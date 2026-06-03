import json
import logging
import os
import subprocess
import sys
import tempfile
from importlib.metadata import version
from pathlib import Path
import platform

import requests
from packaging.version import Version

logger = logging.getLogger(__name__)

try:
    CURRENT_VERSION = version("DocProcessorPro")
except Exception:
    CURRENT_VERSION = "0.3.0-alpha"
VERSION_URL = "https://github.com/ewbussey/DocProcessorPro-REP/releases/latest/download/version.json"
REQUEST_TIMEOUT = 5


_INSTALL_CONFIG = "install_config.json"


def _user_data_dir() -> Path:
    if platform.system() == "Windows":
        return Path(os.environ.get("LOCALAPPDATA", Path.home())) / "DocProcessorPro"
    return Path.home() / "Library" / "Application Support" / "DocProcessorPro"


def save_install_location() -> None:
    """
    Persist the current exe's parent directory to the user-data dir.
    Called once at startup (frozen builds only) so future update installers
    can reinstall to the same path via /DIR=.
    """
    if not hasattr(sys, "_MEIPASS"):
        return
    install_dir = Path(sys.executable).parent
    config_path = _user_data_dir() / _INSTALL_CONFIG
    try:
        config_path.parent.mkdir(parents=True, exist_ok=True)
        config_path.write_text(json.dumps({"install_dir": str(install_dir)}))
    except Exception:
        logger.warning("save_install_location: could not write %s", config_path)


def get_saved_install_dir() -> Path | None:
    """Return the saved install directory, or None if not recorded."""
    config_path = _user_data_dir() / _INSTALL_CONFIG
    try:
        data = json.loads(config_path.read_text())
        return Path(data["install_dir"])
    except Exception:
        return None


def fetch_remote_version() -> dict | None:
    """Fetch version.json from the latest GitHub release."""
    try:
        response = requests.get(VERSION_URL, timeout=REQUEST_TIMEOUT)
        response.raise_for_status()
        return response.json()
    except requests.HTTPError as e:
        logger.error(f"HTTP error checking for updates: {e}")
        return None
    except requests.ConnectionError as e:
        logger.error(f"Connection error checking for updates: {e}")
        return None
    except requests.Timeout as e:
        logger.error(f"Timeout checking for updates: {e}")
        return None
    except requests.exceptions.JSONDecodeError as e:
        logger.error(f"Invalid JSON in version response: {e}")
        return None


def is_update_available(remote_version: dict) -> bool:
    """Return True if the remote version is newer than the installed version."""
    try:
        return Version(remote_version["version"]) > Version(CURRENT_VERSION)
    except Exception:
        return False


def download_installer(download_url: str, dest_dir: Path) -> Path | None:
    """Stream the installer binary into dest_dir, return its path or None on failure."""
    filename = download_url.split("/")[-1]
    download_path = dest_dir / filename
    try:
        with requests.get(download_url, stream=True, timeout=30) as response:
            response.raise_for_status()
            with open(download_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
        return download_path
    except Exception as e:
        logger.error(f"Installer download failed: {e}")
        return None


def launch_installer(installer_path: Path) -> None:
    """
    Launch the installer as a detached process, then exit the app. The installer runs independently.
    """
    if platform.system() == "Windows":
        DETACHED_PROCESS = 0x00000008
        # /DIR= reinstalls to the same location the user originally chose.
        # Fall back to the exe's own directory if the config hasn't been written yet.
        install_dir = get_saved_install_dir() or Path(sys.executable).parent
        # /CLOSEAPPLICATIONS: lets InnoSetup close any still-open handles
        # before overwriting files in the install directory.
        subprocess.Popen(
            [str(installer_path), "/CLOSEAPPLICATIONS", f"/DIR={install_dir}"],
            creationflags=DETACHED_PROCESS,
            close_fds=True,
        )
    elif platform.system() == "Darwin":
        subprocess.Popen(["open", str(installer_path)])
    else:
        installer_path.chmod(installer_path.stat().st_mode | 0o111)
        subprocess.Popen([str(installer_path)])

    try:
        from PySide6.QtWidgets import QApplication

        app = QApplication.instance()
        if app is not None:
            app.quit()
    except ImportError:
        pass
    sys.exit(0)


def check_and_apply_update(prompt_fn=None) -> None:
    """
    Full update flow called at app launch.

    prompt_fn is an optional callable that receives a message string and returns
    True if the user confirms the update. Passes a QMessageBox wrapper in the GUI.
    If prompt_fn is None, the update is applied without prompting.
    """
    remote = fetch_remote_version()
    if remote is None or not is_update_available(remote):
        return

    new_version = remote["version"]
    download_url = remote["download_url"]

    if prompt_fn is not None:
        confirmed = prompt_fn(f"Version {new_version} is available. Install now?")
        if not confirmed:
            return

    with tempfile.TemporaryDirectory() as tmp_dir:
        installer_path = download_installer(download_url, Path(tmp_dir))

        if installer_path is None:
            logger.error("Update download failed.")
            return

        # Move out of the temp dir before it is cleaned up so the detached
        # installer process can still access the file after this context exits.
        stable_path = Path(tempfile.gettempdir()) / installer_path.name
        installer_path.replace(stable_path)

    launch_installer(stable_path)
