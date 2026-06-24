from __future__ import annotations

import sys

import DocProcessorPro.dpp_scripts.gui_files.resources_rc  # noqa: F401  — registers embedded Qt resources
from pathlib import Path

from PySide6.QtGui import QIcon
from PySide6.QtWidgets import QApplication, QDialog
from PySide6.QtCore import Qt


def _apply_win_minmax(dialog: "QDialog") -> None:
    """Add minimize / maximize buttons on Windows.

    QDialog with a parent defaults to Qt.WindowType.Dialog, which maps to
    WS_POPUP on Windows. WS_MINIMIZEBOX / WS_MAXIMIZEBOX are silently ignored
    for popup-style windows — they only work on WS_OVERLAPPEDWINDOW. Switching
    to Qt.WindowType.Window gives the overlapped style that respects the hint.
    Modality from exec() is unaffected by this change.
    """
    if sys.platform == "win32":
        dialog.setWindowFlags(
            Qt.WindowType.Window
            | Qt.WindowType.WindowTitleHint
            | Qt.WindowType.WindowSystemMenuHint
            | Qt.WindowType.WindowMinMaxButtonsHint
            | Qt.WindowType.WindowCloseButtonHint
        )


def _apply_app_icon(widget: "QDialog") -> None:
    """Explicitly set the application icon on a widget.

    app.setWindowIcon() propagates to new windows in Qt6, but QDialog
    instances don't always honour it reliably across platforms. Calling this
    on each dialog guarantees the title-bar icon is correct everywhere.
    """
    app = QApplication.instance()
    if isinstance(app, QApplication):
        widget.setWindowIcon(app.windowIcon())


def _resolve_app_icon() -> "QIcon":
    """Return the application icon with context-aware path resolution.

    PyInstaller bundles: the Qt resource system (resources_rc.py) is compiled
    in, so :/icons/app_icon.svg is tried first; the _MEIPASS extraction
    directory is the fallback if the resource plugin is unavailable.

    Editable / source installs: Qt's SVG iconengine plugin may not be on the
    search path, so the physical SVG file (located relative to __file__) is
    tried first; the Qt resource path is the fallback.
    """
    _SVG_RELPATH = (
        Path(__file__).parent
        / "dpp_scripts"
        / "gui_files"
        / "icons"
        / "delivery_truck_speed_icon.svg"
    )

    if getattr(sys, "_MEIPASS", None):
        # Running from a PyInstaller bundle — resources_rc is embedded and reliable.
        icon = QIcon(":/icons/app_icon.svg")
        if not icon.isNull():
            return icon
        # Secondary: file extracted alongside the exe in _MEIPASS.
        p = Path(sys._MEIPASS) / "dpp_scripts" / "gui_files" / "icons" / "delivery_truck_speed_icon.svg"  # type: ignore[arg-type]
        return QIcon(str(p)) if p.exists() else QIcon()

    # Editable / source install — load the physical file first so icon
    # rendering doesn't depend on Qt's plugin discovery being set up yet.
    if _SVG_RELPATH.exists():
        return QIcon(str(_SVG_RELPATH))

    # Last resort: embedded Qt resource.
    return QIcon(":/icons/app_icon.svg")
