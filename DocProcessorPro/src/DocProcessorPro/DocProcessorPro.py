"""
DocProcessorPro — Keyword Scanner GUI

Select an input folder and an output folder, then run the keyword scanner against every PDF
in the input folder. Matching pages are written to {stem}_matched.pdf and a {stem}_manifest.csv in the output folder
prior to consolidation.
"""

from __future__ import annotations

import sys

import DocProcessorPro.dpp_scripts.gui_files.resources_rc  # noqa: F401  — registers embedded Qt resources

from PySide6.QtCore import QTimer
from PySide6.QtWidgets import QApplication

from ._gui_utils import _resolve_app_icon
from ._scanner_dialog import ScannerDialog

def _suppress_subprocess_windows() -> None:
    """Prevent native-exe subprocess calls (pdftoppm, tesseract) from flashing
    console windows in the frozen build."""
    if sys.platform != "win32" or not getattr(sys, "_MEIPASS", None):
        return
    import subprocess

    _CREATE_NO_WINDOW = 0x08000000
    _orig = subprocess.Popen.__init__

    def _patched(self, *args, creationflags: int = 0, **kwargs) -> None:
        _orig(self, *args, creationflags=creationflags | _CREATE_NO_WINDOW, **kwargs)

    subprocess.Popen.__init__ = _patched  # type: ignore[method-assign]


def main() -> None:
    _suppress_subprocess_windows()

    # Windows: tell the shell this process has its own identity so the taskbar
    # shows the custom icon instead of the generic Python launcher icon.
    if sys.platform == "win32":
        try:
            import ctypes

            ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(
                "DocProcessorPro.KeywordScanner.1"
            )
        except Exception:
            pass

    from DocProcessorPro.logging_resources.log_context import setup_logging
    from DocProcessorPro.dpp_scripts.update_scripts.update_codebase import (
        save_install_location,
    )

    setup_logging()
    save_install_location()

    app: QApplication = QApplication.instance() or QApplication(sys.argv)  # type: ignore[assignment]
    _icon = _resolve_app_icon()
    app.setWindowIcon(_icon)
    dialog = ScannerDialog()
    dialog.show()

    # The platform taskbar/dock button isn't created until after the event loop
    # starts, so re-apply the icon via a short timer once the window handle exists.
    def _apply_taskbar_icon() -> None:
        _app = QApplication.instance()
        if isinstance(_app, QApplication):
            _app.setWindowIcon(_icon)
            for w in _app.topLevelWidgets():
                w.setWindowIcon(_icon)

    QTimer.singleShot(200, _apply_taskbar_icon)

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
