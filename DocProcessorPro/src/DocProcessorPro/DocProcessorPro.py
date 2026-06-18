"""
DocProcessorPro — Keyword Scanner GUI

Select an input folder and an output folder, then run the keyword scanner against every PDF
in the input folder. Matching pages are written to {stem}_matched.pdf and a {stem}_manifest.csv in the output folder
prior to consolidation.
"""

from __future__ import annotations

import sys
from pathlib import Path

from PySide6.QtCore import QThread, QTimer, Signal
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QDialog,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QSizePolicy,
    QSpinBox,
    QVBoxLayout,
)

# BACKGROUND WORKERS


class _ScanWorker(QThread):
    """Runs scan_directory() off the main thread."""

    progress = Signal(str)
    finished = Signal(int, int)  # (pdfs_processed, total_matches)
    error = Signal(str)

    def __init__(
        self,
        input_dir: str,
        output_dir: str,
        min_hits: int,
        page_buffer: int,
        require_categories: frozenset[str] | None,
    ) -> None:
        super().__init__()
        self._input_dir = input_dir
        self._output_dir = output_dir
        self._min_hits = min_hits
        self._page_buffer = page_buffer
        self._require_categories = require_categories

    def run(self) -> None:
        try:
            from DocProcessorPro.dpp_scripts.keyword_scanner_scripts.keyword_scanner_codebase import (
                DEFAULT_CATEGORIES,
                scan_directory,
            )

            results = scan_directory(
                self._input_dir,
                self._output_dir,
                DEFAULT_CATEGORIES,
                min_hits=self._min_hits,
                page_buffer=self._page_buffer,
                require_categories=self._require_categories,
                progress_callback=self.progress.emit,
            )
            total_matches = sum(len(v) for v in results.values())
            self.finished.emit(len(results), total_matches)
        except Exception as exc:
            self.error.emit(str(exc))


class _UpdateChecker(QThread):
    """Fetches version.json and emits update_available if a newer version exists."""

    update_available = Signal(str, str)  # (new_version, download_url)

    def run(self) -> None:
        try:
            from DocProcessorPro.dpp_scripts.update_scripts.update_codebase import (
                fetch_remote_version,
                is_update_available,
            )

            remote = fetch_remote_version()
            if remote and is_update_available(remote):
                self.update_available.emit(remote["version"], remote["download_url"])
        except Exception:
            pass  # silently ignore — update check is best-effort


class _UpdateDownloader(QThread):
    """Downloads the installer to the system temp directory."""

    finished = Signal(str)  # absolute path to downloaded installer
    error = Signal(str)

    def __init__(self, download_url: str) -> None:
        super().__init__()
        self._url = download_url

    def run(self) -> None:
        import tempfile

        try:
            from DocProcessorPro.dpp_scripts.update_scripts.update_codebase import (
                download_installer,
            )

            path = download_installer(self._url, Path(tempfile.gettempdir()))
            if path:
                self.finished.emit(str(path))
            else:
                self.error.emit("Download failed — please update manually.")
        except Exception as exc:
            self.error.emit(str(exc))


# MAIN DIALOG


class ScannerDialog(QDialog):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("DocProcessorPro — Keyword Scanner")
        self.setMinimumWidth(520)
        self._worker: _ScanWorker | None = None
        self._update_checker: _UpdateChecker | None = None
        self._update_downloader: _UpdateDownloader | None = None
        self._build_ui()
        # Defer update check until after the event loop starts
        QTimer.singleShot(2000, self._start_update_check)

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setSpacing(12)
        root.setContentsMargins(16, 16, 16, 16)

        form = QFormLayout()
        form.setSpacing(8)

        # Input directory QLineEdit
        self._input_edit = QLineEdit()
        self._input_edit.setPlaceholderText("Select folder containing PDFs…")
        self._input_edit.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed
        )
        input_browse = QPushButton("Browse…")
        input_browse.clicked.connect(self._browse_input)
        input_row = QHBoxLayout()
        input_row.addWidget(self._input_edit)
        input_row.addWidget(input_browse)
        form.addRow("Input folder:", input_row)

        # Output directory QLineEdit
        self._output_edit = QLineEdit()
        self._output_edit.setPlaceholderText("Select folder for output files…")
        self._output_edit.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed
        )
        output_browse = QPushButton("Browse…")
        output_browse.clicked.connect(self._browse_output)
        output_row = QHBoxLayout()
        output_row.addWidget(self._output_edit)
        output_row.addWidget(output_browse)
        form.addRow("Output folder:", output_row)

        # Min-hits QSpinBox
        self._min_hits_spin = QSpinBox()
        self._min_hits_spin.setMinimum(1)
        self._min_hits_spin.setMaximum(50)
        self._min_hits_spin.setValue(1)
        self._min_hits_spin.setToolTip(
            "Minimum number of distinct keyword/pattern matches for a page to be included."
        )
        form.addRow("Min keyword hits:", self._min_hits_spin)

        # Page buffer QSpinBox
        self._page_buffer_spin = QSpinBox()
        self._page_buffer_spin.setMinimum(2)
        self._page_buffer_spin.setMaximum(10)
        self._page_buffer_spin.setValue(2)
        self._page_buffer_spin.setToolTip(
            "Number of pages before and after each keyword hit to include in the output."
        )
        form.addRow("Page buffer:", self._page_buffer_spin)

        # Document section filter checkbox
        self._doc_section_check = QCheckBox("Require document section match")
        self._doc_section_check.setToolTip(
            "Only include pages that are identified as a clinical note, summary, or "
            "report header (e.g. discharge summary, consultation note, H&P, radiology "
            "report). Filters out nursing flowsheets, medication lists, and billing "
            "pages. Uncheck to restore the default broad-match behaviour."
        )
        form.addRow("", self._doc_section_check)

        root.addLayout(form)

        # Run button
        self._run_btn = QPushButton("Run Scan")
        self._run_btn.setDefault(True)
        self._run_btn.clicked.connect(self._run_scan)
        root.addWidget(self._run_btn)

        # Status label
        self._status_label = QLabel("Ready.")
        self._status_label.setWordWrap(True)
        root.addWidget(self._status_label)

    # BROWSE SLOTS

    def _browse_input(self) -> None:
        path = QFileDialog.getExistingDirectory(self, "Select Input Folder")
        if path:
            self._input_edit.setText(path)

    def _browse_output(self) -> None:
        path = QFileDialog.getExistingDirectory(self, "Select Output Folder")
        if path:
            self._output_edit.setText(path)

    # SCAN

    def _run_scan(self) -> None:
        input_dir = self._input_edit.text().strip()
        output_dir = self._output_edit.text().strip()

        if not input_dir:
            QMessageBox.warning(self, "Missing Input", "Please select an input folder.")
            return
        if not output_dir:
            QMessageBox.warning(
                self, "Missing Output", "Please select an output folder."
            )
            return

        self._run_btn.setEnabled(False)
        self._status_label.setText("Scanning… this may take a while for large batches.")

        require = (
            frozenset({"Document Sections"})
            if self._doc_section_check.isChecked()
            else None
        )
        self._worker = _ScanWorker(
            input_dir,
            output_dir,
            self._min_hits_spin.value(),
            self._page_buffer_spin.value(),
            require,
        )
        self._worker.progress.connect(self._status_label.setText)
        self._worker.finished.connect(self._on_finished)
        self._worker.error.connect(self._on_error)
        self._worker.start()

    def _on_finished(self, pdf_count: int, match_count: int) -> None:
        self._run_btn.setEnabled(True)
        out = self._output_edit.text().strip()
        self._status_label.setText(
            f"Done. Processed {pdf_count} PDF(s); {match_count} total page match(es). "
            f"Consolidated PDF and manifest saved to: {out}"
        )

    def _on_error(self, message: str) -> None:
        self._run_btn.setEnabled(True)
        self._status_label.setText(f"Error: {message}")
        QMessageBox.critical(self, "Scan Error", message)

    # UPDATE CHECK

    def _start_update_check(self) -> None:
        self._update_checker = _UpdateChecker()
        self._update_checker.update_available.connect(self._on_update_available)
        self._update_checker.start()

    def _on_update_available(self, new_version: str, download_url: str) -> None:
        reply = QMessageBox.question(
            self,
            "Update Available",
            f"Version {new_version} is available. Download and install now?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.Yes,
        )
        if reply != QMessageBox.StandardButton.Yes:
            return

        self._status_label.setText(f"Downloading update {new_version}…")
        self._run_btn.setEnabled(False)
        self._update_downloader = _UpdateDownloader(download_url)
        self._update_downloader.finished.connect(self._on_download_finished)
        self._update_downloader.error.connect(self._on_download_error)
        self._update_downloader.start()

    def _on_download_finished(self, installer_path: str) -> None:
        from DocProcessorPro.dpp_scripts.update_scripts.update_codebase import (
            launch_installer,
        )

        launch_installer(Path(installer_path))

    def _on_download_error(self, message: str) -> None:
        self._run_btn.setEnabled(True)
        self._status_label.setText("Update download failed.")
        QMessageBox.critical(self, "Update Failed", message)


# GUI ENTRY POINT


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

    from DocProcessorPro.logging_resources.log_context import setup_logging
    from DocProcessorPro.dpp_scripts.update_scripts.update_codebase import (
        save_install_location,
    )

    setup_logging()
    save_install_location()

    app = QApplication.instance() or QApplication(sys.argv)
    dialog = ScannerDialog()
    dialog.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
