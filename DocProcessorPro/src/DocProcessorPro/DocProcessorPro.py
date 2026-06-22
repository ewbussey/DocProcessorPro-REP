"""
DocProcessorPro — Keyword Scanner GUI

Select an input folder and an output folder, then run the keyword scanner against every PDF
in the input folder. Matching pages are written to {stem}_matched.pdf and a {stem}_manifest.csv in the output folder
prior to consolidation.
"""

from __future__ import annotations

import sys
from pathlib import Path

import DocProcessorPro.resources_rc  # noqa: F401  — registers embedded Qt resources
from PySide6.QtCore import QPointF, QSettings, QSize, QThread, QTimer, QUrl, Qt, Signal
from PySide6.QtGui import QColor, QDesktopServices, QIcon, QImage, QKeySequence, QPainter, QPixmap, QShortcut
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDialog,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QFrame,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSpinBox,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

def _apply_win_minmax(dialog: "QDialog") -> None:
    """Add minimize / maximize buttons on Windows.

    Qt strips them from QDialog title bars by default; this restores the
    standard Windows chrome without affecting macOS or Linux.
    """
    if sys.platform == "win32":
        dialog.setWindowFlags(
            dialog.windowFlags() | Qt.WindowType.WindowMinMaxButtonsHint
        )


# BACKGROUND WORKERS


class _ScanWorker(QThread):
    """Runs scan_directory() off the main thread."""

    progress = Signal(str)
    finished = Signal(int, int, int)  # (pdfs_processed, total_matches, total_exclusions)
    error = Signal(str)

    def __init__(
        self,
        input_dir: str,
        output_dir: str,
        min_hits: float,
        page_buffer: int,
        require_categories: frozenset[str] | None,
        require_anchor: bool = False,
    ) -> None:
        super().__init__()
        self._input_dir = input_dir
        self._output_dir = output_dir
        self._min_hits = min_hits
        self._page_buffer = page_buffer
        self._require_categories = require_categories
        self._require_anchor = require_anchor

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
                require_anchor=self._require_anchor,
                progress_callback=self.progress.emit,
            )
            total_matches = sum(len(r.matches) for r in results.values())
            total_exclusions = sum(len(r.exclusions) for r in results.values())
            self.finished.emit(len(results), total_matches, total_exclusions)
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
        _apply_win_minmax(self)
        self.setWindowTitle("DocProcessorPro — Keyword Scanner")
        self.setMinimumWidth(520)
        self._worker: _ScanWorker | None = None
        self._update_checker: _UpdateChecker | None = None
        self._update_downloader: _UpdateDownloader | None = None
        self._settings = QSettings("DocProcessorPro", "KeywordScanner")
        self._saved_min_hits: float | None = None
        self._last_output_dir: "Path | None" = None
        self._last_scan_settings: "dict | None" = None
        self._last_browsed_dir: str = ""
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

        # Min-score QDoubleSpinBox
        self._min_hits_spin = QDoubleSpinBox()
        self._min_hits_spin.setMinimum(0.5)
        self._min_hits_spin.setMaximum(100.0)
        self._min_hits_spin.setSingleStep(0.5)
        self._min_hits_spin.setValue(3.0)
        self._min_hits_spin.setToolTip(
            "Minimum weighted score for a page to be included. Keywords from specific "
            "clinical categories (Imaging, Therapy, Behavioral Health) contribute more "
            "to the score than broad administrative categories (Billing, Document "
            "Sections). A score of 3.0 requires meaningful clinical content — a single "
            "imaging term alone is not sufficient."
        )
        form.addRow("Min weighted score:", self._min_hits_spin)

        # Page buffer QSpinBox
        self._page_buffer_spin = QSpinBox()
        self._page_buffer_spin.setMinimum(0)
        self._page_buffer_spin.setMaximum(10)
        self._page_buffer_spin.setValue(5)
        self._page_buffer_spin.setToolTip(
            "Number of pages before and after each keyword hit to include in the output."
        )
        form.addRow("Page buffer:", self._page_buffer_spin)

        # Document type filter checkbox
        self._doc_section_check = QCheckBox("Require document type match")
        self._doc_section_check.setToolTip(
            "Only include pages that carry a recognizable clinical document-type label "
            "(e.g. operative note, consultation note, H&P, admission note, psychotherapy "
            "note, imaging report, medical-legal report). Pages that contain clinical "
            "content but no explicit document-type header will be excluded, so leave "
            "unchecked unless output is dominated by untitled pages."
        )
        form.addRow("", self._doc_section_check)

        # Clinical anchor filter checkbox
        self._require_anchor_check = QCheckBox("Require clinical category match")
        self._require_anchor_check.setToolTip(
            "Only include pages that match at least one clinically specific category "
            "(Therapy, Medical Treatment, Injury/Legal, Imaging, Behavioral Health, "
            "Vocational, or Document Type). Pages that only matched Billing terms will "
            "be excluded. Recommended when output contains too many administrative or "
            "billing-only pages."
        )
        form.addRow("", self._require_anchor_check)

        # Affidavits & bills filter checkbox
        self._affidavit_bills_check = QCheckBox("Affidavits + bills only")
        self._affidavit_bills_check.setToolTip(
            "Only include pages that match the Injury/Legal or Billing categories — "
            "useful for quickly locating sworn statements and billing records within a "
            "record set. Lowers the minimum score threshold to 0.3 so that pages with "
            "even a single billing code are captured. Incompatible with the two filters "
            "above, which prioritise clinical content over billing pages."
        )
        form.addRow("", self._affidavit_bills_check)

        # Bidirectional exclusion between "Affidavits & bills" and the clinical filters
        self._affidavit_bills_check.toggled.connect(self._on_affidavit_bills_toggled)
        self._doc_section_check.toggled.connect(self._on_clinical_filter_toggled)
        self._require_anchor_check.toggled.connect(self._on_clinical_filter_toggled)

        root.addLayout(form)

        # Run button
        self._run_btn = QPushButton("Run Scan")
        self._run_btn.setDefault(True)
        self._run_btn.clicked.connect(self._run_scan)
        root.addWidget(self._run_btn)

        # Review button — enabled after a scan produces exclusions
        self._review_btn = QPushButton("Review Excluded Pages")
        self._review_btn.setEnabled(False)
        self._review_btn.setToolTip(
            "Open the differential review window to audit pages the scanner "
            "excluded. Approve pages to merge them into the consolidated output."
        )
        self._review_btn.clicked.connect(self._open_review_dialog)
        root.addWidget(self._review_btn)

        # Matched review button — enabled after a scan produces matched output
        self._matched_review_btn = QPushButton("Review Matched Pages")
        self._matched_review_btn.setEnabled(False)
        self._matched_review_btn.setToolTip(
            "Open the review window to audit pages already in the consolidated output. "
            "Reject duplicates or irrelevant pages to remove them."
        )
        self._matched_review_btn.clicked.connect(self._open_matched_review_dialog)
        root.addWidget(self._matched_review_btn)

        # Load existing review button — always enabled
        self._load_review_btn = QPushButton("Load Existing Review…")
        self._load_review_btn.setToolTip(
            "Open the review window for a previous scan output folder. "
            "Select any file inside the output folder (e.g. a PDF or draft JSON)."
        )
        self._load_review_btn.clicked.connect(self._load_existing_review)
        root.addWidget(self._load_review_btn)

        # Status label
        self._status_label = QLabel("Ready.")
        self._status_label.setWordWrap(True)
        root.addWidget(self._status_label)

        # Restore last-used directories
        self._input_edit.setText(str(self._settings.value("last_input_dir", "")))
        self._output_edit.setText(str(self._settings.value("last_output_dir", "")))

    # FILTER TOGGLE SLOTS

    def _on_affidavit_bills_toggled(self, checked: bool) -> None:
        self._doc_section_check.setEnabled(not checked)
        self._require_anchor_check.setEnabled(not checked)
        if checked:
            self._saved_min_hits = self._min_hits_spin.value()
            self._min_hits_spin.setMinimum(0.3)
            self._min_hits_spin.setValue(0.3)
            self._min_hits_spin.setEnabled(False)
        else:
            self._min_hits_spin.setEnabled(True)
            self._min_hits_spin.setMinimum(0.5)
            self._min_hits_spin.setValue(
                self._saved_min_hits if self._saved_min_hits is not None else 3.0
            )

    def _on_clinical_filter_toggled(self) -> None:
        any_active = (
            self._doc_section_check.isChecked()
            or self._require_anchor_check.isChecked()
        )
        self._affidavit_bills_check.setEnabled(not any_active)

    # BROWSE SLOTS

    def _browse_input(self) -> None:
        start = self._input_edit.text().strip() or self._last_browsed_dir
        path = QFileDialog.getExistingDirectory(self, "Select Input Folder", start)
        if path:
            self._input_edit.setText(path)
            self._settings.setValue("last_input_dir", path)
            self._last_browsed_dir = path

    def _browse_output(self) -> None:
        start = self._output_edit.text().strip() or self._last_browsed_dir
        path = QFileDialog.getExistingDirectory(self, "Select Output Folder", start)
        if path:
            self._output_edit.setText(path)
            self._settings.setValue("last_output_dir", path)
            self._last_browsed_dir = path

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

        if self._affidavit_bills_check.isChecked():
            require: frozenset[str] | None = frozenset({"INJURY_LEGAL", "BILLING"})
        elif self._doc_section_check.isChecked():
            require = frozenset({"DOCUMENT_TYPE"})
        else:
            require = None

        # Derive a human-readable mode label for feedback tagging
        if self._affidavit_bills_check.isChecked():
            _mode = "affidavits_bills"
        elif self._doc_section_check.isChecked() and self._require_anchor_check.isChecked():
            _mode = "document_type_and_anchor"
        elif self._doc_section_check.isChecked():
            _mode = "document_type_filter"
        elif self._require_anchor_check.isChecked():
            _mode = "clinical_anchor"
        else:
            _mode = "standard"
        self._last_scan_settings = {
            "scan_mode": _mode,
            "min_hits": self._min_hits_spin.value(),
            "page_buffer": self._page_buffer_spin.value(),
            "require_categories": sorted(require) if require else None,
            "require_anchor": self._require_anchor_check.isChecked(),
        }

        self._worker = _ScanWorker(
            input_dir,
            output_dir,
            self._min_hits_spin.value(),
            self._page_buffer_spin.value(),
            require,
            require_anchor=self._require_anchor_check.isChecked(),
        )
        self._worker.progress.connect(self._status_label.setText)
        self._worker.finished.connect(self._on_finished)
        self._worker.error.connect(self._on_error)
        self._worker.start()

    def _on_finished(self, pdf_count: int, match_count: int, exclusion_count: int) -> None:
        self._run_btn.setEnabled(True)
        out = self._output_edit.text().strip()
        self._settings.setValue("last_input_dir", self._input_edit.text().strip())
        self._settings.setValue("last_output_dir", out)
        self._last_output_dir = Path(out)
        QDesktopServices.openUrl(QUrl.fromLocalFile(out))
        self._status_label.setText(
            f"Done. Processed {pdf_count} PDF(s); {match_count} match(es), "
            f"{exclusion_count} reviewable exclusion(s). "
            f"Output saved to: {out}"
        )
        if exclusion_count > 0:
            self._review_btn.setEnabled(True)
            self._review_btn.setText(
                f"Review Excluded Pages ({exclusion_count} to review)"
            )
        if match_count > 0:
            self._matched_review_btn.setEnabled(True)
            self._matched_review_btn.setText(
                f"Review Matched Pages ({match_count} matched)"
            )

    def _open_review_dialog(self) -> None:
        if self._last_output_dir is None:
            QMessageBox.warning(
                self, "No Scan Data", "Run a scan first to generate review data."
            )
            return
        pdf_path = self._last_output_dir / "_consolidated_unmatched.pdf"
        csv_path = self._last_output_dir / "_consolidated_unmatched_manifest.csv"
        if not pdf_path.exists():
            QMessageBox.warning(
                self,
                "No Excluded Pages",
                "No consolidated unmatched PDF found. "
                "Run a scan that produces reviewable exclusions first.",
            )
            return
        dlg = ReviewDialog(
            pdf_path, csv_path, self._last_output_dir,
            scan_settings=self._last_scan_settings,
            parent=self,
        )
        self._run_review_dialog(dlg)

    def _open_matched_review_dialog(self) -> None:
        if self._last_output_dir is None:
            QMessageBox.warning(
                self, "No Scan Data", "Run a scan first to generate review data."
            )
            return
        pdf_path = self._last_output_dir / "_consolidated.pdf"
        csv_path = self._last_output_dir / "_consolidated_matched_manifest.csv"
        if not pdf_path.exists():
            QMessageBox.warning(
                self,
                "No Consolidated PDF Found",
                "No _consolidated.pdf found. Run a scan that produces matches first.",
            )
            return
        if not csv_path.exists():
            QMessageBox.warning(
                self,
                "No Matched Manifest Found",
                "_consolidated_matched_manifest.csv not found.\n\n"
                "Re-run the scan to generate the manifest required for matched review.",
            )
            return
        dlg = ReviewDialog(
            pdf_path, csv_path, self._last_output_dir,
            scan_settings=self._last_scan_settings,
            mode="matched",
            parent=self,
        )
        self._run_review_dialog(dlg)

    def _load_existing_review(self) -> None:
        start = (
            self._output_edit.text().strip()
            or self._input_edit.text().strip()
            or self._last_browsed_dir
        )
        # File picker: user selects any file inside the output folder.
        # The output folder is derived from the selected file's parent.
        selected, _ = QFileDialog.getOpenFileName(
            self,
            "Select Any File in Scan Output Folder",
            start,
            "Review files (*_consolidated_unmatched.pdf *_consolidated.pdf "
            "*_review_draft.json *_matched_review_draft.json *_feedback.jsonl);;"
            "All files (*)",
        )
        if not selected:
            return
        output_dir = Path(selected).parent
        self._last_browsed_dir = str(output_dir)

        # Auto-detect which review to open based on filename selected
        selected_name = Path(selected).name
        open_matched = (
            selected_name in ("_consolidated.pdf", "_matched_review_draft.json",
                              "_matched_feedback.jsonl")
            or selected_name.endswith("_matched_manifest.csv")
        )

        if open_matched:
            pdf_path = output_dir / "_consolidated.pdf"
            csv_path = output_dir / "_consolidated_matched_manifest.csv"
            if not pdf_path.exists():
                QMessageBox.warning(self, "No Consolidated PDF Found",
                    "No _consolidated.pdf found in the selected folder.")
                return
            if not csv_path.exists():
                QMessageBox.warning(self, "No Matched Manifest Found",
                    "_consolidated_matched_manifest.csv not found.\n\n"
                    "Re-run the scan to generate this manifest.")
                return
            settings_dlg = _ScanSettingsDialog(parent=self)
            if settings_dlg.exec() != QDialog.DialogCode.Accepted:
                return
            dlg = ReviewDialog(
                pdf_path, csv_path, output_dir,
                scan_settings=settings_dlg.scan_settings,
                mode="matched",
                parent=self,
            )
        else:
            pdf_path = output_dir / "_consolidated_unmatched.pdf"
            csv_path = output_dir / "_consolidated_unmatched_manifest.csv"
            if not pdf_path.exists():
                QMessageBox.warning(self, "No Unmatched PDF Found",
                    "No _consolidated_unmatched.pdf found in the selected folder.\n\n"
                    "Please select a file inside a folder that contains scan output.")
                return
            settings_dlg = _ScanSettingsDialog(parent=self)
            if settings_dlg.exec() != QDialog.DialogCode.Accepted:
                return
            dlg = ReviewDialog(
                pdf_path, csv_path, output_dir,
                scan_settings=settings_dlg.scan_settings,
                parent=self,
            )
        self._run_review_dialog(dlg)

    def _run_review_dialog(self, dlg: "ReviewDialog") -> None:
        """Execute a ReviewDialog and update the status label if Apply was used."""
        dlg.exec()
        if dlg._apply_result is not None:
            primary, skipped = dlg._apply_result
            if dlg._mode == "matched":
                msg = f"Review complete — {primary} page(s) kept in consolidated output."
            else:
                msg = f"Review complete — {primary} page(s) applied to consolidated output."
            if skipped:
                msg += f" ({skipped} page(s) skipped — source file(s) not found.)"
            self._status_label.setText(msg)

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


class _FeedbackWorker(QThread):
    """Runs apply_feedback() or apply_matched_feedback() off the main thread."""

    progress = Signal(str)
    finished = Signal(int, int, int)  # (pages_approved_or_kept, pages_rejected, pages_skipped)
    error = Signal(str)

    def __init__(
        self,
        output_dir: str,
        feedback_path: str,
        mode: str = "unmatched",
    ) -> None:
        super().__init__()
        self._output_dir = output_dir
        self._feedback_path = feedback_path
        self._mode = mode

    def run(self) -> None:
        try:
            if self._mode == "matched":
                from DocProcessorPro.dpp_scripts.keyword_scanner_scripts.keyword_scanner_codebase import (
                    apply_matched_feedback,
                )
                kept, removed, skipped = apply_matched_feedback(
                    self._output_dir,
                    self._feedback_path,
                    progress_callback=self.progress.emit,
                )
                self.finished.emit(kept, removed, skipped)
            else:
                from DocProcessorPro.dpp_scripts.keyword_scanner_scripts.keyword_scanner_codebase import (
                    apply_feedback,
                )
                approved, rejected, skipped = apply_feedback(
                    self._output_dir,
                    self._feedback_path,
                    progress_callback=self.progress.emit,
                )
                self.finished.emit(approved, rejected, skipped)
        except Exception as exc:
            self.error.emit(str(exc))


# Therapy triage classification constants — module level so they're computed once
_THERAPY_CATS: frozenset[str] = frozenset({"THERAPY", "BEHAVIORAL_HEALTH"})
_INTAKE_KW: frozenset[str] = frozenset({
    "initial evaluation", "intake note", "intake assessment",
    "initial assessment", "initial session", "first session",
    "first visit", "initial visit", "initial psychiatric",
    "psychological evaluation", "evaluation and management",
    # regex-produced verbatim matches from CATEGORY_DOCUMENT_TYPE pattern:
    "initial summary", "initial note", "initial report",
    "initial history", "intake summary", "intake report",
})
_DISCHARGE_KW: frozenset[str] = frozenset({
    "discharge note", "discharge summary", "termination note",
    "termination summary", "termination session", "final session",
    "final visit", "final note", "treatment termination", "discharge plan",
    # regex-produced:
    "discharge report", "discharge history", "termination report",
})
# Non-therapy clinical categories that are auto-approved when unique
_AUTO_APPROVE_CATS: frozenset[str] = frozenset({
    "IMAGING", "INJURY_LEGAL", "VOCATIONAL", "BILLING", "MEDICAL_TREATMENT"
})

# Background colors keyed by (decision, decision_source)
_DECISION_COLORS: dict[tuple[str, str], tuple[int, int, int]] = {
    ("approved",            "user"):                 (200, 255, 200),  # green
    ("rejected_duplicate",  "user"):                 (255, 210, 140),  # orange
    ("rejected_irrelevant", "user"):                 (255, 200, 200),  # red
    ("approved",            "smart_triage"):         (185, 215, 255),  # pale blue
    ("rejected_duplicate",  "smart_triage"):         (255, 255, 185),  # pale yellow
    ("rejected_irrelevant", "smart_triage"):         (235, 205, 255),  # pale lavender
    ("approved",            "find_duplicates"):      (200, 255, 200),  # same green
    ("rejected_duplicate",  "find_duplicates"):      (255, 225, 190),  # pale peach
    ("rejected_irrelevant", "find_duplicates"):      (255, 200, 200),  # same red
    ("approved",            "matched_mode_default"): (200, 235, 255),  # pale cyan
    ("rejected_duplicate",  "matched_mode_default"): (255, 210, 140),  # orange
    ("rejected_irrelevant", "matched_mode_default"): (255, 200, 200),  # red
}


def _dhash(qimage: "QImage") -> int:
    """64-bit difference hash: horizontal gradient over a 9×8 grayscale thumbnail."""
    gray = qimage.convertToFormat(QImage.Format.Format_Grayscale8).scaled(
        QSize(9, 8), Qt.AspectRatioMode.IgnoreAspectRatio,
        Qt.TransformationMode.SmoothTransformation,
    )
    bits = gray.bits()
    h = 0
    for row in range(8):
        for col in range(8):
            h = (h << 1) | (1 if bits[row * 9 + col] > bits[row * 9 + col + 1] else 0)
    return h


def _hamming(a: int, b: int) -> int:
    return bin(a ^ b).count("1")


def _cluster_by_hash(
    hashes: list[tuple[int, int]], threshold: int = 10
) -> list[list[int]]:
    """Greedy clustering: each item joins the first cluster within Hamming threshold."""
    clusters: list[list[int]] = []
    representatives: list[int] = []
    for idx, h in hashes:
        placed = False
        for c_i, rep_h in enumerate(representatives):
            if _hamming(h, rep_h) <= threshold:
                clusters[c_i].append(idx)
                placed = True
                break
        if not placed:
            clusters.append([idx])
            representatives.append(h)
    return clusters


class _DedupDialog(QDialog):
    """Side-by-side comparison dialog for groups of visually similar pages.

    Each group shows page thumbnails with source / page / score info and a
    'Keep' checkbox. Default: the highest-scored already-approved page in each
    group is checked; all others are unchecked (will be marked rejected_duplicate
    on Apply). The user can override any checkbox before confirming.
    """

    def __init__(
        self,
        groups: list[list[tuple[int, dict, "QImage"]]],
        decisions: dict[int, str],
        decision_sources: dict[int, str],
        parent=None,
    ) -> None:
        super().__init__(parent)
        _apply_win_minmax(self)
        self._groups = groups
        self._cur_decisions = decisions
        self._cur_sources = decision_sources
        total = sum(len(g) for g in groups)
        self.setWindowTitle(
            f"Find Duplicates — {len(groups)} group(s), {total} similar pages"
        )
        self.setMinimumSize(1100, 700)
        self._keep_checks: list[list[tuple[int, QCheckBox]]] = []
        self._build_ui()

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(10, 10, 10, 10)
        root.setSpacing(8)

        total = sum(len(g) for g in self._groups)
        header = QLabel(
            f"Found {len(self._groups)} group(s) of visually similar pages "
            f"({total} pages total).\n"
            "Check pages to keep as approved; uncheck to mark as duplicate. "
            "The highest-scored page in each group is pre-selected."
        )
        header.setWordWrap(True)
        root.addWidget(header)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        inner = QWidget()
        inner_layout = QVBoxLayout(inner)
        inner_layout.setSpacing(16)

        for g_idx, group in enumerate(self._groups):
            # Prefer already-approved pages as the default canonical
            approved = [t for t in group if self._cur_decisions.get(t[0]) == "approved"]
            if approved:
                canonical_idx = max(
                    approved, key=lambda t: float(t[1].get("total_hits", 0))
                )[0]
            else:
                canonical_idx = max(
                    group, key=lambda t: float(t[1].get("total_hits", 0))
                )[0]

            frame = QFrame()
            frame.setFrameShape(QFrame.Shape.Box)
            frame.setFrameShadow(QFrame.Shadow.Sunken)
            frame_layout = QVBoxLayout(frame)
            frame_layout.setContentsMargins(8, 8, 8, 8)
            frame_layout.setSpacing(4)

            grp_label = QLabel(f"Group {g_idx + 1}")
            grp_label.setStyleSheet("font-weight: bold; font-size: 11px;")
            frame_layout.addWidget(grp_label)

            items_row = QHBoxLayout()
            items_row.setSpacing(12)
            checks: list[tuple[int, QCheckBox]] = []

            for row_idx, row, qimg in sorted(
                group,
                key=lambda t: float(t[1].get("total_hits", 0)),
                reverse=True,
            ):
                item_w = QWidget()
                item_l = QVBoxLayout(item_w)
                item_l.setSpacing(2)
                item_l.setContentsMargins(4, 4, 4, 4)

                # Composite onto a white background so pages with a transparent
                # alpha layer (common in scanned PDFs) render correctly in dark mode
                white = QPixmap(qimg.width(), qimg.height())
                white.fill(Qt.GlobalColor.white)
                painter = QPainter(white)
                painter.drawImage(0, 0, qimg)
                painter.end()
                thumb = QLabel()
                thumb.setPixmap(
                    white.scaled(
                        220, 286,
                        Qt.AspectRatioMode.KeepAspectRatio,
                        Qt.TransformationMode.SmoothTransformation,
                    )
                )
                thumb.setAlignment(Qt.AlignmentFlag.AlignCenter)
                item_l.addWidget(thumb)

                src_path = row.get("source_pdf_path", "")
                filename = src_path.rsplit("/", 1)[-1].rsplit("\\", 1)[-1]
                score = float(row.get("total_hits", 0))
                pg = row.get("page_num", "?")
                cur_dec = self._cur_decisions.get(row_idx)
                cur_src = self._cur_sources.get(row_idx, "—")
                status = (
                    f"{cur_dec} ({cur_src})" if cur_dec else "undecided"
                )
                info = QLabel(f"{filename}\np.{pg}  score {score:.2f}\n{status}")
                info.setWordWrap(True)
                info.setAlignment(Qt.AlignmentFlag.AlignCenter)
                item_l.addWidget(info)

                chk = QCheckBox("Keep")
                chk.setChecked(row_idx == canonical_idx)
                item_l.addWidget(chk, 0, Qt.AlignmentFlag.AlignHCenter)
                checks.append((row_idx, chk))
                items_row.addWidget(item_w)

            items_row.addStretch()
            frame_layout.addLayout(items_row)

            btn_row = QHBoxLayout()
            btn_row.addStretch()
            best_btn = QPushButton("Keep Highest Scored")
            best_btn.clicked.connect(
                lambda _, c=checks: [chk.setChecked(i == 0) for i, (_, chk) in enumerate(c)]
            )
            btn_row.addWidget(best_btn)
            frame_layout.addLayout(btn_row)

            self._keep_checks.append(checks)
            inner_layout.addWidget(frame)

        inner_layout.addStretch()
        scroll.setWidget(inner)
        root.addWidget(scroll)

        bottom = QHBoxLayout()
        bottom.addStretch()
        apply_btn = QPushButton("Apply Selections")
        apply_btn.setDefault(True)
        apply_btn.clicked.connect(self.accept)
        cancel_btn = QPushButton("Cancel (no changes)")
        cancel_btn.clicked.connect(self.reject)
        bottom.addWidget(apply_btn)
        bottom.addWidget(cancel_btn)
        root.addLayout(bottom)

    def results(self) -> tuple[list[int], list[int]]:
        """Return (keep_indices, reject_indices) based on checkbox state."""
        keep: list[int] = []
        reject: list[int] = []
        for checks in self._keep_checks:
            for row_idx, chk in checks:
                (keep if chk.isChecked() else reject).append(row_idx)
        return keep, reject


class _ScanSettingsDialog(QDialog):
    """Prompts the user for the scan settings that produced an existing output folder.

    Used when loading a prior review session so exported feedback can be tagged
    with the correct scan profile. All fields default to the most common values;
    the user may leave the mode as 'Don't specify' to pass scan_settings=None.
    """

    _MODE_ITEMS: list[tuple[str, "str | None"]] = [
        ("Don't specify",                    None),
        ("Standard (full clinical scan)",    "standard"),
        ("Affidavits + Bills only",          "affidavits_bills"),
        ("Document type filter",             "document_type_filter"),
        ("Clinical anchor required",         "clinical_anchor"),
        ("Document type + clinical anchor",  "document_type_and_anchor"),
    ]

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        _apply_win_minmax(self)
        self.setWindowTitle("Scan Settings")
        self.setMinimumWidth(380)
        self._result: "dict | None" = None
        self._build_ui()

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setSpacing(10)
        root.setContentsMargins(16, 16, 16, 16)

        note = QLabel(
            "Specify the scan settings used to produce this output folder.\n"
            "This tags exported feedback for accurate training-data partitioning.\n"
            "Select 'Don't specify' if the original settings are unknown."
        )
        note.setWordWrap(True)
        root.addWidget(note)

        form = QFormLayout()
        form.setSpacing(8)

        self._mode_combo = QComboBox()
        for label, _ in self._MODE_ITEMS:
            self._mode_combo.addItem(label)
        self._mode_combo.currentIndexChanged.connect(self._on_mode_changed)
        form.addRow("Scan mode:", self._mode_combo)

        self._min_hits_spin = QDoubleSpinBox()
        self._min_hits_spin.setMinimum(0.3)
        self._min_hits_spin.setMaximum(100.0)
        self._min_hits_spin.setSingleStep(0.5)
        self._min_hits_spin.setValue(3.0)
        form.addRow("Min weighted score:", self._min_hits_spin)

        self._page_buffer_spin = QSpinBox()
        self._page_buffer_spin.setMinimum(0)
        self._page_buffer_spin.setMaximum(10)
        self._page_buffer_spin.setValue(5)
        form.addRow("Page buffer:", self._page_buffer_spin)

        root.addLayout(form)

        btn_row = QHBoxLayout()
        btn_row.addStretch()
        ok_btn = QPushButton("OK")
        ok_btn.setDefault(True)
        ok_btn.clicked.connect(self._on_ok)
        cancel_btn = QPushButton("Cancel")
        cancel_btn.clicked.connect(self.reject)
        btn_row.addWidget(ok_btn)
        btn_row.addWidget(cancel_btn)
        root.addLayout(btn_row)

    def _on_mode_changed(self, index: int) -> None:
        mode = self._MODE_ITEMS[index][1]
        if mode == "affidavits_bills":
            self._min_hits_spin.setValue(0.3)
            self._min_hits_spin.setEnabled(False)
        else:
            self._min_hits_spin.setEnabled(True)
            if self._min_hits_spin.value() == 0.3:
                self._min_hits_spin.setValue(3.0)

    def _on_ok(self) -> None:
        mode = self._MODE_ITEMS[self._mode_combo.currentIndex()][1]
        if mode is None:
            self._result = None
        else:
            _require: "dict[str, tuple[list[str] | None, bool]]" = {
                "standard":                  (None,                          False),
                "affidavits_bills":          (["BILLING", "INJURY_LEGAL"],   False),
                "document_type_filter":      (["DOCUMENT_TYPE"],             False),
                "clinical_anchor":           (None,                          True),
                "document_type_and_anchor":  (["DOCUMENT_TYPE"],             True),
            }
            require_categories, require_anchor = _require[mode]
            self._result = {
                "scan_mode":          mode,
                "min_hits":           self._min_hits_spin.value(),
                "page_buffer":        self._page_buffer_spin.value(),
                "require_categories": require_categories,
                "require_anchor":     require_anchor,
            }
        self.accept()

    @property
    def scan_settings(self) -> "dict | None":
        return self._result


class ReviewDialog(QDialog):
    """Three-panel review window for auditing keyword-scanner page decisions.

    mode="unmatched" (default): review excluded pages; approve to move to consolidated.
    mode="matched": review matched pages already in consolidated; reject to remove.

    Left:   list of pages sorted by score descending
    Centre: QPdfView showing the consolidated PDF
    Right:  metadata panel + Approve / Reject / Skip buttons

    Keyboard shortcuts: A = approve, D = duplicate, R = irrelevant, S = skip.
    After each decision the view auto-advances to the next unreviewed page.
    """

    def __init__(
        self,
        review_pdf: "Path",
        manifest_csv: "Path",
        output_dir: "Path",
        scan_settings: "dict | None" = None,
        mode: str = "unmatched",
        parent=None,
    ) -> None:
        from pathlib import Path as _Path

        super().__init__(parent)
        _apply_win_minmax(self)
        self._mode = mode
        if mode == "matched":
            self.setWindowTitle("DocProcessorPro — Review Matched Pages")
        else:
            self.setWindowTitle("DocProcessorPro — Review Excluded Pages")
        self.setMinimumSize(1100, 700)

        self._output_dir = _Path(output_dir)
        if mode == "matched":
            self._feedback_path = self._output_dir / "_matched_feedback.jsonl"
            self._draft_path    = self._output_dir / "_matched_review_draft.json"
        else:
            self._feedback_path = self._output_dir / "_feedback.jsonl"
            self._draft_path    = self._output_dir / "_review_draft.json"
        self._fb_worker: _FeedbackWorker | None = None
        self._scan_settings: "dict | None" = scan_settings
        self._apply_result: "tuple[int, int] | None" = None  # (approved, skipped) set on successful apply

        self._rows: list[dict] = []
        self._subtypes: list[str | None] = []  # parallel to _rows
        self._decisions: dict[int, str] = {}         # row_index → "approved" | "rejected_*"
        self._triage_decisions: dict[int, str] = {}  # snapshot after Smart Triage runs
        self._decision_sources: dict[int, str] = {}  # row_index → "user"|"smart_triage"|"find_duplicates"|"matched_mode_default"
        self._dedup_groups: dict[int, int] = {}      # row_index → cluster id from last Find Duplicates run
        self._dedup_canonical: set[int] = set()      # indices identified as canonical in their dedup cluster
        self._current_index: int = -1

        self._load_manifest(manifest_csv)
        # Matched mode: pre-approve every page (already in consolidated; user marks removals)
        if self._mode == "matched":
            for i in range(len(self._rows)):
                self._decisions[i] = "approved"
                self._decision_sources[i] = "matched_mode_default"
        self._build_ui()
        self._load_pdf(review_pdf)
        self._load_existing_feedback()
        self._load_draft()  # overlay any newer in-progress decisions from auto-save

        if self._rows:
            self._select_row(0)

    # ------------------------------------------------------------------ setup

    def _load_manifest(self, csv_path: "Path") -> None:
        import csv as _csv

        if not csv_path.exists():
            return
        with open(csv_path, newline="", encoding="utf-8") as f:
            rows = list(_csv.DictReader(f))
        # Sort by source PDF then page number so whole records appear in sequence
        self._rows = sorted(
            rows,
            key=lambda r: (r.get("source_pdf_path", ""), int(r.get("page_num", 0))),
        )
        self._subtypes = [self._classify_subtype(row) for row in self._rows]

    def _load_pdf(self, pdf_path: "Path") -> None:
        try:
            from PySide6.QtPdf import QPdfDocument

            self._pdf_doc = QPdfDocument(self)
            self._pdf_doc.load(str(pdf_path))
            self._pdf_view.setDocument(self._pdf_doc)
        except Exception:
            self._pdf_view.setEnabled(False)

    def _load_existing_feedback(self) -> None:
        """Pre-populate decisions from a prior-session _feedback.jsonl if present."""
        import json

        if not self._feedback_path.exists():
            return
        for line in self._feedback_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            src = rec.get("source_pdf_path", "")
            pg = rec.get("page_num")
            label = rec.get("label", "")
            if not (src and pg and label in ("include", "exclude")):
                continue
            if label == "include":
                decision = "approved"
            else:
                reason = rec.get("rejection_reason", "irrelevant")
                decision = (
                    "rejected_duplicate" if reason == "duplicate" else "rejected_irrelevant"
                )
            source = rec.get("decision_source", "user")
            for idx, row in enumerate(self._rows):
                if (
                    row.get("source_pdf_path") == src
                    and int(row.get("page_num", 0)) == int(pg)
                ):
                    self._decisions[idx] = decision
                    self._decision_sources[idx] = source
                    break
        self._refresh_list_colors()
        self._update_toolbar()

    def _save_draft(self) -> None:
        """Persist current decisions to a draft file after every change.

        Keyed by "source_pdf_path|page_num" so the draft survives list re-sorts.
        Uses an atomic rename so a crash mid-write never leaves a corrupt file.
        """
        import json

        def _to_stable(d: dict) -> dict:
            out: dict = {}
            for idx, val in d.items():
                row = self._rows[idx]
                key = f"{row.get('source_pdf_path', '')}|{row.get('page_num', '')}"
                out[key] = val
            return out

        # Dedup canonical set → stable keys list
        dedup_canonical_keys = [
            f"{self._rows[i].get('source_pdf_path', '')}|{self._rows[i].get('page_num', '')}"
            for i in self._dedup_canonical
            if i < len(self._rows)
        ]

        draft = {
            "decisions":            _to_stable(self._decisions),
            "triage_decisions":     _to_stable(self._triage_decisions),
            "decision_sources":     _to_stable(self._decision_sources),
            "dedup_group_ids":      _to_stable(self._dedup_groups),
            "dedup_canonical_keys": dedup_canonical_keys,
        }
        tmp = self._draft_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(draft), encoding="utf-8")
        tmp.replace(self._draft_path)

    def _load_draft(self) -> None:
        """Overlay in-progress decisions from the auto-save draft.

        Called after _load_existing_feedback() so the draft (always more recent)
        takes precedence over the last explicit export.
        Supports both the current nested format and the legacy flat format.
        """
        import json

        if not self._draft_path.exists():
            return
        try:
            raw: dict = json.loads(self._draft_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return

        # Detect format: nested (current) vs. legacy flat
        if "decisions" in raw and isinstance(raw["decisions"], dict):
            decisions_map: dict = raw["decisions"]
            triage_map: dict = raw.get("triage_decisions", {})
            sources_map: dict = raw.get("decision_sources", {})
            dedup_groups_map: dict = raw.get("dedup_group_ids", {})
            dedup_canonical_keys: list = raw.get("dedup_canonical_keys", [])
        else:
            decisions_map = raw  # legacy flat
            triage_map = {}
            sources_map = {}
            dedup_groups_map = {}
            dedup_canonical_keys = []

        # Build stable-key → index lookup for current row order
        key_to_idx: dict[str, int] = {}
        for idx, row in enumerate(self._rows):
            key = f"{row.get('source_pdf_path', '')}|{row.get('page_num', '')}"
            key_to_idx[key] = idx

        restored = 0
        for key, decision in decisions_map.items():
            idx = key_to_idx.get(key)
            if idx is not None and self._decisions.get(idx) != decision:
                self._decisions[idx] = decision
                restored += 1

        for key, decision in triage_map.items():
            idx = key_to_idx.get(key)
            if idx is not None:
                self._triage_decisions[idx] = decision

        for key, source in sources_map.items():
            idx = key_to_idx.get(key)
            if idx is not None:
                self._decision_sources[idx] = source

        for key, group_id in dedup_groups_map.items():
            idx = key_to_idx.get(key)
            if idx is not None:
                self._dedup_groups[idx] = group_id

        self._dedup_canonical = {
            key_to_idx[k] for k in dedup_canonical_keys if k in key_to_idx
        }

        if restored:
            self._refresh_list_colors()
            self._update_toolbar()
            QTimer.singleShot(
                200,
                lambda: QMessageBox.information(
                    self,
                    "Session Restored",
                    f"Restored {restored} unsaved decision(s) from your last session.\n\n"
                    "Your progress was automatically saved and is ready to continue.\n"
                    "Click 'Export Feedback' to commit these decisions permanently.",
                ),
            )

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(8, 8, 8, 8)
        root.setSpacing(6)

        # Toolbar row
        toolbar = QHBoxLayout()
        self._progress_label = QLabel("0 approved / 0 rejected / 0 remaining")
        toolbar.addWidget(self._progress_label)
        toolbar.addStretch()
        dedup_btn = QPushButton("Find Duplicates")
        dedup_btn.setToolTip(
            "Render all pages and detect visually similar groups using perceptual hashing. "
            "Runs across all pages regardless of current decision state."
        )
        dedup_btn.clicked.connect(self._run_dedup_pass)
        toolbar.addWidget(dedup_btn)
        triage_btn = QPushButton("Smart Triage")
        triage_btn.setToolTip(
            "Auto-categorize therapy pages (intake/discharge/progress fallback) "
            "and approve non-therapy clinical pages. "
            "Run Find Duplicates first for visual dedup."
        )
        triage_btn.clicked.connect(self._run_smart_triage)
        toolbar.addWidget(triage_btn)
        self._filter_btn = QPushButton("Approved + Duplicates")
        self._filter_btn.setCheckable(True)
        self._filter_btn.setToolTip(
            "Show only approved and duplicate-flagged pages — "
            "helps spot pages that were approved but should be marked as duplicates."
        )
        self._filter_btn.toggled.connect(self._apply_filter)
        toolbar.addWidget(self._filter_btn)
        export_btn = QPushButton("Export Feedback")
        export_btn.setToolTip("Write decisions to _feedback.jsonl in the output folder")
        export_btn.clicked.connect(self._export_feedback)
        toolbar.addWidget(export_btn)
        if self._mode == "matched":
            self._apply_btn = QPushButton("Remove Rejected && Rebuild")
            self._apply_btn.setToolTip(
                "Rebuild _consolidated.pdf without rejected pages"
            )
        else:
            self._apply_btn = QPushButton("Apply Approved && Rebuild")
            self._apply_btn.setToolTip(
                "Append approved pages to _consolidated.pdf and rebuild "
                "_consolidated_unmatched.pdf"
            )
        self._apply_btn.clicked.connect(self._apply_approved)
        toolbar.addWidget(self._apply_btn)
        root.addLayout(toolbar)

        # Three-panel splitter
        splitter = QSplitter()
        splitter.setChildrenCollapsible(False)

        # Left: page list
        self._list = QListWidget()
        self._list.setMaximumWidth(220)
        self._list.currentRowChanged.connect(self._on_list_row_changed)
        for i, row in enumerate(self._rows):
            self._list.addItem(self._make_list_item(i, row))
        splitter.addWidget(self._list)

        # Centre: PDF viewer
        try:
            from PySide6.QtPdfWidgets import QPdfView

            self._pdf_view = QPdfView()
            self._pdf_view.setPageMode(QPdfView.PageMode.SinglePage)
            self._pdf_view.setZoomMode(QPdfView.ZoomMode.FitInView)
        except Exception:
            # Fallback: plain label if QPdfView is unavailable
            self._pdf_view = QLabel("PDF viewer unavailable")  # type: ignore[assignment]
        splitter.addWidget(self._pdf_view)

        # Right: metadata + action buttons
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        right_layout.setContentsMargins(8, 4, 4, 4)
        right_layout.setSpacing(6)
        right_panel.setMaximumWidth(260)

        meta_scroll = QScrollArea()
        meta_scroll.setWidgetResizable(True)
        meta_inner = QWidget()
        self._meta_form = QFormLayout(meta_inner)
        self._meta_form.setSpacing(4)
        self._meta_labels: dict[str, QLabel] = {}
        for field in (
            "Source",
            "Page",
            "Type",
            "Score",
            "Threshold",
            "Excluded because",
            "Categories",
            "Keywords",
            "Dates",
        ):
            lbl = QLabel("—")
            lbl.setWordWrap(True)
            self._meta_labels[field] = lbl
            self._meta_form.addRow(f"{field}:", lbl)
        meta_scroll.setWidget(meta_inner)
        right_layout.addWidget(meta_scroll)

        btn_layout = QVBoxLayout()
        self._approve_btn = QPushButton("Approve          (A)")
        self._reject_dup_btn = QPushButton("Dup / Redundant  (D)")
        self._reject_irr_btn = QPushButton("Irrelevant       (R)")
        self._skip_btn = QPushButton("Skip             (S)")
        for btn in (
            self._approve_btn,
            self._reject_dup_btn,
            self._reject_irr_btn,
            self._skip_btn,
        ):
            btn.setFixedHeight(34)
            btn_layout.addWidget(btn)
        self._approve_btn.clicked.connect(self._approve)
        self._reject_dup_btn.clicked.connect(self._reject_duplicate)
        self._reject_irr_btn.clicked.connect(self._reject_irrelevant)
        self._skip_btn.clicked.connect(self._skip)
        right_layout.addLayout(btn_layout)
        splitter.addWidget(right_panel)

        splitter.setSizes([200, 650, 250])
        root.addWidget(splitter)

        # Keyboard shortcuts
        QShortcut(QKeySequence("A"), self).activated.connect(self._approve)
        QShortcut(QKeySequence("D"), self).activated.connect(self._reject_duplicate)
        QShortcut(QKeySequence("R"), self).activated.connect(self._reject_irrelevant)
        QShortcut(QKeySequence("S"), self).activated.connect(self._skip)

    # --------------------------------------------------------- navigation

    def _select_row(self, index: int) -> None:
        if not (0 <= index < len(self._rows)):
            return
        self._current_index = index
        self._list.setCurrentRow(index)
        self._update_metadata(index)
        self._navigate_pdf(index)

    def _on_list_row_changed(self, row: int) -> None:
        if row == self._current_index or row < 0:
            return
        self._current_index = row
        self._update_metadata(row)
        self._navigate_pdf(row)

    def _navigate_pdf(self, index: int) -> None:
        try:
            from PySide6.QtPdfWidgets import QPdfView

            if not isinstance(self._pdf_view, QPdfView):
                return
            pg_str = self._rows[index].get("consolidated_page_num", "1")
            page_0idx = max(0, int(pg_str) - 1)
            nav = self._pdf_view.pageNavigator()
            nav.jump(page_0idx, QPointF(0, 0), self._pdf_view.zoomFactor())
        except Exception:
            pass

    def _update_metadata(self, index: int) -> None:
        row = self._rows[index]
        src = row.get("source_pdf_path", "")
        filename = src.rsplit("/", 1)[-1].rsplit("\\", 1)[-1]
        score = float(row.get("total_hits", 0))
        threshold = float(row.get("min_hits_threshold", 0))
        self._meta_labels["Source"].setText(filename)
        self._meta_labels["Page"].setText(
            f"{row.get('page_num', '?')}  "
            f"(consolidated p.{row.get('consolidated_page_num', '?')})"
        )
        subtype = self._subtypes[index] if index < len(self._subtypes) else None
        type_display = {
            "intake":     "Intake / Initial",
            "discharge":  "Discharge / Termination",
            "progress":   "Progress Note",
            "imaging":    "Imaging Report",
            "legal":      "Legal / Affidavit",
            "medical":    "Medical Treatment",
            "vocational": "Vocational",
            "billing":    "Billing Record",
        }.get(subtype or "", "—")
        self._meta_labels["Type"].setText(type_display)
        self._meta_labels["Score"].setText(f"{score:.2f}")
        self._meta_labels["Threshold"].setText(f"{threshold:.2f}")
        self._meta_labels["Excluded because"].setText(
            row.get("exclusion_reasons", "").replace("|", ", ")
        )
        self._meta_labels["Categories"].setText(
            row.get("categories_matched", "—").replace("|", ", ") or "—"
        )
        keywords = row.get("keywords_hit", "")
        kw_short = ", ".join(keywords.split("|")[:6])
        if keywords.count("|") >= 6:
            kw_short += "…"
        self._meta_labels["Keywords"].setText(kw_short or "—")
        self._meta_labels["Dates"].setText(
            row.get("dates_on_page", "—").replace("|", ", ") or "—"
        )

    # --------------------------------------------------- triage helpers

    def _make_list_item_text(self, idx: int, row: dict) -> str:
        src = row.get("source_pdf_path", "")
        stem = src.rsplit("/", 1)[-1].rsplit("\\", 1)[-1].removesuffix(".pdf")
        pg = row.get("page_num", "?")
        score = float(row.get("total_hits", 0))
        subtype = self._subtypes[idx] if idx < len(self._subtypes) else None
        # [DUP] prefix whenever the page is marked as a duplicate (any source)
        if self._decisions.get(idx) == "rejected_duplicate":
            prefix = "[DUP]      "
        else:
            prefix = {
                "intake":     "[INTAKE]   ",
                "discharge":  "[DISCHARGE]",
                "progress":   "[PROG]     ",
                "imaging":    "[IMAGING]  ",
                "legal":      "[LEGAL]    ",
                "medical":    "[MEDICAL]  ",
                "vocational": "[VOC]      ",
                "billing":    "[BILLING]  ",
            }.get(subtype or "", "           ")
        return f"{prefix} {stem}\n           p.{pg}  score {score:.2f}"

    def _make_list_item(self, idx: int, row: dict) -> QListWidgetItem:
        return QListWidgetItem(self._make_list_item_text(idx, row))

    def _classify_subtype(self, row: dict) -> str | None:
        cats = {c for c in row.get("categories_matched", "").split("|") if c}
        if not cats:
            return None
        # Therapy subtype takes priority — most granular classification
        if cats & _THERAPY_CATS:
            kw = {k.lower() for k in row.get("keywords_hit", "").split("|") if k}
            if kw & _INTAKE_KW:
                return "intake"
            if kw & _DISCHARGE_KW:
                return "discharge"
            return "progress"
        # Non-therapy clinical categories
        if "IMAGING" in cats:
            return "imaging"
        if "INJURY_LEGAL" in cats:
            return "legal"
        if "MEDICAL_TREATMENT" in cats:
            return "medical"
        if "VOCATIONAL" in cats:
            return "vocational"
        if "BILLING" in cats:
            return "billing"
        return None

    def _run_smart_triage(self) -> None:
        """Clinical auto-categorization pass (no duplicate detection — use Find Duplicates).

        Approves therapy intake/discharge pages, applies first/last progress note
        fallback, pre-rejects remaining progress notes, and auto-approves non-therapy
        clinical pages. All suggestions are tagged source='smart_triage' so they
        appear in the distinct pale-blue / pale-lavender / pale-yellow color tier.
        """
        from collections import defaultdict

        src_items: dict[str, list[tuple[int, str | None]]] = defaultdict(list)
        for idx, row in enumerate(self._rows):
            src_items[row.get("source_pdf_path", "")].append((idx, self._subtypes[idx]))

        # Step 1: therapy — approve intake / discharge per source
        sources_with_intake_discharge: set[str] = set()
        intake_discharge_count = 0
        for src, items in src_items.items():
            if any(sub in ("intake", "discharge") for _, sub in items):
                sources_with_intake_discharge.add(src)
                for idx, sub in items:
                    if sub in ("intake", "discharge") and idx not in self._decisions:
                        self._decisions[idx] = "approved"
                        self._decision_sources[idx] = "smart_triage"
                        intake_discharge_count += 1

        # Step 2: therapy — first / last progress note fallback
        fallback_indices: set[int] = set()
        fallback_count = 0
        for src, items in src_items.items():
            if src in sources_with_intake_discharge:
                continue
            progress = [
                (idx, self._rows[idx])
                for idx, sub in items
                if sub == "progress"
            ]
            if not progress:
                continue

            def _min_date(r: dict) -> str:
                ds = [d for d in r.get("dates_on_page", "").split("|") if d]
                return min(ds) if ds else "9999-99-99"

            def _max_date(r: dict) -> str:
                ds = [d for d in r.get("dates_on_page", "").split("|") if d]
                return max(ds) if ds else "0000-00-00"

            first_idx = min(progress, key=lambda t: _min_date(t[1]))[0]
            last_idx  = max(progress, key=lambda t: _max_date(t[1]))[0]
            for idx in {first_idx, last_idx}:
                if idx not in self._decisions:
                    self._decisions[idx] = "approved"
                    self._decision_sources[idx] = "smart_triage"
                    fallback_indices.add(idx)
                    fallback_count += 1

        # Step 3: therapy — pre-reject remaining progress notes from handled sources
        handled_sources = sources_with_intake_discharge | {
            self._rows[i].get("source_pdf_path", "") for i in fallback_indices
        }
        irr_count = 0
        for src, items in src_items.items():
            if src not in handled_sources:
                continue
            for idx, sub in items:
                if sub == "progress" and idx not in self._decisions:
                    self._decisions[idx] = "rejected_irrelevant"
                    self._decision_sources[idx] = "smart_triage"
                    irr_count += 1

        # Step 4: auto-approve non-therapy clinical pages, tracking per-category counts
        _CAT_LABELS: list[tuple[str, str]] = [
            ("Legal / Affidavit",  "INJURY_LEGAL"),
            ("Imaging",            "IMAGING"),
            ("Billing",            "BILLING"),
            ("Medical Treatment",  "MEDICAL_TREATMENT"),
            ("Vocational",         "VOCATIONAL"),
        ]
        cat_approve_counts: dict[str, int] = {}
        for idx, row in enumerate(self._rows):
            if idx in self._decisions:
                continue
            cats = {c for c in row.get("categories_matched", "").split("|") if c}
            if cats & _AUTO_APPROVE_CATS and not (cats & _THERAPY_CATS):
                self._decisions[idx] = "approved"
                self._decision_sources[idx] = "smart_triage"
                for label, cat_key in _CAT_LABELS:
                    if cat_key in cats:
                        cat_approve_counts[label] = cat_approve_counts.get(label, 0) + 1
                        break

        self._refresh_list_colors()
        self._advance_to_next_unreviewed()
        self._triage_decisions = dict(self._decisions)
        self._save_draft()

        # Build summary — skip zero-count lines so each scan mode sees only relevant info
        summary_lines: list[str] = []
        if intake_discharge_count:
            summary_lines.append(
                f"Therapy — intake / discharge approved:     {intake_discharge_count}"
            )
        if fallback_count:
            summary_lines.append(
                f"Therapy — first / last progress approved:  {fallback_count}"
            )
        if irr_count:
            summary_lines.append(
                f"Therapy — remaining progress pre-rejected: {irr_count}"
            )
        for label, _ in _CAT_LABELS:
            count = cat_approve_counts.get(label, 0)
            if count:
                summary_lines.append(f"{label} pages approved:{'':>4}{count}")
        if not summary_lines:
            summary_lines.append("No pages matched any triage criteria.")

        summary_lines += [
            "",
            "Pale blue     = Smart Triage suggested approve.",
            "Pale lavender = Smart Triage suggested irrelevant.",
            "Run 'Find Duplicates' for visual duplicate detection.",
            "",
            "Review pre-selections and adjust as needed before exporting.",
        ]
        QMessageBox.information(
            self,
            "Smart Triage Complete",
            "\n".join(summary_lines),
        )

    def _run_dedup_pass(self) -> None:
        """Render all pages, compute perceptual hashes, cluster by visual similarity,
        then open _DedupDialog for each group so the user can confirm which to keep.

        Runs against ALL pages regardless of current decision state so every
        potential duplicate pair is surfaced for the feedback record.
        """
        try:
            from PySide6.QtPdf import QPdfDocument
            if not isinstance(self._pdf_doc, QPdfDocument):
                raise RuntimeError("not loaded")
        except Exception:
            QMessageBox.warning(
                self, "PDF Not Available",
                "The PDF viewer must be active to run Find Duplicates."
            )
            return

        if len(self._rows) < 2:
            QMessageBox.information(self, "Not Enough Pages",
                                    "Need at least 2 pages for duplicate detection.")
            return

        prev_label = self._progress_label.text()
        self._progress_label.setText("Find Duplicates: rendering pages…")
        QApplication.processEvents()

        hashes: list[tuple[int, int]] = []
        thumbnails: dict[int, QImage] = {}
        for idx, row in enumerate(self._rows):
            try:
                page_0idx = max(0, int(row.get("consolidated_page_num", "1")) - 1)
                qimg = self._pdf_doc.render(page_0idx, QSize(300, 390))
                if not qimg.isNull():
                    thumbnails[idx] = qimg
                    hashes.append((idx, _dhash(qimg)))
            except Exception:
                pass

        self._progress_label.setText(prev_label)

        if len(hashes) < 2:
            QMessageBox.information(self, "No Results",
                                    "Could not render enough pages for comparison.")
            return

        raw_clusters = _cluster_by_hash(hashes, threshold=10)
        groups = [c for c in raw_clusters if len(c) > 1]

        if not groups:
            QMessageBox.information(
                self, "No Duplicates Found",
                "No visually similar pages were detected across the full page set."
            )
            return

        # Record cluster membership before opening the dialog
        self._dedup_groups = {}
        self._dedup_canonical = set()
        for g_id, group in enumerate(groups):
            canonical = max(group, key=lambda i: float(self._rows[i].get("total_hits", 0)))
            self._dedup_canonical.add(canonical)
            for idx in group:
                self._dedup_groups[idx] = g_id

        dialog_groups = [
            [(idx, self._rows[idx], thumbnails[idx]) for idx in grp if idx in thumbnails]
            for grp in groups
        ]
        dialog_groups = [g for g in dialog_groups if len(g) >= 2]

        dlg = _DedupDialog(
            dialog_groups,
            decisions=self._decisions,
            decision_sources=self._decision_sources,
            parent=self,
        )
        if dlg.exec() != QDialog.DialogCode.Accepted:
            self._save_draft()  # persist dedup group membership even on cancel
            return

        keep_indices, reject_indices = dlg.results()
        changed = 0

        for idx in reject_indices:
            self._decisions[idx] = "rejected_duplicate"
            self._decision_sources[idx] = "find_duplicates"
            item = self._list.item(idx)
            if item:
                item.setText(self._make_list_item_text(idx, self._rows[idx]))
                color = self._item_color(idx)
                if color:
                    item.setBackground(color)
                item.setForeground(QColor(30, 30, 30))
            changed += 1

        for idx in keep_indices:
            if self._decisions.get(idx) not in ("approved",):
                self._decisions[idx] = "approved"
                self._decision_sources[idx] = "find_duplicates"
                item = self._list.item(idx)
                if item:
                    item.setText(self._make_list_item_text(idx, self._rows[idx]))
                    color = self._item_color(idx)
                    if color:
                        item.setBackground(color)
                    item.setForeground(QColor(30, 30, 30))
                changed += 1

        self._apply_filter()
        self._update_toolbar()
        self._save_draft()

        QMessageBox.information(
            self,
            "Find Duplicates Complete",
            f"Detected {len(groups)} group(s) of visually similar pages.\n"
            f"Applied decisions to {changed} page(s).\n\n"
            "Pale peach = Find Duplicates flagged duplicate.\n"
            "Use A / D / R to override any pre-selection as usual.",
        )

    def _item_color(self, idx: int) -> "QColor | None":
        """Return the background QColor for a list item based on decision + source."""
        dec = self._decisions.get(idx)
        if dec is None:
            return None
        src = self._decision_sources.get(idx, "user")
        rgb = _DECISION_COLORS.get((dec, src)) or _DECISION_COLORS.get((dec, "user"))
        return QColor(*rgb) if rgb else None

    # --------------------------------------------------------- decisions

    def _apply_filter(self) -> None:
        """Show only approved + duplicate rows when the filter toggle is active."""
        active = self._filter_btn.isChecked()
        for idx in range(self._list.count()):
            item = self._list.item(idx)
            if item is None:
                continue
            if active:
                dec = self._decisions.get(idx)
                item.setHidden(dec not in ("approved", "rejected_duplicate"))
            else:
                item.setHidden(False)

    def _advance_one(self) -> None:
        """Move to the next row (+1), skipping hidden items when filter is active."""
        active_filter = self._filter_btn.isChecked()
        start = self._current_index + 1
        for i in range(start, len(self._rows)):
            if active_filter:
                item = self._list.item(i)
                if item and item.isHidden():
                    continue
            self._select_row(i)
            return

    def _record_decision(self, decision: str) -> None:
        if self._current_index < 0:
            return
        was_decided = self._current_index in self._decisions
        self._decisions[self._current_index] = decision
        self._decision_sources[self._current_index] = "user"
        item = self._list.item(self._current_index)
        if item:
            item.setText(self._make_list_item_text(self._current_index, self._rows[self._current_index]))
            color = self._item_color(self._current_index)
            if color:
                item.setBackground(color)
            item.setForeground(QColor(30, 30, 30))
        self._apply_filter()
        self._update_toolbar()
        self._save_draft()
        if was_decided:
            # Reassigning a pre-decided page — advance by one rather than jumping
            # to the (potentially distant) next unreviewed page
            self._advance_one()
        else:
            self._advance_to_next_unreviewed()

    def _approve(self) -> None:
        self._record_decision("approved")

    def _reject_duplicate(self) -> None:
        self._record_decision("rejected_duplicate")

    def _reject_irrelevant(self) -> None:
        self._record_decision("rejected_irrelevant")

    def _skip(self) -> None:
        self._advance_to_next_unreviewed()

    def _advance_to_next_unreviewed(self) -> None:
        active_filter = self._filter_btn.isChecked()
        start = self._current_index + 1
        for i in range(start, len(self._rows)):
            if active_filter:
                item = self._list.item(i)
                if item and item.isHidden():
                    continue
            if i not in self._decisions:
                self._select_row(i)
                return
        for i in range(0, start):
            if active_filter:
                item = self._list.item(i)
                if item and item.isHidden():
                    continue
            if i not in self._decisions:
                self._select_row(i)
                return

    def _refresh_list_colors(self) -> None:
        for idx in range(self._list.count()):
            item = self._list.item(idx)
            if item is None:
                continue
            item.setText(self._make_list_item_text(idx, self._rows[idx]))
            color = self._item_color(idx)
            if color:
                item.setBackground(color)
                item.setForeground(QColor(30, 30, 30))
        self._apply_filter()

    def _update_toolbar(self) -> None:
        approved = sum(1 for d in self._decisions.values() if d == "approved")
        dup = sum(1 for d in self._decisions.values() if d == "rejected_duplicate")
        irr = sum(1 for d in self._decisions.values() if d == "rejected_irrelevant")
        remaining = len(self._rows) - len(self._decisions)
        self._progress_label.setText(
            f"{approved} approved / {dup} duplicate / {irr} irrelevant / {remaining} remaining"
        )

    # --------------------------------------------------------- export / apply

    def _export_feedback(self) -> None:
        import json
        from datetime import datetime

        # Load any existing records keyed by (source_pdf_path, page_num)
        existing: dict[tuple[str, int], dict] = {}
        if self._feedback_path.exists():
            for line in self._feedback_path.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                    key = (rec.get("source_pdf_path", ""), int(rec.get("page_num", 0)))
                    existing[key] = rec
                except (json.JSONDecodeError, ValueError):
                    pass

        # Merge current session decisions (overwrite prior decisions for same page)
        now = datetime.now().strftime("%Y-%m-%dT%H:%M:%S")
        for idx, decision in self._decisions.items():
            row = self._rows[idx]
            src = row.get("source_pdf_path", "")
            pg = int(row.get("page_num", 0))
            score = float(row.get("total_hits", 0))
            threshold = float(row.get("min_hits_threshold", 0))
            cats = row.get("categories_matched", "")
            keywords = row.get("keywords_hit", "")
            cat_names = [c for c in cats.split("|") if c]
            kw_list = [k for k in keywords.split("|") if k]
            if decision == "approved":
                label_str = "include"
                rejection_reason = None
            elif decision == "rejected_duplicate":
                label_str = "exclude"
                rejection_reason = "duplicate"
            else:
                label_str = "exclude"
                rejection_reason = "irrelevant"

            triage_dec = self._triage_decisions.get(idx)
            if triage_dec == "approved":
                triage_label_str: "str | None" = "include"
                triage_rejection_reason: "str | None" = None
            elif triage_dec == "rejected_duplicate":
                triage_label_str = "exclude"
                triage_rejection_reason = "duplicate"
            elif triage_dec == "rejected_irrelevant":
                triage_label_str = "exclude"
                triage_rejection_reason = "irrelevant"
            else:
                triage_label_str = None
                triage_rejection_reason = None

            # Find Duplicates provenance
            dedup_group_id = self._dedup_groups.get(idx)
            dedup_is_canonical: "bool | None" = (
                (idx in self._dedup_canonical) if dedup_group_id is not None else None
            )
            if dedup_group_id is None:
                dedup_suggestion = None
            elif idx in self._dedup_canonical:
                dedup_suggestion = "approve"
            else:
                dedup_suggestion = "reject_duplicate"

            rec = {
                "schema_version": 1,
                "label": label_str,
                "rejection_reason": rejection_reason,
                "scanner_decision": self._mode,        # "matched" | "unmatched"
                "decision_source": self._decision_sources.get(idx, "user"),
                "triage_label": triage_label_str,
                "triage_rejection_reason": triage_rejection_reason,
                "triage_override": triage_label_str is not None and triage_label_str != label_str,
                "find_duplicates_group_id": dedup_group_id,
                "find_duplicates_is_canonical": dedup_is_canonical,
                "find_duplicates_suggestion": dedup_suggestion,
                "review_timestamp": now,
                "scan_settings": self._scan_settings,
                "source_pdf_path": src,
                "page_num": pg,
                "total_hits": score,
                "min_hits_threshold": threshold,
                "score_ratio": round(score / threshold, 4) if threshold else 0.0,
                "extraction_method": row.get("extraction_method", ""),
                "exclusion_reasons": [
                    r for r in row.get("exclusion_reasons", "").split("|") if r
                ],
                "categories_matched": cat_names,
                "keywords_hit": kw_list,
                "keyword_count": len(kw_list),
                "dates_on_page": [
                    d for d in row.get("dates_on_page", "").split("|") if d
                ],
                "date_count": len(
                    [d for d in row.get("dates_on_page", "").split("|") if d]
                ),
            }
            existing[(src, pg)] = rec

        with open(self._feedback_path, "w", encoding="utf-8") as f:
            for rec in existing.values():
                f.write(json.dumps(rec) + "\n")

        # Draft is superseded by the committed feedback file — clear it
        try:
            if self._draft_path.exists():
                self._draft_path.unlink()
        except OSError:
            pass

        QDesktopServices.openUrl(QUrl.fromLocalFile(str(self._output_dir)))
        QMessageBox.information(
            self,
            "Feedback Exported",
            f"Feedback written to:\n{self._feedback_path}",
        )

    def _apply_approved(self) -> None:
        if self._mode == "matched":
            remove_count = sum(
                1 for d in self._decisions.values()
                if d in ("rejected_duplicate", "rejected_irrelevant")
            )
            if remove_count == 0:
                QMessageBox.information(
                    self, "Nothing to Remove",
                    "No pages have been marked for removal yet."
                )
                return
            reply = QMessageBox.question(
                self,
                "Remove Pages from Consolidated",
                f"Remove {remove_count} rejected page(s) from _consolidated.pdf?\n\n"
                "This will modify the output file. Make sure you have exported "
                "your feedback first.",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No,
            )
        else:
            approved_count = sum(
                1 for d in self._decisions.values() if d == "approved"
            )
            if approved_count == 0:
                QMessageBox.information(
                    self, "Nothing to Apply", "No pages have been approved yet."
                )
                return
            reply = QMessageBox.question(
                self,
                "Apply Approved Pages",
                f"Append {approved_count} approved page(s) to _consolidated.pdf "
                f"and rebuild _consolidated_unmatched.pdf?\n\n"
                f"This will modify the output files. Make sure you have exported "
                f"your feedback first.",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No,
            )
        if reply != QMessageBox.StandardButton.Yes:
            return

        # Export feedback first so the backend has fresh data
        self._export_feedback()

        self._apply_btn.setEnabled(False)
        self._fb_worker = _FeedbackWorker(
            str(self._output_dir), str(self._feedback_path), mode=self._mode
        )
        self._fb_worker.progress.connect(self._progress_label.setText)
        self._fb_worker.finished.connect(self._on_apply_finished)
        self._fb_worker.error.connect(self._on_apply_error)
        self._fb_worker.start()

    def _on_apply_finished(self, primary: int, _secondary: int, skipped: int) -> None:
        self._apply_result = (primary, skipped)
        self._apply_btn.setEnabled(True)
        if self._mode == "matched":
            msg = f"Consolidated PDF rebuilt — {primary} page(s) kept."
            skip_note = f"\n{skipped} page(s) skipped — source file(s) not found."
        else:
            msg = f"Applied {primary} approved page(s)."
            skip_note = (
                f"\n{skipped} page(s) could not be applied — "
                f"source file(s) not found at stored path."
            )
        if skipped:
            msg += skip_note
        QMessageBox.information(self, "Apply Complete", msg)
        self._update_toolbar()
        self.accept()

    def _on_apply_error(self, message: str) -> None:
        self._apply_btn.setEnabled(True)
        QMessageBox.critical(self, "Apply Failed", message)


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

    app: QApplication = QApplication.instance() or QApplication(sys.argv)  # type: ignore[assignment]
    app.setWindowIcon(QIcon(":/icons/app_icon.svg"))
    dialog = ScannerDialog()
    dialog.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
