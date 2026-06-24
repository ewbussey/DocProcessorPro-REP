from __future__ import annotations

import re
import sys
from pathlib import Path

from PySide6.QtCore import QSettings, QSize, Qt, QTimer, QUrl
from PySide6.QtGui import QDesktopServices
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSpinBox,
    QVBoxLayout,
)

from ._gui_utils import _apply_win_minmax, _apply_app_icon
from ._workers import _LlmBatchWorker, _ScanWorker, _UpdateChecker, _UpdateDownloader
from ._review_dialog import ReviewDialog

# Matches "{LastName(s)}, {FirstName(s)} {CaseNumber}" folder names.
# Allows hyphens, apostrophes, and accented characters in name components.
_CLAIMANT_FOLDER_RE = re.compile(
    r"^[A-Za-z’’À-ɏ\-]+(?:\s+[A-Za-z’’À-ɏ\-]+)*"
    r",\s+\S.*\s+\S+\s*$"
)

_DEFAULT_OUTPUT_SENTINEL = "Default Output Path"

class ScannerDialog(QDialog):
    def __init__(self) -> None:
        super().__init__()
        _apply_win_minmax(self)
        _apply_app_icon(self)
        self.setWindowTitle("DocProcessorPro — Keyword Scanner")
        self.setMinimumWidth(620)
        self.resize(660, 560)
        self._worker: _ScanWorker | None = None
        self._llm_worker: _LlmBatchWorker | None = None
        self._update_checker: _UpdateChecker | None = None
        self._update_downloader: _UpdateDownloader | None = None
        self._settings = QSettings("DocProcessorPro", "KeywordScanner")
        from . import _llm_client
        _llm_client.configure(
            str(self._settings.value("llm_service_url", _llm_client._DEFAULT_URL))
        )
        self._saved_min_hits: float | None = None
        self._last_output_dir: "Path | None" = None
        self._last_scan_settings: "dict | None" = None
        self._last_browsed_dir: str = ""
        self._auto_derived_output: str = ""  # last value written by auto-derivation
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
        output_default = QPushButton("Default")
        output_default.setToolTip(
            "Derive output path from the input folder's claimant structure:\n"
            "{Output root}/{Initial}/{Last, First CaseNum}/"
        )
        output_default.clicked.connect(self._set_default_output)
        output_row = QHBoxLayout()
        output_row.addWidget(self._output_edit)
        output_row.addWidget(output_browse)
        output_row.addWidget(output_default)
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
        self._matched_review_btn = QPushButton("Review Records Pages")
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

        # Settings button — right-aligned
        settings_row = QHBoxLayout()
        settings_row.addStretch()
        settings_btn = QPushButton("Settings…")
        settings_btn.setToolTip("Configure application settings (output root directory, etc.)")
        settings_btn.clicked.connect(self._open_settings)
        settings_row.addWidget(settings_btn)
        root.addLayout(settings_row)

        # Status label
        self._status_label = QLabel("Ready.")
        self._status_label.setWordWrap(True)
        root.addWidget(self._status_label)

        # Restore last-used directories
        self._input_edit.setText(str(self._settings.value("last_input_dir", "")))
        self._output_edit.setText(
            str(self._settings.value("last_output_dir", _DEFAULT_OUTPUT_SENTINEL))
        )

    # FILTER TOGGLE SLOTS

    def _on_clinical_filter_toggled(self) -> None:
        pass  # reserved for future cross-filter exclusion logic

    # BROWSE SLOTS

    def _browse_input(self) -> None:
        start = self._input_edit.text().strip() or self._last_browsed_dir
        path = QFileDialog.getExistingDirectory(self, "Select Input Folder", start)
        if path:
            self._input_edit.setText(path)
            self._settings.setValue("last_input_dir", path)
            self._last_browsed_dir = path
            self._try_derive_output(path)

    def _try_derive_output(self, input_path: str) -> None:
        """Auto-fill the output field from the claimant folder structure, if possible.

        Only overwrites the output field when it is empty or still shows the
        previous auto-derived value (i.e. the user has not manually changed it).
        """
        derived = self._derive_output_from_input(input_path)
        if not derived:
            return
        current = self._output_edit.text().strip()
        if current == _DEFAULT_OUTPUT_SENTINEL:
            return  # user explicitly chose Default mode — resolve at scan time
        if current and current != self._auto_derived_output:
            return  # user has manually set the output — leave it alone
        self._output_edit.setText(derived)
        self._auto_derived_output = derived
        self._settings.setValue("last_output_dir", derived)

    def _derive_output_from_input(self, input_path: str) -> str | None:
        """Walk up from input_path to find a claimant folder, then build the
        canonical output path: {root}/dpp_outputs/{Initial}/{ClaimantFolder}/
        """
        p = Path(input_path)
        if not p.exists():
            return None

        # Walk toward the filesystem root looking for a claimant-pattern folder.
        current = p
        claimant_folder: Path | None = None
        while True:
            if _CLAIMANT_FOLDER_RE.match(current.name):
                claimant_folder = current
                break
            parent = current.parent
            if parent == current:
                break
            current = parent

        if claimant_folder is None:
            return None

        letter_folder = claimant_folder.parent
        if len(letter_folder.name) == 1 and letter_folder.name.isalpha():
            initial = letter_folder.name.upper()
        else:
            # Letter folder absent — derive initial from last name
            last_name = claimant_folder.name.split(",")[0].strip()
            initial = last_name[0].upper() if last_name else "X"

        output_root = Path(
            str(self._settings.value(
                "output_root",
                str(Path.home() / "Documents" / "dpp_outputs"),
            ))
        )
        return str(output_root / initial / claimant_folder.name)

    def _browse_output(self) -> None:
        start = self._output_edit.text().strip() or self._last_browsed_dir
        path = QFileDialog.getExistingDirectory(self, "Select Output Folder", start)
        if path:
            self._output_edit.setText(path)
            self._settings.setValue("last_output_dir", path)

    def _open_settings(self) -> None:
        dlg = _AppSettingsDialog(self._settings, parent=self)
        dlg.exec()

    def _set_default_output(self) -> None:
        self._output_edit.setText(_DEFAULT_OUTPUT_SENTINEL)
        self._auto_derived_output = _DEFAULT_OUTPUT_SENTINEL

    # SCAN

    def _run_scan(self) -> None:
        input_dir = self._input_edit.text().strip()
        output_dir = self._output_edit.text().strip()

        if not input_dir:
            QMessageBox.warning(self, "Missing Input", "Please select an input folder.")
            return
        if output_dir == _DEFAULT_OUTPUT_SENTINEL:
            derived = self._derive_output_from_input(input_dir)
            if not derived:
                QMessageBox.warning(
                    self,
                    "Cannot Derive Output",
                    "Could not find a claimant folder in the input path.\n"
                    "Expected structure: …/{Initial}/{Last, First CaseNum}/…",
                )
                return
            output_dir = derived
            self._output_edit.setText(derived)
            self._auto_derived_output = derived
        if not output_dir:
            QMessageBox.warning(
                self, "Missing Output", "Please select an output folder."
            )
            return

        self._run_btn.setEnabled(False)
        self._status_label.setText("Scanning… this may take a while for large batches.")

        if self._doc_section_check.isChecked():
            require: frozenset[str] | None = frozenset({"DOCUMENT_TYPE"})
        else:
            require = None

        # Derive a human-readable mode label for feedback tagging
        if self._doc_section_check.isChecked() and self._require_anchor_check.isChecked():
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
            require_anchor=self._require_anchor_check.isChecked(),
        )
        self._worker.progress.connect(self._status_label.setText)
        self._worker.finished.connect(self._on_finished)
        self._worker.error.connect(self._on_error)
        self._worker.start()

    def _on_finished(
        self, pdf_count: int, match_count: int, exclusion_count: int
    ) -> None:
        self._run_btn.setEnabled(True)
        out = self._output_edit.text().strip()
        self._settings.setValue("last_input_dir", self._input_edit.text().strip())
        self._settings.setValue("last_output_dir", out)
        self._last_output_dir = Path(out)

        if exclusion_count > 0:
            self._review_btn.setEnabled(True)
            self._review_btn.setText(
                f"Review Excluded Pages ({exclusion_count} to review)"
            )
        if match_count > 0:
            self._matched_review_btn.setEnabled(True)
            self._matched_review_btn.setText(
                f"Review Records Pages ({match_count} matched)"
            )

        # If the LLM service is reachable, run the batch pass before opening the
        # review dialog so pre-classifications are ready when the dialog loads.
        from . import _llm_client
        if _llm_client.is_available():
            self._status_label.setText(
                f"Scan done ({pdf_count} PDF(s), {match_count} match(es)). "
                f"Running LLM analysis…"
            )
            self._llm_worker = _LlmBatchWorker(out)
            self._llm_worker.progress.connect(self._status_label.setText)
            self._llm_worker.finished.connect(self._on_llm_batch_finished)
            self._llm_worker.error.connect(self._on_llm_batch_error)
            self._llm_worker.start()
        else:
            self._status_label.setText(
                f"Done. Processed {pdf_count} PDF(s); {match_count} match(es), "
                f"{exclusion_count} reviewable exclusion(s). "
                f"Output saved to: {out}"
            )
            self._open_combined_review()

    def _on_llm_batch_finished(self, processed: int, _skipped: int) -> None:
        out = str(self._last_output_dir) if self._last_output_dir else ""
        self._status_label.setText(
            f"LLM analysis complete — {processed} page(s) pre-classified. "
            f"Output saved to: {out}"
        )
        self._open_combined_review()

    def _on_llm_batch_error(self, msg: str) -> None:
        out = str(self._last_output_dir) if self._last_output_dir else ""
        self._status_label.setText(
            f"LLM analysis unavailable ({msg}). Output saved to: {out}"
        )
        self._open_combined_review()

    def _open_combined_review(self) -> None:
        """Auto-open the combined review dialog when both PDF and CSV are present."""
        if self._last_output_dir is None:
            return
        review_pdf = self._last_output_dir / "_review" / "_consolidated_review.pdf"
        review_csv = self._last_output_dir / "_review" / "_consolidated_review_manifest.csv"
        if review_pdf.exists() and review_csv.exists():
            dlg = ReviewDialog(
                review_pdf,
                review_csv,
                self._last_output_dir,
                scan_settings=self._last_scan_settings,
                mode="combined",
                parent=self,
            )
            dlg.exec()

        QDesktopServices.openUrl(QUrl.fromLocalFile(str(self._last_output_dir)))

    def _open_review_dialog(self) -> None:
        if self._last_output_dir is None:
            QMessageBox.warning(
                self, "No Scan Data", "Run a scan first to generate review data."
            )
            return
        pdf_path = self._last_output_dir / "_unmatched" / "_consolidated_unmatched.pdf"
        csv_path = self._last_output_dir / "_unmatched" / "_consolidated_unmatched_manifest.csv"
        if not pdf_path.exists():
            QMessageBox.warning(
                self,
                "No Excluded Pages",
                "No consolidated unmatched PDF found. "
                "Run a scan that produces reviewable exclusions first.",
            )
            return
        dlg = ReviewDialog(
            pdf_path,
            csv_path,
            self._last_output_dir,
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
        pdf_path = self._last_output_dir / "_records" / "_consolidated_records.pdf"
        csv_path = self._last_output_dir / "_records" / "_consolidated_records_manifest.csv"
        if not pdf_path.exists():
            QMessageBox.warning(
                self,
                "No Records PDF Found",
                "No _consolidated_records.pdf found. Run a scan that produces records matches first.",
            )
            return
        if not csv_path.exists():
            QMessageBox.warning(
                self,
                "No Records Manifest Found",
                "_consolidated_records_manifest.csv not found.\n\n"
                "Re-run the scan to generate the manifest required for records review.",
            )
            return
        dlg = ReviewDialog(
            pdf_path,
            csv_path,
            self._last_output_dir,
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
            "Review files (*_consolidated_review.pdf *_consolidated_unmatched.pdf "
            "*_consolidated_records.pdf *_consolidated_bills.pdf "
            "*_review_draft.json *_matched_review_draft.json *_feedback.jsonl);;"
            "All files (*)",
        )
        if not selected:
            return
        selected_path = Path(selected)
        # If the user navigated into a subfolder, resolve to the parent output dir.
        _SUBDIR_NAMES = {"_feedback", "_sidecars", "_review", "_records", "_bills", "_unmatched", "_depositions"}
        candidate_dir = selected_path.parent
        if candidate_dir.name in _SUBDIR_NAMES:
            candidate_dir = candidate_dir.parent
        output_dir = candidate_dir
        self._last_browsed_dir = str(output_dir)

        # Auto-detect which review to open based on filename selected
        selected_name = selected_path.name
        open_combined = selected_name in (
            "_consolidated_review.pdf",
            "_combined_review_draft.json",
            "_combined_feedback.jsonl",
        )
        open_matched = not open_combined and (
            selected_name in (
                "_consolidated_records.pdf",
                "_consolidated.pdf",          # legacy name
                "_matched_review_draft.json",
                "_matched_feedback.jsonl",
            ) or selected_name.endswith("_records_manifest.csv")
            or selected_name.endswith("_matched_manifest.csv")
        )

        if open_combined:
            pdf_path = output_dir / "_review" / "_consolidated_review.pdf"
            if not pdf_path.exists():
                pdf_path = output_dir / "_consolidated_review.pdf"  # legacy flat
            csv_path = (
                output_dir / "_review" / "_consolidated_review_manifest.csv"
                if (output_dir / "_review" / "_consolidated_review_manifest.csv").exists()
                else output_dir / "_consolidated_review_manifest.csv"
            )
            if not pdf_path.exists():
                QMessageBox.warning(
                    self,
                    "No Review PDF Found",
                    "No _consolidated_review.pdf found in the selected folder.",
                )
                return
            settings_dlg = _ScanSettingsDialog(parent=self)
            if settings_dlg.exec() != QDialog.DialogCode.Accepted:
                return
            dlg = ReviewDialog(
                pdf_path,
                csv_path,
                output_dir,
                scan_settings=settings_dlg.scan_settings,
                mode="combined",
                parent=self,
            )
        elif open_matched:
            # Look in _records/ subdir first; fall back to root (legacy flat layout)
            pdf_path = output_dir / "_records" / "_consolidated_records.pdf"
            if not pdf_path.exists():
                pdf_path = output_dir / "_consolidated_records.pdf"
            if not pdf_path.exists():
                pdf_path = output_dir / "_consolidated.pdf"  # oldest legacy name
            records_dir = output_dir / "_records"
            csv_path = (
                records_dir / "_consolidated_records_manifest.csv"
                if (records_dir / "_consolidated_records_manifest.csv").exists()
                else output_dir / "_consolidated_records_manifest.csv"
                if (output_dir / "_consolidated_records_manifest.csv").exists()
                else output_dir / "_consolidated_matched_manifest.csv"
            )
            if not pdf_path.exists():
                QMessageBox.warning(
                    self,
                    "No Records PDF Found",
                    "No _consolidated_records.pdf found in the selected folder.",
                )
                return
            if not csv_path.exists():
                QMessageBox.warning(
                    self,
                    "No Records Manifest Found",
                    "_consolidated_records_manifest.csv not found.\n\n"
                    "Re-run the scan to generate this manifest.",
                )
                return
            settings_dlg = _ScanSettingsDialog(parent=self)
            if settings_dlg.exec() != QDialog.DialogCode.Accepted:
                return
            dlg = ReviewDialog(
                pdf_path,
                csv_path,
                output_dir,
                scan_settings=settings_dlg.scan_settings,
                mode="matched",
                parent=self,
            )
        else:
            # Look in _unmatched/ subdir first; fall back to root (legacy flat layout)
            pdf_path = output_dir / "_unmatched" / "_consolidated_unmatched.pdf"
            if not pdf_path.exists():
                pdf_path = output_dir / "_consolidated_unmatched.pdf"
            csv_path = (
                output_dir / "_unmatched" / "_consolidated_unmatched_manifest.csv"
                if (output_dir / "_unmatched" / "_consolidated_unmatched_manifest.csv").exists()
                else output_dir / "_consolidated_unmatched_manifest.csv"
            )
            if not pdf_path.exists():
                QMessageBox.warning(
                    self,
                    "No Unmatched PDF Found",
                    "No _consolidated_unmatched.pdf found in the selected folder.\n\n"
                    "Please select a file inside a folder that contains scan output.",
                )
                return
            settings_dlg = _ScanSettingsDialog(parent=self)
            if settings_dlg.exec() != QDialog.DialogCode.Accepted:
                return
            dlg = ReviewDialog(
                pdf_path,
                csv_path,
                output_dir,
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
                msg = (
                    f"Review complete — {primary} page(s) kept in consolidated output."
                )
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



class _AppSettingsDialog(QDialog):
    """Application-level settings (output root directory, etc.)."""

    def __init__(self, settings: QSettings, parent=None) -> None:
        super().__init__(parent)
        _apply_win_minmax(self)
        _apply_app_icon(self)
        self.setWindowTitle("Settings")
        self.setMinimumWidth(480)
        self._settings = settings
        self._build_ui()

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setSpacing(12)
        root.setContentsMargins(16, 16, 16, 16)

        form = QFormLayout()
        form.setSpacing(8)

        _default_root = str(Path.home() / "Documents" / "dpp_outputs")
        self._output_root_edit = QLineEdit(
            str(self._settings.value("output_root", _default_root))
        )
        self._output_root_edit.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed
        )
        self._output_root_edit.setToolTip(
            "Base directory for the Default output path.\n"
            "Default button will produce: {this folder}/{Initial}/{Last, First CaseNum}/"
        )
        browse_btn = QPushButton("Browse…")
        browse_btn.clicked.connect(self._browse_output_root)
        row = QHBoxLayout()
        row.addWidget(self._output_root_edit)
        row.addWidget(browse_btn)
        form.addRow("Output root:", row)

        from . import _llm_client
        self._llm_url_edit = QLineEdit(
            str(self._settings.value("llm_service_url", _llm_client._DEFAULT_URL))
        )
        self._llm_url_edit.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed
        )
        self._llm_url_edit.setToolTip(
            "Base URL of the local LLM inference service.\n"
            "Leave as default if running on the same machine (port 8765).\n"
            "The service is optional — DocProcessorPro works without it."
        )
        form.addRow("LLM service URL:", self._llm_url_edit)

        root.addLayout(form)

        from PySide6.QtWidgets import QDialogButtonBox
        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self._save_and_accept)
        buttons.rejected.connect(self.reject)
        root.addWidget(buttons)

    def _browse_output_root(self) -> None:
        path = QFileDialog.getExistingDirectory(
            self, "Select Output Root Folder", self._output_root_edit.text().strip()
        )
        if path:
            self._output_root_edit.setText(path)

    def _save_and_accept(self) -> None:
        path = self._output_root_edit.text().strip()
        if path:
            self._settings.setValue("output_root", path)
        llm_url = self._llm_url_edit.text().strip()
        if llm_url:
            self._settings.setValue("llm_service_url", llm_url)
            from . import _llm_client
            _llm_client.configure(llm_url)
        self.accept()



class _ScanSettingsDialog(QDialog):
    """Prompts the user for the scan settings that produced an existing output folder.

    Used when loading a prior review session so exported feedback can be tagged
    with the correct scan profile. All fields default to the most common values;
    the user may leave the mode as 'Don't specify' to pass scan_settings=None.
    """

    _MODE_ITEMS: list[tuple[str, "str | None"]] = [
        ("Don't specify", None),
        ("Standard (full clinical scan)", "standard"),
        ("Affidavits + Bills only", "affidavits_bills"),
        ("Document type filter", "document_type_filter"),
        ("Clinical anchor required", "clinical_anchor"),
        ("Document type + clinical anchor", "document_type_and_anchor"),
    ]

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        _apply_win_minmax(self)
        _apply_app_icon(self)
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
                "standard": (None, False),
                "affidavits_bills": (["BILLING", "INJURY_LEGAL"], False),
                "document_type_filter": (["DOCUMENT_TYPE"], False),
                "clinical_anchor": (None, True),
                "document_type_and_anchor": (["DOCUMENT_TYPE"], True),
            }
            require_categories, require_anchor = _require[mode]
            self._result = {
                "scan_mode": mode,
                "min_hits": self._min_hits_spin.value(),
                "page_buffer": self._page_buffer_spin.value(),
                "require_categories": require_categories,
                "require_anchor": require_anchor,
            }
        self.accept()

    @property
    def scan_settings(self) -> "dict | None":
        return self._result


