"""
DocProcessorPro — Keyword Scanner GUI

Select an input folder and an output folder, then run the keyword scanner against every PDF
in the input folder. Matching pages are written to {stem}_matched.pdf and a {stem}_manifest.csv in the output folder
prior to consolidation.
"""

from __future__ import annotations

import sys
from pathlib import Path

from PySide6.QtCore import QPointF, QSettings, QThread, QTimer, QUrl, Signal
from PySide6.QtGui import QColor, QDesktopServices, QKeySequence, QShortcut
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QDialog,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
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
        self.setWindowTitle("DocProcessorPro — Keyword Scanner")
        self.setMinimumWidth(520)
        self._worker: _ScanWorker | None = None
        self._update_checker: _UpdateChecker | None = None
        self._update_downloader: _UpdateDownloader | None = None
        self._settings = QSettings("DocProcessorPro", "KeywordScanner")
        self._saved_min_hits: float | None = None
        self._last_output_dir: "Path | None" = None
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
        self._page_buffer_spin.setValue(2)
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
        self._affidavit_bills_check = QCheckBox("Affidavits & bills only")
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
        start = self._input_edit.text().strip()
        path = QFileDialog.getExistingDirectory(self, "Select Input Folder", start)
        if path:
            self._input_edit.setText(path)
            self._settings.setValue("last_input_dir", path)

    def _browse_output(self) -> None:
        start = self._output_edit.text().strip()
        path = QFileDialog.getExistingDirectory(self, "Select Output Folder", start)
        if path:
            self._output_edit.setText(path)
            self._settings.setValue("last_output_dir", path)

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
        dlg = ReviewDialog(pdf_path, csv_path, self._last_output_dir, parent=self)
        dlg.exec()

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
    """Runs apply_feedback() off the main thread."""

    progress = Signal(str)
    finished = Signal(int, int, int)  # (pages_approved, pages_rejected, pages_skipped)
    error = Signal(str)

    def __init__(self, output_dir: str, feedback_path: str) -> None:
        super().__init__()
        self._output_dir = output_dir
        self._feedback_path = feedback_path

    def run(self) -> None:
        try:
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


class ReviewDialog(QDialog):
    """Three-panel review window for auditing keyword-scanner exclusions.

    Left:   list of excluded pages sorted by score descending
    Centre: QPdfView showing the consolidated unmatched PDF
    Right:  metadata panel + Approve / Reject / Skip buttons

    Keyboard shortcuts: A = approve, R = reject, S = skip.
    After each decision the view auto-advances to the next unreviewed page.
    """

    def __init__(
        self,
        consolidated_unmatched_pdf: "Path",
        unmatched_manifest_csv: "Path",
        output_dir: "Path",
        parent=None,
    ) -> None:
        from pathlib import Path as _Path

        super().__init__(parent)
        self.setWindowTitle("DocProcessorPro — Review Excluded Pages")
        self.setMinimumSize(1100, 700)

        self._output_dir = _Path(output_dir)
        self._feedback_path = self._output_dir / "_feedback.jsonl"
        self._fb_worker: _FeedbackWorker | None = None

        self._rows: list[dict] = []
        self._decisions: dict[int, str] = {}  # row_index → "approved" | "rejected"
        self._current_index: int = -1

        self._load_manifest(unmatched_manifest_csv)
        self._build_ui()
        self._load_pdf(consolidated_unmatched_pdf)
        self._load_existing_feedback()

        if self._rows:
            self._select_row(0)

    # ------------------------------------------------------------------ setup

    def _load_manifest(self, csv_path: "Path") -> None:
        import csv as _csv

        if not csv_path.exists():
            return
        with open(csv_path, newline="", encoding="utf-8") as f:
            rows = list(_csv.DictReader(f))
        # Sort highest score first so the most-interesting pages appear at top
        self._rows = sorted(
            rows, key=lambda r: float(r.get("total_hits", 0)), reverse=True
        )

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
            # Match to a row by source_pdf_path + page_num
            for idx, row in enumerate(self._rows):
                if (
                    row.get("source_pdf_path") == src
                    and int(row.get("page_num", 0)) == int(pg)
                ):
                    self._decisions[idx] = (
                        "approved" if label == "include" else "rejected"
                    )
                    break
        self._refresh_list_colors()
        self._update_toolbar()

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(8, 8, 8, 8)
        root.setSpacing(6)

        # Toolbar row
        toolbar = QHBoxLayout()
        self._progress_label = QLabel("0 approved / 0 rejected / 0 remaining")
        toolbar.addWidget(self._progress_label)
        toolbar.addStretch()
        export_btn = QPushButton("Export Feedback")
        export_btn.setToolTip("Write decisions to _feedback.jsonl in the output folder")
        export_btn.clicked.connect(self._export_feedback)
        toolbar.addWidget(export_btn)
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
        for row in self._rows:
            src = row.get("source_pdf_path", "")
            stem = src.rsplit("/", 1)[-1].rsplit("\\", 1)[-1].removesuffix(".pdf")
            pg = row.get("page_num", "?")
            score = float(row.get("total_hits", 0))
            item = QListWidgetItem(f"{stem}\np.{pg}  score {score:.2f}")
            self._list.addItem(item)
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
        self._approve_btn = QPushButton("Approve  (A)")
        self._reject_btn = QPushButton("Reject   (R)")
        self._skip_btn = QPushButton("Skip     (S)")
        for btn in (self._approve_btn, self._reject_btn, self._skip_btn):
            btn.setFixedHeight(34)
            btn_layout.addWidget(btn)
        self._approve_btn.clicked.connect(self._approve)
        self._reject_btn.clicked.connect(self._reject)
        self._skip_btn.clicked.connect(self._skip)
        right_layout.addLayout(btn_layout)
        splitter.addWidget(right_panel)

        splitter.setSizes([200, 650, 250])
        root.addWidget(splitter)

        # Keyboard shortcuts
        QShortcut(QKeySequence("A"), self).activated.connect(self._approve)
        QShortcut(QKeySequence("R"), self).activated.connect(self._reject)
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

    # --------------------------------------------------------- decisions

    def _record_decision(self, decision: str) -> None:
        if self._current_index < 0:
            return
        self._decisions[self._current_index] = decision
        item = self._list.item(self._current_index)
        if item:
            if decision == "approved":
                item.setBackground(QColor(200, 255, 200))
            elif decision == "rejected":
                item.setBackground(QColor(255, 200, 200))
        self._update_toolbar()
        self._advance_to_next_unreviewed()

    def _approve(self) -> None:
        self._record_decision("approved")

    def _reject(self) -> None:
        self._record_decision("rejected")

    def _skip(self) -> None:
        self._advance_to_next_unreviewed()

    def _advance_to_next_unreviewed(self) -> None:
        start = self._current_index + 1
        for i in range(start, len(self._rows)):
            if i not in self._decisions:
                self._select_row(i)
                return
        # Wrap around from the beginning
        for i in range(0, start):
            if i not in self._decisions:
                self._select_row(i)
                return

    def _refresh_list_colors(self) -> None:
        for idx, decision in self._decisions.items():
            item = self._list.item(idx)
            if item:
                if decision == "approved":
                    item.setBackground(QColor(200, 255, 200))
                elif decision == "rejected":
                    item.setBackground(QColor(255, 200, 200))

    def _update_toolbar(self) -> None:
        approved = sum(1 for d in self._decisions.values() if d == "approved")
        rejected = sum(1 for d in self._decisions.values() if d == "rejected")
        remaining = len(self._rows) - len(self._decisions)
        self._progress_label.setText(
            f"{approved} approved / {rejected} rejected / {remaining} remaining"
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
            # Build per-category score map
            cat_names = [c for c in cats.split("|") if c]
            kw_list = [k for k in keywords.split("|") if k]
            rec = {
                "schema_version": 1,
                "label": "include" if decision == "approved" else "exclude",
                "review_timestamp": now,
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

        QMessageBox.information(
            self,
            "Feedback Exported",
            f"Feedback written to:\n{self._feedback_path}",
        )

    def _apply_approved(self) -> None:
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

        # Export feedback first so apply_feedback() has fresh data
        self._export_feedback()

        self._apply_btn.setEnabled(False)
        self._fb_worker = _FeedbackWorker(
            str(self._output_dir), str(self._feedback_path)
        )
        self._fb_worker.progress.connect(self._progress_label.setText)
        self._fb_worker.finished.connect(self._on_apply_finished)
        self._fb_worker.error.connect(self._on_apply_error)
        self._fb_worker.start()

    def _on_apply_finished(self, approved: int, _rejected: int, skipped: int) -> None:
        self._apply_btn.setEnabled(True)
        msg = f"Applied {approved} approved page(s)."
        if skipped:
            msg += (
                f"\n{skipped} page(s) could not be applied — "
                f"source file(s) not found at stored path."
            )
        QMessageBox.information(self, "Apply Complete", msg)
        self._update_toolbar()

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

    app = QApplication.instance() or QApplication(sys.argv)
    dialog = ScannerDialog()
    dialog.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
