from __future__ import annotations

import re
import sys
from pathlib import Path
from typing import NamedTuple

from PySide6.QtCore import QPointF, QSettings, QSize, QStringListModel, QTimer, QUrl, Qt
from PySide6.QtGui import (
    QColor,
    QDesktopServices,
    QIcon,
    QImage,
    QKeySequence,
    QPainter,
    QPixmap,
    QShortcut,
)
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QCompleter,
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

from ._gui_utils import _apply_win_minmax, _apply_app_icon
from ._dedup import (
    _DedupPrefetchWorker,
    _DedupDialog,
    _dhash,
    _cluster_by_hash,
    _DEDUP_THRESHOLD,
    _DEDUP_CONFIRM_MAX_DIFF,
    _pixel_mad,
)
from ._workers import _FeedbackWorker

_DECISION_COLORS: dict[tuple[str, str], tuple[int, int, int]] = {
    ("approved", "user"): (200, 255, 200),  # green
    ("rejected_duplicate", "user"): (255, 210, 140),  # orange
    ("rejected_irrelevant", "user"): (255, 200, 200),  # red
    ("approved", "smart_triage"): (185, 215, 255),  # pale blue
    ("rejected_duplicate", "smart_triage"): (255, 255, 185),  # pale yellow
    ("rejected_irrelevant", "smart_triage"): (235, 205, 255),  # pale lavender
    ("approved", "find_duplicates"): (200, 255, 200),  # same green
    ("rejected_duplicate", "find_duplicates"): (255, 225, 190),  # pale peach
    ("rejected_irrelevant", "find_duplicates"): (255, 200, 200),  # same red
    ("approved", "matched_mode_default"): (200, 235, 255),  # pale cyan
    ("rejected_duplicate", "matched_mode_default"): (255, 210, 140),  # orange
    ("rejected_irrelevant", "matched_mode_default"): (255, 200, 200),  # red
    # Combined mode — scanner pre-approved records pages
    ("approved", "scanner"): (200, 255, 200),  # same green as user-approved
    ("rejected_duplicate", "scanner"): (255, 210, 140),
    ("rejected_irrelevant", "scanner"): (255, 200, 200),
    # Combined mode — scanner pre-approved bills pages
    ("approved", "scanner_bills"): (190, 240, 255),  # cyan/teal
    ("rejected_duplicate", "scanner_bills"): (255, 210, 140),
    ("rejected_irrelevant", "scanner_bills"): (255, 200, 200),
    # LLM batch-pass pre-classifications (mint — distinct from smart_triage blue)
    ("approved", "llm_triage"): (175, 240, 215),
    ("rejected_duplicate", "llm_triage"): (255, 210, 140),
    ("rejected_irrelevant", "llm_triage"): (255, 200, 200),
}

# Background color for scanner-excluded pages with no decision yet (combined mode)
_PENDING_EXCLUDED_COLOR: tuple[int, int, int] = (255, 230, 140)  # amber

# Category names for the correction dropdown (ordered by clinical relevance)
_CATEGORY_NAMES: list[str] = [
    "THERAPY",
    "MEDICAL_TREATMENT",
    "BILLING",
    "INJURY_LEGAL",
    "IMAGING",
    "BEHAVIORAL_HEALTH",
    "VOCATIONAL",
    "DOCUMENT_TYPE",
]

# Categories that route a page to the bills stream
_BILLS_CATEGORIES: frozenset[str] = frozenset({"BILLING", "INJURY_LEGAL"})

# LLM record types that warrant automatic approval at high confidence.
# OCR pages are excluded — text quality is too uncertain for unattended approval.
_LLM_AUTO_APPROVE_TYPES: frozenset[str] = frozenset(
    {"bill", "imaging", "pharmacy", "legal_document"}
)
_LLM_MIN_CONFIDENCE: float = 0.85

_THERAPY_CATS: frozenset[str] = frozenset({"THERAPY", "BEHAVIORAL_HEALTH"})

_INTAKE_KW: frozenset[str] = frozenset({
    "initial evaluation", "initial eval", "initial assessment",
    "intake", "initial visit", "new patient", "evaluation and management",
    "initial consultation", "initial treatment",
})

_DISCHARGE_KW: frozenset[str] = frozenset({
    "discharge summary", "discharge note", "discharge planning",
    "final visit", "final treatment", "discharge instructions",
    "completion of treatment", "discharged",
})

_AUTO_APPROVE_CATS: frozenset[str] = frozenset({
    "BILLING", "INJURY_LEGAL", "IMAGING", "MEDICAL_TREATMENT", "VOCATIONAL",
})

# Maps LLM record_type values to the canonical keyword category they override.
# Empty string → fall through to keyword category (used for "other_nec").
_RECORD_TYPE_TO_CATEGORY: dict[str, str] = {
    "office_visit":       "MEDICAL_TREATMENT",
    "therapy_non_psych":  "THERAPY",
    "therapy_psych":      "BEHAVIORAL_HEALTH",
    "inpatient_stay":     "MEDICAL_TREATMENT",
    "imaging":            "IMAGING",
    "bill":               "BILLING",
    "billing_affidavit":  "BILLING",
    "vocational":         "VOCATIONAL",
    "legal_document":     "INJURY_LEGAL",
    "pharmacy":           "BILLING",
    "ime":                "INJURY_LEGAL",
    "neuropsych_testing": "BEHAVIORAL_HEALTH",
    "operative_report":   "MEDICAL_TREATMENT",
    "other_nec":          "",
}

_MAX_UNDO_DEPTH = 50


class _ReviewSnapshot(NamedTuple):
    """Immutable capture of all mutable ReviewDialog decision state."""

    decisions: dict
    decision_sources: dict
    triage_decisions: dict
    service_date_overrides: dict
    provider_npi_overrides: dict
    provider_name_overrides: dict
    provider_assigned: dict
    dedup_groups: dict
    dedup_canonical: set
    rows: list
    subtypes: list
    category_overrides: dict


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
        _apply_app_icon(self)
        self._mode = mode
        if mode == "matched":
            self.setWindowTitle("DocProcessorPro — Review Matched Pages")
        elif mode == "combined":
            self.setWindowTitle("DocProcessorPro — Review All Scored Pages")
        else:
            self.setWindowTitle("DocProcessorPro — Review Excluded Pages")
        self.setMinimumSize(1300, 820)
        self.resize(1300, 820)
        self.setWindowState(Qt.WindowState.WindowMaximized)

        self._output_dir = _Path(output_dir)
        _feedback_dir = self._output_dir / "_feedback"
        _feedback_dir.mkdir(exist_ok=True)
        if mode == "matched":
            self._feedback_path = _feedback_dir / "_matched_feedback.jsonl"
            self._draft_path = _feedback_dir / "_matched_review_draft.json"
        elif mode == "combined":
            self._feedback_path = _feedback_dir / "_combined_feedback.jsonl"
            self._draft_path = _feedback_dir / "_combined_review_draft.json"
        else:
            self._feedback_path = _feedback_dir / "_feedback.jsonl"
            self._draft_path = _feedback_dir / "_review_draft.json"
        self._fb_worker: _FeedbackWorker | None = None
        self._scan_settings: "dict | None" = scan_settings
        self._apply_result: "tuple[int, int] | None" = (
            None  # (approved, skipped) set on successful apply
        )

        self._rows: list[dict] = []
        self._subtypes: list[str | None] = []  # parallel to _rows
        self._decisions: dict[int, str] = {}  # row_index → "approved" | "rejected_*"
        self._triage_decisions: dict[int, str] = {}  # snapshot after Smart Triage runs
        self._decision_sources: dict[
            int, str
        ] = {}  # row_index → "user"|"smart_triage"|"find_duplicates"|"matched_mode_default"
        self._dedup_groups: dict[
            int, int
        ] = {}  # row_index → cluster id from last Find Duplicates run
        self._dedup_canonical: set[int] = (
            set()
        )  # indices identified as canonical in their dedup cluster
        self._current_index: int = -1
        self._row_idx_to_list_row: dict[int, int] = {}
        self._undo_stack: list[_ReviewSnapshot] = []
        self._redo_stack: list[_ReviewSnapshot] = []

        # Field-level correction overrides (user edits auto-extracted scanner values)
        self._service_date_overrides: dict[int, str] = {}
        self._provider_npi_overrides: dict[int, str] = {}
        self._provider_name_overrides: dict[int, str] = {}
        # User-assigned canonical provider name (never auto-filled from scanner)
        self._provider_assigned: dict[int, str] = {}
        # User-corrected primary category (overrides highest-weight system category)
        self._category_overrides: dict[int, str] = {}

        # Dedup prefetch: Pass-1 results (candidate groups + thumbnails) computed in
        # background so Find Duplicates can skip straight to Pass 2 when clicked.
        self._dedup_cache: "dict | None" = None
        self._dedup_prefetch: "_DedupPrefetchWorker | None" = None
        self._review_pdf_path: Path = Path(review_pdf)
        # Ordered list of known provider names for autocomplete
        self._provider_registry: list[str] = []
        self._provider_registry_path = Path(manifest_csv).parent / ".dpp_providers.json"
        self._load_provider_registry()

        self._llm_index: dict[str, dict[int, dict]] = {}  # stem → {page_num → llm_record}

        self._load_manifest(manifest_csv)
        # Matched mode: pre-approve every page (already in consolidated; user marks removals)
        if self._mode == "matched":
            for i in range(len(self._rows)):
                self._decisions[i] = "approved"
                self._decision_sources[i] = "matched_mode_default"
        # Combined mode: pre-approve scanner-matched pages; leave excluded pages pending.
        # Bills pages get a distinct cyan color via "scanner_bills" source key.
        elif self._mode == "combined":
            for i, row in enumerate(self._rows):
                if row.get("scanner_decision") == "matched":
                    stream = row.get("scan_stream", "records")
                    source = "scanner_bills" if stream == "bills" else "scanner"
                    self._decisions[i] = "approved"
                    self._decision_sources[i] = source
        # LLM batch-pass pre-classifications: auto-approve high-confidence routine
        # record types on pages that don't already have a scanner decision.
        # _load_existing_feedback() called below will override with any saved user work.
        self._apply_llm_triage()
        self._build_ui()
        self._load_pdf(review_pdf)
        self._load_existing_feedback()
        self._load_draft()  # overlay any newer in-progress decisions from auto-save

        if self._rows:
            self._select_row(0)

        # Start background dedup prefetch after dialog is fully initialized.
        if len(self._rows) >= 2:
            self._dedup_prefetch = _DedupPrefetchWorker(
                self._review_pdf_path, self._rows, parent=self
            )
            self._dedup_prefetch.ready.connect(self._on_dedup_prefetch_ready)
            self._dedup_prefetch.start()

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
        self._subtypes = [self._classify_subtype(i, row) for i, row in enumerate(self._rows)]

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
                    "rejected_duplicate"
                    if reason == "duplicate"
                    else "rejected_irrelevant"
                )
            source = rec.get("decision_source", "user")
            for idx, row in enumerate(self._rows):
                if row.get("source_pdf_path") == src and int(
                    row.get("page_num", 0)
                ) == int(pg):
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
            "decisions": _to_stable(self._decisions),
            "triage_decisions": _to_stable(self._triage_decisions),
            "decision_sources": _to_stable(self._decision_sources),
            "dedup_group_ids": _to_stable(self._dedup_groups),
            "dedup_canonical_keys": dedup_canonical_keys,
            "service_date_overrides": _to_stable(self._service_date_overrides),
            "provider_npi_overrides": _to_stable(self._provider_npi_overrides),
            "provider_name_overrides": _to_stable(self._provider_name_overrides),
            "provider_assigned": _to_stable(self._provider_assigned),
            "category_overrides": _to_stable(self._category_overrides),
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
            svc_date_map: dict = raw.get("service_date_overrides", {})
            npi_map: dict = raw.get("provider_npi_overrides", {})
            name_hint_map: dict = raw.get("provider_name_overrides", {})
            assigned_map: dict = raw.get("provider_assigned", {})
            cat_overrides_map: dict = raw.get("category_overrides", {})
        else:
            decisions_map = raw  # legacy flat
            triage_map = {}
            sources_map = {}
            dedup_groups_map = {}
            dedup_canonical_keys = []
            svc_date_map = {}
            npi_map = {}
            name_hint_map = {}
            assigned_map = {}
            cat_overrides_map = {}

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

        for key, val in svc_date_map.items():
            idx = key_to_idx.get(key)
            if idx is not None and val:
                self._service_date_overrides[idx] = val

        for key, val in npi_map.items():
            idx = key_to_idx.get(key)
            if idx is not None and val:
                self._provider_npi_overrides[idx] = val

        for key, val in name_hint_map.items():
            idx = key_to_idx.get(key)
            if idx is not None and val:
                self._provider_name_overrides[idx] = val

        for key, val in assigned_map.items():
            idx = key_to_idx.get(key)
            if idx is not None and val:
                self._provider_assigned[idx] = val
                if val not in self._provider_registry:
                    self._provider_registry.append(val)

        for key, val in cat_overrides_map.items():
            idx = key_to_idx.get(key)
            if idx is not None and val:
                self._category_overrides[idx] = val

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

    def _load_provider_registry(self) -> None:
        import json
        try:
            data = json.loads(self._provider_registry_path.read_text(encoding="utf-8"))
            for name in data.get("provider_names", []):
                if name and name not in self._provider_registry:
                    self._provider_registry.append(name)
        except (OSError, json.JSONDecodeError, ValueError):
            pass

    def _save_provider_registry(self) -> None:
        import json
        try:
            self._provider_registry_path.write_text(
                json.dumps({"provider_names": self._provider_registry}, ensure_ascii=False),
                encoding="utf-8",
            )
        except OSError:
            pass

    def _provider_key(self, row: dict, idx: int | None = None) -> str:
        """Return the best available provider grouping key for a manifest row.

        Resolution order (highest confidence first):
          1. User-corrected NPI override
          2. Auto-extracted NPI from scanner
          3. User-assigned canonical provider name
          4. User-corrected provider name hint
          5. Auto-extracted provider name hint
          6. Source PDF path (last resort)
        """
        if idx is not None:
            npi = self._provider_npi_overrides.get(idx) or row.get("provider_npi", "")
        else:
            npi = row.get("provider_npi", "")
        if npi:
            return npi

        if idx is not None:
            assigned = self._provider_assigned.get(idx, "")
            if assigned:
                return assigned
            name_hint = self._provider_name_overrides.get(idx) or row.get("provider_name_hint", "")
        else:
            name_hint = row.get("provider_name_hint", "")
        if name_hint:
            return name_hint

        return row.get("source_pdf_path", "")

    def _effective_stream(self, idx: int) -> str:
        """Return 'records', 'bills', or 'unmatched' for the page at idx.

        Category override takes highest priority; then the scan_stream manifest
        column; then a legacy inference from categories_matched for old manifests.
        """
        override = self._category_overrides.get(idx)
        if override:
            return "bills" if override in _BILLS_CATEGORIES else "records"
        row = self._rows[idx]
        stream = row.get("scan_stream", "")
        if stream in ("records", "bills", "unmatched"):
            return stream
        # Legacy fallback: infer from categories_matched
        cats = {c for c in row.get("categories_matched", "").split("|") if c}
        return "bills" if cats and cats <= _BILLS_CATEGORIES else "records"

    def _llm_record(self, idx: int) -> "dict | None":
        """Return the LLM extraction record for row idx, or None if unavailable."""
        from pathlib import Path as _Path
        row = self._rows[idx]
        stem = _Path(row.get("source_pdf_path", "")).stem
        pg = int(row.get("page_num", 0))
        return self._llm_index.get(stem, {}).get(pg)

    def _effective_category(self, idx: int) -> str:
        """Return the authoritative category for row idx.

        Priority: manual user override → LLM record_type mapping → keyword category.
        Falls back cleanly when LLM data is unavailable or record_type is unknown.
        """
        override = self._category_overrides.get(idx)
        if override:
            return override
        llm = self._llm_record(idx)
        if llm:
            mapped = _RECORD_TYPE_TO_CATEGORY.get(llm.get("record_type", ""), "")
            if mapped:
                return mapped
        row = self._rows[idx]
        from DocProcessorPro.dpp_scripts.keyword_scanner_scripts.keyword_scanner_codebase import (
            DEFAULT_CATEGORIES,
            highest_weight_category,
        )
        cats = [c for c in row.get("categories_matched", "").split("|") if c]
        return highest_weight_category(cats, DEFAULT_CATEGORIES) or ""

    def _on_category_changed(self, _combo_index: int) -> None:
        """Persist a user-selected category override and refresh the list item."""
        idx = self._current_index
        if idx < 0:
            return
        new_cat = self._category_combo.currentText()
        row = self._rows[idx]
        from DocProcessorPro.dpp_scripts.keyword_scanner_scripts.keyword_scanner_codebase import (
            DEFAULT_CATEGORIES,
            highest_weight_category,
        )
        cats = [c for c in row.get("categories_matched", "").split("|") if c]
        system_cat = highest_weight_category(cats, DEFAULT_CATEGORIES) or ""
        if new_cat == system_cat:
            self._category_overrides.pop(idx, None)
        else:
            self._push_undo()
            self._category_overrides[idx] = new_cat
        # Refresh the list item's color and tag immediately
        row = self._rows[idx]
        item = self._list_item_for(idx)
        if item:
            item.setText(self._make_list_item_text(idx, row))
            color = self._item_color(idx)
            if color:
                item.setBackground(color)
            else:
                item.setBackground(self._list.palette().base())
        self._save_draft()

    # ------------------------------------------------------------------ undo / redo

    def _snapshot(self) -> _ReviewSnapshot:
        return _ReviewSnapshot(
            decisions=dict(self._decisions),
            decision_sources=dict(self._decision_sources),
            triage_decisions=dict(self._triage_decisions),
            service_date_overrides=dict(self._service_date_overrides),
            provider_npi_overrides=dict(self._provider_npi_overrides),
            provider_name_overrides=dict(self._provider_name_overrides),
            provider_assigned=dict(self._provider_assigned),
            dedup_groups=dict(self._dedup_groups),
            dedup_canonical=set(self._dedup_canonical),
            rows=list(self._rows),
            subtypes=list(self._subtypes),
            category_overrides=dict(self._category_overrides),
        )

    def _push_undo(self) -> None:
        self._undo_stack.append(self._snapshot())
        if len(self._undo_stack) > _MAX_UNDO_DEPTH:
            self._undo_stack.pop(0)
        self._redo_stack.clear()
        self._update_undo_redo_buttons()

    def _apply_snapshot(self, snap: _ReviewSnapshot) -> None:
        self._decisions = dict(snap.decisions)
        self._decision_sources = dict(snap.decision_sources)
        self._triage_decisions = dict(snap.triage_decisions)
        self._service_date_overrides = dict(snap.service_date_overrides)
        self._provider_npi_overrides = dict(snap.provider_npi_overrides)
        self._provider_name_overrides = dict(snap.provider_name_overrides)
        self._provider_assigned = dict(snap.provider_assigned)
        self._dedup_groups = dict(snap.dedup_groups)
        self._dedup_canonical = set(snap.dedup_canonical)
        self._rows = list(snap.rows)
        self._subtypes = list(snap.subtypes)
        self._category_overrides = dict(snap.category_overrides)
        # Rebuild the list widget (row order may have changed)
        self._populate_list()
        self._refresh_list_colors()
        self._update_toolbar()
        self._update_undo_redo_buttons()
        self._save_draft()
        if 0 <= self._current_index < len(self._rows):
            list_row = self._row_idx_to_list_row.get(self._current_index, -1)
            if list_row >= 0:
                self._list.setCurrentRow(list_row)
            self._update_metadata(self._current_index)

    def _undo(self) -> None:
        if not self._undo_stack:
            return
        self._redo_stack.append(self._snapshot())
        self._apply_snapshot(self._undo_stack.pop())

    def _redo(self) -> None:
        if not self._redo_stack:
            return
        self._undo_stack.append(self._snapshot())
        if len(self._undo_stack) > _MAX_UNDO_DEPTH:
            self._undo_stack.pop(0)
        self._apply_snapshot(self._redo_stack.pop())

    def _update_undo_redo_buttons(self) -> None:
        if hasattr(self, "_undo_btn") and hasattr(self, "_redo_btn"):
            self._undo_btn.setEnabled(bool(self._undo_stack))
            self._redo_btn.setEnabled(bool(self._redo_stack))

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(8, 8, 8, 8)
        root.setSpacing(6)

        # Toolbar row
        toolbar = QHBoxLayout()
        self._progress_label = QLabel("0 approved / 0 rejected / 0 remaining")
        toolbar.addWidget(self._progress_label)
        toolbar.addStretch()
        self._undo_btn = QPushButton("Undo")
        self._undo_btn.setToolTip("Undo last action (Ctrl+Z)")
        self._undo_btn.setEnabled(False)
        self._undo_btn.clicked.connect(self._undo)
        toolbar.addWidget(self._undo_btn)
        self._redo_btn = QPushButton("Redo")
        self._redo_btn.setToolTip("Redo last undone action (Ctrl+Shift+Z)")
        self._redo_btn.setEnabled(False)
        self._redo_btn.clicked.connect(self._redo)
        toolbar.addWidget(self._redo_btn)
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
        therapy_triage_btn = QPushButton("Therapy Triage")
        therapy_triage_btn.setToolTip(
            "Per-provider therapy triage: approve intake/discharge pages; "
            "fall back to first/last progress note; pre-reject the rest. "
            "Groups by NPI when available, otherwise by source PDF."
        )
        therapy_triage_btn.clicked.connect(self._run_therapy_triage)
        toolbar.addWidget(therapy_triage_btn)
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
        elif self._mode == "combined":
            self._apply_btn = QPushButton("Apply to Consolidated…")
            self._apply_btn.setToolTip(
                "Rebuild _consolidated.pdf from approved pages and "
                "_consolidated_unmatched.pdf from the remainder"
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
        self._list.setMinimumWidth(160)
        self._list.currentRowChanged.connect(self._on_list_row_changed)
        self._populate_list()
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
        right_panel.setMinimumWidth(220)

        meta_scroll = QScrollArea()
        meta_scroll.setWidgetResizable(True)
        meta_inner = QWidget()
        self._meta_form = QFormLayout(meta_inner)
        self._meta_form.setSpacing(4)
        self._meta_labels: dict[str, QLabel] = {}
        for field_name in (
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
            self._meta_labels[field_name] = lbl
            self._meta_form.addRow(f"{field_name}:", lbl)

        # Editable correction / assignment fields
        self._category_combo = QComboBox()
        for _cat_name in _CATEGORY_NAMES:
            self._category_combo.addItem(_cat_name)
        self._category_combo.setToolTip(
            "Override the primary category. BILLING or INJURY_LEGAL routes to bills; "
            "all others route to records."
        )
        self._category_combo.currentIndexChanged.connect(self._on_category_changed)
        self._meta_form.addRow("Category:", self._category_combo)

        self._svc_date_edit = QLineEdit()
        self._svc_date_edit.setPlaceholderText("YYYY-MM-DD")
        self._svc_date_edit.setToolTip("Correct the auto-extracted service date")
        self._svc_date_edit.editingFinished.connect(self._on_svc_date_edited)
        self._meta_form.addRow("Service Date:", self._svc_date_edit)

        self._provider_npi_edit = QLineEdit()
        self._provider_npi_edit.setPlaceholderText("10-digit NPI")
        self._provider_npi_edit.setToolTip("Correct the auto-extracted NPI")
        self._provider_npi_edit.editingFinished.connect(self._on_provider_npi_edited)
        self._meta_form.addRow("Provider NPI:", self._provider_npi_edit)

        self._provider_hint_edit = QLineEdit()
        self._provider_hint_edit.setPlaceholderText("auto-extracted hint")
        self._provider_hint_edit.setToolTip("Correct the auto-extracted provider name hint")
        self._provider_hint_edit.editingFinished.connect(self._on_provider_hint_edited)
        self._meta_form.addRow("Extracted Name:", self._provider_hint_edit)

        self._name_model = QStringListModel(self._provider_registry, self)
        self._name_completer = QCompleter(self._name_model, self)
        self._name_completer.setCaseSensitivity(Qt.CaseSensitivity.CaseInsensitive)
        self._name_completer.setFilterMode(Qt.MatchFlag.MatchContains)
        self._provider_name_combo = QComboBox()
        self._provider_name_combo.setEditable(True)
        self._provider_name_combo.setCompleter(self._name_completer)
        self._provider_name_combo.setInsertPolicy(QComboBox.InsertPolicy.NoInsert)
        self._provider_name_combo.lineEdit().setPlaceholderText("Type or select provider…")
        self._provider_name_combo.lineEdit().editingFinished.connect(self._on_provider_name_committed)
        self._meta_form.addRow("Provider Name:", self._provider_name_combo)

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

        splitter.setSizes([200, 580, 320])
        root.addWidget(splitter)

        # Keyboard shortcuts
        QShortcut(QKeySequence("A"), self).activated.connect(self._approve)
        QShortcut(QKeySequence("D"), self).activated.connect(self._reject_duplicate)
        QShortcut(QKeySequence("R"), self).activated.connect(self._reject_irrelevant)
        QShortcut(QKeySequence("S"), self).activated.connect(self._skip)
        QShortcut(QKeySequence("Ctrl+Z"), self).activated.connect(self._undo)
        QShortcut(QKeySequence("Ctrl+Shift+Z"), self).activated.connect(self._redo)
        QShortcut(QKeySequence("Ctrl+Y"), self).activated.connect(self._redo)

    # --------------------------------------------------------- navigation

    def _select_row(self, index: int) -> None:
        if not (0 <= index < len(self._rows)):
            return
        self._current_index = index
        list_row = self._row_idx_to_list_row.get(index, -1)
        if list_row >= 0:
            self._list.setCurrentRow(list_row)
        self._update_metadata(index)
        self._navigate_pdf(index)

    def _on_list_row_changed(self, row: int) -> None:
        if row < 0:
            return
        item = self._list.item(row)
        if item is None:
            return
        logical = item.data(Qt.ItemDataRole.UserRole)
        if logical is None or logical < 0:
            return  # section header — ignore
        if logical == self._current_index:
            return
        self._current_index = logical
        self._update_metadata(logical)
        self._navigate_pdf(logical)

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
            "intake": "Intake / Initial",
            "discharge": "Discharge / Termination",
            "progress": "Progress Note",
            "imaging": "Imaging Report",
            "legal": "Legal / Affidavit",
            "medical": "Medical Treatment",
            "vocational": "Vocational",
            "billing": "Billing Record",
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
        # Category combo — manual override > LLM record_type > keyword category
        eff_cat = self._effective_category(index)
        self._category_combo.blockSignals(True)
        _cat_i = self._category_combo.findText(eff_cat)
        self._category_combo.setCurrentIndex(_cat_i if _cat_i >= 0 else 0)
        self._category_combo.blockSignals(False)

        # Editable correction fields — show override if present, else auto-extracted value
        svc = self._service_date_overrides.get(index) or row.get("service_date", "")
        self._svc_date_edit.blockSignals(True)
        self._svc_date_edit.setText(svc)
        self._svc_date_edit.blockSignals(False)

        npi = self._provider_npi_overrides.get(index) or row.get("provider_npi", "")
        self._provider_npi_edit.blockSignals(True)
        self._provider_npi_edit.setText(npi)
        self._provider_npi_edit.blockSignals(False)

        hint = self._provider_name_overrides.get(index) or row.get("provider_name_hint", "")
        self._provider_hint_edit.blockSignals(True)
        self._provider_hint_edit.setText(hint)
        self._provider_hint_edit.blockSignals(False)

        assigned = self._provider_assigned.get(index, "")
        self._provider_name_combo.blockSignals(True)
        le = self._provider_name_combo.lineEdit()
        if le is not None:
            le.setText(assigned)
        self._provider_name_combo.blockSignals(False)

    # ----------------------------------------- editable field handlers

    def _on_svc_date_edited(self) -> None:
        idx = self._current_index
        if idx < 0:
            return
        val = self._svc_date_edit.text().strip()
        if val == self._service_date_overrides.get(idx, ""):
            return
        self._push_undo()
        if val:
            self._service_date_overrides[idx] = val
        else:
            self._service_date_overrides.pop(idx, None)
        self._save_draft()

    def _on_provider_npi_edited(self) -> None:
        idx = self._current_index
        if idx < 0:
            return
        val = self._provider_npi_edit.text().strip()
        if val == self._provider_npi_overrides.get(idx, ""):
            return
        self._push_undo()
        if val:
            self._provider_npi_overrides[idx] = val
        else:
            self._provider_npi_overrides.pop(idx, None)
        self._save_draft()

    def _on_provider_hint_edited(self) -> None:
        idx = self._current_index
        if idx < 0:
            return
        val = self._provider_hint_edit.text().strip()
        if val == self._provider_name_overrides.get(idx, ""):
            return
        self._push_undo()
        if val:
            self._provider_name_overrides[idx] = val
        else:
            self._provider_name_overrides.pop(idx, None)
        self._save_draft()

    def _on_provider_name_committed(self) -> None:
        idx = self._current_index
        if idx < 0:
            return
        le = self._provider_name_combo.lineEdit()
        val = le.text().strip() if le is not None else ""
        if val == self._provider_assigned.get(idx, ""):
            return
        self._push_undo()
        if val:
            self._provider_assigned[idx] = val
            if val not in self._provider_registry:
                self._provider_registry.append(val)
                self._name_model.setStringList(self._provider_registry)
                self._save_provider_registry()
        else:
            self._provider_assigned.pop(idx, None)
        self._save_draft()

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
                "intake": "[INTAKE]   ",
                "discharge": "[DISCHARGE]",
                "progress": "[PROG]     ",
                "imaging": "[IMAGING]  ",
                "legal": "[LEGAL]    ",
                "medical": "[MEDICAL]  ",
                "vocational": "[VOC]      ",
                "billing": "[BILLING]  ",
            }.get(subtype or "", "           ")
        # In combined mode, bills pages get [BILL] tag; records pages get [M] suffix
        if self._mode == "combined" and row.get("scanner_decision") == "matched":
            if self._effective_stream(idx) == "bills":
                prefix = "[BILL]     "
            else:
                prefix = prefix.rstrip() + "[M]"
        return f"{prefix} {stem}\n           p.{pg}  score {score:.2f}"

    def _make_list_item(self, idx: int, row: dict) -> QListWidgetItem:
        item = QListWidgetItem(self._make_list_item_text(idx, row))
        item.setData(Qt.ItemDataRole.UserRole, idx)
        return item

    def _make_section_header(self, stem: str, count: int) -> QListWidgetItem:
        label = f"  {stem}  ({count} page{'s' if count != 1 else ''})"
        item = QListWidgetItem(label)
        item.setData(Qt.ItemDataRole.UserRole, -1)
        item.setFlags(Qt.ItemFlag.NoItemFlags)
        font = item.font()
        font.setBold(True)
        item.setFont(font)
        item.setForeground(QColor(80, 80, 80))
        item.setBackground(QColor(228, 228, 228))
        return item

    def _populate_list(self) -> None:
        """Rebuild the list widget, inserting a section header each time the source PDF changes."""
        self._list.clear()
        self._row_idx_to_list_row = {}
        stem_counts: dict[str, int] = {}
        for row in self._rows:
            src = row.get("source_pdf_path", "")
            stem = src.rsplit("/", 1)[-1].rsplit("\\", 1)[-1].removesuffix(".pdf") or "Unknown"
            stem_counts[stem] = stem_counts.get(stem, 0) + 1
        prev_stem: "str | None" = None
        for i, row in enumerate(self._rows):
            src = row.get("source_pdf_path", "")
            stem = src.rsplit("/", 1)[-1].rsplit("\\", 1)[-1].removesuffix(".pdf") or "Unknown"
            if stem != prev_stem:
                self._list.addItem(self._make_section_header(stem, stem_counts[stem]))
                prev_stem = stem
            self._list.addItem(self._make_list_item(i, row))
            self._row_idx_to_list_row[i] = self._list.count() - 1

    def _list_item_for(self, row_idx: int) -> "QListWidgetItem | None":
        """Return the QListWidgetItem for a logical row index, or None."""
        list_row = self._row_idx_to_list_row.get(row_idx)
        if list_row is None:
            return None
        return self._list.item(list_row)

    def _apply_llm_triage(self) -> None:
        """Load _llm_fields.jsonl sidecars and auto-approve high-confidence routine pages.

        Only affects pages that have no existing decision (i.e. scanner-excluded pages
        in combined mode, or all pages in unmatched mode).  Pages already approved by
        the scanner are left untouched.  _load_existing_feedback() called after this
        will overwrite any LLM decision with prior user work.
        """
        import json as _json
        from pathlib import Path as _Path

        for i, row in enumerate(self._rows):
            if i in self._decisions:
                continue  # scanner already set a decision; don't interfere

            src = row.get("source_pdf_path", "")
            stem = _Path(src).stem
            pg = int(row.get("page_num", 0))

            # Load sidecar on first access for this stem
            if stem not in self._llm_index:
                sidecar = self._output_dir / "_sidecars" / f"{stem}_llm_fields.jsonl"
                page_map: dict[int, dict] = {}
                if sidecar.exists():
                    for _line in sidecar.read_text(encoding="utf-8").splitlines():
                        _line = _line.strip()
                        if not _line:
                            continue
                        try:
                            _rec = _json.loads(_line)
                            page_map[int(_rec["page_num"])] = _rec
                        except (KeyError, ValueError, _json.JSONDecodeError):
                            pass
                self._llm_index[stem] = page_map

            llm = self._llm_index[stem].get(pg)
            if llm is None:
                continue

            record_type = llm.get("record_type", "")
            confidence = float(llm.get("confidence", 0.0))
            extraction_method = row.get("extraction_method", "")

            if (
                record_type in _LLM_AUTO_APPROVE_TYPES
                and confidence >= _LLM_MIN_CONFIDENCE
                and extraction_method != "liteparse_ocr"
            ):
                self._decisions[i] = "approved"
                self._decision_sources[i] = "llm_triage"

    def _classify_subtype(self, idx: int, row: dict) -> str | None:
        eff = self._effective_category(idx)
        if not eff:
            return None
        # Therapy subtype takes priority — most granular classification
        if eff in _THERAPY_CATS:
            kw = {k.lower() for k in row.get("keywords_hit", "").split("|") if k}
            if kw & _INTAKE_KW:
                return "intake"
            if kw & _DISCHARGE_KW:
                return "discharge"
            return "progress"
        # Non-therapy clinical categories
        if eff == "IMAGING":
            return "imaging"
        if eff == "INJURY_LEGAL":
            return "legal"
        if eff == "MEDICAL_TREATMENT":
            return "medical"
        if eff == "VOCATIONAL":
            return "vocational"
        if eff == "BILLING":
            return "billing"
        return None

    def _detect_therapy_duplicates(self) -> set[int]:
        """Return row indices that are likely therapy-record duplicates.

        Groups rows by (provider_key, frozenset(dates_on_page)).  Within groups
        with >1 row the highest-scoring row is kept; the rest are candidate
        duplicates.  For rows without dates, groups by (provider_key,
        frozenset(keywords_hit)) instead.
        """
        from collections import defaultdict

        groups: dict[tuple, list[int]] = defaultdict(list)
        for idx, row in enumerate(self._rows):
            pk = self._provider_key(row, idx)
            dates = frozenset(d for d in row.get("dates_on_page", "").split("|") if d)
            if dates:
                key: tuple = (pk, dates)
            else:
                kws = frozenset(k for k in row.get("keywords_hit", "").split("|") if k)
                key = (pk, kws)
            groups[key].append(idx)

        duplicates: set[int] = set()
        for indices in groups.values():
            if len(indices) < 2:
                continue
            best = max(indices, key=lambda i: float(self._rows[i].get("total_hits", 0)))
            duplicates.update(i for i in indices if i != best)
        return duplicates

    def _run_therapy_triage(self) -> None:
        """Per-provider therapy triage.

        Groups therapy pages by provider key (NPI > assigned name > name hint >
        source PDF).  For each provider: approves intake/discharge pages; falls
        back to first/last progress note when neither exists; pre-rejects the
        remaining progress notes.  Re-sorts the list so intake/discharge appear
        at the top.  All pre-selections are tagged 'smart_triage' so the user
        can override any of them with A/D/R/S.
        """
        self._push_undo()
        from collections import defaultdict

        # Detect likely duplicates first
        dup_candidates = self._detect_therapy_duplicates()
        dup_count = 0
        for idx in dup_candidates:
            if idx not in self._decisions:
                self._decisions[idx] = "rejected_duplicate"
                self._decision_sources[idx] = "smart_triage"
                dup_count += 1

        # Group therapy rows by provider key
        provider_items: dict[str, list[tuple[int, str | None]]] = defaultdict(list)
        for idx, row in enumerate(self._rows):
            sub = self._subtypes[idx] if idx < len(self._subtypes) else None
            if sub in ("intake", "discharge", "progress"):
                pk = self._provider_key(row, idx)
                provider_items[pk].append((idx, sub))

        providers_with_id: set[str] = set()
        intake_discharge_count = 0
        fallback_count = 0
        prereject_count = 0
        fallback_indices: set[int] = set()

        for pk, items in provider_items.items():
            has_id = any(sub in ("intake", "discharge") for _, sub in items)
            if has_id:
                providers_with_id.add(pk)
                for idx, sub in items:
                    if sub in ("intake", "discharge") and idx not in self._decisions:
                        self._decisions[idx] = "approved"
                        self._decision_sources[idx] = "smart_triage"
                        intake_discharge_count += 1
                continue

            # Fallback: first/last progress note by service_date then dates_on_page
            progress = [(idx, self._rows[idx]) for idx, sub in items if sub == "progress"]
            if not progress:
                continue

            def _best_date(r: dict, i: int, want_min: bool) -> str:
                svc = self._service_date_overrides.get(i) or r.get("service_date", "")
                if svc:
                    return svc
                ds = [d for d in r.get("dates_on_page", "").split("|") if d]
                if not ds:
                    return "9999-99-99" if want_min else "0000-00-00"
                return min(ds) if want_min else max(ds)

            first_idx = min(progress, key=lambda t: _best_date(t[1], t[0], True))[0]
            last_idx = max(progress, key=lambda t: _best_date(t[1], t[0], False))[0]
            for idx in {first_idx, last_idx}:
                if idx not in self._decisions:
                    self._decisions[idx] = "approved"
                    self._decision_sources[idx] = "smart_triage"
                    fallback_indices.add(idx)
                    fallback_count += 1

            for idx, _ in progress:
                if idx not in self._decisions:
                    self._decisions[idx] = "rejected_irrelevant"
                    self._decision_sources[idx] = "smart_triage"
                    prereject_count += 1

        # For providers that had intake/discharge, pre-reject remaining progress notes
        for pk, items in provider_items.items():
            if pk not in providers_with_id:
                continue
            for idx, sub in items:
                if sub == "progress" and idx not in self._decisions:
                    self._decisions[idx] = "rejected_irrelevant"
                    self._decision_sources[idx] = "smart_triage"
                    prereject_count += 1

        # Re-sort: intake → discharge → fallback progress → other progress → rest → dups
        def _triage_key(pair: tuple[int, dict]) -> tuple[int, str, str, float]:
            i, r = pair
            sub = self._subtypes[i] if i < len(self._subtypes) else None
            is_dup = i in dup_candidates
            if is_dup:
                tier = 5
            elif sub == "intake":
                tier = 0
            elif sub == "discharge":
                tier = 1
            elif i in fallback_indices:
                tier = 2
            elif sub == "progress":
                tier = 3
            elif sub is not None:
                tier = 4
            else:
                tier = 4
            pk = self._provider_key(r, i)
            svc = self._service_date_overrides.get(i) or r.get("service_date", "")
            return (tier, pk, svc, -float(r.get("total_hits", 0)))

        # Remap decisions to stable keys before re-sorting
        def _stable(d: dict[int, str]) -> dict[str, str]:
            out: dict[str, str] = {}
            for idx, val in d.items():
                if idx < len(self._rows):
                    row = self._rows[idx]
                    key = f"{row.get('source_pdf_path', '')}|{row.get('page_num', '')}"
                    out[key] = val
            return out

        stable_decisions = _stable(self._decisions)
        stable_sources = _stable(self._decision_sources)
        stable_triage = _stable(self._triage_decisions)
        stable_svc = _stable(self._service_date_overrides)
        stable_npi = _stable(self._provider_npi_overrides)
        stable_hint = _stable(self._provider_name_overrides)
        stable_assigned = _stable(self._provider_assigned)
        stable_dedup_groups = _stable({k: str(v) for k, v in self._dedup_groups.items()})
        stable_canonical = {
            f"{self._rows[i].get('source_pdf_path', '')}|{self._rows[i].get('page_num', '')}"
            for i in self._dedup_canonical if i < len(self._rows)
        }

        self._rows = sorted(enumerate(self._rows), key=_triage_key)  # type: ignore[assignment]
        self._rows = [r for _, r in self._rows]  # type: ignore[misc]
        self._subtypes = [self._classify_subtype(i, row) for i, row in enumerate(self._rows)]

        # Restore all per-index state from stable keys
        key_to_new: dict[str, int] = {}
        for new_idx, row in enumerate(self._rows):
            k = f"{row.get('source_pdf_path', '')}|{row.get('page_num', '')}"
            key_to_new[k] = new_idx

        def _restore(stable: dict[str, str]) -> dict[int, str]:
            return {key_to_new[k]: v for k, v in stable.items() if k in key_to_new}

        self._decisions = _restore(stable_decisions)
        self._decision_sources = _restore(stable_sources)
        self._triage_decisions = _restore(stable_triage)
        self._service_date_overrides = _restore(stable_svc)
        self._provider_npi_overrides = _restore(stable_npi)
        self._provider_name_overrides = _restore(stable_hint)
        self._provider_assigned = _restore(stable_assigned)
        self._dedup_groups = {key_to_new[k]: int(v) for k, v in stable_dedup_groups.items() if k in key_to_new}  # type: ignore[arg-type]
        self._dedup_canonical = {key_to_new[k] for k in stable_canonical if k in key_to_new}

        # Rebuild list widget
        self._populate_list()
        self._refresh_list_colors()
        self._update_toolbar()
        self._save_draft()

        QMessageBox.information(
            self,
            "Therapy Triage Complete",
            f"Triage complete:\n"
            f"  • {intake_discharge_count} intake/discharge page(s) approved\n"
            f"  • {fallback_count} first/last progress note(s) approved\n"
            f"  • {prereject_count} remaining progress note(s) pre-rejected\n"
            f"  • {dup_count} likely duplicate(s) pre-flagged\n\n"
            "Review pre-selections and adjust as needed before exporting.",
        )

    def _run_smart_triage(self) -> None:
        """Clinical auto-categorization pass (no duplicate detection — use Find Duplicates).

        Approves therapy intake/discharge pages, applies first/last progress note
        fallback, pre-rejects remaining progress notes, and auto-approves non-therapy
        clinical pages. All suggestions are tagged source='smart_triage' so they
        appear in the distinct pale-blue / pale-lavender / pale-yellow color tier.
        """
        self._push_undo()
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
                (idx, self._rows[idx]) for idx, sub in items if sub == "progress"
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
            last_idx = max(progress, key=lambda t: _max_date(t[1]))[0]
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
            ("Legal / Affidavit", "INJURY_LEGAL"),
            ("Imaging", "IMAGING"),
            ("Billing", "BILLING"),
            ("Medical Treatment", "MEDICAL_TREATMENT"),
            ("Vocational", "VOCATIONAL"),
        ]
        cat_approve_counts: dict[str, int] = {}
        for idx, row in enumerate(self._rows):
            if idx in self._decisions:
                continue
            eff = self._effective_category(idx)
            if eff in _AUTO_APPROVE_CATS and eff not in _THERAPY_CATS:
                self._decisions[idx] = "approved"
                self._decision_sources[idx] = "smart_triage"
                for label, cat_key in _CAT_LABELS:
                    if cat_key == eff:
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

    def _on_dedup_prefetch_ready(self, candidate_groups: list, thumbnails: object) -> None:
        """Slot called when the background prefetch worker finishes Pass 1."""
        self._dedup_cache = {
            "candidate_groups": candidate_groups,
            "thumbnails": thumbnails,
        }

    def _run_dedup_pass(self) -> None:
        """Render all pages, compute perceptual hashes, cluster by visual similarity,
        then open _DedupDialog for each group so the user can confirm which to keep.

        Runs against ALL pages regardless of current decision state so every
        potential duplicate pair is surfaced for the feedback record.

        If the background prefetch worker has already completed Pass 1, the expensive
        full-page rendering and hashing is skipped and only the targeted high-res
        Pass 2 renders are performed.
        """
        try:
            from PySide6.QtPdf import QPdfDocument

            if not isinstance(self._pdf_doc, QPdfDocument):
                raise RuntimeError("not loaded")
        except Exception:
            QMessageBox.warning(
                self,
                "PDF Not Available",
                "The PDF viewer must be active to run Find Duplicates.",
            )
            return

        if len(self._rows) < 2:
            QMessageBox.information(
                self,
                "Not Enough Pages",
                "Need at least 2 pages for duplicate detection.",
            )
            return

        prev_label = self._progress_label.text()

        # ── Pass 1: use prefetch cache if ready, else render inline ──────────
        if self._dedup_cache is not None:
            candidate_groups: list[list[int]] = self._dedup_cache["candidate_groups"]
            thumbnails: dict[int, QImage] = self._dedup_cache["thumbnails"]
        else:
            # If the background worker is still running, wait for it to finish.
            if self._dedup_prefetch is not None and self._dedup_prefetch.isRunning():
                self._progress_label.setText("Find Duplicates: finishing pre-computation…")
                QApplication.processEvents()
                self._dedup_prefetch.wait()
                QApplication.processEvents()  # process queued ready signal → _dedup_cache

            if self._dedup_cache is not None:
                candidate_groups = self._dedup_cache["candidate_groups"]
                thumbnails = self._dedup_cache["thumbnails"]
            else:
                # Prefetch unavailable — run full inline render + hash pass.
                self._progress_label.setText("Find Duplicates: rendering pages…")
                QApplication.processEvents()

                hashes: list[tuple[int, int]] = []
                thumbnails = {}
                for idx, row in enumerate(self._rows):
                    try:
                        page_0idx = max(0, int(row.get("consolidated_page_num", "1")) - 1)
                        qimg = self._pdf_doc.render(page_0idx, QSize(300, 390))
                        if not qimg.isNull():
                            thumbnails[idx] = qimg
                        hash_img = self._pdf_doc.render(page_0idx, QSize(800, 1040))
                        if not hash_img.isNull():
                            hashes.append((idx, _dhash(hash_img)))
                        elif not qimg.isNull():
                            hashes.append((idx, _dhash(qimg)))
                    except Exception:
                        pass

                self._progress_label.setText(prev_label)

                if len(hashes) < 2:
                    QMessageBox.information(
                        self, "No Results", "Could not render enough pages for comparison."
                    )
                    return

                raw_clusters = _cluster_by_hash(hashes, threshold=_DEDUP_THRESHOLD)
                candidate_groups = [c for c in raw_clusters if len(c) > 1]

        if not candidate_groups:
            QMessageBox.information(
                self,
                "No Duplicates Found",
                "No visually similar pages were detected across the full page set.",
            )
            return

        # ── Pass 2: pixel-level confirmation (high-res render, MAD metric) ──
        # Only pages that survive pass 1 are rendered here; the expensive render
        # is conditional on pass-1 membership, keeping overall cost low.
        self._progress_label.setText(
            f"Find Duplicates: confirming {len(candidate_groups)} candidate group(s) at high resolution…"
        )
        QApplication.processEvents()

        # Cache high-res renders; a page appears in at most one first-pass cluster.
        hi_res_cache: dict[int, QImage] = {}
        for group in candidate_groups:
            for idx in group:
                if idx in hi_res_cache:
                    continue
                row = self._rows[idx]
                page_0idx = max(0, int(row.get("consolidated_page_num", "1")) - 1)
                img = self._pdf_doc.render(page_0idx, QSize(1200, 1560))
                if not img.isNull():
                    hi_res_cache[idx] = img

        # Helper: effective service date for a row (override beats auto-extracted).
        def _eff_svc(i: int) -> str:
            return (
                self._service_date_overrides.get(i)
                or self._rows[i].get("service_date", "")
                or ""
            )

        groups: list[list[int]] = []
        for group in candidate_groups:
            available = [i for i in group if i in hi_res_cache]
            if len(available) < 2:
                continue
            # Complete-linkage within this candidate group using pixel distance AND
            # service-date compatibility.  A page joins a sub-cluster only if:
            #   1. Pixel MAD ≤ threshold for every existing member, AND
            #   2. Its service date does not conflict with any existing member's date.
            # If either date is absent the date gate is skipped — ambiguous pages rely
            # on the pixel check alone.  Two known but different dates → separate bills.
            sub_clusters: list[list[int]] = []
            sub_dates: list[list[str]] = []
            sub_imgs: list[list[QImage]] = []
            for idx in available:
                svc = _eff_svc(idx)
                placed = False
                for ci, imgs in enumerate(sub_imgs):
                    dates_ok = all(
                        not (svc and d and svc != d)
                        for d in sub_dates[ci]
                    )
                    pixels_ok = all(
                        _pixel_mad(hi_res_cache[idx], img) <= _DEDUP_CONFIRM_MAX_DIFF
                        for img in imgs
                    )
                    if dates_ok and pixels_ok:
                        sub_clusters[ci].append(idx)
                        sub_dates[ci].append(svc)
                        sub_imgs[ci].append(hi_res_cache[idx])
                        placed = True
                        break
                if not placed:
                    sub_clusters.append([idx])
                    sub_dates.append([svc])
                    sub_imgs.append([hi_res_cache[idx]])
            for sub in sub_clusters:
                if len(sub) > 1:
                    groups.append(sub)

        self._progress_label.setText(prev_label)

        if not groups:
            QMessageBox.information(
                self,
                "No Confirmed Duplicates",
                f"{len(candidate_groups)} candidate group(s) found by visual hash, "
                "but none were confirmed by the pixel-level and service-date second pass.\n\n"
                "The pages may share a similar layout template but cover different "
                "service dates or have different content.",
            )
            return

        # Record cluster membership before opening the dialog
        self._dedup_groups = {}
        self._dedup_canonical = set()
        for g_id, group in enumerate(groups):
            canonical = max(
                group, key=lambda i: float(self._rows[i].get("total_hits", 0))
            )
            self._dedup_canonical.add(canonical)
            for idx in group:
                self._dedup_groups[idx] = g_id

        dialog_groups = [
            [
                (idx, self._rows[idx], thumbnails[idx])
                for idx in grp
                if idx in thumbnails
            ]
            for grp in groups
        ]
        dialog_groups = [g for g in dialog_groups if len(g) >= 2]

        # Render hi-res versions of only the flagged pages for the zoom popup.
        group_indices = {idx for grp in groups for idx in grp}
        hi_res: dict[int, QImage] = {}
        self._progress_label.setText("Find Duplicates: rendering previews…")
        QApplication.processEvents()
        for idx in group_indices:
            row = self._rows[idx]
            try:
                page_0idx = max(0, int(row.get("consolidated_page_num", "1")) - 1)
                qimg_hi = self._pdf_doc.render(page_0idx, QSize(700, 910))
                if not qimg_hi.isNull():
                    hi_res[idx] = qimg_hi
            except Exception:
                pass
        self._progress_label.setText(prev_label)

        dlg = _DedupDialog(
            dialog_groups,
            decisions=self._decisions,
            decision_sources=self._decision_sources,
            hi_res=hi_res,
            service_date_overrides=self._service_date_overrides,
            cache_path=self._output_dir / "_feedback" / "_dedup_cache.json",
            parent=self,
        )
        if dlg.exec() != QDialog.DialogCode.Accepted:
            self._save_draft()  # persist dedup group membership even on cancel
            return

        keep_indices, reject_indices = dlg.results()
        changed = 0
        self._push_undo()

        for idx in reject_indices:
            self._decisions[idx] = "rejected_duplicate"
            self._decision_sources[idx] = "find_duplicates"
            item = self._list_item_for(idx)
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
                item = self._list_item_for(idx)
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
            # Combined mode: scanner-excluded pages with no decision yet get amber
            if (
                self._mode == "combined"
                and idx < len(self._rows)
                and self._rows[idx].get("scanner_decision") == "excluded"
            ):
                return QColor(*_PENDING_EXCLUDED_COLOR)
            return None
        src = self._decision_sources.get(idx, "user")
        rgb = _DECISION_COLORS.get((dec, src)) or _DECISION_COLORS.get((dec, "user"))
        return QColor(*rgb) if rgb else None

    # --------------------------------------------------------- decisions

    def _apply_filter(self) -> None:
        """Show only approved + duplicate rows when the filter toggle is active."""
        active = self._filter_btn.isChecked()
        for list_row in range(self._list.count()):
            item = self._list.item(list_row)
            if item is None:
                continue
            logical = item.data(Qt.ItemDataRole.UserRole)
            if logical is None or logical < 0:
                # Section headers stay visible regardless of filter
                item.setHidden(False)
                continue
            if active:
                dec = self._decisions.get(logical)
                item.setHidden(dec not in ("approved", "rejected_duplicate"))
            else:
                item.setHidden(False)

    def _advance_one(self) -> None:
        """Move to the next row (+1), skipping hidden items when filter is active."""
        active_filter = self._filter_btn.isChecked()
        start = self._current_index + 1
        for i in range(start, len(self._rows)):
            if active_filter:
                item = self._list_item_for(i)
                if item and item.isHidden():
                    continue
            self._select_row(i)
            return

    def _record_decision(self, decision: str) -> None:
        if self._current_index < 0:
            return
        self._push_undo()
        was_decided = self._current_index in self._decisions
        self._decisions[self._current_index] = decision
        self._decision_sources[self._current_index] = "user"
        item = self._list_item_for(self._current_index)
        if item:
            item.setText(
                self._make_list_item_text(
                    self._current_index, self._rows[self._current_index]
                )
            )
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
                item = self._list_item_for(i)
                if item and item.isHidden():
                    continue
            if i not in self._decisions:
                self._select_row(i)
                return
        for i in range(0, start):
            if active_filter:
                item = self._list_item_for(i)
                if item and item.isHidden():
                    continue
            if i not in self._decisions:
                self._select_row(i)
                return

    def _refresh_list_colors(self) -> None:
        for idx, row in enumerate(self._rows):
            item = self._list_item_for(idx)
            if item is None:
                continue
            item.setText(self._make_list_item_text(idx, row))
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
        from DocProcessorPro.dpp_scripts.keyword_scanner_scripts.keyword_scanner_codebase import (
            extract_page_text,
        )

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

        # Build a lazy-loaded sidecar index: source_pdf_path → {1-indexed page_num → record}
        _sidecar_cache: dict[str, dict[int, dict]] = {}

        def _sidecar_rec(src_path: str, page_1idx: int) -> "dict | None":
            if src_path not in _sidecar_cache:
                from pathlib import Path as _Path
                stem = _Path(src_path).stem
                sidecar_path = self._output_dir / "_sidecars" / f"{stem}_page_texts.jsonl"
                page_map: dict[int, dict] = {}
                if sidecar_path.exists():
                    for _line in sidecar_path.read_text(encoding="utf-8").splitlines():
                        _line = _line.strip()
                        if not _line:
                            continue
                        try:
                            _rec = json.loads(_line)
                            page_map[int(_rec["page_num"])] = _rec
                        except (json.JSONDecodeError, KeyError, ValueError):
                            pass
                _sidecar_cache[src_path] = page_map
            return _sidecar_cache[src_path].get(page_1idx)

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

            # Effective provider/date values (override > auto-extracted)
            auto_svc = row.get("service_date", "") or None
            eff_svc = self._service_date_overrides.get(idx) or auto_svc
            auto_npi = row.get("provider_npi", "") or None
            eff_npi = self._provider_npi_overrides.get(idx) or auto_npi
            auto_hint = row.get("provider_name_hint", "") or None
            eff_hint = self._provider_name_overrides.get(idx) or auto_hint
            eff_assigned = self._provider_assigned.get(idx) or None
            eff_pk = self._provider_key(row, idx)

            # Page text: prefer sidecar (written during scan), fall back to on-demand re-extraction
            _sr = _sidecar_rec(src, pg)
            if _sr and _sr.get("text"):
                _page_text: str = _sr["text"]
                _page_text_source = "sidecar"
            else:
                _page_text = (
                    extract_page_text(src, pg)
                    or existing.get((src, pg), {}).get("page_text", "")
                )
                _page_text_source = "fresh_extraction" if _page_text else "unavailable"

            rec = {
                "schema_version": 1,
                "label": label_str,
                "rejection_reason": rejection_reason,
                "scanner_decision": row.get("scanner_decision")
                    or ("matched" if self._mode == "matched" else "excluded"),
                "user_override": (
                    (label_str == "include" and row.get("scanner_decision") == "excluded")
                    or (label_str == "exclude" and row.get("scanner_decision") == "matched")
                ) if self._mode == "combined" else False,
                "decision_source": self._decision_sources.get(idx, "user"),
                "triage_label": triage_label_str,
                "triage_rejection_reason": triage_rejection_reason,
                "triage_override": triage_label_str is not None
                and triage_label_str != label_str,
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
                "service_date": eff_svc,
                "provider_npi": eff_npi,
                "provider_name_hint": eff_hint,
                "provider_name": eff_assigned,
                "provider_key": eff_pk,
                "page_text": _page_text,
                "page_text_source": _page_text_source,
                "raw_service_date_str": (
                    (_sr.get("raw_service_date_str") if _sr else None)
                    or row.get("raw_service_date_str") or None
                ),
                "provider_name_context": (
                    (_sr.get("provider_name_context") if _sr else None)
                    or row.get("provider_name_context") or None
                ),
                "record_type": (
                    self._llm_index.get(Path(src).stem, {}).get(pg, {}).get("record_type")
                    or None
                ),
                "llm_confidence": (
                    self._llm_index.get(Path(src).stem, {}).get(pg, {}).get("confidence")
                    or None
                ),
            }
            # Category / stream fields
            from DocProcessorPro.dpp_scripts.keyword_scanner_scripts.keyword_scanner_codebase import (
                DEFAULT_CATEGORIES,
                highest_weight_category,
            )
            system_primary = highest_weight_category(cat_names, DEFAULT_CATEGORIES)
            cat_override = self._category_overrides.get(idx)
            eff_primary = cat_override or system_primary
            rec["scan_stream"] = self._effective_stream(idx)
            rec["system_primary_category"] = system_primary
            rec["primary_category"] = eff_primary
            rec["category_override"] = bool(cat_override)

            # Emit _predicted siblings only for fields the user actually corrected
            if cat_override and cat_override != system_primary:
                rec["primary_category_predicted"] = system_primary
            if self._service_date_overrides.get(idx) and eff_svc != auto_svc:
                rec["service_date_predicted"] = auto_svc
            if self._provider_npi_overrides.get(idx) and eff_npi != auto_npi:
                rec["provider_npi_predicted"] = auto_npi
            if self._provider_name_overrides.get(idx) and eff_hint != auto_hint:
                rec["provider_name_hint_predicted"] = auto_hint

            existing[(src, pg)] = rec

        # Build provider summary from approved pages
        from collections import Counter, defaultdict as _defaultdict
        provider_pages: dict[str, list[dict]] = _defaultdict(list)
        for (src, pg), rec in existing.items():
            if rec.get("label") == "include":
                pk = rec.get("provider_key") or src
                provider_pages[pk].append(rec)

        provider_date_ranges: dict[str, dict] = {}
        for pk, recs in provider_pages.items():
            all_dates: list[str] = []
            for r in recs:
                svc = r.get("service_date")
                if svc:
                    all_dates.append(svc)
                else:
                    all_dates.extend(r.get("dates_on_page") or [])
            name_counter: Counter[str] = Counter(
                r["provider_name"] for r in recs if r.get("provider_name")
            )
            provider_date_ranges[pk] = {
                "provider_name": name_counter.most_common(1)[0][0] if name_counter else None,
                "service_date_min": min(all_dates) if all_dates else None,
                "service_date_max": max(all_dates) if all_dates else None,
                "approved_page_count": len(recs),
            }

        summary_rec = {
            "_type": "provider_summary",
            "generated_at": now,
            "provider_date_ranges": provider_date_ranges,
        }

        # Write corrections JSONL (one record per page where any field was corrected)
        corrections_path = self._feedback_path.with_name(
            self._feedback_path.stem + "_corrections.jsonl"
        )
        corrections: list[dict] = []
        for idx, row in enumerate(self._rows):
            if idx not in self._decisions:
                continue
            src = row.get("source_pdf_path", "")
            pg = int(row.get("page_num", 0))
            corr: dict = {}
            if idx in self._service_date_overrides:
                auto = row.get("service_date") or None
                corrected = self._service_date_overrides[idx]
                if corrected != auto:
                    _csr = _sidecar_rec(src, pg)
                    raw_date = (
                        (_csr.get("raw_service_date_str") if _csr else None)
                        or row.get("raw_service_date_str") or None
                    )
                    corr["service_date"] = {"raw": raw_date, "auto": auto, "corrected": corrected}
            if idx in self._provider_npi_overrides:
                auto = row.get("provider_npi") or None
                corrected = self._provider_npi_overrides[idx]
                if corrected != auto:
                    corr["provider_npi"] = {"auto": auto, "corrected": corrected}
            if idx in self._provider_name_overrides:
                auto = row.get("provider_name_hint") or None
                corrected = self._provider_name_overrides[idx]
                if corrected != auto:
                    _csr = _sidecar_rec(src, pg)
                    raw_ctx = (
                        (_csr.get("provider_name_context") if _csr else None)
                        or row.get("provider_name_context") or None
                    )
                    corr["provider_name_hint"] = {"raw_context": raw_ctx, "auto": auto, "corrected": corrected}
            if idx in self._category_overrides:
                from DocProcessorPro.dpp_scripts.keyword_scanner_scripts.keyword_scanner_codebase import (
                    DEFAULT_CATEGORIES,
                    highest_weight_category,
                )
                row_cats = [c for c in row.get("categories_matched", "").split("|") if c]
                auto_cat = highest_weight_category(row_cats, DEFAULT_CATEGORIES)
                corrected_cat = self._category_overrides[idx]
                if corrected_cat != auto_cat:
                    corr["primary_category"] = {
                        "raw_categories": row_cats,
                        "auto": auto_cat,
                        "corrected": corrected_cat,
                    }
            if corr:
                _csr = _sidecar_rec(src, pg)
                _ctext: str = (
                    (_csr.get("text", "") if _csr else "")
                    or existing.get((src, pg), {}).get("page_text", "")
                )
                corrections.append({
                    "source_pdf_path": src,
                    "page_num": pg,
                    "page_text": _ctext[:1500],
                    "corrections": corr,
                })

        if corrections:
            with open(corrections_path, "w", encoding="utf-8") as cf:
                for c in corrections:
                    cf.write(json.dumps(c) + "\n")

        with open(self._feedback_path, "w", encoding="utf-8") as f:
            for rec in existing.values():
                f.write(json.dumps(rec) + "\n")
            f.write(json.dumps(summary_rec) + "\n")

        # Persist provider registry (survives draft deletion)
        self._save_provider_registry()

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
                1
                for d in self._decisions.values()
                if d in ("rejected_duplicate", "rejected_irrelevant")
            )
            if remove_count == 0:
                QMessageBox.information(
                    self,
                    "Nothing to Remove",
                    "No pages have been marked for removal yet.",
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
        elif self._mode == "combined":
            approved_count = sum(1 for d in self._decisions.values() if d == "approved")
            if approved_count == 0:
                QMessageBox.information(
                    self, "Nothing to Apply", "No pages have been approved yet."
                )
                return
            total = len(self._rows)
            unapproved = total - approved_count
            reply = QMessageBox.question(
                self,
                "Apply to Consolidated",
                f"Rebuild _consolidated.pdf from {approved_count} approved page(s) "
                f"and _consolidated_unmatched.pdf from {unapproved} remaining page(s)?\n\n"
                "This will overwrite both output files. Export feedback first.",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No,
            )
        else:
            approved_count = sum(1 for d in self._decisions.values() if d == "approved")
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
        elif self._mode == "combined":
            msg = f"Done — {primary} page(s) written to _consolidated.pdf."
            skip_note = (
                f"\n{skipped} page(s) skipped — source file(s) not found at stored path."
            )
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


