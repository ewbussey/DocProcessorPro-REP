from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import QEvent, QSize, QThread, Qt, Signal
from PySide6.QtGui import QColor, QImage, QPainter, QPixmap
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QDialog,
    QFrame,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)

from ._gui_utils import _apply_app_icon, _apply_win_minmax


_DEDUP_THRESHOLD = 75        # max Hamming distance out of 1024 bits (~7%), first pass
_DEDUP_CONFIRM_MAX_DIFF = 0.10  # max mean absolute pixel difference [0–1], second pass


def _dhash(qimage: "QImage") -> int:
    """1024-bit difference hash: horizontal gradient over a 33×32 grayscale thumbnail.

    Using a 33×32 grid (vs the legacy 9×8) gives 16× more discriminating bits so
    that documents sharing similar letterhead or overall gray-level distribution are
    no longer falsely grouped. Pair with _DEDUP_THRESHOLD = 75 (~7% of 1024 bits).
    """
    gray = qimage.convertToFormat(QImage.Format.Format_Grayscale8).scaled(
        QSize(33, 32),
        Qt.AspectRatioMode.IgnoreAspectRatio,
        Qt.TransformationMode.SmoothTransformation,
    )
    bits = gray.bits()
    h = 0
    for row in range(32):
        for col in range(32):
            h = (h << 1) | (1 if bits[row * 33 + col] > bits[row * 33 + col + 1] else 0)
    return h


def _pixel_mad(img_a: "QImage", img_b: "QImage") -> float:
    """Mean absolute pixel difference between two page images, normalised to [0, 1].

    Both images are reduced to a 128×166 grayscale thumbnail before comparison so
    minor rendering artefacts and scan-angle differences don't inflate the score.
    Lower = more similar (0 = identical, 1 = fully inverted).

    Used as the second-pass confirmation gate: first-pass dHash catches candidates,
    this function confirms only those with genuine pixel-level similarity.
    """
    _W, _H = 128, 166

    def _gray_bytes(img: "QImage") -> bytes:
        return bytes(
            img.convertToFormat(QImage.Format.Format_Grayscale8)
            .scaled(
                QSize(_W, _H),
                Qt.AspectRatioMode.IgnoreAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
            .bits()
        )

    ba, bb = _gray_bytes(img_a), _gray_bytes(img_b)
    return sum(abs(a - b) for a, b in zip(ba, bb)) / (_W * _H * 255)


def _hamming(a: int, b: int) -> int:
    return bin(a ^ b).count("1")


def _cluster_by_hash(
    hashes: list[tuple[int, int]], threshold: int = 10
) -> list[list[int]]:
    """Complete-linkage clustering: joins only if within threshold of ALL existing members."""
    clusters: list[list[int]] = []
    cluster_hashes: list[list[int]] = []
    for idx, h in hashes:
        placed = False
        for c_i, ch_list in enumerate(cluster_hashes):
            if all(_hamming(h, ch) <= threshold for ch in ch_list):
                clusters[c_i].append(idx)
                cluster_hashes[c_i].append(h)
                placed = True
                break
        if not placed:
            clusters.append([idx])
            cluster_hashes.append([h])
    return clusters


class _ClickableLabel(QLabel):
    """QLabel that emits clicked() when left-clicked."""

    clicked = Signal()

    def mousePressEvent(self, event) -> None:
        if event.button() == Qt.MouseButton.LeftButton:
            self.clicked.emit()
        super().mousePressEvent(event)


class _DedupPrefetchWorker(QThread):
    """Loads the review PDF in a background thread, renders all pages, computes
    perceptual hashes, and runs the coarse dHash clustering pass (Pass 1) so the
    results are ready before the user clicks 'Find Duplicates'.

    The worker owns its own QPdfDocument instance (separate from the one used by
    the review dialog) so there is no cross-thread sharing of Qt objects.
    """

    ready = Signal(list, object)  # (candidate_groups: list[list[int]], thumbnails: dict[int, QImage])

    def __init__(self, pdf_path: "Path", rows: list, parent=None) -> None:
        super().__init__(parent)
        self._pdf_path = pdf_path
        self._rows = list(rows)

    def run(self) -> None:
        try:
            from PySide6.QtPdf import QPdfDocument

            doc = QPdfDocument()
            doc.load(str(self._pdf_path))

            hashes: list[tuple[int, int]] = []
            thumbnails: dict[int, QImage] = {}

            for idx, row in enumerate(self._rows):
                page_0idx = max(0, int(row.get("consolidated_page_num", "1")) - 1)
                qimg = doc.render(page_0idx, QSize(300, 390))
                if not qimg.isNull():
                    thumbnails[idx] = qimg
                hash_img = doc.render(page_0idx, QSize(800, 1040))
                if not hash_img.isNull():
                    hashes.append((idx, _dhash(hash_img)))
                elif not qimg.isNull():
                    hashes.append((idx, _dhash(qimg)))

            doc.close()

            if len(hashes) >= 2:
                raw_clusters = _cluster_by_hash(hashes, threshold=_DEDUP_THRESHOLD)
                candidate_groups = [c for c in raw_clusters if len(c) > 1]
            else:
                candidate_groups = []

            self.ready.emit(candidate_groups, thumbnails)
        except Exception:
            self.ready.emit([], {})


class _ZoomDialog(QDialog):
    """Full-resolution page preview that dismisses when the user clicks outside it."""

    def changeEvent(self, event) -> None:
        super().changeEvent(event)
        if event.type() == QEvent.Type.ActivationChange and not self.isActiveWindow():
            self.accept()


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
        hi_res: "dict[int, QImage] | None" = None,
        service_date_overrides: "dict[int, str] | None" = None,
        cache_path: "Path | None" = None,
        parent=None,
    ) -> None:
        super().__init__(parent)
        _apply_win_minmax(self)
        _apply_app_icon(self)
        self._groups = groups
        self._cur_decisions = decisions
        self._cur_sources = decision_sources
        self._hi_res: dict[int, QImage] = hi_res or {}
        self._svc_overrides: dict[int, str] = service_date_overrides or {}
        self._cache_path = cache_path
        self._canonical_indices: list[int] = []
        total = sum(len(g) for g in groups)
        self.setWindowTitle(
            f"Find Duplicates — {len(groups)} group(s), {total} similar pages"
        )
        self.setMinimumSize(1300, 820)
        self.resize(1300, 820)
        self._keep_checks: list[list[tuple[int, QCheckBox]]] = []
        self._build_ui()
        self._load_cache()

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

            self._canonical_indices.append(canonical_idx)
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

                # Use an RGB32 canvas (no alpha channel) so any format returned by
                # QPdfDocument.render() — ARGB32 or ARGB32_Premultiplied — composites
                # correctly over white. A QPixmap canvas has a platform-native format
                # that can mismatch premultiplied sources and produce black thumbnails.
                canvas = QImage(qimg.width(), qimg.height(), QImage.Format.Format_RGB32)
                canvas.fill(Qt.GlobalColor.white)
                painter = QPainter(canvas)
                painter.drawImage(0, 0, qimg)
                painter.end()
                white = QPixmap.fromImage(canvas)
                thumb = _ClickableLabel()
                thumb.setPixmap(
                    white.scaled(
                        220,
                        286,
                        Qt.AspectRatioMode.KeepAspectRatio,
                        Qt.TransformationMode.SmoothTransformation,
                    )
                )
                thumb.setAlignment(Qt.AlignmentFlag.AlignCenter)
                thumb.setCursor(Qt.CursorShape.PointingHandCursor)
                thumb.setToolTip("Click to enlarge")
                thumb.clicked.connect(
                    lambda _checked=False, i=row_idx, q=qimg: self._zoom_page(i, q)
                )
                item_l.addWidget(thumb)

                src_path = row.get("source_pdf_path", "")
                filename = src_path.rsplit("/", 1)[-1].rsplit("\\", 1)[-1]
                score = float(row.get("total_hits", 0))
                pg = row.get("page_num", "?")
                cur_dec = self._cur_decisions.get(row_idx)
                cur_src = self._cur_sources.get(row_idx, "—")
                status = f"{cur_dec} ({cur_src})" if cur_dec else "undecided"

                svc = self._svc_overrides.get(row_idx, "") or row.get("service_date", "")
                raw_dates = row.get("dates_on_page", "")
                extra = [d for d in raw_dates.split("|") if d and d != svc] if raw_dates else []
                date_parts = ([f"Svc: {svc}"] if svc else []) + (
                    [", ".join(extra)] if extra else []
                )
                date_line = "  |  ".join(date_parts) if date_parts else "no dates"

                info = QLabel(f"{filename}\np.{pg}  score {score:.2f}\n{status}\n{date_line}")
                info.setWordWrap(True)
                info.setAlignment(Qt.AlignmentFlag.AlignCenter)
                item_l.addWidget(info)

                chk = QCheckBox("Keep")
                chk.setChecked(row_idx == canonical_idx)
                chk.stateChanged.connect(self._autosave)
                item_l.addWidget(chk, 0, Qt.AlignmentFlag.AlignHCenter)
                checks.append((row_idx, chk))
                items_row.addWidget(item_w)

            items_row.addStretch()
            frame_layout.addLayout(items_row)

            btn_row = QHBoxLayout()
            btn_row.addStretch()
            best_btn = QPushButton("Keep Highest Scored")
            best_btn.clicked.connect(
                lambda _, c=checks: [
                    chk.setChecked(i == 0) for i, (_, chk) in enumerate(c)
                ]
            )
            btn_row.addWidget(best_btn)
            frame_layout.addLayout(btn_row)

            self._keep_checks.append(checks)
            inner_layout.addWidget(frame)

        inner_layout.addStretch()
        scroll.setWidget(inner)
        root.addWidget(scroll)

        bottom = QHBoxLayout()
        reset_btn = QPushButton("Reset to Defaults")
        reset_btn.setToolTip("Restore the algorithm's original Keep selections and clear the saved cache")
        reset_btn.clicked.connect(self._reset_to_defaults)
        bottom.addWidget(reset_btn)
        bottom.addStretch()
        apply_btn = QPushButton("Apply Selections")
        apply_btn.setDefault(True)
        apply_btn.clicked.connect(self.accept)
        cancel_btn = QPushButton("Cancel (no changes)")
        cancel_btn.clicked.connect(self.reject)
        bottom.addWidget(apply_btn)
        bottom.addWidget(cancel_btn)
        root.addLayout(bottom)

    def _load_cache(self) -> None:
        if not self._cache_path or not self._cache_path.exists():
            return
        try:
            import json
            cached: dict = json.loads(self._cache_path.read_text(encoding="utf-8")).get("keep", {})
            for checks in self._keep_checks:
                for row_idx, chk in checks:
                    key = str(row_idx)
                    if key in cached:
                        chk.blockSignals(True)
                        chk.setChecked(bool(cached[key]))
                        chk.blockSignals(False)
        except Exception:
            pass

    def _autosave(self) -> None:
        if not self._cache_path:
            return
        import json
        keep = {str(row_idx): chk.isChecked() for checks in self._keep_checks for row_idx, chk in checks}
        try:
            self._cache_path.parent.mkdir(exist_ok=True)
            self._cache_path.write_text(json.dumps({"keep": keep}, ensure_ascii=False), encoding="utf-8")
        except Exception:
            pass

    def _reset_to_defaults(self) -> None:
        for (group, checks), canonical_idx in zip(zip(self._groups, self._keep_checks), self._canonical_indices):
            for row_idx, chk in checks:
                chk.blockSignals(True)
                chk.setChecked(row_idx == canonical_idx)
                chk.blockSignals(False)
        if self._cache_path and self._cache_path.exists():
            try:
                self._cache_path.unlink()
            except Exception:
                pass

    def _zoom_page(self, row_idx: int, qimg_lo: "QImage") -> None:
        """Open a scrollable full-resolution popup for the given page."""
        # Prefer the hi-res render when available; fall back to the thumbnail image.
        src = self._hi_res.get(row_idx, qimg_lo)

        # Composite onto white (same logic as the thumbnail path).
        canvas = QImage(src.width(), src.height(), QImage.Format.Format_RGB32)
        canvas.fill(Qt.GlobalColor.white)
        p = QPainter(canvas)
        p.drawImage(0, 0, src)
        p.end()
        full_pix = QPixmap.fromImage(canvas)

        # Close any existing zoom popup before opening a new one.
        if hasattr(self, "_zoom_dlg") and self._zoom_dlg is not None:
            self._zoom_dlg.close()

        dlg = _ZoomDialog(self)
        self._zoom_dlg = dlg
        dlg.setWindowTitle("Page Preview")
        _apply_win_minmax(dlg)

        # Scale to fit 85 % of the available screen, preserving aspect ratio.
        screen = QApplication.primaryScreen()
        if screen:
            avail = screen.availableGeometry()
            max_w = int(avail.width() * 0.85)
            max_h = int(avail.height() * 0.85)
        else:
            max_w, max_h = 1200, 900

        scaled_pix = full_pix.scaled(
            max_w,
            max_h,
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )

        img_label = QLabel()
        img_label.setPixmap(scaled_pix)
        img_label.setAlignment(Qt.AlignmentFlag.AlignCenter)

        scroll = QScrollArea()
        scroll.setWidget(img_label)
        scroll.setWidgetResizable(False)
        scroll.setAlignment(Qt.AlignmentFlag.AlignCenter)

        close_btn = QPushButton("Close")
        close_btn.clicked.connect(dlg.accept)

        layout = QVBoxLayout(dlg)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(6)
        layout.addWidget(scroll)
        layout.addWidget(close_btn, 0, Qt.AlignmentFlag.AlignRight)

        dlg.resize(min(scaled_pix.width() + 40, max_w), min(scaled_pix.height() + 80, max_h))
        dlg.show()

    def results(self) -> tuple[list[int], list[int]]:
        """Return (keep_indices, reject_indices) based on checkbox state."""
        keep: list[int] = []
        reject: list[int] = []
        for checks in self._keep_checks:
            for row_idx, chk in checks:
                (keep if chk.isChecked() else reject).append(row_idx)
        return keep, reject
