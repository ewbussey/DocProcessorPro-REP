"""
Pure keyword/regex PDF page scanner.

scan_pdf(pdf_path, categories, min_hits) -> ScanResult
scan_directory(input_dir, output_dir, categories, min_hits) -> dict[str, list[PageMatch]]
"""

from __future__ import annotations

import csv
import logging
import os
import shutil
import threading
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from PIL import Image

Image.MAX_IMAGE_PIXELS = 300_000_000  # large-format medical pages at 300 DPI can exceed the 89 MP default

import pypdf

from .models import KeywordCategory, PageMatch, PageExclusion, ScanResult
from .categories import (
    DEFAULT_CATEGORIES, CLINICAL_ANCHOR_CATEGORIES, highest_weight_category,
    _BILLS_MIN_HITS, _BILLS_REQUIRE_CATEGORIES,
    CATEGORY_THERAPY, CATEGORY_MEDICAL_TREATMENT, CATEGORY_BILLING,
    CATEGORY_INJURY_LEGAL, CATEGORY_IMAGING, CATEGORY_BEHAVIORAL_HEALTH,
    CATEGORY_VOCATIONAL, CATEGORY_DOCUMENT_TYPE,
)
from .extraction import (
    scan_pdf, extract_page_text, _TESSERACT_EXE, _POPPLER_BIN,
    _find_tesseract, _find_poppler,
)
from .pdf_ops import (
    extract_matched_pages, extract_unmatched_pages,
    write_manifest, write_dates_csv, write_matched_manifest, write_unmatched_manifest,
    write_page_texts_sidecar,
    consolidate_records, consolidate_bills, consolidate_to_pdf,
    consolidate_manifests, consolidate_dates, consolidate_unmatched,
    consolidate_all_scored,
    _make_separator_page,
)

log = logging.getLogger(__name__)

# These suffixes are checked when deciding whether a safe_stem already has
# output files in the target directory.
_PER_PDF_SUFFIXES = (
    "_records.pdf",
    "_records_manifest.csv",
    "_bills.pdf",
    "_bills_manifest.csv",
    "_manifest.csv",
    "_dates.csv",
    "_unmatched.pdf",
    "_unmatched_manifest.csv",
    "_page_texts.jsonl",
)


def _unique_stem(out_dir: Path, safe_stem: str) -> str:
    """Return safe_stem if no output files exist for it; otherwise iterate.

    Checks for the existence of any file whose name is
    ``{safe_stem}{suffix}`` for each suffix in _PER_PDF_SUFFIXES.  If a
    conflict is found, tries ``{safe_stem}_001``, ``_002``, … until a free
    slot is found.
    """
    if not any((out_dir / f"{safe_stem}{s}").exists() for s in _PER_PDF_SUFFIXES):
        return safe_stem
    n = 1
    while True:
        candidate = f"{safe_stem}_{n:03d}"
        if not any((out_dir / f"{candidate}{s}").exists() for s in _PER_PDF_SUFFIXES):
            return candidate
        n += 1


def scan_directory(
    input_dir: str,
    output_dir: str,
    categories: list[KeywordCategory],
    min_hits: float = 3.0,
    page_buffer: int = 0,
    require_categories: frozenset[str] | None = None,
    require_anchor: bool = False,
    progress_callback: Callable[[str], None] | None = None,
) -> dict[str, ScanResult]:
    """
    Batch-process all .pdf files in input_dir.

    For each PDF with at least one match:
    - Writes {stem}_matched.pdf to output_dir
    - Writes {stem}_manifest.csv to output_dir

    For each PDF with at least one scored-but-excluded page:
    - Writes {stem}_unmatched.pdf to output_dir
    - Writes {stem}_unmatched_manifest.csv to output_dir

    After all PDFs are processed, consolidates matched and unmatched outputs
    into _consolidated.pdf / _consolidated_unmatched.pdf respectively.

    Args:
        progress_callback: Optional callable receiving a status string before
                        each file and on each OCR page (used by the GUI worker).

    Returns: {pdf_stem: ScanResult} for all processed files.
    """
    in_dir = Path(input_dir)
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    pdfs = sorted(in_dir.rglob("*.pdf"))
    if not pdfs:
        log.warning("No PDF files found in %s (including subfolders).", in_dir)
        return {}

    log.info("Processing %d PDF(s) from %s (recursive).", len(pdfs), in_dir)

    # Pre-compute collision-safe stems in sorted order so that consolidation
    # order is deterministic regardless of which threads finish first.
    items: list[tuple[Path, str]] = []
    for pdf_path in pdfs:
        rel_parts = pdf_path.relative_to(in_dir).with_suffix("").parts
        items.append((pdf_path, "_".join(rel_parts)))

    results: dict[str, ScanResult] = {}
    _depo_lock = threading.Lock()
    # Maps original safe_stem → actual output stem (may differ when files exist).
    # Keys are unique per-thread (one PDF per worker), so no lock is needed.
    _out_stems: dict[str, str] = {}

    def _scan_one(pdf_path: Path, safe_stem: str) -> tuple[str, ScanResult]:
        empty = ScanResult(
            matches=[],
            exclusions=[],
            source_date_first=None,
            source_date_last=None,
            source_date_count=0,
            all_page_dates={},
        )
        if progress_callback:
            progress_callback(f"Scanning {pdf_path.name}…")

        from .extraction import _is_deposition, _next_deposition_path
        if _is_deposition(str(pdf_path)):
            with _depo_lock:
                dest = _next_deposition_path(out_dir)
                shutil.copy2(pdf_path, dest)
            log.info("Deposition detected: %s → %s", pdf_path.name, dest.name)
            if progress_callback:
                progress_callback(f"Deposition saved: {dest.name}")
            return safe_stem, empty

        # ── Pass R: records (clinical content) ──────────────────────────────
        try:
            records_result = scan_pdf(
                str(pdf_path),
                categories,
                min_hits=min_hits,
                require_categories=require_categories,
                require_anchor=require_anchor,
                progress_callback=progress_callback,
            )
        except Exception:
            log.exception("Failed to scan %s (records pass) — skipping.", pdf_path.name)
            return safe_stem, empty

        # ── Pass B: bills/affidavits (low threshold, BILLING + INJURY_LEGAL) ─
        try:
            bills_result = scan_pdf(
                str(pdf_path),
                categories,
                min_hits=_BILLS_MIN_HITS,
                require_categories=_BILLS_REQUIRE_CATEGORIES,
                require_anchor=False,
                progress_callback=progress_callback,
            )
        except Exception:
            log.exception("Failed to scan %s (bills pass) — skipping bills.", pdf_path.name)
            bills_result = empty

        # ── De-duplication: records take precedence over bills ───────────────
        # A page matched by records is excluded from the bills output even if
        # it also satisfies the bills criteria.
        records_page_nums: set[int] = {m.page_num for m in records_result.matches}
        bills_matches = [
            m for m in bills_result.matches if m.page_num not in records_page_nums
        ]

        # ── Unmatched: excluded by records AND not matched by bills ──────────
        all_matched: set[int] = records_page_nums | {m.page_num for m in bills_matches}
        records_excl_nums: set[int] = {e.page_num for e in records_result.exclusions}
        merged_exclusions = list(records_result.exclusions) + [
            e for e in bills_result.exclusions if e.page_num not in records_excl_nums
        ]
        final_unmatched = [e for e in merged_exclusions if e.page_num not in all_matched]

        # ── Synthetic ScanResult (preserves public interface) ────────────────
        combined = ScanResult(
            matches=records_result.matches + bills_matches,
            exclusions=final_unmatched,
            source_date_first=(
                records_result.source_date_first or bills_result.source_date_first
            ),
            source_date_last=max(
                filter(None, [records_result.source_date_last,
                               bills_result.source_date_last]),
                default=None,
            ),
            source_date_count=records_result.source_date_count,
            all_page_dates=records_result.all_page_dates,
        )

        # Resolve a non-conflicting stem before writing any output files.
        out_stem = _unique_stem(out_dir, safe_stem)
        _out_stems[safe_stem] = out_stem
        if out_stem != safe_stem:
            log.info(
                "%s: output files already exist for stem %r — writing as %r.",
                pdf_path.name, safe_stem, out_stem,
            )

        if records_result.matches:
            extract_matched_pages(
                str(pdf_path),
                records_result.matches,
                str(out_dir / f"{out_stem}_records.pdf"),
                page_buffer=page_buffer,
            )
            write_manifest(
                records_result.matches,
                str(out_dir / f"{out_stem}_manifest.csv"),
                source_date_first=records_result.source_date_first,
                source_date_last=records_result.source_date_last,
                source_date_count=records_result.source_date_count,
                scan_stream="records",
            )
            write_matched_manifest(
                str(pdf_path),
                records_result.matches,
                str(out_dir / f"{out_stem}_records_manifest.csv"),
                min_hits_threshold=min_hits,
                scan_stream="records",
            )
            write_dates_csv(
                records_result.all_page_dates,
                str(out_dir / f"{out_stem}_dates.csv"),
            )
        else:
            log.info("%s: no records matches.", pdf_path.name)

        if bills_matches:
            extract_matched_pages(
                str(pdf_path),
                bills_matches,
                str(out_dir / f"{out_stem}_bills.pdf"),
                page_buffer=page_buffer,
            )
            write_matched_manifest(
                str(pdf_path),
                bills_matches,
                str(out_dir / f"{out_stem}_bills_manifest.csv"),
                min_hits_threshold=_BILLS_MIN_HITS,
                scan_stream="bills",
            )
        else:
            log.info("%s: no bills matches.", pdf_path.name)

        if final_unmatched:
            extract_unmatched_pages(
                str(pdf_path),
                final_unmatched,
                str(out_dir / f"{out_stem}_unmatched.pdf"),
            )
            write_unmatched_manifest(
                str(pdf_path),
                final_unmatched,
                str(out_dir / f"{out_stem}_unmatched_manifest.csv"),
                scan_stream="unmatched",
            )

        if records_result.page_texts:
            write_page_texts_sidecar(
                out_dir / f"{out_stem}_page_texts.jsonl",
                records_result.page_texts,
                combined.matches,
                combined.exclusions,
            )

        return safe_stem, combined

    max_workers = min(len(pdfs), os.cpu_count() or 4)
    completed = 0
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {pool.submit(_scan_one, p, s): (p, s) for p, s in items}
        for fut in as_completed(futures):
            pdf_path, safe_stem = futures[fut]
            completed += 1
            try:
                stem, result = fut.result()
                results[stem] = result
            except Exception:
                log.exception("Unexpected error processing %s.", pdf_path.name)
                results[safe_stem] = ScanResult(
                    matches=[],
                    exclusions=[],
                    source_date_first=None,
                    source_date_last=None,
                    source_date_count=0,
                    all_page_dates={},
                )
            if progress_callback:
                progress_callback(
                    f"Finished {pdf_path.name} ({completed}/{len(pdfs)})…"
                )

    safe_stems = [s for _, s in items]

    # Build all three stem lists before any consolidation runs — consolidate_records,
    # consolidate_bills, and consolidate_unmatched delete per-stem files, so
    # consolidate_all_scored must read them first.
    records_stems = [
        _out_stems[s]
        for s in safe_stems
        if _out_stems.get(s) and (out_dir / f"{_out_stems[s]}_records.pdf").exists()
    ]
    bills_stems = [
        _out_stems[s]
        for s in safe_stems
        if _out_stems.get(s) and (out_dir / f"{_out_stems[s]}_bills.pdf").exists()
    ]
    unmatched_stems = [
        _out_stems[s]
        for s in safe_stems
        if _out_stems.get(s) and (out_dir / f"{_out_stems[s]}_unmatched.pdf").exists()
    ]

    # Step 1: combined review PDF first — reads per-stem files before Step 2–4 delete them.
    if records_stems or bills_stems or unmatched_stems:
        consolidate_all_scored(
            output_dir, records_stems, bills_stems, unmatched_stems, progress_callback
        )

    # Step 2: records consolidation (OCR applied; deletes {stem}_records.* files).
    if records_stems:
        consolidate_records(output_dir, records_stems, progress_callback)
        consolidate_manifests(output_dir, records_stems)
        consolidate_dates(output_dir, records_stems)

    # Step 3: bills consolidation (no OCR; deletes {stem}_bills.* files).
    if bills_stems:
        consolidate_bills(output_dir, bills_stems, progress_callback)

    # Step 4: unmatched consolidation (deletes {stem}_unmatched.* files).
    if unmatched_stems:
        consolidate_unmatched(output_dir, unmatched_stems, progress_callback)

    return results


def apply_feedback(
    output_dir: str,
    feedback_path: str,
    progress_callback: Callable[[str], None] | None = None,
) -> tuple[int, int, int]:
    """
    Apply approved feedback decisions to the consolidated outputs.

    Reads _feedback.jsonl, extracts approved pages from their source PDFs,
    appends them (with a separator page) to _consolidated.pdf, and rebuilds
    _consolidated_unmatched.pdf to remove the now-approved pages.

    Approved pages are appended to the end of _consolidated.pdf rather than
    inserted in source-document order — a full order-preserving rebuild would
    require absolute paths in the matched manifest (currently stores bare
    filenames only).

    Args:
        output_dir:    Directory containing _consolidated.pdf and
                       _consolidated_unmatched_manifest.csv.
        feedback_path: Path to _feedback.jsonl.

    Returns: (pages_approved, pages_rejected, pages_skipped_missing_source)
    """
    import json

    out = Path(output_dir)
    fb = Path(feedback_path)

    if not fb.exists():
        raise FileNotFoundError(f"Feedback file not found: {fb}")

    # Parse feedback records
    approved: dict[str, list[int]] = {}  # source_pdf_path → list of 0-indexed page_nums
    rejected_count = 0
    for line in fb.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            log.warning("Skipping malformed feedback line: %s", line[:80])
            continue
        label = rec.get("label", "")
        if label == "include":
            src = rec.get("source_pdf_path", "")
            pg = rec.get("page_num")
            if src and pg is not None:
                approved.setdefault(src, []).append(int(pg) - 1)  # 0-indexed
        elif label == "exclude":
            rejected_count += 1

    if not approved:
        log.info("apply_feedback: no approved pages to apply.")
        return 0, rejected_count, 0

    # Append approved pages to _consolidated.pdf
    consolidated = out / "_consolidated.pdf"
    if not consolidated.exists():
        raise FileNotFoundError(f"Consolidated PDF not found: {consolidated}")

    if progress_callback:
        progress_callback("Applying approved pages to consolidated PDF…")

    existing_reader = pypdf.PdfReader(str(consolidated))
    new_writer = pypdf.PdfWriter()
    for page in existing_reader.pages:
        new_writer.add_page(page)

    pages_approved = 0
    skipped_missing = 0
    for src_path, page_nums in sorted(approved.items()):
        if not Path(src_path).exists():
            log.warning(
                "apply_feedback: source PDF not found at stored path — skipping. (%s)",
                src_path,
            )
            skipped_missing += len(page_nums)
            continue
        sep_buf = _make_separator_page(Path(src_path).name + " [approved]")
        sep_reader = pypdf.PdfReader(sep_buf)
        new_writer.add_page(sep_reader.pages[0])
        src_reader = pypdf.PdfReader(src_path)
        for pn in sorted(page_nums):
            if 0 <= pn < len(src_reader.pages):
                new_writer.add_page(src_reader.pages[pn])
                pages_approved += 1

    tmp = consolidated.with_suffix(".tmp.pdf")
    with open(tmp, "wb") as f:
        new_writer.write(f)
    tmp.replace(consolidated)
    log.info("apply_feedback: appended %d approved page(s) to %s.", pages_approved, consolidated.name)

    # Rebuild _consolidated_unmatched.pdf by removing approved pages
    unmatched_manifest = out / "_consolidated_unmatched_manifest.csv"
    if unmatched_manifest.exists() and pages_approved:
        if progress_callback:
            progress_callback("Rebuilding consolidated unmatched PDF…")

        approved_keys: set[tuple[str, int]] = {
            (src, pn + 1)  # back to 1-indexed to match manifest
            for src, pns in approved.items()
            for pn in pns
        }

        with open(unmatched_manifest, newline="", encoding="utf-8") as f:
            all_rows = list(csv.DictReader(f))

        remaining_rows = [
            r for r in all_rows
            if (r.get("source_pdf_path", ""), int(r.get("page_num", 0)))
            not in approved_keys
        ]

        if len(remaining_rows) < len(all_rows):
            unmatched_pdf_path = out / "_consolidated_unmatched.pdf"
            if not remaining_rows:
                # Every unmatched page was approved — clear the unmatched files.
                for old in (unmatched_pdf_path, unmatched_manifest):
                    if old.exists():
                        old.unlink()
            elif unmatched_pdf_path.exists():
                # Fast path: slice the existing consolidated unmatched PDF
                # using consolidated_page_num — no source-PDF traversal needed.
                if progress_callback:
                    progress_callback("Rebuilding consolidated unmatched PDF…")

                unmatched_reader = pypdf.PdfReader(str(unmatched_pdf_path))
                total_cpages = len(unmatched_reader.pages)

                # 1-indexed content pages to keep
                keep_content: set[int] = {
                    int(r["consolidated_page_num"])
                    for r in remaining_rows
                    if r.get("consolidated_page_num")
                }

                # Separator for each source sits immediately before its first
                # content page in the original consolidated PDF.
                by_source_cpns: dict[str, list[int]] = {}
                for r in all_rows:
                    src = r.get("source_pdf_path", "")
                    cpn = r.get("consolidated_page_num")
                    if src and cpn:
                        by_source_cpns.setdefault(src, []).append(int(cpn))
                remaining_sources = {r.get("source_pdf_path", "") for r in remaining_rows}
                sep_pages: set[int] = set()
                for src, cpns in by_source_cpns.items():
                    if src in remaining_sources:
                        sep_pnum = min(cpns) - 1
                        if 1 <= sep_pnum <= total_cpages:
                            sep_pages.add(sep_pnum)

                keep_all = keep_content | sep_pages  # 1-indexed
                new_writer = pypdf.PdfWriter()
                new_page_map: dict[int, int] = {}  # old cpn → new 1-indexed position
                new_pos = 0
                for old_pnum in range(1, total_cpages + 1):
                    if old_pnum in keep_all:
                        new_writer.add_page(unmatched_reader.pages[old_pnum - 1])
                        new_pos += 1
                        if old_pnum in keep_content:
                            new_page_map[old_pnum] = new_pos

                tmp_unmatched = unmatched_pdf_path.with_suffix(".tmp.pdf")
                with open(tmp_unmatched, "wb") as f:
                    new_writer.write(f)
                tmp_unmatched.replace(unmatched_pdf_path)

                # Rewrite manifest with updated consolidated_page_num values.
                updated_rows = []
                for r in remaining_rows:
                    cpn = r.get("consolidated_page_num")
                    if cpn and int(cpn) in new_page_map:
                        new_r = dict(r)
                        new_r["consolidated_page_num"] = new_page_map[int(cpn)]
                        updated_rows.append(new_r)
                if updated_rows:
                    fieldnames = list(updated_rows[0].keys())
                    with open(unmatched_manifest, "w", newline="", encoding="utf-8") as f:
                        w = csv.DictWriter(f, fieldnames=fieldnames)
                        w.writeheader()
                        w.writerows(updated_rows)
                log.info(
                    "apply_feedback: rebuilt unmatched PDF with %d content page(s) "
                    "from %d source(s).",
                    len(updated_rows),
                    len(remaining_sources),
                )

    log.info(
        "apply_feedback complete: %d approved, %d rejected, %d skipped (missing source).",
        pages_approved,
        rejected_count,
        skipped_missing,
    )
    return pages_approved, rejected_count, skipped_missing


def apply_combined_feedback(
    output_dir: str,
    feedback_path: str,
    progress_callback: Callable[[str], None] | None = None,
) -> tuple[int, int, int]:
    """Rebuild split outputs from a combined-review feedback file.

    Reads _combined_feedback.jsonl, then:
    - Rebuilds _consolidated_records.pdf from approved pages with scan_stream="records"
    - Rebuilds _consolidated_bills.pdf from approved pages with scan_stream="bills"
    - Rebuilds _consolidated_unmatched.pdf from all non-approved pages
    - Writes fresh manifests with new consolidated_page_num values
    - Runs OCR on _consolidated_records.pdf

    Returns: (pages_approved, pages_rejected_or_pending, pages_skipped_missing_source)
    """
    import json

    out = Path(output_dir)
    fb = Path(feedback_path)

    if not fb.exists():
        raise FileNotFoundError(f"Combined feedback file not found: {fb}")

    # Parse approved set from feedback
    approved: dict[str, list[int]] = {}  # source_pdf_path → 0-indexed page nums
    for line in fb.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            continue
        if rec.get("_type"):
            continue
        if rec.get("label") == "include":
            src = rec.get("source_pdf_path", "")
            pg = rec.get("page_num")
            if src and pg is not None:
                approved.setdefault(src, []).append(int(pg) - 1)

    approved_keys: set[tuple[str, int]] = {
        (src, pn + 1) for src, pns in approved.items() for pn in pns
    }

    # Read the combined review manifest to get all scored pages
    review_manifest = out / "_consolidated_review_manifest.csv"
    if not review_manifest.exists():
        raise FileNotFoundError(
            f"Combined review manifest not found: {review_manifest}"
        )
    with open(review_manifest, newline="", encoding="utf-8") as f:
        all_review_rows = list(csv.DictReader(f))

    approved_rows = [
        r for r in all_review_rows
        if (r.get("source_pdf_path", ""), int(r.get("page_num", 0))) in approved_keys
    ]
    unapproved_rows = [
        r for r in all_review_rows
        if (r.get("source_pdf_path", ""), int(r.get("page_num", 0))) not in approved_keys
    ]

    pages_skipped = 0

    def _build_pdf_and_manifest(
        rows: list[dict],
        pdf_dest: Path,
        manifest_dest: Path,
        label_suffix: str,
    ) -> int:
        nonlocal pages_skipped
        by_source: dict[str, list[dict]] = {}
        for r in rows:
            by_source.setdefault(r.get("source_pdf_path", ""), []).append(r)

        writer = pypdf.PdfWriter()
        all_out_rows: list[dict] = []
        current_page = 0
        pages_written = 0

        for src_path, src_rows in sorted(by_source.items()):
            if not Path(src_path).exists():
                log.warning(
                    "apply_combined_feedback: source PDF not found — skipping. (%s)",
                    src_path,
                )
                pages_skipped += len(src_rows)
                continue
            stem = Path(src_path).stem
            sep_buf = _make_separator_page(f"{stem}.pdf{label_suffix}")
            sep_reader = pypdf.PdfReader(sep_buf)
            writer.add_page(sep_reader.pages[0])
            writer.add_outline_item(stem, current_page)
            current_page += 1

            src_reader = pypdf.PdfReader(src_path)
            first_content = current_page
            page_nums_0idx = sorted(int(r["page_num"]) - 1 for r in src_rows)
            for pn in page_nums_0idx:
                if 0 <= pn < len(src_reader.pages):
                    writer.add_page(src_reader.pages[pn])
                    current_page += 1
                    pages_written += 1

            for i, r in enumerate(sorted(src_rows, key=lambda x: int(x["page_num"]))):
                new_r = {k: v for k, v in r.items() if k != "consolidated_page_num"}
                new_r["consolidated_page_num"] = first_content + i + 1
                all_out_rows.append(new_r)

        if not pages_written:
            return 0

        with open(pdf_dest, "wb") as f:
            writer.write(f)

        if all_out_rows:
            fieldnames = list(all_out_rows[0].keys())
            with open(manifest_dest, "w", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=fieldnames)
                w.writeheader()
                w.writerows(all_out_rows)

        return pages_written

    # Split approved rows by scan_stream.  Legacy rows without a scan_stream
    # column (from pre-dual-pass scans) default to "records".
    approved_records = [
        r for r in approved_rows
        if r.get("scan_stream", "records") in ("records", "")
        or r.get("scan_stream") is None
    ]
    approved_bills = [
        r for r in approved_rows if r.get("scan_stream") == "bills"
    ]

    if progress_callback:
        progress_callback("Rebuilding records PDF from approved pages…")

    records_pdf = out / "_consolidated_records.pdf"
    records_manifest = out / "_consolidated_records_manifest.csv"
    pages_records = _build_pdf_and_manifest(
        approved_records, records_pdf, records_manifest, ""
    )

    if pages_records and records_pdf.exists():
        from .pdf_ops import _ocr_consolidated
        _ocr_consolidated(records_pdf, progress_callback)

    if progress_callback:
        progress_callback("Rebuilding bills PDF from approved pages…")

    bills_pdf = out / "_consolidated_bills.pdf"
    bills_manifest = out / "_consolidated_bills_manifest.csv"
    pages_bills = _build_pdf_and_manifest(
        approved_bills, bills_pdf, bills_manifest, ""
    )

    if progress_callback:
        progress_callback("Rebuilding consolidated unmatched PDF…")

    unmatched_pdf = out / "_consolidated_unmatched.pdf"
    unmatched_manifest = out / "_consolidated_unmatched_manifest.csv"
    pages_unapproved = _build_pdf_and_manifest(
        unapproved_rows, unmatched_pdf, unmatched_manifest, " [excluded]"
    )

    pages_approved = pages_records + pages_bills
    log.info(
        "apply_combined_feedback complete: %d approved (%d records, %d bills), "
        "%d unapproved, %d skipped.",
        pages_approved, pages_records, pages_bills, pages_unapproved, pages_skipped,
    )
    return pages_approved, pages_unapproved, pages_skipped


def apply_matched_feedback(
    output_dir: str,
    feedback_path: str,
    progress_callback: Callable[[str], None] | None = None,
) -> tuple[int, int, int]:
    """Remove user-rejected pages from _consolidated.pdf and rebuild it.

    Reads _matched_feedback.jsonl for "exclude" decisions (duplicate or irrelevant),
    then rebuilds _consolidated.pdf from source PDFs using _consolidated_matched_manifest.csv
    as the page index, skipping excluded pages.  Pages appended by a prior apply_feedback()
    call (those not in the matched manifest) are preserved verbatim.

    Args:
        output_dir:    Directory containing _consolidated.pdf and
                       _consolidated_matched_manifest.csv.
        feedback_path: Path to _matched_feedback.jsonl.

    Returns: (pages_kept, pages_removed, pages_skipped_missing_source)
    """
    import json

    out = Path(output_dir)
    fb = Path(feedback_path)

    if not fb.exists():
        raise FileNotFoundError(f"Matched feedback file not found: {fb}")

    consolidated = out / "_consolidated.pdf"
    if not consolidated.exists():
        raise FileNotFoundError(f"Consolidated PDF not found: {consolidated}")

    matched_manifest = out / "_consolidated_matched_manifest.csv"
    if not matched_manifest.exists():
        raise FileNotFoundError(
            f"Consolidated matched manifest not found: {matched_manifest}\n"
            "Re-run the scan to generate the manifest before using matched review."
        )

    # Parse feedback — collect excluded pages
    excluded_keys: set[tuple[str, int]] = set()  # (source_pdf_path, 1-indexed page_num)
    removed_count = 0
    for line in fb.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            log.warning("Skipping malformed matched feedback line: %s", line[:80])
            continue
        if rec.get("label") == "exclude":
            src = rec.get("source_pdf_path", "")
            pg = rec.get("page_num")
            if src and pg is not None:
                excluded_keys.add((src, int(pg)))
                removed_count += 1

    if not excluded_keys:
        log.info("apply_matched_feedback: no excluded pages — nothing to rebuild.")
        return 0, 0, 0

    if progress_callback:
        progress_callback("Rebuilding consolidated matched PDF…")

    # Read manifest to know which original pages to keep
    with open(matched_manifest, newline="", encoding="utf-8") as f:
        manifest_rows = list(csv.DictReader(f))

    # Build a set of consolidated_page_nums (1-indexed) that were in the original
    # matched manifest so we can separate them from appended (post-apply) pages.
    manifest_consolidated_pages: set[int] = {
        int(r["consolidated_page_num"]) for r in manifest_rows
        if r.get("consolidated_page_num")
    }

    # Determine which original matched pages to keep
    keep_keys: set[tuple[str, int]] = set()
    skipped_missing = 0
    for row in manifest_rows:
        src = row.get("source_pdf_path", "")
        pg = int(row.get("page_num", 0))
        key = (src, pg)
        if key in excluded_keys:
            continue
        if not Path(src).exists():
            log.warning(
                "apply_matched_feedback: source PDF missing, skipping page %d from %s",
                pg, src,
            )
            skipped_missing += 1
            continue
        keep_keys.add(key)

    # Rebuild: group kept manifest rows by source PDF, preserving source-document order
    # Then append any pages from the consolidated PDF that were NOT in the original manifest
    # (i.e. those added by prior apply_feedback calls).
    pages_kept = 0
    new_writer = pypdf.PdfWriter()

    # Group manifest rows by source PDF in stem order
    from collections import OrderedDict
    stem_to_rows: OrderedDict[str, list[dict]] = OrderedDict()
    for row in manifest_rows:
        src = row.get("source_pdf_path", "")
        if (src, int(row.get("page_num", 0))) in keep_keys:
            stem_to_rows.setdefault(src, []).append(row)

    for src_path, rows in stem_to_rows.items():
        sep_buf = _make_separator_page(Path(src_path).name)
        sep_reader = pypdf.PdfReader(sep_buf)
        new_writer.add_page(sep_reader.pages[0])
        src_reader = pypdf.PdfReader(src_path)
        for row in rows:
            pn = int(row["page_num"]) - 1  # 0-indexed
            if 0 <= pn < len(src_reader.pages):
                new_writer.add_page(src_reader.pages[pn])
                pages_kept += 1

    # Append pages from consolidated PDF that were not in the matched manifest
    # (these were added by apply_feedback from unmatched review — keep them unchanged)
    existing_reader = pypdf.PdfReader(str(consolidated))
    total_consolidated = len(existing_reader.pages)
    for page_1idx in range(1, total_consolidated + 1):
        if page_1idx not in manifest_consolidated_pages:
            new_writer.add_page(existing_reader.pages[page_1idx - 1])

    tmp = consolidated.with_suffix(".tmp.pdf")
    with open(tmp, "wb") as f:
        new_writer.write(f)
    tmp.replace(consolidated)

    # Update the matched manifest to remove excluded rows
    remaining_manifest = [
        r for r in manifest_rows
        if (r.get("source_pdf_path", ""), int(r.get("page_num", 0))) not in excluded_keys
    ]
    if remaining_manifest:
        fieldnames = list(manifest_rows[0].keys())
        with open(matched_manifest, "w", newline="", encoding="utf-8") as f:
            csv_writer = csv.DictWriter(f, fieldnames=fieldnames)
            csv_writer.writeheader()
            csv_writer.writerows(remaining_manifest)
    else:
        matched_manifest.unlink()

    log.info(
        "apply_matched_feedback complete: %d kept, %d removed, %d skipped (missing source).",
        pages_kept,
        removed_count - skipped_missing,
        skipped_missing,
    )
    return pages_kept, removed_count, skipped_missing
