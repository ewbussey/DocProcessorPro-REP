from __future__ import annotations

import csv
import io
import logging
import os
from collections.abc import Callable
from pathlib import Path

import pypdf
from PIL import Image, ImageDraw, ImageFont

from .models import PageMatch, PageExclusion, ScanResult
from .extraction import _TESSERACT_EXE, _POPPLER_BIN, _find_tesseract, _find_poppler, extract_page_text

log = logging.getLogger(__name__)


def _expand_with_buffer(
    matches: list[PageMatch],
    total_pages: int,
    buffer: int,
) -> list[int]:
    """Return sorted, deduplicated 0-indexed page numbers covering all hit pages ± buffer."""
    hit_pages = {m.page_num for m in matches}
    expanded: set[int] = set()
    for p in hit_pages:
        for offset in range(-buffer, buffer + 1):
            n = p + offset
            if 0 <= n < total_pages:
                expanded.add(n)
    return sorted(expanded)


def extract_matched_pages(
    pdf_path: str,
    matches: list[PageMatch],
    output_path: str,
    page_buffer: int = 0,
) -> None:
    """
    Write all matched pages (plus optional context buffer) to a new PDF.

    Args:
        pdf_path:    Source PDF path.
        matches:     PageMatch list from scan_pdf().
        output_path: Destination .pdf path (parent directory created if absent).
        page_buffer: Number of pages before and after each hit to include.
    """
    if not matches:
        return

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    reader = pypdf.PdfReader(pdf_path)
    total = len(reader.pages)
    page_nums = (
        _expand_with_buffer(matches, total, page_buffer)
        if page_buffer
        else sorted(m.page_num for m in matches)
    )

    writer = pypdf.PdfWriter()
    for n in page_nums:
        writer.add_page(reader.pages[n])

    with open(out, "wb") as f:
        writer.write(f)

    log.info(
        "Wrote %d page(s) to %s (%d hit(s), buffer=%d).",
        len(page_nums),
        out.name,
        len(matches),
        page_buffer,
    )


def extract_unmatched_pages(
    pdf_path: str,
    exclusions: list[PageExclusion],
    output_path: str,
) -> None:
    """
    Write all scored-but-excluded pages to a new PDF.

    No page buffer is applied — each excluded page stands alone for reviewer judgment.

    Args:
        pdf_path:    Source PDF path.
        exclusions:  PageExclusion list from scan_pdf().
        output_path: Destination .pdf path.
    """
    if not exclusions:
        return

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    reader = pypdf.PdfReader(pdf_path)
    writer = pypdf.PdfWriter()
    for exc in sorted(exclusions, key=lambda x: x.page_num):
        writer.add_page(reader.pages[exc.page_num])

    with open(out, "wb") as f:
        writer.write(f)

    log.info("Wrote %d unmatched page(s) to %s.", len(exclusions), out.name)


def write_manifest(
    matches: list[PageMatch],
    output_csv_path: str,
    source_date_first: str | None = None,
    source_date_last: str | None = None,
    source_date_count: int = 0,
    scan_stream: str = "records",
) -> None:
    """
    Write a CSV manifest with one row per matched page.

    Columns: page_num (1-indexed), categories_matched, keywords_hit,
    extraction_method, total_hits, dates_on_page, source_date_first,
    source_date_last, source_date_count.

    source_date_* columns repeat the same value on every row (per-PDF summary).
    Multi-value fields are pipe-separated.
    """
    out = Path(output_csv_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    with open(out, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "page_num",
                "categories_matched",
                "keywords_hit",
                "extraction_method",
                "total_hits",
                "dates_on_page",
                "source_date_first",
                "source_date_last",
                "source_date_count",
                "scan_stream",
            ]
        )
        for m in sorted(matches, key=lambda x: x.page_num):
            writer.writerow(
                [
                    m.page_num + 1,
                    "|".join(m.categories),
                    "|".join(m.keywords_hit),
                    m.extraction_method,
                    m.total_hits,
                    "|".join(m.dates_on_page),
                    source_date_first or "",
                    source_date_last or "",
                    source_date_count,
                    scan_stream,
                ]
            )

    log.info("Manifest written to %s.", out.name)


def write_dates_csv(
    all_page_dates: dict[int, list[str]],
    output_csv_path: str,
) -> None:
    """
    Write a per-PDF dates inventory CSV.

    Columns: date (ISO), pages_found (pipe-separated 1-indexed page numbers).
    Covers all pages in the source document, sorted by date ascending.
    """
    date_to_pages: dict[str, list[int]] = {}
    for page_num, dates in all_page_dates.items():
        for d in dates:
            date_to_pages.setdefault(d, []).append(page_num + 1)

    if not date_to_pages:
        return

    out = Path(output_csv_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["date", "pages_found"])
        for date_iso in sorted(date_to_pages):
            writer.writerow(
                [
                    date_iso,
                    "|".join(str(p) for p in sorted(date_to_pages[date_iso])),
                ]
            )
    log.info("Dates CSV written to %s.", out.name)


def write_matched_manifest(
    pdf_path_absolute: str,
    matches: list[PageMatch],
    output_csv_path: str,
    min_hits_threshold: float = 0.0,
    scan_stream: str = "records",
) -> None:
    """Write a CSV manifest with one row per matched page.

    Mirrors write_unmatched_manifest so ReviewDialog can consume either manifest
    with the same column set.  exclusion_reasons is always empty for matched pages.

    Columns: source_pdf_path (absolute), page_num (1-indexed), categories_matched,
    keywords_hit, extraction_method, total_hits, min_hits_threshold,
    exclusion_reasons, dates_on_page.
    """
    out = Path(output_csv_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    with open(out, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "source_pdf_path",
                "page_num",
                "categories_matched",
                "keywords_hit",
                "extraction_method",
                "total_hits",
                "min_hits_threshold",
                "exclusion_reasons",
                "dates_on_page",
                "service_date",
                "raw_service_date_str",
                "provider_npi",
                "provider_name_hint",
                "provider_name_context",
                "scan_stream",
            ]
        )
        for m in sorted(matches, key=lambda x: x.page_num):
            writer.writerow(
                [
                    pdf_path_absolute,
                    m.page_num + 1,
                    "|".join(m.categories),
                    "|".join(m.keywords_hit),
                    m.extraction_method,
                    m.total_hits,
                    min_hits_threshold,
                    "",  # no exclusion_reasons for matched pages
                    "|".join(m.dates_on_page),
                    m.service_date or "",
                    m.raw_service_date_str or "",
                    m.provider_npi or "",
                    m.provider_name_hint or "",
                    m.provider_name_context or "",
                    scan_stream,
                ]
            )

    log.info("Matched manifest written to %s.", out.name)


def write_unmatched_manifest(
    pdf_path_absolute: str,
    exclusions: list[PageExclusion],
    output_csv_path: str,
    scan_stream: str = "unmatched",
) -> None:
    """
    Write a CSV manifest with one row per scored-but-excluded page.

    Stores the absolute source PDF path (required by apply_feedback() to re-open
    source files). Multi-value fields are pipe-separated.

    Columns: source_pdf_path (absolute), page_num (1-indexed), categories_matched,
    keywords_hit, extraction_method, total_hits, min_hits_threshold,
    exclusion_reasons, dates_on_page.
    """
    out = Path(output_csv_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    with open(out, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "source_pdf_path",
                "page_num",
                "categories_matched",
                "keywords_hit",
                "extraction_method",
                "total_hits",
                "min_hits_threshold",
                "exclusion_reasons",
                "dates_on_page",
                "service_date",
                "raw_service_date_str",
                "provider_npi",
                "provider_name_hint",
                "provider_name_context",
                "scan_stream",
            ]
        )
        for exc in sorted(exclusions, key=lambda x: x.page_num):
            writer.writerow(
                [
                    pdf_path_absolute,
                    exc.page_num + 1,
                    "|".join(exc.categories),
                    "|".join(exc.keywords_hit),
                    exc.extraction_method,
                    exc.total_hits,
                    exc.min_hits_threshold,
                    "|".join(exc.exclusion_reasons),
                    "|".join(exc.dates_on_page),
                    exc.service_date or "",
                    exc.raw_service_date_str or "",
                    exc.provider_npi or "",
                    exc.provider_name_hint or "",
                    exc.provider_name_context or "",
                    scan_stream,
                ]
            )

    log.info("Unmatched manifest written to %s.", out.name)


# CONSOLIDATION HELPERS

_SEPARATOR_FONT_CANDIDATES = (
    "C:/Windows/Fonts/calibri.ttf",
    "C:/Windows/Fonts/arial.ttf",
    "C:/Windows/Fonts/verdana.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "arial.ttf",
    "calibri.ttf",
)


def _make_separator_page(source_filename: str) -> io.BytesIO:
    """
    Return a BytesIO containing a single-page PDF (US Letter) with source_filename centered on a white background.

    Uses Pillow to render the text. At 72 DPI a 612×792 pixel image maps 1:1 to US Letter points.
    """
    W, H = 612, 792
    img = Image.new("RGB", (W, H), color=(255, 255, 255))
    draw = ImageDraw.Draw(img)

    font: ImageFont.FreeTypeFont | ImageFont.ImageFont | None = None
    for path in _SEPARATOR_FONT_CANDIDATES:
        try:
            font = ImageFont.truetype(path, size=22)
            break
        except OSError:
            pass
    if font is None:
        font = ImageFont.load_default(size=22)

    bbox = draw.textbbox((0, 0), source_filename, font=font)
    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
    x, y = (W - tw) / 2, (H - th) / 2

    # Horizontal rules flanking the label
    rule_y_top = int(y) - 16
    rule_y_bot = int(y) + th + 16
    margin = 60
    draw.line(
        [(margin, rule_y_top), (W - margin, rule_y_top)], fill=(180, 180, 180), width=1
    )
    draw.line(
        [(margin, rule_y_bot), (W - margin, rule_y_bot)], fill=(180, 180, 180), width=1
    )

    draw.text((x, y), source_filename, fill=(0, 0, 0), font=font)

    buf = io.BytesIO()
    img.save(buf, format="PDF", resolution=72)
    buf.seek(0)
    return buf


def _ocr_consolidated(
    dest: Path,
    progress_callback: Callable[[str], None] | None = None,
) -> None:
    """
    Add a searchable text layer to dest in-place using ocrmypdf.

    Pages that already carry native text are skipped (skip_text=True) so
    LiteParse-extracted pages are left untouched; only image-only pages
    (scanned originals and separator pages) get an OCR overlay.

    Writes to a sibling .ocr.pdf temp file then atomically replaces dest so
    a failure never corrupts the original consolidated PDF.
    """
    try:
        import ocrmypdf
    except ImportError:
        log.warning(
            "ocrmypdf is not installed — skipping OCR of consolidated PDF. "
            "Add ocrmypdf to your dependencies to enable this step."
        )
        return

    if progress_callback:
        progress_callback("Running OCR on consolidated PDF…")

    # ocrmypdf discovers Tesseract via PATH; patch it in if we have a resolved path.
    old_path: str | None = None
    if _TESSERACT_EXE:
        tess_dir = str(Path(_TESSERACT_EXE).parent)
        current = os.environ.get("PATH", "")
        if tess_dir not in current.split(os.pathsep):
            old_path = current
            os.environ["PATH"] = tess_dir + os.pathsep + current

    tmp = dest.with_suffix(".ocr.pdf")
    try:
        ocrmypdf.ocr(
            dest,
            tmp,
            skip_text=True,
            language="eng",
            progress_bar=False,
        )
        tmp.replace(dest)
        log.info("OCR text layer added to %s.", dest.name)
    except Exception:
        log.warning("OCR of consolidated PDF failed — original kept.", exc_info=True)
        if tmp.exists():
            tmp.unlink()
    finally:
        if old_path is not None:
            os.environ["PATH"] = old_path


def _consolidate_stream(
    output_dir: str,
    pdf_stems: list[str],
    content_suffix: str,
    manifest_suffix: str,
    out_pdf_name: str,
    out_manifest_name: str,
    progress_label: str,
    run_ocr: bool = False,
    out_subdir: str = "",
    progress_callback: Callable[[str], None] | None = None,
) -> Path | None:
    """Shared consolidation logic for a single scan stream (records or bills).

    Reads ``{stem}{content_suffix}`` and ``{stem}{manifest_suffix}`` for each
    stem, builds a combined PDF with separator pages, annotates each manifest
    row with ``consolidated_page_num``, writes ``out_pdf_name`` and
    ``out_manifest_name`` (into ``out_subdir`` if given), then deletes the
    per-stem source files.  Returns the output PDF path or None if nothing was
    written.
    """
    out = Path(output_dir)
    dest_dir = (out / out_subdir) if out_subdir else out
    dest_dir.mkdir(exist_ok=True)

    writer = pypdf.PdfWriter()
    all_manifest_rows: list[dict] = []
    included = 0
    current_page = 0

    for stem in pdf_stems:
        content_pdf = out / f"{stem}{content_suffix}"
        content_csv = out / f"{stem}{manifest_suffix}"
        if not content_pdf.exists():
            continue
        if progress_callback:
            progress_callback(f"Consolidating {progress_label}: {stem}…")

        sep_buf = _make_separator_page(f"{stem}.pdf")
        sep_reader = pypdf.PdfReader(sep_buf)
        writer.add_page(sep_reader.pages[0])
        writer.add_outline_item(f"{stem}.pdf", current_page)
        current_page += 1

        reader = pypdf.PdfReader(str(content_pdf))
        page_count_for_stem = len(reader.pages)
        first_content_page = current_page
        for page in reader.pages:
            writer.add_page(page)
            current_page += 1

        if content_csv.exists():
            with open(content_csv, newline="", encoding="utf-8") as f:
                rows = list(csv.DictReader(f))
            for idx, row in enumerate(rows):
                if idx < page_count_for_stem:
                    row["consolidated_page_num"] = first_content_page + idx + 1
                    all_manifest_rows.append(row)

        included += 1

    if not included:
        return None

    dest = dest_dir / out_pdf_name
    with open(dest, "wb") as f:
        writer.write(f)
    log.info(
        "Consolidated %s PDF written to %s (%d source(s)).",
        progress_label, dest.name, included,
    )

    if all_manifest_rows:
        csv_dest = dest_dir / out_manifest_name
        fieldnames = list(all_manifest_rows[0].keys())
        with open(csv_dest, "w", newline="", encoding="utf-8") as f:
            csv_writer = csv.DictWriter(f, fieldnames=fieldnames)
            csv_writer.writeheader()
            csv_writer.writerows(all_manifest_rows)

    for stem in pdf_stems:
        for suffix in (content_suffix, manifest_suffix):
            p = out / f"{stem}{suffix}"
            if p.exists():
                p.unlink()
                log.debug("Removed individual %s file: %s", progress_label, p.name)

    if run_ocr:
        _ocr_consolidated(dest, progress_callback)

    return dest


def consolidate_records(
    output_dir: str,
    pdf_stems: list[str],
    progress_callback: Callable[[str], None] | None = None,
) -> Path | None:
    """Consolidate per-stem records PDFs into _records/_consolidated_records.pdf."""
    return _consolidate_stream(
        output_dir, pdf_stems,
        content_suffix="_records.pdf",
        manifest_suffix="_records_manifest.csv",
        out_pdf_name="_consolidated_records.pdf",
        out_manifest_name="_consolidated_records_manifest.csv",
        progress_label="records",
        run_ocr=True,
        out_subdir="_records",
        progress_callback=progress_callback,
    )


def consolidate_bills(
    output_dir: str,
    pdf_stems: list[str],
    progress_callback: Callable[[str], None] | None = None,
) -> Path | None:
    """Consolidate per-stem bills PDFs into _bills/_consolidated_bills.pdf."""
    return _consolidate_stream(
        output_dir, pdf_stems,
        content_suffix="_bills.pdf",
        manifest_suffix="_bills_manifest.csv",
        out_pdf_name="_consolidated_bills.pdf",
        out_manifest_name="_consolidated_bills_manifest.csv",
        progress_label="bills",
        run_ocr=False,
        out_subdir="_bills",
        progress_callback=progress_callback,
    )


def consolidate_to_pdf(
    output_dir: str,
    pdf_stems: list[str],
    progress_callback: Callable[[str], None] | None = None,
) -> Path | None:
    """
    Combine every {stem}_matched.pdf in output_dir into a single _consolidated.pdf,
    inserting a labeled separator page before each source document's pages.

    Also merges every {stem}_matched_manifest.csv into _consolidated_matched_manifest.csv,
    adding a consolidated_page_num column (1-indexed position within the consolidated PDF)
    so ReviewDialog can navigate QPdfView directly.

    Returns the path of the consolidated PDF, or None if nothing was written.
    """
    out = Path(output_dir)
    writer = pypdf.PdfWriter()
    all_manifest_rows: list[dict] = []
    included = 0
    current_page = 0

    for stem in pdf_stems:
        matched = out / f"{stem}_matched.pdf"
        matched_csv = out / f"{stem}_matched_manifest.csv"
        if not matched.exists():
            continue
        if progress_callback:
            progress_callback(f"Consolidating: {stem}…")

        # Separator page — bookmark points here
        sep_buf = _make_separator_page(f"{stem}.pdf")
        sep_reader = pypdf.PdfReader(sep_buf)
        writer.add_page(sep_reader.pages[0])
        writer.add_outline_item(f"{stem}.pdf", current_page)
        current_page += 1

        # Matched pages
        reader = pypdf.PdfReader(str(matched))
        page_count_for_stem = len(reader.pages)
        first_content_page = current_page
        for page in reader.pages:
            writer.add_page(page)
            current_page += 1

        # Annotate manifest rows with consolidated_page_num
        if matched_csv.exists():
            with open(matched_csv, newline="", encoding="utf-8") as f:
                rows = list(csv.DictReader(f))
            for idx, row in enumerate(rows):
                if idx < page_count_for_stem:
                    row["consolidated_page_num"] = first_content_page + idx + 1
                    all_manifest_rows.append(row)

        included += 1

    if not included:
        return None

    dest = out / "_consolidated.pdf"
    with open(dest, "wb") as f:
        writer.write(f)
    log.info("Consolidated PDF written to %s (%d source(s)).", dest.name, included)

    # Write consolidated matched manifest
    if all_manifest_rows:
        csv_dest = out / "_consolidated_matched_manifest.csv"
        fieldnames = list(all_manifest_rows[0].keys())
        with open(csv_dest, "w", newline="", encoding="utf-8") as f:
            csv_writer = csv.DictWriter(f, fieldnames=fieldnames)
            csv_writer.writeheader()
            csv_writer.writerows(all_manifest_rows)
        log.info(
            "Consolidated matched manifest written to %s (%d row(s)).",
            csv_dest.name,
            len(all_manifest_rows),
        )

    for stem in pdf_stems:
        for suffix in ("_matched.pdf", "_matched_manifest.csv"):
            f = out / f"{stem}{suffix}"
            if f.exists():
                f.unlink()
                log.debug("Removed individual matched file: %s", f.name)

    return dest


def consolidate_manifests(
    output_dir: str,
    pdf_stems: list[str],
) -> Path | None:
    """
    Merge every {stem}_manifest.csv in output_dir into a single _consolidated_manifest.csv, prepending a source_pdf column so each
    row is traceable back to its origin file.

    Returns: Path to the consolidated CSV, or None if nothing was written.
    """
    out = Path(output_dir)
    all_rows: list[dict] = []

    for stem in pdf_stems:
        csv_path = out / f"{stem}_manifest.csv"
        if not csv_path.exists():
            continue
        with open(csv_path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                all_rows.append({"source_pdf": f"{stem}.pdf", **row})

    if not all_rows:
        return None

    records_dir = out / "_records"
    records_dir.mkdir(exist_ok=True)
    dest = records_dir / "_consolidated_manifest.csv"
    fieldnames = list(all_rows[0].keys())
    with open(dest, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_rows)

    log.info(
        "Consolidated manifest written to %s (%d row(s)).", dest.name, len(all_rows)
    )

    for stem in pdf_stems:
        individual = out / f"{stem}_manifest.csv"
        if individual.exists():
            individual.unlink()
            log.debug("Removed individual manifest: %s", individual.name)

    return dest


def consolidate_dates(
    output_dir: str,
    pdf_stems: list[str],
) -> Path | None:
    """
    Merge every {stem}_dates.csv into _consolidated_dates.csv,
    prepending a source_pdf column so each row is traceable to its origin file.

    Returns: Path to the consolidated CSV, or None if nothing was written.
    """
    out = Path(output_dir)
    all_rows: list[dict] = []

    for stem in pdf_stems:
        csv_path = out / f"{stem}_dates.csv"
        if not csv_path.exists():
            continue
        with open(csv_path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                all_rows.append({"source_pdf": f"{stem}.pdf", **row})

    if not all_rows:
        return None

    records_dir = out / "_records"
    records_dir.mkdir(exist_ok=True)
    dest = records_dir / "_consolidated_dates.csv"
    fieldnames = list(all_rows[0].keys())
    with open(dest, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_rows)

    log.info(
        "Consolidated dates CSV written to %s (%d row(s)).", dest.name, len(all_rows)
    )

    for stem in pdf_stems:
        individual = out / f"{stem}_dates.csv"
        if individual.exists():
            individual.unlink()
            log.debug("Removed individual dates CSV: %s", individual.name)

    return dest


def consolidate_unmatched(
    output_dir: str,
    pdf_stems: list[str],
    progress_callback: Callable[[str], None] | None = None,
) -> tuple[Path | None, Path | None]:
    """
    Combine every {stem}_unmatched.pdf into _consolidated_unmatched.pdf, inserting
    a labeled separator page before each source document's pages.

    Also merges every {stem}_unmatched_manifest.csv into
    _consolidated_unmatched_manifest.csv, adding a consolidated_page_num column
    (1-indexed position within the consolidated PDF) so the ReviewDialog can
    navigate QPdfView directly without parsing filenames.

    Individual per-stem files are deleted after consolidation.

    Returns: (Path to consolidated PDF or None, Path to consolidated manifest or None)
    """
    out = Path(output_dir)
    writer = pypdf.PdfWriter()
    all_manifest_rows: list[dict] = []
    included = 0
    current_page = 0  # 0-indexed running counter (separator + content pages)

    for stem in pdf_stems:
        unmatched_pdf = out / f"{stem}_unmatched.pdf"
        unmatched_csv = out / f"{stem}_unmatched_manifest.csv"
        if not unmatched_pdf.exists():
            continue
        if progress_callback:
            progress_callback(f"Consolidating unmatched: {stem}…")

        # Separator page — bookmark points here
        sep_buf = _make_separator_page(f"{stem}.pdf")
        sep_reader = pypdf.PdfReader(sep_buf)
        writer.add_page(sep_reader.pages[0])
        writer.add_outline_item(f"{stem}.pdf", current_page)
        current_page += 1  # separator occupies one page

        # Unmatched content pages
        reader = pypdf.PdfReader(str(unmatched_pdf))
        page_count_for_stem = len(reader.pages)
        first_content_page = current_page  # 0-indexed position of first content page
        for page in reader.pages:
            writer.add_page(page)
            current_page += 1

        # Read the per-stem manifest and annotate with consolidated_page_num.
        # The nth row in the CSV corresponds to the nth content page for this stem
        # (both are ordered by page_num ascending from write_unmatched_manifest).
        if unmatched_csv.exists():
            with open(unmatched_csv, newline="", encoding="utf-8") as f:
                rows = list(csv.DictReader(f))
            for idx, row in enumerate(rows):
                if idx < page_count_for_stem:
                    # consolidated_page_num is 1-indexed
                    row["consolidated_page_num"] = first_content_page + idx + 1
                    all_manifest_rows.append(row)

        included += 1

    if not included:
        return None, None

    unmatched_dir = out / "_unmatched"
    unmatched_dir.mkdir(exist_ok=True)

    # Write consolidated PDF
    pdf_dest = unmatched_dir / "_consolidated_unmatched.pdf"
    with open(pdf_dest, "wb") as f:
        writer.write(f)
    log.info(
        "Consolidated unmatched PDF written to %s (%d source(s)).",
        pdf_dest.name,
        included,
    )

    # Write consolidated manifest
    csv_dest: Path | None = None
    if all_manifest_rows:
        csv_dest = unmatched_dir / "_consolidated_unmatched_manifest.csv"
        fieldnames = list(all_manifest_rows[0].keys())
        with open(csv_dest, "w", newline="", encoding="utf-8") as f:
            csv_writer = csv.DictWriter(f, fieldnames=fieldnames)
            csv_writer.writeheader()
            csv_writer.writerows(all_manifest_rows)
        log.info(
            "Consolidated unmatched manifest written to %s (%d row(s)).",
            csv_dest.name,
            len(all_manifest_rows),
        )

    # Remove individual files
    for stem in pdf_stems:
        for suffix in ("_unmatched.pdf", "_unmatched_manifest.csv"):
            f = out / f"{stem}{suffix}"
            if f.exists():
                f.unlink()
                log.debug("Removed individual unmatched file: %s", f.name)

    return pdf_dest, csv_dest


def consolidate_all_scored(
    output_dir: str,
    records_stems: list[str],
    bills_stems: list[str],
    unmatched_stems: list[str],
    progress_callback: Callable[[str], None] | None = None,
) -> tuple["Path | None", "Path | None"]:
    """Combine ALL scored pages (records + bills + excluded) into a single review PDF.

    Writes _consolidated_review.pdf and _consolidated_review_manifest.csv.
    Records stems are listed first, then bills, then unmatched. Each stem gets a
    separator page. Each manifest row gains ``scanner_decision`` ("matched" or
    "excluded"), ``scan_stream`` ("records", "bills", or "unmatched"), and
    ``consolidated_page_num``.

    Per-stem files are NOT deleted — they are consumed by the caller's separate
    consolidate_records / consolidate_bills / consolidate_unmatched calls.
    """
    out = Path(output_dir)
    writer = pypdf.PdfWriter()
    all_rows: list[dict] = []
    included = 0
    current_page = 0

    def _add_stem(
        stem: str,
        content_suffix: str,
        manifest_suffix: str,
        decision: str,
        stream: str,
    ) -> None:
        nonlocal current_page, included
        content_pdf = out / f"{stem}{content_suffix}"
        manifest_csv = out / f"{stem}{manifest_suffix}"
        if not content_pdf.exists():
            return

        if progress_callback:
            progress_callback(f"Review consolidation: {stem} ({stream})…")

        sep_buf = _make_separator_page(f"{stem}.pdf [{stream}]")
        sep_reader = pypdf.PdfReader(sep_buf)
        writer.add_page(sep_reader.pages[0])
        writer.add_outline_item(f"{stem}.pdf", current_page)
        current_page += 1

        reader = pypdf.PdfReader(str(content_pdf))
        page_count = len(reader.pages)
        first_content_page = current_page
        for page in reader.pages:
            writer.add_page(page)
            current_page += 1

        if manifest_csv.exists():
            with open(manifest_csv, newline="", encoding="utf-8") as f:
                rows = list(csv.DictReader(f))
            for idx, row in enumerate(rows):
                if idx < page_count:
                    row["scanner_decision"] = decision
                    row["scan_stream"] = stream
                    row["consolidated_page_num"] = first_content_page + idx + 1
                    all_rows.append(row)

        included += 1

    for stem in records_stems:
        _add_stem(stem, "_records.pdf", "_records_manifest.csv", "matched", "records")
    for stem in bills_stems:
        _add_stem(stem, "_bills.pdf", "_bills_manifest.csv", "matched", "bills")
    for stem in unmatched_stems:
        _add_stem(stem, "_unmatched.pdf", "_unmatched_manifest.csv", "excluded", "unmatched")

    if not included:
        return None, None

    review_dir = out / "_review"
    review_dir.mkdir(exist_ok=True)

    pdf_dest = review_dir / "_consolidated_review.pdf"
    with open(pdf_dest, "wb") as f:
        writer.write(f)
    log.info(
        "Consolidated review PDF written to %s (%d source(s)).",
        pdf_dest.name,
        included,
    )

    csv_dest: Path | None = None
    if all_rows:
        csv_dest = review_dir / "_consolidated_review_manifest.csv"
        fieldnames = list(all_rows[0].keys())
        with open(csv_dest, "w", newline="", encoding="utf-8") as f:
            csv_writer = csv.DictWriter(f, fieldnames=fieldnames)
            csv_writer.writeheader()
            csv_writer.writerows(all_rows)
        log.info(
            "Consolidated review manifest written to %s (%d row(s)).",
            csv_dest.name,
            len(all_rows),
        )

    return pdf_dest, csv_dest


def write_page_texts_sidecar(
    out_path: "str | Path",
    page_texts: "dict[int, tuple[str, str]]",
    matches: "list[PageMatch]",
    exclusions: "list[PageExclusion]",
) -> None:
    """Write a JSONL sidecar mapping each page to its extracted text and raw fields.

    One JSON object per line, keyed by 1-indexed page_num.  Only pages with
    non-empty text are written.  Downstream consumers (ReviewDialog, LLM service)
    can load this file to retrieve page text without re-opening source PDFs.

    Schema per line:
        page_num            int  (1-indexed)
        text                str
        extraction_method   str  ("liteparse" or "liteparse_ocr")
        raw_service_date_str str | null
        provider_name_context str | null
    """
    import json

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    # Build a lookup from 0-indexed page_num → raw extraction fields
    raw_lookup: dict[int, tuple[str | None, str | None]] = {}
    for m in matches:
        raw_lookup[m.page_num] = (m.raw_service_date_str, m.provider_name_context)
    for e in exclusions:
        if e.page_num not in raw_lookup:
            raw_lookup[e.page_num] = (e.raw_service_date_str, e.provider_name_context)

    with open(out, "w", encoding="utf-8") as f:
        for page_0idx in sorted(page_texts):
            text, method = page_texts[page_0idx]
            if not text.strip():
                continue
            raw_date, name_ctx = raw_lookup.get(page_0idx, (None, None))
            record = {
                "page_num": page_0idx + 1,
                "text": text,
                "extraction_method": method,
                "raw_service_date_str": raw_date,
                "provider_name_context": name_ctx,
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    log.info("Page-texts sidecar written to %s.", out.name)
