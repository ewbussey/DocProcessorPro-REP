#!/usr/bin/env python3
"""
ocr_pdfs.py — Standalone OCR batch utility (temporary workflow fix).

For every PDF in INPUT_DIR, detects image-only pages (native text below the
same threshold used by the scanner), rasterises them with Poppler, runs
Tesseract, and writes a searchable PDF to OUTPUT_DIR, preserving the input
directory structure.

Pages that already have sufficient native text are copied from the source
unchanged; only image-heavy pages are replaced with Tesseract's PDF output.

Usage
-----
    python ocr_pdfs.py <input_dir> <output_dir> [options]

Options
-------
    --dpi INT      Rasterisation DPI for image pages          (default: 450)
    --workers INT  Max parallel Tesseract threads per PDF     (default: 4)
    --force        OCR every page even if native text exists
"""

from __future__ import annotations

import argparse
import io
import re
import shutil
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from liteparse import LiteParse
import json

import pypdf
import pytesseract
from pdf2image import convert_from_path

# ---------------------------------------------------------------------------
# Reuse Tesseract / Poppler detection from the package.  Since this script is
# only run from an editable install the package is always importable.
# ---------------------------------------------------------------------------
try:
    from DocProcessorPro.dpp_scripts.keyword_scanner_scripts.extraction import (
        _MIN_NATIVE_CHARS,
        _POPPLER_BIN,
        _TESSERACT_EXE,
    )
except ImportError:
    print(
        "Warning: could not import from DocProcessorPro package — "
        "falling back to PATH-based Tesseract/Poppler detection.",
        file=sys.stderr,
    )
    _MIN_NATIVE_CHARS = 100
    _POPPLER_BIN = None
    _TESSERACT_EXE = None

if _TESSERACT_EXE:
    pytesseract.pytesseract.tesseract_cmd = _TESSERACT_EXE


# ---------------------------------------------------------------------------
# Per-file OCR
# ---------------------------------------------------------------------------


def _is_likely_valid_pdf(path: Path) -> bool:
    """Cheap sanity check: non-empty and starts with the %PDF magic bytes.

    Catches cloud-sync placeholder files (e.g. ShareFile, OneDrive) that show
    up in a directory listing but haven't actually been downloaded yet, so
    we can report a clear reason instead of an opaque parser error.
    """
    try:
        if path.stat().st_size == 0:
            return False
        with open(path, "rb") as f:
            return f.read(5) == b"%PDF-"
    except OSError:
        return False


def _native_texts(pdf_path: Path) -> list[str]:
    """Extract native text from every page, preferring LiteParse over pdfplumber."""
    try:
        from liteparse import LiteParse  # type: ignore[import]

        result = LiteParse(ocr_enabled=False, quiet=True, output_format="text").parse(
            str(pdf_path)
        )
        return [p.text or "" for p in result.pages]
    except Exception:
        pass

    try:
        import pdfplumber  # type: ignore[import]

        with pdfplumber.open(str(pdf_path)) as pdf:
            return [p.extract_text() or "" for p in pdf.pages]
    except Exception:
        return []


def _ocr_pdf(
    src: Path,
    dest: Path,
    *,
    dpi: int = 300,
    max_workers: int = 4,
    force: bool = False,
) -> tuple[int, int]:
    """Write a searchable copy of *src* to *dest*.

    Returns (pages_native, pages_ocrd).
    """
    dest.parent.mkdir(parents=True, exist_ok=True)

    reader = pypdf.PdfReader(str(src))
    page_count = len(reader.pages)

    # Determine which pages need OCR
    if force:
        ocr_needed: set[int] = set(range(page_count))
    else:
        texts = _native_texts(src)
        # If text extraction failed entirely, OCR every page
        if not texts:
            ocr_needed = set(range(page_count))
        else:
            ocr_needed = {
                i for i, t in enumerate(texts) if len(t.strip()) < _MIN_NATIVE_CHARS
            }

    pages_native = page_count - len(ocr_needed)
    pages_ocrd = len(ocr_needed)

    if not ocr_needed:
        shutil.copy2(src, dest)
        return pages_native, 0

    # Rasterise and OCR each qualifying page in parallel
    def _ocr_one(page_0idx: int) -> tuple[int, bytes]:
        images = convert_from_path(
            str(src),
            poppler_path=_POPPLER_BIN,  # type: ignore[arg-type]
            first_page=page_0idx + 1,
            last_page=page_0idx + 1,
            dpi=dpi,
        )
        if not images:
            raise RuntimeError(f"pdf2image returned no image for page {page_0idx + 1}")
        return page_0idx, pytesseract.image_to_pdf_or_hocr(images[0], extension="pdf")

    ocr_results: dict[int, bytes] = {}
    with ThreadPoolExecutor(max_workers=min(max_workers, len(ocr_needed))) as pool:
        futures = {pool.submit(_ocr_one, i): i for i in sorted(ocr_needed)}
        for fut in as_completed(futures):
            page_0idx = futures[fut]
            try:
                idx, pdf_bytes = fut.result()
                ocr_results[idx] = pdf_bytes
            except Exception as exc:
                print(
                    f"    [warn] page {page_0idx + 1}: OCR failed — {exc}",
                    file=sys.stderr,
                )

    # Assemble output: native pages preserved, OCR'd pages substituted
    writer = pypdf.PdfWriter()
    for i in range(page_count):
        if i in ocr_results:
            ocr_page_reader = pypdf.PdfReader(io.BytesIO(ocr_results[i]))
            writer.add_page(ocr_page_reader.pages[0])
        else:
            writer.add_page(reader.pages[i])

    with open(dest, "wb") as f:
        writer.write(f)

    return pages_native, pages_ocrd


def _table_to_markdown(table: list[list[str | None]]) -> str:
    """Render a pdfplumber table (list of rows) as a Markdown pipe table."""
    rows = [
        [(cell or "").strip().replace("|", r"\|").replace("\n", " ") for cell in row]
        for row in table
    ]
    rows = [row for row in rows if any(cell for cell in row)]
    if not rows:
        return ""

    width = max(len(row) for row in rows)
    rows = [row + [""] * (width - len(row)) for row in rows]
    header, *body = rows

    lines = [
        "| " + " | ".join(header) + " |",
        "| " + " | ".join(["---"] * width) + " |",
    ]
    lines += ["| " + " | ".join(row) + " |" for row in body]
    return "\n".join(lines)


def _pdfplumber_page_tables_markdown(pdf_path: Path) -> dict[int, str]:
    """Extract tables via pdfplumber, rendered as Markdown, per page.

    Assumes pdf_path already has a text layer on every page (native or
    pre-OCR'd) — pdfplumber only reads a PDF's text/vector layer, it never
    OCRs anything itself.

    Returns a mapping of 1-indexed page number -> markdown table block, for
    every page where at least one table was found.
    """
    import pdfplumber  # type: ignore[import]

    tables_by_page: dict[int, str] = {}

    with pdfplumber.open(str(pdf_path)) as pdf:
        for i, page in enumerate(pdf.pages):
            tables = page.extract_tables()
            blocks = [md for t in tables if t and (md := _table_to_markdown(t))]
            if blocks:
                tables_by_page[i + 1] = "\n\n".join(blocks)

    return tables_by_page


def liteparse_pdf(
    pdf_path: Path,
    output_dir: Path,
    output_format: str = "markdown",
    *,
    dpi: int = 450,
    max_workers: int = 4,
    pdfplumber_tables: bool = True,
) -> Path:
    """Parse pdf_path with LiteParse and save the extracted text.

    Pages lacking native text are first OCR'd with the same Poppler +
    Tesseract pipeline _ocr_pdf uses (at *dpi*), rather than LiteParse's own
    built-in OCR. LiteParse then runs with ocr_enabled=False and only does
    layout/Markdown reconstruction on top of the resulting searchable PDF.

    Writes one file (named after the PDF, extension matching output_format,
    pages separated by "--- Page N ---" markers) into output_dir and returns
    its path. For markdown output, table regions are additionally detected with
    pdfplumber and appended as a dedicated "Extracted Tables" section, since
    pdfplumber's line/text based table detection is generally more reliable
    than LiteParse's layout-inferred markdown tables.
    """
    file_ext_map = {"text": "txt", "markdown": "md", "json": "json"}

    if output_format not in file_ext_map:
        raise ValueError(f"Unsupported output format: {output_format}")
    else:
        file_ext = file_ext_map[output_format]

    with tempfile.TemporaryDirectory() as tmp_dir:
        ocr_source = Path(tmp_dir) / f"{pdf_path.stem}.ocr.pdf"
        _ocr_pdf(pdf_path, ocr_source, dpi=dpi, max_workers=max_workers, force=False)

        result = LiteParse(
            ocr_enabled=False, quiet=True, output_format=output_format
        ).parse(str(ocr_source))

        output_dir.mkdir(parents=True, exist_ok=True)
        dest = output_dir / f"{pdf_path.stem}.{file_ext}"

        if output_format == "json":
            dest.write_text(json.dumps(result, indent=2), encoding="utf-8")
            return dest

        text = "\n\n".join(
            f"--- Page {i} ---\n\n{page.text or ''}"
            for i, page in enumerate(result.pages, 1)
        )

        if output_format == "markdown" and pdfplumber_tables:
            tables_by_page = _pdfplumber_page_tables_markdown(ocr_source)
            if tables_by_page:
                table_section = "\n\n".join(
                    f"### Table(s) — page {page_num}\n\n{md}"
                    for page_num, md in sorted(tables_by_page.items())
                )
                text = f"{text}\n\n## Extracted Tables (pdfplumber)\n\n{table_section}"

        dest.write_text(text, encoding="utf-8")

    return dest





# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    ap = argparse.ArgumentParser(
        description=(
            "Batch-OCR a directory of PDFs, writing searchable copies "
            "to an output directory."
        )
    )

    ap.add_argument(
        "input_dir", 
        type=Path, 
        help="Directory containing source PDFs"
    )
    ap.add_argument(
        "output_dir", 
        type=Path, 
        help="Directory for searchable output PDFs"
    )
    ap.add_argument(
        "--ocr_pdfs",
        action="store_true",
        default=False,
        help="OCR PDFs",
    )
    ap.add_argument(
        "--dpi",
        type=int,
        default=450,
        metavar="INT",
        help="Rasterisation DPI for image-only pages (default: 450)",
    )
    ap.add_argument(
        "--workers",
        type=int,
        default=4,
        metavar="INT",
        help="Max parallel Tesseract threads per PDF (default: 4)",
    )
    ap.add_argument(
        "--force",
        action="store_true",
        help="OCR every page even if it already has native text",
    )
    ap.add_argument(
        "--parse_pdfs",
        action="store_true",
        default=False,
        help="Parse PDFs",
    )
    ap.add_argument(
        "--parse_format",
        type=str,
        default="markdown",
        metavar="STR",
        help="Output format for parsed PDFs (default: markdown)",
    )
    args = ap.parse_args()

    in_dir: Path = args.input_dir.resolve()
    out_dir: Path = args.output_dir.resolve()

    if not in_dir.is_dir():
        sys.exit(f"error: '{in_dir}' is not a directory.")

    if not args.ocr_pdfs and not args.parse_pdfs:
        sys.exit("error: either --ocr_pdfs or --parse_pdfs must be specified.")

    pdfs = sorted(in_dir.rglob("*.pdf"))
    if not pdfs:
        sys.exit(f"No PDFs found under '{in_dir}'.")

    print(f"Input:   {in_dir}")
    print(f"Output:  {out_dir}")
    print(f"PDFs:    {len(pdfs)}")
    if args.ocr_pdfs:
        print("OCR PDFs")
        print(f"DPI:     {args.dpi}   Workers: {args.workers}   Force: {args.force}")
    if args.parse_pdfs:
        print("Parse PDFs")
        print(f"Parse Format: {args.parse_format}")
    print()

    total_native = total_ocrd = copied = failed = 0
    total_parsed = 0

    if args.ocr_pdfs:
        for n, pdf in enumerate(pdfs, 1):
            rel = pdf.relative_to(in_dir)
            dest = out_dir / rel
            print(f"[{n}/{len(pdfs)}] {rel}", end="  ", flush=True)
            if not _is_likely_valid_pdf(pdf):
                print(
                    "SKIPPED — file is empty or not a valid PDF "
                    "(check it's fully downloaded/synced)",
                    file=sys.stderr,
                )
                failed += 1
                continue
            try:
                native, ocrd = _ocr_pdf(
                    pdf,
                    dest,
                    dpi=args.dpi,
                    max_workers=args.workers,
                    force=args.force,
                )
                total_native += native
                total_ocrd += ocrd
                if ocrd == 0:
                    print("copied (all pages had native text)")
                    copied += 1
                else:
                    print(f"{native} native  +  {ocrd} OCR'd")
            except Exception as exc:
                print(f"OCR FAILED — {exc}", file=sys.stderr)
                failed += 1

    if args.parse_pdfs:
        for n, pdf in enumerate(pdfs, 1):
            rel = pdf.relative_to(in_dir)
            dest = out_dir / rel
            print(f"[{n}/{len(pdfs)}] {rel}", end="  ", flush=True)
            if not _is_likely_valid_pdf(pdf):
                print(
                    "SKIPPED — file is empty or not a valid PDF "
                    "(check it's fully downloaded/synced)",
                    file=sys.stderr,
                )
                failed += 1
                continue
            try:
                parsed = liteparse_pdf(
                    pdf,
                    dest.parent,
                    output_format=args.parse_format,
                    dpi=args.dpi,
                    max_workers=args.workers,
                )
                total_parsed += 1
                print(f"parsed -> {parsed.name}")
            except Exception as exc:
                print(f"PARSE FAILED — {exc}", file=sys.stderr)
                failed += 1

    print()
    print("=" * 50)
    print(f"Processed : {len(pdfs) - failed}/{len(pdfs)} file(s)")
    if args.ocr_pdfs:
        print(f"Native    : {total_native} page(s)")
        print(f"OCR'd     : {total_ocrd} page(s)")
        if copied:
            print(f"Copied    : {copied} file(s) (no OCR needed)")
    if args.parse_pdfs:
        print(f"Parsed    : {total_parsed} file(s)")
    if failed:
        print(f"Failed    : {failed} file(s)  ← check stderr above")


if __name__ == "__main__":
    main()
