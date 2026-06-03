"""
Pure keyword/regex PDF page scanner.

scan_pdf(pdf_path, categories, min_hits) -> list[PageMatch]
scan_directory(input_dir, output_dir, categories, min_hits) -> dict[str, list[PageMatch]]
"""

from __future__ import annotations

import csv
import io
import logging
import os
import re
import shutil
import sys
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

import pdfplumber
import pypdf
import pytesseract
import regex
from pdf2image import convert_from_path
from PIL import Image, ImageDraw, ImageFont


log = logging.getLogger(__name__)

_MIN_NATIVE_CHARS = 100


# External binary detection (Tesseract + Poppler)


def _find_tesseract() -> str | None:
    """
    Locate the Tesseract executable. Resolution order:
    0. PyInstaller bundle  (sys._MEIPASS/tesseract/tesseract.exe)
    1. TESSERACT_PATH environment variable
    2. Windows registry  HKLM / HKCU  SOFTWARE\\Tesseract-OCR\\InstallDir
    3. Common installation directories
    4. tesseract on PATH (via shutil.which)
    Returns None if not found.
    """
    meipass = getattr(sys, "_MEIPASS", None)
    if meipass:
        candidate = Path(meipass) / "tesseract" / "tesseract.exe"
        if candidate.is_file():
            log.debug("Tesseract found in PyInstaller bundle: %s", candidate)
            return str(candidate)

    env = os.environ.get("TESSERACT_PATH", "")
    if env and Path(env).is_file():
        log.debug("Tesseract found via TESSERACT_PATH env var: %s", env)
        return env

    try:
        import winreg

        for hive in (winreg.HKEY_LOCAL_MACHINE, winreg.HKEY_CURRENT_USER):
            try:
                with winreg.OpenKey(hive, r"SOFTWARE\Tesseract-OCR") as key:
                    install_dir, _ = winreg.QueryValueEx(key, "InstallDir")
                    candidate = Path(install_dir) / "tesseract.exe"
                    if candidate.is_file():
                        log.debug("Tesseract found via registry: %s", candidate)
                        return str(candidate)
            except OSError:
                pass
    except ImportError:
        pass

    for candidate in (
        r"C:\Program Files\Tesseract-OCR\tesseract.exe",
        r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
    ):
        if Path(candidate).is_file():
            log.debug("Tesseract found at common path: %s", candidate)
            return candidate

    on_path = shutil.which("tesseract")
    if on_path:
        log.debug("Tesseract found on PATH: %s", on_path)
    return on_path


def _find_poppler() -> str | None:
    """
    Locate the Poppler bin directory (the folder containing pdftoppm).
    Resolution order:
    0. PyInstaller bundle  (sys._MEIPASS/poppler/)
    1. POPPLER_PATH environment variable (must point to the bin folder)
    2. Common installation directories (including versioned subdirectories)
    3. Returns None if pdftoppm is on PATH (pdf2image handles it)
       or if Poppler cannot be located (OCR will fail gracefully).
    """
    meipass = getattr(sys, "_MEIPASS", None)
    if meipass:
        candidate = Path(meipass) / "poppler"
        if (candidate / "pdftoppm.exe").is_file():
            log.debug("Poppler found in PyInstaller bundle: %s", candidate)
            return str(candidate)

    env = os.environ.get("POPPLER_PATH", "")
    if env and (Path(env) / "pdftoppm.exe").is_file():
        log.debug("Poppler found via POPPLER_PATH env var: %s", env)
        return env

    search_roots = (
        Path(r"C:\Program Files\poppler"),
        Path(r"C:\Program Files (x86)\poppler"),
        Path(r"C:\poppler"),
    )
    for root in search_roots:
        if not root.exists():
            continue
        # Flat layout:  root/bin/pdftoppm.exe  or  root/Library/bin/pdftoppm.exe
        for rel in ("bin", "Library/bin"):
            candidate = root / rel
            if (candidate / "pdftoppm.exe").is_file():
                log.debug("Poppler found at: %s", candidate)
                return str(candidate)
        # Versioned subdirectories (e.g. poppler-26.02.0/Library/bin) — pick newest
        subdirs = sorted((d for d in root.iterdir() if d.is_dir()), reverse=True)
        for sub in subdirs:
            for rel in ("bin", "Library/bin"):
                candidate = sub / rel
                if (candidate / "pdftoppm.exe").is_file():
                    log.debug("Poppler found at versioned path: %s", candidate)
                    return str(candidate)

    if shutil.which("pdftoppm"):
        log.debug("Poppler (pdftoppm) found on PATH — passing poppler_path=None.")
        return None  # pdf2image will use PATH

    log.warning(
        "Poppler not found. OCR fallback will be unavailable. "
        "Set the POPPLER_PATH environment variable to the Poppler bin directory."
    )
    return None


_TESSERACT_EXE: str | None = _find_tesseract()
_POPPLER_BIN: str | None = _find_poppler()

if _TESSERACT_EXE:
    pytesseract.pytesseract.tesseract_cmd = _TESSERACT_EXE
else:
    log.warning(
        "Tesseract not found. OCR fallback will be unavailable. "
        "Set the TESSERACT_PATH environment variable to the tesseract.exe path."
    )


# DATA STRUCTURES


@dataclass
class KeywordCategory:
    """
    A named category combining plain keywords and raw regex patterns.
    """

    name: str
    keywords: list[str]
    patterns: list[str] = field(default_factory=list)
    _compiled: list[regex.Pattern] = field(default_factory=list, init=False, repr=False)
    _terms: list[str] = field(default_factory=list, init=False, repr=False)

    _CPT_RE = re.compile(r"^[A-Z]?\d{4,5}$", re.IGNORECASE)

    def __post_init__(self) -> None:
        compiled: list[regex.Pattern] = []
        terms: list[str] = []
        for kw in self.keywords:
            if self._CPT_RE.fullmatch(kw):
                pat = r"(?<!\d)" + regex.escape(kw) + r"(?!\d)"
            else:
                pat = r"\b" + regex.escape(kw) + r"\b"
            compiled.append(regex.compile(pat, regex.IGNORECASE))
            terms.append(kw)
        for raw in self.patterns:
            compiled.append(regex.compile(raw, regex.IGNORECASE))
            terms.append(raw)
        self._compiled = compiled
        self._terms = terms

    def hits(self, text: str) -> list[str]:
        """Return list of keyword/pattern strings that match in text (deduped)."""
        matched: list[str] = []
        for pat, term in zip(self._compiled, self._terms):
            if pat.search(text):
                matched.append(term)
        return matched


@dataclass
class PageMatch:
    """A single PDF page that meets the min_hits threshold."""

    page_num: int  # 0-indexed internally; written 1-indexed to CSV
    categories: list[str]
    keywords_hit: list[str]
    extraction_method: str  # "pdfplumber" or "ocr"
    total_hits: int


# DEFAULT KEYWORD CATEGORIES

_THERAPY_PLAIN = [
    "therapeutic",
    "physical therapy",
    "occupational therapy",
    "vestibular",
    "chiropractic",
    "manipulation",
    "reeducation",
]
_THERAPY_CODES = [
    "97110",
    "97112",
    "97140",
    "97162",
    "97163",
    "97164",
    "97530",
    "G0283",
    "97014",
    "97012",
]

CATEGORY_THERAPY = KeywordCategory(
    name="THERAPY",
    keywords=_THERAPY_PLAIN
    + _THERAPY_CODES
    + [
        "PT",
        "OT",
        "speech therapy",
        "aquatic therapy",
        "dry needling",
        "massage therapy",
        "myofascial",
        "ultrasound therapy",
        "electrical stimulation",
        "TENS",
        "exercise therapy",
        "range of motion",
        "ROM",
        "strengthening",
        "stretching",
        "home exercise program",
        "HEP",
        "work hardening",
        "work conditioning",
        "functional capacity evaluation",
        "FCE",
        "rehabilitation",
        "rehab",
    ],
    patterns=[
        r"(?<!\d)971\d\d(?!\d)",
        r"(?<!\d)972\d\d(?!\d)",
        r"\bG02\d\d\b",
        r"\b(hot|cold)\s+pack\b",
        r"\belectrical\s+stim(?:ulation)?\b",
    ],
)

CATEGORY_MEDICAL_TREATMENT = KeywordCategory(
    name="MEDICAL_TREATMENT",
    keywords=[
        "diagnosis",
        "diagnoses",
        "prognosis",
        "treatment",
        "assessment",
        "evaluation",
        "examination",
        "prescription",
        "medication",
        "dosage",
        "referral",
        "follow-up",
        "follow up",
        "chief complaint",
        "history of present illness",
        "HPI",
        "review of systems",
        "ROS",
        "physical examination",
        "impression",
        "discharge summary",
        "discharge instructions",
        "inpatient",
        "outpatient",
        "emergency department",
        "urgent care",
        "clinical note",
        "progress note",
        "SOAP note",
        "subjective",
        "objective",
        "vital signs",
        "blood pressure",
        "heart rate",
        "temperature",
        "presenting complaint",
    ],
    patterns=[
        r"\b[A-Z]\d{2}(?:\.\d{1,4})?\b",
    ],
)

CATEGORY_BILLING = KeywordCategory(
    name="BILLING",
    keywords=[
        "date of service",
        "DOS",
        "date of injury",
        "DOI",
        "claim number",
        "claim no",
        "invoice",
        "statement",
        "account number",
        "insurance",
        "insurer",
        "payer",
        "policyholder",
        "member ID",
        "group number",
        "authorization",
        "pre-authorization",
        "referral number",
        "EOB",
        "explanation of benefits",
        "remittance",
        "ERA",
        "CMS-1500",
        "UB-04",
        "superbill",
        "charge",
        "billed amount",
        "allowed amount",
        "paid amount",
        "adjustment",
        "balance due",
        "copay",
        "coinsurance",
        "deductible",
        "units",
        "modifier",
        "place of service",
        "POS",
        "NPI",
        "TIN",
        "tax ID",
    ],
    patterns=[
        r"(?<!\d)[A-Z]\d{4}(?!\d)",
        r"(?<!\d)\d{5}(?!\d)",
        r"\b[A-Z]\d{2}(?:\.\d{1,4})?\b",
        r"\b\d{1,2}/\d{1,2}/\d{2,4}\b",
        r"\b\d{4}-\d{2}-\d{2}\b",
        r"\$\s?\d{1,3}(?:,\d{3})*(?:\.\d{2})?",
        r"\bNPI[:\s#]*\d{10}\b",
    ],
)

CATEGORY_INJURY_LEGAL = KeywordCategory(
    name="INJURY_LEGAL",
    keywords=[
        "accident",
        "motor vehicle accident",
        "MVA",
        "motor vehicle collision",
        "MVC",
        "workers compensation",
        "workers comp",
        "work comp",
        "slip and fall",
        "trip and fall",
        "personal injury",
        "liability",
        "negligence",
        "causation",
        "maximum medical improvement",
        "MMI",
        "independent medical examination",
        "IME",
        "impairment rating",
        "permanent impairment",
        "whole person impairment",
        "WPI",
        "permanent and stationary",
        "disability",
        "permanent disability",
        "temporary disability",
        "TTD",
        "modified duty",
        "light duty",
        "work restrictions",
        "subrogation",
        "lien",
        "medical lien",
        "deposition",
        "qualified medical evaluator",
        "QME",
        "agreed medical evaluator",
        "AME",
        "panel QME",
        "DWC",
        "WCAB",
        "claimant",
        "plaintiff",
        "defendant",
        "settlement",
        "demand letter",
        "independent medical evaluation",
    ],
    patterns=[
        r"\b\d{1,3}%\s+(?:whole\s+person\s+)?impairment\b",
        r"\b(?:claim|case|file)\s*(?:no|number|#)[:\s]*[\w\-]+\b",
    ],
)

CATEGORY_IMAGING = KeywordCategory(
    name="IMAGING",
    keywords=[
        "MRI",
        "magnetic resonance imaging",
        "CT scan",
        "computed tomography",
        "CAT scan",
        "X-ray",
        "radiograph",
        "radiography",
        "radiology",
        "radiologist",
        "radiology report",
        "imaging study",
        "ultrasound",
        "fluoroscopy",
        "bone scan",
        "DEXA",
        "myelogram",
        "discogram",
        "arthrogram",
        "EMG",
        "electromyography",
        "nerve conduction study",
        "NCS",
        "findings",
        "impression",
        "without contrast",
        "with contrast",
        "T1",
        "T2",
        "FLAIR",
        "axial",
        "sagittal",
        "coronal",
        "herniation",
        "disc herniation",
        "disc bulge",
        "stenosis",
        "spinal stenosis",
        "foraminal",
        "spondylosis",
        "fracture",
        "edema",
        "signal change",
        "hyperintense",
        "hypointense",
    ],
    patterns=[
        r"\b[CTLS]\d{1,2}[-–]\d{1,2}\b",
        r"\b[CTLS]\d{1,2}\b",
        r"\b(?:accession|study|exam)\s*(?:no|number|#)[:\s]*[\w\-]+\b",
    ],
)

CATEGORY_BEHAVIORAL_HEALTH = KeywordCategory(
    name="BEHAVIORAL_HEALTH",
    keywords=[
        "psychiatry",
        "psychiatrist",
        "psychology",
        "psychologist",
        "psychotherapy",
        "counseling",
        "counselor",
        "mental health",
        "behavioral health",
        "depression",
        "anxiety",
        "PTSD",
        "post-traumatic stress",
        "post traumatic stress",
        "panic disorder",
        "OCD",
        "obsessive-compulsive",
        "bipolar",
        "mood disorder",
        "adjustment disorder",
        "GAD",
        "generalized anxiety",
        "major depressive disorder",
        "MDD",
        "cognitive behavioral therapy",
        "CBT",
        "EMDR",
        "DBT",
        "dialectical behavioral",
        "medication management",
        "antidepressant",
        "SSRI",
        "SNRI",
        "antipsychotic",
        "anxiolytic",
        "benzodiazepine",
        "PHQ-9",
        "GAD-7",
        "psychiatric evaluation",
        "mental status examination",
        "MSE",
        "suicidal ideation",
        "homicidal ideation",
    ],
    patterns=[
        r"\b[FZ]\d{2}(?:\.\d{1,4})?\b",
        r"(?<!\d)90[0-9]{3}(?!\d)",
    ],
)

DEFAULT_CATEGORIES: list[KeywordCategory] = [
    CATEGORY_THERAPY,
    CATEGORY_MEDICAL_TREATMENT,
    CATEGORY_BILLING,
    CATEGORY_INJURY_LEGAL,
    CATEGORY_IMAGING,
    CATEGORY_BEHAVIORAL_HEALTH,
]


# CORE SCANNER


def scan_pdf(
    pdf_path: str,
    categories: list[KeywordCategory],
    min_hits: int = 1,
    progress_callback: Callable[[str], None] | None = None,
) -> list[PageMatch]:
    """
    Scan every page of a PDF for keyword matches.

    Opens pdfplumber once for the whole document, then does a second pass
    for any pages whose text content falls below _MIN_NATIVE_CHARS (OCR
    fallback via Tesseract). Keyword matching runs after both passes.

    Returns: PageMatch list sorted by page_num for pages where total_hits >= min_hits.
    """
    path = Path(pdf_path)
    if not path.exists():
        raise FileNotFoundError(pdf_path)

    # Pass 1: native extraction — single pdfplumber open for the whole file
    page_texts: list[tuple[str, str]] = []  # (text, method) indexed by page_num
    ocr_needed: list[int] = []

    with pdfplumber.open(pdf_path) as pdf:
        page_count = len(pdf.pages)
        log.info("Scanning %s (%d pages)...", path.name, page_count)
        for i, page in enumerate(pdf.pages):
            native = page.extract_text() or ""
            if len(native.strip()) >= _MIN_NATIVE_CHARS:
                page_texts.append((native, "pdfplumber"))
            else:
                page_texts.append((native, "pdfplumber"))  # placeholder
                ocr_needed.append(i)

    # Pass 2: OCR for pages with insufficient native text
    if ocr_needed:
        log.debug(
            "%s: %d page(s) below threshold — running OCR.", path.name, len(ocr_needed)
        )
        for idx, page_num in enumerate(ocr_needed, 1):
            if progress_callback:
                progress_callback(
                    f"{path.name}: OCR page {page_num + 1} ({idx}/{len(ocr_needed)})…"
                )
            try:
                images = convert_from_path(
                    pdf_path,
                    poppler_path=_POPPLER_BIN,  # type: ignore[arg-type]
                    first_page=page_num + 1,
                    last_page=page_num + 1,
                    dpi=300,
                )
                text = str(pytesseract.image_to_string(images[0]))
                page_texts[page_num] = (text, "ocr")
            except Exception:
                log.warning("Page %d: OCR failed, keeping native text.", page_num)

    # Pass 3: keyword matching
    matches: list[PageMatch] = []
    for i, (text, method) in enumerate(page_texts):
        all_keywords: list[str] = []
        matched_categories: list[str] = []
        for cat in categories:
            hits = cat.hits(text)
            if hits:
                matched_categories.append(cat.name)
                all_keywords.extend(hits)
        total = len(all_keywords)
        if total >= min_hits:
            matches.append(
                PageMatch(
                    page_num=i,
                    categories=matched_categories,
                    keywords_hit=all_keywords,
                    extraction_method=method,
                    total_hits=total,
                )
            )
            log.debug("Page %d matched: %s", i + 1, matched_categories)

    log.info(
        "%s: %d / %d pages matched (min_hits=%d).",
        path.name,
        len(matches),
        page_count,
        min_hits,
    )
    return matches


# OUTPUT WRITERS


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


def write_manifest(
    pdf_stem: str,
    matches: list[PageMatch],
    output_csv_path: str,
) -> None:
    """
    Write a CSV manifest with one row per matched page.

    Columns: page_num (1-indexed), categories_matched, keywords_hit, extraction_method, total_hits.

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
                ]
            )

    log.info("Manifest written to %s.", out.name)


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


def consolidate_to_pdf(
    output_dir: str,
    pdf_stems: list[str],
    progress_callback: Callable[[str], None] | None = None,
) -> Path | None:
    """
    Combine every {stem}_matched.pdf in output_dir into a single _consolidated.pdf,
    inserting a labeled separator page before each source document's pages.

    Returns the path of the consolidated PDF, or None if nothing was written.
    """
    out = Path(output_dir)
    writer = pypdf.PdfWriter()
    included = 0

    for stem in pdf_stems:
        matched = out / f"{stem}_matched.pdf"
        if not matched.exists():
            continue
        if progress_callback:
            progress_callback(f"Consolidating: {stem}…")

        # Separator page
        sep_buf = _make_separator_page(f"{stem}.pdf")
        sep_reader = pypdf.PdfReader(sep_buf)
        writer.add_page(sep_reader.pages[0])

        # Matched pages
        reader = pypdf.PdfReader(str(matched))
        for page in reader.pages:
            writer.add_page(page)

        included += 1

    if not included:
        return None

    dest = out / "_consolidated.pdf"
    with open(dest, "wb") as f:
        writer.write(f)
    log.info("Consolidated PDF written to %s (%d source(s)).", dest.name, included)

    for stem in pdf_stems:
        individual = out / f"{stem}_matched.pdf"
        if individual.exists():
            individual.unlink()
            log.debug("Removed individual matched PDF: %s", individual.name)

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

    dest = out / "_consolidated_manifest.csv"
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


# BATCH PROCESSOR


def scan_directory(
    input_dir: str,
    output_dir: str,
    categories: list[KeywordCategory],
    min_hits: int = 1,
    page_buffer: int = 0,
    progress_callback: Callable[[str], None] | None = None,
) -> dict[str, list[PageMatch]]:
    """
    Batch-process all .pdf files in input_dir.

    For each PDF with at least one match:
    - Writes {stem}_matched.pdf to output_dir
    - Writes {stem}_manifest.csv to output_dir

    PDFs with zero matches produce no output files.

    Args:
        progress_callback: Optional callable receiving a status string before
                        each file and on each OCR page (used by the GUI worker).

    Returns: {pdf_stem: list[PageMatch]} for all processed files.
    """
    in_dir = Path(input_dir)
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    pdfs = sorted(in_dir.rglob("*.pdf"))
    if not pdfs:
        log.warning("No PDF files found in %s (including subfolders).", in_dir)
        return {}

    log.info("Processing %d PDF(s) from %s (recursive).", len(pdfs), in_dir)
    results: dict[str, list[PageMatch]] = {}
    safe_stems: list[str] = []

    for idx, pdf_path in enumerate(pdfs, 1):
        # Build a collision-safe stem from the relative path so that
        # subfolder/billing/report.pdf and subfolder/records/report.pdf
        # produce distinct output file names.
        rel_parts = pdf_path.relative_to(in_dir).with_suffix("").parts
        safe_stem = "_".join(rel_parts)
        safe_stems.append(safe_stem)

        if progress_callback:
            progress_callback(f"Scanning {pdf_path.name} ({idx}/{len(pdfs)})…")
        try:
            matches = scan_pdf(
                str(pdf_path),
                categories,
                min_hits=min_hits,
                progress_callback=progress_callback,
            )
        except Exception:
            log.exception("Failed to scan %s — skipping.", pdf_path.name)
            results[safe_stem] = []
            continue

        results[safe_stem] = matches

        if not matches:
            log.info("%s: no matches.", pdf_path.name)
            continue

        extract_matched_pages(
            str(pdf_path),
            matches,
            str(out_dir / f"{safe_stem}_matched.pdf"),
            page_buffer=page_buffer,
        )
        write_manifest(
            safe_stem,
            matches,
            str(out_dir / f"{safe_stem}_manifest.csv"),
        )

    # Consolidate across all source PDFs that had matches, preserving scan order
    matched_stems = [s for s in safe_stems if results.get(s)]
    if matched_stems:
        consolidate_to_pdf(output_dir, matched_stems, progress_callback)
        consolidate_manifests(output_dir, matched_stems)

    return results
