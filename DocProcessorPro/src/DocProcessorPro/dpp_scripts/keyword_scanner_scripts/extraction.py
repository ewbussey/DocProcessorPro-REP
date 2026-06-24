from __future__ import annotations

import datetime
import logging
import os
import re
import shutil
import sys
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pdfplumber
import pytesseract
from pdf2image import convert_from_path

from .models import KeywordCategory, PageMatch, PageExclusion, ScanResult
from .categories import CLINICAL_ANCHOR_CATEGORIES, _BILLS_MIN_HITS, _BILLS_REQUIRE_CATEGORIES

log = logging.getLogger(__name__)

_MIN_NATIVE_CHARS = 100

# Date extraction helpers

# Dates immediately preceded by one of these labels are dates of birth and
# should be excluded from the dates manifest.  The lookbehind window is kept
# short (60 chars) so a label earlier on the same page cannot suppress an
# unrelated date.
_DOB_CONTEXT_RE = re.compile(
    r"\b(?:"
    r"date\s+of\s+birth"   # "date of birth"
    r"|birth\s*date"        # "birth date" / "birthdate"
    r"|dob"                 # DOB
    r"|d\.o\.b\.?"          # D.O.B. / D.O.B
    r"|born(?:\s+on)?"      # "born" / "born on"
    r")\b",
    re.IGNORECASE,
)
_DOB_LOOKBEHIND = 60  # characters before the date match to scan

# Dates of service: labels that, when followed within _DOS_LOOKAHEAD chars by a date,
# identify the highest-confidence service date on the page.
_DOS_CONTEXT_RE = re.compile(
    r"\b(?:date\s+of\s+service|d\.?o\.?s\.?|service\s+date|session\s+date"
    r"|visit\s+date|appointment\s+date|date\s*:)\b",
    re.IGNORECASE,
)
_DOS_LOOKAHEAD = 80  # characters after the DOS label to search for a date

# Provider identifier extraction
_NPI_EXTRACT_RE = re.compile(r"\bNPI[:\s#]*(\d{10})\b", re.IGNORECASE)
_PROVIDER_NAME_RE = re.compile(
    r"(?:Provider|Therapist|Clinician|Counselor|Psychologist|Psychiatrist"
    r"|Rendered\s+by|Treating\s+(?:provider|clinician)|Supervising\s+clinician)"
    r"[:\s]+([A-Z][a-z]+(?:[\s,]+[A-Z][a-z]*\.?){1,4})"
    r"(?:[,\s]+(?:LCSW|LMFT|LPC|MFT|PhD|PsyD|MD|DO|NP|PA|LPCC|LMHC))?",
    re.IGNORECASE,
)

_DATE_RE = re.compile(
    r"\b(\d{1,2})[/\-](\d{1,2})[/\-](\d{2,4})\b"
    r"|(\d{4})[/\-](\d{1,2})[/\-](\d{1,2})\b"
    r"|\b(Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?"
    r"|Jul(?:y)?|Aug(?:ust)?|Sep(?:t(?:ember)?)?|Oct(?:ober)?|Nov(?:ember)?"
    r"|Dec(?:ember)?)\s+(\d{1,2}),?\s+(\d{4})\b"
    r"|\b(\d{1,2})\s+(Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May"
    r"|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:t(?:ember)?)?|Oct(?:ober)?"
    r"|Nov(?:ember)?|Dec(?:ember)?)\s+(\d{4})\b",
    re.IGNORECASE,
)
_MONTH_MAP = {
    "jan": 1,
    "feb": 2,
    "mar": 3,
    "apr": 4,
    "may": 5,
    "jun": 6,
    "jul": 7,
    "aug": 8,
    "sep": 9,
    "oct": 10,
    "nov": 11,
    "dec": 12,
}


def _extract_dates(text: str) -> list[datetime.date]:
    """Return sorted unique dates found in text, filtered to 1900-2099.

    Dates that immediately follow a date-of-birth label within
    _DOB_LOOKBEHIND characters are suppressed so patient demographics
    do not contaminate the clinical-event dates manifest.
    """
    found: set[datetime.date] = set()
    for m in _DATE_RE.finditer(text):
        preceding = text[max(0, m.start() - _DOB_LOOKBEHIND) : m.start()]
        if _DOB_CONTEXT_RE.search(preceding):
            continue
        g = m.groups()
        try:
            if g[0] is not None:  # MM/DD/YYYY or M-D-YY
                mo, day, yr = int(g[0]), int(g[1]), int(g[2])
                if yr < 100:
                    yr += 2000 if yr < 50 else 1900
            elif g[3] is not None:  # YYYY-MM-DD
                yr, mo, day = int(g[3]), int(g[4]), int(g[5])
            elif g[6] is not None:  # Month DD, YYYY
                mo = _MONTH_MAP[g[6][:3].lower()]
                day, yr = int(g[7]), int(g[8])
            else:  # DD Month YYYY
                day, yr = int(g[9]), int(g[11])
                mo = _MONTH_MAP[g[10][:3].lower()]
            if 1900 <= yr <= 2099:
                found.add(datetime.date(yr, mo, day))
        except (ValueError, KeyError):
            pass
    return sorted(found)


def _extract_service_date(text: str) -> tuple[datetime.date | None, str | None]:
    """Return (service_date, raw_matched_str) from text.

    Searches for a date immediately following a DOS/session-date label within
    _DOS_LOOKAHEAD characters.  Falls back to the earliest non-DOB date on the
    page if no label is found.  Returns (None, None) if the page has no dates.

    raw_matched_str is the literal string matched by the date regex (e.g.
    "01/05/24", "Jan 5th, 2024") before ISO normalisation — preserved for
    LLM fine-tuning and corrections tracking.  On the fallback path it is None
    since there is no label-adjacent match to report.
    """
    for label_m in _DOS_CONTEXT_RE.finditer(text):
        window = text[label_m.end() : label_m.end() + _DOS_LOOKAHEAD]
        date_m = _DATE_RE.search(window)
        if date_m:
            raw_str = date_m.group(0)
            g = date_m.groups()
            try:
                if g[0] is not None:
                    mo, day, yr = int(g[0]), int(g[1]), int(g[2])
                    if yr < 100:
                        yr += 2000 if yr < 50 else 1900
                elif g[3] is not None:
                    yr, mo, day = int(g[3]), int(g[4]), int(g[5])
                elif g[6] is not None:
                    mo = _MONTH_MAP[g[6][:3].lower()]
                    day, yr = int(g[7]), int(g[8])
                else:
                    day, yr = int(g[9]), int(g[11])
                    mo = _MONTH_MAP[g[10][:3].lower()]
                if 1900 <= yr <= 2099:
                    return datetime.date(yr, mo, day), raw_str
            except (ValueError, KeyError):
                pass
    # Fallback: earliest non-DOB date on the page (no label context to report)
    all_dates = _extract_dates(text)
    return (all_dates[0], None) if all_dates else (None, None)


def _extract_provider_id(text: str) -> tuple[str | None, str | None, str | None]:
    """Return (npi, name_hint, name_context) extracted from text.

    npi is the first 10-digit NPI number found after an 'NPI' label.
    name_hint is the first provider name captured after a structured label
    (Provider:, Therapist:, Clinician:, etc.).  Either may be None.
    name_context is a ~150-char text window surrounding the name match —
    preserved for LLM fine-tuning.  None when no name match is found.
    """
    npi: str | None = None
    m = _NPI_EXTRACT_RE.search(text)
    if m:
        npi = m.group(1)

    name_hint: str | None = None
    name_context: str | None = None
    n = _PROVIDER_NAME_RE.search(text)
    if n:
        name_hint = n.group(1).strip()
        name_context = text[max(0, n.start() - 30) : n.end() + 100].strip()

    return npi, name_hint, name_context


# External binary detection (Tesseract + Poppler)


def _find_tesseract() -> str | None:
    """
    Locate the Tesseract executable. Resolution order:
    0. PyInstaller bundle  (sys._MEIPASS/tesseract/tesseract[.exe])
    1. TESSERACT_PATH environment variable
    2. Windows registry  HKLM / HKCU  SOFTWARE\\Tesseract-OCR\\InstallDir
    3. Common installation directories (Windows and macOS)
    4. tesseract on PATH (via shutil.which)
    Returns None if not found.
    """
    _win = sys.platform == "win32"
    _tess_bin = "tesseract.exe" if _win else "tesseract"

    meipass = getattr(sys, "_MEIPASS", None)
    if meipass:
        candidate = Path(meipass) / "tesseract" / _tess_bin
        if candidate.is_file():
            log.debug("Tesseract found in PyInstaller bundle: %s", candidate)
            return str(candidate)

    env = os.environ.get("TESSERACT_PATH", "")
    if env and Path(env).is_file():
        log.debug("Tesseract found via TESSERACT_PATH env var: %s", env)
        return env

    if _win:
        try:
            import winreg

            for hive in (winreg.HKEY_LOCAL_MACHINE, winreg.HKEY_CURRENT_USER):  # type: ignore[attr-defined]
                try:
                    with winreg.OpenKey(hive, r"SOFTWARE\Tesseract-OCR") as key:  # type: ignore[attr-defined]
                        install_dir, _ = winreg.QueryValueEx(key, "InstallDir")  # type: ignore[attr-defined]
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
            r"C:\Supporting_Libraries\Tesseract-OCR\tesseract.exe",
        ):
            if Path(candidate).is_file():
                log.debug("Tesseract found at common Windows path: %s", candidate)
                return candidate

    elif sys.platform == "darwin":
        for candidate in (
            "/opt/homebrew/bin/tesseract",   # Homebrew — Apple Silicon
            "/usr/local/bin/tesseract",       # Homebrew — Intel
            "/opt/local/bin/tesseract",       # MacPorts
        ):
            if Path(candidate).is_file():
                log.debug("Tesseract found at common macOS path: %s", candidate)
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
    2. Common installation directories (Windows and macOS)
    3. Returns None if pdftoppm is on PATH (pdf2image handles it)
       or if Poppler cannot be located (OCR will fail gracefully).
    """
    _win = sys.platform == "win32"
    _pdftoppm = "pdftoppm.exe" if _win else "pdftoppm"

    meipass = getattr(sys, "_MEIPASS", None)
    if meipass:
        candidate = Path(meipass) / "poppler"
        if (candidate / _pdftoppm).is_file():
            log.debug("Poppler found in PyInstaller bundle: %s", candidate)
            return str(candidate)

    env = os.environ.get("POPPLER_PATH", "")
    if env and (Path(env) / _pdftoppm).is_file():
        log.debug("Poppler found via POPPLER_PATH env var: %s", env)
        return env

    if _win:
        search_roots = (
            Path(r"C:\Program Files\poppler"),
            Path(r"C:\Program Files (x86)\poppler"),
            Path(r"C:\poppler"),
            Path(r"C:\Supporting_Libraries"),
        )
        for root in search_roots:
            if not root.exists():
                continue
            # Flat layout:  root/bin/pdftoppm.exe  or  root/Library/bin/pdftoppm.exe
            for rel in ("bin", "Library/bin"):
                candidate = root / rel
                if (candidate / _pdftoppm).is_file():
                    log.debug("Poppler found at: %s", candidate)
                    return str(candidate)
            # Versioned subdirectories (e.g. poppler-26.02.0/Library/bin) — pick newest
            subdirs = sorted((d for d in root.iterdir() if d.is_dir()), reverse=True)
            for sub in subdirs:
                for rel in ("bin", "Library/bin"):
                    candidate = sub / rel
                    if (candidate / _pdftoppm).is_file():
                        log.debug("Poppler found at versioned path: %s", candidate)
                        return str(candidate)

    elif sys.platform == "darwin":
        for bin_dir in (
            "/opt/homebrew/bin",   # Homebrew — Apple Silicon
            "/usr/local/bin",       # Homebrew — Intel
            "/opt/local/bin",       # MacPorts
        ):
            if (Path(bin_dir) / _pdftoppm).is_file():
                log.debug("Poppler found at common macOS path: %s", bin_dir)
                return bin_dir

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


# Pages matching any of these patterns are excluded regardless of keyword hits.
# They identify known-irrelevant page types (nursing flowsheets, MAR sheets, etc.)
# that would otherwise trigger broad category keywords.
_IRRELEVANT_PAGE_PATTERNS: list[re.Pattern] = [
    re.compile(r"MEDICATION\s+ADMINISTRATION\s+RECORD", re.IGNORECASE),
    re.compile(r"NURSING\s+FLOW\s*SHEET", re.IGNORECASE),
    re.compile(r"INTAKE\s+(?:AND\s+)?OUTPUT", re.IGNORECASE),
    re.compile(r"VITAL\s+SIGNS\s+FLOW\s*SHEET", re.IGNORECASE),
    re.compile(r"MEDICATION\s+RECONCILIATION", re.IGNORECASE),
    # Authorization / consent / privacy forms
    re.compile(r"AUTHORIZATION\s+(?:FOR\s+)?(?:RELEASE|DISCLOSURE)\s+OF\s+(?:PROTECTED\s+)?(?:HEALTH\s+)?(?:INFORMATION|RECORDS)", re.IGNORECASE),
    re.compile(r"INFORMED\s+CONSENT", re.IGNORECASE),
    re.compile(r"CONSENT\s+(?:FOR|TO)\s+(?:TREATMENT|PROCEDURE|SURGERY|RELEASE)", re.IGNORECASE),
    re.compile(r"NOTICE\s+OF\s+PRIVACY\s+PRACTICES", re.IGNORECASE),
    re.compile(r"HIPAA\s+(?:PRIVACY\s+)?NOTICE", re.IGNORECASE),
    # Patient registration / demographics
    re.compile(r"PATIENT\s+(?:DEMOGRAPHICS?|REGISTRATION|INFORMATION\s+SHEET)", re.IGNORECASE),
    re.compile(r"(?:NEW\s+)?PATIENT\s+INTAKE\s+FORM", re.IGNORECASE),
    # Scheduling / administrative summaries
    re.compile(r"APPOINTMENT\s+(?:REMINDER|CONFIRMATION|INSTRUCTIONS)", re.IGNORECASE),
    re.compile(r"AFTER[- ]?VISIT\s+SUMMARY", re.IGNORECASE),
    re.compile(r"TABLE\s+OF\s+CONTENTS", re.IGNORECASE),
    # Financial / billing administrative forms
    re.compile(r"FINANCIAL\s+RESPONSIBILITY\s+(?:AGREEMENT|FORM|STATEMENT)", re.IGNORECASE),
    re.compile(r"ASSIGNMENT\s+OF\s+BENEFITS", re.IGNORECASE),
    # Medication administration pages not caught by MAR pattern
    re.compile(r"\bACTIVE\s+MEDICATION\s+LIST\b", re.IGNORECASE),
    # Routine nursing screening tools
    re.compile(r"FALL\s+RISK\s+(?:ASSESSMENT|SCREENING)", re.IGNORECASE),
]


# DEPOSITION DETECTION

# A single match on any of these is a high-confidence deposition signal.
_DEPO_STRONG_RE = re.compile(
    r"\bDEPOSITION\s+OF\b"
    r"|\bDEPONENT\b"
    r"|\bTRANSCRIPT\s+OF\s+(?:ORAL\s+)?(?:DEPOSITION|PROCEEDINGS)\b",
    re.IGNORECASE,
)

# Supporting signals — two or more required when the strong pattern is absent.
_DEPO_SUPPORTING_RES = [
    re.compile(r"^\s*Q\s*[.:]\s+\S", re.MULTILINE),          # Q. ... transcript line
    re.compile(r"\bEXAMINATION\s+BY\b", re.IGNORECASE),
    re.compile(r"\bCOURT\s+REPORTER\b|\bREPORTER'?S?\s+CERTIFICATE\b", re.IGNORECASE),
    re.compile(r"\b(?:DULY\s+)?SWORN\b", re.IGNORECASE),
    re.compile(
        r"\b(?:CROSS|DIRECT|REDIRECT)[- ]EXAMINATION\b", re.IGNORECASE
    ),
]

_DEPO_CHECK_PAGES = 10


def _is_deposition(pdf_path: str) -> bool:
    """Return True if the PDF appears to be a deposition transcript.

    Checks the first _DEPO_CHECK_PAGES pages using native text extraction.
    Requires either a strong deposition marker or two supporting signals.
    """
    try:
        with pdfplumber.open(pdf_path) as pdf:
            pages = pdf.pages[:_DEPO_CHECK_PAGES]
            text = "\n".join((p.extract_text() or "") for p in pages)
    except Exception:
        return False

    if not text.strip():
        return False

    if _DEPO_STRONG_RE.search(text):
        return True

    return sum(1 for pat in _DEPO_SUPPORTING_RES if pat.search(text)) >= 2


def _next_deposition_path(out_dir: Path) -> Path:
    """Return the next available _deposition[_N].pdf path (not thread-safe on its own)."""
    candidate = out_dir / "_deposition.pdf"
    if not candidate.exists():
        return candidate
    i = 1
    while True:
        candidate = out_dir / f"_deposition_{i}.pdf"
        if not candidate.exists():
            return candidate
        i += 1


# CORE SCANNER


def scan_pdf(
    pdf_path: str,
    categories: list[KeywordCategory],
    min_hits: float = 3.0,
    require_categories: frozenset[str] | None = None,
    require_anchor: bool = False,
    progress_callback: Callable[[str], None] | None = None,
) -> ScanResult:
    """
    Scan every page of a PDF for keyword matches and extract all dates.

    Opens pdfplumber once for the whole document, then does a second pass
    for any pages whose text content falls below _MIN_NATIVE_CHARS (OCR
    fallback via Tesseract). Keyword matching and date extraction run after
    both passes across all pages.

    Returns: ScanResult with matched pages and per-document date summary.
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

    # Pass 2: OCR for pages with insufficient native text.
    # A bounded thread pool (≤4 workers) runs Poppler + Tesseract concurrently across
    # pages. Peak memory stays at workers × ~25 MB (≤100 MB) regardless of document
    # size, while a 200-page OCR job runs ~4× faster than a sequential loop.
    if ocr_needed:
        log.debug(
            "%s: %d page(s) below threshold — running OCR.", path.name, len(ocr_needed)
        )

        def _ocr_one_page(page_num: int) -> tuple[int, str]:
            images = convert_from_path(
                pdf_path,
                poppler_path=_POPPLER_BIN,  # type: ignore[arg-type]
                first_page=page_num + 1,
                last_page=page_num + 1,
                dpi=300,
            )
            if images:
                return page_num, str(pytesseract.image_to_string(images[0]))
            return page_num, page_texts[page_num][0]  # keep native text on empty render

        ocr_workers = min(4, len(ocr_needed))
        completed_ocr = 0
        with ThreadPoolExecutor(max_workers=ocr_workers) as ocr_pool:
            ocr_futures = {ocr_pool.submit(_ocr_one_page, pn): pn for pn in ocr_needed}
            for fut in as_completed(ocr_futures):
                completed_ocr += 1
                if progress_callback:
                    progress_callback(
                        f"{path.name}: OCR {completed_ocr}/{len(ocr_needed)} pages…"
                    )
                try:
                    page_num, text = fut.result()
                    page_texts[page_num] = (text, "ocr")
                except Exception as exc:
                    page_num = ocr_futures[fut]
                    log.warning(
                        "Page %d: OCR failed (%s), keeping native text.", page_num, exc
                    )

    # Pass 3: keyword matching + date extraction (all pages)
    matches: list[PageMatch] = []
    exclusions: list[PageExclusion] = []
    all_page_dates: dict[int, list[str]] = {}
    all_dates: set[datetime.date] = set()

    for i, (text, method) in enumerate(page_texts):
        page_dates = _extract_dates(text)
        all_dates.update(page_dates)
        all_page_dates[i] = [d.isoformat() for d in page_dates]
        svc_date, raw_svc_str = _extract_service_date(text)
        page_service_date = svc_date.isoformat() if svc_date else None
        page_npi, page_name_hint, name_ctx = _extract_provider_id(text)

        # Count keywords first (before blocklist) so blocked-but-scored pages
        # can be captured as reviewable exclusions.
        matched_categories: list[str] = []
        weighted_score: float = 0.0
        for cat in categories:
            hits = cat.hits(text)
            if hits:
                matched_categories.append(cat.name)
                weighted_score += cat.weight * len(hits)

        # Deduplicated keyword list (preserves insertion order across categories)
        all_keywords: list[str] = list(
            dict.fromkeys(kw for cat in categories for kw in cat.hits(text))
        )

        # Blocklist check — skip page, but record as exclusion if it scored > 0
        if any(pat.search(text) for pat in _IRRELEVANT_PAGE_PATTERNS):
            log.debug("Page %d skipped — matched irrelevant-page blocklist.", i + 1)
            if weighted_score > 0:
                exclusions.append(
                    PageExclusion(
                        page_num=i,
                        categories=matched_categories,
                        keywords_hit=all_keywords,
                        extraction_method=method,
                        total_hits=round(weighted_score, 2),
                        dates_on_page=all_page_dates[i],
                        exclusion_reasons=["blocked_pattern"],
                        min_hits_threshold=min_hits,
                        service_date=page_service_date,
                        provider_npi=page_npi,
                        provider_name_hint=page_name_hint,
                        raw_service_date_str=raw_svc_str,
                        provider_name_context=name_ctx,
                    )
                )
            continue

        if weighted_score == 0.0:
            continue  # definitively non-clinical — not worth reviewing

        passes_min_hits = weighted_score >= min_hits
        passes_required = require_categories is None or bool(
            set(matched_categories) & require_categories
        )
        passes_anchor = not require_anchor or bool(
            set(matched_categories) & CLINICAL_ANCHOR_CATEGORIES
        )
        if passes_min_hits and passes_required and passes_anchor:
            matches.append(
                PageMatch(
                    page_num=i,
                    categories=matched_categories,
                    keywords_hit=all_keywords,
                    extraction_method=method,
                    total_hits=round(weighted_score, 2),
                    dates_on_page=all_page_dates[i],
                    service_date=page_service_date,
                    provider_npi=page_npi,
                    provider_name_hint=page_name_hint,
                    raw_service_date_str=raw_svc_str,
                    provider_name_context=name_ctx,
                )
            )
            log.debug("Page %d matched: %s", i + 1, matched_categories)
        else:
            reasons: list[str] = []
            if not passes_min_hits:
                reasons.append("below_threshold")
            if not passes_anchor:
                reasons.append("no_anchor")
            if not passes_required:
                reasons.append("no_required_category")
            exclusions.append(
                PageExclusion(
                    page_num=i,
                    categories=matched_categories,
                    keywords_hit=all_keywords,
                    extraction_method=method,
                    total_hits=round(weighted_score, 2),
                    dates_on_page=all_page_dates[i],
                    exclusion_reasons=reasons,
                    min_hits_threshold=min_hits,
                    service_date=page_service_date,
                    provider_npi=page_npi,
                    provider_name_hint=page_name_hint,
                    raw_service_date_str=raw_svc_str,
                    provider_name_context=name_ctx,
                )
            )

    unique_dates = sorted(all_dates)
    log.info(
        "%s: %d matched, %d reviewable exclusion(s) (min_score=%.2f). %d unique date(s).",
        path.name,
        len(matches),
        len(exclusions),
        min_hits,
        len(unique_dates),
    )
    return ScanResult(
        matches=matches,
        exclusions=exclusions,
        source_date_first=unique_dates[0].isoformat() if unique_dates else None,
        source_date_last=unique_dates[-1].isoformat() if unique_dates else None,
        source_date_count=len(unique_dates),
        all_page_dates=all_page_dates,
        page_texts={i: pt for i, pt in enumerate(page_texts)},
    )


def extract_page_text(pdf_path: str, page_num: int) -> str:
    """Extract the text of a single page from a PDF (1-indexed page_num).

    Mirrors the two-pass strategy used by scan_pdf: native pdfplumber first,
    then Tesseract OCR if the native yield is below _MIN_NATIVE_CHARS.
    Returns an empty string if the file is missing, the page is out of range,
    or extraction fails for any reason.
    """
    try:
        path = Path(pdf_path)
        if not path.exists():
            log.warning("extract_page_text: file not found: %s", pdf_path)
            return ""

        zero_idx = page_num - 1
        with pdfplumber.open(pdf_path) as pdf:
            if zero_idx < 0 or zero_idx >= len(pdf.pages):
                log.warning(
                    "extract_page_text: page %d out of range for %s (%d pages).",
                    page_num,
                    path.name,
                    len(pdf.pages),
                )
                return ""
            native = pdf.pages[zero_idx].extract_text() or ""

        if len(native.strip()) >= _MIN_NATIVE_CHARS:
            return native

        # OCR fallback for image-heavy pages
        try:
            images = convert_from_path(
                pdf_path,
                poppler_path=_POPPLER_BIN,  # type: ignore[arg-type]
                first_page=page_num,
                last_page=page_num,
                dpi=300,
            )
            if images:
                return str(pytesseract.image_to_string(images[0]))
        except Exception as exc:
            log.warning("extract_page_text: OCR failed for page %d of %s: %s", page_num, path.name, exc)

        return native  # return whatever native text we got, even if short

    except Exception as exc:
        log.warning("extract_page_text: failed for %s page %d: %s", pdf_path, page_num, exc)
        return ""
