"""
Pure keyword/regex PDF page scanner.

scan_pdf(pdf_path, categories, min_hits) -> ScanResult
scan_directory(input_dir, output_dir, categories, min_hits) -> dict[str, list[PageMatch]]
"""

from __future__ import annotations

import csv
import datetime
import io
import logging
import os
import re
import shutil
import sys
import threading
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path

import pdfplumber
import pypdf
import pytesseract
import regex
from pdf2image import convert_from_path
from PIL import Image, ImageDraw, ImageFont

Image.MAX_IMAGE_PIXELS = 300_000_000  # large-format medical pages at 300 DPI can exceed the 89 MP default


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


# DATA STRUCTURES


@dataclass
class KeywordCategory:
    """
    A named category combining plain keywords and raw regex patterns.
    """

    name: str
    keywords: list[str]
    patterns: list[str] = field(default_factory=list)
    weight: float = 1.0
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
                pat = r"\b" + regex.escape(kw) + r"s?\b"
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
    total_hits: float
    dates_on_page: list[str] = field(
        default_factory=list
    )  # ISO dates found on this page


@dataclass
class PageExclusion:
    """A page that scored > 0 but failed one or more inclusion gates."""

    page_num: int  # 0-indexed (matches PageMatch convention)
    categories: list[str]  # categories that did match (may be empty)
    keywords_hit: list[str]  # deduplicated matched keywords
    extraction_method: str  # "pdfplumber" or "ocr"
    total_hits: float  # weighted score (> 0.0)
    dates_on_page: list[str]  # ISO dates found on this page
    exclusion_reasons: list[str]  # subset of: "below_threshold", "no_anchor",
    #   "no_required_category", "blocked_pattern"
    min_hits_threshold: float  # threshold in effect at scan time


@dataclass
class ScanResult:
    """Return value of scan_pdf(), bundling page matches with per-document date info."""

    matches: list[PageMatch]
    exclusions: list[PageExclusion]  # scored-but-excluded pages (score > 0)
    source_date_first: str | None  # ISO YYYY-MM-DD, earliest date across all pages
    source_date_last: str | None  # ISO YYYY-MM-DD, latest date across all pages
    source_date_count: int  # number of unique dates found across all pages
    all_page_dates: dict[
        int, list[str]
    ]  # 0-indexed page → ISO date strings (all pages)


# DEFAULT KEYWORD CATEGORIES

_THERAPY_PLAIN = [
    "therapeutic",
    "physical therapy",
    "active therapy",
    "occupational therapy",
    "speech therapy",
    "aquatic therapy",
    "vestibular",
    "vestibular rehabilitation therapy",
    "neurocognitive therapy",
    "neurocognitive rehabilitation",
    "cognitive rehabilitation therapy",
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
        "VRT",
        "CRT",
        "ST",
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
    weight=1.5,
)

CATEGORY_MEDICAL_TREATMENT = KeywordCategory(
    name="MEDICAL_TREATMENT",
    keywords=[
        "prognosis",
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
        "clinical summary",
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
        "presenting complaint",
        "patient complaint",
        "office visit",
        "consultation",
        "consult",
        "discharge prescription",
        "post-operative order",
        "postoperative order",
    ],
    patterns=[
        r"\b[A-Z]\d{2}(?:\.\d{1,4})?\b",
    ],
    weight=2.5,
)

CATEGORY_BILLING = KeywordCategory(
    name="BILLING",
    keywords=[
        # Standard billing identifiers
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
        "explanation of payment",
        "EOP",
        "EOMB",
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
        "unit",
        "modifier",
        "place of service",
        "POS",
        "NPI",
        "TIN",
        "tax ID",
        "total price",
        "billed",
        "unit price",
        "balance",
        "receipt",
        "payment",
        "fee",
        "CPT",
        "ICD",
        "HCPCS",
        "visit code",
        # Hospital / facility billing
        "itemized statement",
        "itemized bill",
        "itemized charges",
        "revenue code",
        "rev code",
        "DRG",
        "APR-DRG",
        "total charges",
        "amount due",
        "patient responsibility",
        "patient portion",
        "outstanding balance",
        "prior balance",
        "contractual adjustment",
        "contractual allowance",
        "write-off",
        "network discount",
        "not covered",
        # Provider identification
        "rendering provider",
        "referring provider",
        "ordering provider",
        "billing provider",
        "subscriber ID",
        "subscriber name",
        "insured ID",
        "group name",
        "plan name",
        # Government payers
        "Medicare",
        "Medicaid",
        "Medi-Cal",
        "Medicare part A",
        "Medicare part B",
        "Medicare summary notice",
        "MSN",
        "CHAMPVA",
        "TRICARE",
        # Coordination of benefits
        "coordination of benefits",
        "COB",
        "primary payer",
        "secondary payer",
        "primary insurance",
        "secondary insurance",
        # Pharmacy billing
        "NDC",
        "national drug code",
        "days supply",
        "day supply",
        "quantity dispensed",
        "qty dispensed",
        "refill",
        "refills remaining",
        "Rx number",
        "prescription number",
        "dispense date",
        "dispensed date",
        "pharmacy",
        "prescription",
        "pharmacist",
        # Medical bill audit
        "medical necessity",
        "not medically necessary",
        "medically necessary",
        "duplicate billing",
        "duplicate claim",
        "unbundling",
        "upcoding",
        "billing error",
        "overcoding",
        "review of charges",
        "charge review",
    ],
    patterns=[
        r"(?<!\d)[A-Z]\d{4}(?!\d)",             # HCPCS level II / CPT alpha-numeric
        r"\b\d{1,2}/\d{1,2}/\d{2,4}\b",         # date MM/DD/YYYY
        r"\b\d{4}-\d{2}-\d{2}\b",               # date YYYY-MM-DD
        r"\$\s?\d{1,3}(?:,\d{3})*(?:\.\d{2})?", # dollar amounts
        r"\bNPI[:\s#]*\d{10}\b",                 # NPI numbers
        r"\bRev(?:enue)?\s*(?:Code\s*)?\d{3,4}\b",  # UB-04 revenue codes
        r"\bNDC[:\s#]*\d{5}[-\s]\d{4}[-\s]\d{2}\b", # NDC drug codes
        r"\b[A-Z]\d{2}\.\d{1,4}\b",             # ICD-10 codes (e.g. M54.5, S13.4)
    ],
    weight=0.3,
)

CATEGORY_INJURY_LEGAL = KeywordCategory(
    name="INJURY_LEGAL",
    keywords=[
        # Accident / injury facts
        "accident",
        "motor vehicle accident",
        "MVA",
        "motor vehicle collision",
        "MVC",
        "collision",
        "bodily injury",
        "personal injury",
        "slip and fall",
        "trip and fall",
        "incident report",
        "witness statement",
        "property damage",
        # Workers compensation
        "workers compensation",
        "workers comp",
        "work comp",
        "first report of injury",
        "employer's first report",
        "FROI",
        "modified duty",
        "light duty",
        "work restrictions",
        "maximum medical improvement",
        "MMI",
        "DWC",
        "WCAB",
        "TTD",
        # Impairment / disability
        "impairment rating",
        "permanent impairment",
        "whole person impairment",
        "WPI",
        "permanent and stationary",
        "disability",
        "permanent disability",
        "temporary disability",
        # Auto insurance
        "personal injury protection",
        "PIP",
        "uninsured motorist",
        "underinsured motorist",
        "third party",
        "third-party claim",
        "liability",
        "subrogation",
        # Evaluations / examinations
        "independent medical examination",
        "IME",
        "independent medical evaluation",
        "qualified medical evaluator",
        "QME",
        "agreed medical evaluator",
        "AME",
        "panel QME",
        "FCE",
        "narrative report",
        # Legal process
        "affidavit",
        "declaration",
        "under penalty of perjury",
        "sworn statement",
        "notarized",
        "notary public",
        "petition",
        "objection to",
        "objections to",
        "interrogatories",
        "request for production",
        "expert witness",
        "deposition",
        "attorney",
        "law firm",
        # Legal parties / outcome
        "negligence",
        "causation",
        "claimant",
        "plaintiff",
        "defendant",
        "settlement",
        "demand letter",
        "lien",
        "medical lien",
        # Work status
        "work status",
        "restricted to",
        "restrict from",
    ],
    patterns=[
        r"\b\d{1,3}%\s+(?:whole\s+person\s+)?impairment\b",
        r"\b(?:claim|case|file)\s*(?:no|number|#)[:\s]*[\w\-]+\b",
        r"\bunder\s+penalty\s+of\s+perjury\b",
    ],
    weight=1.0,
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
        "impression",
        "without contrast",
        "with contrast",
        "FLAIR",
        "flexion",
        "extension",
        "anterior",
        "posterior",
        "superior",
        "inferior",
        "lateral",
        "medial",
        "oblique",
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
        r"\bT[12][-\s]weighted\b",
    ],
    weight=1.2,
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
        "anxious",
        "depress",
        "depressed",
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
        "session",
        "progress note",
        "cognitive",
        "neurocognitive",
    ],
    patterns=[
        r"\b[FZ]\d{2}(?:\.\d{1,4})?\b",
        r"(?<!\d)90[0-9]{3}(?!\d)",
    ],
    weight=1.5,
)

CATEGORY_VOCATIONAL = KeywordCategory(
    name="VOCATIONAL",
    keywords=[
        # Work history
        "work history",
        "employment history",
        "job history",
        "employment record",
        "prior employment",
        "previous employment",
        "occupational history",
        "vocational history",
        "employer",
        "employment",
        # Earnings / wages
        "wage",
        "salary",
        "earning",
        "income",
        "compensation",
        "lost wage",
        "wage loss",
        "earning capacity",
        "loss of earning capacity",
        "lost income",
        "annual income",
        "hourly wage",
        "hourly rate",
        "weekly wage",
        "monthly income",
        "IRS",
        "W2",
        "1099",
        "social security",
        "SSA",
        "bank statement",
        "paystub",
        "deposit",
        # Job descriptions / duties
        "job description",
        "job duty",
        "job duties",
        "job demand",
        "essential function",
        "physical demand",
        "occupational duty",
        "occupational duties",
        "job title",
        "occupation",
        # Physical work restrictions / capacity
        "work capacity",
        "work tolerance",
        "sedentary",
        "sedentary work",
        "light work",
        "medium work",
        "heavy work",
        "lifting limit",
        "lifting restriction",
        "return to work",
        "RTW",
        "unable to work",
        "work ability",
        "full duty",
        "transitional work",
        "job placement",
        "vocational rehabilitation",
        "vocational assessment",
        "vocational evaluation",
        "transferable skill",
        "labor market",
    ],
    patterns=[
        r"\blift(?:ing)?\s+(?:up\s+to\s+)?\d+\s*(?:lbs?|pounds?)\b",
        r"\b(?:stand|sit|walk)\s+(?:up\s+to\s+)?\d+\s*(?:hours?|hrs?)\b",
    ],
    weight=2.0,
)

CATEGORY_DOCUMENT_TYPE = KeywordCategory(
    name="DOCUMENT_TYPE",
    keywords=[
        # Hospital / ER records
        "discharge note",
        "history and physical",
        "history & physical",
        "H&P",
        "consultation note",
        "consult note",
        "consultation report",
        "physician note",
        "attending note",
        "attending physician note",
        "admission note",
        "admission history",
        "admitting note",
        "emergency department note",
        "triage note",
        "intake note",
        "operative note",
        "procedure note",
        "surgical note",
        # Therapy / rehabilitation
        "initial evaluation",
        "re-evaluation",
        # Imaging
        "imaging report",
        # Behavioral health
        "psychotherapy note",
        "initial psychiatric",
        "psychological evaluation",
        # Legal / IME
        "medical legal report",
        # Therapy subtype — intake / initial encounter
        "initial assessment",
        "intake assessment",
        "initial session",
        "first session",
        "first visit",
        "initial visit",
        # Therapy subtype — discharge / termination
        "termination note",
        "termination summary",
        "termination session",
        "final session",
        "final visit",
        "final note",
        "treatment termination",
        "discharge plan",
        # Therapy subtype — progress notes
        "session note",
        "treatment note",
    ],
    patterns=[
        r"\b(?:discharge|admission|progress|consultation|attending|physician|operative|procedure|triage|intake|initial|termination)\s+(?:summary|note|report|history|evaluation)\b",
    ],
    weight=1.5,
)

DEFAULT_CATEGORIES: list[KeywordCategory] = [
    CATEGORY_THERAPY,
    CATEGORY_MEDICAL_TREATMENT,
    CATEGORY_BILLING,
    CATEGORY_INJURY_LEGAL,
    CATEGORY_IMAGING,
    CATEGORY_BEHAVIORAL_HEALTH,
    CATEGORY_VOCATIONAL,
    CATEGORY_DOCUMENT_TYPE,
]

# Categories that indicate genuine clinical content. Used by the require_anchor filter
# to exclude pages that only matched administrative categories (Billing).
CLINICAL_ANCHOR_CATEGORIES: frozenset[str] = frozenset({
    "THERAPY",
    "MEDICAL_TREATMENT",
    "INJURY_LEGAL",
    "IMAGING",
    "BEHAVIORAL_HEALTH",
    "VOCATIONAL",
    "DOCUMENT_TYPE",
})


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
    matches: list[PageMatch],
    output_csv_path: str,
    source_date_first: str | None = None,
    source_date_last: str | None = None,
    source_date_count: int = 0,
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


def write_unmatched_manifest(
    pdf_path_absolute: str,
    exclusions: list[PageExclusion],
    output_csv_path: str,
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
                ]
            )

    log.info("Unmatched manifest written to %s.", out.name)


def write_matched_manifest(
    pdf_path_absolute: str,
    matches: list[PageMatch],
    output_csv_path: str,
    min_hits_threshold: float = 0.0,
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
                ]
            )

    log.info("Matched manifest written to %s.", out.name)


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
    pdfplumber-extracted pages are left untouched; only image-only pages
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

    dest = out / "_consolidated_dates.csv"
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

    # Write consolidated PDF
    pdf_dest = out / "_consolidated_unmatched.pdf"
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
        csv_dest = out / "_consolidated_unmatched_manifest.csv"
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


# BATCH PROCESSOR


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

        if _is_deposition(str(pdf_path)):
            with _depo_lock:
                dest = _next_deposition_path(out_dir)
                shutil.copy2(pdf_path, dest)
            log.info("Deposition detected: %s → %s", pdf_path.name, dest.name)
            if progress_callback:
                progress_callback(f"Deposition saved: {dest.name}")
            return safe_stem, empty

        try:
            result = scan_pdf(
                str(pdf_path),
                categories,
                min_hits=min_hits,
                require_categories=require_categories,
                require_anchor=require_anchor,
                progress_callback=progress_callback,
            )
        except Exception:
            log.exception("Failed to scan %s — skipping.", pdf_path.name)
            return safe_stem, empty

        if result.matches:
            extract_matched_pages(
                str(pdf_path),
                result.matches,
                str(out_dir / f"{safe_stem}_matched.pdf"),
                page_buffer=page_buffer,
            )
            write_manifest(
                result.matches,
                str(out_dir / f"{safe_stem}_manifest.csv"),
                source_date_first=result.source_date_first,
                source_date_last=result.source_date_last,
                source_date_count=result.source_date_count,
            )
            write_matched_manifest(
                str(pdf_path),
                result.matches,
                str(out_dir / f"{safe_stem}_matched_manifest.csv"),
                min_hits_threshold=min_hits,
            )
            write_dates_csv(
                result.all_page_dates,
                str(out_dir / f"{safe_stem}_dates.csv"),
            )
        else:
            log.info("%s: no matches.", pdf_path.name)

        if result.exclusions:
            extract_unmatched_pages(
                str(pdf_path),
                result.exclusions,
                str(out_dir / f"{safe_stem}_unmatched.pdf"),
            )
            write_unmatched_manifest(
                str(pdf_path),
                result.exclusions,
                str(out_dir / f"{safe_stem}_unmatched_manifest.csv"),
            )

        return safe_stem, result

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

    # Consolidate matched outputs
    matched_stems = [s for s in safe_stems if results.get(s) and results[s].matches]
    if matched_stems:
        consolidated = consolidate_to_pdf(output_dir, matched_stems, progress_callback)
        consolidate_manifests(output_dir, matched_stems)
        consolidate_dates(output_dir, matched_stems)
        if consolidated:
            _ocr_consolidated(consolidated, progress_callback)

    # Consolidate unmatched outputs (independent — a PDF with zero matches can
    # still have scored-but-excluded pages)
    unmatched_stems = [
        s for s in safe_stems if (out_dir / f"{s}_unmatched.pdf").exists()
    ]
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
            # Re-extract remaining pages from source PDFs grouped by source
            stems_seen: dict[str, list[dict]] = {}
            for row in remaining_rows:
                src = row.get("source_pdf_path", "")
                stems_seen.setdefault(src, []).append(row)

            tmp_dir = out / "_unmatched_rebuild_tmp"
            tmp_dir.mkdir(exist_ok=True)
            rebuilt_stems: list[str] = []
            try:
                for src_path, rows in stems_seen.items():
                    if not Path(src_path).exists():
                        log.warning(
                            "apply_feedback: cannot rebuild unmatched for missing source: %s",
                            src_path,
                        )
                        continue
                    stem = Path(src_path).stem
                    page_nums_0idx = [int(r["page_num"]) - 1 for r in rows]
                    src_reader = pypdf.PdfReader(src_path)
                    rebuild_writer = pypdf.PdfWriter()
                    for pn in sorted(page_nums_0idx):
                        if 0 <= pn < len(src_reader.pages):
                            rebuild_writer.add_page(src_reader.pages[pn])
                    pdf_out = tmp_dir / f"{stem}_unmatched.pdf"
                    with open(pdf_out, "wb") as f:
                        rebuild_writer.write(f)
                    csv_out = tmp_dir / f"{stem}_unmatched_manifest.csv"
                    # Re-write per-stem manifest (without consolidated_page_num — will be recomputed)
                    with open(csv_out, "w", newline="", encoding="utf-8") as f:
                        base_fields = [
                            k for k in rows[0].keys()
                            if k != "consolidated_page_num"
                        ]
                        w = csv.DictWriter(f, fieldnames=base_fields)
                        w.writeheader()
                        for row in rows:
                            w.writerow({k: row[k] for k in base_fields})
                    rebuilt_stems.append(stem)

                if rebuilt_stems:
                    # Remove old consolidated unmatched files then re-consolidate
                    for old in (
                        out / "_consolidated_unmatched.pdf",
                        out / "_consolidated_unmatched_manifest.csv",
                    ):
                        if old.exists():
                            old.unlink()
                    # Move tmp files into output_dir for consolidation
                    for stem in rebuilt_stems:
                        for suffix in ("_unmatched.pdf", "_unmatched_manifest.csv"):
                            src_f = tmp_dir / f"{stem}{suffix}"
                            if src_f.exists():
                                src_f.rename(out / f"{stem}{suffix}")
                    consolidate_unmatched(output_dir, rebuilt_stems, progress_callback)
                else:
                    # All remaining unmatched pages came from missing sources — clear the files
                    for old in (
                        out / "_consolidated_unmatched.pdf",
                        out / "_consolidated_unmatched_manifest.csv",
                    ):
                        if old.exists():
                            old.unlink()
            finally:
                if tmp_dir.exists():
                    import shutil
                    shutil.rmtree(tmp_dir, ignore_errors=True)

    log.info(
        "apply_feedback complete: %d approved, %d rejected, %d skipped (missing source).",
        pages_approved,
        rejected_count,
        skipped_missing,
    )
    return pages_approved, rejected_count, skipped_missing


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
