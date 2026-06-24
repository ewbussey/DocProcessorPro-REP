from __future__ import annotations

import re
from dataclasses import dataclass, field

import regex


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
    dates_on_page: list[str] = field(default_factory=list)  # ISO dates found on this page
    service_date: str | None = None       # highest-confidence service date (ISO)
    provider_npi: str | None = None       # extracted 10-digit NPI
    provider_name_hint: str | None = None  # extracted provider name, unverified
    raw_service_date_str: str | None = None  # raw regex-matched date string before ISO parsing
    provider_name_context: str | None = None  # ~150-char text window around provider name match


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
    service_date: str | None = None
    provider_npi: str | None = None
    provider_name_hint: str | None = None
    raw_service_date_str: str | None = None  # raw regex-matched date string before ISO parsing
    provider_name_context: str | None = None  # ~150-char text window around provider name match


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
    page_texts: dict[int, tuple[str, str]] = field(default_factory=dict)
    # 0-indexed page_num → (text, extraction_method)
