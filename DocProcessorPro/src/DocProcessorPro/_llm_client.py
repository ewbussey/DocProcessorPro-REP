"""Thin HTTP client for the two local LLM inference services (small + large model).

All public functions return None (or safe empty values) when a service is
unavailable, timed out, or returns an unexpected response.  Callers must treat
None as "fall back to rule-based result" — no exception is ever raised.

Two independent services are configured, each with its own URL, set once at
startup via configure_small()/configure_large(); defaults assume two locally
hosted services on ports 8765 (small) and 8766 (large).  Change them via App
Settings in the GUI.

Expected service APIs (implemented separately, e.g. MLX-LM on Mac — see
llm_server/ in the repo root for a runnable scaffold):

    Small model (page categorization + embedding), default port 8765:
        GET  /health
        POST /extract_page_fields   {"text": str, "extraction_method": str}
                                    → {"output": "record_type: ___ | date: ___ | provider: ___ | ..."}
        POST /embed                 {"texts": list[str]}
                                    → {"embeddings": list[list[float]]}

    Large model (case-level analysis + summary generation), default port 8766:
        GET  /health
        POST /summarize_case        {"case_payload": dict}
                                    → {"output": "<json string — see case_summary.py>"}

    The small model returns a pipe-delimited string wrapped in a JSON envelope.
    _parse_pipe_delimited() maps it to the internal field dict.

    Deprecated small-model endpoints (still wired, ignored if service does not
    support them):
    POST /extract_service_date  {"text": str, "raw": str}  → {"date": str|null}
    POST /extract_provider      {"text": str}               → {"name": str|null, "npi": str|null}
    POST /classify_category     {"text": str, "categories": list[str]} → {"category": str|null}

All text payloads are pre-truncated to 2000 characters before sending.
See llm_extraction_schema.md for the full pipe-delimited field reference.
"""

from __future__ import annotations

import json
import logging
import urllib.request

log = logging.getLogger(__name__)

_DEFAULT_SMALL_URL = "http://localhost:8765"
_DEFAULT_LARGE_URL = "http://localhost:8766"
_DEFAULT_URL = _DEFAULT_SMALL_URL  # deprecated alias, kept for compatibility

_SMALL_TIMEOUT = 3.0   # per-page calls must stay snappy
_LARGE_TIMEOUT = 90.0  # case-level generation is slow
_TEXT_LIMIT = 2000

_small_service_url: str = _DEFAULT_SMALL_URL
_large_service_url: str = _DEFAULT_LARGE_URL


def configure_small(url: str) -> None:
    """Set the small-model service base URL.  Called once at startup from app settings."""
    global _small_service_url
    _small_service_url = url.rstrip("/")


def configure_large(url: str) -> None:
    """Set the large-model service base URL.  Called once at startup from app settings."""
    global _large_service_url
    _large_service_url = url.rstrip("/")


def configure(url: str) -> None:
    """Deprecated alias for configure_small() — kept for backward compatibility."""
    configure_small(url)


def _base_url(which: str) -> str:
    return _large_service_url if which == "large" else _small_service_url


def _timeout(which: str) -> float:
    return _LARGE_TIMEOUT if which == "large" else _SMALL_TIMEOUT


def _post(endpoint: str, payload: dict, which: str = "small") -> dict | None:
    """POST JSON to the service endpoint; return parsed response or None on any failure."""
    base_url = _base_url(which)
    try:
        data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        req = urllib.request.Request(
            f"{base_url}/{endpoint}",
            data=data,
            headers={"Content-Type": "application/json; charset=utf-8"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=_timeout(which)) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except Exception as exc:
        log.debug("LLM service unreachable at %s/%s: %s", base_url, endpoint, exc)
        return None


def is_available(which: str = "small") -> bool:
    """Quick health-check.  Returns True if the service responds within the timeout."""
    base_url = _base_url(which)
    try:
        req = urllib.request.Request(
            f"{base_url}/health",
            method="GET",
        )
        with urllib.request.urlopen(req, timeout=_timeout(which)):
            return True
    except Exception:
        return False


def infer_service_date(page_text: str, raw_str: str | None = None) -> str | None:
    """Ask the LLM to extract / confirm the service date.

    Returns an ISO date string ("YYYY-MM-DD") or None if the service is
    unavailable or returns a non-date value.

    Args:
        page_text: Full extracted page text (truncated internally to 2000 chars).
        raw_str:   The raw string already matched by the regex (e.g. "01/05/24"),
                   giving the model a head-start on the ambiguous token.
    """
    result = _post(
        "extract_service_date",
        {"text": page_text[:_TEXT_LIMIT], "raw": raw_str or ""},
    )
    if not result:
        return None
    date_val = result.get("date")
    return str(date_val) if date_val else None


def infer_provider(page_text: str) -> tuple[str | None, str | None]:
    """Ask the LLM to extract provider name and NPI.

    Returns (name_hint, npi).  Either element may be None.  Returns (None, None)
    if the service is unavailable or the response is malformed.
    """
    result = _post("extract_provider", {"text": page_text[:_TEXT_LIMIT]})
    if not result:
        return None, None
    name = result.get("name") or None
    npi = result.get("npi") or None
    return name, npi


def infer_category(page_text: str, candidates: list[str]) -> str | None:
    """Ask the LLM to classify the page into one of the candidate category names.

    Returns the chosen category name (must be one of ``candidates``) or None if
    the service is unavailable, the response is malformed, or the returned value
    is not in the candidates list.
    """
    result = _post(
        "classify_category",
        {"text": page_text[:_TEXT_LIMIT], "categories": candidates},
    )
    if not result:
        return None
    category = result.get("category")
    if category and category in candidates:
        return str(category)
    return None


def _parse_pipe_delimited(raw: str) -> dict:
    """Parse the model's pipe-delimited output into an internal field dict.

    Expected format: "record_type: ___ | date: ___ | provider: ___ | ..."
    Unknown keys are silently ignored so the parser is forward-compatible with
    additional fields added to the model's training data.
    """
    result: dict = {}
    for segment in raw.split(" | "):
        if ": " not in segment:
            continue
        key, _, value = segment.partition(": ")
        key = key.strip().lower()
        value = value.strip()
        if key == "record_type":
            result["record_type"] = value or None
        elif key == "date":
            result["service_date"] = value or None
        elif key == "provider":
            result["provider_name"] = value or None
        elif key == "location":
            result["location_name"] = value or None
        elif key == "title":
            result["record_title"] = value or None
        elif key == "continues_from":
            result["continues_from_previous"] = value.lower() == "yes"
        elif key == "continues_to":
            result["continues_to_next"] = value.lower() == "yes"
        elif key == "confidence":
            try:
                result["confidence"] = float(value)
            except ValueError:
                pass
    return result


def extract_page_fields(page_text: str, extraction_method: str) -> dict | None:
    """Single-pass extraction: record type + universal fields via pipe-delimited model output.

    The service wraps the model's raw output in a JSON envelope:
        {"output": "record_type: ___ | date: ___ | provider: ___ | ..."}

    Returns a parsed field dict (see llm_extraction_schema.md) or None if the
    service is unavailable, the response is malformed, or output is empty.
    The caller is responsible for attaching page_num from the sidecar record.
    """
    result = _post(
        "extract_page_fields",
        {"text": page_text[:_TEXT_LIMIT], "extraction_method": extraction_method},
    )
    if not result:
        return None
    raw = result.get("output", "")
    if not raw:
        return None
    parsed = _parse_pipe_delimited(raw)
    return parsed or None


def embed_texts(texts: list[str]) -> list[list[float]] | None:
    """Batch-embed page texts via the small model's /embed endpoint.

    Returns one vector per input text (same order), or None if the service is
    unavailable or the response is malformed.  Callers must treat None as
    "no embeddings available" — skip RAG ranking / clustering for this batch,
    do not raise.
    """
    result = _post(
        "embed",
        {"texts": [t[:_TEXT_LIMIT] for t in texts]},
        which="small",
    )
    if not result:
        return None
    embeddings = result.get("embeddings")
    if not embeddings or len(embeddings) != len(texts):
        return None
    return embeddings


def summarize_case(case_payload: dict) -> dict | None:
    """Ask the large model to analyze an assembled case and produce summary content.

    The service wraps the model's raw output in a JSON envelope:
        {"output": "<json string — see dpp_scripts/keyword_scanner_scripts/case_summary.py>"}

    Returns the parsed structured summary dict, or None if the service is
    unavailable, the response is malformed, or the output is not valid JSON.
    There is no rule-based fallback for this step — None means the caller
    must tell the user the large model is unavailable, not silently degrade.
    """
    result = _post("summarize_case", {"case_payload": case_payload}, which="large")
    if not result:
        return None
    raw = result.get("output", "")
    if not raw:
        return None
    try:
        parsed = json.loads(raw)
    except (ValueError, TypeError):
        return None
    return parsed if isinstance(parsed, dict) else None
