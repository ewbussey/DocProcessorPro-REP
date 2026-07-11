"""Prompt templates for the real (non-mock) model paths.

Keep the extraction field order/keys in sync with llm_extraction_schema.md —
this is exactly what _llm_client._parse_pipe_delimited() expects on the
response side.
"""

from __future__ import annotations

import json

_EXTRACT_FIELD_ORDER = (
    "record_type", "date", "provider", "location", "title",
    "continues_from", "continues_to", "confidence",
)

_RECORD_TYPES = (
    "office_visit", "therapy_non_psych", "therapy_psych", "inpatient_stay",
    "imaging", "bill", "billing_affidavit", "vocational", "legal_document",
    "pharmacy", "ime", "neuropsych_testing", "operative_report", "other_nec",
)


def build_extract_prompt(text: str, extraction_method: str) -> str:
    quality_hint = (
        "This page was OCR'd from a scanned image — text may contain recognition errors."
        if extraction_method == "liteparse_ocr"
        else "This page's text was extracted natively (no OCR)."
    )
    field_list = " | ".join(f"{k}: <value>" for k in _EXTRACT_FIELD_ORDER)
    return f"""You are extracting structured fields from a single page of a medical/legal record.
{quality_hint}

Respond with EXACTLY one line, pipe-delimited, in this exact key order:
{field_list}

record_type must be one of: {", ".join(_RECORD_TYPES)}
continues_from / continues_to must be "yes" or "no".
confidence must be a number between 0 and 1.
Leave a field blank (but keep the key) if it cannot be determined from the page.

PAGE TEXT:
\"\"\"
{text}
\"\"\"

Respond with only the pipe-delimited line, no other text."""


def build_case_summary_prompt(case_payload: dict) -> str:
    return f"""You are preparing a medical-record case summary for a legal/insurance reviewer.

Given the following assembled record groups for case "{case_payload.get('case_label', '')}",
produce a JSON object with exactly these keys:
  "records_reviewed": array of strings (echo the provided list verbatim)
  "provider_chronology": array of objects: {{provider_name, location_name, date_range, entries: [{{date, record_type, summary}}]}}
  "narrative_summary": a concise prose paragraph synthesizing the clinically/legally significant findings

CASE DATA:
{json.dumps(case_payload, indent=2)}

Respond with only the JSON object, no other text."""
