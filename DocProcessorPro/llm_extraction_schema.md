# LLM Extraction Schema — Pipe-Delimited Output Format

The Qwen3 8B model outputs a single pipe-delimited line per page. The DocProcessorPro
client wraps this in a JSON envelope and `_parse_pipe_delimited()` in `_llm_client.py`
maps each field to the internal dict consumed by `_LlmBatchWorker` and `ReviewDialog`.

---

## API Contract

```
POST /extract_page_fields
{
  "text": str,              // page text (pre-truncated to 2000 chars by _llm_client.py)
  "extraction_method": str  // "liteparse" | "liteparse_ocr" — quality hint
}
→
{
  "output": "record_type: therapy_non_psych | date: 2024-01-15 | provider: Dr. Jane Smith | location: Main Street PT | title: Progress Note | continues_from: no | continues_to: no | confidence: 0.92"
}
```

The model outputs a single line. The server wraps it in `{"output": "..."}` before
returning JSON to the client.

---

## Pipe-Delimited Fields

Fields appear in this order (all optional — absent fields are omitted, not nulled):

| Pipe key | Internal dict key | Type | Notes |
|---|---|---|---|
| `record_type` | `record_type` | str (enum) | See record types below |
| `date` | `service_date` | ISO date str or `""` | Primary service/encounter date |
| `provider` | `provider_name` | str or `""` | Treating/rendering provider name |
| `location` | `location_name` | str or `""` | Facility or practice name |
| `title` | `record_title` | str or `""` | Document title (legal docs, reports) |
| `continues_from` | `continues_from_previous` | `"yes"` / `"no"` → bool | Multi-page continuation signal |
| `continues_to` | `continues_to_next` | `"yes"` / `"no"` → bool | Multi-page continuation signal |
| `confidence` | `confidence` | float 0–1 | Model self-reported confidence |

Unknown keys are silently ignored — the parser is forward-compatible with additional
fields added to training data.

---

## Record Types (`record_type` enum)

| Value | Category mapping | Description |
|---|---|---|
| `office_visit` | MEDICAL_TREATMENT | Standard outpatient office visit |
| `therapy_non_psych` | THERAPY | PT, OT, chiropractic, speech therapy |
| `therapy_psych` | BEHAVIORAL_HEALTH | Psychology, psychiatry, counseling |
| `inpatient_stay` | MEDICAL_TREATMENT | ER, inpatient rehab, hospital admission |
| `imaging` | IMAGING | X-ray, MRI, CT, ultrasound |
| `bill` | BILLING | CMS-1500, UB-04, EOBs, itemized billing |
| `billing_affidavit` | BILLING | Billing affidavits |
| `vocational` | VOCATIONAL | FCE, impairment ratings, work history |
| `legal_document` | INJURY_LEGAL | Liens, authorizations, non-billing affidavits |
| `pharmacy` | BILLING | Pharmacy dispensing records |
| `ime` | INJURY_LEGAL | Independent Medical Examination (examining, not treating) |
| `neuropsych_testing` | BEHAVIORAL_HEALTH | Neuropsychological / psychological testing reports |
| `operative_report` | MEDICAL_TREATMENT | Surgical and operative reports |
| `other_nec` | (keyword fallback) | Not otherwise classified |

The `Category mapping` column is used by `_RECORD_TYPE_TO_CATEGORY` in
`_review_dialog.py` to make `record_type` the authoritative document category
for all LLM-processed pages.

---

## Multi-Page Record Assembly (Downstream Python)

The batch pass produces one extraction record per page. Multi-page records are
assembled by downstream Python using the continuation flags:

```
group by (source_pdf_path, record_type, location_name, provider_name)
within group: sort by page_num
merge consecutive pages where continues_to_next=True / continues_from_previous=True
result: {page_start, page_end, merged_fields}
```

---

## "List of Records Reviewed" Assembly

Groups all pages by `(location_name, provider_name)` across all source PDFs:
- Date range = `min(service_date)` to `max(service_date)` across the group
- Record description = primary record type(s) seen for that group

Example output line:
> *"1. Medical and Billing Records from One Main Physical Therapy, dated 02/04/2020 – 05/15/2026;"*

---

## Extraction Method Labels (from LiteParse integration)

| Value | Meaning |
|---|---|
| `liteparse` | Native text extracted by LiteParse (no OCR) |
| `liteparse_ocr` | Image page — LiteParse native text was below threshold; Tesseract OCR applied |

The `extraction_method` field is sent to the model as a quality hint. `liteparse_ocr`
pages have lower text fidelity and the model should reflect that in its `confidence` output.

---

## Deprecated Endpoints

These endpoints are wired in `_llm_client.py` but no longer the primary path.
They are kept for backward compatibility and may be removed in a future version:

- `POST /extract_service_date` — superseded by `date` field in pipe-delimited output
- `POST /extract_provider` — superseded by `provider` field
- `POST /classify_category` — superseded by `record_type` → category mapping
