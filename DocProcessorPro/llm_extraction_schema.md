# LLM Extraction Schema — Per-Record-Type Field Reference

This document defines the structured fields to extract per page during the LLM batch pass.
Each extraction result is **page-tagged** (from the `_page_texts.jsonl` sidecar) and includes
continuation flags so downstream Python can assemble multi-page records into page ranges.

---

## "List of Records Reviewed" — Universal Fields

**Every page, regardless of record type or match status, must produce these fields.**
They drive the auto-generated records list at the start of the report, e.g.:

> *"1. Medical and Billing Records from One Main Physical Therapy, dated 02/04/2020 – 05/15/2026;"*

| Field | Type | Notes |
|---|---|---|
| `location_name` | string \| null | Facility, practice, or organization name |
| `provider_name` | string \| null | Treating/rendering provider; null for facilities/legal docs |
| `record_title` | string \| null | Inline document title for legal docs; null for clinical records |
| `dates_on_page` | list[ISO date] | All dates found — downstream takes min/max per location group |

These fields are present on **all** extraction results. The LLM should populate them even when
it cannot confidently classify the record type or extract any type-specific fields.

**Downstream assembly for the records list:**
1. Group all pages by `(location_name, provider_name)` across all source PDFs
2. Date range = `min(dates_on_page)` to `max(dates_on_page)` across the group
3. Record description = primary record type(s) seen for that group

---

## Additional Universal Fields (every record type)

| Field | Type | Notes |
|---|---|---|
| `record_type` | string (enum) | See types below |
| `page_num` | int | 1-indexed, from sidecar |
| `source_pdf_path` | string | From sidecar |
| `confidence` | float 0–1 | Model's self-reported confidence |
| `continues_from_previous` | bool | True if this page continues a record started on the prior page |
| `continues_to_next` | bool | True if this record continues onto the next page |
| `extraction_method` | string | "pdfplumber" or "ocr" — from sidecar; lower confidence on "ocr" |

---

## Record Types and Type-Specific Fields

### `office_visit`

| Field | Type |
|---|---|
| `service_date` | ISO date \| null |
| `chief_complaint_history` | string \| null — free text (H&P / chief complaint section) |
| `procedures` | list[string] — procedure names and/or CPT codes, empty if none |
| `diagnoses` | list[string] — free text or ICD codes as written |
| `recommendations` | list[string] — services, behavior modifications, prescriptions, referrals, imaging orders, etc. |

---

### `therapy_non_psych`

Physical therapy, occupational therapy, chiropractic, speech therapy, etc.

| Field | Type |
|---|---|
| `therapy_type` | string \| null — e.g. "Physical Therapy", "Chiropractic", "OT" |
| `service_date` | ISO date \| null |
| `all_session_dates` | list[ISO date] — all dates mentioned; downstream uses min/max for range |

> Downstream uses `min(all_session_dates)` / `max(all_session_dates)` as first/last relevant
> dates. The full array is preserved to support accurate range construction across pages.

---

### `therapy_psych`

Psychology, psychiatry, counseling, behavioral health.

| Field | Type |
|---|---|
| `service_date` | ISO date \| null |
| `diagnoses` | list[string] |
| `recommendations` | list[string] |

---

### `inpatient_stay`

Emergency room, inpatient rehab, hospital admission, medical extended stay.

| Field | Type |
|---|---|
| `admit_date` | ISO date \| null |
| `admit_diagnoses` | list[string] |
| `services_performed` | list[string] |
| `surgery_procedure_names` | list[string] |
| `surgery_procedure_dates` | list[ISO date] |
| `discharge_date` | ISO date \| null |
| `discharge_next_steps` | list[string] — discharge instructions, follow-up referrals, prescriptions at discharge |

---

### `imaging`

X-ray, MRI, CT, ultrasound, etc.

| Field | Type |
|---|---|
| `service_date` | ISO date \| null |
| `imaging_type` | string \| null — e.g. "MRI Lumbar Spine", "X-Ray Right Knee" |
| `findings_summary` | string \| null — brief free-text impression if present |

---

### `bill`

CMS-1500, UB-04, EOBs, itemized billing statements.

| Field | Type |
|---|---|
| `dates_of_service` | list[ISO date] |
| `cpt_hcpcs_codes` | list[string] |
| `line_item_costs` | list[dict] — `{"code": str, "description": str\|null, "amount": float\|null}` |
| `total_billed` | float \| null |

---

### `billing_affidavit`

| Field | Type |
|---|---|
| `affidavit_date` | ISO date \| null |
| `records_date_range_start` | ISO date \| null |
| `records_date_range_end` | ISO date \| null |

---

### `vocational`

Employment records, FCE reports, impairment ratings, work history, earnings data.
May overlap with other types (e.g. a physician's RTW note is also `office_visit`).

| Field | Type |
|---|---|
| `employer_name` | list[string] |
| `any_dates` | list[ISO date] — catch-all |
| `work_history_duties` | string \| null |
| `earnings_records` | list[dict] — `{"type": str, "date_range_start": ISO\|null, "date_range_end": ISO\|null, "amount": float\|null, "description": str\|null}` |
| `vocational_restrictions` | list[string] |
| `return_to_work_notes` | string \| null |
| `termination_notes` | string \| null |
| `job_application_notes` | string \| null |
| `fce_notes` | string \| null |
| `impairment_rating_notes` | string \| null |

---

### `legal_document`

Non-billing affidavits, legal correspondence, liens, authorizations, etc.
`record_title` and `dates_on_page` from the universal fields cover the primary need.

*(No additional type-specific fields beyond the universal set.)*

---

### `pharmacy`

Pharmacy dispensing records. Granular medication data not required.
`location_name` and `dates_on_page` from the universal fields cover the full need.

*(No additional type-specific fields beyond the universal set.)*

---

### `ime`

Independent Medical Examination — same document structure as `office_visit` but authored
by an *examining* (not treating) provider, typically at the request of an insurer or attorney.
The distinction matters for legal weight and downstream flagging.

| Field | Type |
|---|---|
| `service_date` | ISO date \| null |
| `requesting_party` | string \| null — insurer, defense attorney, etc. |
| `examining_provider_specialty` | string \| null |
| `history_summary` | string \| null — history as reported by patient or gleaned from records |
| `exam_findings` | string \| null — physical / clinical findings from the exam |
| `diagnoses` | list[string] — examiner's diagnostic impressions |
| `opinion_summary` | string \| null — causation opinion, disability opinion, overall conclusion |
| `recommendations` | list[string] — further treatment, work restrictions, follow-up |

> `provider_name` from the universal fields = the examining provider (not the claimant's treater).

---

### `neuropsych_testing`

Neuropsychological and psychological testing reports. Overlaps with `therapy_psych` in
provider type but produces specific scored outputs rather than treatment notes.

| Field | Type |
|---|---|
| `service_date` | ISO date \| null — date of report; may differ from test dates |
| `test_dates` | list[ISO date] — all dates testing was administered |
| `tests_administered` | list[string] — e.g. "MMPI-2", "WAIS-IV", "Beck Depression Inventory" |
| `test_findings` | string \| null — summary of results / score narrative |
| `diagnoses` | list[string] |
| `recommendations` | list[string] |

---

### `operative_report`

Surgical and operative reports. More granular than `inpatient_stay`; may appear as a
standalone document or embedded within inpatient records.

| Field | Type |
|---|---|
| `surgery_date` | ISO date \| null |
| `procedure_names` | list[string] — full procedure names as written |
| `procedure_codes` | list[string] — CPT or other codes if present |
| `pre_op_diagnoses` | list[string] |
| `post_op_diagnoses` | list[string] |
| `anesthesia_type` | string \| null — e.g. "General", "Spinal", "Local" |
| `assistant_surgeons` | list[string] — names of assisting surgeons/staff if listed |
| `intraoperative_findings` | string \| null — findings noted during the procedure |
| `complications` | list[string] — empty list if none documented |

> `provider_name` from the universal fields = the primary/attending surgeon.

---

### `other_nec`

**Not Otherwise Classified** — school/educational records, unknown record types,
non-categorized documents, and anything that doesn't fit a type above.

`location_name`, `record_title`, and `dates_on_page` from the universal fields
cover the full need for these records.

*(No additional type-specific fields beyond the universal set.)*

---

## Page Range Assembly (Downstream Python — No LLM)

The batch pass produces one extraction record per page. Multi-page records (e.g. a
discharge summary spanning pages 45–52) are assembled into ranges by downstream Python:

```
group by (source_pdf_path, record_type, location_name, provider_name)
within group: sort by page_num
merge consecutive pages where continues_to_next=True / continues_from_previous=True
result: {page_start, page_end, merged_fields}
```

The `continues_from_previous` / `continues_to_next` flags are the key signal — the model
sets them when the page is clearly a continuation (e.g. "page 3 of 6", narrative mid-sentence,
no new header block).

---

## Batch Pass Scope

The batch pass runs on the **full `_page_texts.jsonl` sidecar** — all pages of all source PDFs,
not just matched/approved pages. This ensures the "list of records reviewed" captures every
document in the production, including unmatched and excluded pages.

---

## API Contract

Add to the LLM service (separate project):

```
POST /extract_page_fields
{
  "text": str,              // page text (pre-truncated to 2000 chars by _llm_client.py)
  "extraction_method": str  // "pdfplumber" | "ocr" — hint for confidence calibration
}
→
{
  // Universal — always present
  "record_type": str,
  "location_name": str | null,
  "provider_name": str | null,
  "record_title": str | null,
  "dates_on_page": list[str],        // ISO dates
  "confidence": float,
  "continues_from_previous": bool,
  "continues_to_next": bool,
  // Type-specific fields — present only when relevant to record_type
  ...
}
```

Single call per page. Model determines `record_type` and extracts all fields for that type
in one pass. Null for missing scalars; empty list for missing lists.

---

## Potentially Missing Record Types (to revisit)

- **IME (Independent Medical Examination) reports** — office visit fields but examining (not treating) provider; `is_ime: true` flag may be useful
- **Neuropsychological / psychological testing reports** — psych therapy fields + specific test scores
- **Operative / surgical reports** — more granular than inpatient stay
