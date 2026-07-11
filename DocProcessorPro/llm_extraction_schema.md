# LLM Extraction Schema — Pipe-Delimited Output Format

DocProcessorPro talks to two independently-hosted local models, each with its own
service URL configured in App Settings (see `_llm_client.py`):

- **Small model** (default `http://localhost:8765`) — page categorization + embedding.
  Outputs a single pipe-delimited line per page (documented below). This is the model
  the rest of this document describes. It is the **primary** signal for PDF
  splitting-by-relevancy-and-provider: `extraction.classify_page()` classifies every
  non-blocklisted page inline during the scan (`keyword_scanner_codebase.scan_directory()`),
  and `keyword_scanner_codebase._route_page()` routes a page to the records/bills/unmatched
  stream by its `record_type` whenever the model's `confidence` clears
  `categories._LLM_MIN_CONFIDENCE` (0.85). The weighted-keyword scanner (`categories.py`,
  the `min_hits`/`require_categories`/`require_anchor` gates) is the **fallback** path —
  used automatically whenever the service is unavailable or a page's classification isn't
  confident enough — and reproduces the pre-LLM keyword-only behavior exactly.
- **Large model** (default `http://localhost:8766`, Qwen3.6 27B 4-bit) — case-level
  analysis and summary generation, consuming the *output* of the small model's page
  classification plus `record_assembly.py`'s multi-page grouping. See "Large-Model
  Endpoint" below.

A runnable scaffold for both (mock mode + real MLX-LM loading) lives in `llm_server/`
at the repo root — see `llm_server/README.md` for how to run them.

The small model outputs a single pipe-delimited line per page. The DocProcessorPro
client wraps this in a JSON envelope and `_parse_pipe_delimited()` in `_llm_client.py`
maps each field to the internal dict consumed inline during scanning and by `ReviewDialog`.

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

**Implemented** in `dpp_scripts/keyword_scanner_scripts/record_assembly.py`
(`assemble_records()`). The batch pass produces one extraction record per page;
multi-page records are assembled using the continuation flags:

```
group by (source_stem, record_type, location_name, provider_name)
within group: sort by page_num
merge consecutive pages where continues_to_next=True / continues_from_previous=True
result: {page_start, page_end, merged_fields}
```

`source_stem` is the sidecar-file stem (see `load_case_page_records()` in the same
module), not a full filesystem path — sufficient to distinguish source documents
within one scan output directory. When the LLM's `continues_from`/`continues_to`
flags are absent (service unavailable, or the page was never classified), pages
fall back to an embedding-similarity check — `_continues()` merges consecutive
pages whose content embeddings (from the optional `_sidecars/{stem}_embeddings.jsonl`,
written during scanning — see "Embedding Endpoint" below) score `>= 0.90` cosine
similarity. With neither signal available, pages are not merged.

---

## "List of Records Reviewed" Assembly

**Implemented** in `record_assembly.py` (`build_records_reviewed_list()`), computed
deterministically in Python rather than trusted to the model. Groups all *assembled*
records by `(location_name, provider_name)`:
- Date range = `min(service_date)` to `max(service_date)` across the group
- Record description = primary record type(s) seen for that group

Example output line:
> *"1. Medical and Billing Records from One Main Physical Therapy, dated 02/04/2020 – 05/15/2026;"*

This list is passed into the large model's case payload (see below) as ground truth
context — the model is not asked to reproduce it, only to consume it.

---

## Embedding Endpoint (Small Model)

```
POST /embed
{
  "texts": [str, ...]
}
→
{
  "embeddings": [[float, ...], ...]   // one L2-normalized vector per input text, same order
}
```

Used for three purposes (`dpp_scripts/keyword_scanner_scripts/embeddings.py`):

- **RAG context selection** — `case_summary.py` ranks assembled records by relevance
  before building the large model's payload, so case-level prompts stay within a
  context budget instead of growing unbounded with case size.
- **Provider identity clustering** (`provider_clustering.cluster_providers()`) — splits
  matched pages by provider at scan time. Called once per source PDF from
  `keyword_scanner_codebase._scan_one()`, batching one identity-text embedding and one
  content-text embedding per matched page into a single `/embed` call. NPI match wins
  unconditionally; otherwise pages join the nearest cluster centroid at cosine
  similarity `>= 0.86`, falling back to case-insensitive name string equality when
  embeddings are unavailable.
- **Continuation detection fallback** — see the Multi-Page Record Assembly note above.
  Reuses the content embedding computed for provider clustering, persisted to
  `_sidecars/{stem}_embeddings.jsonl`, so no second embedding pass is needed.

Cosine similarity is computed brute-force with numpy — no vector database is used.

---

## Large-Model Endpoint

```
POST /summarize_case
{
  "case_payload": {
    "case_label": str,
    "records_reviewed": [str, ...],          // from build_records_reviewed_list()
    "assembled_records": [
      {
        "provider_name": str | null, "location_name": str | null,
        "record_type": str | null,
        "service_date_start": str | null, "service_date_end": str | null,
        "page_start": int, "page_end": int, "source_stem": str | null,
        "excerpt": str                        // merged_text, possibly RAG-trimmed
      }, ...
    ]
  }
}
→
{
  "output": "<json string>"   // parses to {records_reviewed, provider_chronology, narrative_summary} — see dpp_scripts/keyword_scanner_scripts/case_summary.py and dpp_scripts/docx_scripts/docx_codebase.py
}
```

There is no rule-based fallback for this endpoint — `generate_case_summary()` returns
`None` if the large-model service is unavailable, and the caller (`ReviewDialog`)
surfaces that to the user rather than silently degrading.

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
