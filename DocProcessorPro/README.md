# DocProcessorPro

A PySide6 desktop tool for processing large batches of medical-record PDFs for legal/insurance case work. It scans PDFs, splits pages by relevancy and treating provider, supports human review of that split, and generates a case-summary `.docx` — using two local LLMs where available, with automatic fallback to keyword/regex logic when they're not.

This guide covers the project's design and code organization. It intentionally does **not** duplicate two more detailed docs that already exist:

- [`llm_extraction_schema.md`](llm_extraction_schema.md) — the exact wire format the small model's output must follow.
- [`llm_server/README.md`](llm_server/README.md) — how to run the two LLM server processes.

## Table of contents

1. [Quick start](#quick-start)
2. [Architecture at a glance](#architecture-at-a-glance)
3. [Directory & module map](#directory--module-map)
4. [The scanning/splitting pipeline](#the-scanningsplitting-pipeline)
5. [Output directory reference](#output-directory-reference)
6. [The review workflow](#the-review-workflow)
7. [The two-LLM system](#the-two-llm-system)
8. [Settings reference](#settings-reference)
9. [Known gaps / not-yet-done](#known-gaps--not-yet-done)
10. [Where to go next](#where-to-go-next)

---

## Quick start

There's no `[project.scripts]` entry point declared in `pyproject.toml`, so the GUI is launched directly as a module:

```bash
pip install -e .
python -m DocProcessorPro.DocProcessorPro
```

(equivalently: `python src/DocProcessorPro/DocProcessorPro.py`)

The app works fully offline — the keyword/regex scanner is always available. To exercise the LLM-backed features (page classification, provider splitting quality, case-summary generation) without needing real model weights, run the mock servers:

```bash
pip install -e ".[llm-server]"
python -m llm_server.small_model_server --mock    # port 8765
python -m llm_server.large_model_server --mock    # port 8766
```

Both default ports match the app's default settings, so no configuration is needed to test against them. See [`llm_server/README.md`](llm_server/README.md) for running real model weights instead.

> **Packaging note:** `build.ps1` (Windows PyInstaller build) references a `DocProcessorPro.spec` file that does not currently exist in the repo. Packaging is not reproducible as-is — see [Known gaps](#known-gaps--not-yet-done).

---

## Architecture at a glance

```text
input PDFs
    │
    ▼
scan + classify + route pages          (keyword_scanner_scripts/)
    │   LLM-primary, keyword-fallback
    ▼
per-provider split output              (_records/, _bills/, _unmatched/, provider-partitioned PDFs)
    │
    ▼
human review (ReviewDialog)            (_review_dialog.py)
    │   approve / reject / correct
    ▼
feedback export (_feedback.jsonl)
    │
    ├──▶ case-summary generation ──▶ record_assembly → case_summary → docx_codebase ──▶ {case}_summary.docx
    │        (large LLM, no fallback — requires the service)
    │
    └──▶ fine-tuning data export ──▶ feedback_to_training.py ──▶ src/data/training.jsonl
             (small model, for future fine-tuning)
```

Two independent local LLM services sit behind this pipeline (see [The two-LLM system](#the-two-llm-system)), each optional except where noted.

---

## Directory & module map

### GUI layer (`src/DocProcessorPro/`)

| File | Role |
|---|---|
| `DocProcessorPro.py` | Entry point. `main()` sets up logging, builds the `QApplication`, shows `ScannerDialog`. |
| `_scanner_dialog.py` | Main window: input/output pickers, scan settings, App Settings dialog (output root, LLM URLs). Owns the single `QSettings` instance for the whole app. |
| `_review_dialog.py` | The three-panel page-review UI — by far the largest file in the project. Approve/reject decisions, category/provider/date corrections, Smart Triage, Therapy Triage, Find Duplicates integration, feedback export. |
| `_dedup.py` | Perceptual-hash (dHash) visual duplicate detection — a distinct mechanism from the LLM/embedding-based provider clustering described below; this one compares rendered page images, not text. |
| `_workers.py` | `QThread` subclasses that run long operations off the GUI thread: `_ScanWorker`, `_FeedbackWorker`, `_CaseSummaryWorker`, plus update-check workers. |
| `_gui_utils.py` | Window/icon helpers shared across dialogs. |
| `_llm_client.py` | Thin HTTP client to the two LLM services. Never raises — every function returns `None` on any failure, which callers treat as "fall back." |

### Scanning/splitting pipeline (`dpp_scripts/keyword_scanner_scripts/`)

| File | Role |
|---|---|
| `models.py` | Dataclasses: `KeywordCategory`, `PageMatch`, `PageExclusion`, `ScanResult`. |
| `categories.py` | The 8 hardcoded keyword categories (weights, keyword/regex lists) plus the shared LLM classification metadata: `_RECORD_TYPE_TO_CATEGORY`, `_LLM_MIN_CONFIDENCE`, `_LLM_AUTO_APPROVE_TYPES`, `_BILLS_REQUIRE_CATEGORIES`. |
| `extraction.py` | Per-PDF page extraction (native LiteParse + Tesseract OCR fallback) and `classify_page()` — the LLM-primary/keyword-fallback per-page classifier. |
| `keyword_scanner_codebase.py` | Orchestration: `scan_directory()` batch-processes a folder of PDFs, `_route_page()` decides records/bills/unmatched per page, provider partitioning, sidecar writing, consolidation. |
| `pdf_ops.py` | Low-level PDF page extraction, CSV manifest writers, sidecar JSONL writers, consolidation, `_slugify()`. |
| `provider_clustering.py` | Groups matched pages by treating provider (NPI exact match → embedding centroid similarity → name-string fallback) to drive provider-partitioned output files. |
| `embeddings.py` | Shared cosine-similarity / top-k utilities used by provider clustering, RAG context selection, and continuation detection. Brute-force numpy — no vector database. |
| `keyword_scanner_main.py` | Hardcoded-path fallback script (`INPUT_DIR`/`OUTPUT_DIR` edited in-file) for running a scan without the GUI. |

### Case summary generation

| File | Role |
|---|---|
| `dpp_scripts/keyword_scanner_scripts/record_assembly.py` | Loads per-page sidecar data for a whole output directory and merges consecutive pages into multi-page "records" (via LLM continuation flags, falling back to embedding similarity). Also builds the "List of Records Reviewed" text. |
| `dpp_scripts/keyword_scanner_scripts/case_summary.py` | Builds the payload sent to the large model's `/summarize_case` endpoint, with RAG-style excerpt selection when the case is too large to fit in context whole. |
| `dpp_scripts/docx_scripts/docx_codebase.py` | Renders the large model's structured response into a `.docx` (title, records-reviewed list, provider chronology table, narrative summary). |

### Fine-tuning data

| File | Role |
|---|---|
| `dpp_scripts/training_scripts/feedback_to_training.py` | Converts reviewed `_feedback.jsonl` pages into `{"prompt", "completion"}` pairs for eventual small-model fine-tuning. Runnable as its own CLI (`python -m DocProcessorPro.dpp_scripts.training_scripts.feedback_to_training <output_dir>`). |

### LLM client/server

| Location | Role |
|---|---|
| `src/DocProcessorPro/_llm_client.py` | In-app HTTP client (see GUI layer table above). |
| `llm_server/` | **Ships separately from the GUI** — two standalone FastAPI processes (`small_model_server.py`, `large_model_server.py`) that wrap MLX-LM. Installed via the `llm-server` extra, not part of the main app dependencies. See its own [README](llm_server/README.md). |

### Other

- `dpp_scripts/update_scripts/` — self-update checker (fetches a remote version manifest, downloads/installs updates).
- `dpp_scripts/file_management_scripts/` — generic file-collection helpers (`.txt`/`.docx`/`.dcm`/`.pdf`/image), used elsewhere in the app; unrelated to LLM/docx summary generation despite the similar name.
- `logging_resources/` — logging configuration (`logging_config.yaml`) and context helpers.
- `ocr_pdfs.py` (repo root) — a standalone batch-OCR CLI script, independent of the GUI.

---

## The scanning/splitting pipeline

This is the technical core of the app. For a given input folder, `scan_directory()` processes each PDF (in a thread pool, one PDF per worker) through:

1. **Text extraction** — native text via LiteParse first; any page whose native yield is under 100 characters falls back to Tesseract OCR (via `pdf2image` + `pytesseract`).

2. **Classification — LLM-primary, keyword-fallback.** `extraction.classify_page()` calls the small model's `/extract_page_fields` endpoint for every page that isn't blocklisted. **Availability is checked exactly once per batch** (not per page, not per PDF) — a down service costs one health-check timeout total, not one per page. When the service is unavailable, or a page's classification confidence is below `_LLM_MIN_CONFIDENCE` (0.85, in `categories.py`), the page falls back to the original weighted-keyword scoring — this fallback path is designed to reproduce the pre-LLM keyword-only behavior exactly, so the app degrades gracefully offline.

3. **Routing.** `keyword_scanner_codebase._route_page()` decides records / bills / unmatched per page. A confident LLM classification is authoritative and bypasses the keyword category/anchor requirements (which exist to compensate for keyword scoring's blind spots) — but it can never rescue a page the irrelevant-page blocklist or zero-signal check already dropped.

4. **Provider clustering.** `provider_clustering.cluster_providers()` groups each PDF's matched pages by treating provider: exact NPI match wins unconditionally; otherwise pages join the nearest embedding-centroid cluster (cosine similarity ≥ 0.86); with no embeddings available, it falls back to case-insensitive name-string equality. This drives **provider-partitioned output files** — additional `{stem}__{provider_slug}_records.pdf`-style files alongside the always-written unpartitioned ones.

5. **Sidecar writing.** Every non-blank page's text, LLM classification (if any), and content embedding (if any) are written to `_sidecars/` as JSONL — consumed later by the review UI, case-summary generation, and training-data export.

The exact wire format for step 2 (the pipe-delimited small-model output) is documented in [`llm_extraction_schema.md`](llm_extraction_schema.md) rather than repeated here.

---

## Output directory reference

Every scan run writes into the chosen output directory:

| Path | Contents |
|---|---|
| `{stem}_records.pdf`, `_records_manifest.csv`, etc. (root) | Per-source-PDF outputs before consolidation, one set per input PDF. |
| `{stem}__{provider_slug}_records.pdf` (root) | Provider-partitioned variant, written only when a PDF's matches span more than one distinct provider. |
| `_records/` | Consolidated relevant-records stream: `_consolidated_records.pdf`, its manifest, and `_consolidated_dates.csv`. |
| `_bills/` | Consolidated billing-record stream. |
| `_unmatched/` | Consolidated pages that didn't clear any threshold — the human review queue. |
| `_review/` | `_consolidated_review.pdf` (everything, scored) and the generated case-summary `.docx`, once produced. |
| `_sidecars/` | Per-stem JSONL: `_page_texts.jsonl` (raw page text), `_llm_fields.jsonl` (LLM classification, when available), `_embeddings.jsonl` (content embeddings, when available). |
| `_depositions/` | Whole PDFs detected as deposition transcripts, routed out of the normal pipeline entirely. |

> **Note:** `pdf_ops.consolidate_to_pdf()` and the root-level `_consolidated.pdf`/`_consolidated_matched_manifest.csv` path exist in code but are not called anywhere in the current pipeline — legacy from before the records/bills split. The stream-specific subdirectories above are the live path.

---

## The review workflow

`ReviewDialog` is where a human confirms or corrects the scanner's output before anything downstream happens. It supports approving/rejecting individual pages, correcting category/provider/service-date assignments, semi-automated triage flows (Smart Triage, Therapy Triage), and visual duplicate detection (via `_dedup.py`).

`_export_feedback()` writes the review session's decisions to `_feedback.jsonl` (every field, decision provenance included) and `_corrections.jsonl` (just the fields a human actually changed). These two files are the bridge to everything downstream:

- **Case-summary generation** operates on the reviewed/approved page set, not raw scan output.
- **Training-data export** (`feedback_to_training.py`) converts approved, human-confirmed pages into fine-tuning pairs for the small model.

---

## The two-LLM system

| | Small model | Large model |
|---|---|---|
| Role | Per-page categorization + embedding | Case-level analysis + summary generation |
| Default port | 8765 | 8766 |
| Settings key | `llm_service_url` | `llm_large_service_url` |
| Fallback if unavailable | Keyword/regex scoring (graceful) | **None** — case-summary generation requires it |

Both are configured via the App Settings dialog in the GUI (persisted to `QSettings`), and both are hosted by the standalone `llm_server/` processes — see its [README](llm_server/README.md) for setup, and [`llm_extraction_schema.md`](llm_extraction_schema.md) for the small model's exact output contract.

---

## Settings reference

All persistent app settings live under a single `QSettings("DocProcessorPro", "KeywordScanner")` instance (`_scanner_dialog.py`):

| Key | Default | Controls |
|---|---|---|
| `last_input_dir` | `""` | Last-used input folder. |
| `last_output_dir` | `"Default Output Path"` (sentinel) | Last-used output folder. |
| `output_root` | `~/Documents/dpp_outputs` | Root for auto-derived per-claimant output paths: `{output_root}/{Initial}/{Last, First CaseNum}/`. |
| `llm_service_url` | `http://localhost:8765` | Small-model server URL. |
| `llm_large_service_url` | `http://localhost:8766` | Large-model server URL. |

The min-hits threshold and page-buffer spinboxes are per-session UI values only — not persisted across restarts.

---

## Known gaps / not-yet-done

- **Packaging is broken as-is** — `build.ps1` invokes `pyinstaller DocProcessorPro.spec`, but no `.spec` file exists in the repo.
- **`llm_server/small_model_server.py`'s `_load_embedder()` is a placeholder** — it assumes an `mlx_embeddings`-style loader, to be verified/swapped once the actual small embedding model is chosen.
- **Provider clustering and continuation-detection quality depends on real embeddings.** The mock server's embeddings are deterministic-but-not-semantic (hash-derived), so mock mode proves the plumbing works, not real-world clustering accuracy — that needs real model weights to evaluate.
- **No automated test suite exists.** Verification so far has been manual/scripted (fixture PDFs + checksum/content comparison), not a committed `tests/` directory.
- `requirements.txt` is a secondary pip-freeze snapshot, not the dependency source of truth — `pyproject.toml` is authoritative.

---

## Where to go next

- [`llm_extraction_schema.md`](llm_extraction_schema.md) — the small model's exact pipe-delimited output format, the large model's `/summarize_case` payload shape, and the routing/confidence rules in full detail.
- [`llm_server/README.md`](llm_server/README.md) — running the two LLM server processes, mock mode vs. real weights.
