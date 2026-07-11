"""Case-level payload construction and the call to the large model's
/summarize_case endpoint. See llm_extraction_schema.md and _llm_client.py.

There is no rule-based fallback here — case-level narrative summarization is
squarely the large model's job. If it's unavailable, generate_case_summary()
returns None and the caller must say so plainly, not silently degrade (unlike
page classification, which does degrade to keyword-based routing).
"""

from __future__ import annotations

from DocProcessorPro import _llm_client

from .embeddings import top_k_by_similarity
from .record_assembly import build_records_reviewed_list

_CONTEXT_BUDGET_CHARS = 20_000
_RAG_QUERY = "clinically significant findings, diagnoses, treatment plan, billing summary"
_EXCERPT_CHARS = 1200
_EXCERPT_CHARS_LOW_RELEVANCE = 400


def build_case_payload(assembled: list[dict], case_label: str) -> dict:
    """Builds the request body for POST /summarize_case.

    If the total merged text across all assembled records exceeds the context
    budget, uses embeddings to rank records by relevance to a fixed query and
    keeps more text from the higher-ranked ones (RAG-style context selection)
    instead of sending everything. Falls back to simple truncation if the
    embedding service is unavailable.
    """
    records_reviewed = build_records_reviewed_list(assembled)
    excerpts = _select_excerpts(assembled)

    return {
        "case_label": case_label,
        "records_reviewed": records_reviewed,
        "assembled_records": [
            {
                "provider_name": rec.get("provider_name"),
                "location_name": rec.get("location_name"),
                "record_type": rec.get("record_type"),
                "service_date_start": rec.get("service_date_min"),
                "service_date_end": rec.get("service_date_max"),
                "page_start": rec.get("page_start"),
                "page_end": rec.get("page_end"),
                "source_stem": rec.get("source_stem"),
                "excerpt": excerpt,
            }
            for rec, excerpt in zip(assembled, excerpts)
        ],
    }


def _select_excerpts(assembled: list[dict]) -> list[str]:
    full_texts = [rec.get("merged_text", "") for rec in assembled]
    if sum(len(t) for t in full_texts) <= _CONTEXT_BUDGET_CHARS:
        return full_texts

    embeddings = _llm_client.embed_texts([_RAG_QUERY, *full_texts])
    if not embeddings:
        return [t[:_EXCERPT_CHARS] for t in full_texts]

    query_embedding, *record_embeddings = embeddings
    candidates = list(enumerate(record_embeddings))
    ranked = top_k_by_similarity(query_embedding, candidates, k=len(candidates))
    rank_by_index = {idx: rank for rank, (idx, _score) in enumerate(ranked)}

    half = max(len(full_texts) // 2, 1)
    excerpts: list[str] = []
    for i, text in enumerate(full_texts):
        rank = rank_by_index.get(i, len(full_texts))
        limit = _EXCERPT_CHARS if rank < half else _EXCERPT_CHARS_LOW_RELEVANCE
        excerpts.append(text[:limit])
    return excerpts


def generate_case_summary(assembled: list[dict], case_label: str) -> dict | None:
    """build_case_payload() -> _llm_client.summarize_case() -> parsed dict.

    Returns None if the large-model service is unavailable or returns
    malformed output — there is no fallback summary to substitute.
    """
    payload = build_case_payload(assembled, case_label)
    return _llm_client.summarize_case(payload)
