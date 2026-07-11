"""Multi-page record assembly and the "List of Records Reviewed" text spec.

Implements the design documented in llm_extraction_schema.md ("Multi-Page
Record Assembly" and "List of Records Reviewed" sections) as real Python.

assemble_records() takes a plain list[dict], not the PageMatch/PageExclusion
dataclasses, so it works unchanged against both the sidecar-based source used
today (_page_texts.jsonl + _llm_fields.jsonl, via load_case_page_records())
and a future inline-classification source (PageMatch.record_type etc,
converted to the same dict shape).
"""

from __future__ import annotations

import json
from pathlib import Path

from .embeddings import embedding_continuation_score

# Tighter than provider_clustering's 0.86 threshold — "these two pages are
# one continuous document" is a stronger claim than "same provider"; pages
# from the same provider on different visits should cluster together but
# must not be judged continuous.
_CONTINUATION_SIM_THRESHOLD = 0.90


def load_case_page_records(output_dir: "str | Path") -> list[dict]:
    """Merge every _sidecars/{stem}_page_texts.jsonl with its sibling
    {stem}_llm_fields.jsonl and {stem}_embeddings.jsonl in output_dir into
    one flat list of per-page dicts.

    Each dict has: source_stem, page_num (1-indexed), text, extraction_method,
    plus whichever LLM fields exist for that page (record_type, service_date,
    provider_name, location_name, record_title, continues_from_previous,
    continues_to_next, confidence) — merged in verbatim when present, absent
    (not defaulted) when the LLM never classified that page — and an
    "embedding" key when a content embedding was captured for that page
    (also absent, not defaulted, when unavailable).
    """
    out_dir = Path(output_dir)
    sidecars_dir = out_dir / "_sidecars"
    if not sidecars_dir.exists():
        return []

    records: list[dict] = []
    for text_sidecar in sorted(sidecars_dir.glob("*_page_texts.jsonl")):
        stem = text_sidecar.name[: -len("_page_texts.jsonl")]
        fields_sidecar = sidecars_dir / f"{stem}_llm_fields.jsonl"
        embeddings_sidecar = sidecars_dir / f"{stem}_embeddings.jsonl"

        llm_by_page: dict[int, dict] = {}
        if fields_sidecar.exists():
            for line in fields_sidecar.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                    llm_by_page[int(rec["page_num"])] = rec
                except (KeyError, ValueError, json.JSONDecodeError):
                    continue

        embedding_by_page: dict[int, list[float]] = {}
        if embeddings_sidecar.exists():
            for line in embeddings_sidecar.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                    embedding_by_page[int(rec["page_num"])] = rec["embedding"]
                except (KeyError, ValueError, json.JSONDecodeError):
                    continue

        for line in text_sidecar.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                page = json.loads(line)
            except json.JSONDecodeError:
                continue
            page_num = int(page.get("page_num", 0))
            merged = {"source_stem": stem, **page}
            merged.update(llm_by_page.get(page_num, {}))
            if page_num in embedding_by_page:
                merged["embedding"] = embedding_by_page[page_num]
            records.append(merged)

    return records


def assemble_records(page_records: list[dict]) -> list[dict]:
    """Group pages into multi-page records:

    group by (source_stem, record_type, location_name, provider_name)
    within group: sort by page_num
    merge consecutive pages where continues_to_next=True / continues_from_previous=True

    Pages with no LLM continuation signal fall back to embedding similarity
    (see _continues()) when both pages have a captured content embedding;
    otherwise each becomes its own single-page record — there's nothing left
    to decide a merge on.

    Returns one dict per assembled record: {source_stem, record_type,
    provider_name, location_name, page_start, page_end, service_date_min,
    service_date_max, merged_text}.
    """
    groups: dict[tuple, list[dict]] = {}
    for page in page_records:
        key = (
            page.get("source_stem"),
            page.get("record_type"),
            page.get("location_name"),
            page.get("provider_name"),
        )
        groups.setdefault(key, []).append(page)

    assembled: list[dict] = []
    for (source_stem, record_type, location_name, provider_name), pages in groups.items():
        pages.sort(key=lambda p: int(p.get("page_num", 0)))

        run: list[dict] = []
        for page in pages:
            if run and not _continues(run[-1], page):
                assembled.append(
                    _merge_run(source_stem, record_type, location_name, provider_name, run)
                )
                run = []
            run.append(page)
        if run:
            assembled.append(
                _merge_run(source_stem, record_type, location_name, provider_name, run)
            )

    assembled.sort(key=lambda r: (r.get("source_stem") or "", r["page_start"]))
    return assembled


def _continues(prev: dict, page: dict) -> bool:
    if bool(prev.get("continues_to_next")) or bool(page.get("continues_from_previous")):
        return True
    prev_embedding, page_embedding = prev.get("embedding"), page.get("embedding")
    if prev_embedding and page_embedding:
        return embedding_continuation_score(prev_embedding, page_embedding) >= _CONTINUATION_SIM_THRESHOLD
    return False


def _merge_run(
    source_stem: "str | None",
    record_type: "str | None",
    location_name: "str | None",
    provider_name: "str | None",
    run: list[dict],
) -> dict:
    dates = sorted(d for p in run if (d := p.get("service_date")))
    return {
        "source_stem": source_stem,
        "record_type": record_type,
        "provider_name": provider_name,
        "location_name": location_name,
        "page_start": int(run[0].get("page_num", 0)),
        "page_end": int(run[-1].get("page_num", 0)),
        "service_date_min": dates[0] if dates else None,
        "service_date_max": dates[-1] if dates else None,
        "merged_text": "\n\n".join(p.get("text", "") for p in run),
    }


def build_records_reviewed_list(assembled: list[dict]) -> list[str]:
    """"List of Records Reviewed" per llm_extraction_schema.md: group assembled
    records by (location_name, provider_name), compute the overall date range,
    and format one numbered line per group, e.g.:
    "1. Medical and Billing Records from One Main Physical Therapy, dated
    02/04/2020 – 05/15/2026;"
    """
    groups: dict[tuple, list[dict]] = {}
    for rec in assembled:
        key = (rec.get("location_name"), rec.get("provider_name"))
        groups.setdefault(key, []).append(rec)

    lines: list[str] = []
    ordered_groups = sorted(
        groups.items(), key=lambda kv: (kv[0][1] or "", kv[0][0] or "")
    )
    for i, ((location_name, provider_name), recs) in enumerate(ordered_groups, start=1):
        dates = sorted(
            d
            for r in recs
            for d in (r.get("service_date_min"), r.get("service_date_max"))
            if d
        )
        date_range = f"{_fmt_date(dates[0])} – {_fmt_date(dates[-1])}" if dates else "date unknown"
        who = location_name or provider_name or "Unknown Provider"
        record_types = sorted({r.get("record_type") for r in recs if r.get("record_type")})
        description = _describe_record_types(record_types)
        lines.append(f"{i}. {description} from {who}, dated {date_range};")

    return lines


def _fmt_date(iso_date: str) -> str:
    """YYYY-MM-DD -> MM/DD/YYYY, best-effort — falls back to the raw string."""
    parts = iso_date.split("-")
    if len(parts) == 3:
        year, month, day = parts
        return f"{month}/{day}/{year}"
    return iso_date


def _describe_record_types(record_types: list[str]) -> str:
    if not record_types:
        return "Records"
    billing_types = {"bill", "billing_affidavit", "pharmacy"}
    has_billing = any(rt in billing_types for rt in record_types)
    has_other = any(rt not in billing_types for rt in record_types)
    if has_billing and has_other:
        return "Medical and Billing Records"
    if has_billing:
        return "Billing Records"
    return "Medical Records"
