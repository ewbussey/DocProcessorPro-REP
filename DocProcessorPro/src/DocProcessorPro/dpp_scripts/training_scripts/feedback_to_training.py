"""Converts reviewed-page feedback into small-model fine-tuning pairs.

Reads a _feedback.jsonl (written by ReviewDialog._export_feedback()) and the
matching _sidecars/*_llm_fields.jsonl (via
record_assembly.load_case_page_records()) for the LLM fields the feedback
export doesn't carry over (location_name, record_title, continuation flags,
confidence), and emits {"prompt": <page text>, "completion": <ground-truth
pipe-delimited string>} pairs — the same shape as the placeholder line
already in src/data/training.jsonl.

Only approved ("include") pages are converted. Every field in _feedback.jsonl
is already the human-reviewed *effective* value (post user-correction, if
any) — that file's per-field values, not the separate _corrections.jsonl
diff log, are the source of truth here. The one exception is record_type:
the review UI has no correction control for it today (only the coarser
keyword primary_category can be corrected), so it is taken as-is from the
LLM's original prediction.

Run directly:
    python -m DocProcessorPro.dpp_scripts.training_scripts.feedback_to_training <output_dir>
"""

from __future__ import annotations

import json
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[4]
_DEFAULT_TRAINING_PATH = _REPO_ROOT / "src" / "data" / "training.jsonl"

_EXTRACT_FIELD_ORDER = (
    "record_type", "date", "provider", "location", "title",
    "continues_from", "continues_to", "confidence",
)


def feedback_to_training_pairs(feedback_path: "str | Path") -> list[dict]:
    """Reads one _feedback.jsonl file and returns training pairs for its
    approved pages: [{"prompt": ..., "completion": ...}, ...]."""
    return [pair for _key, pair in _iter_pairs_with_keys(Path(feedback_path))]


def export_training_data(
    output_dir: "str | Path",
    training_jsonl_path: "str | Path" = _DEFAULT_TRAINING_PATH,
    append: bool = True,
) -> int:
    """Finds every _feedback/*feedback.jsonl under output_dir (or treats
    output_dir as a single feedback file if it is one), converts each to
    training pairs, and writes new ones to training_jsonl_path.

    De-duplicates by (source_pdf_path, page_num) against a companion
    "<name>.keys.jsonl" file kept alongside training_jsonl_path, so re-running
    after further review only adds newly-approved/newly-corrected pages
    rather than duplicating rows already captured. training_jsonl_path itself
    stays exactly {"prompt", "completion"} per line — no bookkeeping fields
    — so it can be fed straight into a fine-tuning pipeline.

    Returns the number of new pairs written.
    """
    output_dir = Path(output_dir)
    if output_dir.is_file():
        feedback_files = [output_dir]
    else:
        feedback_files = sorted(output_dir.glob("_feedback/*feedback.jsonl"))
    if not feedback_files:
        return 0

    training_jsonl_path = Path(training_jsonl_path)
    keys_path = training_jsonl_path.with_suffix(".keys.jsonl")

    existing_keys: set[tuple[str, int]] = set()
    if append and keys_path.exists():
        for line in keys_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                key = json.loads(line)
                existing_keys.add((key[0], key[1]))
            except (json.JSONDecodeError, IndexError, TypeError):
                continue

    new_pairs: list[tuple[tuple[str, int], dict]] = []
    for feedback_path in feedback_files:
        for key, pair in _iter_pairs_with_keys(feedback_path):
            if key in existing_keys:
                continue
            existing_keys.add(key)
            new_pairs.append((key, pair))

    if not new_pairs:
        return 0

    training_jsonl_path.parent.mkdir(parents=True, exist_ok=True)
    mode = "a" if append and training_jsonl_path.exists() else "w"
    with open(training_jsonl_path, mode, encoding="utf-8") as f:
        for _key, pair in new_pairs:
            f.write(json.dumps(pair, ensure_ascii=False) + "\n")
    with open(keys_path, mode, encoding="utf-8") as f:
        for key, _pair in new_pairs:
            f.write(json.dumps(list(key)) + "\n")

    return len(new_pairs)


def _iter_pairs_with_keys(feedback_path: Path) -> list[tuple[tuple[str, int], dict]]:
    from DocProcessorPro.dpp_scripts.keyword_scanner_scripts.record_assembly import (
        load_case_page_records,
    )

    output_dir = feedback_path.parent.parent  # {output_dir}/_feedback/{name}.jsonl

    llm_by_stem_page: dict[tuple[str, int], dict] = {
        (page["source_stem"], int(page.get("page_num", 0))): page
        for page in load_case_page_records(output_dir)
    }

    results: list[tuple[tuple[str, int], dict]] = []
    for line in feedback_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            continue
        if rec.get("label") != "include":
            continue
        page_text = rec.get("page_text")
        if not page_text:
            continue

        src = rec.get("source_pdf_path", "")
        page_num = int(rec.get("page_num", 0))
        llm_fields = llm_by_stem_page.get((Path(src).stem, page_num), {})

        pair = {"prompt": page_text, "completion": _build_completion(rec, llm_fields)}
        results.append(((src, page_num), pair))

    return results


def _build_completion(feedback_rec: dict, llm_fields: dict) -> str:
    fields = {
        "record_type": feedback_rec.get("record_type") or llm_fields.get("record_type") or "",
        "date": feedback_rec.get("service_date") or llm_fields.get("service_date") or "",
        "provider": (
            feedback_rec.get("provider_name")
            or feedback_rec.get("provider_name_hint")
            or llm_fields.get("provider_name")
            or ""
        ),
        "location": llm_fields.get("location_name") or "",
        "title": llm_fields.get("record_title") or "",
        "continues_from": "yes" if llm_fields.get("continues_from_previous") else "no",
        "continues_to": "yes" if llm_fields.get("continues_to_next") else "no",
        "confidence": _fmt_confidence(
            feedback_rec.get("llm_confidence") or llm_fields.get("confidence")
        ),
    }
    return " | ".join(f"{key}: {fields[key]}" for key in _EXTRACT_FIELD_ORDER)


def _fmt_confidence(value: "float | str | None") -> str:
    if value is None:
        return "1.0"  # human-approved ground truth — treat as fully confident
    try:
        return f"{float(value):.2f}"
    except (TypeError, ValueError):
        return "1.0"


def _main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="Convert reviewed _feedback.jsonl pages into small-model fine-tuning pairs."
    )
    parser.add_argument("output_dir", help="Scan output directory (or a single _feedback.jsonl file)")
    parser.add_argument(
        "--training-path", default=str(_DEFAULT_TRAINING_PATH),
        help=f"Training JSONL to write to (default: {_DEFAULT_TRAINING_PATH})",
    )
    parser.add_argument(
        "--overwrite", action="store_true",
        help="Overwrite training-path instead of appending/deduplicating against it",
    )
    args = parser.parse_args()

    count = export_training_data(args.output_dir, args.training_path, append=not args.overwrite)
    print(f"Wrote {count} new training pair(s) to {args.training_path}")


if __name__ == "__main__":
    _main()
