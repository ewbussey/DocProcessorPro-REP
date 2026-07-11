"""Local inference server for the large case-analysis/summary model (Qwen3.6 27B 4-bit).

Run in --mock mode (no model weights needed) to develop/test the rest of the
pipeline, then point --model-path at real weights once ready:

    python -m llm_server.large_model_server --mock
    python -m llm_server.large_model_server --model-path <path-to-qwen3.6-27b-4bit>
"""

from __future__ import annotations

import json
import logging

import uvicorn
from fastapi import FastAPI

from ._common import (
    HealthResponse,
    SummarizeCaseRequest,
    SummarizeCaseResponse,
    build_arg_parser,
)

log = logging.getLogger(__name__)

app = FastAPI(title="DocProcessorPro large-model server")

_state: dict = {"mock": True, "generator": None, "tokenizer": None}


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse()


@app.post("/summarize_case", response_model=SummarizeCaseResponse)
def summarize_case(req: SummarizeCaseRequest) -> SummarizeCaseResponse:
    if _state["mock"]:
        output = _mock_summarize(req.case_payload)
    else:
        output = _generate_summarize(req.case_payload)
    return SummarizeCaseResponse(output=output)


def _mock_summarize(case_payload: dict) -> str:
    records_reviewed = case_payload.get("records_reviewed", [])
    assembled = case_payload.get("assembled_records", [])

    by_provider: dict[str, list[dict]] = {}
    for rec in assembled:
        key = rec.get("provider_name") or "Unknown Provider"
        by_provider.setdefault(key, []).append(rec)

    provider_chronology = []
    for provider_name, recs in by_provider.items():
        dates = sorted(
            d for r in recs
            for d in (r.get("service_date_start"), r.get("service_date_end"))
            if d
        )
        provider_chronology.append({
            "provider_name": provider_name,
            "location_name": recs[0].get("location_name"),
            "date_range": f"{dates[0]} – {dates[-1]}" if dates else "",
            "entries": [
                {
                    "date": r.get("service_date_start") or "",
                    "record_type": r.get("record_type") or "",
                    "summary": f"(mock) {r.get('record_type', 'record')} — pages {r.get('page_start')}-{r.get('page_end')}",
                }
                for r in recs
            ],
        })

    narrative = (
        f"(mock summary) Reviewed {len(assembled)} record group(s) across "
        f"{len(by_provider)} provider(s) for case '{case_payload.get('case_label', 'unknown')}'."
    )

    return json.dumps({
        "records_reviewed": records_reviewed,
        "provider_chronology": provider_chronology,
        "narrative_summary": narrative,
    })


def _generate_summarize(case_payload: dict) -> str:
    generator, tokenizer = _state["generator"], _state["tokenizer"]
    if generator is None or tokenizer is None:
        raise RuntimeError("Model not loaded — pass --model-path, or run with --mock.")
    from mlx_lm import generate

    from ._prompts import build_case_summary_prompt

    prompt = build_case_summary_prompt(case_payload)
    return generate(generator, tokenizer, prompt=prompt, max_tokens=4000, verbose=False).strip()


def main() -> None:
    parser = build_arg_parser(
        default_port=8766,
        model_help="Path or HF repo id of Qwen3.6-27B-4bit",
    )
    args = parser.parse_args()

    _state["mock"] = args.mock
    if not args.mock:
        if not args.model_path:
            raise SystemExit("--model-path is required unless --mock is set")
        from mlx_lm import load

        log.info("Loading large model from %s ... (this can take a while)", args.model_path)
        _state["generator"], _state["tokenizer"] = load(args.model_path)

    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
