"""Local inference server for the small page-classification/embedding model.

Run in --mock mode (no model weights needed) to develop/test the rest of the
pipeline, then point --model-path / --embed-model-path at real weights once
the small model is chosen:

    python -m llm_server.small_model_server --mock
    python -m llm_server.small_model_server --model-path <path> --embed-model-path <path>
"""

from __future__ import annotations

import hashlib
import logging

import uvicorn
from fastapi import FastAPI

from ._common import (
    EmbedRequest,
    EmbedResponse,
    ExtractPageFieldsRequest,
    ExtractPageFieldsResponse,
    HealthResponse,
    build_arg_parser,
    deterministic_vector,
)

log = logging.getLogger(__name__)

app = FastAPI(title="DocProcessorPro small-model server")

_state: dict = {"mock": True, "generator": None, "tokenizer": None, "embedder": None}

_MOCK_RECORD_TYPES = (
    "office_visit", "therapy_non_psych", "imaging", "bill", "legal_document",
)


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse()


@app.post("/extract_page_fields", response_model=ExtractPageFieldsResponse)
def extract_page_fields(req: ExtractPageFieldsRequest) -> ExtractPageFieldsResponse:
    if _state["mock"]:
        return ExtractPageFieldsResponse(output=_mock_extract(req.text, req.extraction_method))
    return ExtractPageFieldsResponse(output=_generate_extract(req.text, req.extraction_method))


@app.post("/embed", response_model=EmbedResponse)
def embed(req: EmbedRequest) -> EmbedResponse:
    if _state["mock"]:
        return EmbedResponse(embeddings=[deterministic_vector(t) for t in req.texts])
    return EmbedResponse(embeddings=_generate_embeddings(req.texts))


def _mock_extract(text: str, extraction_method: str) -> str:
    idx = int(hashlib.sha256(text.encode("utf-8")).hexdigest(), 16) % len(_MOCK_RECORD_TYPES)
    record_type = _MOCK_RECORD_TYPES[idx]
    confidence = 0.75 if extraction_method == "liteparse_ocr" else 0.92
    fields = {
        "record_type": record_type,
        "date": "2024-01-15",
        "provider": "Mock Provider Clinic",
        "location": "Mock Main Street Location",
        "title": "",
        "continues_from": "no",
        "continues_to": "no",
        "confidence": f"{confidence:.2f}",
    }
    return " | ".join(f"{k}: {v}" for k, v in fields.items())


def _generate_extract(text: str, extraction_method: str) -> str:
    generator, tokenizer = _state["generator"], _state["tokenizer"]
    if generator is None or tokenizer is None:
        raise RuntimeError("Model not loaded — pass --model-path, or run with --mock.")
    from mlx_lm import generate

    from ._prompts import build_extract_prompt

    prompt = build_extract_prompt(text, extraction_method)
    return generate(generator, tokenizer, prompt=prompt, max_tokens=200, verbose=False).strip()


def _generate_embeddings(texts: list[str]) -> list[list[float]]:
    embedder = _state["embedder"]
    if embedder is None:
        raise RuntimeError("Embedding model not loaded — pass --embed-model-path, or run with --mock.")
    return embedder.encode(texts)


def _load_embedder(model_path: str):
    # Placeholder integration point: swap for whichever embedding runtime the
    # chosen small embedding model needs (mlx_embeddings, sentence-transformers,
    # etc). Verify this against the real API once the model is picked.
    from mlx_embeddings import load as load_embedder

    return load_embedder(model_path)


def main() -> None:
    parser = build_arg_parser(
        default_port=8765,
        model_help="Path or HF repo id of the small generative (page-classification) model",
    )
    parser.add_argument(
        "--embed-model-path", default=None,
        help="Path or HF repo id of the dedicated embedding model",
    )
    args = parser.parse_args()

    _state["mock"] = args.mock
    if not args.mock:
        if not args.model_path:
            raise SystemExit("--model-path is required unless --mock is set")
        from mlx_lm import load

        log.info("Loading small generative model from %s ...", args.model_path)
        _state["generator"], _state["tokenizer"] = load(args.model_path)

        if args.embed_model_path:
            log.info("Loading embedding model from %s ...", args.embed_model_path)
            _state["embedder"] = _load_embedder(args.embed_model_path)
        else:
            log.warning("No --embed-model-path given — /embed will error until one is configured.")

    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
