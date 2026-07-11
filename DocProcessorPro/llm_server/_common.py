"""Shared FastAPI plumbing for the two local MLX-LM model servers.

Neither server is part of the shipped DocProcessorPro application — they are
standalone processes you run yourself on the machine hosting the models.
Install their dependencies with `pip install -e ".[llm-server]"`.
"""

from __future__ import annotations

import argparse
import hashlib
from typing import Any

from pydantic import BaseModel


class HealthResponse(BaseModel):
    status: str = "ok"


class ExtractPageFieldsRequest(BaseModel):
    text: str
    extraction_method: str = "liteparse"


class ExtractPageFieldsResponse(BaseModel):
    output: str


class EmbedRequest(BaseModel):
    texts: list[str]


class EmbedResponse(BaseModel):
    embeddings: list[list[float]]


class SummarizeCaseRequest(BaseModel):
    case_payload: dict[str, Any]


class SummarizeCaseResponse(BaseModel):
    output: str


def build_arg_parser(default_port: int, model_help: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=default_port)
    parser.add_argument("--model-path", default=None, help=model_help)
    parser.add_argument(
        "--mock",
        action="store_true",
        help="Serve deterministic canned responses without loading any model weights.",
    )
    return parser


def deterministic_vector(text: str, dims: int = 32) -> list[float]:
    """Stable pseudo-embedding derived from a text hash — for --mock mode only.

    Same input always yields the same vector; different inputs yield different
    (but not semantically meaningful) vectors. This is enough to exercise
    cosine-similarity plumbing (provider clustering, RAG ranking, continuation
    detection) end-to-end before real embedding-model weights are wired up.
    """
    digest = hashlib.sha256(text.encode("utf-8")).digest()
    raw = [(digest[i % len(digest)] / 255.0) * 2 - 1 for i in range(dims)]
    norm = sum(v * v for v in raw) ** 0.5 or 1.0
    return [v / norm for v in raw]
