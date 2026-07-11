"""Shared embedding math: cosine similarity and top-k ranking.

Brute-force numpy over in-memory vectors — sufficient at single-case/
single-batch scale (dozens to low-thousands of pages per run). No vector
database is used or needed here; revisit only if that scale assumption
stops holding.
"""

from __future__ import annotations

import numpy as np


def cosine_similarity(a: list[float], b: list[float]) -> float:
    va, vb = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    denom = np.linalg.norm(va) * np.linalg.norm(vb)
    if denom == 0:
        return 0.0
    return float(np.dot(va, vb) / denom)


def top_k_by_similarity(
    query_embedding: list[float],
    candidates: list[tuple[int, list[float]]],
    k: int,
) -> list[tuple[int, float]]:
    """candidates: list of (id, embedding). Returns the k highest-scoring
    (id, similarity) pairs, descending by similarity."""
    if not candidates:
        return []
    query = np.asarray(query_embedding, dtype=np.float64)
    query_norm = np.linalg.norm(query)
    if query_norm == 0:
        return []

    ids = [c[0] for c in candidates]
    matrix = np.asarray([c[1] for c in candidates], dtype=np.float64)
    norms = np.linalg.norm(matrix, axis=1)
    norms[norms == 0] = 1.0  # avoid divide-by-zero; those rows score 0 anyway
    sims = (matrix @ query) / (norms * query_norm)

    order = np.argsort(-sims)[:k]
    return [(ids[i], float(sims[i])) for i in order]


def embedding_continuation_score(embedding_a: list[float], embedding_b: list[float]) -> float:
    """Similarity between two consecutive pages' embeddings — a fallback
    continuation signal for when the LLM's continues_from/continues_to flags
    are absent or low-confidence. Higher means more likely the same document
    continues from one page to the next."""
    return cosine_similarity(embedding_a, embedding_b)
