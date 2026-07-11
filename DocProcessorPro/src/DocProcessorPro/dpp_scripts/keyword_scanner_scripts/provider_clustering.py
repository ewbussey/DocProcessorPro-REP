"""Provider-identity clustering for splitting matched pages by provider.

Scan-time/ephemeral — used only to decide PDF split boundaries (see
keyword_scanner_codebase.py's _cluster_and_embed()/_scan_one()). Independent
of record_assembly.py's own (unchanged) provider grouping by raw name
string, which serves a different purpose (grouping records for a case
summary, after review).
"""

from __future__ import annotations

from .embeddings import cosine_similarity

_PROVIDER_CLUSTER_THRESHOLD = 0.86


def cluster_providers(page_records: list[dict], stem: str) -> dict[int, str]:
    """page_records: [{"page_num", "provider_npi", "provider_name",
    "identity_embedding"}, ...] (identity_embedding may be None).

    Returns page_num -> provider_key. Precedence mirrors
    ReviewDialog._provider_key()'s established GUI-side scheme (NPI >
    assigned/corrected name > raw hint > fallback, in _review_dialog.py),
    adapted for scan time where no human corrections exist yet:

      1. Exact NPI match -> same cluster, no embedding needed (free, exact).
      2. Embedding centroid similarity >= threshold -> join that cluster
         (centroid-based greedy clustering — cheaper than _dedup.py's
         complete-linkage-against-all-members, and text embeddings cluster
         more predictably per-provider than page-layout hashes do).
      3. No embedding available -> case-insensitive string equality on
         provider_name (today's degraded behavior when the service is down).
      4. Neither NPI, embedding, nor name -> its own singleton cluster, keyed
         by page number so unrelated unknown-provider pages never merge.
    """
    provider_key_by_page: dict[int, str] = {}

    # Pass 1: NPI is an unconditional, free same-cluster join.
    remaining: list[dict] = []
    for rec in page_records:
        npi = rec.get("provider_npi")
        if npi:
            provider_key_by_page[rec["page_num"]] = npi
        else:
            remaining.append(rec)

    # Pass 2: embedding-based centroid clustering (or string-equality
    # fallback) for everything without an NPI.
    clusters: list[dict] = []  # [{"centroid", "sum", "count", "key"}]
    string_clusters: dict[str, str] = {}

    for rec in remaining:
        embedding = rec.get("identity_embedding")
        name = (rec.get("provider_name") or "").strip()

        if embedding:
            best_idx, best_sim = None, -1.0
            for idx, cluster in enumerate(clusters):
                sim = cosine_similarity(embedding, cluster["centroid"])
                if sim > best_sim:
                    best_idx, best_sim = idx, sim
            if best_idx is not None and best_sim >= _PROVIDER_CLUSTER_THRESHOLD:
                cluster = clusters[best_idx]
                cluster["sum"] = [a + b for a, b in zip(cluster["sum"], embedding)]
                cluster["count"] += 1
                cluster["centroid"] = [v / cluster["count"] for v in cluster["sum"]]
                provider_key_by_page[rec["page_num"]] = cluster["key"]
            else:
                key = name or f"{stem}__unresolved_p{rec['page_num'] + 1}"
                clusters.append({
                    "centroid": list(embedding),
                    "sum": list(embedding),
                    "count": 1,
                    "key": key,
                })
                provider_key_by_page[rec["page_num"]] = key
        elif name:
            key = string_clusters.setdefault(name.lower(), name)
            provider_key_by_page[rec["page_num"]] = key
        else:
            provider_key_by_page[rec["page_num"]] = f"{stem}__unresolved_p{rec['page_num'] + 1}"

    return provider_key_by_page
