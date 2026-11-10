"""
Hybrid retrieval: dense (cosine) + lexical (BM25) search over the SAME
authorized candidate set, fused by Reciprocal Rank Fusion:

    RRF(d) = sum over methods r of  1 / (k + rank_r(d))

Ranks, not raw scores, are fused — cosine similarity and BM25 scores live on
incomparable scales. A chunk two independent methods agree on outranks one
only a single method favors. Dense search handles paraphrase; BM25 catches
exact-match cases (clause numbers, identifiers, acronyms) that embeddings
smooth over.
"""

from __future__ import annotations
import logging
import re
from typing import List, Dict, Optional

from config import DENSE_CANDIDATE_K, BM25_CANDIDATE_K, RRF_K
from ingestion.vector_store import VectorStore

logger = logging.getLogger("query.hybrid_retrieval")


def _tokenize(text: str) -> List[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


def _bm25_search(vector_store: VectorStore, where: Dict, query_text: str, k: int) -> List[Dict]:
    from rank_bm25 import BM25Okapi  # ImportError propagates; caller decides the fallback

    candidates = vector_store.get_by_filter(where)  # the FULL authorized set
    if not candidates:
        return []
    bm25 = BM25Okapi([_tokenize(c["text"]) for c in candidates])
    scores = bm25.get_scores(_tokenize(query_text))
    ranked = sorted(zip(candidates, scores), key=lambda p: p[1], reverse=True)
    hits = []
    for chunk, score in ranked[:k]:
        hit = {key: v for key, v in chunk.items() if key != "embedding"}
        hit["bm25_score"] = float(score)
        hits.append(hit)
    return hits


def reciprocal_rank_fusion(ranked_lists: List[List[Dict]], k: int = RRF_K) -> List[Dict]:
    """Sum 1/(k+rank) per list a chunk appears in. A chunk in only one list
    still contributes from that list alone, so a BM25-only or dense-only
    match can still surface."""
    scores: Dict[str, float] = {}
    hit_by_id: Dict[str, Dict] = {}
    for ranked_list in ranked_lists:
        for rank, hit in enumerate(ranked_list, start=1):
            cid = hit["id"]
            scores[cid] = scores.get(cid, 0.0) + 1.0 / (k + rank)
            if cid not in hit_by_id or hit.get("similarity") is not None:
                hit_by_id[cid] = hit
    out = []
    for cid, s in sorted(scores.items(), key=lambda p: p[1], reverse=True):
        h = dict(hit_by_id[cid])
        h["rrf_score"] = s
        out.append(h)
    return out


def hybrid_retrieve(
    vector_store: VectorStore,
    query_embedding: List[float],
    query_text: str,
    where: Dict,
    dense_k: int = DENSE_CANDIDATE_K,
    bm25_k: int = BM25_CANDIDATE_K,
    shortlist_size: Optional[int] = None,
) -> List[Dict]:
    dense_hits = vector_store.query(query_embedding, top_k=dense_k, where=where)
    bm25_hits = _bm25_search(vector_store, where, query_text, bm25_k)
    fused = reciprocal_rank_fusion([dense_hits, bm25_hits])
    return fused[:shortlist_size] if shortlist_size else fused
