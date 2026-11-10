"""
Cross-encoder reranking of the fused hybrid-retrieval shortlist.

Runs AFTER the authorization filter and RRF fusion, so no compute is spent
on chunks the user can't see. A cross-encoder scores each (query, chunk)
pair jointly — finer than a bi-encoder, but too slow for a whole corpus,
hence it only sees the small shortlist. It re-orders; it doesn't recall.
"""

from __future__ import annotations
import logging
from typing import List, Dict

from config import CROSS_ENCODER_MODEL

logger = logging.getLogger("query.reranker")
_model = None


def _get_model():
    global _model
    if _model is None:
        from sentence_transformers import CrossEncoder  # ImportError if unavailable
        _model = CrossEncoder(CROSS_ENCODER_MODEL)
    return _model


def rerank(query_text: str, hits: List[Dict], top_k: int) -> List[Dict]:
    if not hits:
        return []
    scores = _get_model().predict([(query_text, h["text"]) for h in hits])
    ranked = sorted(zip(hits, scores), key=lambda p: p[1], reverse=True)
    out = []
    for hit, score in ranked[:top_k]:
        h = dict(hit)
        h["rerank_score"] = float(score)
        out.append(h)
    return out
