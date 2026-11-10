"""
Shared metric computation for retrieval and access-control evaluation.
Plain set-based precision/recall/F1 — no external eval library needed.
"""

from __future__ import annotations
from typing import Set, Dict


def precision_recall_f1(retrieved: Set[str], relevant: Set[str]) -> Dict[str, float]:
    """
    Standard set-based Precision/Recall/F1:
        precision = |retrieved ∩ relevant| / |retrieved|   (of what we returned, how much was right)
        recall    = |retrieved ∩ relevant| / |relevant|    (of what mattered, how much did we return)
        f1        = harmonic mean of the two

    Edge cases: an empty `retrieved` set gives precision=0.0 (nothing returned,
    nothing right by definition here — some conventions use 1.0 for "vacuously
    correct"; we use 0.0 since an empty result set is never useful in a RAG
    context). An empty `relevant` set gives recall=1.0 (nothing to find, so
    trivially all of it was found) — only relevant when a query genuinely has
    no ground-truth matches in the corpus.
    """
    true_positives = retrieved & relevant

    precision = len(true_positives) / len(retrieved) if retrieved else 0.0
    recall = len(true_positives) / len(relevant) if relevant else 1.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "true_positives": sorted(true_positives),
        "false_positives": sorted(retrieved - relevant),
        "false_negatives": sorted(relevant - retrieved),
    }


def macro_average(per_query_metrics, keys=("precision", "recall", "f1")) -> Dict[str, float]:
    """Simple average across queries (each query weighted equally, regardless
    of how many relevant/retrieved items it has)."""
    n = len(per_query_metrics)
    if n == 0:
        return {k: 0.0 for k in keys}
    return {k: sum(m[k] for m in per_query_metrics) / n for k in keys}
