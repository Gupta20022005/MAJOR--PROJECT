"""
Evaluates retrieval quality — Precision@k / Recall@k / F1 — against a
hand-labeled test set of (query, relevant_items) pairs. This is the
"retrieval accuracy" experiment your synopsis's evaluation section calls for.

Ground truth can't be generated automatically: for each test query, you
decide which chunks or source documents are actually relevant, by reading
your own documents. See test_set.example.json for the format — copy it,
rename to test_set.json (or anything), and fill in real queries/answers
based on your ingested documents.

Two ways to specify relevance per test item:
    "relevant_sources":    ["policy.pdf", "handbook.pdf"]   — document-level, easier to label
    "relevant_chunk_ids":  ["<doc_id>::chunk-4", ...]         — chunk-level, more precise

Get exact chunk_ids by running:  python inspect_doc.py --file <path> --stored

Usage:
    python -m evaluation.eval_retrieval --test-set evaluation/test_set.json
    python -m evaluation.eval_retrieval --test-set evaluation/test_set.json --top-k 3 --output results.json
"""

from __future__ import annotations
import argparse
import json
from typing import List, Dict

from query.pipeline import QueryPipeline
from evaluation.metrics import precision_recall_f1, macro_average


def load_test_set(path: str) -> List[Dict]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def evaluate_query(pipeline: QueryPipeline, item: Dict) -> Dict:
    retrieval = pipeline.retrieve(auth_token=item["auth_token"], query=item["query"])
    hits = retrieval.hits

    if "relevant_chunk_ids" in item:
        retrieved = {h["id"] for h in hits}
        relevant = set(item["relevant_chunk_ids"])
        granularity = "chunk"
    elif "relevant_sources" in item:
        retrieved = {h["metadata"]["source"] for h in hits}
        relevant = set(item["relevant_sources"])
        granularity = "source"
    else:
        raise ValueError(
            f"Test item for query '{item['query']}' needs either "
            f"'relevant_chunk_ids' or 'relevant_sources'."
        )

    metrics = precision_recall_f1(retrieved, relevant)
    return {
        "query": item["query"],
        "granularity": granularity,
        "retrieved": sorted(retrieved),
        "relevant": sorted(relevant),
        **metrics,
    }


def run_evaluation(test_set_path: str, top_k: int = 5, pipeline: QueryPipeline = None) -> Dict:
    # enable_query_refinement=False: retrieval evaluation should measure the
    # retrieval step itself, deterministically, without an LLM call (and its
    # cost/latency/variability) in the loop on every single test item.
    pipeline = pipeline or QueryPipeline(top_k=top_k, enable_query_refinement=False)
    test_set = load_test_set(test_set_path)

    per_query = [evaluate_query(pipeline, item) for item in test_set]
    averages = macro_average(per_query)

    return {
        "num_queries": len(per_query),
        "top_k": top_k,
        "avg_precision": averages["precision"],
        "avg_recall": averages["recall"],
        "avg_f1": averages["f1"],
        "per_query": per_query,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test-set", required=True, help="Path to a JSON test set (see test_set.example.json)")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--output", default=None, help="Optional path to save full JSON results")
    args = parser.parse_args()

    report = run_evaluation(args.test_set, top_k=args.top_k)

    print(f"\nEvaluated {report['num_queries']} quer{'y' if report['num_queries'] == 1 else 'ies'} at top_k={report['top_k']}\n")
    for r in report["per_query"]:
        print(f"  P={r['precision']:.2f}  R={r['recall']:.2f}  F1={r['f1']:.2f}  | {r['query']}")
        if r["false_positives"]:
            print(f"      false positives (retrieved but not relevant): {r['false_positives']}")
        if r["false_negatives"]:
            print(f"      false negatives (relevant but missed): {r['false_negatives']}")

    print(f"\n--- Averages (top_k={report['top_k']}) ---")
    print(f"Precision: {report['avg_precision']:.3f}")
    print(f"Recall:    {report['avg_recall']:.3f}")
    print(f"F1:        {report['avg_f1']:.3f}")

    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)
        print(f"\nFull results saved to {args.output}")


if __name__ == "__main__":
    main()
