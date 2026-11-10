"""
Security / access-control evaluation — "leak testing" from your synopsis's
evaluation section. This is a DIFFERENT question from eval_retrieval.py:

    eval_retrieval.py asks:      "did we find the right content?"
    eval_access_control.py asks: "did we ever return content this user
                                   is NOT authorized to see?"

A leak (false positive, security sense) is far worse than a low-relevance
result — it's the exact failure mode RBAC exists to prevent. This script
runs real queries through the real retrieval path (same auth + role check +
metadata-filtered cosine similarity search) and checks whether anything
in `forbidden_sources` ever appears in the results, for that user.

It can also flag over-blocking: if `expected_accessible_sources` never show
up for a user who SHOULD be able to see them, that's a sign the sensitivity
tier / allowed_roles metadata (or the user's clearance) is misconfigured —
legitimate access being wrongly denied.

Ground truth here means: you decide, per query + user, which sources must
NEVER appear (forbidden) and/or which sources should appear if relevant
(expected). See test_set_access.example.json.

Usage:
    python -m evaluation.eval_access_control --test-set evaluation/test_set_access.json
    python -m evaluation.eval_access_control --test-set evaluation/test_set_access.json --output leak_report.json
"""

from __future__ import annotations
import argparse
import json
from typing import List, Dict

from query.pipeline import QueryPipeline


def load_test_set(path: str) -> List[Dict]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def evaluate_query(pipeline: QueryPipeline, item: Dict) -> Dict:
    retrieval = pipeline.retrieve(auth_token=item["auth_token"], query=item["query"])
    retrieved_sources = {h["metadata"]["source"] for h in retrieval.hits}

    forbidden = set(item.get("forbidden_sources", []))
    expected = set(item.get("expected_accessible_sources", []))

    leaked_sources = retrieved_sources & forbidden           # SECURITY FAILURE if non-empty
    missing_expected = expected - retrieved_sources           # over-blocking signal (informational)

    return {
        "query": item["query"],
        "auth_token": item["auth_token"],
        "retrieved_sources": sorted(retrieved_sources),
        "forbidden_sources": sorted(forbidden),
        "leaked": sorted(leaked_sources),
        "leak_detected": len(leaked_sources) > 0,
        "expected_accessible_sources": sorted(expected),
        "missing_expected": sorted(missing_expected),
    }


def run_evaluation(test_set_path: str, top_k: int = 5, pipeline: QueryPipeline = None) -> Dict:
    pipeline = pipeline or QueryPipeline(top_k=top_k, enable_query_refinement=False)
    test_set = load_test_set(test_set_path)

    per_query = [evaluate_query(pipeline, item) for item in test_set]

    num_leaked_queries = sum(1 for r in per_query if r["leak_detected"])
    total_leaked_chunks = sum(len(r["leaked"]) for r in per_query)
    num_with_missing_expected = sum(1 for r in per_query if r["missing_expected"])

    return {
        "num_queries": len(per_query),
        "num_leaked_queries": num_leaked_queries,
        "leak_rate": num_leaked_queries / len(per_query) if per_query else 0.0,
        "total_leaked_chunks": total_leaked_chunks,
        "num_queries_with_missing_expected_access": num_with_missing_expected,
        "per_query": per_query,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test-set", required=True, help="Path to a JSON test set (see test_set_access.example.json)")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--output", default=None, help="Optional path to save full JSON results")
    args = parser.parse_args()

    report = run_evaluation(args.test_set, top_k=args.top_k)

    print(f"\nEvaluated {report['num_queries']} quer{'y' if report['num_queries'] == 1 else 'ies'} for access control violations\n")
    for r in report["per_query"]:
        status = "LEAK DETECTED" if r["leak_detected"] else "OK"
        print(f"  [{status}] {r['query']}  (user: {r['auth_token']})")
        if r["leak_detected"]:
            print(f"      Leaked sources: {r['leaked']}")
        if r["missing_expected"]:
            print(f"      Expected-accessible but missing: {r['missing_expected']}")

    print(f"\n--- Summary ---")
    print(f"Leak rate: {report['leak_rate']:.1%}  ({report['num_leaked_queries']}/{report['num_queries']} queries leaked)")
    print(f"Total leaked chunks across all queries: {report['total_leaked_chunks']}")
    print(f"Queries with unexpectedly missing (over-blocked) access: {report['num_queries_with_missing_expected_access']}")

    if report["num_leaked_queries"] > 0:
        print("\n⚠ SECURITY: at least one query returned content the user is not authorized to see. "
              "Check sensitivity_tier / sensitivity_rank / allowed_roles metadata and the conjunctive "
              "predicate in query/access_control.py.build_access_filter.")

    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)
        print(f"\nFull results saved to {args.output}")


if __name__ == "__main__":
    main()
