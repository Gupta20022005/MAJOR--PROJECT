"""
Simplest way to run a query: from the command line.

    export GEMINI_API_KEY=...
    python run_query.py --token demo-token-engineer --q "What is our refund policy?"
"""

import argparse
from query.pipeline import QueryPipeline
from query.auth import AuthenticationError
from logging_setup import setup_logging

setup_logging()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--token", required=True, help="Auth token identifying the user")
    parser.add_argument("--q", required=True, help="The question to ask")
    parser.add_argument("--top-k", type=int, default=5)
    args = parser.parse_args()

    pipeline = QueryPipeline(top_k=args.top_k)
    try:
        result = pipeline.handle_query(auth_token=args.token, query=args.q)
    except AuthenticationError as e:
        print(f"Auth failed: {e}")
        return
    except PermissionError as e:
        print(f"Permission denied: {e}")
        return

    print(f"\n--- Refined Query ---\n{result.refined_query}")

    print("\n--- Answer ---")
    print(result.answer)

    print("\n--- Sources ---")
    for h in result.source_chunks:
        m = h["metadata"]
        print(f"- {m['source']} (page {m.get('page')}, section {m.get('section')}) [{m['sensitivity_tier']}]")


if __name__ == "__main__":
    main()
