"""Show exactly which keywords drive the rule-based role and tier scores for a document.

    python diagnose_rules.py data/extracted_markdown/<id>_<name>.md
    python diagnose_rules.py "data/source_docs/some.txt"        (plain text files only)

The pipeline already saves each document's text in data/extracted_markdown/, so point this at that .md file.
"""
import os, sys
from config import RULE_MIN_ROLE_SCORE, RULE_MIN_TIER_SCORE, RULE_TIER_FLOOR_SCORE
from ingestion.rules import ROLE_KEYWORDS, TIER_KEYWORDS, MAX_HITS_PER_TERM, _term_regex


def breakdown(haystack: str, kw_map, min_score):
    low = haystack.lower()
    for cat, terms in kw_map.items():
        rows, total = [], 0.0
        for term, w in terms.items():
            n = len(_term_regex(term).findall(low))
            if n:
                pts = w * min(n, MAX_HITS_PER_TERM)
                total += pts
                rows.append((pts, term, n, w))
        mark = "  <-- qualifies" if total >= min_score else ""
        print(f"  {cat:<13} total={total:5.1f}{mark}")
        for pts, term, n, w in sorted(rows, reverse=True):
            print(f"      {term!r:<26} x{n} (weight {w}, counted up to {MAX_HITS_PER_TERM}) = {pts:.0f}")


if __name__ == "__main__":
    path = sys.argv[1]
    name = os.path.basename(path)
    text = open(path, encoding="utf-8", errors="ignore").read()
    hay = f"{name.replace('_', ' ')}\n{text}"
    print(f"\nTIERS  (a tier needs >= {RULE_MIN_TIER_SCORE}; >= {RULE_TIER_FLOOR_SCORE} counts as strong evidence)")
    breakdown(hay, TIER_KEYWORDS, RULE_MIN_TIER_SCORE)
    print(f"\nROLES  (a role needs >= {RULE_MIN_ROLE_SCORE})")
    breakdown(hay, ROLE_KEYWORDS, RULE_MIN_ROLE_SCORE)
