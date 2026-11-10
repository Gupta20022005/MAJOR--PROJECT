"""Score the classifier against labels YOU have confirmed, and track every run.

    # 1. one-off: write a labels file pre-filled from the current index (you then correct it)
    python -m evaluation.eval_classification --make-template

    # 2. after each (re)ingest: score it and append the run to the history
    python -m evaluation.eval_classification --label "after prompt v3"

Labels live in evaluation/classification_truth.csv:
    source,roles,tier,confirmed,note
    HD-MKT-BRO-7000-2026_....pdf,all,public,yes,
    MC-REN-2026-....docx,legal|sales,confidential,yes,
`roles` is a |-separated list from {finance,hr,legal,engineering,sales,executive} or the single word `all`.
Only rows with confirmed=yes are scored, so a guessed label can never inflate or deflate the score.

Reading the result (access-control view):
  LEAK RISK       tier LOWER than truth, or roles the document should NOT be visible to  (unsafe)
  OVER-RESTRICTED tier HIGHER than truth, or a rightful role locked out                  (safe but costly)
"""
import argparse, csv, os, sys
from collections import Counter
from datetime import datetime

from config import ORGANIZATIONAL_ROLES, ALL_ROLES_PSEUDO_ROLE, SENSITIVITY_TIERS, TIER_RANK, BASE_DIR

EVAL_DIR = os.path.join(BASE_DIR, "evaluation")
TRUTH_PATH = os.path.join(EVAL_DIR, "classification_truth.csv")
HISTORY_PATH = os.path.join(EVAL_DIR, "classification_history.csv")


# ----------------------------------------------------------------- pure scoring (no database needed)
def effective_audience(roles):
    """'all' means every role."""
    roles = set(roles)
    return set(ORGANIZATIONAL_ROLES) if ALL_ROLES_PSEUDO_ROLE in roles else roles


def parse_roles(cell: str):
    roles = [r.strip().lower() for r in cell.replace(",", "|").split("|") if r.strip()]
    bad = [r for r in roles if r not in ORGANIZATIONAL_ROLES + [ALL_ROLES_PSEUDO_ROLE]]
    if bad or not roles:
        raise ValueError(f"bad roles {cell!r}")
    return roles


def judge(truth_roles, truth_tier, pred_roles, pred_tier):
    t_aud, p_aud = effective_audience(truth_roles), effective_audience(pred_roles)
    extra, missing = sorted(p_aud - t_aud), sorted(t_aud - p_aud)
    tier_delta = TIER_RANK[pred_tier] - TIER_RANK[truth_tier]
    leak = tier_delta < 0 or bool(extra)
    over = tier_delta > 0 or bool(missing)
    status = "OK" if not (leak or over) else "MIXED" if (leak and over) else "LEAK RISK" if leak else "OVER-RESTRICTED"
    return dict(status=status, tier_delta=tier_delta, extra=extra, missing=missing,
                tier_ok=tier_delta == 0, roles_ok=not extra and not missing)


def score(truth: dict, preds: dict):
    rows, tp = [], dict(extra=0, missing=0, correct=0)
    for src, (t_roles, t_tier) in sorted(truth.items()):
        if src not in preds:
            rows.append(dict(source=src, status="NOT INDEXED")); continue
        p = preds[src]
        j = judge(t_roles, t_tier, p["roles"], p["tier"])
        j.update(source=src, truth=f"{'|'.join(t_roles)} / {t_tier}", got=f"{'|'.join(p['roles'])} / {p['tier']}",
                 review=p.get("needs_review", False), method=p.get("method", ""))
        tp["extra"] += len(j["extra"]); tp["missing"] += len(j["missing"])
        tp["correct"] += len(effective_audience(t_roles) & effective_audience(p["roles"]))
        rows.append(j)
    scored = [r for r in rows if r["status"] != "NOT INDEXED"]
    n = len(scored)
    pr = tp["correct"] / (tp["correct"] + tp["extra"]) if tp["correct"] + tp["extra"] else 1.0
    rc = tp["correct"] / (tp["correct"] + tp["missing"]) if tp["correct"] + tp["missing"] else 1.0
    m = dict(n=n, fully_correct=sum(r["status"] == "OK" for r in scored),
             tier_exact=sum(r["tier_ok"] for r in scored), roles_exact=sum(r["roles_ok"] for r in scored),
             tier_leak=sum(r["tier_delta"] < 0 for r in scored), tier_over=sum(r["tier_delta"] > 0 for r in scored),
             docs_leak_risk=sum(r["status"] in ("LEAK RISK", "MIXED") for r in scored),
             role_precision=round(pr, 3), role_recall=round(rc, 3),
             flagged_review=sum(bool(r["review"]) for r in scored),
             not_indexed=len(rows) - n)
    return rows, m


# ----------------------------------------------------------------- IO
def load_predictions():
    """Document-level classification per source, read from the vector store."""
    from ingestion.vector_store import VectorStore
    got = VectorStore()._collection.get(include=["metadatas"])
    per_src = {}
    for md in got["metadatas"]:
        per_src.setdefault(md["source"], []).append(md)
    preds = {}
    for src, mds in per_src.items():
        base = [m for m in mds if not m.get("chunk_override")] or mds     # ignore chunk-level raises
        key = Counter((m["allowed_roles"], m["sensitivity_tier"]) for m in base).most_common(1)[0][0]
        first = next(m for m in base if (m["allowed_roles"], m["sensitivity_tier"]) == key)
        preds[src] = dict(roles=[r for r in key[0].split(",") if r], tier=key[1],
                          needs_review=bool(first.get("needs_review")), method=first.get("classification_method", ""),
                          overrides=sum(bool(m.get("chunk_override")) for m in mds))
    return preds


def load_truth(path, include_unconfirmed):
    truth, skipped = {}, 0
    with open(path, newline="", encoding="utf-8") as f:
        lines = [l for l in f if l.strip() and not l.lstrip().startswith("#")]
    for row in csv.DictReader(lines):
        if not include_unconfirmed and row.get("confirmed", "").strip().lower() not in ("yes", "y", "true", "1"):
            skipped += 1; continue
        tier = row["tier"].strip().lower()
        if tier not in SENSITIVITY_TIERS:
            sys.exit(f"{row['source']}: bad tier {tier!r}; use one of {SENSITIVITY_TIERS}")
        try:
            truth[row["source"].strip()] = (parse_roles(row["roles"]), tier)
        except ValueError as e:
            sys.exit(f"{row['source']}: {e}")
    return truth, skipped


def make_template(path):
    preds = load_predictions()
    if os.path.exists(path):
        sys.exit(f"{path} already exists; delete it first or edit it by hand.")
    with open(path, "w", newline="", encoding="utf-8") as f:
        f.write("# Pre-filled from the CURRENT classifier output. Correct every wrong row, then set confirmed=yes.\n")
        w = csv.writer(f); w.writerow(["source", "roles", "tier", "confirmed", "note"])
        for src in sorted(preds):
            w.writerow([src, "|".join(preds[src]["roles"]), preds[src]["tier"], "no", ""])
    print(f"Wrote {len(preds)} rows to {path}. Fix the labels, set confirmed=yes, then run without --make-template.")


def append_history(label, m):
    new = not os.path.exists(HISTORY_PATH)
    prev = None
    if not new:
        with open(HISTORY_PATH, newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f)); prev = rows[-1] if rows else None
    with open(HISTORY_PATH, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if new: w.writerow(["timestamp", "label", *m.keys()])
        w.writerow([datetime.now().isoformat(timespec="seconds"), label, *m.values()])
    return prev


# ----------------------------------------------------------------- report
def report(rows, m, prev):
    bad = [r for r in rows if r["status"] != "OK"]
    print(f"\n{'STATUS':<16}{'FILE':<58}{'TRUTH':<30}GOT")
    for r in sorted(bad, key=lambda r: r["status"]):
        if r["status"] == "NOT INDEXED":
            print(f"{'NOT INDEXED':<16}{r['source'][:56]:<58}(labelled but not in the vector store)"); continue
        extra = f"  +roles {','.join(r['extra'])}" if r["extra"] else ""
        miss = f"  -roles {','.join(r['missing'])}" if r["missing"] else ""
        rev = "  [review]" if r["review"] else ""
        print(f"{r['status']:<16}{r['source'][:56]:<58}{r['truth']:<30}{r['got']}{extra}{miss}{rev}")
    n = m["n"] or 1
    print(f"\nScored {m['n']} confirmed documents")
    print(f"  fully correct          : {m['fully_correct']}/{m['n']}  ({100*m['fully_correct']/n:.0f}%)")
    print(f"  tier exact             : {m['tier_exact']}/{m['n']}   (too LOW: {m['tier_leak']} <- unsafe, too HIGH: {m['tier_over']})")
    print(f"  roles exact            : {m['roles_exact']}/{m['n']}   precision {m['role_precision']}  recall {m['role_recall']}")
    print(f"  documents with LEAK RISK: {m['docs_leak_risk']}   (precision < 1 means roles that should not see it can)")
    print(f"  flagged for review     : {m['flagged_review']}")
    if m["not_indexed"]: print(f"  labelled but not indexed: {m['not_indexed']}")
    if prev:
        print(f"\nvs previous run ({prev['timestamp']}, '{prev['label']}'):")
        for k in ("fully_correct", "tier_leak", "tier_over", "docs_leak_risk", "role_precision", "role_recall"):
            try:
                d = float(m[k]) - float(prev[k]); print(f"  {k:<16} {prev[k]} -> {m[k]}  ({d:+g})")
            except (KeyError, ValueError): pass


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--truth", default=TRUTH_PATH)
    ap.add_argument("--label", default="", help="short note saved with this run in the history")
    ap.add_argument("--make-template", action="store_true")
    ap.add_argument("--include-unconfirmed", action="store_true", help="score every row, even confirmed=no (first look only)")
    ap.add_argument("--no-history", action="store_true")
    ap.add_argument("--fail-on-leak", action="store_true", help="exit 1 if any document has a leak risk (for CI)")
    a = ap.parse_args()
    if a.make_template:
        return make_template(a.truth)
    truth, skipped = load_truth(a.truth, a.include_unconfirmed)
    if not truth:
        sys.exit(f"No confirmed labels in {a.truth}. Set confirmed=yes on rows you have checked.")
    if skipped: print(f"({skipped} unconfirmed rows ignored)")
    rows, m = score(truth, load_predictions())
    prev = None if a.no_history else append_history(a.label, m)
    report(rows, m, prev)
    if a.fail_on_leak and m["docs_leak_risk"]:
        sys.exit(1)


if __name__ == "__main__":
    main()