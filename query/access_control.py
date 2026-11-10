"""
Compiles an authenticated principal into the access-control predicate
evaluated INSIDE the vector search — not applied to its results.

    authorized(u, c)  <=>  ( any role of u is a role of c )
                       AND ( sensitivity_rank(c) <= clearance_rank(u) )

A chunk may belong to several roles (stored as one boolean flag per role),
so the role condition is an OR over the user's role flags. Both conditions
must hold: role alone isn't enough (right department, over clearance) and
clearance alone isn't enough (same tier, wrong department). Passed as
Chroma's `where=`, so an unauthorized chunk is never a candidate, never
occupies a top-k slot, and is never seen by the reranker or the LLM.
"""

from __future__ import annotations
from typing import Dict

from config import TIER_RANK, role_flag
from query.auth import User


def build_access_filter(user: User) -> Dict:
    flags = [{role_flag(r): True} for r in dict.fromkeys(user.roles)]
    role_condition = flags[0] if len(flags) == 1 else {"$or": flags}  # Chroma needs >=2 clauses in $or
    # Fail closed: unreviewed or legacy chunks without an explicit approval flag
    # must never enter dense retrieval or the hybrid retriever's candidate set.
    return {"$and": [
        role_condition,
        {"sensitivity_rank": {"$lte": TIER_RANK[user.sensitivity_clearance]}},
        {"needs_review": False},
    ]}
