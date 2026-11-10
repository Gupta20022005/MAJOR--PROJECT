"""
Attaches the per-chunk metadata contract:

    doc_id, source, page, section, ingested_at,
    allowed_roles ("hr,legal"), role_<name> flags, sensitivity_tier,
    sensitivity_rank, classification_method, needs_review, chunk_override

Chunks INHERIT the document-level classification, then ingestion.classify.
validate_chunk() may tighten (never loosen) an individual chunk.

Multi-role storage: Chroma filters compare scalars, so besides the readable
`allowed_roles` string each chunk carries one boolean flag per role
(role_hr=True, role_legal=False, ...). The query-side filter is then "any of
the user's role flags is True" (query/access_control.py).
sensitivity_rank is stored beside the tier so the clearance check is a
numeric $lte.
"""

from __future__ import annotations
import hashlib
import os
from datetime import datetime, timezone
from typing import List, Dict, Optional, Union

from config import (
    DEFAULT_ALLOWED_ROLE, DEFAULT_SENSITIVITY_TIER, TIER_RANK, ALL_ROLE_VALUES, role_flag,
)
from ingestion.classify import DocumentClassification, validate_chunk, _apply_policy


def make_doc_id(file_path: str) -> str:
    return hashlib.sha1(os.path.abspath(file_path).encode("utf-8")).hexdigest()[:16]


def make_chunk_id(doc_id: str, chunk_index: int) -> str:
    return f"{doc_id}::chunk-{chunk_index}"


def _normalize_roles(roles: Union[str, List[str]]) -> List[str]:
    if isinstance(roles, str):
        roles = [r for r in roles.split(",")]
    cleaned = list(dict.fromkeys(r.strip().lower() for r in roles if r.strip()))
    bad = [r for r in cleaned if r not in ALL_ROLE_VALUES]
    if bad or not cleaned:
        raise ValueError(f"Invalid role(s) {bad or roles}; expected a subset of {ALL_ROLE_VALUES}")
    return cleaned


def access_metadata(roles: List[str], tier: str) -> Dict:
    """The access-control fields for one chunk."""
    meta = {
        "allowed_roles": ",".join(roles),
        "sensitivity_tier": tier,
        "sensitivity_rank": TIER_RANK[tier],
    }
    for r in ALL_ROLE_VALUES:
        meta[role_flag(r)] = r in roles
    return meta


def attach_metadata(
    chunks: List[Dict],
    file_path: str,
    doc_classification: Optional[DocumentClassification] = None,
    allowed_roles: Optional[Union[str, List[str]]] = None,
    sensitivity_tier: Optional[str] = None,
) -> List[Dict]:
    """
    Pass `doc_classification` (from ingestion.classify.classify_document) for
    the normal automatic path. Manual overrides are policy-validated. With no
    classification, use a restricted executive-only fallback and mark it for
    review; query-time access filters exclude all documents needing review.
    Chunk-level tightening applies in every case.
    """
    doc_id = make_doc_id(file_path)
    source = os.path.basename(file_path)
    ingested_at = datetime.now(timezone.utc).isoformat()

    if allowed_roles is not None or sensitivity_tier is not None:
        roles = _normalize_roles(allowed_roles if allowed_roles is not None else DEFAULT_ALLOWED_ROLE)
        tier = sensitivity_tier if sensitivity_tier is not None else DEFAULT_SENSITIVITY_TIER
        if tier not in TIER_RANK:
            raise ValueError(f"Invalid tier {tier!r}; expected one of {list(TIER_RANK)}")
        roles, tier, notes = _apply_policy(roles, tier)
        doc = DocumentClassification(
            roles, tier, "manual", None,
            "manual override" + (f"; policy: {'; '.join(notes)}" if notes else ""),
            bool(notes),
        )
    elif doc_classification is not None:
        # Revalidate at the metadata boundary too: callers may construct a
        # DocumentClassification directly instead of using classify_document.
        roles = _normalize_roles(doc_classification.roles)
        tier = doc_classification.tier
        if tier not in TIER_RANK:
            raise ValueError(f"Invalid tier {tier!r}; expected one of {list(TIER_RANK)}")
        roles, tier, notes = _apply_policy(roles, tier)
        doc = DocumentClassification(
            roles, tier, doc_classification.method, doc_classification.confidence,
            doc_classification.reason + (f"; policy: {'; '.join(notes)}" if notes else ""),
            doc_classification.needs_review or bool(notes), doc_classification.rule_status,
        )
    else:
        # Missing classification is NOT public/org-wide. Quarantine by default.
        doc = DocumentClassification([DEFAULT_ALLOWED_ROLE], DEFAULT_SENSITIVITY_TIER,
                                     "default", None, "no classification supplied; quarantined pending review", True)

    enriched = []
    for chunk in chunks:
        cc = validate_chunk(chunk["text"], doc)
        metadata = {
            "doc_id": doc_id, "source": source, "page": chunk.get("page"),
            "section": chunk.get("section"), "ingested_at": ingested_at,
            "classification_method": doc.method,
            "needs_review": doc.needs_review,
            "chunk_override": cc.overridden,
            **access_metadata(cc.roles, cc.tier),
        }
        enriched.append({
            "id": make_chunk_id(doc_id, chunk["chunk_index"]),
            "text": chunk["text"],
            "metadata": metadata,
        })
    return enriched
