"""
Document-level classification: RULES FIRST, LLM ONLY WHEN NEEDED.

    classify_document(text)  -> roles + tier for the whole document
    validate_chunk(chunk, doc) -> the chunk's final roles + tier

Flow
----
1. Rules: weighted keyword lists decide roles and tier. If they give a clear
   answer that is consistent with the access policy, it is accepted and the
   LLM is never called.
2. LLM fallback: used when no rule matches, only part of the answer is known,
   or the matched roles conflict under the policy. It returns roles, tier,
   confidence, a reason, and whether manual review is needed.
3. Chunks inherit the document's label. A validation pass may then make an
   individual chunk STRICTER (a salary table inside a public handbook) —
   never looser.

Failure rule (deliberately lopsided)
------------------------------------
An unreachable LLM, an unparseable reply, or an unsure answer never resolves
permissively. The tier is escalated one rank and the document is flagged for
review; if nothing at all is known it becomes restricted + executive-only.
Over-restriction is visible and self-correcting; over-permission is a silent
confidentiality breach.
"""

from __future__ import annotations
import json
import logging
import re
from dataclasses import dataclass, field
from itertools import combinations
from typing import Dict, List, Optional

from config import (
    ORGANIZATIONAL_ROLES, ALL_ROLES_PSEUDO_ROLE, SENSITIVITY_TIERS, TIER_RANK,
    COMPATIBLE_ROLE_PAIRS, MAX_TIER_FOR_ALL_ROLES, LLM_MIN_CONFIDENCE,
    RULE_MIN_ROLE_SCORE, RULE_ROLE_RELATIVE_SHARE, RULE_MIN_TIER_SCORE,
    RULE_MAX_ROLES, LLM_MAX_ROLES, LLM_REVIEW_ESCALATE_BELOW,
    RULE_TIER_FLOOR_SCORE, LLM_DEFAULT_CONFIDENCE, PUBLIC_TITLE_OVERRIDE,
    LLM_SAMPLE_HEAD_CHARS, LLM_SAMPLE_TAIL_CHARS,
)
from ingestion.rules import (
    ROLE_KEYWORDS, TIER_KEYWORDS, SENSITIVE_PATTERNS, SENSITIVE_TERMS,
    keyword_scores, term_hits, PUBLIC_DOC_TYPES, _term_regex, header_tier, HEADER_WEIGHT,
)
from query.llm_providers import generate_with_retry, LLMError, BaseLLMProvider
from logging_setup import get_audit_logger

logger = logging.getLogger("ingestion.classify")
audit = get_audit_logger()

VALID_ROLES = set(ORGANIZATIONAL_ROLES) | {ALL_ROLES_PSEUDO_ROLE}


@dataclass
class DocumentClassification:
    roles: List[str]
    tier: str
    method: str                      # "rules" | "llm" | "failsafe" | "manual" | "default"
    confidence: Optional[float]      # LLM's confidence; None when rules decided
    reason: str
    needs_review: bool
    rule_status: str = ""


@dataclass
class ChunkClassification:
    roles: List[str]
    tier: str
    overridden: bool
    reasons: List[str] = field(default_factory=list)


@dataclass
class _RuleResult:
    status: str                      # clear | no_match | partial | conflict
    roles: List[str]
    tier: Optional[str]
    top_role: Optional[str]
    role_scores: Dict[str, float]
    tier_scores: Dict[str, float]


def _escalate(tier: str) -> str:
    return SENSITIVITY_TIERS[min(TIER_RANK[tier] + 1, len(SENSITIVITY_TIERS) - 1)]


def _max_tier(a: str, b: str) -> str:
    return a if TIER_RANK[a] >= TIER_RANK[b] else b


def _compatible(roles: List[str]) -> bool:
    return all(frozenset(p) in COMPATIBLE_ROLE_PAIRS for p in combinations(roles, 2))


# ---------------------------------------------------------------------------
# 1. Rules
# ---------------------------------------------------------------------------
def _rule_classify(text: str, source_name: str) -> _RuleResult:
    haystack = f"{source_name.replace('_', ' ')}\n{text}"  # underscores are \\w, so they would hide every word in the file name from \\b matching
    role_scores = keyword_scores(haystack, ROLE_KEYWORDS)
    # "Confidential Information" is a defined term in NDAs and similar agreements, not a sensitivity marking.
    tier_scores = keyword_scores(re.sub(r"confidential information", " ", haystack, flags=re.I), TIER_KEYWORDS)
    banner = header_tier(text)
    if banner:
        tier_scores[banner] += HEADER_WEIGHT

    top_role, top_score = max(role_scores.items(), key=lambda kv: kv[1])
    roles = []
    if top_score >= RULE_MIN_ROLE_SCORE:
        roles = [r for r, sc in role_scores.items()
                 if sc >= RULE_MIN_ROLE_SCORE and sc >= RULE_ROLE_RELATIVE_SHARE * top_score]
        roles.sort(key=lambda r: -role_scores[r])

    # The MOST restrictive tier that clears the bar wins: a document that says
    # both "press release" and "confidential" is confidential.
    tier = next((t for t in reversed(SENSITIVITY_TIERS) if tier_scores[t] >= RULE_MIN_TIER_SCORE), None)

    # A public-type document (notice, press release, shareholder letter) that also mentions a sensitive
    # word once or twice is ambiguous: weak higher-tier evidence must not win by default. Let the LLM decide.
    # Strong evidence (>= RULE_TIER_FLOOR_SCORE) is still accepted by the rules directly.
    if (tier not in (None, "public") and tier_scores["public"] >= RULE_MIN_TIER_SCORE
            and tier_scores[tier] < RULE_TIER_FLOOR_SCORE):
        return _RuleResult("conflict", roles, tier, roles[0] if roles else None, role_scores, tier_scores)
    if not roles and tier == "public":
        return _RuleResult("clear", [ALL_ROLES_PSEUDO_ROLE], "public", None, role_scores, tier_scores)
    if not roles and tier is None:
        return _RuleResult("no_match", [], None, None, role_scores, tier_scores)
    if not roles or tier is None:
        return _RuleResult("partial", roles, tier, roles[0] if roles else None, role_scores, tier_scores)
    if len(roles) > RULE_MAX_ROLES or not _compatible(roles):   # too broad / not an approved mix -> LLM
        return _RuleResult("conflict", roles, tier, roles[0], role_scores, tier_scores)
    return _RuleResult("clear", roles, tier, roles[0], role_scores, tier_scores)


# ---------------------------------------------------------------------------
# 2. LLM fallback
# ---------------------------------------------------------------------------
_LLM_PROMPT = """You are classifying a document for a role-based access-control system.
Over-labelling is a bug: every extra role exposes the document to more people.

ROLES. Pick the ONE primary owning role first. Add a second role ONLY if the document is
substantially about both. Never return more than {max_roles} roles. Roles: {roles}
- finance: accounting, budgets, payroll, financial reporting, investor/unitholder results
- hr: hiring, compensation, performance, employee relations, leave and policies for staff
- legal: contracts and agreements, risk registers, litigation, compliance, regulatory, IP
- engineering: ONLY technical specifications, software/system architecture, product design.
  NOT logistics, operations, contracts, risk or supply chain.
- sales: customers, quotes, proposals, pipeline, pricing, discounts, renewals
- executive: board, strategy, M&A, restructuring, leadership decisions, enterprise risk
Logistics, supply-chain and operations material has no role of its own: choose the owner of the
decision (legal for agreements, executive for strategy or risk, finance for cost).
Mentioning a topic in passing does NOT make a role relevant. Judge what the document IS, not
every word it contains. Use "all" ONLY for genuinely org-wide or public material (press
releases, brochures, notices), alone, and only with tier public or internal.

TIERS, least to most sensitive: {tiers}
- public: written for external audiences (press releases, brochures, published notices, public filings)
- internal: routine company work not meant for outsiders (SOPs, reports, internal memos, plans)
- confidential: agreements, renewals and proposals with a named counterparty, customer-specific
  pricing, non-public financials, personnel data
- restricted: M&A or divestiture plans, legal strategy, board executive sessions, credentials
Choose the tier the content itself justifies. A "Confidential" footer or boilerplate alone does
not raise a tier, but do not go below the descriptions above.

Keyword rules found (a weak hint, may be wrong): {rule_hint}

Respond with ONLY a JSON object, no other text:
{{"roles": ["..."], "tier": "...", "confidence": 0.0-1.0, "reason": "one short sentence", "needs_review": true or false}}

Set needs_review to true ONLY if you genuinely cannot decide between roles or tiers.

Document: {source}
---
{sample}
"""


def _sample(text: str) -> str:
    if len(text) <= LLM_SAMPLE_HEAD_CHARS + LLM_SAMPLE_TAIL_CHARS:
        return text
    return text[:LLM_SAMPLE_HEAD_CHARS] + "\n[...]\n" + text[-LLM_SAMPLE_TAIL_CHARS:]


def _parse_llm(response: str) -> Optional[dict]:
    """Returns a validated dict, or None if the reply isn't usable."""
    m = re.search(r"\{.*\}", response, re.DOTALL)
    if not m:
        return None
    try:
        data = json.loads(m.group(0))
        roles = [str(r).lower().strip() for r in data.get("roles", [])]
        tier = str(data.get("tier", "")).lower().strip()
        if not roles or any(r not in VALID_ROLES for r in roles) or tier not in TIER_RANK:
            return None
        conf = data.get("confidence")
        conf = max(0.0, min(1.0, float(conf))) if conf is not None else LLM_DEFAULT_CONFIDENCE
        return {
            "roles": list(dict.fromkeys(roles)),
            "tier": tier,
            "confidence": conf,
            "reason": str(data.get("reason", ""))[:300],
            "needs_review": bool(data.get("needs_review", False)),
        }
    except (ValueError, TypeError, AttributeError):
        return None


def _llm_classify(text: str, source_name: str, rule: _RuleResult, provider: BaseLLMProvider) -> Optional[dict]:
    hint = (f"status={rule.status}, roles={rule.roles or 'none'}, tier={rule.tier or 'none'}")
    prompt = _LLM_PROMPT.format(
        roles=", ".join(ORGANIZATIONAL_ROLES), max_roles=LLM_MAX_ROLES, tiers=", ".join(SENSITIVITY_TIERS),
        rule_hint=hint, source=source_name, sample=_sample(text),
    )
    try:
        raw = generate_with_retry(prompt, provider, max_retries=2)
        parsed = _parse_llm(raw)
        if parsed is None:
            logger.warning(f"{source_name}: LLM reply unusable (bad JSON, unknown role/tier, or empty): {str(raw)[:300]!r}")
        return parsed
    except LLMError as e:
        logger.warning(f"LLM fallback unavailable: {e}")
    except Exception as e:  # noqa: BLE001 - a bug here must not fail ingestion
        logger.warning(f"LLM fallback error: {e}")
    return None


# ---------------------------------------------------------------------------
# Policy + failure handling
# ---------------------------------------------------------------------------
def _apply_policy(roles: List[str], tier: str, narrow_incompatible: bool = False):
    """Returns (roles, tier, notes). Only ever narrows or flags, never widens."""
    notes = []
    roles = list(dict.fromkeys(roles))
    if ALL_ROLES_PSEUDO_ROLE in roles and len(roles) > 1:
        roles = [r for r in roles if r != ALL_ROLES_PSEUDO_ROLE]
        notes.append('dropped "all" because specific roles were also given')
    if roles == [ALL_ROLES_PSEUDO_ROLE] and TIER_RANK[tier] > TIER_RANK[MAX_TIER_FOR_ALL_ROLES]:
        roles = ["executive"]
        notes.append(f'"all" is not permitted above {MAX_TIER_FOR_ALL_ROLES}; restricted to executive')
    if ALL_ROLES_PSEUDO_ROLE not in roles and not _compatible(roles):
        if narrow_incompatible:
            notes.append(f"roles {roles} not an approved combination; narrowed to primary role {roles[0]}")
            roles = [roles[0]]
        else:
            notes.append(f"roles {roles} are not a policy-approved combination")
    return roles, tier, notes


def _failsafe(rule: _RuleResult, why: str) -> DocumentClassification:
    """Nothing reliable available: resolve to the stricter outcome."""
    roles, tier = list(rule.roles), rule.tier
    if rule.status == "conflict" and rule.top_role:
        roles = [rule.top_role]                     # narrowest: the single strongest role
    if not roles and tier is None:
        return DocumentClassification(["executive"], "restricted", "failsafe", None,
                                      f"{why}; no rules matched, so restricted to executive pending review",
                                      True, rule.status)
    if not roles:
        roles = ["executive"]                       # tier known, role unknown
    tier = _escalate(tier or "internal")            # unknown tier is assumed internal, then escalated
    return DocumentClassification(roles, tier, "failsafe", None,
                                  f"{why}; partial rule result escalated one tier pending review",
                                  True, rule.status)


def _is_public_document(text: str, source_name: str, rule: _RuleResult) -> bool:
    """A press release / brochure / AGM notice / shareholder letter, with NO sensitive keyword evidence."""
    opening = f"{source_name.replace('_', ' ')}\n{text[:600]}".lower()
    if not any(_term_regex(t).search(opening) for t in PUBLIC_DOC_TYPES):
        return False
    ts = rule.tier_scores
    return (ts.get("restricted", 0) == 0 and ts.get("confidential", 0) < RULE_TIER_FLOOR_SCORE
            and ts.get("internal", 0) < RULE_MIN_TIER_SCORE)


def classify_document(text: str, source_name: str, llm_provider: BaseLLMProvider,
                      doc_id: str = "") -> DocumentClassification:
    rule = _rule_classify(text, source_name)

    if PUBLIC_TITLE_OVERRIDE and _is_public_document(text, source_name, rule):
        doc = DocumentClassification(
            [ALL_ROLES_PSEUDO_ROLE], "public", "rules", None,
            "public document type (press release / brochure / notice / shareholder letter) with no sensitive keywords",
            False, rule.status)
    elif rule.status == "clear":
        roles, tier, notes = _apply_policy(rule.roles, rule.tier)
        doc = DocumentClassification(
            roles, tier, "rules", None,
            f"rules matched roles={roles} (scores " +
            ", ".join(f"{r}={rule.role_scores.get(r, 0):.0f}" for r in roles) +
            f"), tier={tier}" + (f"; {'; '.join(notes)}" if notes else ""),
            bool(notes), rule.status)
    else:
        llm = _llm_classify(text, source_name, rule, llm_provider)
        if llm is None:
            doc = _failsafe(rule, "LLM fallback unavailable or unusable")
        else:
            capped = llm["roles"][:LLM_MAX_ROLES]          # primary role first; extras are dropped
            roles, tier, notes = _apply_policy(capped, llm["tier"], narrow_incompatible=True)
            # needs_review alone is not enough: models set it liberally. Escalate only when
            # confidence is actually low.
            unsure = (llm["confidence"] < LLM_MIN_CONFIDENCE or
                      (llm["needs_review"] and llm["confidence"] < LLM_REVIEW_ESCALATE_BELOW))
            reason = llm["reason"] or "no reason given"
            # (a) A department-owned document is never "public": that tier is for org-wide material.
            if tier == "public" and ALL_ROLES_PSEUDO_ROLE not in roles:
                tier = "internal"
                notes.append("specific roles with tier public -> raised to internal")
            # (b) Strong keyword evidence (e.g. "acquisition", "divestiture") is a floor the LLM cannot undercut.
            floor = next((t for t in reversed(SENSITIVITY_TIERS)
                          if rule.tier_scores.get(t, 0) >= RULE_TIER_FLOOR_SCORE), None)
            if floor and TIER_RANK[floor] > TIER_RANK[tier]:
                notes.append(f"keyword evidence floors tier at {floor}")
                tier = floor
            if unsure:
                tier = _escalate(tier)
                reason += f" [unsure (confidence {llm['confidence']:.2f}); tier escalated one rank]"
            if notes:
                reason += " [policy: " + "; ".join(notes) + "]"
            # Narrowing to the primary role is routine; any other policy action still needs a human look.
            policy_flag = any("narrowed to primary role" not in n and "raised to internal" not in n and "floors tier" not in n for n in notes)
            # Cross-check: the LLM's primary role has NO keyword support while the rules clearly see another.
            # Surface the disagreement for a human; do not silently override either side.
            best_role, best_score = max(rule.role_scores.items(), key=lambda kv: kv[1])
            disagree = (roles and roles[0] in rule.role_scores and rule.role_scores[roles[0]] == 0
                        and best_score >= RULE_MIN_ROLE_SCORE)
            if disagree:
                reason += f" [LLM role '{roles[0]}' has no keyword support; rules suggest '{best_role}']"
            doc = DocumentClassification(roles, tier, "llm", llm["confidence"], reason,
                                         unsure or policy_flag or bool(disagree), rule.status)

    audit.log("document_classified", doc_id=doc_id, source=source_name, method=doc.method,
              roles=doc.roles, tier=doc.tier, rule_status=doc.rule_status,
              confidence=doc.confidence, needs_review=doc.needs_review, reason=doc.reason)
    if doc.needs_review:
        audit.log("document_flagged_for_review", doc_id=doc_id, source=source_name,
                  roles=doc.roles, tier=doc.tier, reason=doc.reason)
        logger.info(f"{source_name}: flagged for review ({doc.reason})")
    return doc


# ---------------------------------------------------------------------------
# 3. Chunk-level validation: inherit, then tighten only
# ---------------------------------------------------------------------------
def validate_chunk(chunk_text: str, doc: DocumentClassification) -> ChunkClassification:
    """Chunks inherit the document's roles and tier. If the chunk itself
    contains sensitive content demanding more, it is made stricter. This can
    raise the tier and narrow the roles; it can never lower or widen either."""
    required, reasons = doc.tier, []

    for pattern, tier, label in SENSITIVE_PATTERNS:
        if pattern.search(chunk_text) and TIER_RANK[tier] > TIER_RANK[required]:
            required = tier
            reasons.append(f"{label} -> {tier}")
    for tier, terms in SENSITIVE_TERMS.items():
        hits = term_hits(chunk_text, terms)
        if hits and TIER_RANK[tier] > TIER_RANK[required]:
            required = tier
            reasons.append(f"term '{hits[0]}' -> {tier}")

    roles = list(doc.roles)
    if TIER_RANK[required] > TIER_RANK[doc.tier]:
        # Sensitive content found: narrow the audience to the roles this chunk
        # is actually about, when that's a subset of the document's roles.
        scores = keyword_scores(chunk_text, ROLE_KEYWORDS)
        about = [r for r, sc in scores.items() if sc >= RULE_MIN_ROLE_SCORE]
        if roles == [ALL_ROLES_PSEUDO_ROLE]:
            if TIER_RANK[required] > TIER_RANK[MAX_TIER_FOR_ALL_ROLES]:
                roles = about or ["executive"]
                reasons.append(f"narrowed from 'all' to {roles}")
        elif len(roles) > 1:
            narrowed = [r for r in roles if r in about]
            if narrowed and len(narrowed) < len(roles):
                roles = narrowed
                reasons.append(f"narrowed roles to {roles}")

    overridden = required != doc.tier or roles != doc.roles
    return ChunkClassification(roles, required, overridden, reasons)