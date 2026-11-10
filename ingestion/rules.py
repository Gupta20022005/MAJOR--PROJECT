"""
Predefined classification rules.

ROLE_KEYWORDS / TIER_KEYWORDS drive document-level rule classification.
SENSITIVE_PATTERNS / SENSITIVE_TERMS drive chunk-level validation, which can
only ever make a chunk STRICTER than its document, never looser.

Weights are coarse (1-3). These lists are starting points for a demo corpus,
not a production taxonomy — extend them to your organization's vocabulary.
Matching is on whole words (plurals allowed), so "contract" matches
"contracts" but "tax" does not match "taxonomy".
"""

import re
from typing import Dict

MAX_HITS_PER_TERM = 5  # one repeated word shouldn't dominate a long document

ROLE_KEYWORDS = {
    "finance": {"payroll": 3, "invoice": 2, "budget": 2, "revenue": 2, "accounting": 2, "expense": 1,
                "forecast": 1, "audit": 2, "ledger": 2, "tax": 1, "salary": 2, "balance sheet": 3},
    "hr": {"employee": 2, "appraisal": 3, "disciplinary": 3, "recruitment": 2, "onboarding": 2,
           "benefits": 1, "compensation": 2, "performance review": 3, "termination": 2,
           "grievance": 3, "leave policy": 2, "payroll": 2, "salary": 2},
    "legal": {"litigation": 3, "contract": 2, "compliance": 2, "regulatory": 2, "intellectual property": 2,
              "liability": 2, "clause": 1, "plaintiff": 3, "settlement": 3, "counsel": 2, "indemnity": 3},
    "engineering": {"specification": 2, "architecture": 2, "api": 1, "schema": 1, "deployment": 1,
                    "codebase": 2, "system design": 2, "protocol": 1, "requirements": 1, "changelog": 1,
                    "microservice": 2},
    "sales": {"pipeline": 2, "quota": 2, "prospect": 2, "deal": 2, "discount": 2, "customer account": 2,
              "proposal": 1, "renewal": 2, "upsell": 2, "crm": 1, "lead": 1,
              "key account": 2, "quotation": 2, "pricing": 1},
    "executive": {"board": 3, "merger": 3, "acquisition": 3, "strategic plan": 3, "shareholder": 2,
                  "leadership": 1, "restructuring": 3, "confidential memo": 3, "divestiture": 3},
}

TIER_KEYWORDS = {
    "restricted": {"unreleased": 3, "litigation strategy": 3, "trade secret": 3, "merger": 2,
                   "acquisition": 2, "divestiture": 3, "embargoed": 3, "do not distribute": 3, "privileged": 2},
    "confidential": {"confidential": 3, "internal use only": 2, "salary": 2, "compensation": 2,
                     "proprietary": 2, "non-disclosure": 2, "sensitive": 1, "payroll": 2},
    "internal": {"internal": 2, "staff only": 2, "company policy": 1, "draft": 1, "working document": 1},
    "public": {"press release": 2, "public": 1, "announcement": 1, "marketing": 1, "brochure": 1,
               "notice of annual general meeting": 3, "notice of meeting": 3, "annual general meeting": 2,
               "letter to unitholders": 3, "letter to shareholders": 3},
}

# Document types written for an outside audience. If one of these is in the file name or the opening of
# the text (and nothing sensitive is found), the document is org-wide and public.
PUBLIC_DOC_TYPES = ["press release", "brochure", "notice of annual general meeting", "notice of meeting",
                    "letter to unitholders", "letter to shareholders",
                    "published document", "published policy", "published supplier document", "published service terms",
                    "published investor communication", "published shareholder notice", "published on our",
                    "public whitepaper", "public policy statement", "public developer documentation"]

# Classification banners near the top of a document ("Restricted - Board only", "Privileged and confidential").
# A banner is the author's own statement of sensitivity, so it counts as strong evidence for that tier.
HEADER_WEIGHT = 8
HEADER_SCAN_CHARS = 700
HEADER_MARKERS = {
    "restricted": [r"\brestricted\b\s*[-:\u2013\u2014]", r"\bprivileged (?:and|&) confidential\b", r"\bprivileged legal advice\b",
                   r"\battorney[- ]client\b", r"\bstrictly confidential\b", r"\bprivileged in part\b",
                   r"\brestricted distribution\b", r"\bnot to be circulated\b"],
    "confidential": [r"\bconfidential\b\s*[-:\u2013\u2014]", r"\bconfidential and not to be disclosed\b"],
    "internal": [r"\binternal [a-z ,]{0,30}use only\b"],
}


def header_tier(text: str):
    """Most restrictive tier announced by a banner in the opening of the text, or None."""
    import re
    head = text[:HEADER_SCAN_CHARS].lower()
    for tier in ("restricted", "confidential", "internal"):
        if any(re.search(p, head) for p in HEADER_MARKERS[tier]):
            return tier
    return None

# Chunk-level: (pattern, minimum tier it forces, label)
SENSITIVE_PATTERNS = [
    (re.compile(r"\b\d{4}[ -]\d{4}[ -]\d{4}\b"), "restricted", "12-digit national ID number"),
    (re.compile(r"\b\d{3}-\d{2}-\d{4}\b"), "restricted", "SSN-style ID number"),
    (re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----"), "restricted", "private key"),
    (re.compile(r"(?i)\b(?:password|passwd|api[_ -]?key|secret[_ -]?key)\s*[:=]\s*\S+"), "restricted", "credential"),
    (re.compile(r"\b[A-Z]{5}\d{4}[A-Z]\b"), "confidential", "PAN-style tax ID"),
    (re.compile(r"(?i)\b(?:account|a/c)\s*(?:no\.?|number|#)?\s*[:#-]?\s*\d{9,18}\b"), "confidential", "bank account number"),
    (re.compile(r"(?i)\b(?:salary|ctc|compensation|stipend)\b[^.\n]{0,40}?(?:₹|rs\.?|inr|\$|usd)\s?\d"),
     "confidential", "salary figure"),
]

# Chunk-level: whole-word terms that force a minimum tier wherever they appear
SENSITIVE_TERMS = {
    "restricted": ["do not distribute", "trade secret", "litigation strategy", "embargoed",
                   "unreleased", "attorney-client privileged"],
    "confidential": ["confidential", "internal use only", "non-disclosure", "disciplinary action",
                     "termination of employment", "performance review", "salary", "payroll"],
}


def _term_regex(term: str) -> re.Pattern:
    return re.compile(r"\b" + re.escape(term.lower()) + r"(?:s|es)?\b")


_COMPILED: Dict[str, Dict[str, re.Pattern]] = {}


def keyword_scores(text: str, keyword_map: Dict[str, Dict[str, int]]) -> Dict[str, float]:
    """Weighted keyword score per category (raw, not normalized)."""
    low = text.lower()
    out = {}
    for category, terms in keyword_map.items():
        score = 0.0
        for term, weight in terms.items():
            rx = _COMPILED.setdefault(term, _term_regex(term))
            score += weight * min(len(rx.findall(low)), MAX_HITS_PER_TERM)
        out[category] = score
    return out


def term_hits(text: str, terms) -> list:
    low = text.lower()
    return [t for t in terms if _COMPILED.setdefault(t, _term_regex(t)).search(low)]