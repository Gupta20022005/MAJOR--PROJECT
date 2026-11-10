"""
Central configuration for the RAG ingestion + query pipeline.
Adjust these values to match your environment (paths, model names, etc).
"""

import os

try:
    from dotenv import load_dotenv
    load_dotenv()  # loads .env from the current working directory, if present
except ImportError:
    pass  # python-dotenv not installed — fall back to real environment variables only

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

SOURCE_DOCS_DIR = os.path.join(BASE_DIR, "data", "source_docs")   # drop raw files here
STATE_DIR = os.path.join(BASE_DIR, "data", "state")
INGESTION_MANIFEST_PATH = os.path.join(STATE_DIR, "ingestion_manifest.json")
VECTOR_STORE_DIR = os.path.join(BASE_DIR, "data", "vector_store")
MARKDOWN_DIR = os.path.join(BASE_DIR, "data", "extracted_markdown")  # per-doc .md saved during extraction

# ---------------------------------------------------------------------------
# Supported input formats
# ---------------------------------------------------------------------------
SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".doc", ".txt"}

# ---------------------------------------------------------------------------
# Chunking
# ---------------------------------------------------------------------------
CHUNK_SIZE_TOKENS = 500
CHUNK_OVERLAP_TOKENS = 50

# Separators tried in order, largest semantic unit first (recursive splitting)
RECURSIVE_SEPARATORS = ["\n\n", "\n", ". ", " ", ""]

# ---------------------------------------------------------------------------
# Embeddings
# ---------------------------------------------------------------------------
# Any local sentence-transformers model works out of the box. Swap this for
# an API-backed embedder (OpenAI, Voyage, Anthropic-compatible, etc.) by
# implementing the same interface in ingestion/embedder.py.
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_BATCH_SIZE = 32

# ---------------------------------------------------------------------------
# Vector store
# ---------------------------------------------------------------------------
VECTOR_STORE_COLLECTION = "rag_documents"

# ---------------------------------------------------------------------------
# LLM (query pipeline) — switch providers via LLM_PROVIDER, no code changes
# ---------------------------------------------------------------------------
LLM_PROVIDER = os.environ.get("LLM_PROVIDER", "gemini")  # "gemini" | "ollama"

GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-2.5-flash")

OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "llama3.2")
OLLAMA_BASE_URL = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
LOG_DIR = os.path.join(BASE_DIR, "logs")
LOG_LEVEL = os.environ.get("LOG_LEVEL", "INFO")
LOG_MAX_BYTES = 5 * 1024 * 1024   # 5 MB per log file before rotating
LOG_BACKUP_COUNT = 5               # keep 5 rotated files (app.log.1 ... app.log.5)

# ---------------------------------------------------------------------------
# Access control (attached at metadata stage, enforced at query stage)
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Access control model (Phase 1)
# ---------------------------------------------------------------------------
# Chunk metadata stores allowed_roles as a SCALAR (the single owning role, or
# the pseudo-role "all" for org-wide material) — not a list. Chroma metadata
# filters compare scalars, so a scalar owning-role field lets the access
# predicate use a plain {"$in": user.roles} clause. A user can still hold
# several roles (Executive holds membership in every departmental role).
ORGANIZATIONAL_ROLES = ["finance", "hr", "legal", "engineering", "sales", "executive"]
ALL_ROLES_PSEUDO_ROLE = "all"
DEFAULT_ALLOWED_ROLE = ALL_ROLES_PSEUDO_ROLE
DEFAULT_SENSITIVITY_TIER = "public"

# Four tiers, least -> most sensitive. sensitivity_rank (ordinal) is stored
# next to sensitivity_tier so the clearance check is a numeric $lte.
SENSITIVITY_TIERS = ["public", "internal", "confidential", "restricted"]
TIER_RANK = {tier: rank for rank, tier in enumerate(SENSITIVITY_TIERS)}

# ---------------------------------------------------------------------------
# Document classification: rules first, LLM fallback (ingestion/classify.py)
# ---------------------------------------------------------------------------
# The whole DOCUMENT is classified once; chunks inherit the label, then a
# restrictive-only validation pass may tighten individual chunks.
#
#   1. Keyword rules run first. If they give a clear, policy-consistent
#      answer (roles AND tier), it is accepted and the LLM is never called.
#   2. If no rule matches, only part of the answer is known, or the matched
#      roles conflict under the access policy, the LLM classifies the document.
#   3. Anything unreliable (LLM unreachable, unparseable, or unsure) resolves
#      to the stricter outcome: tier escalated one rank (or restricted +
#      executive when nothing at all is known) and flagged for review.
#
# A document may legitimately belong to several roles (e.g. HR + Finance for
# payroll); multiple matching roles alone are not treated as ambiguity.
RULE_MIN_ROLE_SCORE = 6          # weighted keyword score a role needs to count as matched
RULE_ROLE_RELATIVE_SHARE = 0.7   # ...and it must reach this share of the top role's score
RULE_MIN_TIER_SCORE = 4          # weighted score a tier needs (one weight-3 mention is NOT enough); the MOST restrictive qualifying tier wins
RULE_MAX_ROLES = 2               # rules matching MORE roles than this are too broad to trust -> LLM decides
LLM_MAX_ROLES = 2                # the LLM's role list is capped to its first N (primary role first)
LLM_REVIEW_ESCALATE_BELOW = 0.75 # an LLM "needs_review" only escalates the tier / flags review below this confidence
RULE_TIER_FLOOR_SCORE = 6        # a rule tier scoring this high is a FLOOR the LLM may not go below (e.g. "acquisition" -> restricted)
LLM_DEFAULT_CONFIDENCE = 0.7     # used when the LLM omits "confidence" (0.0 would wrongly count as unsure)
PUBLIC_TITLE_OVERRIDE = True     # press releases, brochures, AGM notices, shareholder letters -> all/public (only when no sensitive keyword evidence)
LLM_MIN_CONFIDENCE = 0.6         # below this an LLM answer counts as "unsure"
LLM_SAMPLE_HEAD_CHARS = 2500     # the LLM sees the start and end of a long document,
LLM_SAMPLE_TAIL_CHARS = 1000     # not all of it

# Organizational access policy: which roles may legitimately share a document.
# A rule-based match across roles NOT paired here is a conflict and goes to the
# LLM. Executive pairs with every department.
COMPATIBLE_ROLE_PAIRS = {frozenset(p) for p in [
    ("finance", "hr"), ("finance", "legal"), ("hr", "legal"), ("finance", "sales"),
    ("finance", "executive"), ("hr", "executive"), ("legal", "executive"),
    ("engineering", "executive"), ("sales", "executive"),
]}
# Org-wide ("all") material may not be tagged above this tier.
MAX_TIER_FOR_ALL_ROLES = "internal"

# Multi-role storage: Chroma metadata filters compare scalars, so each chunk
# stores one boolean flag per role (role_hr=True, role_legal=False, ...) plus
# a human-readable comma-joined allowed_roles string. The access filter is
# then "any of the user's role flags is True" — an $or over flags.
ROLE_FLAG_PREFIX = "role_"
ALL_ROLE_VALUES = ORGANIZATIONAL_ROLES + [ALL_ROLES_PSEUDO_ROLE]


def role_flag(role: str) -> str:
    return f"{ROLE_FLAG_PREFIX}{role}"


# ---------------------------------------------------------------------------
# Hybrid retrieval (query/hybrid_retrieval.py, query/reranker.py)
# ---------------------------------------------------------------------------
ENABLE_HYBRID_RETRIEVAL = True
DENSE_CANDIDATE_K = 30
BM25_CANDIDATE_K = 30
RRF_K = 60
RERANK_SHORTLIST_SIZE = 30
CROSS_ENCODER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"