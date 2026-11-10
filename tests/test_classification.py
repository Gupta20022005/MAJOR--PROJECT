"""
Tests for rules-first document classification, chunk inheritance with
restrictive-only validation, multi-role storage, and access enforcement.

Run from the project root:   python -m tests.test_classification

No network, no real LLM, no embedding model: the LLM is replaced by a
scripted fake and embeddings by a deterministic stand-in. What this proves
is the LOGIC (routing, failure handling, access control); it says nothing
about how accurate the keyword lists or a real LLM would be on your documents.
"""

import json
import os
import random
import shutil
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import ingestion.classify as clf
from ingestion.classify import classify_document, validate_chunk, DocumentClassification
from ingestion.metadata import attach_metadata
from ingestion.pipeline import IngestionPipeline
from ingestion.vector_store import VectorStore
from query.llm_providers import LLMError
from query.pipeline import QueryPipeline
from config import TIER_RANK, SENSITIVITY_TIERS
from query.access_control import build_access_filter
from query.auth import User

PASSED, FAILED = [], []


def check(name, cond, detail=""):
    (PASSED if cond else FAILED).append(name)
    print(f"  {'PASS' if cond else 'FAIL'}  {name}" + (f"   [{detail}]" if detail and not cond else ""))


class FakeEmbedder:
    def embed(self, texts):
        return [[float((hash(t) >> i) % 100) / 100 for i in range(8)] for t in texts]


class ScriptedLLM:
    """Replaces generate_with_retry inside ingestion.classify."""
    def __init__(self):
        self.calls, self.reply = 0, None

    def __call__(self, prompt, provider, max_retries=2):
        self.calls += 1
        if isinstance(self.reply, Exception):
            raise self.reply
        return self.reply


llm = ScriptedLLM()
clf.generate_with_retry = llm
PROVIDER = object()


def doc(text, name="doc.txt", reply=None):
    llm.calls, llm.reply = 0, reply
    return classify_document(text, name, PROVIDER)


def js(roles, tier, conf=0.9, review=False, reason="because"):
    return json.dumps({"roles": roles, "tier": tier, "confidence": conf, "reason": reason, "needs_review": review})


HR_TEXT = "Employee performance review and appraisal. Disciplinary action and grievance handling. Confidential. " * 3
PAYROLL_TEXT = "Payroll and salary budget. Employee compensation and benefits. Payroll audit ledger. Confidential. " * 3
VAGUE_TEXT = "Quarterly notes about the new initiative and general discussion topics. " * 3

# ---------------------------------------------------------------- routing
print("\nRules first, LLM only when needed")
d = doc(HR_TEXT)
check("clear rules accepted without calling the LLM", d.method == "rules" and llm.calls == 0, f"{d.method}/{llm.calls}")
check("rules gave HR + confidential", d.roles == ["hr"] and d.tier == "confidential", f"{d.roles}/{d.tier}")

d = doc(PAYROLL_TEXT)
check("multiple valid roles are NOT treated as ambiguity", d.method == "rules" and set(d.roles) == {"finance", "hr"} and llm.calls == 0,
      f"{d.method}/{d.roles}/{llm.calls}")

d = doc(VAGUE_TEXT, reply=js(["sales"], "internal"))
check("no rule match -> LLM classifies", d.method == "llm" and llm.calls == 1 and d.roles == ["sales"], f"{d.method}/{llm.calls}")
check("confident LLM answer not flagged", not d.needs_review and d.tier == "internal")

conflict = "Litigation settlement counsel. Sales pipeline quota prospect deal renewal. Internal draft. " * 3
d = doc(conflict, reply=js(["legal"], "confidential"))
check("policy-conflicting rule roles (legal+sales) -> LLM", d.method == "llm" and llm.calls == 1 and d.rule_status == "conflict",
      f"{d.method}/{d.rule_status}")

partial = "Invoice budget revenue accounting audit ledger expense. " * 3     # roles known, no tier keywords
d = doc(partial, reply=js(["finance"], "internal"))
check("roles known but tier unknown (partial) -> LLM", d.method == "llm" and d.rule_status == "partial")

public = "Press release and public announcement. Marketing brochure. " * 3
d = doc(public)
check("public material with no role -> 'all', no LLM", d.roles == ["all"] and d.tier == "public" and llm.calls == 0)

# ---------------------------------------------------------------- failures
print("\nFailures resolve to the stricter outcome")
d = doc(VAGUE_TEXT, reply=LLMError("down"))
check("LLM down + nothing known -> restricted, executive only, review",
      d.method == "failsafe" and d.tier == "restricted" and d.roles == ["executive"] and d.needs_review, f"{d.roles}/{d.tier}")

d = doc(partial, reply=LLMError("down"))
check("LLM down + roles known, tier unknown -> internal escalated to confidential",
      d.roles == ["finance"] and d.tier == "confidential" and d.needs_review, f"{d.roles}/{d.tier}")

d = doc(conflict, reply=LLMError("down"))
check("LLM down + conflict -> narrowed to single strongest role, escalated, review",
      len(d.roles) == 1 and d.needs_review and TIER_RANK[d.tier] >= TIER_RANK["confidential"], f"{d.roles}/{d.tier}")

d = doc(VAGUE_TEXT, reply="I think this is about sales, not sure")
check("unparseable LLM reply -> failsafe", d.method == "failsafe" and d.needs_review)

d = doc(VAGUE_TEXT, reply=js(["wizard"], "internal"))
check("LLM invents a role -> rejected, failsafe", d.method == "failsafe")

d = doc(VAGUE_TEXT, reply=js(["sales"], "internal", conf=0.3))
check("unsure LLM (low confidence) -> tier escalated + review",
      d.tier == "confidential" and d.needs_review, f"{d.tier}")

d = doc(VAGUE_TEXT, reply=js(["sales"], "internal", conf=0.95, review=True))
check("LLM asks for review -> tier escalated + review", d.tier == "confidential" and d.needs_review)

d = doc(VAGUE_TEXT, reply=js(["all"], "confidential"))
check("policy: 'all' above internal is replaced by executive", d.roles == ["executive"] and d.needs_review, f"{d.roles}")

d = doc(VAGUE_TEXT, reply=js(["hr", "finance"], "confidential"))
check("LLM may assign several roles", set(d.roles) == {"hr", "finance"} and not d.needs_review)

d = doc(VAGUE_TEXT, reply="Sure!\n```json\n" + js(["legal"], "restricted") + "\n```")
check("JSON wrapped in prose/code fence still parses", d.method == "llm" and d.roles == ["legal"])

# ---------------------------------------------------------------- chunk validation
print("\nChunk validation: inherit, then tighten only")
pub = DocumentClassification(["all"], "public", "rules", None, "t", False)
cc = validate_chunk("Employee record. SSN 123-45-6789 on file.", pub)
check("SSN in a public chunk -> restricted and no longer 'all'",
      cc.tier == "restricted" and cc.roles != ["all"] and cc.overridden, f"{cc.roles}/{cc.tier}")

cc = validate_chunk("Opening hours are 9 to 5.", pub)
check("benign chunk inherits unchanged", cc.tier == "public" and cc.roles == ["all"] and not cc.overridden)

dept = DocumentClassification(["hr", "finance"], "internal", "rules", None, "t", False)
cc = validate_chunk("Account number 123456789012 for the finance ledger audit budget and invoice.", dept)
check("sensitive chunk raises tier and narrows multi-role doc",
      cc.tier == "confidential" and cc.roles == ["finance"], f"{cc.roles}/{cc.tier}")

cc = validate_chunk("The api key = sk_live_abcdef123456 was rotated.", DocumentClassification(["engineering"], "internal", "rules", None, "t", False))
check("credential in a chunk -> restricted", cc.tier == "restricted")

# property test: a chunk is NEVER looser than its document
random.seed(7)
snippets = ["salary payroll", "SSN 123-45-6789", "press release", "do not distribute", "api = 1", "hello world",
            "litigation settlement counsel", "confidential memo board merger", "account number 123456789012", "PAN ABCDE1234F"]
role_sets = [["all"], ["hr"], ["finance"], ["hr", "finance"], ["legal", "executive"], ["engineering"]]
violations = 0
for _ in range(500):
    base = DocumentClassification(random.choice(role_sets), random.choice(SENSITIVITY_TIERS), "rules", None, "t", False)
    text = " ".join(random.sample(snippets, random.randint(1, 4)))
    r = validate_chunk(text, base)
    tier_ok = TIER_RANK[r.tier] >= TIER_RANK[base.tier]
    roles_ok = base.roles == ["all"] or set(r.roles) <= set(base.roles)
    violations += not (tier_ok and roles_ok)
check("500 random chunks: tier never lower, roles never wider than the document", violations == 0, f"{violations} violations")

# ---------------------------------------------------------------- metadata
print("\nMulti-role metadata")
chunks = [{"text": "plain text", "page": None, "section": None, "chunk_index": 0}]
m = attach_metadata(chunks, "x/doc.txt", doc_classification=DocumentClassification(["hr", "legal"], "confidential", "rules", None, "t", False))[0]["metadata"]
check("flags set per role", m["role_hr"] and m["role_legal"] and not m["role_finance"] and not m["role_all"])
check("readable allowed_roles + numeric rank", m["allowed_roles"] == "hr,legal" and m["sensitivity_rank"] == 2)
m = attach_metadata(chunks, "x/doc.txt", allowed_roles="finance", sensitivity_tier="internal")[0]["metadata"]
check("manual override still works", m["role_finance"] and m["classification_method"] == "manual")
try:
    attach_metadata(chunks, "x/doc.txt", allowed_roles="wizard")
    check("invalid manual role rejected", False)
except ValueError:
    check("invalid manual role rejected", True)

# Fail-closed metadata and retrieval policy
print("\nFail-closed approval gate")
default_meta = attach_metadata(chunks, "x/unclassified.txt")[0]["metadata"]
check("missing classification defaults to restricted executive-only review",
      default_meta["needs_review"] is True and default_meta["sensitivity_tier"] == "restricted"
      and default_meta["role_executive"] and not default_meta["role_all"])
manual_bad = attach_metadata(chunks, "x/bad-manual.txt", allowed_roles=["engineering", "legal"],
                             sensitivity_tier="internal")[0]["metadata"]
check("incompatible manual roles are quarantined", manual_bad["needs_review"] is True)
invalid_doc = DocumentClassification(["engineering", "sales"], "internal", "manual", None, "test", False)
invalid_meta = attach_metadata(chunks, "x/bad-doc.txt", doc_classification=invalid_doc)[0]["metadata"]
check("directly supplied invalid classification is revalidated", invalid_meta["needs_review"] is True)
flt = build_access_filter(User("test", ["finance"], "restricted"))
check("retrieval filter excludes documents needing review", {"needs_review": False} in flt["$and"])

# ---------------------------------------------------------------- end to end
print("\nEnd to end: ingestion -> storage -> access control")
try:
    import chromadb  # noqa: F401
    HAS_CHROMA = True
except ImportError:
    HAS_CHROMA = False
    print("  SKIP  end-to-end vector-store tests (install chromadb to run them)")

if HAS_CHROMA:
    tmp = tempfile.mkdtemp()
    try:
        docs = {
            "hr_policy.txt": HR_TEXT,
            "payroll_summary.txt": PAYROLL_TEXT,
            "company_faq.txt": "Press release and public announcement for all staff. Marketing brochure. " * 3,
            "eng_spec.txt": "System architecture specification for the microservice deployment. API schema and codebase changelog. Internal draft. " * 3,
            "legal_memo.txt": "Litigation strategy memo. Settlement position and counsel advice. Do not distribute. " * 3,
            "vague_notes.txt": VAGUE_TEXT,
        }
        for n, t in docs.items():
            open(os.path.join(tmp, n), "w").write(t)

        vs = VectorStore(persist_dir=os.path.join(tmp, "vs"), collection_name="test_collection")
        emb = FakeEmbedder()
        pipe = IngestionPipeline(embedder=emb, vector_store=vs, llm_provider=PROVIDER, source_dir=tmp)

        llm.calls, llm.reply = 0, js(["sales"], "internal")
        for n in docs:
            pipe.ingest_file(os.path.join(tmp, n))
        check("LLM called once total (only the vague document), not once per chunk", llm.calls == 1, f"{llm.calls} calls")

        import query.pipeline as qpm
        qpm.generate_with_retry = lambda *a, **k: "ok"
        expected = {
            "demo-token-finance": {"payroll_summary.txt", "company_faq.txt"},
            "demo-token-hr": {"hr_policy.txt", "payroll_summary.txt", "company_faq.txt"},
            "demo-token-legal": {"legal_memo.txt", "company_faq.txt"},
            "demo-token-engineer": {"eng_spec.txt", "company_faq.txt"},
            "demo-token-sales": {"vague_notes.txt", "company_faq.txt"},
            "demo-token-exec": set(docs),
        }
        for hybrid in (False, True):
            qp = QueryPipeline(embedder=emb, vector_store=vs, top_k=20, enable_query_refinement=False,
                               enable_hybrid_retrieval=hybrid, llm_provider=PROVIDER)
            label = "hybrid" if hybrid else "dense"
            for token, exp in expected.items():
                got = {h["metadata"]["source"] for h in qp.retrieve(token, "payroll policy memo notes").hits}
                ok = got <= exp and (hybrid or got == exp)
                check(f"[{label}] {token:20s} sees only {sorted(exp)}" if not ok else f"[{label}] {token:20s} access correct", ok,
                      f"got {sorted(got)}")

        # shared document is visible to BOTH of its roles, hidden from the rest
        fin = {h["metadata"]["source"] for h in QueryPipeline(embedder=emb, vector_store=vs, top_k=20, enable_query_refinement=False,
               enable_hybrid_retrieval=False, llm_provider=PROVIDER).retrieve("demo-token-finance", "x").hits}
        check("multi-role payroll document reachable via its finance role", "payroll_summary.txt" in fin)

        # chunk-level override inside a real ingestion
        long_public = ("Press release and public announcement about our marketing brochure. " * 60 +
                       "\n\nEmployee record: SSN 123-45-6789 held on file.\n\n" +
                       "Press release and public announcement about our marketing brochure. " * 60)
        p = os.path.join(tmp, "mixed.txt")
        open(p, "w").write(long_public)
        n = pipe.ingest_file(p)
        from ingestion.metadata import make_doc_id
        stored = vs.get_by_doc_id(make_doc_id(p))
        tiers = sorted({c["metadata"]["sensitivity_tier"] for c in stored})
        ssn = [c for c in stored if "123-45-6789" in c["text"]]
        check("one document, mixed sensitivity: only the SSN chunk is tightened",
              n > 1 and ssn and ssn[0]["metadata"]["sensitivity_tier"] == "restricted"
              and any(c["metadata"]["sensitivity_tier"] == "public" for c in stored), f"tiers={tiers}")
        sales = {h["id"] for h in QueryPipeline(embedder=emb, vector_store=vs, top_k=50, enable_query_refinement=False,
                 enable_hybrid_retrieval=False, llm_provider=PROVIDER).retrieve("demo-token-sales", "SSN record").hits}
        check("sales user cannot retrieve the SSN chunk of a public document", ssn[0]["id"] not in sales)

        # stale-chunk cleanup still works with the new flow
        q = os.path.join(tmp, "shrink.txt")
        open(q, "w").write("\n\n".join("Press release announcement. " * 150 for _ in range(6)))
        n1 = pipe.ingest_file(q)
        open(q, "w").write("Press release announcement. " * 10)
        n2 = pipe.ingest_file(q)
        left = len(vs.get_by_doc_id(make_doc_id(q)))
        check("shrunk document leaves no stale chunks", n1 > n2 and left == n2, f"{n1}->{n2}, stored {left}")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

print(f"\n{len(PASSED)} passed, {len(FAILED)} failed")
if FAILED:
    print("FAILED:", *FAILED, sep="\n  - ")
    sys.exit(1)
