# RAG Ingestion + Query Pipeline

## Layout

```
rag_pipeline/
├── config.py                 # all tunables: paths, chunk size/overlap, model names, log settings
├── logging_setup.py           # app.log (rotating) + audit.log (structured JSON events)
├── requirements.txt
├── data/
│   ├── source_docs/          # drop raw .pdf / .docx / .doc / .txt files here
│   ├── extracted_markdown/    # per-doc .md files, tables preserved (written every ingestion)
│   ├── state/                # ingestion_manifest.json (tracks what's already ingested)
│   └── vector_store/         # Chroma's persistent DB files
├── logs/
│   ├── app.log                # general operational logs (rotating)
│   └── audit.log               # structured JSON audit trail (rotating)
├── ingestion/                 # OFFLINE pipeline
│   ├── extract.py             # plain-text extraction (pdf/docx/doc/txt) — tables NOT preserved
│   ├── markdown_extract.py     # markdown extraction — tables preserved as proper pipe tables
│   ├── chunker.py               # recursive character chunking (500 tok / 50 tok overlap)
│   ├── rules.py                   # keyword rules, tier terms, chunk-level sensitive patterns
│   ├── classify.py                # document classification: rules first, LLM fallback; chunk validation
│   ├── metadata.py                 # chunk inheritance + per-role flags, tier, tier rank, review flags
│   ├── embedder.py                # pluggable embedding model (local ST by default)
│   ├── vector_store.py             # Chroma wrapper with metadata filtering, cosine similarity
│   ├── manifest.py                  # hash-based "new or changed file" detection
│   ├── pipeline.py                   # orchestrates the full ingestion flow
│   └── watcher.py                     # triggers pipeline.run_once() — polling or event-driven
└── query/                      # ONLINE pipeline
    ├── auth.py                  # authentication + role verification (6 roles, 4 tiers)
    ├── access_control.py          # compiles a User into the conjunctive role+clearance predicate
    ├── hybrid_retrieval.py          # dense + BM25 search, fused by Reciprocal Rank Fusion
    ├── reranker.py                   # cross-encoder reranking of the fused shortlist
    ├── llm_providers.py           # common LLM interface: GeminiProvider + OllamaProvider, shared retry
    └── pipeline.py                # auth -> role check -> query refinement -> retrieval -> LLM
    (retrieve() = steps 1-5 only, reused by evaluation/)

tests/
└── test_classification.py       # 44 checks: routing, failure handling, chunk rules, access control

evaluation/                    # retrieval + access-control evaluation (Precision/Recall/F1, leak testing)
├── metrics.py                  # precision/recall/F1 computation
├── eval_retrieval.py            # "did we find the right content?"
├── eval_access_control.py        # "did we ever leak unauthorized content?"
├── test_set.example.json          # copy -> test_set.json, fill in ground truth
└── test_set_access.example.json    # copy -> test_set_access.json, fill in forbidden sources
```

## 1. Ingestion pipeline (offline, runs only on new/changed documents)

```
Raw Documents → Markdown Extraction (tables preserved) → Save .md file
→ Recursive Character Chunking (500 tok / 50 tok overlap)
→ Embedding Model → Metadata Attachment + Automatic Role/Sensitivity Classification
→ Vector Store
```

**Tables are preserved.** Plain text extraction (`ingestion/extract.py`) flattens table
cells into unstructured, jumbled text — worse, python-docx's `.paragraphs` list
skips table content entirely, so tables are silently **lost**, not just mangled.
`ingestion/markdown_extract.py` fixes this: PDF tables are detected with `pdfplumber`
and DOCX tables are walked in document order (interleaved correctly with surrounding
paragraphs), and both are rendered as real Markdown pipe tables (`| col | col |`).
Every ingestion writes the resulting Markdown to `data/extracted_markdown/{doc_id}_{name}.md`
— an inspectable, durable record of exactly what got chunked and embedded — before chunking runs.

```bash
# See it directly: compare plain extraction (table lost) vs markdown extraction (table preserved)
python inspect_doc.py --file "data/source_docs/yourfile.docx" --raw
python inspect_doc.py --file "data/source_docs/yourfile.docx" --markdown
```

**"Only when new documents are added"** is enforced by `ingestion/manifest.py`:
every source file is hashed (SHA-256), and `IngestionPipeline.run_once()` diffs
the current directory listing against `data/state/ingestion_manifest.json`.
Unchanged files are skipped entirely — no re-extraction, re-chunking, or
re-embedding. You can therefore call `run_once()` as often as you like
(cron, Lambda, CI hook) without wasting compute.

Two ways to trigger it:

```bash
# One-off / cron-friendly:
python -m ingestion.pipeline

# Polling loop (checks every 60s, no-ops when nothing changed):
python -m ingestion.watcher poll

# Real-time, event-driven (requires `watchdog`, debounces bursts of file events):
python -m ingestion.watcher watch
```

### Re-ingesting an edited document

Chunk ids are positional (`{doc_id}::chunk-{n}`), so re-ingesting overwrites chunks
`0..n-1`. If the new version has *fewer* chunks, the leftovers from the old version would
otherwise stay searchable, so the pipeline records what's stored before writing, upserts the
new chunks, and only then deletes the stale ones (`stale_chunks_removed` in `audit.log`). Deleting
*after* a successful write means a failed upsert leaves the old version intact and the file
is retried on the next run, rather than the document disappearing.

### Classification: rules first, LLM only when needed

The whole **document** is classified once (`ingestion/classify.py`); its chunks inherit the label.

1. **Rules** — weighted keyword lists (`ingestion/rules.py`) score each role and tier. If they give
   a clear answer consistent with the access policy, it's accepted and **the LLM is never called**.
   The most restrictive tier that clears the bar wins. A document may belong to several roles
   (payroll → Finance + HR); multiple matches are not treated as ambiguity.
2. **LLM fallback** — only when no rule matches, only part of the answer is known (roles but no
   tier, or the reverse), or the matched roles conflict under the policy
   (`config.COMPATIBLE_ROLE_PAIRS`; e.g. Legal + Sales). The LLM returns JSON:
   roles, tier, confidence, a one-line reason, and a `needs_review` flag. It sees the start and
   end of long documents, not all of it. One call per document at most, not per chunk.
3. **Policy check** — `"all"` can't be tagged above `internal` (it's replaced by executive-only),
   and `"all"` is dropped if specific roles are also given. Incompatible role combinations are
   retained in the audit metadata but marked for review; they are not searchable until approved.

**Failures resolve to the stricter outcome.** An unreachable LLM, an unparseable reply, an invented
role, a low-confidence answer (`< LLM_MIN_CONFIDENCE`, 0.6), or `needs_review: true` never resolves
permissively: the tier is escalated one rank and the document is flagged
(`document_flagged_for_review` in `audit.log`). If nothing at all is known it becomes
**restricted + executive-only**. Every chunk with `needs_review=True` is excluded by the
query-time access filter, for both dense and hybrid retrieval. Missing classifications also default
to restricted + executive-only + review, never public/all. Manual overrides pass through the same
role policy validation. Over-restriction is visible and self-correcting; over-permission is a silent
confidentiality breach.

**Chunk-level validation.** After chunking, each chunk inherits the document's roles and tier,
then `validate_chunk` may make it *stricter* — never looser — when the chunk itself contains
sensitive content (ID numbers, credentials, salary figures, terms like "do not distribute").
A salary table inside a public handbook becomes confidential/restricted and is narrowed to the
roles it concerns, while the rest of the handbook stays public. A property test checks 500 random
chunks: tier never lower, roles never wider.

```bash
# preview a classification without storing anything
python inspect_doc.py --file "data/source_docs/yourfile.pdf" --classify
```

To manually classify a document whose role/tier you already know, pass
`allowed_roles=` (a string like `"hr,legal"` or a list) and/or `sensitivity_tier=` to `attach_metadata()`.
The same policy validation still applies; incompatible combinations are quarantined for review.

### Metadata stored per chunk
`doc_id`, `source`, `page`, `section`, `ingested_at`, `allowed_roles` (readable, e.g. `"hr,legal"`),
one boolean flag per role (`role_hr`, `role_legal`, … `role_all`), `sensitivity_tier`,
`sensitivity_rank`, `classification_method` (`rules`/`llm`/`failsafe`/`manual`), `needs_review`,
`chunk_override`. Chroma filters compare scalars, so multi-role ownership is stored as one flag per
role; `sensitivity_rank` (`public`=0 … `restricted`=3) makes the clearance check a numeric `$lte`.

## 2. Query pipeline (online, runs on every request)

```
User → Authentication → Role Verification → User Query → LLM Query Refinement
→ Embedding Model → Access-Filtered Hybrid Retrieval (dense + BM25 + RRF)
→ Cross-Encoder Reranking → Top-k Relevant Chunks → LLM → Generated Answer
```

```python
from query.pipeline import QueryPipeline

qp = QueryPipeline()
result = qp.handle_query(auth_token="demo-token-engineer", query="What is our refund policy?")
print(result.refined_query)  # the LLM-rewritten search query
print(result.answer)
```

- **Authentication** (`query/auth.py`): validates the credential, returns a `User`
  with `roles` and `sensitivity_clearance`. Swap `authenticate_user` for your
  real IdP/JWT verification — nothing downstream needs to change.
- **Role verification**: gates whether the user may run this query at all
  (e.g. certain query types restricted to certain roles).
- **LLM query refinement** (`_refine_query`): rewrites the raw user question into a clearer,
  self-contained search query (resolves pronouns, expands ambiguous terms) before embedding it.
  Purely an enhancement — if the LLM call fails, it falls back to the original query untouched
  rather than failing the request.
- **Embedding model**: the refined query is embedded with the same model used at ingestion time.
- **Access control** (`query/access_control.py`): compiles the user into a **conjunctive** predicate —

  ```
  authorized(user, chunk) <=> any role of user is a role of chunk  AND  sensitivity_rank(chunk) ≤ clearance_rank(user)
  ```

  Chunks can belong to several roles, so the role condition is an `$or` over the user's role flags.
  Neither condition suffices alone (right department/over clearance; same tier/wrong department).
  It's evaluated *inside* retrieval (`where=`), not on its results: an unauthorized chunk is never
  a candidate, never ranked or reranked, and never reaches the LLM. Executive authority is
  membership in every departmental role — the predicate has no special case for it.
- **Hybrid retrieval** (`query/hybrid_retrieval.py`): dense (cosine) and BM25 search run over the
  *same* authorized set and fuse by Reciprocal Rank Fusion, `RRF(d) = Σ 1/(k + rank_i(d))`, `k=60`.
  Ranks are fused, not scores (cosine and BM25 are on incomparable scales); agreement between
  methods outranks a single method's pick. Dense handles paraphrase; BM25 catches exact identifiers.
- **Cross-encoder reranking** (`query/reranker.py`): the fused shortlist (30) is rescored by
  `cross-encoder/ms-marco-MiniLM-L-6-v2`, which jointly encodes (query, chunk) — finer ordering,
  too slow for the full set, hence shortlist only.
- **Top-k relevant chunks**: the reranked `top_k` hits.
- **LLM → generated answer**: `_call_llm` calls whichever provider is configured (Gemini or Ollama — see
  below) with the retrieved chunks as grounding context, retrying on transient errors.

**Graceful degradation:** if `rank_bm25` or the cross-encoder isn't installed, or hybrid retrieval
throws, the query doesn't fail — it falls back to dense-only (`retrieval_mode` = `dense_only` /
`hybrid_no_rerank` in the audit log). Disable with `QueryPipeline(enable_hybrid_retrieval=False)`.

### Switching between Gemini and Ollama

`query/llm_providers.py` defines one common interface (`BaseLLMProvider`) that both `GeminiProvider`
(cloud) and `OllamaProvider` (local) implement — `_generate(prompt)`, `is_transient(error)`,
`is_rate_limited(error)`. `QueryPipeline` only ever calls `generate_with_retry(prompt, self.llm_provider)`;
it has no idea which provider it's holding, so the retry loop, `_refine_query`, and `_call_llm` never
need to change when you switch backends. The only thing that changes is which provider object gets
constructed.

**Switch via `.env` (no code changes):**
```
LLM_PROVIDER=gemini   # or: ollama
```
`QueryPipeline()` reads this through `get_llm_provider()` automatically.

**Or pass a provider explicitly (e.g. to mix providers, or for tests):**
```python
from query.pipeline import QueryPipeline
from query.llm_providers import OllamaProvider, GeminiProvider

qp = QueryPipeline(llm_provider=OllamaProvider(model="llama3.2"))     # local, no API key
qp = QueryPipeline(llm_provider=GeminiProvider(model="gemini-2.5-flash"))  # cloud
```

Running with Ollama requires `ollama serve` running locally and the model already pulled
(`ollama pull llama3.2`) — no `GEMINI_API_KEY` needed in that mode. Each provider defines its own
notion of a "transient" error (Gemini: rate limits/5xx/network; Ollama: connection-refused/timeout/
model-still-loading), so retry behavior stays sensible for either backend without any shared-code changes.

## Setup

```bash
pip install -r requirements.txt
cp .env.example .env   # then fill in GEMINI_API_KEY (if LLM_PROVIDER=gemini)
# Legacy .doc support additionally needs one of: antiword, libreoffice (system packages)
```

`.env` is loaded automatically by `config.py` (via `python-dotenv`), so every
entry point — `run_query.py`, `api.py`, `ingestion/pipeline.py`,
`ingestion/watcher.py` — picks up the keys without extra setup. It's already
listed in `.gitignore` so it won't get committed.

### Demo users (`query/auth.py`)

In-memory stand-ins — replace with a real identity provider (JWT) before production.

| Token | Roles | Clearance |
|---|---|---|
| `demo-token-finance` | finance, all | confidential |
| `demo-token-hr` | hr, all | confidential |
| `demo-token-legal` | legal, all | restricted |
| `demo-token-engineer` | engineering, all | internal |
| `demo-token-sales` | sales, all | internal |
| `demo-token-exec` | every department, all | restricted |

## Fallbacks & resilience

Every external call in both pipelines is wrapped with retries — so a
transient network blip or a rate limit doesn't fail the whole
request/ingestion run.

**Query pipeline (`query/pipeline.py`, `query/llm_providers.py`):**
- **Embedding call**: retried with exponential backoff (`embed_retries`, default 2).
- **Vector store query**: retried with exponential backoff (`retrieval_retries`, default 2). Empty results (no matching docs) are returned as-is, not retried — that's not a failure, just nothing to show.
- **LLM generation**: retried up to `llm_max_retries` (default 4) with exponential backoff, via the shared `generate_with_retry(prompt, provider)` loop in `query/llm_providers.py` — the same retry logic applies regardless of which provider is configured. Each provider decides what counts as transient for itself: **Gemini** flags rate limits (`429`, `quota`, `resource_exhausted`) for a longer backoff and timeouts/5xx for normal backoff; **Ollama** flags connection-refused/timeout/model-still-loading as transient (it has no concept of rate limits). Non-transient errors (bad request, invalid API key) fail immediately instead of wasting retries. If the provider is still failing after all retries, the pipeline doesn't error the request out — it returns the raw retrieved excerpts instead, with `QueryResult.degraded=True` so the caller can show an "answer generation unavailable" notice.

Tune retry behavior:
```python
from query.pipeline import QueryPipeline

qp = QueryPipeline(llm_max_retries=6)  # more patience under heavy rate limiting
```

**Ingestion pipeline (`ingestion/pipeline.py`):**
- Embedding and vector-store-write steps are retried with exponential backoff per file.
- Failures are isolated per file — one bad/unreachable document doesn't block the rest of the batch, and is only marked "ingested" in the manifest on success, so it's automatically retried on the *next* run instead of being silently skipped.

## Evaluating retrieval quality (Precision / Recall / F1)

Two separate evaluations under `evaluation/`, both directly matching your synopsis's evaluation
section (retrieval accuracy, false-positive/false-negative rates, leak testing):

**1. Retrieval accuracy** — "did we find the right content?"
```bash
cp evaluation/test_set.example.json evaluation/test_set.json
# edit test_set.json: real queries + which sources/chunks are actually relevant
python -m evaluation.eval_retrieval --test-set evaluation/test_set.json --top-k 5
```
Reports Precision@k, Recall@k, and F1, per query and macro-averaged. You define ground truth two ways:
`relevant_sources` (document-level, easier to label by hand) or `relevant_chunk_ids` (chunk-level,
more precise — get exact IDs with `python inspect_doc.py --file <path> --stored`).

**2. Access-control leak testing** — "did we ever return content this user isn't authorized to see?"
```bash
cp evaluation/test_set_access.example.json evaluation/test_set_access.json
# edit: for each (query, user) pair, list forbidden_sources that must NEVER appear
python -m evaluation.eval_access_control --test-set evaluation/test_set_access.json
```
Reports a leak rate (fraction of queries where a forbidden source leaked through) and flags exactly
which sources leaked, for exactly which query/user — this is the false-positive-rate metric for RBAC
itself, distinct from retrieval relevance. It can also flag over-blocking (`expected_accessible_sources`
that never showed up), which usually means a `sensitivity_tier`/`allowed_roles` metadata mismatch.

Both scripts run through `QueryPipeline.retrieve()` — the exact same auth → role-check → refine →
embed → cosine-similarity-search path a real query uses — with `enable_query_refinement=False` by
default so results are deterministic and don't cost an LLM call per test item. `handle_query()` (used
by `run_query.py`/`api.py`) calls `retrieve()` internally and then adds the final LLM generation step.

## Checking extracted / chunked / stored content

`inspect_doc.py` lets you verify each stage of ingestion without re-running embedding or writing anything:

```bash
# Plain extraction (tables NOT preserved — for comparison/debugging only)
python inspect_doc.py --file "data/source_docs/Major project Synopsis.pdf" --raw

# Markdown extraction — tables ARE preserved, and this writes the .md file to disk.
# This is what actually gets chunked.
python inspect_doc.py --file "data/source_docs/Major project Synopsis.pdf" --markdown

# How it would be split into chunks (500 tok / 50 tok overlap) — still no storage
python inspect_doc.py --file "data/source_docs/Major project Synopsis.pdf" --chunks

# What's actually sitting in the vector store right now (i.e. what a query can retrieve)
python inspect_doc.py --file "data/source_docs/Major project Synopsis.pdf" --stored
```

`--stored` looks the document up by `doc_id` (the same hash-of-path used at ingestion) directly against Chroma's metadata — no embedding call needed — so it's the fastest way to confirm ingestion actually wrote what you expect, or to check for the stale-chunk issue if you've been editing a document (see `VectorStore.delete_by_doc_id`).

## Logging

Two separate log files, both under `logs/` and both rotating (5MB per file, 5 backups kept):

- **`logs/app.log`** — general operational logging (info/warning/error), human-readable. Same content as what prints to the console, kept for later debugging.
- **`logs/audit.log`** — structured, newline-delimited JSON. One line per security/activity event, matching the "Audit Logs / Query Logs / Access Logs / Monitoring" component in the RB-RAG architecture. Events currently logged:

  | Event | When | Fields |
  |---|---|---|
  | `auth_success` / `auth_failed` | Every authentication attempt | `user_id`, `roles`, `clearance` (or just a token prefix on failure) |
  | `role_verification_passed` / `role_verification_failed` | After auth, before any retrieval | `user_id`, `required_roles` |
  | `query_submitted` | Every query, right after role check | `user_id`, `query` |
  | `query_refined` | Only if the LLM rewrote the query | `original_query`, `refined_query` |
  | `retrieval` | After the cosine similarity search | `chunk_count`, `sources`, `doc_ids`, `similarities` |
  | `answer_generated` | After the final LLM call | `degraded` (True if the LLM failed and raw excerpts were returned instead), `answer_length` |
  | `document_ingested` | Ingestion pipeline, per file | `doc_id`, `source`, `chunk_count` |
  | `document_ingestion_failed` | Ingestion pipeline, per file | `source`, `reason` |

  Being newline-delimited JSON, it's directly loadable with `pandas.read_json("logs/audit.log", lines=True)`, `jq`, or any log-shipping tool — useful for the "false-positive/false-negative analysis" and "leak testing" your synopsis's evaluation section calls for, since you can reconstruct exactly which chunks were returned to which user for any query.

Both logs are configured in `logging_setup.py` and initialized automatically the first time any pipeline module is imported — no manual setup needed. Tune `LOG_LEVEL`, `LOG_MAX_BYTES`, `LOG_BACKUP_COUNT` in `config.py` (or set `LOG_LEVEL=DEBUG` in `.env` for more verbose app.log output).

## Extending

| Want to change...            | Edit...                                   |
|-------------------------------|--------------------------------------------|
| Chunk size / overlap           | `config.py` (`CHUNK_SIZE_TOKENS`, `CHUNK_OVERLAP_TOKENS`) |
| Embedding model / provider      | `ingestion/embedder.py`                    |
| Vector DB backend                | `ingestion/vector_store.py`                |
| Add a new input file type         | `ingestion/extract.py`                     |
| Access-control granularity         | `ingestion/metadata.py`, `query/auth.py`, `query/pipeline.py._build_access_filter` |
