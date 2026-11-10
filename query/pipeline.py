"""
Query pipeline (online) — executes on every user request:

    User Query -> LLM Query Refinement -> Embedding Model
    -> Access-Filtered Hybrid Retrieval (dense + BM25 + RRF)
    -> Cross-Encoder Reranking -> LLM -> Generated Answer

Authentication and role verification still gate the request before any of
this runs, and the user's access level is folded into the similarity search
as a metadata filter — so "Cosine Similarity Search" here means "cosine
similarity search restricted to chunks this user is authorized to see."
"""

from __future__ import annotations
import logging
import time
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple

from query.auth import authenticate_user, verify_role, User, AuthenticationError
from query.access_control import build_access_filter
from query.llm_providers import generate_with_retry, get_llm_provider, BaseLLMProvider, LLMError
from query.hybrid_retrieval import hybrid_retrieve
from query.reranker import rerank as cross_encoder_rerank
from ingestion.embedder import BaseEmbedder, get_default_embedder
from ingestion.vector_store import VectorStore
from logging_setup import setup_logging, get_audit_logger
from config import ENABLE_HYBRID_RETRIEVAL, RERANK_SHORTLIST_SIZE

setup_logging()
logger = logging.getLogger("query.pipeline")
audit = get_audit_logger()

QUERY_REFINEMENT_PROMPT = (
    "Rewrite the user's question into a clear, specific search query optimized "
    "for retrieving relevant passages from a document collection. Expand "
    "ambiguous terms, resolve pronouns, and make it self-contained. "
    "Return ONLY the rewritten query text, with no explanation or quotes.\n\n"
    "User question: {query}"
)


@dataclass
class RetrievalResult:
    user: User
    refined_query: str
    hits: List[Dict]


@dataclass
class QueryResult:
    answer: str
    source_chunks: List[Dict]
    refined_query: str
    degraded: bool = False  # True if we fell back to a non-LLM answer


class QueryPipeline:
    def __init__(
        self,
        embedder: Optional[BaseEmbedder] = None,
        vector_store: Optional[VectorStore] = None,
        top_k: int = 5,
        llm_provider: Optional[BaseLLMProvider] = None,
        llm_max_retries: int = 4,
        embed_retries: int = 2,
        retrieval_retries: int = 2,
        enable_query_refinement: bool = True,
        enable_hybrid_retrieval: bool = ENABLE_HYBRID_RETRIEVAL,
        rerank_shortlist_size: int = RERANK_SHORTLIST_SIZE,
    ):
        self.embedder = embedder or get_default_embedder()
        self.vector_store = vector_store or VectorStore()
        self.top_k = top_k
        # Resolved from config.LLM_PROVIDER (env var LLM_PROVIDER=gemini|ollama)
        # unless a specific provider instance is passed in. Every call site
        # below (_refine_query, _call_llm) just calls generate_with_retry(...,
        # self.llm_provider) — switching providers never touches this code.
        self.llm_provider = llm_provider or get_llm_provider()
        self.llm_max_retries = llm_max_retries
        self.embed_retries = embed_retries
        self.retrieval_retries = retrieval_retries
        self.enable_query_refinement = enable_query_refinement
        self.enable_hybrid_retrieval = enable_hybrid_retrieval
        self.rerank_shortlist_size = rerank_shortlist_size

    def retrieve(self, auth_token: str, query: str, required_roles: Optional[List[str]] = None) -> RetrievalResult:
        """
        Runs steps 1–5 only (auth -> role check -> query refinement -> embed
        -> cosine similarity search), without calling the LLM for a final
        answer. This is the exact same retrieval path handle_query() uses —
        pulled out separately so evaluation scripts (see evaluation/) can
        measure retrieval precision/recall without paying for or waiting on
        LLM generation on every run.
        """
        # 1. Authentication
        try:
            user = authenticate_user(auth_token)  # raises AuthenticationError if invalid
        except AuthenticationError:
            audit.log("auth_failed", auth_token_prefix=auth_token[:8])
            raise
        audit.log("auth_success", user_id=user.user_id, roles=user.roles, clearance=user.sensitivity_clearance)

        # 2. Role verification (gate on whether this user may query at all)
        if not verify_role(user, required_roles):
            audit.log("role_verification_failed", user_id=user.user_id, roles=user.roles, required_roles=required_roles)
            raise PermissionError(f"User {user.user_id} lacks required role(s): {required_roles}")
        audit.log("role_verification_passed", user_id=user.user_id, required_roles=required_roles)

        audit.log("query_submitted", user_id=user.user_id, query=query)

        # 3. User Query -> LLM Query Refinement
        refined_query = self._refine_query(query)
        if refined_query != query:
            audit.log("query_refined", user_id=user.user_id, original_query=query, refined_query=refined_query)

        # 4. Embedding Model
        query_embedding = self._embed_with_retry(refined_query)

        # 5. Cosine Similarity Search -> Top-k Relevant Chunks
        #    (restricted to the user's access level via metadata filter)
        where_filter = build_access_filter(user)  # conjunctive role + clearance predicate
        hits, retrieval_mode = self._retrieve_with_fallback(query_embedding, refined_query, where_filter)
        audit.log(
            "retrieval",
            user_id=user.user_id,
            retrieval_mode=retrieval_mode,
            chunk_count=len(hits),
            sources=[h["metadata"]["source"] for h in hits],
            doc_ids=[h["metadata"]["doc_id"] for h in hits],
            scores=[
                round(h["rerank_score"], 4) if h.get("rerank_score") is not None
                else round(h["similarity"], 4) if h.get("similarity") is not None
                else None
                for h in hits
            ],
        )

        return RetrievalResult(user=user, refined_query=refined_query, hits=hits)

    def handle_query(self, auth_token: str, query: str, required_roles: Optional[List[str]] = None) -> QueryResult:
        retrieval = self.retrieve(auth_token, query, required_roles)

        # 6. LLM -> Generated Answer
        answer, degraded = self._call_llm(query, retrieval.hits)
        audit.log("answer_generated", user_id=retrieval.user.user_id, degraded=degraded, answer_length=len(answer))

        return QueryResult(
            answer=answer,
            source_chunks=retrieval.hits,
            refined_query=retrieval.refined_query,
            degraded=degraded,
        )

    # -- LLM Query Refinement ---------------------------------------------
    def _refine_query(self, query: str) -> str:
        """
        Rewrites the raw user query into a more retrieval-friendly form
        before embedding. Falls back to the original query untouched if
        refinement is disabled or the LLM call fails — refinement is an
        enhancement, not a hard dependency for the pipeline to function.
        """
        if not self.enable_query_refinement:
            return query

        prompt = QUERY_REFINEMENT_PROMPT.format(query=query)
        try:
            refined = generate_with_retry(prompt, self.llm_provider, max_retries=self.llm_max_retries)
            refined = refined.strip().strip('"')
            return refined if refined else query
        except LLMError as e:
            logger.warning(f"Query refinement failed, falling back to original query: {e}")
            return query

    # -- Fallback: embedding call -------------------------------------
    def _embed_with_retry(self, query: str) -> List[float]:
        last_error = None
        for attempt in range(self.embed_retries + 1):
            try:
                return self.embedder.embed([query])[0]
            except Exception as e:  # noqa: BLE001
                last_error = e
                if attempt < self.embed_retries:
                    delay = 1.0 * (2 ** attempt)
                    logger.warning(f"Embedding call failed (attempt {attempt + 1}), retrying in {delay:.1f}s: {e}")
                    time.sleep(delay)
        raise RuntimeError(f"Embedding model unavailable after {self.embed_retries + 1} attempts: {last_error}")

    # -- Retrieval: hybrid (dense + BM25 + RRF + cross-encoder), degrading to dense-only --
    def _retrieve_with_fallback(
        self, query_embedding: List[float], query_text: str, where_filter: Dict
    ) -> Tuple[List[Dict], str]:
        """Returns (hits, retrieval_mode): "hybrid", "hybrid_no_rerank" (cross-encoder
        unavailable), or "dense_only". A missing optional dependency degrades
        retrieval quality; it never fails the query."""
        if self.enable_hybrid_retrieval:
            try:
                fused = hybrid_retrieve(
                    self.vector_store, query_embedding, query_text, where_filter,
                    shortlist_size=self.rerank_shortlist_size,
                )
                if not fused:
                    return [], "hybrid"
                try:
                    return cross_encoder_rerank(query_text, fused, top_k=self.top_k), "hybrid"
                except ImportError as e:
                    logger.warning(f"Cross-encoder unavailable, returning fused (unreranked) shortlist: {e}")
                    return fused[: self.top_k], "hybrid_no_rerank"
            except ImportError as e:
                logger.warning(f"Hybrid retrieval dependency unavailable, falling back to dense-only: {e}")
            except Exception as e:  # noqa: BLE001
                logger.warning(f"Hybrid retrieval failed, falling back to dense-only: {e}")

        last_error = None
        for attempt in range(self.retrieval_retries + 1):
            try:
                return self.vector_store.query(query_embedding, top_k=self.top_k, where=where_filter), "dense_only"
            except Exception as e:  # noqa: BLE001
                last_error = e
                if attempt < self.retrieval_retries:
                    delay = 1.0 * (2 ** attempt)
                    logger.warning(f"Vector store query failed (attempt {attempt + 1}), retrying in {delay:.1f}s: {e}")
                    time.sleep(delay)
        raise RuntimeError(f"Vector store unavailable after {self.retrieval_retries + 1} attempts: {last_error}")

    # -- LLM call: common interface, retry logic shared across providers --
    def _call_llm(self, query: str, hits: List[Dict]) -> tuple[str, bool]:
        """
        Returns (answer, degraded). `degraded=True` means the LLM provider
        failed after exhausting retries and we fell back to a raw-excerpt
        answer instead of erroring the whole request out.
        """
        if not hits:
            return "No documents matched this query at your access level.", False

        context = "\n\n".join(
            f"[Source: {h['metadata']['source']} p.{h['metadata'].get('page')}]\n{h['text']}"
            for h in hits
        )
        prompt = (
            "Answer the user's question using only the context below. "
            "If the context doesn't contain the answer, say so.\n\n"
            f"Context:\n{context}\n\nQuestion: {query}"
        )

        try:
            answer = generate_with_retry(prompt, self.llm_provider, max_retries=self.llm_max_retries)
            return answer, False
        except LLMError as e:
            logger.error(f"[{self.llm_provider.name}] call failed after all retries: {e}")
            # Last-resort fallback: don't fail the request, surface the raw
            # retrieved excerpts so the user still gets something useful.
            excerpt_answer = (
                "The answer-generation model is currently unavailable. Here are "
                "the most relevant source excerpts instead:\n\n" + context[:2000]
            )
            return excerpt_answer, True
