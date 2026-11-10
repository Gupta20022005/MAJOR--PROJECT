"""
Ingestion pipeline orchestrator (offline):

    Raw Documents -> Markdown Extraction (tables preserved) -> Save .md file
    -> Recursive Character Chunking -> Metadata Attachment -> Embedding Model
    -> Vector Store

Runs only over files that are new or changed since the last run, tracked
via ingestion/manifest.py. Call `run_once()` from a cron job, a file-watcher
(see watcher.py), or a CI hook — it is idempotent and safe to call anytime.
"""

from __future__ import annotations
import logging
import os
import time
from typing import List, Optional

from config import SOURCE_DOCS_DIR, SUPPORTED_EXTENSIONS
from logging_setup import setup_logging, get_audit_logger
from ingestion.extract import ExtractionError
from ingestion.markdown_extract import extract_to_markdown_blocks, save_markdown_file
from ingestion.chunker import chunk_extracted_blocks
from ingestion.metadata import attach_metadata, make_doc_id
from ingestion.classify import classify_document
from ingestion.embedder import BaseEmbedder, get_default_embedder
from ingestion.vector_store import VectorStore
from ingestion.manifest import load_manifest, save_manifest, is_new_or_changed, mark_ingested
from query.llm_providers import BaseLLMProvider, get_llm_provider

setup_logging()
logger = logging.getLogger("ingestion.pipeline")
audit = get_audit_logger()

TRANSIENT_ERROR_SUBSTRINGS = (
    "timeout", "timed out", "rate limit", "429", "500", "502", "503", "504",
    "overloaded", "unavailable", "connection",
)


def _is_transient(error: Exception) -> bool:
    return any(sub in str(error).lower() for sub in TRANSIENT_ERROR_SUBSTRINGS)


class IngestionPipeline:
    def __init__(
        self,
        embedder: Optional[BaseEmbedder] = None,
        vector_store: Optional[VectorStore] = None,
        llm_provider: Optional[BaseLLMProvider] = None,
        source_dir: str = SOURCE_DOCS_DIR,
        max_retries: int = 2,
        auto_classify: bool = True,
    ):
        self.embedder = embedder or get_default_embedder()
        self.vector_store = vector_store or VectorStore()
        # Classification's LLM signal uses the same provider abstraction as the
        # query pipeline (Gemini/Ollama via config.LLM_PROVIDER).
        self.llm_provider = llm_provider or get_llm_provider()
        self.auto_classify = auto_classify
        self.source_dir = source_dir
        self.max_retries = max_retries

    def discover_files(self) -> List[str]:
        os.makedirs(self.source_dir, exist_ok=True)
        found = []
        for root, _, files in os.walk(self.source_dir):
            for name in files:
                ext = os.path.splitext(name)[1].lower()
                if ext in SUPPORTED_EXTENSIONS:
                    found.append(os.path.join(root, name))
        return found

    def _with_retries(self, fn, description: str):
        """Retries transient failures (network/rate-limit) with exponential
        backoff. Non-transient errors (bad input, auth) fail immediately."""
        last_error = None
        for attempt in range(self.max_retries + 1):
            try:
                return fn()
            except Exception as e:  # noqa: BLE001
                last_error = e
                if attempt < self.max_retries and _is_transient(e):
                    delay = 1.0 * (2 ** attempt)
                    logger.warning(
                        f"{description} failed (attempt {attempt + 1}/{self.max_retries + 1}), "
                        f"retrying in {delay:.1f}s: {e}"
                    )
                    time.sleep(delay)
                    continue
                raise
        raise last_error  # pragma: no cover — unreachable, satisfies type checkers

    def ingest_file(self, file_path: str) -> int:
        """Markdown extraction (tables preserved) -> document-level
        classification (rules first, LLM only if needed) -> chunking ->
        chunk inheritance + restrictive-only validation -> embedding -> storage.
        Returns the number of chunks written."""
        logger.info(f"Ingesting: {file_path}")

        # 1. Extract to Markdown — tables come out as real Markdown tables.
        blocks = extract_to_markdown_blocks(file_path)

        # 2. Save the Markdown (data/extracted_markdown/) — an inspectable
        #    record of exactly what will be classified and chunked.
        md_path = save_markdown_file(file_path, blocks)
        logger.info(f"  -> markdown saved to {md_path}")

        # 3. Classify the WHOLE document once. Keyword rules decide when they
        #    can; the LLM is called only when they can't (one call per
        #    document at most — not per chunk).
        doc_cls = None
        if self.auto_classify:
            full_text = "\n\n".join(b["text"] for b in blocks)
            doc_cls = classify_document(
                full_text, os.path.basename(file_path), self.llm_provider,
                doc_id=make_doc_id(file_path),
            )
            logger.info(f"  -> classified by {doc_cls.method}: roles={doc_cls.roles}, tier={doc_cls.tier}"
                        + (" [NEEDS REVIEW]" if doc_cls.needs_review else ""))

        # 4. Chunk the Markdown (tables stay intact if they fit one chunk).
        raw_chunks = chunk_extracted_blocks(blocks)
        if not raw_chunks:
            logger.warning(f"No chunks produced for {file_path}, skipping.")
            return 0

        # 5. Chunks inherit the document's classification; a validation pass
        #    may tighten individual chunks (never loosen them).
        enriched = attach_metadata(raw_chunks, file_path, doc_classification=doc_cls)
        texts = [c["text"] for c in raw_chunks]

        # 6. Embed.
        embeddings = self._with_retries(lambda: self.embedder.embed(texts), "Embedding call")

        doc_id = make_doc_id(file_path)
        new_ids = [c["id"] for c in enriched]

        # Chunk ids are positional ("{doc_id}::chunk-{n}"), so upsert overwrites
        # chunks 0..n-1 of a re-ingested document but can't remove old chunks
        # past the new count. Record what's stored BEFORE writing so stale
        # leftovers (from a longer earlier version) can be removed afterwards.
        previous_ids = {c["id"] for c in self.vector_store.get_by_doc_id(doc_id)}

        self._with_retries(
            lambda: self.vector_store.upsert(
                ids=new_ids,
                embeddings=embeddings,
                documents=texts,
                metadatas=[c["metadata"] for c in enriched],
            ),
            "Vector store upsert",
        )

        # Delete only AFTER the new version is safely written: if the upsert
        # fails, the old version stays intact and the file is retried next run
        # (the manifest isn't updated), rather than the document vanishing.
        stale_ids = sorted(previous_ids - set(new_ids))
        if stale_ids:
            self.vector_store.delete_by_ids(stale_ids)
            logger.info(f"  -> removed {len(stale_ids)} stale chunk(s) from the previous version.")
            audit.log("stale_chunks_removed", doc_id=doc_id,
                      source=os.path.basename(file_path), removed_count=len(stale_ids))


        logger.info(f"  -> {len(enriched)} chunks embedded and stored.")
        audit.log(
            "document_ingested",
            doc_id=make_doc_id(file_path),
            source=os.path.basename(file_path),
            file_path=file_path,
            markdown_path=md_path,
            chunk_count=len(enriched),
            classification_method=enriched[0]["metadata"]["classification_method"],
            roles=enriched[0]["metadata"]["allowed_roles"],
            tier=enriched[0]["metadata"]["sensitivity_tier"],
            chunk_overrides=sum(1 for c in enriched if c["metadata"]["chunk_override"]),
        )
        return len(enriched)

    def run_once(self) -> int:
        """
        Scans the source directory and ingests only files that are new or
        changed since the last run. Returns count of files ingested.
        This is the entry point that satisfies: 'runs only when new
        documents are added.'

        Failures are isolated per file: one bad or unreachable document
        doesn't block the rest of the batch, and is only marked ingested
        in the manifest on success — so it's automatically retried on the
        next run instead of being silently skipped forever.
        """
        manifest = load_manifest()
        candidates = self.discover_files()
        new_or_changed = [f for f in candidates if is_new_or_changed(f, manifest)]

        if not new_or_changed:
            logger.info("No new or changed documents found. Nothing to do.")
            return 0

        logger.info(f"Found {len(new_or_changed)} new/changed document(s) to ingest.")
        ingested_count = 0
        failed_files = []
        for file_path in new_or_changed:
            try:
                chunk_count = self.ingest_file(file_path)
                mark_ingested(file_path, manifest, chunk_count)
                ingested_count += 1
            except ExtractionError as e:
                logger.error(f"Extraction failed for {file_path}: {e}")
                audit.log("document_ingestion_failed", source=os.path.basename(file_path), file_path=file_path, reason=str(e))
                failed_files.append(file_path)
            except Exception as e:
                logger.exception(f"Unexpected error ingesting {file_path}: {e}")
                audit.log("document_ingestion_failed", source=os.path.basename(file_path), file_path=file_path, reason=str(e))
                failed_files.append(file_path)

        save_manifest(manifest)
        if failed_files:
            logger.warning(
                f"{len(failed_files)} file(s) failed and will be retried on the next run: {failed_files}"
            )
        return ingested_count


if __name__ == "__main__":
    pipeline = IngestionPipeline()
    n = pipeline.run_once()
    print(f"Ingested {n} document(s).")
