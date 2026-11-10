"""
Embedding model wrapper.

Default implementation uses a local sentence-transformers model (no API key,
no network dependency at query time). Swap in an API-backed embedder by
implementing the same `embed(texts: List[str]) -> List[List[float]]` interface,
e.g. OpenAIEmbedder, VoyageEmbedder, etc.
"""

from __future__ import annotations
import os
from typing import List
from config import EMBEDDING_MODEL_NAME, EMBEDDING_BATCH_SIZE

# Once a model is cached locally (after its first successful load), stop
# checking huggingface.co for "is this still the latest version" metadata
# on every run. This makes ingestion fully offline-capable and removes the
# per-run network calls entirely — nothing about your documents was ever
# sent over the network, but this also stops the "am I up to date?" pings.
os.environ.setdefault("HF_HUB_OFFLINE", "1")


class BaseEmbedder:
    def embed(self, texts: List[str]) -> List[List[float]]:
        raise NotImplementedError


class LocalSentenceTransformerEmbedder(BaseEmbedder):
    def __init__(self, model_name: str = EMBEDDING_MODEL_NAME, batch_size: int = EMBEDDING_BATCH_SIZE):
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as e:
            raise ImportError(
                "sentence-transformers is required. Install with "
                "`pip install sentence-transformers`."
            ) from e

        try:
            self._model = SentenceTransformer(model_name)
        except Exception as e:
            # Most likely cause: HF_HUB_OFFLINE=1 but the model was never
            # actually cached yet. Retry once with network access allowed
            # so the first run still works; subsequent runs stay offline.
            if os.environ.get("HF_HUB_OFFLINE") == "1":
                os.environ["HF_HUB_OFFLINE"] = "0"
                self._model = SentenceTransformer(model_name)
                os.environ["HF_HUB_OFFLINE"] = "1"
            else:
                raise

        self.batch_size = batch_size

    def embed(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            return []
        vectors = self._model.encode(
            texts,
            batch_size=self.batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        return vectors.tolist()


class APIEmbedder(BaseEmbedder):
    """
    Example skeleton for wiring in a hosted embeddings API. Fill in `_call_api`
    with the provider's client call. Kept separate so ingestion code never
    needs to change when you switch providers — only this class does.
    """
    def __init__(self, model_name: str, client, batch_size: int = EMBEDDING_BATCH_SIZE):
        self.model_name = model_name
        self.client = client
        self.batch_size = batch_size

    def embed(self, texts: List[str]) -> List[List[float]]:
        vectors = []
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i:i + self.batch_size]
            vectors.extend(self._call_api(batch))
        return vectors

    def _call_api(self, batch: List[str]) -> List[List[float]]:
        raise NotImplementedError("Wire this up to your embeddings provider's API.")


def get_default_embedder() -> BaseEmbedder:
    return LocalSentenceTransformerEmbedder()
