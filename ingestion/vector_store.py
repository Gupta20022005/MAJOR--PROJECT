"""
Vector store wrapper around Chroma (local, persistent, metadata-filterable).
Swap the backend (Pinecone, Weaviate, pgvector, Qdrant...) by implementing
the same interface: upsert() / query() / delete_by_doc_id().
"""

from __future__ import annotations
from typing import List, Dict, Optional
from config import VECTOR_STORE_DIR, VECTOR_STORE_COLLECTION


class VectorStore:
    def __init__(self, persist_dir: str = VECTOR_STORE_DIR, collection_name: str = VECTOR_STORE_COLLECTION):
        try:
            import chromadb
        except ImportError as e:
            raise ImportError("chromadb is required. Install with `pip install chromadb`.") from e

        self._client = chromadb.PersistentClient(path=persist_dir)
        # Explicit cosine similarity space (Chroma's default is squared L2).
        self._collection = self._client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    def upsert(self, ids: List[str], embeddings: List[List[float]], documents: List[str], metadatas: List[Dict]):
        # Chroma metadata values must be scalar and non-None; allowed_roles is
        # already a scalar here (see ingestion/metadata.py).
        safe_metadatas = [_flatten_metadata(m) for m in metadatas]
        self._collection.upsert(
            ids=ids,
            embeddings=embeddings,
            documents=documents,
            metadatas=safe_metadatas,
        )

    def query(
        self,
        query_embedding: List[float],
        top_k: int = 5,
        where: Optional[Dict] = None,
    ) -> List[Dict]:
        """
        Cosine similarity search over the collection (space set to "cosine" at
        collection creation). `where` is a Chroma-style metadata filter, e.g.:
            {"sensitivity_tier": {"$in": ["public", "internal"]}}
        """
        results = self._collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            where=where,
        )
        hits = []
        for i in range(len(results["ids"][0])):
            distance = results["distances"][0][i] if results.get("distances") else None
            # With hnsw:space="cosine", Chroma returns cosine *distance*
            # (0 = identical, 2 = opposite); similarity = 1 - distance.
            similarity = (1 - distance) if distance is not None else None
            hits.append({
                "id": results["ids"][0][i],
                "text": results["documents"][0][i],
                "metadata": _unflatten_metadata(results["metadatas"][0][i]),
                "distance": distance,
                "similarity": similarity,
            })
        return hits

    def delete_by_doc_id(self, doc_id: str):
        self._collection.delete(where={"doc_id": doc_id})

    def delete_by_ids(self, ids: List[str]):
        if ids:
            self._collection.delete(ids=ids)

    def get_by_doc_id(self, doc_id: str) -> List[Dict]:
        """
        Fetches every stored chunk for a document by doc_id — no embedding or
        similarity search involved, just a direct metadata lookup. Useful for
        checking exactly what ended up in the vector store after ingestion.
        """
        results = self._collection.get(where={"doc_id": doc_id})
        chunks = []
        for i in range(len(results["ids"])):
            chunks.append({
                "id": results["ids"][i],
                "text": results["documents"][i],
                "metadata": _unflatten_metadata(results["metadatas"][i]),
            })
        # Chunks aren't guaranteed back in order from Chroma; sort by the
        # numeric suffix in "{doc_id}::chunk-{n}" so output reads top-to-bottom.
        chunks.sort(key=lambda c: int(c["id"].split("chunk-")[-1]))
        return chunks

    def get_by_filter(self, where: Dict) -> List[Dict]:
        """Every chunk matching a metadata predicate, WITH its embedding, no
        similarity search. Hybrid retrieval's BM25 side needs the full
        authorized candidate set so lexical search runs over the same
        authorization boundary as dense search."""
        r = self._collection.get(where=where, include=["documents", "metadatas", "embeddings"])
        emb = r.get("embeddings")
        return [{
            "id": r["ids"][i],
            "text": r["documents"][i],
            "metadata": _unflatten_metadata(r["metadatas"][i]),
            "embedding": (emb[i] if emb is not None and len(emb) > i else None),
        } for i in range(len(r["ids"]))]


def _flatten_metadata(metadata: Dict) -> Dict:
    """allowed_roles is a genuine scalar so the access predicate can use a
    plain {"$in": user.roles}. Chroma rejects None, so those become ""."""
    return {k: ("" if v is None else v) for k, v in metadata.items()}


def _unflatten_metadata(metadata: Dict) -> Dict:
    return dict(metadata)
