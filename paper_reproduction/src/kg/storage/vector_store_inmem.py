"""In-memory vector store for ephemeral KG (RT-KG per-query mini-KGs).

Drop-in replacement for KGVectorStore that avoids all Qdrant HTTP roundtrips.
Use when the KG is created, queried a few times, then discarded — which is
exactly the RT-KG retrieve-then-KG pattern.

Benefits vs Qdrant-backed store:
- No HTTP roundtrips (10-15 ops per query saved)
- No collection create/delete overhead
- No Qdrant contention when multiple workers run in parallel
- ~10x speedup for small per-question graphs (< 1000 entities/relations)

API matches KGVectorStore so downstream LightRAG/PathRAG/GraphRAG query
executors work unchanged.
"""

from __future__ import annotations

from typing import Literal

import numpy as np

from ..base import Entity, Relation

try:
    from utils.clients import embed_client
except ImportError:
    from src.utils.clients import embed_client


class InMemoryKGVectorStore:
    """In-memory replacement for KGVectorStore.

    Stores entities/relations with their embeddings in numpy arrays and does
    brute-force cosine similarity search. For per-query mini-KGs with
    O(100) entities, this is much faster than any remote vector DB.

    API is intentionally identical to KGVectorStore:
        - init()
        - upsert_entities(entities)
        - upsert_relations(relations)
        - search_entities(query, top_k, entity_types)
        - search_relations(query, top_k, relation_types)
        - get_entity_by_name(name)
        - delete_collection(type)
        - close()
    """

    def __init__(self, namespace: str = "kg", qdrant_client=None):
        """Initialize in-memory store.

        Args:
            namespace: Kept for API compatibility (ignored — no collections).
            qdrant_client: Kept for API compatibility (ignored).
        """
        self._namespace = namespace
        # Lists for insertion order; numpy arrays built lazily before search.
        self._entities: list[Entity] = []
        self._entity_embeddings: list[list[float]] = []
        self._entity_matrix: np.ndarray | None = None  # (N, D) normalized
        self._entity_name_index: dict[str, int] = {}

        self._relations: list[Relation] = []
        self._relation_embeddings: list[list[float]] = []
        self._relation_matrix: np.ndarray | None = None

        self._initialized = False

    async def init(self) -> None:
        """No-op (no collections to create)."""
        self._initialized = True

    # ------------------------------------------------------------------
    # Upsert
    # ------------------------------------------------------------------

    async def upsert_entities(
        self,
        entities: list[Entity],
        batch_size: int = 100,
    ) -> int:
        """Store entities + embeddings in memory."""
        to_embed: list[Entity] = []
        for e in entities:
            if e.embedding is None and e.description:
                to_embed.append(e)

        # Batch-embed any entities missing vectors
        if to_embed:
            texts = [f"{e.name}: {e.description}" for e in to_embed]
            embs = await embed_client.embed(texts)
            for e, emb in zip(to_embed, embs):
                e.embedding = emb

        added = 0
        for e in entities:
            if e.embedding is None:
                continue
            self._entities.append(e)
            self._entity_embeddings.append(e.embedding)
            self._entity_name_index[e.name] = len(self._entities) - 1
            added += 1

        # Invalidate the prebuilt matrix — will rebuild on next search.
        self._entity_matrix = None
        return added

    async def upsert_relations(
        self,
        relations: list[Relation],
        batch_size: int = 100,
    ) -> int:
        """Store relations + embeddings in memory."""
        to_embed: list[Relation] = []
        for r in relations:
            if r.embedding is None and r.description:
                to_embed.append(r)

        if to_embed:
            texts = [
                f"{r.source} {r.type} {r.target}: {r.description}" for r in to_embed
            ]
            embs = await embed_client.embed(texts)
            for r, emb in zip(to_embed, embs):
                r.embedding = emb

        added = 0
        for r in relations:
            if r.embedding is None:
                continue
            self._relations.append(r)
            self._relation_embeddings.append(r.embedding)
            added += 1

        self._relation_matrix = None
        return added

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------

    def _build_entity_matrix(self) -> None:
        """Lazily build the normalized entity embedding matrix."""
        if not self._entity_embeddings:
            self._entity_matrix = np.zeros((0, 0), dtype=np.float32)
            return
        m = np.asarray(self._entity_embeddings, dtype=np.float32)
        norms = np.linalg.norm(m, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1.0, norms)
        self._entity_matrix = m / norms

    def _build_relation_matrix(self) -> None:
        if not self._relation_embeddings:
            self._relation_matrix = np.zeros((0, 0), dtype=np.float32)
            return
        m = np.asarray(self._relation_embeddings, dtype=np.float32)
        norms = np.linalg.norm(m, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1.0, norms)
        self._relation_matrix = m / norms

    @staticmethod
    def _normalize_query(vec: list[float]) -> np.ndarray:
        q = np.asarray(vec, dtype=np.float32)
        n = np.linalg.norm(q)
        return q / n if n > 0 else q

    async def search_entities(
        self,
        query: str,
        top_k: int = 20,
        entity_types: list[str] | None = None,
    ) -> list[tuple[Entity, float]]:
        """Cosine similarity search over stored entities."""
        if not self._entities:
            return []
        if self._entity_matrix is None:
            self._build_entity_matrix()
        if self._entity_matrix.size == 0:
            return []

        q_emb = await embed_client.embed_single(query)
        q = self._normalize_query(q_emb)
        if q.shape[0] != self._entity_matrix.shape[1]:
            return []

        scores = self._entity_matrix @ q  # (N,)

        # Optional type filter: mask out non-matching entities
        if entity_types:
            types = {t.lower() for t in entity_types}
            mask = np.array(
                [e.type and e.type.lower() in types for e in self._entities],
                dtype=bool,
            )
            scores = np.where(mask, scores, -np.inf)

        top = min(top_k, len(self._entities))
        if top <= 0:
            return []
        # argpartition for O(N) top-k then sort the top-k subset
        idx = np.argpartition(-scores, top - 1)[:top]
        idx = idx[np.argsort(-scores[idx])]
        return [
            (self._entities[i], float(scores[i]))
            for i in idx
            if np.isfinite(scores[i])
        ]

    async def search_relations(
        self,
        query: str,
        top_k: int = 20,
        relation_types: list[str] | None = None,
    ) -> list[tuple[Relation, float]]:
        """Cosine similarity search over stored relations."""
        if not self._relations:
            return []
        if self._relation_matrix is None:
            self._build_relation_matrix()
        if self._relation_matrix.size == 0:
            return []

        q_emb = await embed_client.embed_single(query)
        q = self._normalize_query(q_emb)
        if q.shape[0] != self._relation_matrix.shape[1]:
            return []

        scores = self._relation_matrix @ q

        if relation_types:
            types = {t.lower() for t in relation_types}
            mask = np.array(
                [r.type and r.type.lower() in types for r in self._relations],
                dtype=bool,
            )
            scores = np.where(mask, scores, -np.inf)

        top = min(top_k, len(self._relations))
        if top <= 0:
            return []
        idx = np.argpartition(-scores, top - 1)[:top]
        idx = idx[np.argsort(-scores[idx])]
        return [
            (self._relations[i], float(scores[i]))
            for i in idx
            if np.isfinite(scores[i])
        ]

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    async def get_entity_by_name(self, name: str) -> Entity | None:
        idx = self._entity_name_index.get(name)
        if idx is None:
            return None
        return self._entities[idx]

    async def delete_collection(
        self,
        collection_type: Literal["entities", "relations", "all"] = "all",
    ) -> None:
        if collection_type in ("entities", "all"):
            self._entities.clear()
            self._entity_embeddings.clear()
            self._entity_name_index.clear()
            self._entity_matrix = None
        if collection_type in ("relations", "all"):
            self._relations.clear()
            self._relation_embeddings.clear()
            self._relation_matrix = None
        self._initialized = False

    async def close(self) -> None:
        """No-op — no network connections."""
        await self.delete_collection("all")
