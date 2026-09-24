"""Qdrant-based vector storage for KG entities and relations.

Provides semantic search over entity/relation descriptions.
"""

import hashlib
from typing import Literal

from qdrant_client import AsyncQdrantClient
from qdrant_client.models import (
    Distance,
    PointStruct,
    VectorParams,
    Filter,
    FieldCondition,
    MatchValue,
    MatchAny,
)

from ..base import Entity, Relation

try:
    from config import get_config
    from utils.clients import embed_client
except ImportError:
    from src.config import get_config
    from src.utils.clients import embed_client


class KGVectorStore:
    """Qdrant vector store for KG entity and relation embeddings.

    Provides semantic search over entities and relations based on
    their descriptions.

    Example:
        >>> store = KGVectorStore(namespace="lightrag")
        >>> await store.init()
        >>> await store.upsert_entities([entity1, entity2])
        >>> similar = await store.search_entities("cancer treatment", top_k=10)
    """

    def __init__(
        self,
        namespace: str = "kg",
        qdrant_client: AsyncQdrantClient | None = None,
    ):
        """Initialize vector store.

        Args:
            namespace: Namespace prefix for collection names
            qdrant_client: Optional pre-configured Qdrant client
        """
        self._namespace = namespace
        self._config = get_config()

        # Collection names
        self._entity_collection = f"{namespace}_entities"
        self._relation_collection = f"{namespace}_relations"

        self._client = qdrant_client
        self._initialized = False

    async def _ensure_client(self) -> AsyncQdrantClient:
        """Ensure Qdrant client is initialized.

        Uses a FIXED URL (first in the round-robin list) rather than the
        rotating property, because KGVectorStore creates/deletes ephemeral
        collections per query — those write ops need a single stable endpoint.
        """
        if self._client is None:
            # Prefer the first static URL from the config's urls list.
            urls = getattr(self._config.qdrant, "urls", None)
            fixed_url = urls[0] if urls else self._config.qdrant.url
            self._client = AsyncQdrantClient(
                url=fixed_url,
                check_compatibility=False,
                timeout=60,
            )
        return self._client

    async def init(self) -> None:
        """Initialize collections if they don't exist."""
        if self._initialized:
            return

        client = await self._ensure_client()

        # Check/create entity collection
        collections = await client.get_collections()
        existing = {c.name for c in collections.collections}

        vector_params = VectorParams(
            size=self._config.embedding.dimension,
            distance=Distance.COSINE,
        )

        if self._entity_collection not in existing:
            await client.create_collection(
                collection_name=self._entity_collection,
                vectors_config=vector_params,
            )

        if self._relation_collection not in existing:
            await client.create_collection(
                collection_name=self._relation_collection,
                vectors_config=vector_params,
            )

        self._initialized = True

    async def upsert_entities(
        self,
        entities: list[Entity],
        batch_size: int = 100,
    ) -> int:
        """Upsert entities with embeddings into vector store.

        Args:
            entities: List of entities to upsert
            batch_size: Batch size for embedding calls

        Returns:
            Number of entities upserted
        """
        await self.init()
        client = await self._ensure_client()

        # Filter entities that need embeddings
        to_embed = []
        with_embeddings = []

        for entity in entities:
            if entity.embedding is not None:
                with_embeddings.append(entity)
            elif entity.description:
                to_embed.append(entity)

        # Generate embeddings for entities without them
        if to_embed:
            texts = [f"{e.name}: {e.description}" for e in to_embed]
            embeddings = await embed_client.embed(texts)

            for entity, embedding in zip(to_embed, embeddings):
                entity.embedding = embedding
                with_embeddings.append(entity)

        # Build points
        points = []
        for entity in with_embeddings:
            point_id = int(hashlib.sha256(entity.id.encode()).hexdigest()[:15], 16)
            points.append(
                PointStruct(
                    id=point_id,
                    vector=entity.embedding,
                    payload={
                        "entity_id": entity.id,
                        "name": entity.name,
                        "type": entity.type,
                        "description": entity.description[:1000],
                        "mentions": entity.mentions,
                    },
                )
            )

        # Upsert in batches
        for i in range(0, len(points), batch_size):
            batch = points[i : i + batch_size]
            await client.upsert(
                collection_name=self._entity_collection,
                points=batch,
            )

        return len(points)

    async def upsert_relations(
        self,
        relations: list[Relation],
        batch_size: int = 100,
    ) -> int:
        """Upsert relations with embeddings into vector store.

        Args:
            relations: List of relations to upsert
            batch_size: Batch size for embedding calls

        Returns:
            Number of relations upserted
        """
        await self.init()
        client = await self._ensure_client()

        # Filter relations that need embeddings
        to_embed = []
        with_embeddings = []

        for relation in relations:
            if relation.embedding is not None:
                with_embeddings.append(relation)
            elif relation.description:
                to_embed.append(relation)

        # Generate embeddings
        if to_embed:
            texts = [
                f"{r.source} {r.type} {r.target}: {r.description}" for r in to_embed
            ]
            embeddings = await embed_client.embed(texts)

            for relation, embedding in zip(to_embed, embeddings):
                relation.embedding = embedding
                with_embeddings.append(relation)

        # Build points
        points = []
        for relation in with_embeddings:
            point_id = int(hashlib.sha256(relation.id.encode()).hexdigest()[:15], 16)
            points.append(
                PointStruct(
                    id=point_id,
                    vector=relation.embedding,
                    payload={
                        "relation_id": relation.id,
                        "source": relation.source,
                        "target": relation.target,
                        "type": relation.type,
                        "description": relation.description[:1000],
                        "weight": relation.weight,
                        "keywords": relation.keywords,
                    },
                )
            )

        # Upsert in batches
        for i in range(0, len(points), batch_size):
            batch = points[i : i + batch_size]
            await client.upsert(
                collection_name=self._relation_collection,
                points=batch,
            )

        return len(points)

    async def search_entities(
        self,
        query: str,
        top_k: int = 20,
        entity_types: list[str] | None = None,
    ) -> list[tuple[Entity, float]]:
        """Search for similar entities by query.

        Args:
            query: Search query
            top_k: Number of results
            entity_types: Optional filter by entity types

        Returns:
            List of (Entity, score) tuples
        """
        await self.init()
        client = await self._ensure_client()

        # Get query embedding
        query_embedding = await embed_client.embed_single(query)

        # Build filter
        qdrant_filter = None
        if entity_types:
            qdrant_filter = Filter(
                must=[
                    FieldCondition(
                        key="type",
                        match=MatchAny(any=entity_types),
                    )
                ]
            )

        # Search
        results = await client.query_points(
            collection_name=self._entity_collection,
            query=query_embedding,
            query_filter=qdrant_filter,
            limit=top_k,
            with_payload=True,
        )

        # Convert to Entity objects
        entities = []
        for point in results.points:
            payload = point.payload
            entity = Entity(
                id=payload["entity_id"],
                name=payload["name"],
                type=payload["type"],
                description=payload.get("description", ""),
                mentions=payload.get("mentions", 1),
            )
            entities.append((entity, point.score))

        return entities

    async def search_relations(
        self,
        query: str,
        top_k: int = 20,
        relation_types: list[str] | None = None,
    ) -> list[tuple[Relation, float]]:
        """Search for similar relations by query.

        Args:
            query: Search query
            top_k: Number of results
            relation_types: Optional filter by relation types

        Returns:
            List of (Relation, score) tuples
        """
        await self.init()
        client = await self._ensure_client()

        # Get query embedding
        query_embedding = await embed_client.embed_single(query)

        # Build filter
        qdrant_filter = None
        if relation_types:
            qdrant_filter = Filter(
                must=[
                    FieldCondition(
                        key="type",
                        match=MatchAny(any=relation_types),
                    )
                ]
            )

        # Search
        results = await client.query_points(
            collection_name=self._relation_collection,
            query=query_embedding,
            query_filter=qdrant_filter,
            limit=top_k,
            with_payload=True,
        )

        # Convert to Relation objects
        relations = []
        for point in results.points:
            payload = point.payload
            relation = Relation(
                id=payload["relation_id"],
                source=payload["source"],
                target=payload["target"],
                type=payload["type"],
                description=payload.get("description", ""),
                weight=payload.get("weight", 1.0),
                keywords=payload.get("keywords", []),
            )
            relations.append((relation, point.score))

        return relations

    async def get_entity_by_name(self, name: str) -> Entity | None:
        """Get entity by exact name match.

        Args:
            name: Entity name

        Returns:
            Entity or None
        """
        await self.init()
        client = await self._ensure_client()

        results = await client.scroll(
            collection_name=self._entity_collection,
            scroll_filter=Filter(
                must=[FieldCondition(key="name", match=MatchValue(value=name))]
            ),
            limit=1,
            with_payload=True,
        )

        if results[0]:
            payload = results[0][0].payload
            return Entity(
                id=payload["entity_id"],
                name=payload["name"],
                type=payload["type"],
                description=payload.get("description", ""),
                mentions=payload.get("mentions", 1),
            )
        return None

    async def delete_collection(self, collection_type: Literal["entities", "relations", "all"] = "all") -> None:
        """Delete collections.

        Args:
            collection_type: Which collections to delete
        """
        client = await self._ensure_client()

        if collection_type in ("entities", "all"):
            try:
                await client.delete_collection(self._entity_collection)
            except Exception:
                pass

        if collection_type in ("relations", "all"):
            try:
                await client.delete_collection(self._relation_collection)
            except Exception:
                pass

        self._initialized = False

    async def close(self) -> None:
        """Close the client connection."""
        if self._client:
            await self._client.close()
            self._client = None
            self._initialized = False
