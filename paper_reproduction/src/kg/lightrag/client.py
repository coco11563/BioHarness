"""LightRAG client implementing KGClient protocol.

LightRAG is an entity/relation-centric approach to KG-RAG that supports
multiple query modes for different use cases.
"""

import time
from pathlib import Path
from typing import Literal

from ..base import Entity, Relation, KGStats, QueryResult, ModelResponse
from ..extraction import EntityExtractor, RelationExtractor
from ..storage import GraphStore, KGVectorStore, ArtifactStore
from .query import QueryExecutor, QueryMode


class LightRAGClient:
    """LightRAG implementation of KGClient protocol.

    Provides entity/relation-centric retrieval with 4 query modes:
    - naive: Pure vector search (baseline)
    - local: Entity-centric search with neighbor expansion
    - global: Relation-centric search with aggregation
    - hybrid: Combined local + global (recommended)

    Example:
        >>> client = LightRAGClient(storage_dir="data/kg/lightrag")
        >>> await client.index(chunks, chunk_ids)
        >>> result = await client.query("What treats diabetes?", mode="hybrid")
        >>> print(result.answer)
    """

    def __init__(
        self,
        storage_dir: str | Path = "data/kg/lightrag",
        namespace: str = "lightrag",
    ):
        """Initialize LightRAG client.

        Args:
            storage_dir: Directory for artifact persistence
            namespace: Namespace for vector collections
        """
        self._storage_dir = Path(storage_dir)
        self._namespace = namespace

        # Storage components
        self._graph_store = GraphStore()
        self._vector_store = KGVectorStore(namespace=namespace)
        self._artifact_store = ArtifactStore(storage_dir)

        # Extraction components
        self._entity_extractor = EntityExtractor()
        self._relation_extractor = RelationExtractor()

        # Query executor (initialized after index/load)
        self._query_executor: QueryExecutor | None = None

        self._indexed = False

    async def index(
        self,
        chunks: list[str],
        chunk_ids: list[str] | None = None,
        **kwargs,
    ) -> None:
        """Build knowledge graph from text chunks.

        Extracts entities and relations from each chunk, builds graph,
        and creates vector embeddings.

        Args:
            chunks: List of text chunks to process
            chunk_ids: Optional IDs for chunks (auto-generated if not provided)
            **kwargs: Additional options:
                - batch_size: Number of chunks to process in parallel
                - save: Whether to save artifacts after indexing
        """
        batch_size = kwargs.get("batch_size", 15)
        save_artifacts = kwargs.get("save", True)

        # Generate chunk IDs if not provided
        if chunk_ids is None:
            chunk_ids = [f"chunk_{i:06d}" for i in range(len(chunks))]

        # Step 1: Extract entities from all chunks
        entity_results = await self._entity_extractor.extract_batch(
            list(zip(chunk_ids, chunks)),
            concurrency=batch_size,
        )

        # Collect all entities
        chunk_entities: dict[str, list[Entity]] = {}
        for result in entity_results:
            chunk_entities[result.chunk_id] = result.entities
            for entity in result.entities:
                self._graph_store.add_entity(entity)

        # Step 2: Extract relations from each chunk
        relation_inputs = [
            (cid, chunks[i], chunk_entities.get(cid, []))
            for i, cid in enumerate(chunk_ids)
            if chunk_entities.get(cid)
        ]

        relation_results = await self._relation_extractor.extract_batch(
            relation_inputs,
            concurrency=batch_size,
        )

        for result in relation_results:
            for relation in result.relations:
                self._graph_store.add_relation(relation)

        # Step 3: Build vector indices
        entities = list(self._graph_store.iter_entities())
        relations = list(self._graph_store.iter_relations())

        await self._vector_store.upsert_entities(entities)
        await self._vector_store.upsert_relations(relations)

        # Step 4: Save artifacts
        if save_artifacts:
            chunk_data = [
                {"id": cid, "text": text[:500]}
                for cid, text in zip(chunk_ids, chunks)
            ]
            await self._artifact_store.save(self._graph_store)
            await self._artifact_store.save_chunks(chunk_data)

        # Initialize query executor
        self._query_executor = QueryExecutor(self._graph_store, self._vector_store)
        self._indexed = True

    async def query(
        self,
        question: str,
        mode: str = "hybrid",
        top_k: int = 20,
        **kwargs,
    ) -> QueryResult:
        """Query the knowledge graph.

        Args:
            question: Natural language question
            mode: Query mode (naive, local, global, hybrid)
            top_k: Number of results to retrieve
            **kwargs: Additional options

        Returns:
            QueryResult with answer and supporting context
        """
        if self._query_executor is None:
            raise RuntimeError("Client not indexed. Call index() or load() first.")

        return await self._query_executor.query(question, mode=mode, top_k=top_k)

    def get_stats(self) -> KGStats:
        """Get knowledge graph statistics.

        Returns:
            KGStats with entity/relation/community counts
        """
        return self._graph_store.get_stats()

    async def save(self, path: str | None = None) -> None:
        """Save knowledge graph to disk.

        Args:
            path: Optional override for storage directory
        """
        if path:
            store = ArtifactStore(path)
        else:
            store = self._artifact_store

        await store.save(self._graph_store)

    async def load(self, path: str | None = None) -> None:
        """Load knowledge graph from disk.

        Args:
            path: Optional override for storage directory
        """
        if path:
            store = ArtifactStore(path)
        else:
            store = self._artifact_store

        self._graph_store = await store.load()

        # Rebuild vector indices from loaded entities/relations
        entities = list(self._graph_store.iter_entities())
        relations = list(self._graph_store.iter_relations())

        await self._vector_store.upsert_entities(entities)
        await self._vector_store.upsert_relations(relations)

        # Initialize query executor
        self._query_executor = QueryExecutor(self._graph_store, self._vector_store)
        self._indexed = True

    async def generate(
        self,
        question: str,
        question_type: str,
        context: list[str] | None = None,
        options: dict[str, str] | None = None,
    ) -> ModelResponse:
        """Generate answer for benchmark question.

        Implements BenchmarkModelClient protocol.

        Args:
            question: Question text
            question_type: Type (yesno, mcq, factoid, list, summary)
            context: Optional context passages (ignored, uses KG)
            options: MCQ options dict

        Returns:
            ModelResponse with answer and metadata
        """
        start_time = time.perf_counter()

        # Use hybrid mode for best results
        result = await self.query(question, mode="hybrid", top_k=20)

        # Extract final answer based on question type
        answer = self._extract_answer(result.answer, question_type, options)

        latency_ms = (time.perf_counter() - start_time) * 1000

        return ModelResponse(
            answer=answer,
            response_text=result.answer,
            latency_ms=latency_ms,
            metadata={
                "mode": result.mode,
                "num_entities": len(result.entities),
                "num_relations": len(result.relations),
            },
        )

    def _extract_answer(
        self,
        response: str,
        question_type: str,
        options: dict[str, str] | None = None,
    ) -> str:
        """Extract structured answer from LLM response.

        Args:
            response: Raw LLM response
            question_type: Question type
            options: MCQ options if applicable

        Returns:
            Extracted answer
        """
        response_lower = response.lower().strip()

        if question_type == "yesno":
            # Check for yes/no indicators
            if any(w in response_lower[:50] for w in ["yes", "correct", "true", "affirmative"]):
                return "yes"
            elif any(w in response_lower[:50] for w in ["no", "incorrect", "false", "negative"]):
                return "no"
            return "yes" if "yes" in response_lower else "no"

        elif question_type == "mcq" and options:
            # Find which option is mentioned
            for key, value in options.items():
                if key.lower() in response_lower or value.lower() in response_lower:
                    return key
            # Default to first option
            return list(options.keys())[0] if options else ""

        elif question_type == "factoid":
            # Return first sentence or up to 100 chars
            sentences = response.split(".")
            return sentences[0].strip()[:100] if sentences else response[:100]

        elif question_type == "list":
            # Extract list items
            lines = [l.strip() for l in response.split("\n") if l.strip()]
            items = []
            for line in lines:
                if line.startswith(("-", "*", "•")) or line[0].isdigit():
                    item = line.lstrip("-*•0123456789.) ")
                    if item:
                        items.append(item)
            return ", ".join(items[:10]) if items else response[:200]

        else:  # summary or unknown
            return response[:500]

    @property
    def graph_store(self) -> GraphStore:
        """Access underlying graph store."""
        return self._graph_store

    @property
    def vector_store(self) -> KGVectorStore:
        """Access underlying vector store."""
        return self._vector_store

    async def close(self) -> None:
        """Clean up resources."""
        await self._vector_store.close()
