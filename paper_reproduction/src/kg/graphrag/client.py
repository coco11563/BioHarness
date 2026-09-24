"""MS GraphRAG client implementing KGClient protocol.

GraphRAG builds hierarchical communities and uses them for
both local and global search.
"""

import time
from pathlib import Path as FilePath
from typing import Literal

from ..base import Entity, Relation, Community, KGStats, QueryResult, ModelResponse
from ..extraction import EntityExtractor, RelationExtractor
from ..storage import GraphStore, KGVectorStore, ArtifactStore
from .community import CommunityDetector, CommunityReportGenerator
from .search import LocalSearch, GlobalSearch


class GraphRAGClient:
    """MS GraphRAG implementation of KGClient protocol.

    Builds hierarchical communities and supports two search modes:
    - local: Entity/relation focused search
    - global: Community report aggregation via map-reduce

    Example:
        >>> client = GraphRAGClient(storage_dir="data/kg/graphrag")
        >>> await client.index(chunks, chunk_ids)
        >>> # Local search for specific information
        >>> result = await client.query("What is BRCA1?", mode="local")
        >>> # Global search for overview/synthesis
        >>> result = await client.query("Overview of cancer genetics", mode="global")
    """

    def __init__(
        self,
        storage_dir: str | FilePath = "data/kg/graphrag",
        namespace: str = "graphrag",
        community_levels: int = 3,
    ):
        """Initialize GraphRAG client.

        Args:
            storage_dir: Directory for artifact persistence
            namespace: Namespace for vector collections
            community_levels: Number of community hierarchy levels
        """
        self._storage_dir = FilePath(storage_dir)
        self._namespace = namespace
        self._community_levels = community_levels

        # Storage components
        self._graph_store = GraphStore()
        self._vector_store = KGVectorStore(namespace=namespace)
        self._artifact_store = ArtifactStore(storage_dir)

        # Extraction components
        self._entity_extractor = EntityExtractor()
        self._relation_extractor = RelationExtractor()

        # Community components
        self._community_detector: CommunityDetector | None = None
        self._report_generator: CommunityReportGenerator | None = None

        # Search components
        self._local_search: LocalSearch | None = None
        self._global_search: GlobalSearch | None = None

        self._indexed = False

    async def index(
        self,
        chunks: list[str],
        chunk_ids: list[str] | None = None,
        **kwargs,
    ) -> None:
        """Build knowledge graph with communities from text chunks.

        In addition to LightRAG-style extraction, also:
        1. Detects hierarchical communities
        2. Generates LLM summaries for each community

        Args:
            chunks: List of text chunks to process
            chunk_ids: Optional IDs for chunks
            **kwargs: Additional options:
                - batch_size: Concurrent extractions
                - generate_reports: Whether to generate community reports
                - save: Whether to save artifacts
        """
        batch_size = kwargs.get("batch_size", 15)
        generate_reports = kwargs.get("generate_reports", True)
        save_artifacts = kwargs.get("save", True)

        if chunk_ids is None:
            chunk_ids = [f"chunk_{i:06d}" for i in range(len(chunks))]

        # Step 1: Extract entities
        entity_results = await self._entity_extractor.extract_batch(
            list(zip(chunk_ids, chunks)),
            concurrency=batch_size,
        )

        chunk_entities: dict[str, list[Entity]] = {}
        for result in entity_results:
            chunk_entities[result.chunk_id] = result.entities
            for entity in result.entities:
                self._graph_store.add_entity(entity)

        # Step 2: Extract relations
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

        # Step 3: Detect communities
        self._community_detector = CommunityDetector(self._graph_store)
        communities = await self._community_detector.detect(
            levels=self._community_levels
        )

        # Step 4: Generate community reports
        if generate_reports and communities:
            self._report_generator = CommunityReportGenerator(self._graph_store)
            communities = await self._report_generator.generate_reports(
                communities,
                concurrency=batch_size,
            )

        # Add communities to graph store
        for community in communities:
            self._graph_store.add_community(community)

        # Step 5: Build vector indices
        entities = list(self._graph_store.iter_entities())
        relations = list(self._graph_store.iter_relations())

        await self._vector_store.upsert_entities(entities)
        await self._vector_store.upsert_relations(relations)

        # Step 6: Save artifacts
        if save_artifacts:
            chunk_data = [
                {"id": cid, "text": text[:500]}
                for cid, text in zip(chunk_ids, chunks)
            ]
            await self._artifact_store.save(self._graph_store)
            await self._artifact_store.save_chunks(chunk_data)

        # Initialize search components
        self._local_search = LocalSearch(self._graph_store, self._vector_store)
        self._global_search = GlobalSearch(self._graph_store, self._vector_store)
        self._indexed = True

    async def query(
        self,
        question: str,
        mode: str = "local",
        top_k: int = 20,
        **kwargs,
    ) -> QueryResult:
        """Query using local or global search.

        Args:
            question: Natural language question
            mode: "local" for entity focus, "global" for synthesis
            top_k: Number of results
            **kwargs: Additional options:
                - community_level: Level for global search (default: 1)

        Returns:
            QueryResult with answer and context
        """
        if self._local_search is None or self._global_search is None:
            raise RuntimeError("Client not indexed. Call index() or load() first.")

        community_level = kwargs.get("community_level", 1)

        if mode == "global":
            return await self._global_search.search(
                question,
                community_level=community_level,
                max_communities=top_k,
            )
        else:  # local is default
            return await self._local_search.search(question, top_k=top_k)

    def get_stats(self) -> KGStats:
        """Get knowledge graph statistics."""
        return self._graph_store.get_stats()

    async def save(self, path: str | None = None) -> None:
        """Save knowledge graph to disk."""
        if path:
            store = ArtifactStore(path)
        else:
            store = self._artifact_store
        await store.save(self._graph_store)

    async def load(self, path: str | None = None) -> None:
        """Load knowledge graph from disk."""
        if path:
            store = ArtifactStore(path)
        else:
            store = self._artifact_store

        self._graph_store = await store.load()

        # Rebuild vector indices
        entities = list(self._graph_store.iter_entities())
        relations = list(self._graph_store.iter_relations())

        await self._vector_store.upsert_entities(entities)
        await self._vector_store.upsert_relations(relations)

        # Initialize search components
        self._local_search = LocalSearch(self._graph_store, self._vector_store)
        self._global_search = GlobalSearch(self._graph_store, self._vector_store)
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

        Uses local search for specific questions (factoid, yesno, mcq)
        and global search for synthesis questions (summary, list).
        """
        start_time = time.perf_counter()

        # Choose mode based on question type
        if question_type in ("summary", "list"):
            mode = "global"
        else:
            mode = "local"

        result = await self.query(question, mode=mode, top_k=20)

        # Extract answer
        answer = self._extract_answer(result.answer, question_type, options)

        latency_ms = (time.perf_counter() - start_time) * 1000

        return ModelResponse(
            answer=answer,
            response_text=result.answer,
            latency_ms=latency_ms,
            metadata={
                "mode": mode,
                "num_entities": len(result.entities),
                "num_communities": len(result.communities),
            },
        )

    def _extract_answer(
        self,
        response: str,
        question_type: str,
        options: dict[str, str] | None = None,
    ) -> str:
        """Extract structured answer from response."""
        response_lower = response.lower().strip()

        if question_type == "yesno":
            if any(w in response_lower[:50] for w in ["yes", "correct", "true"]):
                return "yes"
            elif any(w in response_lower[:50] for w in ["no", "incorrect", "false"]):
                return "no"
            return "yes" if "yes" in response_lower else "no"

        elif question_type == "mcq" and options:
            for key, value in options.items():
                if key.lower() in response_lower or value.lower() in response_lower:
                    return key
            return list(options.keys())[0] if options else ""

        elif question_type == "factoid":
            sentences = response.split(".")
            return sentences[0].strip()[:100] if sentences else response[:100]

        elif question_type == "list":
            lines = [l.strip() for l in response.split("\n") if l.strip()]
            items = []
            for line in lines:
                if line.startswith(("-", "*", "•")) or line[0].isdigit():
                    item = line.lstrip("-*•0123456789.) ")
                    if item:
                        items.append(item)
            return ", ".join(items[:10]) if items else response[:200]

        else:
            return response[:500]

    @property
    def graph_store(self) -> GraphStore:
        """Access underlying graph store."""
        return self._graph_store

    @property
    def communities(self) -> list[Community]:
        """Get all communities."""
        return list(self._graph_store.iter_communities())

    async def close(self) -> None:
        """Clean up resources."""
        await self._vector_store.close()
