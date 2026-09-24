"""PathRAG client implementing KGClient protocol.

PathRAG uses multi-hop path finding for retrieval, connecting
query-relevant entities through graph paths.
"""

import time
from pathlib import Path as FilePath
from typing import Literal

from ..base import Entity, Relation, Path, KGStats, QueryResult, ModelResponse
from ..extraction import EntityExtractor, RelationExtractor
from ..storage import GraphStore, KGVectorStore, ArtifactStore
from .pathfinder import PathFinder

try:
    from config import get_config
    from utils.clients import llm_client
except ImportError:
    from src.config import get_config
    from src.utils.clients import llm_client


PATH_RAG_PROMPT = """Answer the question based on the knowledge graph paths provided.

## Path-based Evidence
{paths}

## Related Entities
{entities}

## Question
{question}

## Instructions
1. Use the path evidence to reason about connections between entities
2. Follow the logical chain of relationships shown in the paths
3. If the paths don't directly answer the question, explain what they show
4. Be concise and cite the specific paths that support your answer

## Answer
"""


class PathRAGClient:
    """PathRAG implementation of KGClient protocol.

    Uses multi-hop path finding to answer questions by:
    1. Identifying query-relevant entities
    2. Finding paths connecting these entities
    3. Using path evidence for LLM reasoning

    Example:
        >>> client = PathRAGClient(storage_dir="data/kg/pathrag")
        >>> await client.index(chunks, chunk_ids)
        >>> result = await client.query("How is BRCA1 related to breast cancer?")
        >>> print(result.paths)  # Shows connecting paths
    """

    def __init__(
        self,
        storage_dir: str | FilePath = "data/kg/pathrag",
        namespace: str = "pathrag",
        alpha: float = 0.8,
        max_hops: int = 3,
    ):
        """Initialize PathRAG client.

        Args:
            storage_dir: Directory for artifact persistence
            namespace: Namespace for vector collections
            alpha: Path decay factor (0-1)
            max_hops: Maximum path length to search
        """
        self._storage_dir = FilePath(storage_dir)
        self._namespace = namespace
        self._alpha = alpha
        self._max_hops = max_hops

        # Storage components
        self._graph_store = GraphStore()
        self._vector_store = KGVectorStore(namespace=namespace)
        self._artifact_store = ArtifactStore(storage_dir)

        # Extraction components
        self._entity_extractor = EntityExtractor()
        self._relation_extractor = RelationExtractor()

        # Path finder (initialized after index/load)
        self._path_finder: PathFinder | None = None

        self._indexed = False
        self._config = get_config()

    async def index(
        self,
        chunks: list[str],
        chunk_ids: list[str] | None = None,
        **kwargs,
    ) -> None:
        """Build knowledge graph from text chunks.

        Same as LightRAG - extracts entities and relations.

        Args:
            chunks: List of text chunks to process
            chunk_ids: Optional IDs for chunks
            **kwargs: Additional options
        """
        batch_size = kwargs.get("batch_size", 15)
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

        # Initialize path finder with reference parameters
        self._path_finder = PathFinder(
            self._graph_store,
            alpha=self._alpha,
            threshold=0.3,  # Reference: PathRAG default
            max_total_edges=15,  # Reference: PathRAG default
        )
        self._indexed = True

    async def query(
        self,
        question: str,
        mode: str = "hybrid",
        top_k: int = 40,
        **kwargs,
    ) -> QueryResult:
        """Query using path-based retrieval.

        Args:
            question: Natural language question
            mode: Query mode (ignored, always uses path mode)
            top_k: Number of paths to retrieve
            **kwargs: Additional options:
                - max_hops: Override default max hops
                - path_depth: Alias for max_hops

        Returns:
            QueryResult with answer and path evidence
        """
        if self._path_finder is None:
            raise RuntimeError("Client not indexed. Call index() or load() first.")

        start_time = time.perf_counter()
        max_hops = kwargs.get("max_hops", kwargs.get("path_depth", self._max_hops))

        # Step 1: Find query-relevant entities via vector search
        entity_results = await self._vector_store.search_entities(
            question, top_k=top_k // 2
        )
        seed_entities = [e.name for e, _ in entity_results]

        if not seed_entities:
            return QueryResult(
                answer="Could not find relevant entities for this question.",
                latency_ms=(time.perf_counter() - start_time) * 1000,
                mode="path",
            )

        # Step 2: Find paths between seed entities (using new weighted BFS)
        path_scores = await self._path_finder.find_paths(
            source_entities=seed_entities,
            max_hops=max_hops,
        )

        paths = [ps.path for ps in path_scores]

        # Step 3: Get all entities involved in paths
        entity_names = set(seed_entities)
        for path in paths:
            entity_names.update(path.nodes)

        entities = []
        for name in entity_names:
            entity = self._graph_store.get_entity(name)
            if entity:
                entities.append(entity)

        # Step 4: Generate answer with new evidence generation
        path_evidence = await self._path_finder.generate_path_evidence(paths)
        entity_context = self._format_entities(entities[:20])

        prompt = PATH_RAG_PROMPT.format(
            paths=path_evidence,
            entities=entity_context,
            question=question,
        )

        answer = await llm_client.chat(prompt)

        latency_ms = (time.perf_counter() - start_time) * 1000

        # Collect relations from paths
        relations = []
        for path in paths:
            for rel_id in path.edge_ids:
                rel = self._graph_store.get_relation(rel_id)
                if rel:
                    relations.append(rel)

        return QueryResult(
            answer=answer,
            entities=entities,
            relations=relations,
            paths=paths,
            context_text=f"{path_evidence}\n\n{entity_context}",
            latency_ms=latency_ms,
            mode="path",
            metadata={
                "num_paths": len(paths),
                "max_hops": max_hops,
                "seed_entities": seed_entities[:10],
            },
        )

    def _format_entities(self, entities: list[Entity]) -> str:
        """Format entities for context."""
        lines = ["## Entities\n"]
        for e in entities:
            lines.append(f"- **{e.name}** ({e.type}): {e.description[:150]}")
        return "\n".join(lines)

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

        # Initialize path finder with reference parameters
        self._path_finder = PathFinder(
            self._graph_store,
            alpha=self._alpha,
            threshold=0.3,
            max_total_edges=15,
        )
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
        """
        start_time = time.perf_counter()

        result = await self.query(question, top_k=40)

        # Extract answer based on question type
        answer = self._extract_answer(result.answer, question_type, options)

        latency_ms = (time.perf_counter() - start_time) * 1000

        return ModelResponse(
            answer=answer,
            response_text=result.answer,
            latency_ms=latency_ms,
            metadata={
                "mode": "path",
                "num_paths": len(result.paths),
                "num_entities": len(result.entities),
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
    def path_finder(self) -> PathFinder | None:
        """Access path finder."""
        return self._path_finder

    async def close(self) -> None:
        """Clean up resources."""
        await self._vector_store.close()
