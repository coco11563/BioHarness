"""Retrieve-then-KG ModelClient implementations.

Two-stage retrieval pipeline:
1. Stage 1: Use dense vector retriever to get top-k documents
2. Stage 2: Dynamically build mini-KG from retrieved docs, then apply KG strategy

This approach enables KG-RAG methods to work with large corpora (38M+ docs)
by first narrowing down to relevant documents.

Usage:
    from benchmark.retrieve_then_kg import RetrieveThenLightRAG

    async with RetrieveThenLightRAG(mode="hybrid") as client:
        response = await client.generate(
            question="Is metformin effective for diabetes?",
            question_type="yesno",
        )
"""

from __future__ import annotations

import sys
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, TYPE_CHECKING

import asyncpg
# Add project paths while preserving benchmark/code/src ahead of src/.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent.parent
_SRC_PATH = _PROJECT_ROOT / "src"
_BENCHMARK_SRC = Path(__file__).resolve().parents[1]

if str(_BENCHMARK_SRC) in sys.path:
    sys.path.remove(str(_BENCHMARK_SRC))
sys.path.insert(0, str(_BENCHMARK_SRC))

if str(_SRC_PATH) in sys.path:
    sys.path.remove(str(_SRC_PATH))
sys.path.insert(1, str(_SRC_PATH))

if str(_PROJECT_ROOT) not in sys.path:
    sys.path.append(str(_PROJECT_ROOT))

from qdrant_client import AsyncQdrantClient

# Import from local benchmark package (this directory)
from .client_protocol import ModelClient, ModelResponse
from .shared_pipeline import (
    DenseEvidenceDoc,
    generate_constrained_answer,
    retrieve_dense_evidence,
)

# Import retriever components using absolute path
from retriever import VectorSearch
from retriever.models import RetrievedAbstract

# Import config
from config import get_config

if TYPE_CHECKING:
    from kg.base import QueryResult
    from kg.storage import GraphStore
    from kg.extraction import EntityExtractor, RelationExtractor


def _doc_key(abstract) -> str:
    """Extraction-cache key for one retrieved document.

    Abstract mode keeps the PMID, so existing caches and results are unchanged. With
    XC_FULLTEXT_CHUNKS=1 many full-text chunks of one paper share a PMID: keyed by PMID,
    the dicts in _build_mini_kg kept only the last chunk of each paper and the cache
    returned that paper's abstract extraction for every chunk. Key each chunk by PMID plus
    a hash of its text instead.
    """
    import os as _os
    if _os.environ.get("XC_FULLTEXT_CHUNKS") != "1":
        return abstract.pmid
    import hashlib as _hl
    h = _hl.sha1(f"{abstract.title}\n\n{abstract.abstract}".encode("utf-8")).hexdigest()[:12]
    return f"{abstract.pmid}_ft{h}"


def _extractor_id() -> str:
    """Separate cache namespace for full-text chunks, so chunk and abstract extractions never mix."""
    import os as _os
    return "v1_qwen_default_ftchunk" if _os.environ.get("XC_FULLTEXT_CHUNKS") == "1" else "v1_qwen_default"


@dataclass
class RetrieveThenKGConfig:
    """Configuration for Retrieve-then-KG pipeline."""
    # Retriever settings
    retrieval_top_k: int = 20
    qdrant_url: str = None  # Will use config.qdrant.url if None
    paper_collection: str = "paper-full"

    def __post_init__(self):
        if self.qdrant_url is None:
            self.qdrant_url = get_config().qdrant.url

    # KG extraction settings
    extraction_batch_size: int = 10

    # KG namespace (for vector store)
    kg_namespace: str = "retrieve_then_kg"


class _RetrievedChunkSearcher:
    """Chunk searcher backed by stage-1 retrieved abstracts.

    This keeps RT-LightRAG naive mode aligned with a document-chunk retrieval
    baseline instead of falling back to entity-vector proxy retrieval.
    """

    def __init__(self, abstracts: list[RetrievedAbstract], max_chars: int = 1400):
        self._chunks: list[tuple[str, float]] = []
        for abstract in abstracts:
            text = f"PMID {abstract.pmid}\n{abstract.title}\n\n{abstract.abstract}"
            self._chunks.append((text[:max_chars], float(abstract.score)))

    async def search(self, query: str, limit: int = 20) -> list[tuple[str, float]]:
        del query  # ranking comes from stage-1 retrieval scores
        if limit <= 0:
            return []
        return self._chunks[:limit]


def _golden_context_to_abstracts(context: list[str]) -> list[RetrievedAbstract]:
    abstracts: list[RetrievedAbstract] = []
    for i, ctx_text in enumerate(context):
        abstracts.append(
            RetrievedAbstract(
                pmid=f"golden_context_{i}",
                title=f"Golden Context {i + 1}",
                abstract=ctx_text,
                score=2.0,
                rank=i,
            )
        )
    return abstracts


def _dense_docs_to_abstracts(docs: list[DenseEvidenceDoc]) -> list[RetrievedAbstract]:
    abstracts: list[RetrievedAbstract] = []
    for doc in docs:
        abstracts.append(
            RetrievedAbstract(
                pmid=doc.pmid,
                title=doc.title,
                abstract=doc.abstract,
                score=doc.score,
                rank=doc.rank,
            )
        )
    return abstracts


async def _retrieve_stage1_abstracts(
    *,
    question: str,
    context: list[str] | None,
    qdrant: AsyncQdrantClient,
    pool: asyncpg.Pool,
    retrieval_top_k: int,
    collection: str,
) -> tuple[list[RetrievedAbstract], dict]:
    """Shared stage-1 retrieval for all retrieve-then-KG variants."""
    if context:
        return _golden_context_to_abstracts(context), {
            "embed_ms": 0.0,
            "search_ms": 0.0,
            "abstract_ms": 0.0,
            "retrieval_ms": 0.0,
            "docs_retrieved": len(context),
            "docs_loaded": len(context),
            "top_k": retrieval_top_k,
            "source": "golden_context",
        }

    docs, retrieval_meta = await retrieve_dense_evidence(
        question,
        qdrant=qdrant,
        pool=pool,
        top_k=retrieval_top_k,
        collection=collection,
    )
    retrieval_meta = dict(retrieval_meta)
    retrieval_meta["source"] = "shared_dense_stage1"
    return _dense_docs_to_abstracts(docs), retrieval_meta


class RetrieveThenLightRAG(ModelClient):
    """Retrieve-then-LightRAG: Dense retrieval followed by LightRAG reasoning.

    Pipeline:
    1. VectorSearch → top-k abstracts
    2. Entity/Relation extraction from abstracts
    3. Build mini-KG in memory
    4. LightRAG query (naive/local/global/hybrid)

    Example:
        async with RetrieveThenLightRAG(mode="hybrid") as client:
            response = await client.generate(
                question="What treats diabetes?",
                question_type="factoid",
            )
    """

    def __init__(
        self,
        mode: Literal["naive", "local", "global", "hybrid"] = "hybrid",
        retrieval_top_k: int = 20,
        kg_top_k: int = 20,
        qdrant_url: str = None,  # Uses config.qdrant.url if None
        pg_dsn: str | None = None,
        paper_collection: str = "paper-full",
    ):
        """Initialize Retrieve-then-LightRAG client.

        Args:
            mode: LightRAG query mode
            retrieval_top_k: Number of documents to retrieve
            kg_top_k: Number of KG results to use
            qdrant_url: Qdrant server URL
            paper_collection: Collection name for paper vectors
        """
        self._mode = mode
        self._retrieval_top_k = retrieval_top_k
        self._kg_top_k = kg_top_k
        cfg = get_config()
        self._qdrant_url = qdrant_url if qdrant_url else cfg.qdrant.url
        self._pg_dsn = pg_dsn or cfg.postgres.pubmed_url
        self._paper_collection = paper_collection

        self._qdrant: AsyncQdrantClient | None = None
        self._pool: asyncpg.Pool | None = None
        self._vector_search: VectorSearch | None = None
        self._entity_extractor: EntityExtractor | None = None
        self._relation_extractor: RelationExtractor | None = None
        self._initialized = False

    async def __aenter__(self):
        """Initialize async resources."""
        from kg.extraction import EntityExtractor, RelationExtractor

        self._qdrant = AsyncQdrantClient(
            self._qdrant_url,
            timeout=60,
        )
        self._pool = await asyncpg.create_pool(
            self._pg_dsn,
            min_size=2,
            max_size=10,
        )
        self._vector_search = VectorSearch(
            self._qdrant,
            collection=self._paper_collection,
        )
        self._entity_extractor = EntityExtractor()
        self._relation_extractor = RelationExtractor()
        self._initialized = True
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Close async resources."""
        if self._pool:
            await self._pool.close()
        if self._qdrant:
            await self._qdrant.close()
        self._initialized = False

    async def generate(
        self,
        question: str,
        question_type: str,
        context: list[str] | None = None,
        options: dict[str, str] | None = None,
    ) -> ModelResponse:
        """Generate answer using retrieve-then-KG pipeline.

        Args:
            question: Question text
            question_type: Type (yesno, mcq, factoid, list, summary)
            context: Optional golden context to include in KG building
            options: MCQ options dict

        Returns:
            ModelResponse with answer and metadata
        """
        if not self._initialized:
            raise RuntimeError("Client not initialized. Use async context manager.")

        start_time = time.perf_counter()

        abstracts, retrieval_meta = await _retrieve_stage1_abstracts(
            question=question,
            context=context,
            qdrant=self._qdrant,
            pool=self._pool,
            retrieval_top_k=self._retrieval_top_k,
            collection=self._paper_collection,
        )
        retrieval_ms = retrieval_meta.get("retrieval_ms", 0.0)

        if not abstracts:
            return ModelResponse(
                answer="No relevant documents found.",
                response_text="No relevant documents found.",
                latency_ms=(time.perf_counter() - start_time) * 1000,
                metadata={"error": "no_documents"},
            )

        # Stage 2: Build mini-KG from retrieved abstracts
        graph_store, build_stats = await self._build_mini_kg(abstracts)

        # Stage 3: Build vector indices for KG
        from kg.storage.vector_store_inmem import InMemoryKGVectorStore as KGVectorStore
        from kg.lightrag.query import QueryExecutor as LightRAGQueryExecutor

        vector_store = KGVectorStore(namespace=f"rtkg_{uuid.uuid4().hex[:12]}")
        entities = list(graph_store.iter_entities())
        relations = list(graph_store.iter_relations())

        if entities:
            await vector_store.upsert_entities(entities)
        if relations:
            await vector_store.upsert_relations(relations)

        # Stage 4: Query using LightRAG with constrained output
        kg_query_start = time.perf_counter()
        chunk_searcher = _RetrievedChunkSearcher(abstracts)
        query_executor = LightRAGQueryExecutor(
            graph_store,
            vector_store,
            chunk_searcher=chunk_searcher,
        )
        result = await query_executor.query(
            question,
            mode=self._mode,
            top_k=self._kg_top_k,
            question_type=None,
            options=None,
            context_only=True,
        )
        kg_query_ms = (time.perf_counter() - kg_query_start) * 1000

        # Stage 5: Shared final answer generation from KG-built context
        llm_start = time.perf_counter()
        answer, answer_text = await generate_constrained_answer(
            question=question,
            question_type=question_type,
            context=result.context_text or "No knowledge graph context available.",
            options=options,
            temperature=0.1,
        )
        llm_ms = (time.perf_counter() - llm_start) * 1000

        # Cleanup vector store - delete collections to avoid accumulation
        await vector_store.delete_collection("all")
        await vector_store.close()

        total_ms = (time.perf_counter() - start_time) * 1000

        return ModelResponse(
            answer=answer,
            response_text=answer_text,
            latency_ms=total_ms,
            metadata={
                "strategy": "retrieve_then_lightrag",
                "mode": self._mode,
                "retrieval_top_k": self._retrieval_top_k,
                "retrieval_ms": retrieval_ms,
                "embed_ms": retrieval_meta.get("embed_ms"),
                "search_ms": retrieval_meta.get("search_ms"),
                "abstract_ms": retrieval_meta.get("abstract_ms"),
                "retrieval_source": retrieval_meta.get("source"),
                "kg_query_ms": kg_query_ms,
                "llm_ms": llm_ms,
                "kg_context_chars": len(result.context_text or ""),
                "num_abstracts": len(abstracts),
                "num_entities": len(entities),
                "num_relations": len(relations),
                **build_stats,
            },
        )

    async def _build_mini_kg(
        self,
        abstracts: list[RetrievedAbstract],
    ) -> tuple[GraphStore, dict]:
        """Build mini-KG from retrieved abstracts with extraction caching.

        Uses ExtractionCache to avoid redundant LLM calls:
        1. Check cache for each PMID
        2. Load cached entities/relations for hits
        3. Extract and cache for misses
        4. Build graph from all results

        Args:
            abstracts: Retrieved abstracts

        Returns:
            Tuple of (GraphStore, build_stats_dict)
        """
        from src.cache.extraction_cache import ExtractionCache
        from kg.storage import GraphStore

        start_time = time.perf_counter()
        graph_store = GraphStore()
        extraction_cache = ExtractionCache(extractor_id=_extractor_id())

        # Prepare data from abstracts
        pmid_to_abstract = {}
        pmid_to_text = {}
        for abstract in abstracts:
            text = f"{abstract.title}\n\n{abstract.abstract}"
            pmid_to_abstract[_doc_key(abstract)] = abstract
            pmid_to_text[_doc_key(abstract)] = text

        pmids = list(pmid_to_abstract.keys())

        # Partition into cached vs uncached
        cached_pmids, uncached_pmids = await extraction_cache.partition_pmids(pmids)
        cache_hit_rate = len(cached_pmids) / max(1, len(pmids))

        # Load cached extractions
        chunk_entities = {}
        for pmid in cached_pmids:
            result = await extraction_cache.load(pmid)
            if result:
                entities, relations = result
                chunk_entities[pmid] = entities
                for entity in entities:
                    graph_store.add_entity(entity)
                for relation in relations:
                    graph_store.add_relation(relation)

        # Extract for uncached PMIDs
        extraction_start = time.perf_counter()
        extraction_ms = 0.0

        if uncached_pmids:
            # Prepare chunks for extraction
            chunks_to_extract = [
                (pmid, pmid_to_text[pmid])
                for pmid in uncached_pmids
            ]

            # Extract entities
            entity_results = await self._entity_extractor.extract_batch(
                chunks_to_extract,
                concurrency=10,
            )

            for result in entity_results:
                chunk_entities[result.chunk_id] = result.entities
                for entity in result.entities:
                    graph_store.add_entity(entity)

            # Extract relations
            relation_inputs = [
                (pmid, pmid_to_text[pmid], chunk_entities.get(pmid, []))
                for pmid in uncached_pmids
                if chunk_entities.get(pmid)
            ]

            pmid_relations = {pmid: [] for pmid in uncached_pmids}
            if relation_inputs:
                relation_results = await self._relation_extractor.extract_batch(
                    relation_inputs,
                    concurrency=10,
                )

                for result in relation_results:
                    pmid_relations[result.chunk_id] = result.relations
                    for relation in result.relations:
                        graph_store.add_relation(relation)

            # Cache newly extracted results
            for pmid in uncached_pmids:
                abstract = pmid_to_abstract[pmid]
                text = pmid_to_text[pmid]
                entities = chunk_entities.get(pmid, [])
                relations = pmid_relations.get(pmid, [])
                await extraction_cache.set(
                    pmid=pmid,
                    title=abstract.title,
                    text=text,
                    entities=entities,
                    relations=relations,
                )

            extraction_ms = (time.perf_counter() - extraction_start) * 1000

        build_ms = (time.perf_counter() - start_time) * 1000
        build_stats = {
            "kg_build_ms": build_ms,
            "extraction_ms": extraction_ms,
            "cache_hits": len(cached_pmids),
            "cache_total": len(pmids),
            "cache_hit_rate": cache_hit_rate,
        }
        return graph_store, build_stats

    def _extract_answer(
        self,
        response: str,
        question_type: str,
        options: dict[str, str] | None = None,
    ) -> str:
        """Extract structured answer from response."""
        import re
        response_lower = response.lower().strip()

        if question_type == "yesno":
            if any(w in response_lower[:50] for w in ["yes", "correct", "true"]):
                return "yes"
            elif any(w in response_lower[:50] for w in ["no", "incorrect", "false"]):
                return "no"
            return "yes" if "yes" in response_lower else "no"

        elif question_type == "mcq" and options:
            # Use smarter extraction for MCQ to avoid false positives
            keys = list(options.keys())
            keys_pattern = "|".join(re.escape(k) for k in keys)

            # Priority 1: Explicit answer patterns (e.g., "Answer: A", "The answer is B")
            explicit_patterns = [
                r"(?:answer|correct answer|correct option)(?:\s+is)?[:\s]+[(]?(" + keys_pattern + r")[).]?",
                r"\*\*(" + keys_pattern + r")\*\*",  # Bold answer
                r"^(" + keys_pattern + r")[.)\s]",   # Answer at start of line
                r"\n(" + keys_pattern + r")[.)\s]",  # Answer at start of new line
            ]
            for pattern in explicit_patterns:
                match = re.search(pattern, response, re.IGNORECASE | re.MULTILINE)
                if match:
                    matched_key = match.group(1).upper()
                    if matched_key in keys:
                        return matched_key

            # Priority 2: Check for option letter with answer-indicating punctuation
            for key in keys:
                # Only match single letters when followed by answer punctuation: "A.", "A)", "A:"
                # or in parentheses "(A)" - not standalone article "a"
                answer_indicator_patterns = [
                    r"(?<![a-zA-Z])(" + re.escape(key) + r")[.):,]",  # A. A) A: A,
                    r"\((" + re.escape(key) + r")\)",  # (A)
                ]
                for pattern in answer_indicator_patterns:
                    if re.search(pattern, response[:150], re.IGNORECASE):
                        return key

            # Priority 3: Check if full option text appears in response
            for key, value in options.items():
                if len(value) > 5 and value.lower() in response_lower:
                    return key

            # Fallback: return first option
            return keys[0] if keys else ""

        elif question_type == "factoid":
            sentences = response.split(".")
            return sentences[0].strip()[:100] if sentences else response[:100]

        elif question_type == "list":
            lines = [l.strip() for l in response.split("\n") if l.strip()]
            items = []
            for line in lines:
                if line.startswith(("-", "*", "•")) or (line and line[0].isdigit()):
                    item = line.lstrip("-*•0123456789.) ")
                    if item:
                        items.append(item)
            return ", ".join(items[:10]) if items else response[:200]

        else:
            return response[:500]


class RetrieveThenPathRAG(ModelClient):
    """Retrieve-then-PathRAG: Dense retrieval followed by PathRAG reasoning.

    Pipeline:
    1. VectorSearch → top-k abstracts
    2. Entity/Relation extraction from abstracts
    3. Build mini-KG in memory
    4. PathRAG multi-hop path finding
    """

    def __init__(
        self,
        max_hops: int = 3,
        alpha: float = 0.8,
        retrieval_top_k: int = 20,
        qdrant_url: str = None,  # Uses config.qdrant.url if None
        pg_dsn: str | None = None,
        paper_collection: str = "paper-full",
    ):
        """Initialize Retrieve-then-PathRAG client.

        Args:
            max_hops: Maximum path length
            alpha: Decay factor per hop
            retrieval_top_k: Number of documents to retrieve
            qdrant_url: Qdrant server URL
            paper_collection: Collection name for paper vectors
        """
        self._max_hops = max_hops
        self._alpha = alpha
        self._retrieval_top_k = retrieval_top_k
        cfg = get_config()
        self._qdrant_url = qdrant_url if qdrant_url else cfg.qdrant.url
        self._pg_dsn = pg_dsn or cfg.postgres.pubmed_url
        self._paper_collection = paper_collection

        self._qdrant: AsyncQdrantClient | None = None
        self._pool: asyncpg.Pool | None = None
        self._vector_search: VectorSearch | None = None
        self._entity_extractor: EntityExtractor | None = None
        self._relation_extractor: RelationExtractor | None = None
        self._initialized = False

    async def __aenter__(self):
        """Initialize async resources."""
        from kg.extraction import EntityExtractor, RelationExtractor

        self._qdrant = AsyncQdrantClient(self._qdrant_url, timeout=60)
        self._pool = await asyncpg.create_pool(
            self._pg_dsn,
            min_size=2,
            max_size=10,
        )
        self._vector_search = VectorSearch(self._qdrant, collection=self._paper_collection)
        self._entity_extractor = EntityExtractor()
        self._relation_extractor = RelationExtractor()
        self._initialized = True
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Close async resources."""
        if self._pool:
            await self._pool.close()
        if self._qdrant:
            await self._qdrant.close()
        self._initialized = False

    async def generate(
        self,
        question: str,
        question_type: str,
        context: list[str] | None = None,
        options: dict[str, str] | None = None,
    ) -> ModelResponse:
        """Generate answer using retrieve-then-PathRAG pipeline."""
        if not self._initialized:
            raise RuntimeError("Client not initialized. Use async context manager.")

        start_time = time.perf_counter()

        abstracts, retrieval_meta = await _retrieve_stage1_abstracts(
            question=question,
            context=context,
            qdrant=self._qdrant,
            pool=self._pool,
            retrieval_top_k=self._retrieval_top_k,
            collection=self._paper_collection,
        )
        retrieval_ms = retrieval_meta.get("retrieval_ms", 0.0)

        if not abstracts:
            return ModelResponse(
                answer="No relevant documents found.",
                response_text="No relevant documents found.",
                latency_ms=(time.perf_counter() - start_time) * 1000,
                metadata={"error": "no_documents"},
            )

        # Stage 2: Build mini-KG
        graph_store, build_stats = await self._build_mini_kg(abstracts)

        # Stage 3: Build vector indices
        from kg.storage.vector_store_inmem import InMemoryKGVectorStore as KGVectorStore
        from kg.pathrag.pathfinder import PathFinder

        vector_store = KGVectorStore(namespace=f"rtkg_path_{uuid.uuid4().hex[:12]}")
        entities = list(graph_store.iter_entities())
        relations = list(graph_store.iter_relations())

        if entities:
            await vector_store.upsert_entities(entities)

        # Stage 4: PathRAG query
        kg_query_start = time.perf_counter()

        # Find seed entities via vector search
        entity_results = await vector_store.search_entities(question, top_k=10)
        seed_entities = [e.name for e, _ in entity_results]

        if len(seed_entities) < 2:
            # Not enough entities for path finding
            await vector_store.delete_collection("all")
            await vector_store.close()
            return ModelResponse(
                answer="Insufficient entities for path-based reasoning.",
                response_text="Could not find enough related entities.",
                latency_ms=(time.perf_counter() - start_time) * 1000,
                metadata={"error": "insufficient_entities", "found": len(seed_entities), **build_stats},
            )

        # Find paths
        path_finder = PathFinder(
            graph_store,
            alpha=self._alpha,
            threshold=0.3,
            max_total_edges=15,
        )
        path_scores = await path_finder.find_paths(
            source_entities=seed_entities,
            max_hops=self._max_hops,
        )

        paths = [ps.path for ps in path_scores]

        # Generate evidence
        path_evidence = await path_finder.generate_path_evidence(paths)

        # Format entity context
        entity_context = "\n".join([
            f"- {e.name} ({e.type}): {e.description[:100]}"
            for e in entities[:15]
        ])

        # Build context from paths and entities
        context_text = f"## Path-based Evidence\n{path_evidence}\n\n## Related Entities\n{entity_context}"
        llm_start = time.perf_counter()
        answer, answer_text = await generate_constrained_answer(
            question=question,
            question_type=question_type,
            context=context_text,
            options=options,
            temperature=0.1,
        )
        llm_ms = (time.perf_counter() - llm_start) * 1000
        kg_query_ms = (time.perf_counter() - kg_query_start) * 1000

        await vector_store.delete_collection("all")
        await vector_store.close()
        total_ms = (time.perf_counter() - start_time) * 1000

        return ModelResponse(
            answer=answer,
            response_text=answer_text,
            latency_ms=total_ms,
            metadata={
                "strategy": "retrieve_then_pathrag",
                "max_hops": self._max_hops,
                "alpha": self._alpha,
                "retrieval_top_k": self._retrieval_top_k,
                "retrieval_ms": retrieval_ms,
                "embed_ms": retrieval_meta.get("embed_ms"),
                "search_ms": retrieval_meta.get("search_ms"),
                "abstract_ms": retrieval_meta.get("abstract_ms"),
                "retrieval_source": retrieval_meta.get("source"),
                "kg_query_ms": kg_query_ms,
                "llm_ms": llm_ms,
                "num_abstracts": len(abstracts),
                "num_entities": len(entities),
                "num_relations": len(relations),
                "num_paths": len(paths),
                "seed_entities": seed_entities[:5],
                **build_stats,
            },
        )

    async def _build_mini_kg(
        self,
        abstracts: list[RetrievedAbstract],
    ) -> tuple[GraphStore, dict]:
        """Build mini-KG from retrieved abstracts with extraction caching.

        Uses ExtractionCache to avoid redundant LLM calls.
        """
        from src.cache.extraction_cache import ExtractionCache
        from kg.storage import GraphStore

        start_time = time.perf_counter()
        graph_store = GraphStore()
        extraction_cache = ExtractionCache(extractor_id=_extractor_id())

        # Prepare data from abstracts
        pmid_to_abstract = {}
        pmid_to_text = {}
        for abstract in abstracts:
            text = f"{abstract.title}\n\n{abstract.abstract}"
            pmid_to_abstract[_doc_key(abstract)] = abstract
            pmid_to_text[_doc_key(abstract)] = text

        pmids = list(pmid_to_abstract.keys())

        # Partition into cached vs uncached
        cached_pmids, uncached_pmids = await extraction_cache.partition_pmids(pmids)
        cache_hit_rate = len(cached_pmids) / max(1, len(pmids))

        # Load cached extractions
        chunk_entities = {}
        for pmid in cached_pmids:
            result = await extraction_cache.load(pmid)
            if result:
                entities, relations = result
                chunk_entities[pmid] = entities
                for entity in entities:
                    graph_store.add_entity(entity)
                for relation in relations:
                    graph_store.add_relation(relation)

        # Extract for uncached PMIDs
        extraction_start = time.perf_counter()
        extraction_ms = 0.0

        if uncached_pmids:
            chunks_to_extract = [
                (pmid, pmid_to_text[pmid])
                for pmid in uncached_pmids
            ]

            entity_results = await self._entity_extractor.extract_batch(
                chunks_to_extract,
                concurrency=10,
            )

            for result in entity_results:
                chunk_entities[result.chunk_id] = result.entities
                for entity in result.entities:
                    graph_store.add_entity(entity)

            relation_inputs = [
                (pmid, pmid_to_text[pmid], chunk_entities.get(pmid, []))
                for pmid in uncached_pmids
                if chunk_entities.get(pmid)
            ]

            pmid_relations = {pmid: [] for pmid in uncached_pmids}
            if relation_inputs:
                relation_results = await self._relation_extractor.extract_batch(
                    relation_inputs,
                    concurrency=10,
                )

                for result in relation_results:
                    pmid_relations[result.chunk_id] = result.relations
                    for relation in result.relations:
                        graph_store.add_relation(relation)

            # Cache newly extracted results
            for pmid in uncached_pmids:
                abstract = pmid_to_abstract[pmid]
                text = pmid_to_text[pmid]
                entities = chunk_entities.get(pmid, [])
                relations = pmid_relations.get(pmid, [])
                await extraction_cache.set(
                    pmid=pmid,
                    title=abstract.title,
                    text=text,
                    entities=entities,
                    relations=relations,
                )

            extraction_ms = (time.perf_counter() - extraction_start) * 1000

        build_ms = (time.perf_counter() - start_time) * 1000
        build_stats = {
            "kg_build_ms": build_ms,
            "extraction_ms": extraction_ms,
            "cache_hits": len(cached_pmids),
            "cache_total": len(pmids),
            "cache_hit_rate": cache_hit_rate,
        }
        return graph_store, build_stats

    def _extract_answer(self, response, question_type, options=None):
        """Extract structured answer."""
        import re
        response_lower = response.lower().strip()

        if question_type == "yesno":
            if any(w in response_lower[:50] for w in ["yes", "correct", "true"]):
                return "yes"
            elif any(w in response_lower[:50] for w in ["no", "incorrect", "false"]):
                return "no"
            return "yes" if "yes" in response_lower else "no"

        elif question_type == "mcq" and options:
            # Use smarter extraction for MCQ to avoid false positives
            keys = list(options.keys())
            keys_pattern = "|".join(re.escape(k) for k in keys)

            # Priority 1: Explicit answer patterns (e.g., "Answer: A", "The answer is B")
            explicit_patterns = [
                r"(?:answer|correct answer|correct option)(?:\s+is)?[:\s]+[(]?(" + keys_pattern + r")[).]?",
                r"\*\*(" + keys_pattern + r")\*\*",  # Bold answer
                r"^(" + keys_pattern + r")[.)\s]",   # Answer at start of line
                r"\n(" + keys_pattern + r")[.)\s]",  # Answer at start of new line
            ]
            for pattern in explicit_patterns:
                match = re.search(pattern, response, re.IGNORECASE | re.MULTILINE)
                if match:
                    matched_key = match.group(1).upper()
                    if matched_key in keys:
                        return matched_key

            # Priority 2: Check for option letter with answer-indicating punctuation
            for key in keys:
                # Only match single letters when followed by answer punctuation: "A.", "A)", "A:"
                # or in parentheses "(A)" - not standalone article "a"
                answer_indicator_patterns = [
                    r"(?<![a-zA-Z])(" + re.escape(key) + r")[.):,]",  # A. A) A: A,
                    r"\((" + re.escape(key) + r")\)",  # (A)
                ]
                for pattern in answer_indicator_patterns:
                    if re.search(pattern, response[:150], re.IGNORECASE):
                        return key

            # Priority 3: Check if full option text appears in response
            for key, value in options.items():
                if len(value) > 5 and value.lower() in response_lower:
                    return key

            # Fallback: return first option
            return keys[0] if keys else ""

        elif question_type == "factoid":
            sentences = response.split(".")
            return sentences[0].strip()[:100] if sentences else response[:100]

        elif question_type == "list":
            lines = [l.strip() for l in response.split("\n") if l.strip()]
            items = []
            for line in lines:
                if line.startswith(("-", "*", "•")) or (line and line[0].isdigit()):
                    item = line.lstrip("-*•0123456789.) ")
                    if item:
                        items.append(item)
            return ", ".join(items[:10]) if items else response[:200]

        return response[:500]


class RetrieveThenGraphRAG(ModelClient):
    """Retrieve-then-GraphRAG: Dense retrieval followed by GraphRAG reasoning.

    Pipeline:
    1. VectorSearch → top-k abstracts
    2. Entity/Relation extraction from abstracts
    3. Build mini-KG with community detection
    4. GraphRAG local/global search
    """

    def __init__(
        self,
        mode: Literal["local", "global"] = "local",
        community_level: int = 1,
        retrieval_top_k: int = 20,
        generate_reports: bool = True,
        qdrant_url: str = None,  # Uses config.qdrant.url if None
        pg_dsn: str | None = None,
        paper_collection: str = "paper-full",
    ):
        """Initialize Retrieve-then-GraphRAG client.

        Args:
            mode: Search mode (local or global)
            community_level: Community hierarchy level for global search
            retrieval_top_k: Number of documents to retrieve
            generate_reports: Whether to generate community reports
            qdrant_url: Qdrant server URL
            paper_collection: Collection name for paper vectors
        """
        self._mode = mode
        self._community_level = community_level
        self._retrieval_top_k = retrieval_top_k
        self._generate_reports = generate_reports
        cfg = get_config()
        self._qdrant_url = qdrant_url if qdrant_url else cfg.qdrant.url
        self._pg_dsn = pg_dsn or cfg.postgres.pubmed_url
        self._paper_collection = paper_collection

        self._qdrant: AsyncQdrantClient | None = None
        self._pool: asyncpg.Pool | None = None
        self._vector_search: VectorSearch | None = None
        self._entity_extractor: EntityExtractor | None = None
        self._relation_extractor: RelationExtractor | None = None
        self._initialized = False

    async def __aenter__(self):
        """Initialize async resources."""
        from kg.extraction import EntityExtractor, RelationExtractor

        self._qdrant = AsyncQdrantClient(self._qdrant_url, timeout=60)
        self._pool = await asyncpg.create_pool(
            self._pg_dsn,
            min_size=2,
            max_size=10,
        )
        self._vector_search = VectorSearch(self._qdrant, collection=self._paper_collection)
        self._entity_extractor = EntityExtractor()
        self._relation_extractor = RelationExtractor()
        self._initialized = True
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Close async resources."""
        if self._pool:
            await self._pool.close()
        if self._qdrant:
            await self._qdrant.close()
        self._initialized = False

    async def generate(
        self,
        question: str,
        question_type: str,
        context: list[str] | None = None,
        options: dict[str, str] | None = None,
    ) -> ModelResponse:
        """Generate answer using retrieve-then-GraphRAG pipeline."""
        if not self._initialized:
            raise RuntimeError("Client not initialized. Use async context manager.")

        # Import GraphRAG lazily so LightRAG/PathRAG runs do not pay the import
        # cost or fail on optional GraphRAG dependencies during fresh process start.
        from kg.graphrag.community import CommunityDetector, CommunityReportGenerator
        from kg.graphrag.search import LocalSearch, GlobalSearch
        from kg.storage.vector_store_inmem import InMemoryKGVectorStore as KGVectorStore

        start_time = time.perf_counter()

        abstracts, retrieval_meta = await _retrieve_stage1_abstracts(
            question=question,
            context=context,
            qdrant=self._qdrant,
            pool=self._pool,
            retrieval_top_k=self._retrieval_top_k,
            collection=self._paper_collection,
        )
        retrieval_ms = retrieval_meta.get("retrieval_ms", 0.0)

        if not abstracts:
            return ModelResponse(
                answer="No relevant documents found.",
                response_text="No relevant documents found.",
                latency_ms=(time.perf_counter() - start_time) * 1000,
                metadata={"error": "no_documents"},
            )

        # Stage 2: Build mini-KG
        graph_store, build_stats = await self._build_mini_kg(abstracts)

        # Stage 3: Detect communities
        communities = []
        community_detection_error = None
        if graph_store.num_entities >= 2 and graph_store.num_relations > 0:
            community_detector = CommunityDetector(graph_store)
            try:
                communities = await community_detector.detect(
                    max_cluster_size=10,
                    levels=2,
                )
            except Exception as exc:
                community_detection_error = f"{type(exc).__name__}: {exc}"
        else:
            community_detection_error = "insufficient_graph_structure"

        # Stage 4: Generate community reports (optional, for global search)
        if self._generate_reports and communities and self._mode == "global":
            report_generator = CommunityReportGenerator(graph_store)
            communities = await report_generator.generate_reports(
                communities, concurrency=5
            )

        # Add communities to graph store
        for community in communities:
            graph_store.add_community(community)

        # Stage 5: Build vector indices
        vector_store = KGVectorStore(namespace=f"rtkg_graph_{uuid.uuid4().hex[:12]}")
        entities = list(graph_store.iter_entities())
        relations = list(graph_store.iter_relations())

        if entities:
            await vector_store.upsert_entities(entities)
        if relations:
            await vector_store.upsert_relations(relations)

        # Stage 6: GraphRAG search with constrained output
        kg_query_start = time.perf_counter()
        global_fallback_used = False
        local_fallback_used = False
        graph_search_error = None
        if self._mode == "global":
            search = GlobalSearch(graph_store, vector_store)
            try:
                result = await search.search(
                    question,
                    community_level=self._community_level,
                    max_communities=10,
                    question_type=None,
                    options=None,
                    context_only=True,
                )
            except Exception as exc:
                graph_search_error = f"{type(exc).__name__}: {exc}"
                result = await self._fallback_global_answer(
                    question=question,
                    question_type=question_type,
                    options=options,
                    communities=communities,
                    entities=entities,
                    relations=relations,
                )
                global_fallback_used = True
            if not (result.context_text or "").strip():
                result = await self._fallback_global_answer(
                    question=question,
                    question_type=question_type,
                    options=options,
                    communities=communities,
                    entities=entities,
                    relations=relations,
                )
                global_fallback_used = True
        else:
            search = LocalSearch(graph_store, vector_store)
            try:
                result = await search.search(
                    question,
                    top_k=20,
                    question_type=None,
                    options=None,
                    context_only=True,
                )
            except Exception as exc:
                graph_search_error = f"{type(exc).__name__}: {exc}"
                result = self._fallback_local_result(
                    abstracts=abstracts,
                    entities=entities,
                    relations=relations,
                )
                local_fallback_used = True
            if not (result.context_text or "").strip():
                result = self._fallback_local_result(
                    abstracts=abstracts,
                    entities=entities,
                    relations=relations,
                )
                local_fallback_used = True
        kg_query_ms = (time.perf_counter() - kg_query_start) * 1000

        llm_start = time.perf_counter()
        answer, answer_text = await generate_constrained_answer(
            question=question,
            question_type=question_type,
            context=result.context_text or "No graph context available.",
            options=options,
            temperature=0.1,
        )
        llm_ms = (time.perf_counter() - llm_start) * 1000

        await vector_store.delete_collection("all")
        await vector_store.close()
        total_ms = (time.perf_counter() - start_time) * 1000

        return ModelResponse(
            answer=answer,
            response_text=answer_text,
            latency_ms=total_ms,
            metadata={
                "strategy": "retrieve_then_graphrag",
                "mode": self._mode,
                "community_level": self._community_level,
                "retrieval_top_k": self._retrieval_top_k,
                "retrieval_ms": retrieval_ms,
                "embed_ms": retrieval_meta.get("embed_ms"),
                "search_ms": retrieval_meta.get("search_ms"),
                "abstract_ms": retrieval_meta.get("abstract_ms"),
                "retrieval_source": retrieval_meta.get("source"),
                "kg_query_ms": kg_query_ms,
                "llm_ms": llm_ms,
                "kg_context_chars": len(result.context_text or ""),
                "num_abstracts": len(abstracts),
                "num_entities": len(entities),
                "num_relations": len(relations),
                "num_communities": len(communities),
                "community_detection_error": community_detection_error,
                "graph_search_error": graph_search_error,
                "global_fallback_used": global_fallback_used,
                "local_fallback_used": local_fallback_used,
                **build_stats,
            },
        )

    @staticmethod
    def _is_no_data_response(answer: str) -> bool:
        if not answer:
            return True
        normalized = answer.strip().lower()
        return (
            "unable to answer" in normalized
            or "insufficient information" in normalized
            or normalized == "i don't know"
            or normalized == "i do not know"
        )

    def _fallback_local_result(
        self,
        abstracts,
        entities,
        relations,
    ):
        """Build a direct context fallback when local GraphRAG cannot search."""
        from kg.base import QueryResult

        abstract_lines = []
        for abstract in abstracts[:6]:
            text = (abstract.abstract or "").replace("\n", " ")
            abstract_lines.append(
                f"[PMID {abstract.pmid}] {abstract.title}\n{text[:500]}"
            )

        entity_lines = []
        for entity in entities[:15]:
            desc = (entity.description or "")[:120]
            entity_lines.append(f"- {entity.name} ({entity.type}): {desc}")

        relation_lines = []
        for relation in relations[:15]:
            desc = f": {(relation.description or '')[:80]}" if relation.description else ""
            relation_lines.append(
                f"- {relation.source} --[{relation.type}]--> {relation.target}{desc}"
            )

        context_parts = []
        if abstract_lines:
            context_parts.append("Retrieved Abstracts:\n" + "\n\n".join(abstract_lines))
        if entity_lines:
            context_parts.append("Entities:\n" + "\n".join(entity_lines))
        if relation_lines:
            context_parts.append("Relations:\n" + "\n".join(relation_lines))

        context_text = "\n\n".join(context_parts)[:8000] or "No graph context available."

        return QueryResult(
            answer="",
            entities=list(entities[:15]),
            relations=list(relations[:15]),
            communities=[],
            context_text=context_text,
            mode="local",
            metadata={"fallback": "direct_graph_context"},
        )

    async def _fallback_global_answer(
        self,
        question: str,
        question_type: str,
        options: dict[str, str] | None,
        communities,
        entities,
        relations,
    ) -> QueryResult:
        """Fallback for GraphRAG global when map-reduce yields no usable points.

        Use existing community summaries directly as context when global map
        extraction yields no usable points.
        """
        from kg.base import QueryResult

        community_lines = []
        for community in communities[:8]:
            title = getattr(community, "title", "") or f"Community {community.id[:8]}"
            summary = getattr(community, "summary", "") or ""
            member_count = len(getattr(community, "member_entities", []) or [])
            if summary:
                community_lines.append(
                    f"[{title}] members={member_count}\n{summary[:600]}"
                )

        entity_lines = []
        for entity in entities[:15]:
            desc = (entity.description or "")[:120]
            entity_lines.append(f"- {entity.name} ({entity.type}): {desc}")

        relation_lines = []
        for relation in relations[:15]:
            desc = f": {(relation.description or '')[:80]}" if relation.description else ""
            relation_lines.append(
                f"- {relation.source} --[{relation.type}]--> {relation.target}{desc}"
            )

        context_parts = []
        if community_lines:
            context_parts.append("Community Reports:\n" + "\n\n".join(community_lines))
        if entity_lines:
            context_parts.append("Entities:\n" + "\n".join(entity_lines))
        if relation_lines:
            context_parts.append("Relations:\n" + "\n".join(relation_lines))

        context_text = "\n\n".join(context_parts)[:8000] or "No graph context available."

        return QueryResult(
            answer="",
            communities=list(communities[:8]),
            entities=list(entities[:15]),
            relations=list(relations[:15]),
            context_text=context_text,
            mode="global",
            metadata={"fallback": "community_summary_direct"},
        )

    async def _build_mini_kg(
        self,
        abstracts: list[RetrievedAbstract],
    ) -> tuple[GraphStore, dict]:
        """Build mini-KG from retrieved abstracts with extraction caching.

        Uses ExtractionCache to avoid redundant LLM calls.
        """
        from src.cache.extraction_cache import ExtractionCache
        from kg.storage import GraphStore

        start_time = time.perf_counter()
        graph_store = GraphStore()
        extraction_cache = ExtractionCache(extractor_id=_extractor_id())

        # Prepare data from abstracts
        pmid_to_abstract = {}
        pmid_to_text = {}
        for abstract in abstracts:
            text = f"{abstract.title}\n\n{abstract.abstract}"
            pmid_to_abstract[_doc_key(abstract)] = abstract
            pmid_to_text[_doc_key(abstract)] = text

        pmids = list(pmid_to_abstract.keys())

        # Partition into cached vs uncached
        cached_pmids, uncached_pmids = await extraction_cache.partition_pmids(pmids)
        cache_hit_rate = len(cached_pmids) / max(1, len(pmids))

        # Load cached extractions
        chunk_entities = {}
        for pmid in cached_pmids:
            result = await extraction_cache.load(pmid)
            if result:
                entities, relations = result
                chunk_entities[pmid] = entities
                for entity in entities:
                    graph_store.add_entity(entity)
                for relation in relations:
                    graph_store.add_relation(relation)

        # Extract for uncached PMIDs
        extraction_start = time.perf_counter()
        extraction_ms = 0.0

        if uncached_pmids:
            chunks_to_extract = [
                (pmid, pmid_to_text[pmid])
                for pmid in uncached_pmids
            ]

            entity_results = await self._entity_extractor.extract_batch(
                chunks_to_extract,
                concurrency=10,
            )

            for result in entity_results:
                chunk_entities[result.chunk_id] = result.entities
                for entity in result.entities:
                    graph_store.add_entity(entity)

            relation_inputs = [
                (pmid, pmid_to_text[pmid], chunk_entities.get(pmid, []))
                for pmid in uncached_pmids
                if chunk_entities.get(pmid)
            ]

            pmid_relations = {pmid: [] for pmid in uncached_pmids}
            if relation_inputs:
                relation_results = await self._relation_extractor.extract_batch(
                    relation_inputs,
                    concurrency=10,
                )

                for result in relation_results:
                    pmid_relations[result.chunk_id] = result.relations
                    for relation in result.relations:
                        graph_store.add_relation(relation)

            # Cache newly extracted results
            for pmid in uncached_pmids:
                abstract = pmid_to_abstract[pmid]
                text = pmid_to_text[pmid]
                entities = chunk_entities.get(pmid, [])
                relations = pmid_relations.get(pmid, [])
                await extraction_cache.set(
                    pmid=pmid,
                    title=abstract.title,
                    text=text,
                    entities=entities,
                    relations=relations,
                )

            extraction_ms = (time.perf_counter() - extraction_start) * 1000

        build_ms = (time.perf_counter() - start_time) * 1000
        build_stats = {
            "kg_build_ms": build_ms,
            "extraction_ms": extraction_ms,
            "cache_hits": len(cached_pmids),
            "cache_total": len(pmids),
            "cache_hit_rate": cache_hit_rate,
        }
        return graph_store, build_stats

    def _extract_answer(self, response, question_type, options=None):
        """Extract structured answer."""
        import re
        response_lower = response.lower().strip()

        if question_type == "yesno":
            if any(w in response_lower[:50] for w in ["yes", "correct", "true"]):
                return "yes"
            elif any(w in response_lower[:50] for w in ["no", "incorrect", "false"]):
                return "no"
            return "yes" if "yes" in response_lower else "no"

        elif question_type == "mcq" and options:
            # Use smarter extraction for MCQ to avoid false positives
            keys = list(options.keys())
            keys_pattern = "|".join(re.escape(k) for k in keys)

            # Priority 1: Explicit answer patterns (e.g., "Answer: A", "The answer is B")
            explicit_patterns = [
                r"(?:answer|correct answer|correct option)(?:\s+is)?[:\s]+[(]?(" + keys_pattern + r")[).]?",
                r"\*\*(" + keys_pattern + r")\*\*",  # Bold answer
                r"^(" + keys_pattern + r")[.)\s]",   # Answer at start of line
                r"\n(" + keys_pattern + r")[.)\s]",  # Answer at start of new line
            ]
            for pattern in explicit_patterns:
                match = re.search(pattern, response, re.IGNORECASE | re.MULTILINE)
                if match:
                    matched_key = match.group(1).upper()
                    if matched_key in keys:
                        return matched_key

            # Priority 2: Check for option letter with answer-indicating punctuation
            for key in keys:
                # Only match single letters when followed by answer punctuation: "A.", "A)", "A:"
                # or in parentheses "(A)" - not standalone article "a"
                answer_indicator_patterns = [
                    r"(?<![a-zA-Z])(" + re.escape(key) + r")[.):,]",  # A. A) A: A,
                    r"\((" + re.escape(key) + r")\)",  # (A)
                ]
                for pattern in answer_indicator_patterns:
                    if re.search(pattern, response[:150], re.IGNORECASE):
                        return key

            # Priority 3: Check if full option text appears in response
            for key, value in options.items():
                if len(value) > 5 and value.lower() in response_lower:
                    return key

            # Fallback: return first option
            return keys[0] if keys else ""

        elif question_type == "factoid":
            sentences = response.split(".")
            return sentences[0].strip()[:100] if sentences else response[:100]

        elif question_type == "list":
            lines = [l.strip() for l in response.split("\n") if l.strip()]
            items = []
            for line in lines:
                if line.startswith(("-", "*", "•")) or (line and line[0].isdigit()):
                    item = line.lstrip("-*•0123456789.) ")
                    if item:
                        items.append(item)
            return ", ".join(items[:10]) if items else response[:200]

        return response[:500]


# =============================================================================
# Convenience factories
# =============================================================================

RETRIEVE_THEN_KG_STRATEGIES = [
    "rt_lightrag_naive",
    "rt_lightrag_local",
    "rt_lightrag_global",
    "rt_lightrag_hybrid",
    "rt_pathrag",
    "rt_graphrag_local",
    "rt_graphrag_global",
]


def create_retrieve_then_kg_client(
    strategy: str,
    retrieval_top_k: int = 20,
    **kwargs,
) -> ModelClient:
    """Create Retrieve-then-KG client by strategy name.

    Args:
        strategy: Strategy name (e.g., rt_lightrag_hybrid, rt_pathrag)
        retrieval_top_k: Number of documents to retrieve
        **kwargs: Additional options

    Returns:
        ModelClient instance
    """
    if strategy.startswith("rt_lightrag"):
        mode = strategy.split("_")[2]  # rt_lightrag_hybrid → hybrid
        return RetrieveThenLightRAG(
            mode=mode,
            retrieval_top_k=retrieval_top_k,
            **kwargs,
        )

    elif strategy == "rt_pathrag":
        return RetrieveThenPathRAG(
            retrieval_top_k=retrieval_top_k,
            **kwargs,
        )

    elif strategy.startswith("rt_graphrag"):
        mode = strategy.split("_")[2]  # rt_graphrag_local → local
        return RetrieveThenGraphRAG(
            mode=mode,
            retrieval_top_k=retrieval_top_k,
            **kwargs,
        )

    else:
        raise ValueError(f"Unknown strategy: {strategy}")
