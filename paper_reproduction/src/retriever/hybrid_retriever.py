"""Hybrid PubMed Retriever with Multi-Path Fusion and Reranking.

This module implements a more sophisticated retrieval pipeline:
1. Multi-path retrieval (MeSH, Keyword, Dense)
2. RRF (Reciprocal Rank Fusion) for combining results
3. Reranking with Qwen3-Reranker-8B for final ordering

Architecture:
    Query → Parser → [MeSH Path, Keyword Path, Dense Path]
                              ↓
                    RRF Fusion → Top 200
                              ↓
                    Rerank → Final Top 50
"""

import asyncio
import time
from dataclasses import dataclass, field
from typing import Sequence

import asyncpg
from qdrant_client import AsyncQdrantClient

from config import get_config as _get_config

from .models import (
    FilterStats,
    ParsedQuery,
    RetrievalResult,
    RetrievedAbstract,
    FilterMode,
)
from .query_parser import QueryParser
from .mesh_filter import MeSHFilter
from .keyword_filter import KeywordFilter
from .vector_search import VectorSearch
from utils.clients import rerank_client

# Get config for defaults
_cfg = _get_config()


@dataclass
class PathResult:
    """Result from a single retrieval path."""
    path_name: str
    pmids: list[str]  # Ordered by score (best first)
    scores: dict[str, float]  # pmid -> score
    time_ms: float


@dataclass
class HybridConfig:
    """Configuration for hybrid retriever.

    Default configuration is B6 (MeSH+Keyword+Vector+Rerank), optimal for
    multi-document retrieval tasks like BioASQ.

    Experimental Results (exp02-1, 500 questions each):
    ┌─────────┬──────────┬──────────┬──────────┬──────────┐
    │ Config  │ PQAL R@1 │ PQAL R@10│BioASQ R@1│BioASQ R@10│
    ├─────────┼──────────┼──────────┼──────────┼──────────┤
    │ B6      │  38.0%   │  42.8%   │  29.8%   │  58.6%   │
    │ C2      │  46.4%   │  53.4%   │  21.8%   │  39.8%   │
    └─────────┴──────────┴──────────┴──────────┴──────────┘

    Strategy Selection:
    - Single-doc retrieval (PQAL-like): Use C2_FAST preset
    - Multi-doc retrieval (BioASQ-like): Use B6_DEFAULT preset (default)

    Example:
        # Default (B6 - multi-doc)
        retriever = HybridRetriever()

        # Fast mode (C2 - single-doc)
        retriever = HybridRetriever(C2_FAST)
    """
    # Path enables (B6 default: MeSH + Keyword paths, no pure Dense)
    enable_mesh_path: bool = True
    enable_keyword_path: bool = True
    enable_dense_path: bool = False  # B6 uses filter+vector, not pure dense

    # Path modes: "filter_only" skips vector search, "vector" does filter+vector
    mesh_path_mode: str = "vector"      # "filter_only" | "vector"
    keyword_path_mode: str = "vector"   # "filter_only" | "vector"

    # Vector query source (for fair comparison without PubMedQA title bias)
    vector_query_type: str = "entity"   # "entity" | "keyword" | "full"

    # Per-path limits
    mesh_path_limit: int = 100      # Top results from MeSH path
    keyword_path_limit: int = 100   # Top results from Keyword path
    dense_path_limit: int = 100     # Top results from Dense path

    # MeSH filter parameters
    mesh_filter_limit: int = 5000   # Max PMIDs from MeSH filter before vector search
    mesh_top_k: int = 20
    mesh_llm_select: int = 15

    # Keyword filter parameters
    keyword_filter_limit: int = 5000

    # Fusion parameters
    rrf_k: int = 60                 # RRF constant (typically 60)
    fusion_limit: int = 200         # Top candidates after fusion

    # Rerank parameters
    enable_rerank: bool = True
    rerank_limit: int = 50          # Final results after rerank

    # Database config (from config.py)
    pubmed_db_url: str = field(default_factory=lambda: _cfg.postgres.pubmed_url)
    qdrant_url: str = field(default_factory=lambda: _cfg.qdrant.url)
    mesh_collection: str = field(default_factory=lambda: _cfg.qdrant.mesh_collection)
    paper_collection: str = field(default_factory=lambda: _cfg.qdrant.paper_collection)


# =============================================================================
# Preset Configurations
# =============================================================================

def B6_DEFAULT() -> HybridConfig:
    """B6: MeSH + Keyword + Vector + Rerank (Default).

    Optimal for multi-document retrieval (BioASQ-like tasks).
    - BioASQ: R@1=29.8%, R@10=58.6%, GT Ceiling=66.8%
    - PQAL: R@1=38.0%, R@10=42.8%
    - Latency: ~25s
    """
    return HybridConfig(
        enable_mesh_path=True,
        enable_keyword_path=True,
        enable_dense_path=False,
        mesh_path_mode="vector",
        keyword_path_mode="vector",
        vector_query_type="entity",
        enable_rerank=True,
    )


def C2_FAST() -> HybridConfig:
    """C2: Entity-Only Dense + Rerank (Fast Mode).

    Optimal for single-document retrieval (PQAL-like tasks).
    - PQAL: R@1=46.4%, R@10=53.4%, GT Ceiling=53.6%
    - BioASQ: R@1=21.8%, R@10=39.8%
    - Latency: ~2s (12x faster than B6)
    """
    return HybridConfig(
        enable_mesh_path=False,
        enable_keyword_path=False,
        enable_dense_path=True,
        vector_query_type="entity",
        enable_rerank=True,
    )


def B2_BALANCED() -> HybridConfig:
    """B2: MeSH + Vector + Rerank (Balanced).

    Middle ground between B6 and C2.
    - BioASQ: R@1=24.2%, R@10=47.4%
    - PQAL: R@1=34.0%, R@10=37.4%
    - Latency: ~15s
    """
    return HybridConfig(
        enable_mesh_path=True,
        enable_keyword_path=False,
        enable_dense_path=False,
        mesh_path_mode="vector",
        vector_query_type="entity",
        enable_rerank=True,
    )


# =============================================================================
# All 14 Strategy Presets (from exp02-1 ablation study)
# =============================================================================

# Group A: Filter-Only (No Vector Search)

def A1_MESH_ONLY() -> HybridConfig:
    """A1: MeSH Filter Only (no vector search, no rerank)."""
    return HybridConfig(
        enable_mesh_path=True,
        enable_keyword_path=False,
        enable_dense_path=False,
        mesh_path_mode="filter_only",
        enable_rerank=False,
    )


def A2_MESH_RERANK() -> HybridConfig:
    """A2: MeSH Filter + Rerank."""
    return HybridConfig(
        enable_mesh_path=True,
        enable_keyword_path=False,
        enable_dense_path=False,
        mesh_path_mode="filter_only",
        enable_rerank=True,
    )


def A3_KEYWORD_ONLY() -> HybridConfig:
    """A3: Keyword Filter Only (no vector search, no rerank)."""
    return HybridConfig(
        enable_mesh_path=False,
        enable_keyword_path=True,
        enable_dense_path=False,
        keyword_path_mode="filter_only",
        enable_rerank=False,
    )


def A4_KEYWORD_RERANK() -> HybridConfig:
    """A4: Keyword Filter + Rerank."""
    return HybridConfig(
        enable_mesh_path=False,
        enable_keyword_path=True,
        enable_dense_path=False,
        keyword_path_mode="filter_only",
        enable_rerank=True,
    )


def A5_MESH_KW_RRF() -> HybridConfig:
    """A5: MeSH + Keyword RRF (filter_only, no rerank)."""
    return HybridConfig(
        enable_mesh_path=True,
        enable_keyword_path=True,
        enable_dense_path=False,
        mesh_path_mode="filter_only",
        keyword_path_mode="filter_only",
        enable_rerank=False,
    )


def A6_MESH_KW_RRF_RERANK() -> HybridConfig:
    """A6: MeSH + Keyword RRF + Rerank."""
    return HybridConfig(
        enable_mesh_path=True,
        enable_keyword_path=True,
        enable_dense_path=False,
        mesh_path_mode="filter_only",
        keyword_path_mode="filter_only",
        enable_rerank=True,
    )


# Group B: Filter + Vector Search

def B1_MESH_VECTOR() -> HybridConfig:
    """B1: MeSH + Vector (no rerank)."""
    return HybridConfig(
        enable_mesh_path=True,
        enable_keyword_path=False,
        enable_dense_path=False,
        mesh_path_mode="vector",
        vector_query_type="entity",
        enable_rerank=False,
    )


def B3_KEYWORD_VECTOR() -> HybridConfig:
    """B3: Keyword + Vector (no rerank)."""
    return HybridConfig(
        enable_mesh_path=False,
        enable_keyword_path=True,
        enable_dense_path=False,
        keyword_path_mode="vector",
        vector_query_type="entity",
        enable_rerank=False,
    )


def B4_KEYWORD_VECTOR_RERANK() -> HybridConfig:
    """B4: Keyword + Vector + Rerank."""
    return HybridConfig(
        enable_mesh_path=False,
        enable_keyword_path=True,
        enable_dense_path=False,
        keyword_path_mode="vector",
        vector_query_type="entity",
        enable_rerank=True,
    )


def B5_MESH_KW_VECTOR() -> HybridConfig:
    """B5: MeSH + Keyword + Vector (no rerank)."""
    return HybridConfig(
        enable_mesh_path=True,
        enable_keyword_path=True,
        enable_dense_path=False,
        mesh_path_mode="vector",
        keyword_path_mode="vector",
        vector_query_type="entity",
        enable_rerank=False,
    )


# Group C: Dense-Only

def C1_DENSE_ONLY() -> HybridConfig:
    """C1: Entity-Only Dense (no rerank)."""
    return HybridConfig(
        enable_mesh_path=False,
        enable_keyword_path=False,
        enable_dense_path=True,
        vector_query_type="entity",
        enable_rerank=False,
    )


# =============================================================================
# Strategy Preset Mapping
# =============================================================================

from typing import Callable

STRATEGY_PRESETS: dict[str, Callable[[], HybridConfig]] = {
    # Group A: Filter-Only
    "A1": A1_MESH_ONLY,
    "A2": A2_MESH_RERANK,
    "A3": A3_KEYWORD_ONLY,
    "A4": A4_KEYWORD_RERANK,
    "A5": A5_MESH_KW_RRF,
    "A6": A6_MESH_KW_RRF_RERANK,
    # Group B: Filter + Vector
    "B1": B1_MESH_VECTOR,
    "B2": B2_BALANCED,  # MeSH + Vector + Rerank
    "B3": B3_KEYWORD_VECTOR,
    "B4": B4_KEYWORD_VECTOR_RERANK,
    "B5": B5_MESH_KW_VECTOR,
    "B6": B6_DEFAULT,    # MeSH + Keyword + Vector + Rerank (DEFAULT)
    # Group C: Dense-Only
    "C1": C1_DENSE_ONLY,
    "C2": C2_FAST,       # Dense + Rerank (FAST)
}


def get_config_by_strategy(strategy: str) -> HybridConfig:
    """Get HybridConfig by strategy name (A1-A6, B1-B6, C1-C2).

    Args:
        strategy: Strategy ID (e.g., "B6", "C2")

    Returns:
        HybridConfig for the specified strategy

    Raises:
        ValueError: If strategy is unknown
    """
    strategy = strategy.upper()
    if strategy not in STRATEGY_PRESETS:
        available = ", ".join(sorted(STRATEGY_PRESETS.keys()))
        raise ValueError(f"Unknown strategy: {strategy}. Available: {available}")
    return STRATEGY_PRESETS[strategy]()


class HybridRetriever:
    """Hybrid retriever with multi-path fusion and reranking."""

    def __init__(self, config: HybridConfig | None = None):
        self._config = config or HybridConfig()
        self._initialized = False

        self._db_pool: asyncpg.Pool | None = None
        self._qdrant: AsyncQdrantClient | None = None

        self._query_parser: QueryParser | None = None
        self._mesh_filter: MeSHFilter | None = None
        self._keyword_filter: KeywordFilter | None = None
        self._vector_search: VectorSearch | None = None

    async def initialize(self) -> None:
        """Initialize connections and components."""
        if self._initialized:
            return

        self._db_pool = await asyncpg.create_pool(
            self._config.pubmed_db_url,
            min_size=2,
            max_size=10,
        )

        self._qdrant = AsyncQdrantClient(
            self._config.qdrant_url,
            check_compatibility=False,
            timeout=180,  # Increased for high-latency VPN connections
        )

        self._query_parser = QueryParser()
        self._mesh_filter = MeSHFilter(
            self._qdrant,
            self._db_pool,
            mesh_collection=self._config.mesh_collection,
            top_k=self._config.mesh_top_k,
            llm_select=self._config.mesh_llm_select,
        )
        self._keyword_filter = KeywordFilter(self._db_pool)
        self._vector_search = VectorSearch(
            self._qdrant,
            collection=self._config.paper_collection,
        )

        self._initialized = True

    async def close(self) -> None:
        """Close all connections."""
        if self._db_pool:
            await self._db_pool.close()
        if self._qdrant:
            await self._qdrant.close()
        self._initialized = False

    async def _mesh_path(
        self,
        query: str,
        parsed: ParsedQuery,
    ) -> PathResult:
        """MeSH-based retrieval path.

        entities → MeSH matching → filter PMIDs → vector search → Top N
        """
        start = time.perf_counter()

        # Match MeSH terms using entities
        search_terms = parsed.entities + parsed.mesh_candidates
        mesh_matches, _ = await self._mesh_filter.match_mesh_terms(
            query, entities=search_terms if search_terms else None
        )

        if not mesh_matches:
            return PathResult("mesh", [], {}, (time.perf_counter() - start) * 1000)

        # Get PMIDs from MeSH filter
        mesh_pmids, _ = await self._mesh_filter.get_pmids_by_mesh(
            mesh_matches, limit=self._config.mesh_filter_limit
        )

        if not mesh_pmids:
            return PathResult("mesh", [], {}, (time.perf_counter() - start) * 1000)

        # Vector search within MeSH-filtered PMIDs
        # Use config-based query (entity by default for fair comparison)
        vector_query = self._build_vector_query(query, parsed)
        results, _ = await self._vector_search.search(
            vector_query,
            pmid_filter=mesh_pmids,
            limit=self._config.mesh_path_limit,
        )

        pmids = [r.pmid for r in results]
        scores = {r.pmid: r.score for r in results}

        return PathResult(
            "mesh", pmids, scores,
            (time.perf_counter() - start) * 1000
        )

    async def _keyword_path(
        self,
        query: str,
        parsed: ParsedQuery,
    ) -> PathResult:
        """Keyword-based retrieval path.

        keywords → FTS filter → vector search → Top N
        """
        start = time.perf_counter()

        if not parsed.keywords:
            return PathResult("keyword", [], {}, (time.perf_counter() - start) * 1000)

        # Get PMIDs from keyword filter
        keyword_pmids, _ = await self._keyword_filter.get_pmids_by_keywords(
            parsed.keywords,
            limit=self._config.keyword_filter_limit,
            match_any=True,
        )

        if not keyword_pmids:
            return PathResult("keyword", [], {}, (time.perf_counter() - start) * 1000)

        # Vector search within keyword-filtered PMIDs
        # Use config-based query (entity by default for fair comparison)
        vector_query = self._build_vector_query(query, parsed)
        results, _ = await self._vector_search.search(
            vector_query,
            pmid_filter=keyword_pmids,
            limit=self._config.keyword_path_limit,
        )

        pmids = [r.pmid for r in results]
        scores = {r.pmid: r.score for r in results}

        return PathResult(
            "keyword", pmids, scores,
            (time.perf_counter() - start) * 1000
        )

    def _build_vector_query(self, query: str, parsed: ParsedQuery) -> str:
        """Build vector search query based on config.

        Args:
            query: Original query string
            parsed: Parsed query with entities/keywords

        Returns:
            Query string for vector embedding based on vector_query_type config
        """
        if self._config.vector_query_type == "entity":
            if parsed.entities:
                return " ".join(parsed.entities[:5])
            # Fallback to keywords if no entities
            if parsed.keywords:
                return " ".join(parsed.keywords[:3])
            return query
        elif self._config.vector_query_type == "keyword":
            if parsed.keywords:
                return " ".join(parsed.keywords[:5])
            return query
        else:  # "full" - original behavior (unfair for PubMedQA)
            return query

    async def _dense_path(
        self,
        query: str,
        parsed: ParsedQuery,
    ) -> PathResult:
        """Dense retrieval path (entity-based vector search by default).

        entities → vector search on all papers → Top N
        """
        start = time.perf_counter()

        # Build entity-based query for fair comparison
        vector_query = self._build_vector_query(query, parsed)

        # Direct vector search without filtering
        results, _ = await self._vector_search.search(
            vector_query,
            pmid_filter=None,  # No filtering
            limit=self._config.dense_path_limit,
        )

        pmids = [r.pmid for r in results]
        scores = {r.pmid: r.score for r in results}

        return PathResult(
            "dense", pmids, scores,
            (time.perf_counter() - start) * 1000
        )

    async def _mesh_filter_only_path(
        self,
        query: str,
        parsed: ParsedQuery,
    ) -> PathResult:
        """MeSH-based retrieval with native match_count ranking (no vector search).

        entities → MeSH matching → PMID lookup (ranked by match_count) → Top N
        """
        start = time.perf_counter()

        # Match MeSH terms using entities
        search_terms = parsed.entities + parsed.mesh_candidates
        mesh_matches, _ = await self._mesh_filter.match_mesh_terms(
            query, entities=search_terms if search_terms else None
        )

        if not mesh_matches:
            return PathResult("mesh", [], {}, (time.perf_counter() - start) * 1000)

        # Get PMIDs from MeSH filter (ranked by match_count, return ordered list)
        mesh_pmids, _ = await self._mesh_filter.get_pmids_by_mesh(
            mesh_matches, limit=self._config.mesh_path_limit, return_ordered=True
        )

        if not mesh_pmids:
            return PathResult("mesh", [], {}, (time.perf_counter() - start) * 1000)

        # mesh_pmids is already an ordered list from DB
        pmids = mesh_pmids[:self._config.mesh_path_limit]
        # Assign scores based on rank (higher rank = higher score)
        scores = {pmid: 1.0 - i / len(pmids) for i, pmid in enumerate(pmids)}

        return PathResult(
            "mesh", pmids, scores,
            (time.perf_counter() - start) * 1000
        )

    async def _keyword_filter_only_path(
        self,
        query: str,
        parsed: ParsedQuery,
    ) -> PathResult:
        """Keyword-based retrieval with FTS ranking (no vector search).

        keywords → FTS filter → PMID lookup (FTS order) → Top N
        """
        start = time.perf_counter()

        if not parsed.keywords:
            return PathResult("keyword", [], {}, (time.perf_counter() - start) * 1000)

        # Get PMIDs from keyword filter (return as ordered list)
        keyword_pmids, _ = await self._keyword_filter.get_pmids_by_keywords(
            parsed.keywords,
            limit=self._config.keyword_path_limit,
            match_any=True,
            return_ordered=True,
        )

        if not keyword_pmids:
            return PathResult("keyword", [], {}, (time.perf_counter() - start) * 1000)

        # keyword_pmids is already a list (order is arbitrary from DB, but consistent)
        pmids = keyword_pmids[:self._config.keyword_path_limit]
        # Assign scores based on rank
        scores = {pmid: 1.0 - i / len(pmids) for i, pmid in enumerate(pmids)}

        return PathResult(
            "keyword", pmids, scores,
            (time.perf_counter() - start) * 1000
        )

    def _rrf_fusion(
        self,
        path_results: list[PathResult],
        k: int = 60,
    ) -> list[tuple[str, float]]:
        """Reciprocal Rank Fusion to combine multiple ranked lists.

        RRF_score(d) = Σ 1/(k + rank_i(d))

        Args:
            path_results: Results from each retrieval path
            k: RRF constant (default 60)

        Returns:
            List of (pmid, rrf_score) sorted by score descending
        """
        # Filter out empty results
        non_empty_results = [r for r in path_results if r.pmids]

        if not non_empty_results:
            return []

        # Short-circuit: single path, skip RRF computation
        if len(non_empty_results) == 1:
            result = non_empty_results[0]
            return [(pmid, result.scores.get(pmid, 0.0)) for pmid in result.pmids]

        # Multi-path RRF fusion
        rrf_scores: dict[str, float] = {}

        for result in non_empty_results:
            for rank, pmid in enumerate(result.pmids, start=1):
                if pmid not in rrf_scores:
                    rrf_scores[pmid] = 0.0
                rrf_scores[pmid] += 1.0 / (k + rank)

        # Sort by RRF score descending
        sorted_results = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
        return sorted_results

    async def _rerank(
        self,
        query: str,
        pmids: list[str],
        limit: int = 50,
    ) -> list[tuple[str, float]]:
        """Rerank candidates using Qwen3-Reranker-8B.

        Args:
            query: Original query
            pmids: Candidate PMIDs to rerank
            limit: Number of top results to return

        Returns:
            List of (pmid, rerank_score) sorted by score descending
        """
        if not pmids:
            return []

        # Fetch abstracts for reranking
        async with self._db_pool.acquire() as conn:
            placeholders = ", ".join(f"${i+1}" for i in range(len(pmids)))
            rows = await conn.fetch(f"""
                SELECT pmid::text, title, abstract
                FROM articles
                WHERE pmid IN ({placeholders})
            """, *[int(p) for p in pmids])

        pmid_to_text = {}
        for row in rows:
            title = row["title"] or ""
            abstract = row["abstract"] or ""
            pmid_to_text[row["pmid"]] = f"{title}\n{abstract}"[:1500]  # Truncate

        # Prepare documents for reranking
        documents = []
        valid_pmids = []
        for pmid in pmids:
            if pmid in pmid_to_text:
                documents.append(pmid_to_text[pmid])
                valid_pmids.append(pmid)

        if not documents:
            return []

        # Call rerank service
        try:
            # rerank returns list of (doc_index, score) sorted by score descending
            rerank_results = await rerank_client.rerank(query, documents, top_k=limit)

            # Convert to (pmid, score) pairs
            pmid_scores = [(valid_pmids[idx], score) for idx, score in rerank_results]

            return pmid_scores
        except Exception as e:
            print(f"[HybridRetriever] Rerank failed: {e}, using RRF scores")
            # Fallback: return original order
            return [(pmid, 1.0 - i/len(valid_pmids)) for i, pmid in enumerate(valid_pmids)][:limit]

    async def retrieve(
        self,
        query: str,
        gt_pmids: Sequence[str] | None = None,
    ) -> RetrievalResult:
        """Retrieve relevant abstracts using hybrid multi-path fusion.

        Args:
            query: Natural language query
            gt_pmids: Optional ground truth PMIDs for evaluation

        Returns:
            RetrievalResult with retrieved abstracts and metrics
        """
        if not self._initialized:
            await self.initialize()

        start_time = time.perf_counter()
        gt_set = set(gt_pmids) if gt_pmids else set()

        # 1. Parse query
        parsed = await self._query_parser.parse(query)
        parse_time = parsed.parse_time_ms

        # 2. Run retrieval paths in parallel
        tasks = []
        if self._config.enable_mesh_path:
            if self._config.mesh_path_mode == "filter_only":
                tasks.append(self._mesh_filter_only_path(query, parsed))
            else:
                tasks.append(self._mesh_path(query, parsed))
        if self._config.enable_keyword_path:
            if self._config.keyword_path_mode == "filter_only":
                tasks.append(self._keyword_filter_only_path(query, parsed))
            else:
                tasks.append(self._keyword_path(query, parsed))
        if self._config.enable_dense_path:
            tasks.append(self._dense_path(query, parsed))

        path_results = await asyncio.gather(*tasks)

        # Calculate path times
        mesh_time = sum(r.time_ms for r in path_results if r.path_name == "mesh")
        keyword_time = sum(r.time_ms for r in path_results if r.path_name == "keyword")
        vector_time = sum(r.time_ms for r in path_results if r.path_name == "dense")

        # 3. RRF Fusion
        fusion_start = time.perf_counter()
        fused_results = self._rrf_fusion(path_results, k=self._config.rrf_k)
        fusion_candidates = [pmid for pmid, _ in fused_results[:self._config.fusion_limit]]
        fusion_time = (time.perf_counter() - fusion_start) * 1000

        # 4. Rerank (optional)
        rerank_time = 0.0
        if self._config.enable_rerank and fusion_candidates:
            rerank_start = time.perf_counter()
            reranked = await self._rerank(
                query,
                fusion_candidates,
                limit=self._config.rerank_limit,
            )
            final_pmids = [pmid for pmid, _ in reranked]
            final_scores = {pmid: score for pmid, score in reranked}
            rerank_time = (time.perf_counter() - rerank_start) * 1000
        else:
            final_pmids = fusion_candidates[:self._config.rerank_limit]
            final_scores = {pmid: score for pmid, score in fused_results[:self._config.rerank_limit]}

        # 5. Fetch full abstracts
        abstracts = await self._fetch_abstracts(final_pmids, final_scores)

        # 6. Track GT and build stats
        mesh_pmids = set()
        keyword_pmids = set()
        for r in path_results:
            if r.path_name == "mesh":
                mesh_pmids = set(r.pmids)
            elif r.path_name == "keyword":
                keyword_pmids = set(r.pmids)

        filter_stats = FilterStats(
            mesh_pmid_count=len(mesh_pmids),
            keyword_pmid_count=len(keyword_pmids),
            combined_pmid_count=len(fusion_candidates),
            gt_in_mesh=bool(gt_set & mesh_pmids),
            gt_in_keyword=bool(gt_set & keyword_pmids),
            gt_in_combined=bool(gt_set & set(fusion_candidates)),
        )

        # Track GT ranks
        gt_ranks = {}
        for pmid in gt_set:
            rank = -1
            for abstract in abstracts:
                if abstract.pmid == pmid:
                    rank = abstract.rank
                    break
            gt_ranks[pmid] = rank

        total_time = (time.perf_counter() - start_time) * 1000

        return RetrievalResult(
            query=query,
            abstracts=abstracts,
            parsed_query=parsed,
            mesh_matches=[],  # Not tracking individual matches in hybrid mode
            filter_stats=filter_stats,
            filter_mode=FilterMode.MESH_OR_KEYWORD,  # Hybrid mode
            total_time_ms=total_time,
            parse_time_ms=parse_time,
            mesh_time_ms=mesh_time,
            keyword_time_ms=keyword_time,
            vector_time_ms=vector_time + fusion_time + rerank_time,
            gt_pmids=list(gt_set),
            gt_ranks=gt_ranks,
        )

    async def _fetch_abstracts(
        self,
        pmids: list[str],
        scores: dict[str, float],
    ) -> list[RetrievedAbstract]:
        """Fetch full abstract data for PMIDs."""
        if not pmids:
            return []

        async with self._db_pool.acquire() as conn:
            placeholders = ", ".join(f"${i+1}" for i in range(len(pmids)))
            rows = await conn.fetch(f"""
                SELECT pmid::text, title, abstract, paper_uuid, pmc
                FROM articles
                WHERE pmid IN ({placeholders})
            """, *[int(p) for p in pmids])

        pmid_to_row = {row["pmid"]: row for row in rows}

        abstracts = []
        for rank, pmid in enumerate(pmids, start=1):
            row = pmid_to_row.get(pmid)
            if row:
                abstracts.append(RetrievedAbstract(
                    pmid=pmid,
                    title=row["title"] or "",
                    abstract=row["abstract"] or "",
                    score=scores.get(pmid, 0.0),
                    rank=rank,
                    have_fulltext=bool(row["pmc"]),
                    paper_uuid=str(row["paper_uuid"]) if row["paper_uuid"] else None,
                    pmc=row["pmc"],
                ))

        return abstracts


async def test_hybrid_retriever():
    """Test the hybrid retriever."""
    config = HybridConfig(
        enable_mesh_path=True,
        enable_keyword_path=True,
        enable_dense_path=True,
        enable_rerank=False,  # Test without rerank first
    )

    retriever = HybridRetriever(config)
    await retriever.initialize()

    query = "Does metformin help with type 2 diabetes?"
    result = await retriever.retrieve(query, gt_pmids=["12345678"])

    print(f"Query: {query}")
    print(f"Retrieved {len(result.abstracts)} abstracts")
    print(f"Total time: {result.total_time_ms:.0f}ms")
    print(f"Filter stats: mesh={result.filter_stats.mesh_pmid_count}, "
          f"keyword={result.filter_stats.keyword_pmid_count}, "
          f"combined={result.filter_stats.combined_pmid_count}")

    for abstract in result.abstracts[:5]:
        print(f"  [{abstract.rank}] {abstract.pmid}: {abstract.title[:60]}...")

    await retriever.close()


if __name__ == "__main__":
    import asyncio
    asyncio.run(test_hybrid_retriever())
