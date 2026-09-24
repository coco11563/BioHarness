"""Cached Hybrid Retriever with automatic cache lookup and storage.

This module wraps HybridRetriever to provide:
1. Automatic cache lookup before running retrieval
2. Automatic cache storage after retrieval (optional)
3. GT injection at load time for ablation studies

Usage:
    # Initialize with strategy
    retriever = CachedHybridRetriever(strategy="B6")
    await retriever.initialize()

    # Retrieve - checks cache first, runs retrieval if miss
    pmids = await retriever.retrieve(
        question_id="bioasq_001",
        query="Does metformin help diabetes?",
        top_k=50,
        inject_gt=False,
        gt_pmids=["12345"],  # For caching only
    )

    # Close when done
    await retriever.close()

    # Or use context manager
    async with CachedHybridRetriever(strategy="C2") as retriever:
        pmids = await retriever.retrieve(...)
"""

from pathlib import Path
from typing import Sequence

from .hybrid_retriever import HybridRetriever, get_config_by_strategy, STRATEGY_PRESETS
from .models import RetrievedAbstract, RetrievalResult
from cache.retrieval_cache import RetrievalCache, extract_pmids_from_metadata


class CachedHybridRetriever:
    """HybridRetriever with automatic caching.

    Wraps the HybridRetriever to provide transparent caching:
    1. On retrieve(), check cache first for (question_id, strategy)
    2. If cache hit: return cached PMIDs (with optional GT injection)
    3. If cache miss: run actual retrieval, optionally cache results

    Args:
        strategy: Retrieval strategy (A1-A6, B1-B6, C1-C2)
        cache_dir: Cache directory path (default: .cache/retrieval)
        auto_cache: If True, cache results on miss (default: True)

    Example:
        retriever = CachedHybridRetriever(strategy="B6")
        await retriever.initialize()

        # First call: cache miss, runs retrieval, caches result
        pmids = await retriever.retrieve("bioasq_001", "Does aspirin help?", gt_pmids=["123"])

        # Second call: cache hit, returns instantly
        pmids = await retriever.retrieve("bioasq_001", "Does aspirin help?")

        await retriever.close()
    """

    def __init__(
        self,
        strategy: str = "B6",
        cache_dir: str | Path = ".cache/retrieval",
        auto_cache: bool = True,
    ):
        self._strategy = strategy.upper()
        self._config = get_config_by_strategy(self._strategy)
        self._retriever = HybridRetriever(self._config)
        self._cache = RetrievalCache(cache_dir=cache_dir)
        self._auto_cache = auto_cache
        self._initialized = False

    @property
    def strategy(self) -> str:
        """Get current strategy name."""
        return self._strategy

    @property
    def cache(self) -> RetrievalCache:
        """Get cache instance for direct access if needed."""
        return self._cache

    async def initialize(self) -> None:
        """Initialize the underlying retriever."""
        if not self._initialized:
            await self._retriever.initialize()
            self._initialized = True

    async def close(self) -> None:
        """Close the underlying retriever."""
        if self._initialized:
            await self._retriever.close()
            self._initialized = False

    async def __aenter__(self) -> "CachedHybridRetriever":
        await self.initialize()
        return self

    async def __aexit__(self, *args) -> None:
        await self.close()

    async def retrieve(
        self,
        question_id: str,
        query: str,
        top_k: int = 50,
        inject_gt: bool = False,
        gt_pmids: Sequence[str] | None = None,
    ) -> list[str]:
        """Retrieve PMIDs with automatic caching.

        1. Check cache for (question_id, strategy)
        2. If hit: return cached PMIDs (with optional GT injection)
        3. If miss: run retrieval, cache results (if auto_cache), return PMIDs

        Args:
            question_id: Question identifier (used as cache key)
            query: Natural language query
            top_k: Number of PMIDs to return
            inject_gt: If True, inject GT PMIDs at top (oracle mode)
            gt_pmids: Ground truth PMIDs for caching (not used if cache hit)

        Returns:
            List of PMIDs (length = top_k or less)
        """
        # Try cache first
        cached = await self._cache.load(
            question_id,
            strategy=self._strategy,
            top_k=top_k,
            inject_gt=inject_gt,
        )
        if cached:
            return cached

        # Cache miss - ensure initialized
        if not self._initialized:
            await self.initialize()

        # Run actual retrieval
        result = await self._retriever.retrieve(
            query,
            gt_pmids=list(gt_pmids) if gt_pmids else None,
        )

        # Cache results if auto_cache enabled
        if self._auto_cache:
            await self._cache.set(
                question_id=question_id,
                strategy=self._strategy,
                results=[
                    {
                        "pmid": a.pmid,
                        "score": a.score,
                        "rank": a.rank,
                        "has_pmc": a.have_fulltext,
                        "paper_uuid": a.paper_uuid,
                    }
                    for a in result.abstracts
                ],
                gt_pmids=list(gt_pmids) if gt_pmids else None,
            )

        # Return PMIDs (applying inject_gt if needed)
        if not inject_gt:
            return [a.pmid for a in result.abstracts[:top_k]]
        else:
            # Apply GT injection to fresh results
            gt_set = set(gt_pmids) if gt_pmids else set()
            result_pmids = [a.pmid for a in result.abstracts]
            non_gt_pmids = [p for p in result_pmids if p not in gt_set]

            gt_list = list(gt_pmids) if gt_pmids else []
            remaining_slots = max(0, top_k - len(gt_list))
            combined = gt_list + non_gt_pmids[:remaining_slots]

            return combined[:top_k]

    async def retrieve_full(
        self,
        question_id: str,
        query: str,
        top_k: int = 50,
        inject_gt: bool = False,
        gt_pmids: Sequence[str] | None = None,
    ) -> RetrievalResult:
        """Retrieve with full result (including abstracts, timing info).

        Unlike retrieve(), this returns the full RetrievalResult object
        instead of just PMIDs. Does not use cache for return value
        (but still caches PMIDs for future calls).

        Args:
            question_id: Question identifier
            query: Natural language query
            top_k: Number of results to return
            inject_gt: If True, inject GT PMIDs at top
            gt_pmids: Ground truth PMIDs

        Returns:
            Full RetrievalResult with abstracts, timing info, etc.
        """
        # Ensure initialized
        if not self._initialized:
            await self.initialize()

        # Run actual retrieval
        result = await self._retriever.retrieve(
            query,
            gt_pmids=list(gt_pmids) if gt_pmids else None,
        )

        # Cache results if auto_cache enabled
        if self._auto_cache:
            await self._cache.set(
                question_id=question_id,
                strategy=self._strategy,
                results=[
                    {
                        "pmid": a.pmid,
                        "score": a.score,
                        "rank": a.rank,
                        "has_pmc": a.have_fulltext,
                        "paper_uuid": a.paper_uuid,
                    }
                    for a in result.abstracts
                ],
                gt_pmids=list(gt_pmids) if gt_pmids else None,
            )

        return result

    async def is_cached(self, question_id: str) -> bool:
        """Check if a question is cached for current strategy."""
        return await self._cache.exists(question_id, self._strategy)

    def get_cache_stats(self) -> dict:
        """Get cache statistics for current strategy."""
        return self._cache.get_stats(self._strategy)


def list_available_strategies() -> list[str]:
    """List all available retrieval strategies."""
    return sorted(STRATEGY_PRESETS.keys())


def get_strategy_description(strategy: str) -> str:
    """Get a short description of a strategy."""
    descriptions = {
        "A1": "MeSH Filter Only",
        "A2": "MeSH Filter + Rerank",
        "A3": "Keyword Filter Only",
        "A4": "Keyword Filter + Rerank",
        "A5": "MeSH + Keyword RRF",
        "A6": "MeSH + Keyword RRF + Rerank",
        "B1": "MeSH + Vector",
        "B2": "MeSH + Vector + Rerank",
        "B3": "Keyword + Vector",
        "B4": "Keyword + Vector + Rerank",
        "B5": "MeSH + Keyword + Vector",
        "B6": "MeSH + Keyword + Vector + Rerank (DEFAULT)",
        "C1": "Dense Only",
        "C2": "Dense + Rerank (FAST)",
    }
    return descriptions.get(strategy.upper(), "Unknown strategy")
