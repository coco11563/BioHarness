"""Vector-based semantic search in paper-full collection.

This module provides vector similarity search for PubMed abstracts:
1. Generate query embedding
2. Search in paper-full collection (38M+ vectors)
3. Optional PMID filtering to narrow search scope
4. Return top-k abstracts with metadata
"""

import time
from typing import Sequence

from qdrant_client import AsyncQdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchAny, SearchParams

from .models import RetrievedAbstract
from utils.clients import embed_client


class VectorSearch:
    """Vector-based semantic search in paper-full collection.

    Uses the global embed_client singleton for load-balanced embedding calls.
    Supports optional PMID filtering to search only within
    pre-filtered candidates from MeSH/keyword filters.

    Example:
        >>> vs = VectorSearch(qdrant)
        >>> abstracts = await vs.search(
        ...     query="Does metformin help with diabetes?",
        ...     pmid_filter=filtered_pmids,
        ...     limit=50
        ... )
    """

    def __init__(
        self,
        qdrant_client: AsyncQdrantClient,
        collection: str = "paper-full",
    ):
        """Initialize vector search.

        Args:
            qdrant_client: Async Qdrant client
            collection: Qdrant collection name for paper vectors
        """
        self._qdrant = qdrant_client
        self._collection = collection

    async def search(
        self,
        query: str,
        pmid_filter: set[str] | None = None,
        limit: int = 50,
    ) -> tuple[list[RetrievedAbstract], float]:
        """Search for similar abstracts with optional PMID filtering.

        Args:
            query: Natural language query
            pmid_filter: Optional set of PMIDs to restrict search to
            limit: Maximum number of results

        Returns:
            Tuple of (list of RetrievedAbstract, time_ms)
        """
        start_time = time.perf_counter()

        # 1. Generate query embedding using global singleton
        query_vector = await embed_client.embed_single(query)

        # 2. Build filter if PMIDs provided
        qdrant_filter = None
        if pmid_filter:
            # PMIDs are stored as strings in the collection
            pmid_strs = list(pmid_filter)
            if pmid_strs:
                qdrant_filter = Filter(
                    must=[
                        FieldCondition(
                            key="pmid",
                            match=MatchAny(any=pmid_strs),
                        )
                    ]
                )

        # 3. Search in paper-full collection
        search_response = await self._qdrant.query_points(
            collection_name=self._collection,
            query=query_vector,
            query_filter=qdrant_filter,
            limit=limit,
            with_payload=True,
            search_params=SearchParams(hnsw_ef=128),
        )

        # 4. Build RetrievedAbstract list
        abstracts = []
        for rank, point in enumerate(search_response.points, start=1):
            payload = point.payload
            abstracts.append(
                RetrievedAbstract(
                    pmid=str(payload.get("pmid", "")),
                    title=payload.get("title", ""),
                    abstract=payload.get("abstract", ""),
                    score=point.score,
                    rank=rank,
                    have_fulltext=bool(payload.get("pmc")),
                    paper_uuid=None,  # Will be populated later if needed
                    pmc=payload.get("pmc"),
                    matched_mesh=[],  # Will be populated by orchestrator
                )
            )

        time_ms = (time.perf_counter() - start_time) * 1000
        return abstracts, time_ms

    async def search_by_pmids(
        self,
        query: str,
        pmids: Sequence[str],
        limit: int = 50,
    ) -> tuple[list[RetrievedAbstract], float]:
        """Search and rank specific PMIDs by similarity to query.

        Use this when you have a fixed set of PMIDs and want to rank them.

        Args:
            query: Natural language query
            pmids: List of PMIDs to rank
            limit: Maximum number of results

        Returns:
            Tuple of (list of RetrievedAbstract sorted by score, time_ms)
        """
        return await self.search(query, set(pmids), limit)


async def test_vector_search():
    """Test vector search with sample queries."""
    from qdrant_client import AsyncQdrantClient

    # Initialize clients with increased timeout for large collection
    qdrant = AsyncQdrantClient(
        __import__("config").get_config().qdrant.url,
        check_compatibility=False,
        timeout=60,  # 60 second timeout for 38M+ vectors
    )

    try:
        vs = VectorSearch(qdrant)

        test_queries = [
            "Does metformin help with type 2 diabetes?",
            "Are innate lymphoid cells involved in chronic rhinosinusitis?",
            "Is FGFR2 methylation associated with birth weight?",
        ]

        print("=" * 60)
        print("Vector Search Test")
        print("=" * 60)

        for i, query in enumerate(test_queries, 1):
            print(f"\n[Query {i}] {query[:55]}...")

            # Test without filter
            abstracts, time_ms = await vs.search(query, limit=5)
            print(f"  No filter: {len(abstracts)} results ({time_ms:.0f}ms)")
            for a in abstracts[:3]:
                print(f"    #{a.rank} [{a.pmid}] {a.title[:50]}... (score={a.score:.3f})")

            # Test with filter (use returned PMIDs as filter)
            filter_pmids = {a.pmid for a in abstracts}
            abstracts_f, time_f = await vs.search(query, filter_pmids, limit=5)
            print(f"  With filter ({len(filter_pmids)} PMIDs): {len(abstracts_f)} results ({time_f:.0f}ms)")

    finally:
        await qdrant.close()


if __name__ == "__main__":
    import asyncio
    asyncio.run(test_vector_search())
