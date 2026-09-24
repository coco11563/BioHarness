"""MeSH-based PMID filtering using vector search.

This module provides MeSH term matching via:
1. Per-entity vector search in mesh-term-only collection (30,956 terms)
2. Score-based filtering (threshold >= 0.75)
3. PostgreSQL query to get PMIDs via indexed descriptor_ui

Experimental Results (mesh-ablation-2025-12-21):
- mesh-term-only + all expansion: 95.4% overlap rate
- Name-only embedding contributes 66.5% of performance
"""

import re
import time

import asyncpg
from qdrant_client import AsyncQdrantClient
from qdrant_client.models import SearchParams

from .models import MeSHMatch
from utils.clients import embed_client


class MeSHFilter:
    """MeSH-based filtering using per-entity vector search.

    Uses the global embed_client singleton for load-balanced embedding calls.
    Uses descriptor_ui for PostgreSQL queries (indexed!) instead of
    descriptor_name (not indexed).

    Example:
        >>> mesh_filter = MeSHFilter(qdrant, pool)
        >>> matches = await mesh_filter.match_mesh_terms(query)
        >>> pmids = await mesh_filter.get_pmids_by_mesh(matches, limit=5000)
    """

    def __init__(
        self,
        qdrant_client: AsyncQdrantClient,
        db_pool: asyncpg.Pool,
        mesh_collection: str = "mesh-term-only",
        top_k: int = 20,
        llm_select: int = 8,  # Legacy name, now just max_terms
    ):
        """Initialize the MeSH filter.

        Args:
            qdrant_client: Async Qdrant client
            db_pool: PostgreSQL connection pool for paper-graph-pubmed
            mesh_collection: Qdrant collection name for MeSH terms
            top_k: Number of MeSH candidates from vector search per entity
            llm_select: Max MeSH terms to return (legacy name)
        """
        self._qdrant = qdrant_client
        self._pool = db_pool
        self._collection = mesh_collection
        self._top_k = top_k
        self._llm_select = llm_select

    async def match_mesh_terms(
        self,
        query: str,
        entities: list[str] | None = None,
    ) -> tuple[list[MeSHMatch], float]:
        """Match query to MeSH terms via per-entity vector search.

        Searches each entity separately for better matching, since individual
        terms like "eosinophilia" match MeSH terms much better than full queries.

        Args:
            query: Natural language biomedical query (used if entities not provided)
            entities: Optional list of entities from query parser

        Returns:
            Tuple of (list of MeSHMatch, time_ms)
        """
        start_time = time.perf_counter()

        # Determine search terms: use entities if provided, else extract from query
        search_terms = entities if entities else self._extract_search_terms(query)
        if not search_terms:
            return [], (time.perf_counter() - start_time) * 1000

        # 1. Embed all search terms in batch
        embeddings = await embed_client.embed(search_terms)

        # 2. Search for each term and collect unique candidates
        seen_uis: set[str] = set()
        all_candidates: list[tuple[float, dict]] = []  # (score, payload)

        for i, term_embedding in enumerate(embeddings):
            search_response = await self._qdrant.query_points(
                collection_name=self._collection,
                query=term_embedding,
                limit=5,  # Top 5 per term
                with_payload=True,
                search_params=SearchParams(hnsw_ef=128),
            )

            for p in search_response.points:
                ui = p.payload["descriptor_ui"]
                # Only add if score >= 0.75 and not seen
                if p.score >= 0.75 and ui not in seen_uis:
                    seen_uis.add(ui)
                    all_candidates.append((p.score, p.payload))

        if not all_candidates:
            # Fallback: just use full query
            return await self._match_mesh_terms_fullquery(query, start_time)

        # 3. Sort by score and take top matches
        all_candidates.sort(key=lambda x: x[0], reverse=True)

        matches = []
        for score, payload in all_candidates[: self._llm_select]:
            matches.append(
                MeSHMatch(
                    descriptor_ui=payload["descriptor_ui"],
                    descriptor_name=payload["descriptor_name"],
                    score=score,
                    tree_numbers=payload.get("tree_numbers", []),
                )
            )

        time_ms = (time.perf_counter() - start_time) * 1000
        return matches, time_ms

    async def _match_mesh_terms_fullquery(
        self,
        query: str,
        start_time: float,
    ) -> tuple[list[MeSHMatch], float]:
        """Fallback: match using full query embedding."""
        query_vector = await embed_client.embed_single(query)

        search_response = await self._qdrant.query_points(
            collection_name=self._collection,
            query=query_vector,
            limit=self._top_k,
            with_payload=True,
            search_params=SearchParams(hnsw_ef=128),
        )

        matches = []
        for p in search_response.points[: self._llm_select]:
            matches.append(
                MeSHMatch(
                    descriptor_ui=p.payload["descriptor_ui"],
                    descriptor_name=p.payload["descriptor_name"],
                    score=p.score,
                    tree_numbers=p.payload.get("tree_numbers", []),
                )
            )

        time_ms = (time.perf_counter() - start_time) * 1000
        return matches, time_ms

    def _extract_search_terms(self, query: str) -> list[str]:
        """Extract simple search terms from query when entities not provided."""
        import re
        # Remove common question words and split
        stopwords = {'are', 'is', 'do', 'does', 'the', 'a', 'an', 'in', 'of', 'with', 'or', 'and', 'to', 'for', 'by', 'on', 'at', 'from', 'as'}
        words = re.findall(r'\b\w+\b', query.lower())
        terms = [w for w in words if len(w) > 2 and w not in stopwords]
        return terms[:10]  # Limit to prevent too many searches

    async def get_pmids_by_mesh(
        self,
        matches: list[MeSHMatch],
        limit: int = 10000,
        max_term_papers: int = 500000,
        return_ordered: bool = False,
    ) -> tuple[set[str] | list[str], float]:
        """Get PMIDs that have any of the given MeSH terms.

        Uses descriptor_ui for queries (indexed in mesh_headings table).
        Papers are ranked by how many of the MeSH terms they match.
        Overly broad terms (>max_term_papers) are excluded.

        Args:
            matches: List of MeSHMatch with descriptor_ui
            limit: Maximum number of PMIDs to return
            max_term_papers: Exclude MeSH terms with more papers than this
            return_ordered: If True, return ordered list instead of set (preserves DB ranking)

        Returns:
            Tuple of (set or list of PMID strings, time_ms)
        """
        if not matches:
            return [] if return_ordered else set(), 0.0

        start_time = time.perf_counter()

        descriptor_uis = [m.descriptor_ui for m in matches]
        placeholders = ", ".join(f"${i+1}" for i in range(len(descriptor_uis)))

        # Query that returns PMIDs matching ANY of the MeSH terms
        # Ranked by number of matching terms (papers matching more terms first)
        # This prioritizes papers more relevant to the query
        query = f"""
            WITH term_counts AS (
                SELECT descriptor_ui, COUNT(DISTINCT pmid) as cnt
                FROM mesh_headings
                WHERE descriptor_ui IN ({placeholders})
                GROUP BY descriptor_ui
                HAVING COUNT(DISTINCT pmid) <= {max_term_papers}
            ),
            paper_scores AS (
                SELECT mh.pmid, COUNT(DISTINCT mh.descriptor_ui) as match_count
                FROM mesh_headings mh
                JOIN term_counts tc ON mh.descriptor_ui = tc.descriptor_ui
                GROUP BY mh.pmid
            )
            SELECT pmid::text
            FROM paper_scores
            ORDER BY match_count DESC, pmid DESC
            LIMIT {limit}
        """

        async with self._pool.acquire() as conn:
            rows = await conn.fetch(query, *descriptor_uis)

        # Return ordered list or set based on parameter
        if return_ordered:
            pmids = [row["pmid"] for row in rows]
        else:
            pmids = {row["pmid"] for row in rows}
        time_ms = (time.perf_counter() - start_time) * 1000

        return pmids, time_ms


async def test_mesh_filter():
    """Test the MeSH filter with sample queries."""
    import asyncpg
    from qdrant_client import AsyncQdrantClient

    # Initialize clients
    qdrant = AsyncQdrantClient(__import__("config").get_config().qdrant.url, check_compatibility=False)
    pool = await asyncpg.create_pool(
        __import__("config").get_config().postgres.pubmed_url,
        min_size=2,
        max_size=5,
    )

    try:
        mesh_filter = MeSHFilter(qdrant, pool)

        test_queries = [
            "Does metformin help with type 2 diabetes?",
            "Are group 2 innate lymphoid cells increased in chronic rhinosinusitis?",
            "Is methylation of the FGFR2 gene associated with high birth weight?",
        ]

        print("=" * 60)
        print("MeSH Filter Test")
        print("=" * 60)

        for i, query in enumerate(test_queries, 1):
            print(f"\n[Query {i}] {query[:60]}...")

            # Match MeSH terms
            matches, match_time = await mesh_filter.match_mesh_terms(query)
            print(f"  Matched MeSH ({len(matches)} terms, {match_time:.0f}ms):")
            for m in matches[:5]:
                print(f"    - {m.descriptor_name} ({m.descriptor_ui}, score={m.score:.3f})")

            # Get PMIDs
            pmids, pmid_time = await mesh_filter.get_pmids_by_mesh(matches)
            print(f"  PMIDs: {len(pmids):,} ({pmid_time:.0f}ms)")
            print(f"  Total: {match_time + pmid_time:.0f}ms")

    finally:
        await pool.close()
        await qdrant.close()


if __name__ == "__main__":
    import asyncio
    asyncio.run(test_mesh_filter())
