"""Keyword-based PMID filtering using PostgreSQL full-text search.

This module provides keyword-based filtering via:
1. Parse extracted keywords from query parser
2. Build tsquery for PostgreSQL FTS
3. Query articles table using indexed tsvector
"""

import time
from typing import Sequence

import asyncpg


class KeywordFilter:
    """Keyword-based filtering using PostgreSQL GIN-indexed FTS.

    Uses the existing index:
    idx_articles_fulltext ON articles USING gin(
        to_tsvector('english', coalesce(title,'') || ' ' || coalesce(abstract,''))
    )

    Example:
        >>> kw_filter = KeywordFilter(pool)
        >>> pmids = await kw_filter.get_pmids_by_keywords(
        ...     ['metformin', 'diabetes', 'type 2'],
        ...     limit=5000
        ... )
        >>> print(len(pmids))
        4523
    """

    def __init__(self, db_pool: asyncpg.Pool):
        """Initialize the keyword filter.

        Args:
            db_pool: PostgreSQL connection pool for paper-graph-pubmed
        """
        self._pool = db_pool

    async def get_pmids_by_keywords(
        self,
        keywords: Sequence[str],
        limit: int = 10000,
        match_any: bool = True,
        return_ordered: bool = False,
    ) -> tuple[set[str] | list[str], float]:
        """Get PMIDs matching keywords via full-text search.

        Args:
            keywords: List of keywords to search for
            limit: Maximum number of PMIDs to return
            match_any: If True, match any keyword (OR). If False, match all (AND).
            return_ordered: If True, return list instead of set (order is arbitrary from DB)

        Returns:
            Tuple of (set or list of PMID strings, time_ms)
        """
        if not keywords:
            return [] if return_ordered else set(), 0.0

        start_time = time.perf_counter()

        # Build tsquery from keywords
        # Sanitize keywords: remove special chars, keep alphanumeric and spaces
        sanitized = []
        for kw in keywords:
            # Handle multi-word phrases: "type 2 diabetes" -> "type <-> 2 <-> diabetes"
            words = [w.strip() for w in kw.split() if w.strip()]
            if not words:
                continue
            if len(words) == 1:
                # Single word: just the term
                sanitized.append(self._escape_tsquery_term(words[0]))
            else:
                # Multi-word: use phrase matching (<->)
                phrase = " <-> ".join(self._escape_tsquery_term(w) for w in words)
                sanitized.append(f"({phrase})")

        if not sanitized:
            empty = [] if return_ordered else set()
            return empty, (time.perf_counter() - start_time) * 1000

        # Combine with OR or AND
        operator = " | " if match_any else " & "
        tsquery_str = operator.join(sanitized)

        # Execute query using the GIN index
        # Note: We skip ts_rank ordering as it's extremely slow (70s+ on 600K+ rows)
        # The downstream vector search will reorder by semantic relevance anyway
        query = """
            SELECT pmid::text
            FROM articles
            WHERE to_tsvector('english', coalesce(title,'') || ' ' || coalesce(abstract,''))
                  @@ to_tsquery('english', $1)
            LIMIT $2
        """

        try:
            async with self._pool.acquire() as conn:
                # Force index usage - PostgreSQL planner often incorrectly chooses SeqScan
                # for GIN indexes on large tables, but index scan is actually 4-5x faster
                await conn.execute("SET LOCAL enable_seqscan = off")
                rows = await conn.fetch(query, tsquery_str, limit)
            if return_ordered:
                pmids = [row["pmid"] for row in rows]
            else:
                pmids = {row["pmid"] for row in rows}
        except Exception as e:
            # Fallback: try simpler query with just first few keywords
            print(f"[KeywordFilter] FTS query failed: {e}, trying fallback")
            fallback_pmids = await self._fallback_search(keywords[:3], limit)
            pmids = list(fallback_pmids) if return_ordered else fallback_pmids

        time_ms = (time.perf_counter() - start_time) * 1000
        return pmids, time_ms

    async def _fallback_search(
        self,
        keywords: Sequence[str],
        limit: int,
    ) -> set[str]:
        """Simple ILIKE fallback when tsquery fails.

        Args:
            keywords: Keywords to search for
            limit: Maximum results

        Returns:
            Set of PMID strings
        """
        if not keywords:
            return set()

        # Build simple ILIKE conditions
        conditions = []
        params = []
        for i, kw in enumerate(keywords):
            clean_kw = "".join(c for c in kw if c.isalnum() or c.isspace())
            if clean_kw:
                conditions.append(
                    f"(title ILIKE ${i+1} OR abstract ILIKE ${i+1})"
                )
                params.append(f"%{clean_kw}%")

        if not conditions:
            return set()

        query = f"""
            SELECT pmid::text
            FROM articles
            WHERE {" OR ".join(conditions)}
            LIMIT {limit}
        """

        async with self._pool.acquire() as conn:
            rows = await conn.fetch(query, *params)

        return {row["pmid"] for row in rows}

    def _escape_tsquery_term(self, term: str) -> str:
        """Escape a term for use in tsquery.

        Args:
            term: Raw search term

        Returns:
            Escaped term safe for tsquery
        """
        # Keep only alphanumeric characters
        clean = "".join(c for c in term if c.isalnum())
        if not clean:
            return ""
        return clean


async def test_keyword_filter():
    """Test the keyword filter with sample queries."""
    import asyncpg

    pool = await asyncpg.create_pool(
        __import__("config").get_config().postgres.pubmed_url,
        min_size=2,
        max_size=5,
    )

    try:
        kw_filter = KeywordFilter(pool)

        test_cases = [
            (["metformin", "type 2 diabetes"], "Metformin + T2D"),
            (["FGFR2", "methylation", "birth weight"], "FGFR2 methylation"),
            (["innate lymphoid cells", "rhinosinusitis"], "ILC2 + rhinosinusitis"),
            (["vagus nerve", "steatohepatitis", "PEMT"], "Vagus + PEMT"),
        ]

        print("=" * 60)
        print("Keyword Filter Test")
        print("=" * 60)

        for keywords, desc in test_cases:
            print(f"\n[{desc}] Keywords: {keywords}")

            # Test match_any (OR)
            pmids_or, time_or = await kw_filter.get_pmids_by_keywords(
                keywords, limit=10000, match_any=True
            )
            print(f"  OR mode: {len(pmids_or):,} PMIDs ({time_or:.0f}ms)")

            # Test match_all (AND)
            pmids_and, time_and = await kw_filter.get_pmids_by_keywords(
                keywords, limit=10000, match_any=False
            )
            print(f"  AND mode: {len(pmids_and):,} PMIDs ({time_and:.0f}ms)")

    finally:
        await pool.close()


if __name__ == "__main__":
    import asyncio
    asyncio.run(test_keyword_filter())
