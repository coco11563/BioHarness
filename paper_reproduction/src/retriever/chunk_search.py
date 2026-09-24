"""Vector-based semantic search in chunks collection (129M+ chunks).

This module provides vector similarity search for PMC full-text chunks:
1. Generate query embedding
2. Search in chunks collection (129M vectors)
3. Optional PMID filtering to narrow search scope
4. Fetch text content from PostgreSQL
5. Return top-k chunks with metadata
"""

import asyncio
import time
from typing import Sequence

import asyncpg
from qdrant_client import AsyncQdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchAny, SearchParams

from config import get_config
from .models import RetrievedChunk
from utils.clients import embed_client


_QDRANT_GATE = asyncio.Semaphore(3)


class ChunkSearch:
    """Vector-based semantic search in chunks collection.

    Uses the global embed_client singleton for load-balanced embedding calls.
    Supports optional PMID filtering to search only within specific papers.

    Example:
        >>> cs = ChunkSearch(qdrant, db_pool)
        >>> chunks = await cs.search(
        ...     query="metformin mechanism of action",
        ...     pmid_filter={"12345678", "23456789"},
        ...     limit=50
        ... )
    """

    def __init__(
        self,
        qdrant_client: AsyncQdrantClient,
        db_pool: asyncpg.Pool | None = None,
        collection: str = "chunks",
    ):
        """Initialize chunk search.

        Args:
            qdrant_client: Async Qdrant client
            db_pool: AsyncPG connection pool for papergraph database
            collection: Qdrant collection name for chunk vectors
        """
        self._qdrant = qdrant_client
        self._db_pool = db_pool
        self._collection = collection
        self._config = get_config()

    async def _ensure_db_pool(self) -> asyncpg.Pool:
        """Ensure database pool is initialized."""
        if self._db_pool is None:
            # asyncpg uses plain postgresql:// not postgresql+asyncpg://
            db_url = self._config.postgres.papergraph_url
            self._db_pool = await asyncpg.create_pool(
                db_url,
                min_size=2,
                max_size=10,
            )
        return self._db_pool

    async def search(
        self,
        query: str,
        pmid_filter: set[str] | None = None,
        limit: int = 50,
    ) -> tuple[list[RetrievedChunk], float]:
        """Search for similar chunks with optional PMID filtering.

        Args:
            query: Natural language query
            pmid_filter: Optional set of PMIDs to restrict search to
            limit: Maximum number of results

        Returns:
            Tuple of (list of RetrievedChunk, time_ms)
        """
        start_time = time.perf_counter()

        # 1. Generate query embedding using global singleton
        query_vector = await embed_client.embed_single(query)

        # 2. Build filter if PMIDs provided
        qdrant_filter = None
        if pmid_filter:
            # PMIDs are stored as strings in chunks collection
            pmid_strs = [str(p) for p in pmid_filter]
            if pmid_strs:
                qdrant_filter = Filter(
                    must=[
                        FieldCondition(
                            key="pmid",
                            match=MatchAny(any=pmid_strs),
                        )
                    ]
                )

        # 3. Search in chunks collection.
        # The caller's item-level concurrency is multiplied by the number of
        # retrievals each item issues (dual retrieval fires three), so a nominal
        # concurrency of 3 put 9 searches on this 128M-point on-disk collection.
        # Measured: 3 concurrent queries -> 3.9 s median, 6 -> 85.1 s, which blows
        # past the server's timeout and returns 408. Cap the real concurrency here.
        # The box is shared and its load swings widely, so a search occasionally
        # comes back 408 or the SSH tunnel blips mid-request. Without a retry a
        # single transient failure discarded the whole benchmark item, which is
        # what emptied three runs; embed_client already retries the same way.
        search_response = None
        last_err: Exception | None = None
        for attempt in range(4):
            try:
                async with _QDRANT_GATE:
                    search_response = await self._qdrant.query_points(
                        collection_name=self._collection,
                        query=query_vector,
                        query_filter=qdrant_filter,
                        limit=limit,
                        with_payload=True,
                        search_params=SearchParams(hnsw_ef=128),
                        # A cold query takes ~15 s; the server's own 60 s default
                        # was returning 408 under concurrent load.
                        timeout=240,
                    )
                break
            except Exception as e:
                last_err = e
                if attempt == 3:
                    raise
                await asyncio.sleep(min(30.0, 3.0 * (2 ** attempt)))
        if search_response is None:  # pragma: no cover - defensive
            raise RuntimeError(f"chunk search failed: {last_err}")

        if not search_response.points:
            return [], (time.perf_counter() - start_time) * 1000

        # 4. Get chunk texts from PostgreSQL
        chunk_ids = [
            p.payload.get("chunk_id")
            for p in search_response.points
            if p.payload.get("chunk_id")
        ]

        # Short-circuit if no valid chunk_ids (avoid invalid SQL)
        if not chunk_ids:
            return [], (time.perf_counter() - start_time) * 1000

        pool = await self._ensure_db_pool()
        async with pool.acquire() as conn:
            placeholders = ", ".join(f"${i+1}" for i in range(len(chunk_ids)))
            rows = await conn.fetch(f"""
                SELECT
                    id::text as chunk_id,
                    text_content,
                    section_id::text,
                    sequence_order
                FROM chunks
                WHERE id IN ({placeholders})
            """, *chunk_ids)

        # Build text lookup
        text_map = {row["chunk_id"]: row for row in rows}

        # 5. Build RetrievedChunk list
        chunks = []
        for rank, point in enumerate(search_response.points, start=1):
            payload = point.payload
            chunk_id = payload.get("chunk_id")
            text_data = text_map.get(chunk_id, {})

            chunks.append(
                RetrievedChunk(
                    chunk_id=chunk_id or "",
                    pmid=str(payload.get("pmid", "")),
                    pmcid=str(payload.get("pmcid", "")),
                    text=text_data.get("text_content", ""),
                    section_id=text_data.get("section_id") or payload.get("section_id", ""),
                    score=point.score,
                    rank=rank,
                    token_count=payload.get("token_count", 0),
                    sequence_order=text_data.get("sequence_order", 0),
                )
            )

        time_ms = (time.perf_counter() - start_time) * 1000
        return chunks, time_ms

    async def search_by_pmids(
        self,
        query: str,
        pmids: Sequence[str],
        limit: int = 50,
    ) -> tuple[list[RetrievedChunk], float]:
        """Search and rank chunks from specific PMIDs by similarity to query.

        Use this when you have a set of papers and want to find relevant chunks.

        Args:
            query: Natural language query
            pmids: List of PMIDs to search within
            limit: Maximum number of results

        Returns:
            Tuple of (list of RetrievedChunk sorted by score, time_ms)
        """
        return await self.search(query, set(pmids), limit)

    async def close(self):
        """Close database connections."""
        if self._db_pool:
            await self._db_pool.close()
            self._db_pool = None


async def test_chunk_search():
    """Test chunk search with sample queries."""
    from qdrant_client import AsyncQdrantClient

    cfg = get_config()

    # Initialize clients
    qdrant = AsyncQdrantClient(
        cfg.qdrant.url,
        check_compatibility=False,
        timeout=60,
    )

    cs = ChunkSearch(qdrant)

    try:
        test_queries = [
            "metformin mechanism of action in diabetes",
            "BRCA1 breast cancer treatment",
            "COVID-19 vaccine efficacy",
        ]

        print("=" * 60)
        print("Chunk Search Test (129M chunks)")
        print("=" * 60)

        for i, query in enumerate(test_queries, 1):
            print(f"\n[Query {i}] {query[:55]}...")

            # Test without filter
            chunks, time_ms = await cs.search(query, limit=5)
            print(f"  Results: {len(chunks)} chunks ({time_ms:.0f}ms)")

            for c in chunks[:3]:
                text_preview = c.text[:80].replace("\n", " ") + "..." if c.text else "N/A"
                print(f"    #{c.rank} [PMID:{c.pmid}] score={c.score:.3f}")
                print(f"        {text_preview}")

    finally:
        await cs.close()
        await qdrant.close()


if __name__ == "__main__":
    import asyncio
    asyncio.run(test_chunk_search())
