"""Minimal vector search wrapper for benchmark-safe RT-KG runs."""

import time
from typing import Sequence

from qdrant_client import AsyncQdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchAny, SearchParams

from .models import RetrievedAbstract
from utils.clients import embed_client


class VectorSearch:
    """Vector-based semantic search in paper-full collection."""

    def __init__(
        self,
        qdrant_client: AsyncQdrantClient,
        collection: str = "paper-full",
    ):
        self._qdrant = qdrant_client
        self._collection = collection

    async def search(
        self,
        query: str,
        pmid_filter: set[str] | None = None,
        limit: int = 50,
    ) -> tuple[list[RetrievedAbstract], float]:
        start_time = time.perf_counter()
        query_vector = await embed_client.embed_single(query)

        qdrant_filter = None
        if pmid_filter:
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

        search_response = await self._qdrant.query_points(
            collection_name=self._collection,
            query=query_vector,
            query_filter=qdrant_filter,
            limit=limit,
            with_payload=True,
            search_params=SearchParams(hnsw_ef=128),
        )

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
                    paper_uuid=None,
                    pmc=payload.get("pmc"),
                    matched_mesh=[],
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
        return await self.search(query, set(pmids), limit)
