"""Thin async Qdrant client over httpx (avoids the qdrant-client lib dep
when we only need search)."""

from __future__ import annotations

from typing import Any

import httpx


class QdrantClient:
    def __init__(self, base_url: str, *, timeout: float = 60.0) -> None:
        self._base_url = base_url.rstrip("/")
        self._client = httpx.AsyncClient(timeout=timeout)

    async def aclose(self) -> None:
        await self._client.aclose()

    async def search(
        self, collection: str, vector: list[float], *, top_k: int = 50,
    ) -> list[dict[str, Any]]:
        url = f"{self._base_url}/collections/{collection}/points/search"
        payload = {"vector": vector, "limit": top_k, "with_payload": True}
        r = await self._client.post(url, json=payload)
        r.raise_for_status()
        out = []
        for hit in r.json().get("result", []):
            payload = hit.get("payload") or {}
            out.append(
                {
                    "id": hit.get("id"),
                    "score": hit.get("score"),
                    "text": payload.get("text") or payload.get("abstract") or "",
                    "metadata": payload,
                }
            )
        return out
