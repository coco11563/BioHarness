"""OpenAI-compatible embedding client (1024-dim by default)."""

from __future__ import annotations

import httpx


class EmbedClient:
    def __init__(self, base_url: str, api_key: str, *, timeout: float = 60.0) -> None:
        self._base_url = base_url.rstrip("/")
        self._headers = {"Authorization": f"Bearer {api_key}"}
        self._client = httpx.AsyncClient(timeout=timeout)

    async def aclose(self) -> None:
        await self._client.aclose()

    async def embed(self, texts: list[str], *, model: str = "embedding") -> list[list[float]]:
        if not texts:
            return []
        r = await self._client.post(
            f"{self._base_url}/embeddings",
            json={"model": model, "input": texts},
            headers=self._headers,
        )
        r.raise_for_status()
        return [item["embedding"] for item in r.json()["data"]]

    async def embed_one(self, text: str) -> list[float]:
        out = await self.embed([text])
        return out[0]
