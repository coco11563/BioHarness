"""OpenAI-compatible embedding client (1024-dim by default)."""

from __future__ import annotations

import httpx


class EmbedClient:
    def __init__(self, base_url: str, api_key: str, *, timeout: float = 60.0) -> None:
        self._base_url = base_url.rstrip("/")
        self._headers = {"Authorization": f"Bearer {api_key}"}
        self._client = httpx.AsyncClient(timeout=timeout)
        self._model: str | None = None

    async def aclose(self) -> None:
        await self._client.aclose()

    async def _resolve_model(self) -> str:
        """Auto-discover the served model id from ``/models``.

        vLLM rejects requests with an unknown model name (HTTP 404), so we
        cannot hard-code a placeholder. Probing once and caching keeps the
        hot path zero-overhead.
        """
        if self._model is not None:
            return self._model
        r = await self._client.get(f"{self._base_url}/models", headers=self._headers)
        r.raise_for_status()
        data = r.json().get("data") or []
        if not data:
            raise RuntimeError(f"no models served at {self._base_url}/models")
        self._model = data[0]["id"]
        return self._model

    async def embed(self, texts: list[str], *, model: str | None = None) -> list[list[float]]:
        if not texts:
            return []
        resolved = model or await self._resolve_model()
        r = await self._client.post(
            f"{self._base_url}/embeddings",
            json={"model": resolved, "input": texts},
            headers=self._headers,
        )
        r.raise_for_status()
        return [item["embedding"] for item in r.json()["data"]]

    async def embed_one(self, text: str) -> list[float]:
        out = await self.embed([text])
        return out[0]
