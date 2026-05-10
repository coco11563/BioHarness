"""OpenAI-compatible rerank (cross-encoder) client."""

from __future__ import annotations

import httpx


class RerankClient:
    def __init__(self, base_url: str, api_key: str, *, timeout: float = 60.0) -> None:
        self._base_url = base_url.rstrip("/")
        self._headers = {"Authorization": f"Bearer {api_key}"}
        self._client = httpx.AsyncClient(timeout=timeout)

    async def aclose(self) -> None:
        await self._client.aclose()

    async def score(
        self, query: str, documents: list[str], *, top_k: int | None = None,
        model: str = "rerank",
    ) -> list[float]:
        payload = {"model": model, "query": query, "documents": documents}
        if top_k is not None:
            payload["top_k"] = top_k
        r = await self._client.post(
            f"{self._base_url}/rerank", json=payload, headers=self._headers,
        )
        r.raise_for_status()
        data = r.json()
        # Expect data["results"] = [{"index": int, "relevance_score": float}, ...]
        results = data.get("results") or []
        scores = [0.0] * len(documents)
        for r_ in results:
            idx = r_.get("index")
            if isinstance(idx, int) and 0 <= idx < len(scores):
                scores[idx] = float(r_.get("relevance_score", 0.0))
        return scores
