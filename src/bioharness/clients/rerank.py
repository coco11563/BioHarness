"""Rerank client.

Two implementations behind the same ``score(query, docs)`` API:

* **API mode** — sends ``/v1/rerank`` (Jina / Cohere style) when the
  backend exposes that endpoint. Single batched call.
* **Logprob mode** — for vLLM-served cross-encoders without
  ``/v1/rerank`` (Qwen3-Reranker etc.), scores each (query, document)
  pair via a yes/no chat completion with ``logprobs=True`` and uses
  the probability of the affirmative token as the relevance score.

The client auto-detects which path the endpoint supports on first use
and caches the choice. Both paths share the same model-id auto-discovery
the embedding client uses.
"""

from __future__ import annotations

import asyncio
import math

import httpx

_YES_TOKENS = {"yes", "Yes", "YES"}


class RerankClient:
    def __init__(self, base_url: str, api_key: str, *, timeout: float = 60.0) -> None:
        self._base_url = base_url.rstrip("/")
        self._headers = {"Authorization": f"Bearer {api_key}"}
        self._client = httpx.AsyncClient(timeout=timeout)
        self._model: str | None = None
        self._mode: str | None = None  # "api" | "logprob"

    async def aclose(self) -> None:
        await self._client.aclose()

    async def _resolve_model(self) -> str:
        if self._model is not None:
            return self._model
        r = await self._client.get(f"{self._base_url}/models", headers=self._headers)
        r.raise_for_status()
        data = r.json().get("data") or []
        if not data:
            raise RuntimeError(f"no models served at {self._base_url}/models")
        self._model = data[0]["id"]
        return self._model

    async def _detect_mode(self) -> str:
        if self._mode is not None:
            return self._mode
        # Try API mode with a 1x1 ping; fall through to logprob mode on 404.
        model = await self._resolve_model()
        try:
            r = await self._client.post(
                f"{self._base_url}/rerank",
                json={"model": model, "query": "ping", "documents": ["ping"]},
                headers=self._headers, timeout=5.0,
            )
            if r.status_code < 400:
                self._mode = "api"
                return self._mode
        except httpx.HTTPError:
            pass
        self._mode = "logprob"
        return self._mode

    async def score(
        self, query: str, documents: list[str], *, top_k: int | None = None,
    ) -> list[float]:
        if not documents:
            return []
        mode = await self._detect_mode()
        if mode == "api":
            return await self._score_api(query, documents, top_k=top_k)
        return await self._score_logprob(query, documents)

    # ---- API mode -----------------------------------------------------

    async def _score_api(
        self, query: str, documents: list[str], *, top_k: int | None,
    ) -> list[float]:
        model = await self._resolve_model()
        payload: dict[str, object] = {
            "model": model, "query": query, "documents": documents,
        }
        if top_k is not None:
            payload["top_k"] = top_k
        r = await self._client.post(
            f"{self._base_url}/rerank", json=payload, headers=self._headers,
        )
        r.raise_for_status()
        results = r.json().get("results") or []
        scores = [0.0] * len(documents)
        for entry in results:
            idx = entry.get("index")
            if isinstance(idx, int) and 0 <= idx < len(scores):
                scores[idx] = float(entry.get("relevance_score", 0.0))
        return scores

    # ---- Logprob mode -------------------------------------------------

    _RELEVANCE_PROMPT = (
        "Decide whether the document is relevant to the query.\n\n"
        "Query: {query}\n"
        "Document: {document}\n\n"
        "Answer with a single word: yes or no."
    )

    async def _score_one(self, query: str, document: str) -> float:
        model = await self._resolve_model()
        payload = {
            "model": model,
            "messages": [
                {"role": "user",
                 "content": self._RELEVANCE_PROMPT.format(
                     query=query, document=document[:1600],
                 )},
            ],
            "max_tokens": 1,
            "temperature": 0.0,
            "logprobs": True,
            "top_logprobs": 5,
            "chat_template_kwargs": {"enable_thinking": False},
        }
        r = await self._client.post(
            f"{self._base_url}/chat/completions",
            json=payload, headers=self._headers,
        )
        r.raise_for_status()
        choice = r.json()["choices"][0]
        lp = choice.get("logprobs") or {}
        content = lp.get("content") or []
        if not content:
            return 0.0
        top = content[0].get("top_logprobs") or []
        for cand in top:
            tok = (cand.get("token") or "").strip()
            if tok in _YES_TOKENS:
                return math.exp(cand["logprob"])
        return 0.0

    async def _score_logprob(self, query: str, documents: list[str]) -> list[float]:
        # Concurrency-bounded fan-out keeps the rerank cost predictable.
        sem = asyncio.Semaphore(8)

        async def one(doc: str) -> float:
            async with sem:
                try:
                    return await self._score_one(query, doc)
                except Exception:
                    return 0.0

        return await asyncio.gather(*[one(d) for d in documents])
