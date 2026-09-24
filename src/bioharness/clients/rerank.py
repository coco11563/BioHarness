"""Rerank client.

Two implementations behind the same ``score(query, docs)`` API:

* **API mode** -- sends ``/v1/rerank`` (Jina / Cohere style) when the
  backend exposes that endpoint. Single batched call.
* **Completion mode** -- for Qwen3-Reranker served as a generative model
  (vLLM without ``/v1/rerank``). Each (query, document) pair is written
  into the model's own raw prompt template (judge system prompt,
  ``<Instruct>/<Query>/<Document>`` user turn, empty think block) and sent
  to ``/v1/completions`` with ``max_tokens=1`` and ``logprobs``. The score
  is ``P(yes) / (P(yes) + P(no))`` over the next-token distribution. This
  is the research code's ``XC_RERANK_RAW=1`` scorer, set by the runs from
  2026-09-08 on (MedXpertQA, LitQA2); the April and June runs used the
  research code's default chat-completions scorer.

There is no chat-completions path.

The client auto-detects which path the endpoint supports on first use
and caches the choice (API mode wins when ``/v1/rerank`` answers); pass
``mode="completion"`` to force the paper's September scorer, or
``mode="api"`` to skip detection. The model id is auto-discovered from ``/v1/models``.
"""

from __future__ import annotations

import math
from typing import Any

import httpx

# Qwen3-Reranker prompt pieces (verbatim from the model's reference usage).
SYSTEM_PROMPT = (
    "Judge whether the Document meets the requirements based on the Query "
    'and the Instruct provided. Note that the answer can only be "yes" or "no".'
)
DEFAULT_INSTRUCTION = "Given a web search query, retrieve relevant passages that answer the query"

QUERY_CHARS = 500  # query cap
DOC_CHARS = 1500  # document cap (title + abstract fits)
BATCH_SIZE = 64  # prompts per /v1/completions request
_MISSING_LOGPROB = -10.0  # used when "yes" / "no" is absent from the top logprobs

_MODES = ("auto", "api", "completion")


def build_prompt(query: str, document: str, instruction: str = DEFAULT_INSTRUCTION) -> str:
    """Raw Qwen3-Reranker completion prompt for one (query, document) pair."""
    return (
        f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n"
        f"<|im_start|>user\n<Instruct>: {instruction}\n"
        f"<Query>: {query[:QUERY_CHARS]}\n<Document>: {document[:DOC_CHARS]}<|im_end|>\n"
        f"<|im_start|>assistant\n<think>\n\n</think>\n\n"
    )


def yes_no_score(top_logprobs: dict[str, float] | None) -> float:
    """``P(yes) / (P(yes) + P(no))`` from one position's top-logprob map.

    A token missing from the map counts as logprob -10, so a position that
    shows neither token scores 0.5.
    """
    top = top_logprobs or {}
    yes = max(
        (v for k, v in top.items() if k.strip().lower() in ("yes", '"yes"')),
        default=_MISSING_LOGPROB,
    )
    no = max(
        (v for k, v in top.items() if k.strip().lower() in ("no", '"no"')),
        default=_MISSING_LOGPROB,
    )
    return math.exp(yes) / (math.exp(yes) + math.exp(no))


class RerankClient:
    def __init__(
        self,
        base_url: str,
        api_key: str,
        *,
        timeout: float = 60.0,
        mode: str = "auto",
        instruction: str = DEFAULT_INSTRUCTION,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        if mode not in _MODES:
            raise ValueError(f"mode must be one of {_MODES}, got {mode!r}")
        self._base_url = base_url.rstrip("/")
        self._headers = {"Authorization": f"Bearer {api_key}"}
        self._client = httpx.AsyncClient(timeout=timeout, transport=transport)
        self._model: str | None = None
        self._mode: str | None = None if mode == "auto" else mode
        self._instruction = instruction

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
        # Try API mode with a 1x1 ping; fall through to completion mode otherwise.
        model = await self._resolve_model()
        try:
            r = await self._client.post(
                f"{self._base_url}/rerank",
                json={"model": model, "query": "ping", "documents": ["ping"]},
                headers=self._headers,
                timeout=5.0,
            )
            if r.status_code < 400:
                self._mode = "api"
                return self._mode
        except httpx.HTTPError:
            pass
        self._mode = "completion"
        return self._mode

    async def score(
        self,
        query: str,
        documents: list[str],
        *,
        top_k: int | None = None,
    ) -> list[float]:
        """Relevance scores aligned with ``documents``. Raises on HTTP errors."""
        if not documents:
            return []
        mode = await self._detect_mode()
        if mode == "api":
            return await self._score_api(query, documents, top_k=top_k)
        return await self._score_completion(query, documents)

    # ---- API mode -----------------------------------------------------

    async def _score_api(
        self,
        query: str,
        documents: list[str],
        *,
        top_k: int | None,
    ) -> list[float]:
        model = await self._resolve_model()
        payload: dict[str, object] = {
            "model": model,
            "query": query[:QUERY_CHARS],
            "documents": [d[:DOC_CHARS] for d in documents],
        }
        if top_k is not None:
            payload["top_n"] = top_k  # vLLM / Jina /v1/rerank field name
        r = await self._client.post(
            f"{self._base_url}/rerank",
            json=payload,
            headers=self._headers,
        )
        r.raise_for_status()
        results = r.json().get("results") or []
        scores = [0.0] * len(documents)
        for entry in results:
            idx = entry.get("index")
            if isinstance(idx, int) and 0 <= idx < len(scores):
                scores[idx] = float(entry.get("relevance_score", 0.0))
        return scores

    # ---- Completion mode (raw Qwen3-Reranker template) ----------------

    async def _score_completion(self, query: str, documents: list[str]) -> list[float]:
        model = await self._resolve_model()
        scores: list[float] = []
        for start in range(0, len(documents), BATCH_SIZE):
            batch = documents[start : start + BATCH_SIZE]
            payload = {
                "model": model,
                "prompt": [build_prompt(query, d, self._instruction) for d in batch],
                "max_tokens": 1,
                "temperature": 0.0,
                "logprobs": 10,
            }
            r = await self._client.post(
                f"{self._base_url}/completions",
                json=payload,
                headers=self._headers,
            )
            r.raise_for_status()
            by_index: dict[int, Any] = {
                c.get("index", i): c for i, c in enumerate(r.json().get("choices") or [])
            }
            for j in range(len(batch)):
                lp = (by_index.get(j) or {}).get("logprobs") or {}
                top = lp.get("top_logprobs") or []
                scores.append(yes_no_score(top[0] if top else None))
        return scores
