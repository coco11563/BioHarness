"""OpenAI-compatible chat completion client with logprob support."""

from __future__ import annotations

import math
from typing import Any

import httpx


class LLMClient:
    def __init__(self, base_url: str, api_key: str, *, timeout: float = 120.0) -> None:
        self._base_url = base_url.rstrip("/")
        self._headers = {"Authorization": f"Bearer {api_key}"}
        self._client = httpx.AsyncClient(timeout=timeout)

    async def aclose(self) -> None:
        await self._client.aclose()

    async def chat_with_logprob(
        self,
        *,
        model: str,
        messages: list[dict[str, str]],
        max_tokens: int = 256,
        temperature: float = 0.0,
        top_logprobs: int = 1,
    ) -> tuple[str, float]:
        """Chat completion that also returns the mean logprob of the
        emitted tokens. The mean is used as the cascade routing signal.
        """
        payload: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "logprobs": True,
            "top_logprobs": top_logprobs,
        }
        r = await self._client.post(
            f"{self._base_url}/chat/completions",
            json=payload, headers=self._headers,
        )
        r.raise_for_status()
        data = r.json()
        choice = data["choices"][0]
        text = choice["message"]["content"] or ""
        logprobs = choice.get("logprobs") or {}
        token_logprobs = [
            t["logprob"]
            for t in (logprobs.get("content") or [])
            if t.get("logprob") is not None
        ]
        if token_logprobs:
            mean_lp = sum(token_logprobs) / len(token_logprobs)
            # Convert mean logprob to a 0-1 confidence via softmax-like clamp.
            confidence = math.exp(mean_lp)
        else:
            confidence = 0.0
        return text, confidence
