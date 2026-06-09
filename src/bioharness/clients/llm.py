"""OpenAI-compatible chat completion client."""

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

    async def chat(
        self,
        *,
        model: str,
        messages: list[dict[str, str]],
        max_tokens: int = 256,
        temperature: float = 0.0,
        enable_thinking: bool = False,
    ) -> str:
        """Plain chat completion, no logprobs. Used by query rewrite and
        agent escalation where the routing signal is not needed."""
        payload: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "chat_template_kwargs": {"enable_thinking": enable_thinking},
        }
        r = await self._client.post(
            f"{self._base_url}/chat/completions",
            json=payload, headers=self._headers,
        )
        r.raise_for_status()
        return r.json()["choices"][0]["message"]["content"] or ""

    async def chat_with_first_token_confidence(
        self,
        *,
        model: str,
        messages: list[dict[str, str]],
        max_tokens: int = 256,
        temperature: float = 0.1,
        top_logprobs: int = 5,
        enable_thinking: bool = False,
    ) -> tuple[str, float]:
        """Chat completion returning ``(response_text, confidence)``.

        Confidence = ``exp(first_token.logprob)`` — only the first emitted
        token is used as the cascade routing signal, matching the
        upstream pipeline.
        """
        payload: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "logprobs": True,
            "top_logprobs": top_logprobs,
            "chat_template_kwargs": {"enable_thinking": enable_thinking},
        }
        r = await self._client.post(
            f"{self._base_url}/chat/completions",
            json=payload, headers=self._headers,
        )
        r.raise_for_status()
        choice = r.json()["choices"][0]
        text = choice["message"]["content"] or ""
        content = (choice.get("logprobs") or {}).get("content") or []
        if content and content[0].get("logprob") is not None:
            confidence = math.exp(content[0]["logprob"])
        else:
            confidence = 0.0
        return text, confidence

    # Backwards-compatible alias used by some older call sites that wanted
    # a mean-logprob signal. Now identical to the first-token variant.
    chat_with_logprob = chat_with_first_token_confidence
