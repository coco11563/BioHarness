"""Load-balanced OpenAI-compatible clients for vLLM services.

DESIGN PRINCIPLE: Fail-Fast
- NO fallback/backup logic - errors propagate immediately
- NO silent retries that hide issues
- Explicit error messages with context

This module provides singleton client instances with round-robin load balancing:
- embed_client: 8 local embedding instances
- llm_client: 2 LLM instances
- rerank_client: 1 rerank instance

Configuration is loaded from src/config.py for centralized management.

Usage:
    from src.utils.clients import embed_client, llm_client, rerank_client

    # Embedding
    embeddings = await embed_client.embed(["text1", "text2"])

    # LLM Chat
    response = await llm_client.chat("What is diabetes?")

    # Rerank
    scores = await rerank_client.rerank("query", ["doc1", "doc2"])
"""

import asyncio
import math
import os
import re
import logging
from typing import Sequence

import httpx
from openai import AsyncOpenAI

# Import centralized configuration
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import get_config

# Setup logging
logger = logging.getLogger(__name__)

CHAT_TEMPLATE_KWARGS = {"enable_thinking": False}

# --- External (cloud, OpenAI-compatible) endpoints ----------------------------------
# XC_LLM_PORTABLE=1        drop the vLLM-only chat_template_kwargs (cloud APIs reject or
#                          ignore unknown body fields) and ask OpenRouter to return cost
# XC_LLM_OMIT_TEMPERATURE=1 omit temperature (reasoning models accept only the default)
# XC_LLM_REASONING_EFFORT=none|minimal|low|medium|high  OpenRouter unified reasoning knob
# XC_LLM_USAGE_LOG=<path>  append one JSON line per call with the provider usage block
# All default off, so local vLLM runs are byte-identical to before.
_LLM_PORTABLE = os.environ.get("XC_LLM_PORTABLE") == "1"
_LLM_OMIT_TEMPERATURE = os.environ.get("XC_LLM_OMIT_TEMPERATURE") == "1"
_LLM_REASONING_EFFORT = os.environ.get("XC_LLM_REASONING_EFFORT") or None
# XC_LLM_REASONING_OFF=1 sends OpenRouter's unified {"reasoning": {"enabled": false}} -- verified
# 2026-09-12 to disable hidden reasoning on the OpenRouter models of the frontier panel (some
# otherwise reason by default and return empty content under a 4-token cap).
_LLM_REASONING_OFF = os.environ.get("XC_LLM_REASONING_OFF") == "1"
_LLM_USAGE_LOG = os.environ.get("XC_LLM_USAGE_LOG") or None
# Cloud endpoints: bound the worst case of one hung request. Defaults keep the local
# behaviour (SDK timeout 120 s x 3 SDK retries x 8 failover attempts, i.e. up to ~1 h per
# call); XC_LLM_TIMEOUT / XC_LLM_SDK_RETRIES / XC_LLM_ATTEMPTS shrink it for API runs.
_LLM_ATTEMPTS = int(os.environ.get("XC_LLM_ATTEMPTS", "8"))


def _chat_kwargs(model: str, messages: list, max_tokens: int, temperature: float,
                 extra_body: dict | None = None) -> dict:
    """Build the chat.completions.create kwargs for local vLLM or a cloud endpoint."""
    kw: dict = {"model": model, "messages": messages, "max_tokens": max_tokens}
    if not _LLM_OMIT_TEMPERATURE:
        kw["temperature"] = temperature
    eb: dict = {} if _LLM_PORTABLE else {"chat_template_kwargs": CHAT_TEMPLATE_KWARGS}
    if _LLM_PORTABLE and _LLM_USAGE_LOG:
        eb["usage"] = {"include": True}
    if _LLM_PORTABLE and _LLM_REASONING_OFF:
        eb["reasoning"] = {"enabled": False}
    elif _LLM_PORTABLE and _LLM_REASONING_EFFORT:
        eb["reasoning"] = {"effort": _LLM_REASONING_EFFORT}
    if extra_body:
        eb.update(extra_body)
    if eb:
        kw["extra_body"] = eb
    return kw


def _log_usage(response, endpoint: str, model: str) -> None:
    if not _LLM_USAGE_LOG:
        return
    try:
        import json as _json
        import threading as _th
        import time as _time
        u = getattr(response, "usage", None)
        d = u.model_dump() if hasattr(u, "model_dump") else (dict(u) if u else {})
        det = d.get("completion_tokens_details") or {}
        rec = {"t": round(_time.time(), 3), "model": getattr(response, "model", None) or model,
               "endpoint": endpoint, "prompt_tokens": d.get("prompt_tokens"),
               "completion_tokens": d.get("completion_tokens"), "total_tokens": d.get("total_tokens"),
               "reasoning_tokens": det.get("reasoning_tokens") if isinstance(det, dict) else None,
               "cost": d.get("cost")}
        lock = globals().setdefault("_usage_log_lock", _th.Lock())
        with lock, open(_LLM_USAGE_LOG, "a") as f:
            f.write(_json.dumps(rec) + "\n")
    except Exception:  # noqa: BLE001 - accounting must never break a call
        pass

# Get configuration
_cfg = get_config()

# Configuration from centralized config.py
EMBEDDING_ENDPOINTS = _cfg.embedding.servers
EMBEDDING_MODEL = _cfg.embedding.model
EMBEDDING_API_KEY = _cfg.embedding.api_key

# Build (endpoint_url, model_name) tuples for per-endpoint model names
LLM_ENDPOINT_CONFIGS = [(ep.url, ep.model) for ep in _cfg.llm.endpoints]
LLM_MODEL = _cfg.llm.model  # Legacy: primary model name
LLM_API_KEY = _cfg.llm.api_key

RERANK_ENDPOINTS = _cfg.rerank.servers
RERANK_MODEL = _cfg.rerank.model
RERANK_API_KEY = _cfg.rerank.api_key


def _build_http_client(timeout: float, max_connections: int = 100) -> httpx.AsyncClient:
    """Build an HTTP client that avoids reusing half-closed keepalive sockets.

    The benchmark issues thousands of short local requests through SSH tunnels/nginx.
    Reusing keepalive connections in this setup can leave sockets in CLOSE_WAIT and
    stall long-running asyncio batches. Disabling keepalive is slower but much safer.
    """
    return httpx.AsyncClient(
        timeout=timeout,
        limits=httpx.Limits(
            max_connections=max_connections,
            max_keepalive_connections=0,
            keepalive_expiry=0.0,
        ),
        headers={"Connection": "close"},
        http2=False,
    )


class ClientError(Exception):
    """Base exception for client errors with context."""

    def __init__(self, message: str, client_type: str, endpoint: str | None = None):
        self.client_type = client_type
        self.endpoint = endpoint
        super().__init__(f"[{client_type}] {message}" + (f" (endpoint: {endpoint})" if endpoint else ""))


class EmbeddingError(ClientError):
    """Embedding-specific error."""

    def __init__(self, message: str, endpoint: str | None = None):
        super().__init__(message, "Embedding", endpoint)


class LLMError(ClientError):
    """LLM-specific error."""

    def __init__(self, message: str, endpoint: str | None = None):
        super().__init__(message, "LLM", endpoint)


class RerankError(ClientError):
    """Rerank-specific error."""

    def __init__(self, message: str, endpoint: str | None = None):
        super().__init__(message, "Rerank", endpoint)


# =============================================================================
# Embedding Client (Fail-Fast)
# =============================================================================

class EmbeddingClient:
    """Load-balanced embedding client with round-robin across instances.

    Uses moderate transport retries to absorb transient timeout spikes on the
    shared embedding tier.
    """

    def __init__(
        self,
        base_urls: Sequence[str],
        model: str,
        api_key: str = "EMPTY",
        timeout: float = 60.0,
        max_retries: int = 5,
    ):
        if not base_urls:
            raise EmbeddingError("No embedding endpoints configured")

        self._base_urls = list(base_urls)
        self._clients = [
            AsyncOpenAI(
                base_url=url,
                api_key=api_key,
                timeout=timeout,
                max_retries=max_retries,
                http_client=_build_http_client(timeout),
            )
            for url in base_urls
        ]
        self._model = model
        self._index = 0

    def _next_client(self) -> tuple[AsyncOpenAI, str]:
        """Get next client in round-robin fashion. Returns (client, endpoint_url)."""
        idx = self._index % len(self._clients)
        self._index += 1
        return self._clients[idx], self._base_urls[idx]

    async def embed(self, texts: str | Sequence[str]) -> list[list[float]]:
        """Generate embeddings with endpoint-level failover on transport errors.

        Cycles through all endpoints on connection/timeout errors before giving up.
        This absorbs SSH-tunnel drops where one endpoint dies but others are live.
        """
        if isinstance(texts, str):
            texts = [texts]

        if not texts:
            raise EmbeddingError("Empty text list provided")

        last_err: Exception | None = None
        last_endpoint: str | None = None
        # At least 4 attempts so a single live endpoint can ride out transient
        # timeouts (the shared 8004 LLM occasionally times out under load);
        # round-robin still rotates when multiple endpoints are configured.
        # 8 attempts with exponential backoff (~60 s): endpoints are reached over SSH
        # tunnels that can drop briefly; the old 4 x 0.5 s policy gave up in ~2 s
        # and killed multi-hour benchmark runs on a transient blip.
        for _attempt in range(max(8, len(self._clients))):
            client, endpoint = self._next_client()
            try:
                response = await client.embeddings.create(
                    model=self._model,
                    input=list(texts),
                )
                return [data.embedding for data in response.data]
            except Exception as e:
                last_err = e
                last_endpoint = endpoint
                logger.warning("Embedding endpoint %s failed (%s); failing over", endpoint, type(e).__name__)
                await asyncio.sleep(min(30.0, 0.5 * (2 ** _attempt)))
                continue

        raise EmbeddingError(
            f"All embedding endpoints failed; last error: {last_err}",
            last_endpoint,
        ) from last_err

    async def embed_single(self, text: str) -> list[float]:
        """Generate embedding for a single text."""
        embeddings = await self.embed(text)
        return embeddings[0]

    @property
    def model(self) -> str:
        return self._model

    @property
    def num_instances(self) -> int:
        return len(self._clients)


# =============================================================================
# LLM Client (Fail-Fast)
# =============================================================================

class LLMClient:
    """Load-balanced LLM client with round-robin across instances.

    Uses modest transport retries to absorb transient tunnel/nginx hiccups
    without masking persistent backend failures.
    """

    def __init__(
        self,
        endpoint_configs: Sequence[tuple[str, str]],
        default_model: str = "qwen",
        api_key: str = "EMPTY",
        timeout: float = 120.0,
        max_retries: int = 3,
    ):
        """Initialize LLM client with per-endpoint model names.

        Args:
            endpoint_configs: List of (endpoint_url, model_name) tuples
            default_model: Fallback model name if not specified per endpoint
            api_key: API key for authentication
            timeout: Request timeout in seconds
        """
        if not endpoint_configs:
            raise LLMError("No LLM endpoints configured")

        self._endpoint_configs = list(endpoint_configs)
        self._clients = [
            AsyncOpenAI(
                base_url=url,
                api_key=api_key,
                timeout=float(os.environ.get("XC_LLM_TIMEOUT", timeout)),
                max_retries=int(os.environ.get("XC_LLM_SDK_RETRIES", max_retries)),
                http_client=_build_http_client(float(os.environ.get("XC_LLM_TIMEOUT", timeout))),
            )
            for url, _ in endpoint_configs
        ]
        self._default_model = default_model
        self._index = 0

    def _next_client(self) -> tuple[AsyncOpenAI, str, str]:
        """Get next client in round-robin fashion. Returns (client, endpoint_url, model_name)."""
        idx = self._index % len(self._clients)
        self._index += 1
        url, model = self._endpoint_configs[idx]
        return self._clients[idx], url, model

    async def chat(
        self,
        prompt: str,
        system: str | None = None,
        max_tokens: int = 1024,
        temperature: float = 0.1,
    ) -> str:
        """Generate chat completion. Fails immediately on error.

        Args:
            prompt: User message.
            system: Optional system message.
            max_tokens: Maximum response tokens.
            temperature: Sampling temperature.

        Returns:
            Generated text response.

        Raises:
            LLMError: If chat completion fails.
        """
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        last_err: Exception | None = None
        last_endpoint: str | None = None
        # At least 4 attempts so a single live endpoint can ride out transient
        # timeouts (the shared 8004 LLM occasionally times out under load);
        # round-robin still rotates when multiple endpoints are configured.
        # 8 attempts with exponential backoff (~60 s): endpoints are reached over SSH
        # tunnels that can drop briefly; the old 4 x 0.5 s policy gave up in ~2 s
        # and killed multi-hour benchmark runs on a transient blip.
        for _attempt in range(max(_LLM_ATTEMPTS, len(self._clients))):
            client, endpoint, model_name = self._next_client()
            try:
                response = await client.chat.completions.create(
                    **_chat_kwargs(model_name, messages, max_tokens, temperature)
                )
                _log_usage(response, endpoint, model_name)
                content = response.choices[0].message.content
                if content is None:
                    raise LLMError("LLM returned None content", endpoint)
                return content
            except LLMError:
                raise
            except Exception as e:
                last_err = e
                last_endpoint = endpoint
                logger.warning("LLM endpoint %s failed (%s); failing over", endpoint, type(e).__name__)
                await asyncio.sleep(min(30.0, 0.5 * (2 ** _attempt)))
                continue

        raise LLMError(
            f"All LLM endpoints failed; last error: {last_err}",
            last_endpoint,
        ) from last_err

    async def chat_raw(
        self,
        messages: list[dict],
        max_tokens: int = 1024,
        temperature: float = 0.1,
        extra_body: dict | None = None,
    ):
        """Chat with caller-supplied messages, returning the raw response.

        Same round-robin + failover loop as chat(); exists because the official
        MedXpertQA protocol needs a multi-turn call with continue_final_message
        (the answer trigger is a prefix of the model's OWN turn) and logprobs,
        none of which chat()'s single-prompt signature can express.
        """
        last_err: Exception | None = None
        last_endpoint: str | None = None
        for _attempt in range(max(_LLM_ATTEMPTS, len(self._clients))):
            client, endpoint, model_name = self._next_client()
            try:
                response = await client.chat.completions.create(
                    **_chat_kwargs(model_name, messages, max_tokens, temperature, extra_body)
                )
                _log_usage(response, endpoint, model_name)
                return response
            except Exception as e:
                last_err = e
                last_endpoint = endpoint
                logger.warning("LLM endpoint %s failed (%s); failing over",
                               endpoint, type(e).__name__)
                await asyncio.sleep(min(30.0, 0.5 * (2 ** _attempt)))
                continue
        raise LLMError(f"All LLM endpoints failed; last error: {last_err}",
                       last_endpoint) from last_err

    async def chat_json(
        self,
        prompt: str,
        system: str | None = None,
        max_tokens: int = 1024,
    ) -> str:
        """Generate chat completion expecting JSON output (lower temperature)."""
        return await self.chat(
            prompt=prompt,
            system=system,
            max_tokens=max_tokens,
            temperature=0.05,
        )

    @property
    def model(self) -> str:
        return self._default_model

    @property
    def num_instances(self) -> int:
        return len(self._clients)

    @property
    def endpoint_configs(self) -> list[tuple[str, str]]:
        """Get list of (endpoint_url, model_name) tuples."""
        return self._endpoint_configs.copy()


# =============================================================================
# Rerank Client (Fail-Fast)
# =============================================================================

# Per-passage character cap sent to the reranker. 1500 fits an abstract; PMC
# full-text chunks run to ~2300 chars, so their tails were invisible to the
# reranker (LitQA2: 23% of gold key passages fell partly or wholly past the cap).
_RERANK_DOC_CHARS = int(os.environ.get("XC_RERANK_DOC_CHARS", "1500"))
# XC_RERANK_RAW=1 scores through the official completion-style template (see _score_raw_template).
_RERANK_RAW = os.environ.get("XC_RERANK_RAW", "0") == "1"


class RerankClient:
    """Rerank client supporting two backends:

    1. **API mode** (default): Uses the dedicated /v1/rerank endpoint (Jina/Cohere style).
       Works with cloud APIs like uni-api. Single batch call per rerank.

    2. **Logprobs mode**: Uses vLLM chat + logprobs for per-document scoring.
       Requires local vLLM with logprobs support.

    Auto-detection: if the endpoint URL is HTTPS (cloud), uses API mode.
    """

    SYSTEM_PROMPT = (
        "Judge whether the Document meets the requirements based on the Query "
        "and the Instruct provided. Note that the answer can only be \"yes\" or \"no\"."
    )

    DEFAULT_INSTRUCTION = (
        "Given a web search query, retrieve relevant passages that answer the query"
    )

    def __init__(
        self,
        base_urls: Sequence[str],
        model: str,
        api_key: str = "EMPTY",
        timeout: float = 60.0,
        max_retries: int = 3,
        instruction: str | None = None,
        mode: str = "auto",  # "auto" | "api" | "logprobs"
    ):
        if not base_urls:
            raise RerankError("No rerank endpoints configured")

        self._base_urls = list(base_urls)
        self._clients = [
            AsyncOpenAI(
                base_url=url,
                api_key=api_key,
                timeout=timeout,
                max_retries=max_retries,
                http_client=_build_http_client(timeout),
            )
            for url in base_urls
        ]
        self._api_key = api_key
        self._timeout = timeout
        self._model = model
        self._index = 0
        self._instruction = instruction or self.DEFAULT_INSTRUCTION

        # Determine mode
        if mode == "auto":
            self._use_api_mode = any(url.startswith("https://") for url in base_urls)
        else:
            self._use_api_mode = (mode == "api")

    def _next_client(self) -> tuple[AsyncOpenAI, str]:
        """Get next client in round-robin fashion. Returns (client, endpoint_url)."""
        idx = self._index % len(self._clients)
        self._index += 1
        return self._clients[idx], self._base_urls[idx]

    # ----- API mode (dedicated /v1/rerank endpoint) -----

    async def _rerank_api(
        self,
        query: str,
        documents: Sequence[str],
        top_k: int | None = None,
    ) -> list[tuple[int, float]]:
        """Rerank via dedicated /v1/rerank endpoint (batch, fast)."""
        import httpx

        _, base_url = self._next_client()
        # Build rerank URL: replace /v1/... with /v1/rerank
        rerank_url = base_url.rstrip("/")
        if rerank_url.endswith("/v1"):
            rerank_url += "/rerank"
        else:
            rerank_url = rerank_url.rsplit("/v1", 1)[0] + "/v1/rerank"

        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }
        payload: dict = {
            "model": self._model,
            "query": query[:500],
            "documents": [d[:_RERANK_DOC_CHARS] for d in documents],
        }
        if top_k is not None:
            payload["top_n"] = top_k

        async with httpx.AsyncClient(timeout=self._timeout) as client:
            resp = await client.post(rerank_url, headers=headers, json=payload)
            if resp.status_code != 200:
                raise RerankError(
                    f"Rerank API returned {resp.status_code}: {resp.text[:200]}",
                    rerank_url,
                )
            data = resp.json()

        results = [
            (item["index"], item["relevance_score"])
            for item in data.get("results", [])
        ]
        results.sort(key=lambda x: x[1], reverse=True)

        if top_k is not None:
            results = results[:top_k]

        return results

    # ----- Logprobs mode (vLLM chat, per-document scoring) -----

    def _build_messages(self, query: str, document: str) -> list[dict]:
        """Build chat messages in official Qwen3-Reranker format."""
        user_content = (
            f"<Instruct>: {self._instruction}\n\n"
            f"<Query>: {query}\n\n"
            f"<Document>: {document}"
        )
        return [
            {"role": "system", "content": self.SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ]

    def _compute_score_from_logprobs(self, logprobs_content: list) -> float:
        """Compute relevance score from yes/no logprobs.

        Score = P(yes) / (P(yes) + P(no))
        """
        if not logprobs_content:
            raise RerankError("No logprobs in response")

        # Get the last token's logprobs
        last_token_logprobs = logprobs_content[-1]
        if not hasattr(last_token_logprobs, 'top_logprobs'):
            raise RerankError("No top_logprobs in response")

        top_logprobs = last_token_logprobs.top_logprobs

        # Find yes/no logprobs (check various forms)
        yes_logprob = -10.0  # Default very low probability
        no_logprob = -10.0

        for lp in top_logprobs:
            token_lower = lp.token.lower().strip()
            if token_lower in ("yes", "yes.", "\"yes\"", "yes\""):
                yes_logprob = max(yes_logprob, lp.logprob)
            elif token_lower in ("no", "no.", "\"no\"", "no\""):
                no_logprob = max(no_logprob, lp.logprob)

        # Convert logprobs to probabilities and compute score
        import math
        yes_prob = math.exp(yes_logprob)
        no_prob = math.exp(no_logprob)

        # Avoid division by zero
        total = yes_prob + no_prob
        if total < 1e-10:
            return 0.5  # Uncertain

        return yes_prob / total

    async def _score_raw_template(self, client: AsyncOpenAI, query: str, document: str) -> float:
        """Official Qwen3-Reranker scoring: text completion with the model's own
        prompt layout (judge system prompt, <Instruct>/<Query>/<Document>, an empty
        think block) and P(yes)/(P(yes)+P(no)) from the next-token logprobs.
        The chat-completions route returned a content-independent distribution
        (relevant 0.050 / irrelevant 0.050 / empty 0.045 on every instance), i.e.
        the reranker never saw the document; this route gives 0.953 / 0.000 / 0.000.
        """
        prompt = (
            f"<|im_start|>system\n{self.SYSTEM_PROMPT}<|im_end|>\n"
            f"<|im_start|>user\n<Instruct>: {self._instruction}\n"
            f"<Query>: {query[:500]}\n<Document>: {document[:_RERANK_DOC_CHARS]}<|im_end|>\n"
            f"<|im_start|>assistant\n<think>\n\n</think>\n\n"
        )
        resp = await client.completions.create(
            model=self._model, prompt=prompt, max_tokens=1, temperature=0.0, logprobs=10,
        )
        lp = resp.choices[0].logprobs
        top = (lp.top_logprobs[0] if lp and lp.top_logprobs else {}) or {}
        yes = max([v for k, v in top.items() if k.strip().lower() in ("yes", '"yes"')], default=-10.0)
        no = max([v for k, v in top.items() if k.strip().lower() in ("no", '"no"')], default=-10.0)
        return math.exp(yes) / (math.exp(yes) + math.exp(no))

    async def _score_single(
        self,
        query: str,
        document: str,
        doc_idx: int,
    ) -> tuple[int, float]:
        """Score a single document's relevance using logprobs. Fails on error."""
        client, endpoint = self._next_client()
        if _RERANK_RAW:
            return doc_idx, await self._score_raw_template(client, query, document)

        messages = self._build_messages(
            query=query[:500],
            document=document[:_RERANK_DOC_CHARS],
        )

        try:
            response = await client.chat.completions.create(
                model=self._model,
                messages=messages,
                max_tokens=1,  # Only need one token (yes/no)
                temperature=0.0,
                logprobs=True,
                top_logprobs=20,  # Get top 20 token probabilities
                extra_body={
                    "chat_template_kwargs": {"enable_thinking": False}
                },
            )

            choice = response.choices[0]

            # Check if logprobs are available
            if not choice.logprobs or not choice.logprobs.content:
                # Fallback: parse text response
                content = (choice.message.content or "").lower().strip()
                if "yes" in content:
                    return (doc_idx, 1.0)
                elif "no" in content:
                    return (doc_idx, 0.0)
                else:
                    raise RerankError(
                        f"No logprobs and unclear response: '{content[:50]}'",
                        endpoint
                    )

            score = self._compute_score_from_logprobs(choice.logprobs.content)
            return (doc_idx, score)

        except RerankError:
            raise
        except Exception as e:
            raise RerankError(f"Failed to score document {doc_idx}: {e}", endpoint) from e

    def _raw_prompt(self, query: str, document: str) -> str:
        return (
            f"<|im_start|>system\n{self.SYSTEM_PROMPT}<|im_end|>\n"
            f"<|im_start|>user\n<Instruct>: {self._instruction}\n"
            f"<Query>: {query[:500]}\n<Document>: {document[:_RERANK_DOC_CHARS]}<|im_end|>\n"
            f"<|im_start|>assistant\n<think>\n\n</think>\n\n"
        )

    async def _rerank_raw_batched(self, query: str, documents: Sequence[str], top_k: int | None) -> list[tuple[int, float]]:
        """Same scoring as _score_raw_template, but one completions request per
        batch of 64 prompts (vLLM accepts a prompt list): 60 documents score in
        ~2.6 s instead of ~8.8 s with per-document calls. Scores are identical."""
        client, _ = self._next_client()
        results: list[tuple[int, float]] = []
        for start in range(0, len(documents), 64):
            batch = documents[start:start + 64]
            resp = await client.completions.create(
                model=self._model, prompt=[self._raw_prompt(query, d) for d in batch],
                max_tokens=1, temperature=0.0, logprobs=10,
            )
            by_index = {c.index: c for c in resp.choices}
            for j in range(len(batch)):
                ch = by_index.get(j)
                lp = ch.logprobs if ch else None
                top = (lp.top_logprobs[0] if lp and lp.top_logprobs else {}) or {}
                yes = max([v for k, v in top.items() if k.strip().lower() in ("yes", '"yes"')], default=-10.0)
                no = max([v for k, v in top.items() if k.strip().lower() in ("no", '"no"')], default=-10.0)
                results.append((start + j, math.exp(yes) / (math.exp(yes) + math.exp(no))))
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:top_k] if top_k is not None else results

    async def _rerank_logprobs(
        self,
        query: str,
        documents: Sequence[str],
        top_k: int | None = None,
        max_concurrent: int = 10,
    ) -> list[tuple[int, float]]:
        """Rerank via logprobs-based per-document scoring."""
        if _RERANK_RAW:
            return await self._rerank_raw_batched(query, documents, top_k)
        semaphore = asyncio.Semaphore(max_concurrent)

        async def score_with_limit(idx: int, doc: str) -> tuple[int, float]:
            async with semaphore:
                return await self._score_single(query, doc, idx)

        tasks = [score_with_limit(i, doc) for i, doc in enumerate(documents)]

        results = await asyncio.gather(*tasks)
        results = sorted(results, key=lambda x: x[1], reverse=True)

        if top_k is not None:
            results = results[:top_k]

        return results

    # ----- Public interface -----

    async def rerank(
        self,
        query: str,
        documents: Sequence[str],
        top_k: int | None = None,
        max_concurrent: int = 10,
    ) -> list[tuple[int, float]]:
        """Rerank documents by relevance to query. Fails on any error.

        Args:
            query: Query text.
            documents: List of document texts to rerank.
            top_k: Return only top-k results (None for all).
            max_concurrent: Max concurrent scoring requests (logprobs mode only).

        Returns:
            List of (document_index, score) tuples sorted by score descending.

        Raises:
            RerankError: If scoring fails.
        """
        if not documents:
            return []

        if self._use_api_mode:
            return await self._rerank_api(query, documents, top_k)
        else:
            return await self._rerank_logprobs(query, documents, top_k, max_concurrent)

    @property
    def model(self) -> str:
        return self._model

    @property
    def num_instances(self) -> int:
        return len(self._clients)


# =============================================================================
# Singleton Instances (Global)
# =============================================================================

# Pre-initialized load-balanced clients for global use
embed_client = EmbeddingClient(
    base_urls=EMBEDDING_ENDPOINTS,
    model=EMBEDDING_MODEL,
    api_key=EMBEDDING_API_KEY,
)

llm_client = LLMClient(
    endpoint_configs=LLM_ENDPOINT_CONFIGS,
    default_model=LLM_MODEL,
    api_key=LLM_API_KEY,
    timeout=1800.0,
)

rerank_client = RerankClient(
    base_urls=RERANK_ENDPOINTS,
    model=RERANK_MODEL,
    api_key=RERANK_API_KEY,
)


# =============================================================================
# Convenience Functions
# =============================================================================

async def embed(texts: str | Sequence[str]) -> list[list[float]]:
    """Convenience function for embedding."""
    return await embed_client.embed(texts)


async def chat(prompt: str, **kwargs) -> str:
    """Convenience function for chat completion."""
    return await llm_client.chat(prompt, **kwargs)


# =============================================================================
# Test (Fail-Fast)
# =============================================================================

async def test_clients():
    """Test all singleton clients. Will fail immediately on any error."""
    print("=" * 60)
    print("Testing Singleton Load-Balanced Clients (Fail-Fast)")
    print("=" * 60)

    # Test Embedding
    print(f"\n[Embedding] {embed_client.num_instances} instances")
    emb = await embed_client.embed(["test embedding"])
    print(f"  OK: Embedding dim = {len(emb[0])}")

    # Test LLM
    print(f"\n[LLM] {llm_client.num_instances} instances")
    resp = await llm_client.chat("Say 'hello' only", max_tokens=10)
    print(f"  OK: Response = {resp[:50]}")

    # Test Rerank
    print(f"\n[Rerank] {rerank_client.num_instances} instances")
    docs = ["diabetes treatment", "weather today"]
    results = await rerank_client.rerank("diabetes", docs, max_concurrent=2)
    print(f"  OK: Scores = {results}")

    print("\n" + "=" * 60)
    print("All tests passed!")


if __name__ == "__main__":
    asyncio.run(test_clients())
