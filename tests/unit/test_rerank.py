"""Reranker client tests against a mocked HTTP transport (no network)."""

from __future__ import annotations

import json
import math

import httpx
import pytest

from bioharness.cascade.retrieval import dual_rerank
from bioharness.clients.rerank import (
    DOC_CHARS,
    QUERY_CHARS,
    RerankClient,
    build_prompt,
    yes_no_score,
)

EXPECTED_PROMPT = (
    "<|im_start|>system\n"
    "Judge whether the Document meets the requirements based on the Query and the "
    'Instruct provided. Note that the answer can only be "yes" or "no".<|im_end|>\n'
    "<|im_start|>user\n"
    "<Instruct>: Given a web search query, retrieve relevant passages that answer the query\n"
    "<Query>: q?\n<Document>: d.<|im_end|>\n"
    "<|im_start|>assistant\n<think>\n\n</think>\n\n"
)


def _completion_choice(index: int, yes: float | None, no: float | None) -> dict:
    top: dict[str, float] = {"maybe": -5.0}
    if yes is not None:
        top["yes"] = yes
    if no is not None:
        top["no"] = no
    return {"index": index, "text": "yes", "logprobs": {"top_logprobs": [top]}}


class _Server:
    """Records requests; serves /models, /rerank and /completions."""

    def __init__(self, *, rerank_status: int = 404, relevance: dict[str, float] | None = None):
        self.rerank_status = rerank_status
        self.relevance = relevance or {}
        self.requests: list[tuple[str, dict]] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        path = request.url.path
        body = json.loads(request.content) if request.content else {}
        self.requests.append((path, body))
        if path.endswith("/models"):
            return httpx.Response(200, json={"data": [{"id": "Qwen3-Reranker-8B"}]})
        if path.endswith("/rerank"):
            if self.rerank_status >= 400:
                return httpx.Response(self.rerank_status, json={"error": "not found"})
            return httpx.Response(
                200,
                json={
                    "results": [
                        {"index": i, "relevance_score": 1.0 - 0.1 * i}
                        for i in range(len(body["documents"]))
                    ]
                },
            )
        if path.endswith("/completions"):
            # Relevance depends on the document text inside each prompt.
            choices = []
            for i, prompt in enumerate(body["prompt"]):
                p_yes = next((v for k, v in self.relevance.items() if k in prompt), 0.5)
                choices.append(_completion_choice(i, math.log(p_yes), math.log(1 - p_yes)))
            return httpx.Response(200, json={"choices": choices})
        return httpx.Response(404)


def _client(server: _Server, **kw: object) -> RerankClient:
    return RerankClient(
        "http://rerank.test/v1",
        "EMPTY",
        transport=httpx.MockTransport(server),
        **kw,
    )


def test_build_prompt_matches_qwen3_reranker_template() -> None:
    assert build_prompt("q?", "d.") == EXPECTED_PROMPT


def test_build_prompt_truncates_query_and_document() -> None:
    prompt = build_prompt("Q" * 2000, "D" * 5000)
    assert "Q" * QUERY_CHARS + "\n" in prompt
    assert "Q" * (QUERY_CHARS + 1) not in prompt
    assert "D" * DOC_CHARS + "<|im_end|>" in prompt
    assert "D" * (DOC_CHARS + 1) not in prompt


def test_yes_no_score_is_normalised() -> None:
    assert yes_no_score({"yes": math.log(0.6), "no": math.log(0.2)}) == pytest.approx(0.75)
    # Variants: whitespace and quoted tokens, best logprob wins.
    top = {" Yes": math.log(0.3), '"yes"': math.log(0.6), "No": math.log(0.2)}
    assert yes_no_score(top) == pytest.approx(0.75)
    # Neither token present -> both at -10 -> 0.5.
    assert yes_no_score({"maybe": -0.1}) == pytest.approx(0.5)
    assert yes_no_score(None) == pytest.approx(0.5)
    # Only "yes" present -> close to 1.
    assert yes_no_score({"yes": -0.01}) > 0.999


@pytest.mark.asyncio
async def test_completion_mode_scores_each_document() -> None:
    server = _Server(relevance={"RELEVANT": 0.95, "OFFTOPIC": 0.02})
    client = _client(server)
    scores = await client.score("what binds TP53?", ["RELEVANT doc", "OFFTOPIC doc", "other"])
    await client.aclose()

    assert scores == pytest.approx([0.95, 0.02, 0.5])
    paths = [p for p, _ in server.requests]
    assert paths == ["/v1/models", "/v1/rerank", "/v1/completions"]
    body = server.requests[-1][1]
    assert body["model"] == "Qwen3-Reranker-8B"
    assert body["max_tokens"] == 1
    assert body["temperature"] == 0.0
    assert body["logprobs"] == 10
    assert body["prompt"][0] == build_prompt("what binds TP53?", "RELEVANT doc")
    # No chat-completions request is ever made.
    assert not any(p.endswith("/chat/completions") for p in paths)


@pytest.mark.asyncio
async def test_completion_mode_batches_and_handles_out_of_order_choices() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/models"):
            return httpx.Response(200, json={"data": [{"id": "m"}]})
        body = json.loads(request.content)
        n = len(body["prompt"])
        choices = [
            _completion_choice(
                i, math.log(0.9 if i == 0 else 0.1), math.log(0.1 if i == 0 else 0.9)
            )
            for i in range(n)
        ]
        return httpx.Response(200, json={"choices": list(reversed(choices))})

    client = RerankClient(
        "http://t/v1", "EMPTY", mode="completion", transport=httpx.MockTransport(handler)
    )
    scores = await client.score("q", [f"doc {i}" for i in range(70)])
    await client.aclose()
    assert len(scores) == 70
    # First document of each batch (0 and 64) is the relevant one.
    assert scores[0] == pytest.approx(0.9)
    assert scores[64] == pytest.approx(0.9)
    assert scores[1] == pytest.approx(0.1)
    assert scores[69] == pytest.approx(0.1)


@pytest.mark.asyncio
async def test_api_mode_used_when_rerank_endpoint_answers() -> None:
    server = _Server(rerank_status=200)
    client = _client(server)
    scores = await client.score("q" * 900, ["a" * 3000, "b"], top_k=2)
    await client.aclose()
    assert scores == pytest.approx([1.0, 0.9])
    path, body = server.requests[-1]
    assert path == "/v1/rerank"
    assert len(body["query"]) == QUERY_CHARS
    assert len(body["documents"][0]) == DOC_CHARS
    assert body["top_n"] == 2
    assert not any(p.endswith("/completions") for p, _ in server.requests)


@pytest.mark.asyncio
async def test_completion_http_error_raises_and_dual_rerank_falls_back() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/models"):
            return httpx.Response(200, json={"data": [{"id": "m"}]})
        return httpx.Response(500)

    client = RerankClient(
        "http://t/v1", "EMPTY", mode="completion", transport=httpx.MockTransport(handler)
    )
    with pytest.raises(httpx.HTTPStatusError):
        await client.score("q", ["a"])
    fallback = await dual_rerank(client, "q", [{"text": "a"}, {"text": "b"}], top_k=2)
    await client.aclose()
    assert fallback == [1.0, 0.5]


@pytest.mark.asyncio
async def test_dual_rerank_sends_title_and_text() -> None:
    server = _Server()
    client = _client(server, mode="completion")
    passages = [
        {"text": "Abstract body.", "metadata": {"title": "A Title"}},
        {"text": "No title here.", "metadata": {}},
    ]
    await dual_rerank(client, "q", passages, top_k=2)
    await client.aclose()
    prompts = server.requests[-1][1]["prompt"]
    assert "<Document>: A Title\nAbstract body.<|im_end|>" in prompts[0]
    assert "<Document>: No title here.<|im_end|>" in prompts[1]


def test_invalid_mode_rejected() -> None:
    with pytest.raises(ValueError):
        RerankClient("http://t/v1", "EMPTY", mode="chat")
