"""Cascade-routing tests using monkey-patched sub-stages."""

from __future__ import annotations

from typing import Any

import pytest
from framework_eval.eval.types import Item

from bioharness.cascade.client import (
    AgentOutcome,
    FastPathOutcome,
    PipelineCascadeClient,
    RetrievalContext,
    StubPipelineCascadeClient,
)
from bioharness.config import ServiceConfig


def _item(qtype: str = "yesno", **kw: Any) -> Item:
    base = dict(
        id="x", dataset="bioasq", question="Does X help Y?",
        question_type=qtype, answer="yes",
    )
    base.update(kw)
    return Item(**base)


def _services(force_agent: bool = False) -> ServiceConfig:
    return ServiceConfig(
        llm_url="http://t/v1", embed_url="http://t/v1", rerank_url="http://t/v1",
        qdrant_url="http://t", pubmed_pg_url="postgresql://t/x",
        papergraph_pg_url="postgresql://t/x", model_name="t", api_key="EMPTY",
        force_agent=force_agent,
    )


def _patch(monkeypatch, *, fast_logprob: float, grounded: bool,
           fast_answer: str = "yes", rejudge_text: str = "no"):
    async def fake_retrieve(self, item):
        return RetrievalContext(
            passages=[{"text": "supporting evidence"}],
            rerank_scores=[1.0], rewritten_query="rewritten",
        )

    async def fake_fast(self, item, ctx):
        return FastPathOutcome(
            answer=fast_answer, response_text="raw",
            logprob=fast_logprob, grounded=grounded,
        )

    async def fake_agent(self, item, ctx, fp):
        return AgentOutcome(
            answer="agent draft", response_text="agent reasoning",
            iterations=1, tool_calls=["pubmed_search"],
        )

    async def fake_rejudge(self, item, agent, retrieval=None):
        return rejudge_text

    monkeypatch.setattr(PipelineCascadeClient, "_retrieve", fake_retrieve)
    monkeypatch.setattr(PipelineCascadeClient, "_fast_path", fake_fast)
    monkeypatch.setattr(PipelineCascadeClient, "_agent",     fake_agent)
    monkeypatch.setattr(PipelineCascadeClient, "_rejudge",   fake_rejudge)


@pytest.mark.asyncio
async def test_stub_returns_extracted_answer() -> None:
    client = StubPipelineCascadeClient(fixed_answer="yes", services=_services())
    pred = await client.generate(_item("yesno"))
    assert pred.answer == "yes"
    assert pred.extras["stage"] == "stub"


@pytest.mark.asyncio
async def test_high_logprob_returns_fast_path(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch(monkeypatch, fast_logprob=0.95, grounded=True)
    client = PipelineCascadeClient(services=_services())
    pred = await client.generate(_item("mcq", answer="A"))
    assert pred.extras["stage"] == "fast_path"


@pytest.mark.asyncio
async def test_low_logprob_escalates(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch(monkeypatch, fast_logprob=0.1, grounded=True, rejudge_text="no")
    client = PipelineCascadeClient(services=_services())
    pred = await client.generate(_item("mcq", answer="A"))
    assert pred.extras["stage"] == "agent_rejudged"
    assert pred.answer == "no"


@pytest.mark.asyncio
async def test_ungrounded_factoid_escalates(monkeypatch: pytest.MonkeyPatch) -> None:
    """Grounded gate applies to factoid/list (pipeline GROUNDED_CHECK_TYPES)."""
    _patch(monkeypatch, fast_logprob=0.95, grounded=False, rejudge_text="no")
    client = PipelineCascadeClient(services=_services())
    pred = await client.generate(_item("factoid", answer="entity"))
    assert pred.extras["stage"] == "agent_rejudged"


@pytest.mark.asyncio
async def test_ungrounded_mcq_does_not_escalate(monkeypatch: pytest.MonkeyPatch) -> None:
    """Grounded gate must NOT fire for mcq: substring grounding is meaningless
    for one-token labels, and the pipeline restricts it to factoid/list."""
    _patch(monkeypatch, fast_logprob=0.95, grounded=False)
    client = PipelineCascadeClient(services=_services())
    pred = await client.generate(_item("mcq", answer="A"))
    assert pred.extras["stage"] == "fast_path"


@pytest.mark.asyncio
async def test_expression_always_takes_fast_path(monkeypatch: pytest.MonkeyPatch) -> None:
    """Expression is fully handled by the repair-context fast path; it never
    escalates (would discard the +D atlas signal and diverge from the pipeline),
    even with low confidence, ungrounded, and force_agent=True."""
    _patch(monkeypatch, fast_logprob=0.0, grounded=False)
    client = PipelineCascadeClient(services=_services(force_agent=True))
    pred = await client.generate(_item("expression", answer='{"tissue_list": []}'))
    assert pred.extras["stage"] == "fast_path"


@pytest.mark.asyncio
async def test_yesno_always_takes_fast_path(monkeypatch: pytest.MonkeyPatch) -> None:
    """Paper §5.2: yesno always fast-paths; agent over-analyses."""
    _patch(monkeypatch, fast_logprob=0.0, grounded=False)
    client = PipelineCascadeClient(services=_services(force_agent=True))
    pred = await client.generate(_item("yesno"))
    assert pred.extras["stage"] == "fast_path"


@pytest.mark.asyncio
async def test_force_agent_overrides_high_logprob(monkeypatch: pytest.MonkeyPatch) -> None:
    """force_agent=True escalates even high-confidence non-yesno items."""
    _patch(monkeypatch, fast_logprob=0.99, grounded=True, rejudge_text="no")
    client = PipelineCascadeClient(services=_services(force_agent=True))
    pred = await client.generate(_item("mcq", answer="A"))
    assert pred.extras["stage"] == "agent_rejudged"


@pytest.mark.asyncio
async def test_constrained_generate_prepends_tool_evidence() -> None:
    """Pre-fetched tool evidence must reach the fast-path constrained prompt
    (so gene-DB lookups are not answered 'unknown' before escalation)."""
    from bioharness.cascade.constrained import constrained_generate

    captured: dict[str, Any] = {}

    class _FakeLLM:
        async def chat_with_first_token_confidence(self, *, model, messages, max_tokens, temperature):  # noqa: ANN001
            captured["user"] = messages[-1]["content"]
            return "BRCA1", 0.9

    await constrained_generate(
        _FakeLLM(), model_name="m", item=_item("factoid", answer="BRCA1"),
        passages=[{"text": "doc"}],
        tool_evidence="## Tool Results (pre-fetched)\n[snp_lookup(rs1)] gene=BRCA1, chromosome=chr17",
    )
    assert "[snp_lookup(rs1)] gene=BRCA1" in captured["user"]
