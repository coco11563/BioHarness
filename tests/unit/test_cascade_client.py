"""Cascade-routing tests using the StubV14CascadeClient + monkey-patched
sub-stages."""

from __future__ import annotations

from typing import Any

import pytest

from framework_eval.eval.types import Item

from framework_chi.cascade.client import (
    AgentOutcome,
    FastPathOutcome,
    RetrievalContext,
    StubV14CascadeClient,
    V14CascadeClient,
)
from framework_chi.config import CascadeOptions


def _item(qtype: str = "yesno", **kw: Any) -> Item:
    base = dict(
        id="x", dataset="bioasq", question="Does X help Y?",
        question_type=qtype, answer="yes",
    )
    base.update(kw)
    return Item(**base)


@pytest.mark.asyncio
async def test_stub_returns_extracted_answer() -> None:
    client = StubV14CascadeClient(fixed_answer="FINAL(yes)")
    pred = await client.generate(_item("yesno"))
    assert pred.answer == "yes"
    assert pred.extras["stage"] == "stub"


def _patch(monkeypatch, *, fast_logprob: float, grounded: bool, rejudge_text: str = "no"):
    async def fake_retrieve(self, item):  # type: ignore[no-redef]
        return RetrievalContext(passages=[{"text": "supporting evidence"}], rerank_scores=[1.0])

    async def fake_fast(self, item, ctx):
        return FastPathOutcome(
            answer="yes" if grounded else "maybe",
            response_text="raw",
            logprob=fast_logprob,
            grounded=grounded,
        )

    async def fake_agent(self, item, ctx, fp):
        return AgentOutcome(
            answer="agent draft",
            response_text="agent reasoning",
            iterations=1,
            tool_calls=["pubmed_search"],
        )

    async def fake_rejudge(self, item, agent):
        return rejudge_text

    monkeypatch.setattr(V14CascadeClient, "_retrieve", fake_retrieve)
    monkeypatch.setattr(V14CascadeClient, "_fast_path", fake_fast)
    monkeypatch.setattr(V14CascadeClient, "_agent",     fake_agent)
    monkeypatch.setattr(V14CascadeClient, "_rejudge",   fake_rejudge)


@pytest.mark.asyncio
async def test_high_logprob_returns_fast_path(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch(monkeypatch, fast_logprob=0.95, grounded=True, rejudge_text="no")
    client = V14CascadeClient(options=CascadeOptions(cascade_threshold=0.7))
    pred = await client.generate(_item("mcq"))
    assert pred.extras["stage"] == "fast_path"
    assert pred.answer == "yes"


@pytest.mark.asyncio
async def test_low_logprob_escalates_to_agent(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch(monkeypatch, fast_logprob=0.1, grounded=True, rejudge_text="no")
    client = V14CascadeClient(options=CascadeOptions(cascade_threshold=0.7))
    pred = await client.generate(_item("mcq"))
    assert pred.extras["stage"] == "agent_rejudged"
    assert pred.answer == "no"


@pytest.mark.asyncio
async def test_grounded_gate_escalates_when_not_grounded(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch(monkeypatch, fast_logprob=0.95, grounded=False, rejudge_text="no")
    client = V14CascadeClient(
        options=CascadeOptions(cascade_threshold=0.7, enable_grounded_gate=True),
    )
    pred = await client.generate(_item("mcq"))
    assert pred.extras["stage"] == "agent_rejudged"


@pytest.mark.asyncio
async def test_yesno_always_takes_fast_path(monkeypatch: pytest.MonkeyPatch) -> None:
    """Per paper §5.2: agent over-analyses yes/no; cascade always fast-paths."""
    _patch(monkeypatch, fast_logprob=0.0, grounded=False, rejudge_text="ignored")
    client = V14CascadeClient(
        options=CascadeOptions(cascade_threshold=0.7, enable_grounded_gate=True),
    )
    pred = await client.generate(_item("yesno"))
    assert pred.extras["stage"] == "fast_path"
