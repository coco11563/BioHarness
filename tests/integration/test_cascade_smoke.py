"""Cascade smoke test that does not need pytest-asyncio."""

from __future__ import annotations

import asyncio

from framework_eval.eval.types import Item

from framework_chi.cascade.client import StubPipelineCascadeClient


def test_stub_round_trip() -> None:
    client = StubPipelineCascadeClient(fixed_answer="FINAL(yes)")
    item = Item(
        id="x", dataset="bioasq", question="?",
        question_type="yesno", answer="yes",
    )
    pred = asyncio.run(client.generate(item))
    assert pred.answer == "yes"
    asyncio.run(client.aclose())
