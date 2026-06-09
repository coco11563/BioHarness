"""Unit tests for the atlas (D) component — pure functions + wiring (no network)."""

from __future__ import annotations

import asyncio
from typing import Any

import pytest
from framework_eval.eval.types import Item

from bioharness.cascade import atlas, constrained
from bioharness.cascade.client import PipelineCascadeClient
from bioharness.config import ServiceConfig


def _services(**kw: Any) -> ServiceConfig:
    base = dict(
        llm_url="http://t/v1", embed_url="http://t/v1", rerank_url="http://t/v1",
        qdrant_url="http://t", pubmed_pg_url="postgresql://t/x",
        papergraph_pg_url="postgresql://t/x", model_name="t", api_key="EMPTY",
    )
    base.update(kw)
    return ServiceConfig(**base)


def _expr_item(gene: str = "TP53") -> Item:
    return Item(
        id="e", dataset="scihorizon_hgkb",
        question=f"What is the expression pattern of {gene} gene?",
        question_type="expression", answer='{"tissue_list": ["liver"]}',
    )


# ----------------------------------------------------------------------
# parse_expression_tissues
# ----------------------------------------------------------------------

@pytest.mark.parametrize("text, expected", [
    ('["liver", "kidney"]', ["liver", "kidney"]),
    ('Here you go: ["liver", "kidney"]', ["liver", "kidney"]),   # prose prefix
    ("liver\nkidney", ["liver", "kidney"]),                       # newline fallback
    ("liver; kidney; banana", ["liver", "kidney"]),               # semicolon + vocab filter
    ('["liver","liver","fat"]', ["liver", "fat"]),                # dedup
    ('{"tissue_list": ["heart","lung"]}', ["heart", "lung"]),     # dict form
    ('["low expression"]', ["low expression"]),                   # sentinel allowed
    ("", []),
    ("not json at all", []),
])
def test_parse_expression_tissues(text, expected):
    assert atlas.parse_expression_tissues(text) == expected


def test_vocab_is_27_unique():
    vocab = atlas.EXPRESSION_TISSUE_VOCAB
    assert len(vocab) == 27 == len(set(vocab))


# ----------------------------------------------------------------------
# gene extraction + atlas rows
# ----------------------------------------------------------------------

def test_gene_from_expression_question():
    q = "What is the expression pattern of TP53 gene?"
    assert atlas.gene_from_expression_question(q) == "TP53"
    assert atlas.gene_from_expression_question("unrelated") is None


def test_atlas_rows_from_hpa_normalises_and_filters():
    rows = atlas.atlas_rows_from_hpa("G", [
        {"tissue": "adipose tissue", "nx": 18.0},   # -> fat
        {"tissue": "cerebellum", "nx": 2.0},        # -> brain
        {"tissue": "liver", "nx": 23.0},
        {"tissue": "made up", "nx": 99.0},          # filtered out
        {"tissue": "kidney", "nx": 0.0},            # non-positive dropped
    ])
    assert rows == "G: liver:23, fat:18, brain:2"   # sorted by nx desc


def test_atlas_rows_from_hpa_empty_is_none():
    assert atlas.atlas_rows_from_hpa("G", []) is None


# ----------------------------------------------------------------------
# build_expression_messages (-D vs +D)
# ----------------------------------------------------------------------

def test_build_expression_messages_minus_d():
    msg = atlas.build_expression_messages("Q?")
    assert "ALLOWED TISSUES" in msg[0]["content"]
    assert "Reference tissue expression" not in msg[0]["content"]


def test_build_expression_messages_plus_d():
    msg = atlas.build_expression_messages("Q?", atlas_rows="G: liver:23")
    assert "Reference tissue expression" in msg[0]["content"]
    assert "G: liver:23" in msg[0]["content"]


# ----------------------------------------------------------------------
# client wiring + constrained_generate injection
# ----------------------------------------------------------------------

def test_atlas_client_disabled_by_default():
    client = PipelineCascadeClient(services=_services())
    assert client.services.enable_atlas is False
    assert client._atlas_client() is None


def test_atlas_client_constructs_when_enabled():
    client = PipelineCascadeClient(services=_services(enable_atlas=True))
    ac = client._atlas_client()
    assert type(ac).__name__ == "AtlasClient"


class _RecordingLLM:
    def __init__(self) -> None:
        self.messages: list[dict[str, str]] = []

    async def chat_with_first_token_confidence(self, *, model, messages, max_tokens, temperature):
        self.messages = messages
        return '["liver"]', 0.9


def test_constrained_generate_injects_atlas_for_expression():
    llm = _RecordingLLM()
    asyncio.run(constrained.constrained_generate(
        llm, model_name="m", item=_expr_item(), passages=None, atlas_rows="TP53: liver:23",
    ))
    user = llm.messages[-1]["content"]
    assert "Reference tissue expression" in user and "TP53: liver:23" in user


def test_constrained_generate_no_atlas_block_without_rows():
    llm = _RecordingLLM()
    asyncio.run(constrained.constrained_generate(
        llm, model_name="m", item=_expr_item(), passages=None, atlas_rows=None,
    ))
    assert "Reference tissue expression" not in llm.messages[-1]["content"]
