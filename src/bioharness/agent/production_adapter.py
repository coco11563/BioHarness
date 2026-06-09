"""Production REPL-agent adapter.

The default ``run_agent`` shipped under ``bioharness.agent.repl_agent``
is a single-pass LLM call; it is sufficient for ``yesno`` / ``mcq`` /
``summary`` items but underperforms on ``list`` and entity-lookup
``factoid`` items where the production cascade uses the multi-iteration
``BiomedicalRLMPipeline`` (REPL + tool dispatch + evidence-state).

When the user has the production source tree available locally, this
adapter wires the real pipeline into Chi's ``_agent`` hook so the agent
escalation is **identical** to the headline run:

  export BIOHARNESS_PRODUCTION_SRC=/path/to/PaperAsKnowledgeGraph-RAG

The adapter then loads ``src.rlm.pipeline.BiomedicalRLMPipeline`` and
delegates to it. When ``BIOHARNESS_PRODUCTION_SRC`` is unset, callers
should keep using the simpler ``bioharness.agent.repl_agent.run_agent``
default.
"""

from __future__ import annotations

import logging
import os
import sys
from typing import Any

from framework_eval.eval.types import Item

from bioharness.cascade.client import AgentOutcome
from bioharness.config import ServiceConfig

LOGGER = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# sys.path injection
# ---------------------------------------------------------------------------


def _production_src_root() -> str | None:
    """Resolve the path that should be added to ``sys.path``.

    Order of precedence:
      1. ``BIOHARNESS_PRODUCTION_SRC`` env var (must point at the directory
         that contains ``src/rlm/pipeline.py``).
      2. A sibling ``PaperAsKnowledgeGraph-RAG`` directory next to the
         current working directory (covers a common dev layout).
    Returns ``None`` if neither resolves.
    """
    explicit = os.environ.get("BIOHARNESS_PRODUCTION_SRC")
    if explicit:
        return explicit
    cwd = os.getcwd()
    sibling = os.path.join(os.path.dirname(cwd), "PaperAsKnowledgeGraph-RAG")
    if os.path.isdir(os.path.join(sibling, "src", "rlm")):
        return sibling
    return None


_PIPELINE_CACHE: tuple[Any, Any] | None = None


def _load_production_pipeline() -> tuple[Any, Any]:
    """Import and cache ``(BiomedicalRLMPipeline, PipelineConfig)``.

    Raises:
        RuntimeError: when the production source tree is not on disk or
            the pipeline module fails to import.
    """
    global _PIPELINE_CACHE  # noqa: PLW0603
    if _PIPELINE_CACHE is not None:
        return _PIPELINE_CACHE

    root = _production_src_root()
    if root is None:
        raise RuntimeError(
            "BIOHARNESS_PRODUCTION_SRC is not set and no sibling "
            "PaperAsKnowledgeGraph-RAG directory was found. The production "
            "REPL agent cannot be loaded; set the env var or install the "
            "production source tree."
        )
    if root not in sys.path:
        sys.path.insert(0, root)

    try:
        from src.rlm.pipeline import BiomedicalRLMPipeline, PipelineConfig
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(
            f"Could not import src.rlm.pipeline from {root!r}: {exc}. "
            "Verify that the directory contains src/rlm/pipeline.py and "
            "that all transitive dependencies (rlm framework, qdrant-client, "
            "asyncpg, etc.) are installed in the active Python environment."
        ) from exc

    _PIPELINE_CACHE = (BiomedicalRLMPipeline, PipelineConfig)
    return _PIPELINE_CACHE


def production_agent_available() -> bool:
    """Return True if ``BIOHARNESS_PRODUCTION_SRC`` resolves to a usable tree."""
    try:
        _load_production_pipeline()
        return True
    except Exception:  # noqa: BLE001
        return False


# ---------------------------------------------------------------------------
# Adapter
# ---------------------------------------------------------------------------


async def run_production_agent(
    *,
    llm: Any,                       # noqa: ARG001 - the production pipeline manages its own LLM
    services: ServiceConfig,        # noqa: ARG001 - production uses its own service config
    item: Item,
    retrieval: Any,
    fast_path: Any,
) -> AgentOutcome:
    """Delegate agent escalation to the production ``BiomedicalRLMPipeline``.

    The production pipeline accepts the assembled context as a dict and
    returns a result object with ``.answer`` and (optionally)
    ``.metadata``. We forward the retrieval passages as ``evidence`` and
    let the pipeline run its full REPL agent + tool dispatch.
    """
    pipeline_cls, config_cls = _load_production_pipeline()

    config = config_cls(
        version="v14",
        max_iterations=8,
        use_gold_context=True,
        yesno_dual_hypothesis=False,
        enable_kg_tools=True,
    )
    pipeline = pipeline_cls(config=config)

    # Build evidence text in the exact ``PubMed Evidence`` shape the
    # production pipeline expects.
    evidence_lines: list[str] = ["PubMed Evidence:"]
    for i, p in enumerate(retrieval.passages if retrieval else [], 1):
        text = (p.get("text") or "").strip()
        if not text:
            continue
        meta = p.get("metadata") or {}
        pmid = meta.get("pmid") or "unknown"
        title = (meta.get("title") or "").strip()
        evidence_lines.append(f"[{i}] PMID {pmid}")
        if title:
            evidence_lines.append(f"Title: {title}")
        evidence_lines.append(f"Abstract: {text[:1500]}")
        evidence_lines.append("")
    if len(evidence_lines) == 1:
        evidence_lines.append("No PubMed evidence found.")
    evidence = "\n".join(evidence_lines).strip()

    ctx: dict[str, Any] = {"evidence": evidence}
    if item.options:
        ctx["options"] = dict(item.options)

    try:
        result = await pipeline.answer_async(
            question=item.question,
            question_type=item.question_type,
            context=ctx,
        )
    except Exception as exc:  # noqa: BLE001
        LOGGER.warning(
            "production REPL agent failed for %s: %s; using fast-path answer",
            item.id, exc,
        )
        return AgentOutcome(
            answer=fast_path.answer,
            response_text=fast_path.response_text,
            iterations=0,
            tool_calls=[],
            tool_evidence="",
        )

    answer = getattr(result, "answer", None) or ""
    metadata = getattr(result, "metadata", None) or {}
    tool_calls = metadata.get("tool_calls") or []
    iterations = metadata.get("iterations") or 1

    return AgentOutcome(
        answer=str(answer).strip(),
        response_text=str(answer),
        iterations=int(iterations),
        tool_calls=list(tool_calls),
        tool_evidence="",
    )
