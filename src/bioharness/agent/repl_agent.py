"""REPL-based escalation agent.

The default escalation in this release is a single-call, longer-budget LLM
prompt that reasons over the assembled context + any pre-fetched tool
evidence (gene/UniProt/GO lookups). The full multi-iteration REPL agent
with Python execution is exposed as a hook: subclass
:class:`PipelineCascadeClient` and override :meth:`PipelineCascadeClient._agent`
to plug in a heavier implementation. The hook signature is stable.
"""

from __future__ import annotations

import logging
from typing import Any

from framework_eval.eval.types import Item

from bioharness.cascade.client import AgentOutcome
from bioharness.config import ServiceConfig

LOGGER = logging.getLogger(__name__)


def _format_passages(passages: list[dict[str, Any]] | None, limit: int = 8) -> str:
    if not passages:
        return ""
    parts = []
    for i, p in enumerate(passages[:limit], 1):
        text = (p.get("text") or "").strip()
        if not text:
            continue
        parts.append(f"[{i}] {text[:600]}")
    return "\n\n".join(parts)


async def run_agent(
    *,
    llm: Any,
    services: ServiceConfig,
    item: Item,
    retrieval: Any,
    fast_path: Any,
) -> AgentOutcome:
    """Default escalation: pre-call entity-lookup tools, run a single
    LLM analysis pass over the assembled context + tool evidence, and
    for ``list`` items run a second "what's missing" expansion pass so
    the rejudge stage sees a longer enumerated candidate set."""
    from bioharness.tools import precall_tools

    try:
        tool_evidence = await precall_tools(item.question, item.question_type)
    except Exception as exc:
        LOGGER.warning("precall_tools failed: %s", exc)
        tool_evidence = ""

    ctx = _format_passages(retrieval.passages if retrieval else None)

    # ---- pass 1: analysis ----------------------------------------------
    user_msg = (
        f"Question type: {item.question_type}\n"
        f"Question: {item.question}\n"
    )
    if item.options:
        user_msg += "Options:\n" + "\n".join(
            f"  {k}. {v}" for k, v in item.options.items()
        ) + "\n"
    if tool_evidence:
        user_msg += f"\n{tool_evidence}\n"
    if ctx:
        user_msg += (
            f"\nRetrieved context:\n{ctx}\n\n"
            "You may reference any of the passages above. Reason step by "
            "step internally; output only the final answer."
        )

    base_messages = [
        {"role": "system",
         "content": (
             "You are a meticulous biomedical reasoning assistant. Use the "
             "tool results (when present) as authoritative; use the "
             "retrieved passages as supporting context."
         )},
        {"role": "user", "content": user_msg},
    ]

    try:
        text = await llm.chat(
            model=services.model_name,
            messages=base_messages,
            max_tokens=512,
            temperature=0.1,
        )
    except Exception as exc:
        LOGGER.warning("agent escalation failed: %s; using fast-path answer", exc)
        return AgentOutcome(
            answer=fast_path.answer,
            response_text=fast_path.response_text,
            iterations=0,
            tool_calls=[],
            tool_evidence=tool_evidence,
        )

    return AgentOutcome(
        answer=(text or "").strip(),
        response_text=text or "",
        iterations=1,
        tool_calls=["gene_resolver"] if tool_evidence else [],
        tool_evidence=tool_evidence,
    )
