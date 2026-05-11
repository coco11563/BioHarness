"""REPL-based escalation agent.

The default escalation in this release is a single-call, longer-budget LLM
prompt that reasons over the assembled context. The full multi-iteration
REPL agent with Python execution and multi-tool dispatch is exposed as a
hook: subclass :class:`V14CascadeClient` and override
:meth:`V14CascadeClient._agent` to plug in a heavier implementation. The
hook signature is stable.
"""

from __future__ import annotations

import logging
from typing import Any

from framework_eval.eval.types import Item

from framework_chi.cascade.client import AgentOutcome
from framework_chi.config import ServiceConfig

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
    """Default escalation: a longer LLM call with the assembled context.

    Returns the constrained-rejudge candidate together with iteration
    count and tool-call log. When the LLM call fails, the agent returns
    the fast-path answer so the cascade still produces output.
    """
    prompt_suffix = (
        "\n\nYou may reference any of the passages above. Reason step by "
        "step internally; output only the final answer."
    )

    ctx = _format_passages(retrieval.passages if retrieval else None)
    user_msg = (
        f"Question type: {item.question_type}\n"
        f"Question: {item.question}\n"
    )
    if item.options:
        user_msg += "Options:\n" + "\n".join(
            f"  {k}. {v}" for k, v in item.options.items()
        ) + "\n"
    if ctx:
        user_msg += f"\nRetrieved context:\n{ctx}{prompt_suffix}"

    messages = [
        {"role": "system",
         "content": (
             "You are a meticulous biomedical reasoning assistant. Use the "
             "retrieved passages and your domain knowledge to answer "
             "carefully."
         )},
        {"role": "user", "content": user_msg},
    ]

    # The default escalation makes a single iteration; ``max_agent_iterations``
    # is the upper bound that downstream subclasses may use, and it is
    # surfaced in the AgentOutcome for telemetry.
    try:
        text, _ = await llm.chat_with_logprob(
            model=services.model_name,
            messages=messages,
            max_tokens=512,
            temperature=0.0,
        )
    except Exception as exc:  # noqa: BLE001
        LOGGER.warning("agent escalation failed: %s; using fast-path answer", exc)
        return AgentOutcome(
            answer=fast_path.answer,
            response_text=fast_path.response_text,
            iterations=0,
            tool_calls=[],
        )

    return AgentOutcome(
        answer=text.strip(),
        response_text=text,
        iterations=1,
        tool_calls=[],
    )
