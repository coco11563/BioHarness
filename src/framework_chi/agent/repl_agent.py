"""REPL-based escalation agent.

The reference implementation in this skeleton is a *single-shot* tool-aware
LLM call: it prompts the LLM to think + answer with up to
``options.max_agent_iterations`` reasoning rounds. The full
``BiomedicalRLMPipeline`` from the paper (Python REPL execution, multi-tool
dispatch, self-revision) is out of scope for this open-source release; the
hook signature here is identical so a downstream user can subclass
``V14CascadeClient`` and swap in the heavier agent.
"""

from __future__ import annotations

import logging
from typing import Any

from framework_eval.eval.types import Item

from framework_chi.cascade.client import AgentOutcome
from framework_chi.config import CascadeOptions, ServiceConfig

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
    options: CascadeOptions,
    item: Item,
    retrieval: Any,
    fast_path: Any,
) -> AgentOutcome:
    """Default escalation: a longer LLM call with the assembled context.

    Returns the constrained-rejudge candidate together with iteration
    count and tool-call log. When the LLM call fails, the agent returns
    the fast-path answer so the cascade still produces output.
    """
    if options.no_tools:
        prompt_suffix = ""
    else:
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
