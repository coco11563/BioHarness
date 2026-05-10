"""Constrained generation + re-judgment.

The fast path uses ``max_tokens=4`` so the answer is forced into the
benchmark-compliant shape immediately, and the routing logprob is read
off of the same call. Re-judgment runs the agent's free-form answer
back through the same constrained pathway so the output schema stays
consistent.
"""

from __future__ import annotations

import logging
from typing import Any

from framework_eval.eval.types import Item

LOGGER = logging.getLogger(__name__)


_TYPE_INSTRUCTION = {
    "yesno":      "Answer with exactly one word: yes, no, or maybe.",
    "mcq":        "Answer with exactly one capital letter (A-E).",
    "mcq_multi":  "Answer with comma-separated capital letters, e.g. 'A, C'.",
    "factoid":    "Answer with a short phrase or entity.",
    "list":       "Answer with a comma-separated list.",
    "summary":    "Answer with one to three concise sentences.",
    "expression": "Answer with a comma-separated list of tissues.",
}


def _format_passages(passages: list[dict[str, Any]] | None, limit: int = 6) -> str:
    if not passages:
        return ""
    parts = []
    for i, p in enumerate(passages[:limit], 1):
        text = (p.get("text") or "").strip()
        if not text:
            continue
        parts.append(f"[{i}] {text[:400]}")
    return "\n\n".join(parts)


def _format_options(item: Item) -> str:
    if not item.options:
        return ""
    return "\n".join(f"  {k}. {v}" for k, v in item.options.items())


async def constrained_generate(
    llm_client: Any,
    *,
    model_name: str,
    item: Item,
    passages: list[dict[str, Any]] | None,
) -> tuple[str, float]:
    """Returns (response_text, mean_logprob)."""
    instr = _TYPE_INSTRUCTION.get(item.question_type, "Answer concisely.")
    ctx = _format_passages(passages)
    user_msg = (
        f"Question type: {item.question_type}\n{instr}\n\n"
        f"Question: {item.question}"
    )
    if item.options:
        user_msg += f"\nOptions:\n{_format_options(item)}"
    if ctx:
        user_msg += f"\n\nContext (top retrieved passages):\n{ctx}"

    messages = [
        {"role": "system", "content": "You are a careful biomedical QA assistant."},
        {"role": "user",   "content": user_msg},
    ]
    try:
        text, logprob = await llm_client.chat_with_logprob(
            model=model_name, messages=messages, max_tokens=4, temperature=0.0,
        )
    except Exception as exc:  # noqa: BLE001
        LOGGER.warning("constrained_generate failed: %s", exc)
        return "", 0.0
    return text, logprob


async def rejudge(
    llm_client: Any,
    *,
    model_name: str,
    item: Item,
    agent_text: str,
) -> str:
    """Re-judge the agent's free-form answer through a constrained call."""
    instr = _TYPE_INSTRUCTION.get(item.question_type, "Answer concisely.")
    user_msg = (
        f"Question type: {item.question_type}\n{instr}\n\n"
        f"Question: {item.question}\n\n"
        f"Candidate answer (free-form): {agent_text}\n\n"
        f"Rewrite the candidate answer in the requested format only."
    )
    if item.options:
        user_msg += f"\nOptions:\n{_format_options(item)}"
    messages = [
        {"role": "system", "content": "You rewrite candidate answers into the requested format."},
        {"role": "user",   "content": user_msg},
    ]
    try:
        text, _ = await llm_client.chat_with_logprob(
            model=model_name, messages=messages, max_tokens=4, temperature=0.0,
        )
    except Exception as exc:  # noqa: BLE001
        LOGGER.warning("rejudge failed: %s", exc)
        return agent_text
    return text
