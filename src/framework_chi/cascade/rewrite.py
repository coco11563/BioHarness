"""Query rewrite for retrieval.

Generates a single rewritten query optimised for dense retrieval before
the cascade fast path. The rewrite step is a {framework}^χ component: it
prompts the LLM to produce a search-engine-ready phrase that surfaces
the kind of biomedical evidence needed to answer the question. For
``yesno`` items, a paired negative-hypothesis query is also produced
(see ``yesno_dual_query``) so the retrieval set covers both polarities.
"""

from __future__ import annotations

import logging
from typing import Any

from framework_eval.eval.types import Item

LOGGER = logging.getLogger(__name__)


_REWRITE_SYSTEM = (
    "You rewrite questions into short retrieval queries. "
    "Return only the query, no quotes, no explanation."
)

_REWRITE_USER = (
    "Rewrite the following question as a concise search query that would "
    "retrieve biomedical evidence relevant to answering it. Keep entities "
    "and modifiers; drop interrogative scaffolding. Max 20 words.\n\n"
    "Question: {question}\n\n"
    "Query:"
)


# Question types where rewrite often hurts more than it helps because the
# answer hinges on an entity lookup (e.g. GeneTuring "official symbol of X")
# rather than evidence synthesis. Mirrors the upstream pipeline's policy.
SKIP_REWRITE_TYPES = frozenset({"factoid"})


async def rewrite_query(
    llm_client: Any, *, model_name: str, item: Item,
) -> str:
    """Return the rewritten retrieval query for ``item``.

    Falls back to the original question text on any failure or for
    question types listed in ``SKIP_REWRITE_TYPES``.
    """
    if item.question_type in SKIP_REWRITE_TYPES:
        return item.question
    messages = [
        {"role": "system", "content": _REWRITE_SYSTEM},
        {"role": "user",   "content": _REWRITE_USER.format(question=item.question)},
    ]
    try:
        text, _ = await llm_client.chat_with_logprob(
            model=model_name, messages=messages,
            max_tokens=64, temperature=0.0,
        )
    except Exception as exc:  # noqa: BLE001
        LOGGER.warning("rewrite_query failed: %s; falling back to original question", exc)
        return item.question
    text = (text or "").strip().strip("\"'`")
    return text or item.question


_NEG_REWRITE_USER = (
    "Rewrite the question as a search query that would retrieve evidence "
    "AGAINST the proposition (the 'no' case). Same constraints as before: "
    "short, no scaffolding, max 20 words. Keep entities.\n\n"
    "Question: {question}\n\n"
    "Negative-hypothesis query:"
)


async def yesno_dual_query(
    llm_client: Any, *, model_name: str, item: Item,
) -> tuple[str, str]:
    """Return ``(positive_query, negative_query)`` for a yesno item.

    The positive query is the standard rewrite; the negative query asks
    the LLM for terms that would surface counter-evidence. The cascade
    retrieves with both and merges the result sets so the constrained
    fast path sees both polarities.
    """
    pos = await rewrite_query(llm_client, model_name=model_name, item=item)
    messages = [
        {"role": "system", "content": _REWRITE_SYSTEM},
        {"role": "user",   "content": _NEG_REWRITE_USER.format(question=item.question)},
    ]
    try:
        text, _ = await llm_client.chat_with_logprob(
            model=model_name, messages=messages,
            max_tokens=64, temperature=0.0,
        )
    except Exception as exc:  # noqa: BLE001
        LOGGER.warning("yesno_dual_query negative pass failed: %s", exc)
        return pos, item.question
    neg = (text or "").strip().strip("\"'`") or item.question
    return pos, neg
