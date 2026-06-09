"""Retrieval augmentation primitives.

Two distinct retrieval augmentations are exposed:

* :func:`pseudo_answer_text` — the LLM drafts a candidate answer paragraph
  for the question, and the embedding of *that paragraph* is used as a
  retrieval query. This catches concept-level passages that the literal
  question text would miss.
* :func:`negative_evidence_query` — appends literal counter-evidence
  terms to the question, used only for yesno to surface "no" passages
  that the positive retrieval would not return.

Neither augmentation is run for question types whose answer is an
entity lookup (e.g. ``factoid``); see :data:`SKIP_REWRITE_TYPES`.
"""

from __future__ import annotations

import logging
from typing import Any

from framework_eval.eval.types import Item

LOGGER = logging.getLogger(__name__)


# Question types where the answer is an entity lookup and rewrite
# augmentation typically hurts. Matches the upstream policy.
SKIP_REWRITE_TYPES = frozenset({"factoid"})

# Fixed counter-evidence terms appended verbatim to yesno questions.
# These terms are deterministic so we do not pay an extra LLM call per item.
NEGATIVE_EVIDENCE_TERMS = (
    "no effect OR not effective OR no association OR failed OR no benefit"
)


_PSEUDO_ANSWER_SYSTEM = (
    "You generate a short hypothetical scientific abstract that would "
    "directly answer the question. Output the abstract as if it were a "
    "single paragraph from a published paper. No quotes, no markdown."
)

_PSEUDO_ANSWER_USER = (
    "Question: {question}\n\n"
    "Write a 3-5 sentence hypothetical abstract that would answer this "
    "question. State the finding directly and cite typical biomedical "
    "terminology. Do not refuse; this is a retrieval-augmentation step, "
    "not a final answer."
)


async def pseudo_answer_text(
    llm_client: Any, *, model_name: str, item: Item,
) -> str:
    """Generate a hypothetical answer paragraph for retrieval.

    The output is embedded and used as a *second* dense retrieval query
    alongside the literal question. Falls back to the question text on
    failure.
    """
    if item.question_type in SKIP_REWRITE_TYPES:
        return item.question
    messages = [
        {"role": "system", "content": _PSEUDO_ANSWER_SYSTEM},
        {"role": "user",   "content": _PSEUDO_ANSWER_USER.format(question=item.question)},
    ]
    try:
        text = await llm_client.chat(
            model=model_name, messages=messages,
            max_tokens=256, temperature=0.1,
        )
    except Exception as exc:
        LOGGER.warning("pseudo_answer_text failed: %s; using question text", exc)
        return item.question
    text = (text or "").strip()
    return text or item.question


def negative_evidence_query(item: Item) -> str:
    """Return the literal negative-evidence retrieval query for yesno
    items; empty string otherwise."""
    if item.question_type != "yesno":
        return ""
    return f"{item.question} {NEGATIVE_EVIDENCE_TERMS}"


# Back-compat names kept so external callers do not break if they
# imported the old surface (returns the verbatim question; query rewrite
# is now done implicitly via pseudo_answer_text instead).
async def rewrite_query(_llm_client: Any, *, model_name: str, item: Item) -> str:
    return item.question
