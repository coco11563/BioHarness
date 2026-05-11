"""Constrained answer generation, per-type prompts, and extraction.

Prompt templates and per-type ``max_tokens`` budgets are the
{framework}^χ headline values. The fast-path call always returns
benchmark-shaped output (a single token for yesno/mcq, a short phrase
for factoid, etc.) and the routing logprob is read off the same call.
"""

from __future__ import annotations

import logging
from typing import Any

from framework_eval.eval.types import Item

LOGGER = logging.getLogger(__name__)


# ----------------------------------------------------------------------
# Per-type prompt registry
# ----------------------------------------------------------------------

_BIOMED_SYSTEM_SHORT = "You are a careful biomedical QA assistant."
_BIOMED_SYSTEM_FORMAT = (
    "You are a careful biomedical QA assistant. Respond with the answer "
    "in the requested format only; no explanation, no chain of thought."
)

SYSTEM_PROMPTS: dict[str, str] = {
    # yesno/mcq use the short system prompt — the long form's "no caveats"
    # wording encourages the model to default to 'maybe' on yesno.
    "yesno": _BIOMED_SYSTEM_SHORT,
    "mcq":   _BIOMED_SYSTEM_FORMAT,
    "mcq_multi": (
        "You are a strict answer formatter.\n"
        "Output a comma-separated list of uppercase letters from the "
        "provided options.\n"
        "No punctuation other than commas, no explanation, no extra words."
    ),
    "factoid": (
        "You are a strict answer formatter.\n"
        "Output a single short biomedical entity or phrase (max 6 words).\n"
        "No complete sentence, no explanation, no extra punctuation.\n"
        "If evidence is insufficient, output: unknown"
    ),
    "list": (
        "You are a strict answer formatter.\n"
        "Output a comma-separated list of biomedical entities.\n"
        "No brackets, no bullets, no numbering, no explanation.\n"
        "Each item should be 1-5 words.\n"
        "If no items are supported by evidence, output: none"
    ),
    "summary": (
        "You are a biomedical research summarizer.\n"
        "Write a concise 2-4 sentence summary of the key findings from "
        "the evidence.\n"
        "Focus on the main results and conclusions. No bullet points."
    ),
    "expression": (
        "You are a strict answer formatter.\n"
        "Output a comma-separated list of tissue names where the gene is "
        "expressed.\n"
        "Lowercase tissue names only. No brackets, no explanation."
    ),
}

USER_PROMPTS: dict[str, str] = {
    "yesno": (
        "Question type: yesno\n"
        "Answer with exactly one word: yes, no, or maybe.\n\n"
        "Question: {question}\n\n"
        "Context (top retrieved passages):\n{context}"
    ),
    "mcq": (
        "Question type: mcq\n"
        "Answer with exactly one capital letter (A-E).\n\n"
        "Question: {question}\n"
        "Options:\n{options}\n\n"
        "Context (top retrieved passages):\n{context}\n\n"
        "Answer:"
    ),
    "mcq_multi": (
        "Based on the evidence below, select all correct options.\n\n"
        "Question: {question}\n\n"
        "Options:\n{options}\n\n"
        "Evidence:\n{context}\n\n"
        "Answer (comma-separated letters):"
    ),
    "factoid": (
        "Based on the evidence below, answer the question with a short "
        "phrase.\n\n"
        "Question: {question}\n\n"
        "Evidence:\n{context}\n\n"
        "Answer:"
    ),
    "list": (
        "Based on the evidence below, list all relevant items.\n\n"
        "Question: {question}\n\n"
        "Evidence:\n{context}\n\n"
        "Answer (comma-separated):"
    ),
    "summary": (
        "Based on the evidence below, summarize the key findings.\n\n"
        "Question: {question}\n\n"
        "Evidence:\n{context}\n\n"
        "Summary (2-4 sentences):"
    ),
    "expression": (
        "Based on the evidence below, list the tissues where the gene "
        "is expressed.\n\n"
        "Question: {question}\n\n"
        "Evidence:\n{context}\n\n"
        "Answer (comma-separated tissues):"
    ),
}

MAX_TOKENS: dict[str, int] = {
    "yesno":      4,
    "mcq":        4,
    "mcq_multi":  16,
    "factoid":    32,
    "list":       128,
    "summary":    256,
    "expression": 64,
}


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------


def _format_options(options: dict[str, str] | None) -> str:
    if not options:
        return ""
    lines = []
    for k in sorted(options):
        lines.append(f"{k}. {options[k]}")
    return "\n".join(lines)


def _format_context(passages: list[dict[str, Any]] | None, limit: int = 10) -> str:
    if not passages:
        return "(no retrieved evidence)"
    parts = []
    for i, p in enumerate(passages[:limit], 1):
        title = (p.get("metadata") or {}).get("title", "") or ""
        text = (p.get("text") or "").strip()
        if not text:
            continue
        header = f"[{i}] PMID {p.get('metadata', {}).get('pmid', '?')}"
        if title:
            header += f" — {title[:120]}"
        parts.append(f"{header}\n{text[:1200]}")
    return "\n\n".join(parts) if parts else "(no usable evidence)"


def build_messages(
    item: Item, passages: list[dict[str, Any]] | None,
) -> tuple[str, str]:
    qt = item.question_type if item.question_type in SYSTEM_PROMPTS else "factoid"
    system = SYSTEM_PROMPTS[qt]
    user = USER_PROMPTS[qt].format(
        question=item.question,
        context=_format_context(passages)[:8000],
        options=_format_options(item.options),
    )
    return system, user


# ----------------------------------------------------------------------
# Calls
# ----------------------------------------------------------------------


async def constrained_generate(
    llm_client: Any,
    *,
    model_name: str,
    item: Item,
    passages: list[dict[str, Any]] | None,
) -> tuple[str, float]:
    """Single-call fast path: returns (response_text, mean_logprob)."""
    system, user = build_messages(item, passages)
    messages = [
        {"role": "system", "content": system},
        {"role": "user",   "content": user},
    ]
    max_tokens = MAX_TOKENS.get(item.question_type, 64)
    try:
        text, logprob = await llm_client.chat_with_logprob(
            model=model_name, messages=messages,
            max_tokens=max_tokens, temperature=0.0,
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
    """Re-judge the agent's free-form answer through the same constrained
    prompt, with the agent's text appended as additional evidence."""
    system, user = build_messages(
        item,
        passages=[{"text": f"Agent draft answer:\n{agent_text}", "metadata": {}}],
    )
    messages = [
        {"role": "system", "content": system},
        {"role": "user",   "content": user},
    ]
    max_tokens = MAX_TOKENS.get(item.question_type, 64)
    try:
        text, _ = await llm_client.chat_with_logprob(
            model=model_name, messages=messages,
            max_tokens=max_tokens, temperature=0.0,
        )
    except Exception as exc:  # noqa: BLE001
        LOGGER.warning("rejudge failed: %s", exc)
        return agent_text
    return text


# ----------------------------------------------------------------------
# Extraction
# ----------------------------------------------------------------------


def extract_constrained_answer(
    response: str, question_type: str, options: dict[str, str] | None = None,
) -> str:
    """Mirror the production extractor: strict prompts → simple parsing.

    Falls back to keyword scan for yesno (catches hedging) and first-letter
    scan for mcq.
    """
    import re

    response = (response or "").strip()
    if not response:
        return ""

    if question_type == "yesno":
        low = response.lower()
        first = low.split()[0] if low.split() else ""
        if first in ("yes", "no", "maybe"):
            return first
        if "maybe" in low or "inconclusive" in low:
            return "maybe"
        if "yes" in low:
            return "yes"
        if "no" in low:
            return "no"
        return ""

    if question_type == "mcq":
        upper = response.upper()
        valid = set(options) if options else set("ABCDE")
        for ch in upper:
            if ch in valid:
                return ch
        return ""

    if question_type == "mcq_multi":
        letters = re.findall(r"[A-E]", response.upper())
        return str(sorted(set(letters))) if letters else ""

    if question_type == "factoid":
        first_line = response.split("\n")[0].strip()
        for prefix in ("Answer:", "The answer is", "It is"):
            if first_line.lower().startswith(prefix.lower()):
                first_line = first_line[len(prefix):].strip()
        return first_line[:100]

    if question_type == "list":
        cleaned = response.replace("\n", ", ")
        cleaned = re.sub(r"^\s*[-*\d.)\]]+\s*", "", cleaned, flags=re.MULTILINE)
        return cleaned[:200]

    if question_type == "expression":
        cleaned = response.replace("\n", ", ")
        return cleaned[:200]

    if question_type == "summary":
        return response[:1024]

    return response[:500]
