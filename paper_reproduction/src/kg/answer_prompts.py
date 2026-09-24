"""Constrained answer prompts for RT-KG benchmark.

Forces LLM to output structured, parseable answers:
- yesno: "yes" or "no"
- mcq: single letter "A", "B", "C", "D"
- factoid: short entity/phrase
- list: comma-separated items
"""

from __future__ import annotations

import os

# Character cap on the evidence block handed to the constrained answerer. 8000 was
# sized for abstracts; with 20 full-text chunks the evidence block is ~25k chars, so
# the rejudge that produces the FINAL answer saw ~6 of 20 documents while stage 1 saw
# all of them uncapped. XC_ANSWER_CONTEXT_CHARS lifts it; default unchanged.
_ANSWER_CONTEXT_CHARS = int(os.environ.get("XC_ANSWER_CONTEXT_CHARS", "8000"))


def format_options(options: dict[str, str] | None) -> str:
    """Format MCQ options as A. option_text lines."""
    if not options:
        return ""
    lines = []
    for k in sorted(options.keys()):
        lines.append(f"{k}. {options[k]}")
    return "\n".join(lines)


# System prompts enforce strict output format
SYSTEM_PROMPTS = {
    "yesno": (
        "You are a strict answer formatter.\n"
        "Output EXACTLY one word: yes or no.\n"
        "Commit to yes or no based on the weight of evidence.\n"
        "Lowercase only. No punctuation, no explanation, no extra words."
    ),
    "mcq": (
        "You are a strict answer formatter.\n"
        "Output EXACTLY one uppercase letter from the provided options (A, B, C, D, etc.).\n"
        "No punctuation, no explanation, no extra words."
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
        "Write a concise 2-4 sentence summary of the key findings from the evidence.\n"
        "Focus on the main results and conclusions. No bullet points."
    ),
}

# User prompts provide question and evidence
USER_PROMPTS = {
    "yesno": (
        "Based on the evidence below, answer the question.\n\n"
        "Question: {question}\n\n"
        "Evidence:\n{context}\n\n"
        "Answer (yes or no):"
    ),
    "mcq": (
        "Based on the evidence below, select the correct option.\n\n"
        "Question: {question}\n\n"
        "Options:\n{options}\n\n"
        "Evidence:\n{context}\n\n"
        "Answer (single letter):"
    ),
    "factoid": (
        "Based on the evidence below, answer the question with a short phrase.\n\n"
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
}

# Recommended max_tokens for each question type
MAX_TOKENS = {
    "yesno": 4,
    "mcq": 4,
    "factoid": 32,
    "list": 128,
    "summary": 256,
}


def build_answer_messages(
    question_type: str,
    question: str,
    context: str,
    options: dict[str, str] | None = None,
) -> tuple[str, str]:
    """Build system and user messages for constrained answer generation.

    Args:
        question_type: One of "yesno", "mcq", "factoid", "list"
        question: The question text
        context: Evidence text (KG query results, retrieved docs, etc.)
        options: MCQ options dict (e.g., {"A": "option1", "B": "option2"})

    Returns:
        Tuple of (system_prompt, user_prompt)
    """
    # Normalize question type
    qt = question_type if question_type in SYSTEM_PROMPTS else "factoid"

    system = SYSTEM_PROMPTS[qt]
    if qt == "mcq" and options:
        # Enumerate the letters this item actually offers. The static prompt
        # listed "(A, B, C, D, etc.)", which anchors the model to the first
        # four options: on MedXpertQA (ten options, A-J) predictions were
        # skewed 1.65:1 toward A-E while gold is uniform, and accuracy when
        # the gold letter was J ran ~18 pp below gold=D.
        letters = ", ".join(sorted(options))
        system = system.replace(
            "one uppercase letter from the provided options (A, B, C, D, etc.)",
            f"one uppercase letter from the provided options ({letters})",
        )
    user = USER_PROMPTS[qt].format(
        question=question,
        context=context[:_ANSWER_CONTEXT_CHARS],  # Truncate long context
        options=format_options(options),
    )

    return system, user


def get_max_tokens(question_type: str) -> int:
    """Get recommended max_tokens for question type."""
    return MAX_TOKENS.get(question_type, 64)


def extract_constrained_answer(
    response: str,
    question_type: str,
    options: dict[str, str] | None = None,
) -> str:
    """Extract answer from constrained LLM response.

    Since we use strict prompts, extraction is simple:
    - yesno: normalize to "yes" or "no"
    - mcq: extract first letter A-Z
    - factoid/list: return cleaned response

    Args:
        response: LLM response text
        question_type: Question type
        options: MCQ options (for validation)

    Returns:
        Extracted answer string
    """
    response = response.strip()

    if question_type == "yesno":
        response_lower = response.lower()
        # Check first word first (most reliable)
        first_word = response_lower.split()[0] if response_lower.split() else ""
        if first_word in ("yes", "no", "maybe"):
            return first_word
        # Keyword search — maybe before yes/no to catch hedging
        if "maybe" in response_lower or "inconclusive" in response_lower:
            return "maybe"
        if "yes" in response_lower:
            return "yes"
        if "no" in response_lower:
            return "no"
        return ""  # Let evaluator score as incorrect, don't guess

    elif question_type == "mcq":
        # Extract first letter that's a valid option
        response_upper = response.upper()
        valid_keys = set(options.keys()) if options else set("ABCDE")

        # First, try to find the letter directly
        for char in response_upper:
            if char in valid_keys:
                return char

        # Fallback to first option
        return list(valid_keys)[0] if valid_keys else "A"

    elif question_type == "factoid":
        # Return first line, cleaned
        first_line = response.split("\n")[0].strip()
        # Remove common prefixes
        for prefix in ["Answer:", "The answer is", "It is"]:
            if first_line.lower().startswith(prefix.lower()):
                first_line = first_line[len(prefix):].strip()
        return first_line[:100]

    elif question_type == "list":
        # Return comma-separated items, cleaned
        response = response.replace("\n", ", ")
        # Remove bullets/numbers
        import re
        response = re.sub(r"^\s*[-*\d.)\]]+\s*", "", response, flags=re.MULTILINE)
        return response[:200]

    else:
        return response[:500]
