"""Centralized answer extraction for benchmark compliance.

This module is the SINGLE SOURCE OF TRUTH for extracting structured
answers from LLM/RLM responses. Pipeline code must NOT duplicate
extraction logic — it should call extract_answer() exclusively.

Answer formats per question type (from benchmark/code/README.md):
- yesno: "yes", "no", or "maybe"
- mcq: Single letter A-E
- mcq_multi: List of letters, e.g., "['A', 'C']" or "A, C"
- factoid: Short text answer
- list: Comma-separated items
- summary: Text summary
- expression: Comma-separated tissues/expressions

Modes:
- strict: For benchmark / paper results. Rejects ambiguous inputs.
- lenient: For interactive / demo. Allows more heuristic fallback.
"""

import re
from typing import Literal

QuestionType = Literal[
    "yesno", "mcq", "mcq_multi", "factoid", "list", "summary", "expression"
]
Mode = Literal["strict", "lenient"]


# ============================================================================
# FINAL() Unwrapper
# ============================================================================

def _extract_final(text: str) -> str:
    """Extract content from the last FINAL(...) in text.

    Uses O(n) parenthesis counting instead of regex to avoid catastrophic
    backtracking on biomedical text with unmatched parentheses like "(P < 0.05)".
    """
    idx = text.rfind("FINAL")
    if idx < 0:
        return text
    rest = text[idx + 5:]
    # Find opening paren right after FINAL
    paren_start = -1
    for i, ch in enumerate(rest):
        if ch == '(':
            paren_start = i
            break
        elif not ch.isspace():
            return text  # Non-whitespace before ( means not FINAL(...)
    if paren_start < 0:
        return text
    # Count parens to find matching close
    depth = 0
    for i in range(paren_start, len(rest)):
        if rest[i] == '(':
            depth += 1
        elif rest[i] == ')':
            depth -= 1
            if depth == 0:
                content = rest[paren_start + 1:i].strip()
                # Strip outer quotes
                if len(content) >= 2 and content[0] == content[-1] and content[0] in '"\'':
                    content = content[1:-1].strip()
                return content
    # No matching close paren — return original text
    return text


# ============================================================================
# Main Entry Point
# ============================================================================

def extract_answer(
    text: str,
    question_type: QuestionType,
    options: dict[str, str] | None = None,
    mode: Mode = "strict",
) -> str:
    """Extract structured answer from response text.

    This is the ONLY entry point for answer extraction. Pipeline code
    should call this function, not implement its own parsing.

    Args:
        text: Raw response text from LLM/RLM
        question_type: Type of question determining extraction logic
        options: MCQ options dict (e.g., {"A": "...", "B": "..."})
        mode: "strict" (benchmark) or "lenient" (interactive)

    Returns:
        Extracted answer in benchmark-compliant format.
        Empty string on failure (benchmark scores as incorrect).
    """
    if not text:
        return ""

    # Step 1: Unwrap FINAL() and strip outer quotes
    text = _extract_final(text.strip())
    text = text.strip()
    # Strip outer quotes left by RLM responses like '"A"' or "'yes'"
    if len(text) >= 2 and text[0] == text[-1] and text[0] in '"\'':
        text = text[1:-1].strip()

    # Step 2+3: Type-specific extraction (strict, then lenient if enabled)
    extractors = {
        "yesno": _extract_yesno,
        "mcq": _extract_mcq,
        "mcq_multi": _extract_mcq_multi,
        "factoid": _extract_factoid,
        "list": _extract_list,
        "summary": _extract_summary,
        "expression": _extract_expression,
    }

    extractor = extractors.get(question_type)
    if extractor is None:
        return text

    return extractor(text, options, mode)


# ============================================================================
# YesNo Extraction
# ============================================================================

# Word-boundary patterns for polarity detection (no bare substring matching)
_YESNO_AFFIRMATIVE = [
    r"\b(beneficial|effective|useful|helpful|positive|protective)\b",
    r"\b(correct|true|indeed|affirmative|confirmed)\b",
    r"\b(supports?|improves?|enhances?|promotes?)\b",
]
_YESNO_NEGATIVE = [
    r"\b(ineffective|harmful|useless|detrimental|negative)\b",
    r"\b(incorrect|false|disproven|refuted)\b",
    r"\bno\s+(association|effect|benefit|improvement|difference|correlation|evidence)\b",
    r"\b(does\s+not|doesn'?t|isn'?t|aren'?t|cannot|can'?t)\b",
    r"\bnot\s+(associated|effective|beneficial|useful|supported)\b",
]
_YESNO_UNCERTAIN = [
    r"\b(uncertain|unclear|inconclusive|equivocal|ambiguous)\b",
    r"\b(insufficient\s+evidence|limited\s+evidence|mixed\s+evidence)\b",
    r"\b(maybe|possibly|potentially|might)\b",
]


def _extract_yesno(text: str, options: dict | None = None, mode: Mode = "strict") -> str:
    """Extract yes/no/maybe answer.

    Strict mode: only explicit labels and word-boundary phrase patterns.
    No bare "yes" in text / "no" in text matching.
    """
    lower = text.lower().strip()

    # 1. Exact label (entire text is just yes/no/maybe)
    if lower in ("yes", "no", "maybe"):
        return lower

    # 2. First word is an explicit label (strip trailing punctuation so that
    #    verbose answers like "Yes, ..." / "No." resolve correctly)
    first_word = lower.split()[0].strip(".,;:!?\"'") if lower.split() else ""
    if first_word in ("yes", "no", "maybe"):
        return first_word

    # 2b. Explicit "(the) answer is yes/no/maybe" lead-in (still an explicit
    #     label, not a bare-substring guess) — common in verbose generations.
    m = re.match(r"(?:the\s+)?answer\s+is[:\s]+(yes|no|maybe)\b", lower)
    if m:
        return m.group(1)

    # 3. Word-boundary phrase detection (strict: no bare substring)
    # Check uncertainty first (maybe > yes > no priority)
    for pattern in _YESNO_UNCERTAIN:
        if re.search(pattern, lower):
            return "maybe"

    # Count affirmative vs negative signals
    aff_count = sum(1 for p in _YESNO_AFFIRMATIVE if re.search(p, lower))
    neg_count = sum(1 for p in _YESNO_NEGATIVE if re.search(p, lower))

    if aff_count > 0 and neg_count == 0:
        return "yes"
    if neg_count > 0 and aff_count == 0:
        return "no"
    if aff_count > 0 and neg_count > 0:
        # Conflicting signals — strict returns empty, lenient returns maybe
        return "maybe" if mode == "lenient" else ""

    if mode == "lenient":
        # Lenient fallback: bare substring (last resort)
        text_50 = lower[:50]
        if "yes" in text_50:
            return "yes"
        if "no" in text_50:
            return "no"

    # No signal found
    return ""


# ============================================================================
# MCQ Extraction
# ============================================================================

def _extract_mcq(text: str, options: dict | None = None, mode: Mode = "strict") -> str:
    """Extract a single MCQ choice.

    The valid letters come from ``options`` rather than a fixed A-E: MedXpertQA
    has ten (A-J), and hard-coding A-E silently dropped every F-J answer to ""
    (observed: a free-form method answering "I" scored as empty). Datasets with
    <=5 options are unaffected, and with no options we keep the A-E default.
    """
    letters = "".join(sorted({k.strip().upper() for k in options if k and k.strip()})) \
        if options else "ABCDE"
    if not letters:
        letters = "ABCDE"
    cls = f"[{re.escape(letters)}]"
    upper = text.upper().strip()
    text_lower = text.lower()

    # Strict: single letter answer
    if len(upper) == 1 and upper in letters:
        return upper

    # Pattern 1: "answer is X" or "correct answer is X" (case-insensitive)
    match = re.search(rf"(?:answer|choice|option)\s*(?:is|:)?\s*({cls})\b", text, re.IGNORECASE)
    if match:
        return match.group(1).upper()

    # Pattern 2: "X is correct"
    match = re.search(rf"\b({cls})\)?(?:\s*[).\]]?\s*(?:is|appears?|seems?)\s*(?:correct|right|the\s*answer))", text, re.IGNORECASE)
    if match:
        return match.group(1).upper()

    # Pattern 3: "X)" or "X." at line start
    match = re.search(rf"^\s*({cls})\s*[).:]", text, re.MULTILINE)
    if match:
        return match.group(1).upper()

    # Pattern 4: Options text matching
    if options:
        for letter, option_text in options.items():
            if option_text and option_text.lower() in text_lower:
                return letter.upper()

    if mode == "lenient":
        # Lenient: first standalone option letter. Note "I" is also an English
        # pronoun, so this can misfire on prose in 10-option sets; benchmark
        # runs use mode="strict" and never reach here.
        match = re.search(rf"\b({cls})\b", text, re.IGNORECASE)
        if match:
            return match.group(1).upper()

    return ""


# ============================================================================
# MCQ-Multi Extraction
# ============================================================================

# Negation phrases that disqualify a letter
_MCQ_MULTI_NEGATION = re.compile(
    r"(?:not|incorrect|wrong|excluded?|excluding|except|rather\s+than|instead\s+of|"
    r"is\s+(?:not|incorrect|wrong)|(?:isn'?t|aren'?t))\s+(?:option\s+)?([A-Ea-e])"
    r"(?:\s+(?:or|and|,)\s*([A-Ea-e]))*|"
    r"\b([A-Ea-e])\s+(?:is\s+)?(?:not|incorrect|wrong|excluded)",
    re.IGNORECASE,
)


def _extract_mcq_multi(text: str, options: dict | None = None, mode: Mode = "strict") -> str:
    """Extract multiple MCQ choices with negation filtering.

    Strict: only accepts structured formats (comma-separated, JSON array).
    Lenient: also parses natural language, filtering negated options.
    """
    upper = text.upper().strip()

    # Format 1: JSON-like array — ['A', 'C'] or ["A", "C"]
    match = re.search(r"\[(['\"]?[A-E]['\"]?(?:\s*,\s*['\"]?[A-E]['\"]?)*)\]", upper)
    if match:
        letters = re.findall(r"[A-E]", match.group(1))
        if letters:
            return str(sorted(set(letters)))

    # Format 2: Comma-separated letters — "A, C" or "A,C"
    match = re.match(r"^([A-E](?:\s*,\s*[A-E])+)$", upper.strip())
    if match:
        letters = re.findall(r"[A-E]", match.group(0))
        return str(sorted(set(letters)))

    # Format 3: Single letter
    if len(upper) == 1 and upper in "ABCDE":
        return str([upper])

    if mode == "lenient":
        # Lenient: find all standalone letters, then filter negated ones
        all_letters = set(re.findall(r"\b([A-E])\b", upper))

        # Find negated letters
        negated = set()
        for m in _MCQ_MULTI_NEGATION.finditer(text):
            for g in m.groups():
                if g:
                    negated.add(g.upper())

        positive = all_letters - negated
        if positive:
            return str(sorted(positive))

    return ""


# ============================================================================
# Factoid Extraction (pure function — no external calls)
# ============================================================================

def _extract_factoid(text: str, options: dict | None = None, mode: Mode = "strict") -> str:
    """Extract factoid answer (short phrase/entity).

    Pure function — no external API calls, no UniProt lookups.
    """
    # Remove common answer prefixes
    prefixes = [
        r"^(?:the\s+)?answer\s+(?:is|:)\s*",
        r"^based\s+on\s+.*?,\s*",
        r"^according\s+to\s+.*?,\s*",
        r"^(?:it\s+is|this\s+is)\s+",
    ]
    cleaned = text
    for prefix in prefixes:
        cleaned = re.sub(prefix, "", cleaned, flags=re.IGNORECASE)

    # Get first sentence or line
    sentences = re.split(r"[.!?\n]", cleaned)
    if sentences:
        result = sentences[0].strip()
    else:
        result = cleaned.strip()

    return result


# ============================================================================
# List Extraction
# ============================================================================

def _extract_list(text: str, options: dict | None = None, mode: Mode = "strict") -> str:
    """Extract list items as comma-separated string."""
    # Try bullet/numbered list extraction
    bullet_items = re.findall(r"(?:^|\n)\s*[-\u2022*]\s*(.+)", text)
    if bullet_items:
        items = [item.strip().rstrip(".") for item in bullet_items if item.strip()]
        return ", ".join(dict.fromkeys(items))  # dedup, preserve order

    numbered_items = re.findall(r"(?:^|\n)\s*\d+[.)]\s*(.+?)(?:\n|$)", text)
    if numbered_items:
        items = [item.strip().rstrip(".") for item in numbered_items if item.strip()]
        return ", ".join(dict.fromkeys(items))

    # Try inline list (semicolon or comma separated)
    if ";" in text:
        items = [item.strip().rstrip(".") for item in text.split(";") if item.strip()]
        if len(items) > 1:
            return ", ".join(dict.fromkeys(items))

    # Already comma-separated or single item
    return text.strip()


# ============================================================================
# Summary Extraction
# ============================================================================

def _extract_summary(text: str, options: dict | None = None, mode: Mode = "strict") -> str:
    """Extract summary text. Light cleanup only."""
    prefixes = [
        r"^(?:in\s+)?summary[,:]\s*",
        r"^(?:to\s+)?summarize[,:]\s*",
        r"^based\s+on\s+the\s+(?:evidence|literature)[,:]\s*",
    ]
    cleaned = text
    for prefix in prefixes:
        cleaned = re.sub(prefix, "", cleaned, flags=re.IGNORECASE)

    return cleaned.strip()


# ============================================================================
# Expression Extraction
# ============================================================================

def _extract_expression(text: str, options: dict | None = None, mode: Mode = "strict") -> str:
    """Extract expression/tissue list as comma-separated string."""
    prefixes = [
        r"^(?:the\s+)?(?:gene\s+)?(?:is\s+)?expressed\s+in[:\s]*",
        r"^(?:expression\s+)?(?:is\s+)?(?:found|detected)\s+in[:\s]*",
    ]
    cleaned = text
    for prefix in prefixes:
        cleaned = re.sub(prefix, "", cleaned, flags=re.IGNORECASE)

    return _extract_list(cleaned, options, mode)


# ============================================================================
# REPL Helper Function
# ============================================================================

def format_for_repl() -> str:
    """Return docstring for REPL injection."""
    return """extract_answer(text, question_type, options=None, mode="strict")

    Extract structured answer from text for benchmark evaluation.

    Args:
        text: Raw response text
        question_type: One of: yesno, mcq, mcq_multi, factoid, list, summary, expression
        options: Optional MCQ options dict
        mode: "strict" (benchmark) or "lenient" (interactive)

    Returns:
        Benchmark-compliant answer string

    Examples:
        >>> extract_answer("Yes, metformin helps diabetes", "yesno")
        'yes'
        >>> extract_answer("The answer is B", "mcq")
        'B'
        >>> extract_answer("A and C are correct", "mcq_multi")
        "['A', 'C']"
    """
