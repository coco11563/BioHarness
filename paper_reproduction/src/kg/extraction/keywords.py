"""Keyword extraction for KG query processing.

Extracts high-level (concepts, themes) and low-level (entities, terms) keywords
from queries to guide entity/relation search.

Reference: LightRAG operate.py:3257-3364
"""

import json
import re
from dataclasses import dataclass

from .prompts import KEYWORDS_EXTRACTION_PROMPT

try:
    from utils.clients import llm_client
except ImportError:
    from src.utils.clients import llm_client


@dataclass
class ExtractedKeywords:
    """Keywords extracted from a query."""
    high_level: list[str]  # Concepts, themes, question types
    low_level: list[str]   # Entities, technical terms, proper nouns
    raw_response: str = ""


def _parse_json_response(response: str) -> dict:
    """Parse JSON from LLM response."""
    # Try to extract JSON from markdown code blocks
    json_match = re.search(r"```(?:json)?\s*([\s\S]*?)```", response)
    if json_match:
        response = json_match.group(1)

    response = response.strip()
    if not response:
        return {}

    try:
        return json.loads(response)
    except json.JSONDecodeError:
        # Try to find object in response
        obj_match = re.search(r"\{[\s\S]*\}", response)
        if obj_match:
            try:
                return json.loads(obj_match.group())
            except json.JSONDecodeError:
                pass
        return {}


async def extract_keywords(query: str) -> ExtractedKeywords:
    """Extract high-level and low-level keywords from a query.

    High-level keywords capture:
    - Overarching concepts or themes
    - User's core intent
    - Subject area or question type

    Low-level keywords capture:
    - Specific entities
    - Proper nouns
    - Technical terms
    - Concrete items

    Args:
        query: Natural language query

    Returns:
        ExtractedKeywords with high_level and low_level lists

    Example:
        >>> kw = await extract_keywords("How does metformin treat diabetes?")
        >>> print(kw.high_level)  # ["treatment mechanism", "drug action"]
        >>> print(kw.low_level)   # ["metformin", "diabetes"]
    """
    prompt = KEYWORDS_EXTRACTION_PROMPT.format(query=query)

    response = await llm_client.chat(prompt, max_tokens=256, temperature=0.1)

    data = _parse_json_response(response)

    high_level = data.get("high_level_keywords", [])
    low_level = data.get("low_level_keywords", [])

    # Ensure we have lists
    if isinstance(high_level, str):
        high_level = [k.strip() for k in high_level.split(",")]
    if isinstance(low_level, str):
        low_level = [k.strip() for k in low_level.split(",")]

    return ExtractedKeywords(
        high_level=high_level[:10],
        low_level=low_level[:10],
        raw_response=response,
    )


async def extract_query_entities(query: str) -> list[str]:
    """Extract entity mentions from a query for entity search.

    Simplified extraction focusing on entity names.

    Args:
        query: Natural language query

    Returns:
        List of entity name strings
    """
    keywords = await extract_keywords(query)
    # Low-level keywords are typically entities
    return keywords.low_level
