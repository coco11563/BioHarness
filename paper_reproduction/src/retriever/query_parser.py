"""LLM-based query parser for biomedical entity and keyword extraction.

This module provides a QueryParser class that uses an LLM to extract
biomedical entities, keywords, and potential MeSH terms from natural
language queries.
"""

import json
import re
import time

from .models import ParsedQuery
from utils.clients import llm_client


# English prompt for query parsing
QUERY_PARSE_PROMPT = """Analyze this biomedical question and extract:
1. Biomedical entities (drugs, diseases, genes, proteins, pathways, organisms)
2. Keywords for full-text search (important terms that should appear in relevant papers)
3. Potential MeSH descriptors (standardized medical terms)

Question: {query}

Output JSON only, no explanation:
{{"entities": ["entity1", "entity2"], "keywords": ["keyword1", "keyword2"], "mesh_candidates": ["MeSH Term 1", "MeSH Term 2"]}}"""


class QueryParser:
    """LLM-based query parser for biomedical entity/keyword extraction.

    Uses the global llm_client singleton for load-balanced LLM calls.

    Example:
        >>> parser = QueryParser()
        >>> result = await parser.parse("Does metformin help with type 2 diabetes?")
        >>> print(result.entities)
        ['metformin', 'type 2 diabetes']
    """

    def __init__(
        self,
        max_tokens: int = 512,
        temperature: float = 0.1,
    ):
        """Initialize the query parser.

        Args:
            max_tokens: Max tokens for LLM response
            temperature: LLM temperature (low for deterministic output)
        """
        self._max_tokens = max_tokens
        self._temperature = temperature

    async def parse(self, query: str) -> ParsedQuery:
        """Parse a query to extract entities, keywords, and MeSH candidates.

        Args:
            query: Natural language biomedical query

        Returns:
            ParsedQuery with extracted entities, keywords, and mesh_candidates
        """
        start_time = time.perf_counter()

        try:
            content = await llm_client.chat(
                prompt=QUERY_PARSE_PROMPT.format(query=query),
                max_tokens=self._max_tokens,
                temperature=self._temperature,
            )

            parsed = self._extract_json(content)

            return ParsedQuery(
                original_query=query,
                entities=parsed.get("entities", []),
                keywords=parsed.get("keywords", []),
                mesh_candidates=parsed.get("mesh_candidates", []),
                parse_time_ms=(time.perf_counter() - start_time) * 1000,
            )
        except Exception as e:
            # Fallback: simple keyword extraction
            print(f"[QueryParser] LLM parsing failed: {e}, using fallback")
            keywords = self._fallback_extract(query)
            return ParsedQuery(
                original_query=query,
                entities=[],
                keywords=keywords,
                mesh_candidates=[],
                parse_time_ms=(time.perf_counter() - start_time) * 1000,
            )

    def _extract_json(self, content: str) -> dict:
        """Extract JSON object from LLM response.

        Handles cases where LLM includes extra text before/after JSON.

        Args:
            content: Raw LLM response

        Returns:
            Parsed JSON dict, or empty dict if parsing fails
        """
        # Try to find JSON object in response
        json_match = re.search(r'\{[^{}]*\}', content, re.DOTALL)
        if json_match:
            try:
                return json.loads(json_match.group())
            except json.JSONDecodeError:
                pass

        # Try parsing entire content as JSON
        try:
            return json.loads(content)
        except json.JSONDecodeError:
            return {}

    def _fallback_extract(self, query: str) -> list[str]:
        """Simple fallback keyword extraction when LLM fails.

        Extracts words that are:
        - Longer than 2 characters
        - Not common stopwords
        - Likely to be meaningful

        Args:
            query: Natural language query

        Returns:
            List of extracted keywords
        """
        stopwords = {
            'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been',
            'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will',
            'would', 'could', 'should', 'may', 'might', 'can', 'of', 'in',
            'on', 'at', 'to', 'for', 'with', 'by', 'from', 'as', 'and',
            'or', 'but', 'if', 'then', 'than', 'that', 'this', 'these',
            'those', 'what', 'which', 'who', 'whom', 'when', 'where',
            'why', 'how', 'all', 'each', 'every', 'both', 'few', 'more',
            'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only',
            'same', 'so', 'too', 'very', 'just', 'also', 'now', 'here',
            'there', 'about', 'after', 'before', 'between', 'into',
            'through', 'during', 'above', 'below', 'up', 'down', 'out',
            'off', 'over', 'under', 'again', 'further', 'once',
        }

        # Extract words, keeping multi-word entities intact if quoted
        words = []
        query_lower = query.lower()

        # Simple word extraction (improve if needed)
        for word in re.findall(r'\b\w+\b', query_lower):
            if len(word) > 2 and word not in stopwords:
                words.append(word)

        return words


async def test_query_parser():
    """Test the query parser with sample queries."""
    parser = QueryParser()

    # Test queries from PubMedQA
    test_queries = [
        "Are group 2 innate lymphoid cells ( ILC2s ) increased in chronic rhinosinusitis with nasal polyps or eosinophilia?",
        "Does vagus nerve contribute to the development of steatohepatitis and obesity in phosphatidylethanolamine N-methyltransferase deficient mice?",
        "Is methylation of the FGFR2 gene associated with high birth weight centile in humans?",
        "Do tumor-infiltrating immune cell profiles and their change after neoadjuvant chemotherapy predict response and prognosis of breast cancer?",
        "Does metformin help with type 2 diabetes?",
    ]

    print("=" * 60)
    print("Query Parser Test")
    print("=" * 60)

    for i, query in enumerate(test_queries, 1):
        print(f"\n[Query {i}] {query[:80]}...")
        result = await parser.parse(query)
        print(f"  Entities: {result.entities}")
        print(f"  Keywords: {result.keywords}")
        print(f"  MeSH: {result.mesh_candidates}")
        print(f"  Time: {result.parse_time_ms:.0f}ms")


if __name__ == "__main__":
    import asyncio
    asyncio.run(test_query_parser())
