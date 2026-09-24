"""Entity and Relation extraction module for KG construction.

This module provides LLM-based extraction of:
- Biomedical entities (genes, proteins, diseases, drugs, etc.)
- Relations between entities (treats, causes, inhibits, etc.)

Design: Fail-fast, no fallbacks.
"""

from .entity_extractor import EntityExtractor, ExtractionResult
from .relation_extractor import RelationExtractor, RelationExtractionResult
from .keywords import extract_keywords, extract_query_entities, ExtractedKeywords
from .prompts import (
    ENTITY_EXTRACTION_PROMPT,
    RELATION_EXTRACTION_PROMPT,
    BIOMEDICAL_ENTITY_TYPES,
    BIOMEDICAL_RELATION_TYPES,
)

__all__ = [
    "EntityExtractor",
    "RelationExtractor",
    "ExtractionResult",
    "RelationExtractionResult",
    "extract_keywords",
    "extract_query_entities",
    "ExtractedKeywords",
    "ENTITY_EXTRACTION_PROMPT",
    "RELATION_EXTRACTION_PROMPT",
    "BIOMEDICAL_ENTITY_TYPES",
    "BIOMEDICAL_RELATION_TYPES",
]
