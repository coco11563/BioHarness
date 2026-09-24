"""Entity extraction using LLM.

Design: Fail-fast, no fallbacks.
"""

import hashlib
import json
import re
from dataclasses import dataclass, field

from ..base import Entity, EntityType
from .prompts import ENTITY_EXTRACTION_PROMPT, BIOMEDICAL_ENTITY_TYPES

# Use absolute imports for config/clients to avoid import issues
try:
    from config import get_config
    from utils.clients import llm_client
except ImportError:
    from src.config import get_config
    from src.utils.clients import llm_client


@dataclass
class ExtractionResult:
    """Result of entity/relation extraction from a chunk."""
    chunk_id: str
    entities: list[Entity] = field(default_factory=list)
    raw_response: str = ""
    error: str | None = None


def _generate_entity_id(name: str, entity_type: str) -> str:
    """Generate deterministic entity ID from name and type."""
    key = f"{name.lower().strip()}:{entity_type.lower()}"
    return hashlib.sha256(key.encode()).hexdigest()[:16]


def _normalize_entity_type(raw_type: str) -> str:
    """Normalize entity type to valid EntityType value."""
    raw_lower = raw_type.lower().strip()

    # Direct matches
    if raw_lower in [e.value for e in EntityType]:
        return raw_lower

    # Common aliases
    aliases = {
        "genes": EntityType.GENE.value,
        "proteins": EntityType.PROTEIN.value,
        "diseases": EntityType.DISEASE.value,
        "drugs": EntityType.DRUG.value,
        "medications": EntityType.DRUG.value,
        "chemicals": EntityType.CHEMICAL.value,
        "compounds": EntityType.CHEMICAL.value,
        "pathways": EntityType.PATHWAY.value,
        "cells": EntityType.CELL_TYPE.value,
        "cell_types": EntityType.CELL_TYPE.value,
        "organisms": EntityType.ORGANISM.value,
        "species": EntityType.ORGANISM.value,
        "anatomies": EntityType.ANATOMY.value,
        "anatomy": EntityType.ANATOMY.value,
        "processes": EntityType.PROCESS.value,
        "biological_processes": EntityType.PROCESS.value,
        "methods": EntityType.METHOD.value,
        "techniques": EntityType.METHOD.value,
        "concepts": EntityType.CONCEPT.value,
    }

    return aliases.get(raw_lower, EntityType.OTHER.value)


def _parse_json_response(response: str) -> list[dict]:
    """Parse JSON from LLM response, handling markdown code blocks."""
    # Try to extract JSON from markdown code blocks
    json_match = re.search(r"```(?:json)?\s*([\s\S]*?)```", response)
    if json_match:
        response = json_match.group(1)

    # Clean up common issues
    response = response.strip()
    if not response:
        return []

    try:
        parsed = json.loads(response)
        if isinstance(parsed, list):
            return parsed
        elif isinstance(parsed, dict) and "entities" in parsed:
            return parsed["entities"]
        else:
            return [parsed]
    except json.JSONDecodeError:
        # Try to find array in response
        array_match = re.search(r"\[[\s\S]*\]", response)
        if array_match:
            try:
                return json.loads(array_match.group())
            except json.JSONDecodeError:
                pass
        raise


class EntityExtractor:
    """Extract biomedical entities from text using LLM.

    Uses JSON output format for reliable parsing.

    Example:
        >>> extractor = EntityExtractor()
        >>> result = await extractor.extract(
        ...     text="Metformin treats type 2 diabetes by activating AMPK.",
        ...     chunk_id="chunk_001"
        ... )
        >>> print([e.name for e in result.entities])
        ['Metformin', 'Type 2 Diabetes', 'AMPK']
    """

    def __init__(
        self,
        entity_types: list[str] | None = None,
        max_entities_per_chunk: int = 50,
    ):
        """Initialize entity extractor.

        Args:
            entity_types: List of entity types to extract (default: biomedical types)
            max_entities_per_chunk: Maximum entities to extract per chunk
        """
        self._entity_types = entity_types or BIOMEDICAL_ENTITY_TYPES
        self._max_entities = max_entities_per_chunk
        self._config = get_config()

    async def extract(
        self,
        text: str,
        chunk_id: str,
    ) -> ExtractionResult:
        """Extract entities from a single text chunk.

        Args:
            text: Text content to extract from
            chunk_id: ID of the source chunk

        Returns:
            ExtractionResult with extracted entities
        """
        prompt = ENTITY_EXTRACTION_PROMPT.format(
            entity_types=", ".join(self._entity_types),
            text=text[:8000],  # Truncate to avoid context limits
        )

        response = await llm_client.chat(prompt)

        try:
            raw_entities = _parse_json_response(response)
        except json.JSONDecodeError as e:
            return ExtractionResult(
                chunk_id=chunk_id,
                raw_response=response,
                error=f"JSON parse error: {e}",
            )

        entities = []
        for raw in raw_entities[:self._max_entities]:
            if not isinstance(raw, dict):
                continue

            name = raw.get("name", "").strip()
            if not name:
                continue

            entity_type = _normalize_entity_type(raw.get("type", "other"))
            description = raw.get("description", "")

            entity = Entity(
                id=_generate_entity_id(name, entity_type),
                name=name,
                type=entity_type,
                description=description,
                mentions=1,
                source_chunks=[chunk_id],
            )
            entities.append(entity)

        return ExtractionResult(
            chunk_id=chunk_id,
            entities=entities,
            raw_response=response,
        )

    async def extract_batch(
        self,
        chunks: list[tuple[str, str]],
        concurrency: int = 5,
    ) -> list[ExtractionResult]:
        """Extract entities from multiple chunks.

        Args:
            chunks: List of (chunk_id, text) tuples
            concurrency: Number of concurrent extractions

        Returns:
            List of ExtractionResult for each chunk
        """
        import asyncio

        semaphore = asyncio.Semaphore(concurrency)

        async def extract_with_semaphore(chunk_id: str, text: str):
            async with semaphore:
                return await self.extract(text, chunk_id)

        tasks = [extract_with_semaphore(cid, txt) for cid, txt in chunks]
        return await asyncio.gather(*tasks)
