"""Relation extraction using LLM.

Design: Fail-fast, no fallbacks.
"""

import hashlib
import json
import re
from dataclasses import dataclass, field

from ..base import Entity, Relation, RelationType
from .prompts import RELATION_EXTRACTION_PROMPT, BIOMEDICAL_RELATION_TYPES

try:
    from config import get_config
    from utils.clients import llm_client
except ImportError:
    from src.config import get_config
    from src.utils.clients import llm_client


@dataclass
class RelationExtractionResult:
    """Result of relation extraction from a chunk."""
    chunk_id: str
    relations: list[Relation] = field(default_factory=list)
    raw_response: str = ""
    error: str | None = None


def _generate_relation_id(source: str, target: str, rel_type: str) -> str:
    """Generate deterministic relation ID."""
    key = f"{source.lower().strip()}:{target.lower().strip()}:{rel_type.lower()}"
    return hashlib.sha256(key.encode()).hexdigest()[:16]


def _normalize_relation_type(raw_type: str) -> str:
    """Normalize relation type to valid RelationType value."""
    raw_lower = raw_type.lower().strip().replace(" ", "_").replace("-", "_")

    # Direct matches
    if raw_lower in [r.value for r in RelationType]:
        return raw_lower

    # Common aliases
    aliases = {
        "treatment": RelationType.TREATS.value,
        "treats_disease": RelationType.TREATS.value,
        "caused_by": RelationType.CAUSES.value,
        "associated": RelationType.ASSOCIATED_WITH.value,
        "association": RelationType.ASSOCIATED_WITH.value,
        "inhibition": RelationType.INHIBITS.value,
        "blocks": RelationType.INHIBITS.value,
        "suppresses": RelationType.INHIBITS.value,
        "activation": RelationType.ACTIVATES.value,
        "upregulates": RelationType.ACTIVATES.value,
        "induces": RelationType.ACTIVATES.value,
        "regulation": RelationType.REGULATES.value,
        "modulates": RelationType.REGULATES.value,
        "expression": RelationType.EXPRESSED_IN.value,
        "expressed": RelationType.EXPRESSED_IN.value,
        "interaction": RelationType.INTERACTS_WITH.value,
        "binds": RelationType.INTERACTS_WITH.value,
        "binds_to": RelationType.INTERACTS_WITH.value,
        "component_of": RelationType.PART_OF.value,
        "member_of": RelationType.PART_OF.value,
        "location": RelationType.LOCATED_IN.value,
        "found_in": RelationType.LOCATED_IN.value,
        "relation": RelationType.RELATED_TO.value,
        "related": RelationType.RELATED_TO.value,
    }

    return aliases.get(raw_lower, RelationType.OTHER.value)


def _parse_json_response(response: str) -> list[dict]:
    """Parse JSON from LLM response, handling markdown code blocks."""
    json_match = re.search(r"```(?:json)?\s*([\s\S]*?)```", response)
    if json_match:
        response = json_match.group(1)

    response = response.strip()
    if not response:
        return []

    try:
        parsed = json.loads(response)
        if isinstance(parsed, list):
            return parsed
        elif isinstance(parsed, dict) and "relations" in parsed:
            return parsed["relations"]
        elif isinstance(parsed, dict) and "relationships" in parsed:
            return parsed["relationships"]
        else:
            return [parsed]
    except json.JSONDecodeError:
        array_match = re.search(r"\[[\s\S]*\]", response)
        if array_match:
            try:
                return json.loads(array_match.group())
            except json.JSONDecodeError:
                pass
        raise


class RelationExtractor:
    """Extract relations between entities using LLM.

    Given a set of entities and text, extracts relationships between them.

    Example:
        >>> extractor = RelationExtractor()
        >>> entities = [Entity(id="1", name="Metformin", type="drug", description="...")]
        >>> result = await extractor.extract(
        ...     text="Metformin treats type 2 diabetes.",
        ...     entities=entities,
        ...     chunk_id="chunk_001"
        ... )
    """

    def __init__(
        self,
        relation_types: list[str] | None = None,
        max_relations_per_chunk: int = 100,
    ):
        """Initialize relation extractor.

        Args:
            relation_types: List of relation types to extract
            max_relations_per_chunk: Maximum relations to extract per chunk
        """
        self._relation_types = relation_types or BIOMEDICAL_RELATION_TYPES
        self._max_relations = max_relations_per_chunk
        self._config = get_config()

    async def extract(
        self,
        text: str,
        entities: list[Entity],
        chunk_id: str,
    ) -> RelationExtractionResult:
        """Extract relations from text given known entities.

        Args:
            text: Text content to extract from
            entities: List of entities found in this text
            chunk_id: ID of the source chunk

        Returns:
            RelationExtractionResult with extracted relations
        """
        if len(entities) < 2:
            return RelationExtractionResult(chunk_id=chunk_id)

        # Build entity list for prompt
        entity_list = "\n".join(
            f"- {e.name} ({e.type}): {e.description[:100]}" for e in entities[:30]
        )

        prompt = RELATION_EXTRACTION_PROMPT.format(
            entities=entity_list,
            relation_types=", ".join(self._relation_types),
            text=text[:8000],
        )

        response = await llm_client.chat(prompt)

        try:
            raw_relations = _parse_json_response(response)
        except json.JSONDecodeError as e:
            return RelationExtractionResult(
                chunk_id=chunk_id,
                raw_response=response,
                error=f"JSON parse error: {e}",
            )

        # Build entity name lookup for validation
        entity_names = {e.name.lower(): e.name for e in entities}

        relations = []
        for raw in raw_relations[:self._max_relations]:
            if not isinstance(raw, dict):
                continue

            source = raw.get("source", "").strip()
            target = raw.get("target", "").strip()

            if not source or not target:
                continue

            # Validate entities exist (case-insensitive)
            source_normalized = entity_names.get(source.lower())
            target_normalized = entity_names.get(target.lower())

            if not source_normalized or not target_normalized:
                continue

            rel_type = _normalize_relation_type(raw.get("type", "related_to"))
            description = raw.get("description", "")
            keywords = raw.get("keywords", [])

            if isinstance(keywords, str):
                keywords = [k.strip() for k in keywords.split(",")]

            relation = Relation(
                id=_generate_relation_id(source_normalized, target_normalized, rel_type),
                source=source_normalized,
                target=target_normalized,
                type=rel_type,
                description=description,
                keywords=keywords[:5],
                source_chunks=[chunk_id],
            )
            relations.append(relation)

        return RelationExtractionResult(
            chunk_id=chunk_id,
            relations=relations,
            raw_response=response,
        )

    async def extract_batch(
        self,
        chunks: list[tuple[str, str, list[Entity]]],
        concurrency: int = 5,
    ) -> list[RelationExtractionResult]:
        """Extract relations from multiple chunks.

        Args:
            chunks: List of (chunk_id, text, entities) tuples
            concurrency: Number of concurrent extractions

        Returns:
            List of RelationExtractionResult for each chunk
        """
        import asyncio

        semaphore = asyncio.Semaphore(concurrency)

        async def extract_with_semaphore(chunk_id: str, text: str, entities: list[Entity]):
            async with semaphore:
                return await self.extract(text, entities, chunk_id)

        tasks = [extract_with_semaphore(cid, txt, ents) for cid, txt, ents in chunks]
        return await asyncio.gather(*tasks)
