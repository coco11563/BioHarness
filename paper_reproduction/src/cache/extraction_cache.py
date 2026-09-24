"""Document-level Entity/Relation extraction cache.

Caches entity and relation extraction results per document (PMID).
This enables ~7.5x speedup for RT-KG pipelines by avoiding redundant
LLM calls when the same document is retrieved for multiple questions.

Cache structure:
    .cache/extraction/{extractor_id}/{pmid_prefix}/{pmid}.json

Each file:
    {
        "meta": {
            "pmid": "12345678",
            "doc_hash": "abc123...",
            "extractor_id": "v1_qwen_default",
            "model": "qwen",
            "timestamp": "2026-01-15T10:00:00"
        },
        "data": {
            "entities": [...],
            "relations": [...]
        },
        "status": "success" | "error" | "empty"
    }

Usage:
    cache = ExtractionCache(extractor_id="v1_qwen")

    # Check and load
    result = await cache.load(pmid)
    if result is not None:
        entities, relations = result
    else:
        # Extract and cache
        entities, relations = await extract(doc)
        await cache.set(pmid, title, text, entities, relations)

    # Batch operations
    cached, missing = await cache.partition_pmids(pmid_list)
"""

import hashlib
import json
import os
import tempfile
from dataclasses import dataclass, asdict, field
from datetime import datetime
from pathlib import Path
from typing import Literal

# Import Entity/Relation from kg.base
try:
    from kg.base import Entity, Relation
except ImportError:
    from src.kg.base import Entity, Relation


def compute_doc_hash(text: str) -> str:
    """Compute hash of document text (normalized)."""
    # Normalize: lowercase, collapse whitespace
    normalized = " ".join(text.lower().split())
    return hashlib.sha256(normalized.encode()).hexdigest()[:16]


def compute_extractor_id(
    model: str = "qwen",
    version: str = "v1",
    prompt_variant: str = "default",
) -> str:
    """Compute extractor ID from configuration."""
    return f"{version}_{model}_{prompt_variant}"


@dataclass
class CachedEntity:
    """Cached entity data."""
    id: str
    name: str
    type: str
    description: str
    mentions: int = 1
    source_chunks: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "CachedEntity":
        return cls(
            id=d["id"],
            name=d["name"],
            type=d["type"],
            description=d.get("description", ""),
            mentions=d.get("mentions", 1),
            source_chunks=d.get("source_chunks", []),
        )

    def to_entity(self) -> Entity:
        """Convert to kg.base.Entity."""
        return Entity(
            id=self.id,
            name=self.name,
            type=self.type,
            description=self.description,
            mentions=self.mentions,
            source_chunks=self.source_chunks,
        )


@dataclass
class CachedRelation:
    """Cached relation data."""
    id: str
    source: str
    target: str
    type: str
    description: str = ""
    weight: float = 1.0
    keywords: list[str] = field(default_factory=list)
    source_chunks: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "CachedRelation":
        return cls(
            id=d["id"],
            source=d["source"],
            target=d["target"],
            type=d["type"],
            description=d.get("description", ""),
            weight=d.get("weight", 1.0),
            keywords=d.get("keywords", []),
            source_chunks=d.get("source_chunks", []),
        )

    def to_relation(self) -> Relation:
        """Convert to kg.base.Relation."""
        return Relation(
            id=self.id,
            source=self.source,
            target=self.target,
            type=self.type,
            description=self.description,
            weight=self.weight,
            keywords=self.keywords,
            source_chunks=self.source_chunks,
        )


@dataclass
class ExtractionMeta:
    """Metadata for cached extraction."""
    pmid: str
    doc_hash: str
    extractor_id: str
    model: str
    timestamp: str = ""
    title: str = ""

    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.now().isoformat()

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "ExtractionMeta":
        return cls(
            pmid=d["pmid"],
            doc_hash=d.get("doc_hash", ""),
            extractor_id=d.get("extractor_id", ""),
            model=d.get("model", ""),
            timestamp=d.get("timestamp", ""),
            title=d.get("title", ""),
        )


@dataclass
class ExtractionData:
    """Extraction results data."""
    entities: list[CachedEntity]
    relations: list[CachedRelation]

    def to_dict(self) -> dict:
        return {
            "entities": [e.to_dict() for e in self.entities],
            "relations": [r.to_dict() for r in self.relations],
        }

    @classmethod
    def from_dict(cls, d: dict) -> "ExtractionData":
        return cls(
            entities=[CachedEntity.from_dict(e) for e in d.get("entities", [])],
            relations=[CachedRelation.from_dict(r) for r in d.get("relations", [])],
        )


@dataclass
class ExtractionEntry:
    """Complete extraction cache entry."""
    meta: ExtractionMeta
    data: ExtractionData
    status: Literal["success", "error", "empty"] = "success"
    error_message: str = ""

    def to_dict(self) -> dict:
        result = {
            "meta": self.meta.to_dict(),
            "data": self.data.to_dict(),
            "status": self.status,
        }
        if self.error_message:
            result["error_message"] = self.error_message
        return result

    @classmethod
    def from_dict(cls, d: dict) -> "ExtractionEntry":
        return cls(
            meta=ExtractionMeta.from_dict(d.get("meta", {})),
            data=ExtractionData.from_dict(d.get("data", {})),
            status=d.get("status", "success"),
            error_message=d.get("error_message", ""),
        )


class ExtractionCache:
    """Document-level entity/relation extraction cache.

    Stores extraction results per PMID with extractor versioning.
    Uses sharded directory structure and atomic writes for safety.

    Args:
        cache_dir: Base cache directory
        extractor_id: Extractor version identifier
        model: Model name (for metadata)

    Example:
        cache = ExtractionCache(extractor_id="v1_qwen_default")

        # Load or extract
        result = await cache.load("12345678")
        if result is None:
            entities, relations = await extract(doc)
            await cache.set("12345678", "Title", text, entities, relations)
        else:
            entities, relations = result
    """

    def __init__(
        self,
        cache_dir: str | Path = ".cache/extraction",
        extractor_id: str = "v1_qwen_default",
        model: str = "qwen",
    ):
        self._base_dir = Path(cache_dir)
        self._extractor_id = extractor_id
        self._model = model
        self._cache_dir = self._base_dir / extractor_id
        self._cache_dir.mkdir(parents=True, exist_ok=True)

    @property
    def extractor_id(self) -> str:
        return self._extractor_id

    def _get_cache_path(self, pmid: str) -> Path:
        """Get cache file path for a PMID (sharded by first 2 chars)."""
        prefix = pmid[:2] if len(pmid) >= 2 else "00"
        shard_dir = self._cache_dir / prefix
        shard_dir.mkdir(exist_ok=True)
        return shard_dir / f"{pmid}.json"

    def _atomic_write(self, path: Path, data: dict) -> None:
        """Write JSON atomically using temp file + rename."""
        # Write to temp file in same directory
        fd, tmp_path = tempfile.mkstemp(
            suffix=".tmp",
            prefix=path.stem + "_",
            dir=path.parent,
        )
        try:
            with os.fdopen(fd, "w") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            # Atomic rename
            os.replace(tmp_path, path)
        except Exception:
            # Clean up temp file on error
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
            raise

    async def exists(self, pmid: str) -> bool:
        """Check if extraction cache exists for a PMID."""
        return self._get_cache_path(pmid).exists()

    async def load(self, pmid: str) -> tuple[list[Entity], list[Relation]] | None:
        """Load cached entities and relations for a PMID.

        Returns:
            Tuple of (entities, relations) or None if not cached/error
        """
        cache_path = self._get_cache_path(pmid)
        if not cache_path.exists():
            return None

        try:
            with open(cache_path) as f:
                data = json.load(f)
            entry = ExtractionEntry.from_dict(data)

            # Check extractor compatibility
            if entry.meta.extractor_id != self._extractor_id:
                return None

            # Skip error entries
            if entry.status == "error":
                return None

            # Return empty lists for empty status
            if entry.status == "empty":
                return [], []

            entities = [e.to_entity() for e in entry.data.entities]
            relations = [r.to_relation() for r in entry.data.relations]
            return entities, relations

        except (json.JSONDecodeError, KeyError, TypeError):
            return None

    async def load_raw(self, pmid: str) -> ExtractionEntry | None:
        """Load raw cache entry (including error status)."""
        cache_path = self._get_cache_path(pmid)
        if not cache_path.exists():
            return None

        try:
            with open(cache_path) as f:
                data = json.load(f)
            return ExtractionEntry.from_dict(data)
        except (json.JSONDecodeError, KeyError, TypeError):
            return None

    async def set(
        self,
        pmid: str,
        title: str,
        text: str,
        entities: list[Entity],
        relations: list[Relation],
    ) -> None:
        """Cache extraction results for a PMID.

        Args:
            pmid: Document PMID
            title: Document title
            text: Document text (for hash computation)
            entities: Extracted entities
            relations: Extracted relations
        """
        doc_hash = compute_doc_hash(text)

        cached_entities = [
            CachedEntity(
                id=e.id,
                name=e.name,
                type=e.type,
                description=e.description,
                mentions=e.mentions,
                source_chunks=e.source_chunks,
            )
            for e in entities
        ]

        cached_relations = [
            CachedRelation(
                id=r.id,
                source=r.source,
                target=r.target,
                type=r.type,
                description=r.description,
                weight=r.weight,
                keywords=r.keywords,
                source_chunks=r.source_chunks,
            )
            for r in relations
        ]

        status = "success" if entities or relations else "empty"

        entry = ExtractionEntry(
            meta=ExtractionMeta(
                pmid=pmid,
                doc_hash=doc_hash,
                extractor_id=self._extractor_id,
                model=self._model,
                title=title,
            ),
            data=ExtractionData(
                entities=cached_entities,
                relations=cached_relations,
            ),
            status=status,
        )

        cache_path = self._get_cache_path(pmid)
        self._atomic_write(cache_path, entry.to_dict())

    async def set_error(
        self,
        pmid: str,
        title: str,
        text: str,
        error_message: str,
    ) -> None:
        """Cache extraction error for a PMID (to avoid retry)."""
        doc_hash = compute_doc_hash(text)

        entry = ExtractionEntry(
            meta=ExtractionMeta(
                pmid=pmid,
                doc_hash=doc_hash,
                extractor_id=self._extractor_id,
                model=self._model,
                title=title,
            ),
            data=ExtractionData(entities=[], relations=[]),
            status="error",
            error_message=error_message[:500],  # Truncate long errors
        )

        cache_path = self._get_cache_path(pmid)
        self._atomic_write(cache_path, entry.to_dict())

    async def partition_pmids(
        self,
        pmids: list[str],
    ) -> tuple[list[str], list[str]]:
        """Partition PMIDs into cached and missing.

        Args:
            pmids: List of PMIDs to check

        Returns:
            Tuple of (cached_pmids, missing_pmids)
        """
        cached = []
        missing = []

        for pmid in pmids:
            if await self.exists(pmid):
                cached.append(pmid)
            else:
                missing.append(pmid)

        return cached, missing

    async def load_batch(
        self,
        pmids: list[str],
    ) -> dict[str, tuple[list[Entity], list[Relation]]]:
        """Load cached extractions for multiple PMIDs.

        Args:
            pmids: List of PMIDs to load

        Returns:
            Dict mapping pmid -> (entities, relations) for cached PMIDs
        """
        results = {}
        for pmid in pmids:
            data = await self.load(pmid)
            if data is not None:
                results[pmid] = data
        return results

    def get_stats(self) -> dict:
        """Get cache statistics."""
        total_files = 0
        success_count = 0
        error_count = 0
        empty_count = 0
        total_entities = 0
        total_relations = 0

        for shard_dir in self._cache_dir.iterdir():
            if shard_dir.is_dir():
                for cache_file in shard_dir.glob("*.json"):
                    total_files += 1
                    try:
                        with open(cache_file) as f:
                            data = json.load(f)
                        status = data.get("status", "success")
                        if status == "success":
                            success_count += 1
                            total_entities += len(data.get("data", {}).get("entities", []))
                            total_relations += len(data.get("data", {}).get("relations", []))
                        elif status == "error":
                            error_count += 1
                        elif status == "empty":
                            empty_count += 1
                    except:
                        pass

        return {
            "extractor_id": self._extractor_id,
            "total_documents": total_files,
            "success": success_count,
            "error": error_count,
            "empty": empty_count,
            "total_entities": total_entities,
            "total_relations": total_relations,
            "cache_dir": str(self._cache_dir),
        }

    def clear(self) -> int:
        """Clear all cached extractions for current extractor. Returns count."""
        count = 0
        for shard_dir in self._cache_dir.iterdir():
            if shard_dir.is_dir():
                for cache_file in shard_dir.glob("*.json"):
                    cache_file.unlink()
                    count += 1
        return count
