"""Artifact persistence for KG data.

Supports JSON and Parquet formats for different use cases:
- JSON: Human-readable, good for inspection and debugging
- Parquet: Efficient columnar storage for large datasets
"""

import json
from pathlib import Path
from typing import Any

from .graph_store import GraphStore


class ArtifactStore:
    """File-based persistence for KG artifacts.

    Stores KG data in organized directory structure:
        {base_dir}/
            graph.json          # Full graph data
            entities.json       # Entity list
            relations.json      # Relation list
            communities.json    # Community reports
            metadata.json       # Stats and config

    Example:
        >>> store = ArtifactStore("data/kg/lightrag")
        >>> await store.save(graph_store)
        >>> loaded = await store.load()
    """

    def __init__(self, base_dir: str | Path):
        """Initialize artifact store.

        Args:
            base_dir: Base directory for artifacts
        """
        self._base_dir = Path(base_dir)

    @property
    def base_dir(self) -> Path:
        """Get base directory path."""
        return self._base_dir

    def _ensure_dir(self) -> None:
        """Ensure base directory exists."""
        self._base_dir.mkdir(parents=True, exist_ok=True)

    async def save(
        self,
        graph_store: GraphStore,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Save graph store to disk.

        Args:
            graph_store: GraphStore to persist
            metadata: Optional metadata to include
        """
        self._ensure_dir()

        # Save full graph
        graph_data = graph_store.to_dict()
        graph_path = self._base_dir / "graph.json"
        with open(graph_path, "w") as f:
            json.dump(graph_data, f, indent=2)

        # Save entities separately for easy inspection
        entities_path = self._base_dir / "entities.json"
        with open(entities_path, "w") as f:
            json.dump(graph_data["entities"], f, indent=2)

        # Save relations separately
        relations_path = self._base_dir / "relations.json"
        with open(relations_path, "w") as f:
            json.dump(graph_data["relations"], f, indent=2)

        # Save communities separately
        communities_path = self._base_dir / "communities.json"
        with open(communities_path, "w") as f:
            json.dump(graph_data["communities"], f, indent=2)

        # Save metadata with stats
        stats = graph_store.get_stats()
        meta = {
            "stats": {
                "num_entities": stats.num_entities,
                "num_relations": stats.num_relations,
                "num_communities": stats.num_communities,
                "num_chunks": stats.num_chunks,
                "entity_types": stats.entity_types,
                "relation_types": stats.relation_types,
                "avg_entity_degree": stats.avg_entity_degree,
                "max_community_level": stats.max_community_level,
            },
            **(metadata or {}),
        }
        metadata_path = self._base_dir / "metadata.json"
        with open(metadata_path, "w") as f:
            json.dump(meta, f, indent=2)

    async def load(self) -> GraphStore:
        """Load graph store from disk.

        Returns:
            GraphStore with loaded data

        Raises:
            FileNotFoundError: If graph.json doesn't exist
        """
        graph_path = self._base_dir / "graph.json"
        if not graph_path.exists():
            raise FileNotFoundError(f"Graph file not found: {graph_path}")

        with open(graph_path) as f:
            graph_data = json.load(f)

        return GraphStore.from_dict(graph_data)

    async def load_metadata(self) -> dict[str, Any]:
        """Load metadata from disk.

        Returns:
            Metadata dictionary

        Raises:
            FileNotFoundError: If metadata.json doesn't exist
        """
        metadata_path = self._base_dir / "metadata.json"
        if not metadata_path.exists():
            raise FileNotFoundError(f"Metadata file not found: {metadata_path}")

        with open(metadata_path) as f:
            return json.load(f)

    def exists(self) -> bool:
        """Check if artifact store has saved data.

        Returns:
            True if graph.json exists
        """
        return (self._base_dir / "graph.json").exists()

    async def save_embeddings(
        self,
        entity_embeddings: dict[str, list[float]],
        relation_embeddings: dict[str, list[float]],
    ) -> None:
        """Save embeddings to separate files.

        Args:
            entity_embeddings: Dict mapping entity_id to embedding
            relation_embeddings: Dict mapping relation_id to embedding
        """
        self._ensure_dir()

        # Save as JSON (simple but larger)
        entity_embed_path = self._base_dir / "entity_embeddings.json"
        with open(entity_embed_path, "w") as f:
            json.dump(entity_embeddings, f)

        relation_embed_path = self._base_dir / "relation_embeddings.json"
        with open(relation_embed_path, "w") as f:
            json.dump(relation_embeddings, f)

    async def load_embeddings(
        self,
    ) -> tuple[dict[str, list[float]], dict[str, list[float]]]:
        """Load embeddings from disk.

        Returns:
            Tuple of (entity_embeddings, relation_embeddings)
        """
        entity_embeds = {}
        relation_embeds = {}

        entity_embed_path = self._base_dir / "entity_embeddings.json"
        if entity_embed_path.exists():
            with open(entity_embed_path) as f:
                entity_embeds = json.load(f)

        relation_embed_path = self._base_dir / "relation_embeddings.json"
        if relation_embed_path.exists():
            with open(relation_embed_path) as f:
                relation_embeds = json.load(f)

        return entity_embeds, relation_embeds

    async def save_chunks(
        self,
        chunks: list[dict[str, Any]],
    ) -> None:
        """Save source chunks for reference.

        Args:
            chunks: List of chunk data dicts with id, text, etc.
        """
        self._ensure_dir()
        chunks_path = self._base_dir / "chunks.json"
        with open(chunks_path, "w") as f:
            json.dump(chunks, f, indent=2)

    async def load_chunks(self) -> list[dict[str, Any]]:
        """Load source chunks.

        Returns:
            List of chunk data dicts
        """
        chunks_path = self._base_dir / "chunks.json"
        if not chunks_path.exists():
            return []

        with open(chunks_path) as f:
            return json.load(f)

    def clear(self) -> None:
        """Remove all artifacts."""
        import shutil

        if self._base_dir.exists():
            shutil.rmtree(self._base_dir)
