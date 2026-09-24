"""Storage layer for KG data.

Provides:
- GraphStore: NetworkX-based graph storage
- VectorStore: Qdrant vector storage for entities/relations
- ArtifactStore: JSON/Parquet persistence

Design: Fail-fast, no fallbacks.
"""

from .graph_store import GraphStore
from .vector_store import KGVectorStore
from .artifact_store import ArtifactStore

__all__ = [
    "GraphStore",
    "KGVectorStore",
    "ArtifactStore",
]
