"""KG-based RAG implementations for biomedical QA.

This package provides three KG-RAG paradigms:
- LightRAG: Entity/relation-centric with 4 query modes
- PathRAG: Multi-hop path-based retrieval
- GraphRAG: Hierarchical community-based retrieval (MS-style)

All implementations share:
- Common data models (Entity, Relation, Community, Path)
- Shared extraction layer (LLM-based entity/relation extraction)
- Shared storage layer (NetworkX graph + Qdrant vectors)
- Unified benchmark interface

Usage:
    from src.kg import LightRAGClient, PathRAGClient, GraphRAGClient

    # LightRAG
    lightrag = LightRAGClient()
    await lightrag.index(chunks)
    result = await lightrag.query("What treats diabetes?", mode="hybrid")

    # PathRAG
    pathrag = PathRAGClient()
    await pathrag.index(chunks)
    result = await pathrag.query("How is BRCA1 related to cancer?")

    # GraphRAG
    graphrag = GraphRAGClient()
    await graphrag.index(chunks)
    result = await graphrag.query("Overview of cancer treatment", mode="global")

    # Benchmark integration
    from src.kg import KGBenchmarkClient, BenchmarkConfig

    config = BenchmarkConfig(implementation="lightrag")
    client = KGBenchmarkClient(config)
    response = await client.generate(question, question_type)
"""

# Base models and protocols
from .base import (
    Entity,
    Relation,
    Community,
    Path,
    KGStats,
    QueryResult,
    ModelResponse,
    EntityType,
    RelationType,
    QueryMode,
    KGClient,
    BenchmarkModelClient,
)

# Implementations
from .lightrag import LightRAGClient
from .pathrag import PathRAGClient
from .graphrag import GraphRAGClient

# Benchmark integration
from .benchmark_client import (
    KGBenchmarkClient,
    BenchmarkConfig,
    create_benchmark_client,
    run_benchmark_comparison,
)

# Storage components
from .storage import GraphStore, KGVectorStore, ArtifactStore

# Extraction components
from .extraction import (
    EntityExtractor,
    RelationExtractor,
    ExtractionResult,
    RelationExtractionResult,
)

__all__ = [
    # Data models
    "Entity",
    "Relation",
    "Community",
    "Path",
    "KGStats",
    "QueryResult",
    "ModelResponse",
    # Enums
    "EntityType",
    "RelationType",
    "QueryMode",
    # Protocols
    "KGClient",
    "BenchmarkModelClient",
    # Client implementations
    "LightRAGClient",
    "PathRAGClient",
    "GraphRAGClient",
    # Benchmark
    "KGBenchmarkClient",
    "BenchmarkConfig",
    "create_benchmark_client",
    "run_benchmark_comparison",
    # Storage
    "GraphStore",
    "KGVectorStore",
    "ArtifactStore",
    # Extraction
    "EntityExtractor",
    "RelationExtractor",
    "ExtractionResult",
    "RelationExtractionResult",
]
