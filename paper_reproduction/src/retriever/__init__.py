"""PubMed Abstract Retriever package.

This package provides a configurable retriever for PubMed abstracts
with support for MeSH filtering, keyword filtering, and vector search.

Main Components:
    - HybridRetriever: Main retriever with multi-path fusion and reranking
    - CachedHybridRetriever: HybridRetriever with automatic caching
    - HybridConfig: Configuration dataclass
    - Preset configs: B6_DEFAULT, C2_FAST, B2_BALANCED
    - Strategy presets: A1-A6, B1-B6, C1-C2 (14 total)

Load-balanced clients are available via:
    from utils.clients import embed_client, llm_client, rerank_client

Example:
    from src.retriever import HybridRetriever, C2_FAST

    retriever = HybridRetriever(C2_FAST())
    await retriever.initialize()
    result = await retriever.retrieve("Does metformin help with diabetes?")

Example with caching:
    from src.retriever import CachedHybridRetriever

    async with CachedHybridRetriever(strategy="B6") as retriever:
        pmids = await retriever.retrieve(
            question_id="bioasq_001",
            query="Does metformin help diabetes?",
            top_k=50,
        )
"""

from .models import (
    FilterMode,
    FilterStats,
    ParsedQuery,
    MeSHMatch,
    RetrievedAbstract,
    RetrievedChunk,
    RetrievalResult,
    RetrieverConfig,
)
from .hybrid_retriever import (
    HybridRetriever,
    HybridConfig,
    PathResult,
    B6_DEFAULT,
    C2_FAST,
    B2_BALANCED,
    # All 14 strategy presets
    A1_MESH_ONLY,
    A2_MESH_RERANK,
    A3_KEYWORD_ONLY,
    A4_KEYWORD_RERANK,
    A5_MESH_KW_RRF,
    A6_MESH_KW_RRF_RERANK,
    B1_MESH_VECTOR,
    B3_KEYWORD_VECTOR,
    B4_KEYWORD_VECTOR_RERANK,
    B5_MESH_KW_VECTOR,
    C1_DENSE_ONLY,
    STRATEGY_PRESETS,
    get_config_by_strategy,
)
from .cached_retriever import (
    CachedHybridRetriever,
    list_available_strategies,
    get_strategy_description,
)
from .chunk_search import ChunkSearch
from .query_parser import QueryParser
from .mesh_filter import MeSHFilter
from .keyword_filter import KeywordFilter
from .vector_search import VectorSearch

__all__ = [
    # Models
    "FilterMode",
    "FilterStats",
    "ParsedQuery",
    "MeSHMatch",
    "RetrievedAbstract",
    "RetrievedChunk",
    "RetrievalResult",
    "RetrieverConfig",
    # Main Retriever
    "HybridRetriever",
    "HybridConfig",
    "PathResult",
    # Cached Retriever
    "CachedHybridRetriever",
    # Preset Configurations (legacy)
    "B6_DEFAULT",
    "C2_FAST",
    "B2_BALANCED",
    # All 14 Strategy Presets
    "A1_MESH_ONLY",
    "A2_MESH_RERANK",
    "A3_KEYWORD_ONLY",
    "A4_KEYWORD_RERANK",
    "A5_MESH_KW_RRF",
    "A6_MESH_KW_RRF_RERANK",
    "B1_MESH_VECTOR",
    "B3_KEYWORD_VECTOR",
    "B4_KEYWORD_VECTOR_RERANK",
    "B5_MESH_KW_VECTOR",
    "C1_DENSE_ONLY",
    "STRATEGY_PRESETS",
    "get_config_by_strategy",
    "list_available_strategies",
    "get_strategy_description",
    # Components
    "ChunkSearch",
    "QueryParser",
    "MeSHFilter",
    "KeywordFilter",
    "VectorSearch",
]
