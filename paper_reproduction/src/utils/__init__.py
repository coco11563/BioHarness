"""Shared utilities for PaperAsKnowledgeGraph-RAG.

Provides:
- Load-balanced async clients (embedding, LLM, rerank)
- Query-level caching with execution traces for case study
- Async execution helpers for sync-to-async bridging
- Centralized answer extraction for benchmark compliance
"""

from .clients import (
    # Client classes
    EmbeddingClient,
    LLMClient,
    RerankClient,
    # Singleton instances
    embed_client,
    llm_client,
    rerank_client,
    # Convenience functions
    embed,
    chat,
    # Error classes
    ClientError,
    EmbeddingError,
    LLMError,
    RerankError,
)

from .cache import (
    QueryTrace,
    BenchmarkCache,
    ErrorCategory,
    classify_error,
)

from .async_helper import (
    AsyncExecutor,
    run_async,
    get_executor,
)

from .answer_extraction import (
    extract_answer,
    QuestionType,
)

__all__ = [
    # Client classes
    "EmbeddingClient",
    "LLMClient",
    "RerankClient",
    # Singleton instances
    "embed_client",
    "llm_client",
    "rerank_client",
    # Convenience functions
    "embed",
    "chat",
    # Error classes
    "ClientError",
    "EmbeddingError",
    "LLMError",
    "RerankError",
    # Cache
    "QueryTrace",
    "BenchmarkCache",
    "ErrorCategory",
    "classify_error",
    # Async helpers
    "AsyncExecutor",
    "run_async",
    "get_executor",
    # Answer extraction
    "extract_answer",
    "QuestionType",
]
