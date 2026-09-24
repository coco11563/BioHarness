"""Unified benchmark client for KG-based RAG implementations.

Provides a consistent interface for benchmarking LightRAG, PathRAG,
and GraphRAG against the benchmark suite.
"""

import time
from dataclasses import dataclass
from typing import Literal

from .base import ModelResponse, KGStats
from .lightrag import LightRAGClient
from .pathrag import PathRAGClient
from .graphrag import GraphRAGClient


@dataclass
class BenchmarkConfig:
    """Configuration for benchmark client."""
    implementation: Literal["lightrag", "pathrag", "graphrag"] = "lightrag"
    storage_dir: str = "data/kg"
    query_mode: str = "hybrid"  # lightrag: naive/local/global/hybrid
    top_k: int = 20
    max_hops: int = 3  # pathrag specific
    community_level: int = 1  # graphrag specific


class KGBenchmarkClient:
    """Unified benchmark client for all KG implementations.

    Wraps LightRAG, PathRAG, and GraphRAG with consistent interface
    for benchmark evaluation.

    Example:
        >>> config = BenchmarkConfig(implementation="lightrag")
        >>> client = KGBenchmarkClient(config)
        >>> await client.load_or_index(chunks, chunk_ids)
        >>> response = await client.generate(question, question_type)
        >>> print(response.answer)
    """

    def __init__(self, config: BenchmarkConfig):
        """Initialize benchmark client.

        Args:
            config: Benchmark configuration
        """
        self._config = config
        self._impl_name = config.implementation

        # Create appropriate client
        storage_path = f"{config.storage_dir}/{config.implementation}"

        if config.implementation == "lightrag":
            self._client = LightRAGClient(
                storage_dir=storage_path,
                namespace=f"kg_{config.implementation}",
            )
        elif config.implementation == "pathrag":
            self._client = PathRAGClient(
                storage_dir=storage_path,
                namespace=f"kg_{config.implementation}",
                max_hops=config.max_hops,
            )
        elif config.implementation == "graphrag":
            self._client = GraphRAGClient(
                storage_dir=storage_path,
                namespace=f"kg_{config.implementation}",
            )
        else:
            raise ValueError(f"Unknown implementation: {config.implementation}")

    async def index(
        self,
        chunks: list[str],
        chunk_ids: list[str] | None = None,
        **kwargs,
    ) -> None:
        """Build knowledge graph from text chunks.

        Args:
            chunks: Text chunks to process
            chunk_ids: Optional chunk IDs
            **kwargs: Implementation-specific options
        """
        await self._client.index(chunks, chunk_ids, **kwargs)

    async def load(self, path: str | None = None) -> None:
        """Load existing knowledge graph.

        Args:
            path: Optional path override
        """
        await self._client.load(path)

    async def load_or_index(
        self,
        chunks: list[str],
        chunk_ids: list[str] | None = None,
        force_reindex: bool = False,
        **kwargs,
    ) -> bool:
        """Load existing KG or index if not available.

        Args:
            chunks: Text chunks (used if indexing needed)
            chunk_ids: Optional chunk IDs
            force_reindex: Force re-indexing even if KG exists
            **kwargs: Indexing options

        Returns:
            True if indexed, False if loaded
        """
        if not force_reindex:
            try:
                await self._client.load()
                return False
            except (FileNotFoundError, Exception):
                pass

        await self._client.index(chunks, chunk_ids, **kwargs)
        return True

    async def generate(
        self,
        question: str,
        question_type: str,
        context: list[str] | None = None,
        options: dict[str, str] | None = None,
    ) -> ModelResponse:
        """Generate answer for benchmark question.

        Args:
            question: Question text
            question_type: Type (yesno, mcq, factoid, list, summary)
            context: Optional context (ignored, uses KG)
            options: MCQ options

        Returns:
            ModelResponse with answer and metadata
        """
        return await self._client.generate(
            question, question_type, context, options
        )

    async def query(
        self,
        question: str,
        **kwargs,
    ):
        """Direct query access for debugging.

        Args:
            question: Question text
            **kwargs: Query options

        Returns:
            QueryResult from underlying implementation
        """
        query_mode = kwargs.get("mode", self._config.query_mode)
        top_k = kwargs.get("top_k", self._config.top_k)

        return await self._client.query(question, mode=query_mode, top_k=top_k, **kwargs)

    def get_stats(self) -> KGStats:
        """Get knowledge graph statistics."""
        return self._client.get_stats()

    @property
    def implementation(self) -> str:
        """Get implementation name."""
        return self._impl_name

    @property
    def config(self) -> BenchmarkConfig:
        """Get configuration."""
        return self._config

    async def close(self) -> None:
        """Clean up resources."""
        await self._client.close()


def create_benchmark_client(
    implementation: str = "lightrag",
    **kwargs,
) -> KGBenchmarkClient:
    """Factory function to create benchmark client.

    Args:
        implementation: "lightrag", "pathrag", or "graphrag"
        **kwargs: Additional configuration options

    Returns:
        Configured KGBenchmarkClient
    """
    config = BenchmarkConfig(implementation=implementation, **kwargs)
    return KGBenchmarkClient(config)


async def run_benchmark_comparison(
    question: str,
    question_type: str,
    chunks: list[str] | None = None,
    implementations: list[str] | None = None,
) -> dict[str, ModelResponse]:
    """Run question through all implementations for comparison.

    Args:
        question: Question to answer
        question_type: Question type
        chunks: Optional chunks for indexing (if not already indexed)
        implementations: Which implementations to test (default: all)

    Returns:
        Dict mapping implementation name to response
    """
    if implementations is None:
        implementations = ["lightrag", "pathrag", "graphrag"]

    results = {}

    for impl in implementations:
        client = create_benchmark_client(impl)

        try:
            if chunks:
                await client.load_or_index(chunks)
            else:
                await client.load()

            response = await client.generate(question, question_type)
            results[impl] = response

        except Exception as e:
            results[impl] = ModelResponse(
                answer=f"Error: {e}",
                response_text="",
                latency_ms=0,
                metadata={"error": str(e)},
            )
        finally:
            await client.close()

    return results
