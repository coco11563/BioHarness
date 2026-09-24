"""
Biomedical QA Benchmark Pipeline.

A structured evaluation pipeline for biomedical question answering systems.
Supports multiple question types with fine-grained metrics collection.

Example usage:
    from benchmark import BenchmarkLoader, MetricsEvaluator, BenchmarkRunner, LLMClient

    loader = BenchmarkLoader()
    evaluator = MetricsEvaluator()
    runner = BenchmarkRunner(loader, evaluator)

    async with LLMClient(backends=[...]) as client:
        report = await runner.run(client, "model-name")
"""

from .metrics import (
    ClassificationMetrics,
    SetMetrics,
    RougeMetrics,
    RawMetrics,
    ScoreResult,
    DatasetInfo,
    SingleResult,
    SubtaskReport,
    DatasetReport,
    BenchmarkReport,
    OverallStats,
    ModelResponse,
)
from .evaluator import MetricsEvaluator, aggregate_subtask_results, get_subtask_score
from .loader import BenchmarkLoader, QAItem, DATASET_CONFIG
from .client_protocol import ModelClient, LLMClient, RAGClient, ClientConfig
from .runner import BenchmarkRunner, RunConfig, print_report
from .logger import BenchmarkLogger

__all__ = [
    # Metrics
    "ClassificationMetrics",
    "SetMetrics",
    "RougeMetrics",
    "RawMetrics",
    "ScoreResult",
    "DatasetInfo",
    "SingleResult",
    "SubtaskReport",
    "DatasetReport",
    "BenchmarkReport",
    "OverallStats",
    "ModelResponse",
    # Evaluator
    "MetricsEvaluator",
    "aggregate_subtask_results",
    "get_subtask_score",
    # Loader
    "BenchmarkLoader",
    "QAItem",
    "DATASET_CONFIG",
    # Client
    "ModelClient",
    "LLMClient",
    "RAGClient",
    "ClientConfig",
    # Runner
    "BenchmarkRunner",
    "RunConfig",
    "print_report",
    # Logger
    "BenchmarkLogger",
]
