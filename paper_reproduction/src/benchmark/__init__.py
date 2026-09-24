"""Benchmark module for RLM-based biomedical QA evaluation.

Provides:
- RLMModelClient: ModelClient implementation wrapping BiomedicalRLMPipeline
- ParametricRunner: CLI interface for ablation experiments

Usage:
    from src.benchmark import RLMModelClient, run_benchmark

    client = RLMModelClient(max_iterations=30, max_depth=1)
    async with client:
        report = await run_benchmark(client, datasets=["bioasq"])
"""

from .rlm_client import RLMModelClient
from .runner import ParametricRunner, run_benchmark

__all__ = [
    "RLMModelClient",
    "ParametricRunner",
    "run_benchmark",
]
