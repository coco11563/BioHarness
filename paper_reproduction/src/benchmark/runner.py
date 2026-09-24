"""Parametric benchmark runner for RLM ablation experiments.

Provides CLI interface for running benchmarks with configurable:
- max_iterations (RLM iterations per query)
- max_depth (RLM recursive depth)
- datasets/subtasks filtering
- concurrency control

Usage:
    # Run full benchmark
    python -m src.benchmark.runner --datasets bioasq medmcqa --limit 100

    # Ablation: vary max_iterations
    python -m src.benchmark.runner --rlm-max-iterations 5 10 20 30 --limit 500

    # Ablation: vary max_depth
    python -m src.benchmark.runner --rlm-max-depth 0 1 2 --limit 500
"""

import argparse
import asyncio
import json
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from .rlm_client import RLMModelClient, ModelClient, ModelResponse
from ..utils.cache import BenchmarkCache


# Dynamically import from existing benchmark code if available
# Strategy: Use importlib to load modules with unique names to avoid namespace conflicts
import importlib.util as _iu

_benchmark_code_path = Path(__file__).parent.parent.parent / "benchmark" / "code"
_HAS_BENCHMARK_CODE = False
_BenchmarkLoader = None
_MetricsEvaluator = None
_BaseBenchmarkRunner = None
_print_report = None
_ExternalRunConfig = None


def _load_module_from_path(name: str, file_path: Path):
    """Load a Python module from a file path with a unique name."""
    spec = _iu.spec_from_file_location(name, file_path)
    module = _iu.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


if _benchmark_code_path.exists():
    _ext_src = _benchmark_code_path / "src"
    _ext_pkg = _ext_src / "benchmark"
    _loader_path = _ext_pkg / "loader.py"
    _evaluator_path = _ext_pkg / "evaluator.py"
    _runner_path = _ext_pkg / "runner.py"
    _metrics_path = _ext_pkg / "metrics.py"
    _client_protocol_path = _ext_pkg / "client_protocol.py"

    all_exist = all(p.exists() for p in [
        _loader_path, _evaluator_path, _runner_path,
        _metrics_path, _client_protocol_path
    ])

    if all_exist:
        try:
            # Load modules with unique prefix to avoid conflicts
            # Must load dependencies first
            _ext_metrics = _load_module_from_path("_ext_bm_metrics", _metrics_path)
            _ext_client_protocol = _load_module_from_path("_ext_bm_client_protocol", _client_protocol_path)

            # Patch relative imports in loader - it uses .metrics
            sys.modules["_ext_bm_loader.metrics"] = _ext_metrics

            # For evaluator, it needs .metrics
            # Create a fake parent package
            import types
            _ext_bm_pkg = types.ModuleType("_ext_bm")
            _ext_bm_pkg.metrics = _ext_metrics
            _ext_bm_pkg.client_protocol = _ext_client_protocol
            sys.modules["_ext_bm"] = _ext_bm_pkg

            # Manually load loader with patched imports
            with open(_loader_path) as f:
                loader_code = f.read()
            loader_code = loader_code.replace("from .metrics import", "from _ext_bm_metrics import")
            loader_code = loader_code.replace("from .client_protocol import", "from _ext_bm_client_protocol import")
            _ext_loader = types.ModuleType("_ext_bm_loader")
            exec(compile(loader_code, str(_loader_path), "exec"), _ext_loader.__dict__)
            sys.modules["_ext_bm_loader"] = _ext_loader

            # Load evaluator with patched imports
            with open(_evaluator_path) as f:
                evaluator_code = f.read()
            evaluator_code = evaluator_code.replace("from .metrics import", "from _ext_bm_metrics import")
            evaluator_code = evaluator_code.replace("from .client_protocol import", "from _ext_bm_client_protocol import")
            evaluator_code = evaluator_code.replace("from .loader import", "from _ext_bm_loader import")
            _ext_evaluator = types.ModuleType("_ext_bm_evaluator")
            exec(compile(evaluator_code, str(_evaluator_path), "exec"), _ext_evaluator.__dict__)
            sys.modules["_ext_bm_evaluator"] = _ext_evaluator

            # Load runner with patched imports
            with open(_runner_path) as f:
                runner_code = f.read()
            runner_code = runner_code.replace("from .metrics import", "from _ext_bm_metrics import")
            runner_code = runner_code.replace("from .client_protocol import", "from _ext_bm_client_protocol import")
            runner_code = runner_code.replace("from .loader import", "from _ext_bm_loader import")
            runner_code = runner_code.replace("from .evaluator import", "from _ext_bm_evaluator import")
            _ext_runner = types.ModuleType("_ext_bm_runner")
            exec(compile(runner_code, str(_runner_path), "exec"), _ext_runner.__dict__)
            sys.modules["_ext_bm_runner"] = _ext_runner

            # Extract classes
            _BenchmarkLoader = _ext_loader.BenchmarkLoader
            _MetricsEvaluator = _ext_evaluator.MetricsEvaluator
            _BaseBenchmarkRunner = _ext_runner.BenchmarkRunner
            _ExternalRunConfig = _ext_runner.RunConfig
            _print_report = _ext_runner.print_report

            _HAS_BENCHMARK_CODE = True

        except Exception as _e:
            import traceback as _tb
            # Uncomment for debugging:
            # print(f"Failed to load benchmark code: {_e}")
            # _tb.print_exc()
            _HAS_BENCHMARK_CODE = False


# Minimal implementations if benchmark code not available
if not _HAS_BENCHMARK_CODE:
    @dataclass
    class _StubRunConfig:
        """Minimal run config."""
        datasets: list[str] | None = None
        subtasks: list[str] | None = None
        limit: int | None = None
        concurrency: int = 50
    _ExternalRunConfig = _StubRunConfig

    class _StubBenchmarkLoader:
        """Stub loader - requires actual benchmark data."""
        def __init__(self, data_dir: str = ""):
            self.data_dir = data_dir

        def list_datasets(self) -> list[str]:
            return []

        def load_dataset(self, name: str):
            return iter([])
    _BenchmarkLoader = _StubBenchmarkLoader

    class _StubMetricsEvaluator:
        """Stub evaluator."""
        pass
    _MetricsEvaluator = _StubMetricsEvaluator

    def _stub_print_report(report):
        """Print report stub."""
        print(f"Score: {report.get('score', 'N/A')}")
    _print_report = _stub_print_report


# Public aliases for backwards compatibility
BenchmarkLoader = _BenchmarkLoader
MetricsEvaluator = _MetricsEvaluator
print_report = _print_report
RunConfig = _ExternalRunConfig


@dataclass
class AblationConfig:
    """Configuration for ablation experiments."""
    # RLM parameters (can be lists for ablation)
    max_iterations_values: list[int] = field(default_factory=lambda: [30])
    max_depth_values: list[int] = field(default_factory=lambda: [1])

    # Benchmark parameters
    datasets: list[str] | None = None
    subtasks: list[str] | None = None
    limit: int | None = None
    concurrency: int = 50

    # KG tools
    enable_kg_tools: bool = True

    # Output
    output_dir: Path = Path("./results")
    export_json: bool = True
    export_md: bool = True


class ParametricRunner:
    """Run benchmarks with parameter sweeps for ablation.

    Supports running multiple configurations in sequence or parallel
    for ablation experiments.

    Example:
        runner = ParametricRunner()

        # Run ablation over max_iterations
        results = await runner.run_ablation(
            max_iterations_values=[5, 10, 20, 30],
            datasets=["bioasq"],
            limit=100,
        )

        # Export results
        runner.export_ablation_results(results, Path("./ablation_results"))
    """

    def __init__(self, data_dir: Path | None = None):
        """Initialize runner.

        Args:
            data_dir: Path to benchmark data directory
        """
        self.data_dir = data_dir or (Path(__file__).resolve().parents[2] / "benchmark" / "unified")
        self._loader: BenchmarkLoader | None = None
        self._evaluator: MetricsEvaluator | None = None

    def _get_loader(self) -> BenchmarkLoader:
        """Get or create benchmark loader."""
        if self._loader is None:
            self._loader = BenchmarkLoader(str(self.data_dir))
        return self._loader

    def _get_evaluator(self) -> MetricsEvaluator:
        """Get or create evaluator."""
        if self._evaluator is None:
            self._evaluator = MetricsEvaluator()
        return self._evaluator

    async def run_single(
        self,
        max_iterations: int = 30,
        max_depth: int = 1,
        enable_kg_tools: bool = True,
        datasets: list[str] | None = None,
        subtasks: list[str] | None = None,
        limit: int | None = None,
        concurrency: int = 50,
    ) -> dict[str, Any]:
        """Run single benchmark configuration.

        Args:
            max_iterations: RLM max iterations
            max_depth: RLM recursive depth
            enable_kg_tools: Whether to enable KG tools
            datasets: Dataset names to run
            subtasks: Question types to include
            limit: Max items per dataset
            concurrency: Max concurrent requests

        Returns:
            Dict with benchmark results and config
        """
        # Create client
        client = RLMModelClient(
            max_iterations=max_iterations,
            max_depth=max_depth,
            enable_kg_tools=enable_kg_tools,
        )

        # Setup runner
        loader = self._get_loader()
        evaluator = self._get_evaluator()

        if not _HAS_BENCHMARK_CODE:
            raise RuntimeError(
                "Benchmark code not available. Please ensure benchmark/code exists."
            )

        runner = _BaseBenchmarkRunner(loader, evaluator)

        # Run benchmark
        config = _ExternalRunConfig(
            datasets=datasets,
            subtasks=subtasks,
            limit=limit,
            concurrency=concurrency,
        )

        model_name = f"rlm_iter{max_iterations}_depth{max_depth}"
        if not enable_kg_tools:
            model_name += "_nokg"

        async with client:
            report = await runner.run(client, model_name, config)

        # Get traces for case study
        cache = client.get_cache()

        return {
            "config": {
                "max_iterations": max_iterations,
                "max_depth": max_depth,
                "enable_kg_tools": enable_kg_tools,
                "datasets": datasets,
                "subtasks": subtasks,
                "limit": limit,
            },
            "report": report,
            "traces": cache.trace_store,
            "cache_stats": cache.get_stats(),
        }

    async def run_ablation(
        self,
        max_iterations_values: list[int] | None = None,
        max_depth_values: list[int] | None = None,
        enable_kg_tools: bool = True,
        datasets: list[str] | None = None,
        subtasks: list[str] | None = None,
        limit: int | None = None,
        concurrency: int = 50,
    ) -> list[dict[str, Any]]:
        """Run ablation experiment over parameter values.

        Args:
            max_iterations_values: Values to test for max_iterations
            max_depth_values: Values to test for max_depth
            enable_kg_tools: Whether to enable KG tools
            datasets: Dataset names to run
            subtasks: Question types to include
            limit: Max items per dataset
            concurrency: Max concurrent requests

        Returns:
            List of results for each configuration
        """
        max_iterations_values = max_iterations_values or [30]
        max_depth_values = max_depth_values or [1]

        results = []

        # Run all combinations
        for max_iterations in max_iterations_values:
            for max_depth in max_depth_values:
                print(f"\n{'='*60}")
                print(f"Running: max_iterations={max_iterations}, max_depth={max_depth}")
                print(f"{'='*60}")

                result = await self.run_single(
                    max_iterations=max_iterations,
                    max_depth=max_depth,
                    enable_kg_tools=enable_kg_tools,
                    datasets=datasets,
                    subtasks=subtasks,
                    limit=limit,
                    concurrency=concurrency,
                )
                results.append(result)

                # Print intermediate report
                print_report(result["report"])

        return results

    def export_ablation_results(
        self,
        results: list[dict[str, Any]],
        output_dir: Path,
    ) -> None:
        """Export ablation results to files.

        Args:
            results: List of ablation results
            output_dir: Directory to write files
        """
        output_dir.mkdir(parents=True, exist_ok=True)

        # Summary JSON
        summary = []
        for r in results:
            report = r["report"]
            summary.append({
                "config": r["config"],
                "overall_score": report.overall.score,
                "total_items": report.overall.total,
                "correct_count": report.overall.correct_count,
                "duration_seconds": report.overall.duration_seconds,
                "by_dataset": {
                    name: {
                        "score": ds.score,
                        "total": ds.total,
                        "correct": ds.correct_count,
                    }
                    for name, ds in report.datasets.items()
                },
                "by_type": report.overall.by_type,
            })

        with open(output_dir / "summary.json", "w") as f:
            json.dump(summary, f, indent=2, default=str)

        # Individual result files
        for i, r in enumerate(results):
            config = r["config"]
            prefix = f"iter{config['max_iterations']}_depth{config['max_depth']}"

            # Traces
            cache = BenchmarkCache()
            cache.trace_store = r["traces"]
            cache.export_case_study(output_dir / prefix)

        # Markdown report
        self._write_markdown_report(results, output_dir / "report.md")

        print(f"\nResults exported to: {output_dir}")

    def _write_markdown_report(
        self,
        results: list[dict[str, Any]],
        output_path: Path,
    ) -> None:
        """Write markdown summary report."""
        lines = [
            "# RLM Ablation Experiment Results",
            "",
            f"Generated: {datetime.now().isoformat()}",
            "",
            "## Summary",
            "",
            "| max_iter | max_depth | KG | Score | Total | Correct | Duration |",
            "|----------|-----------|-----|-------|-------|---------|----------|",
        ]

        for r in results:
            config = r["config"]
            report = r["report"]
            kg = "Yes" if config["enable_kg_tools"] else "No"
            lines.append(
                f"| {config['max_iterations']} | {config['max_depth']} | {kg} | "
                f"{report.overall.score:.1%} | {report.overall.total} | "
                f"{report.overall.correct_count} | {report.overall.duration_seconds:.0f}s |"
            )

        lines.extend([
            "",
            "## By Dataset",
            "",
        ])

        for r in results:
            config = r["config"]
            report = r["report"]
            lines.append(f"### iter={config['max_iterations']}, depth={config['max_depth']}")
            lines.append("")
            lines.append("| Dataset | Score | Total | Correct |")
            lines.append("|---------|-------|-------|---------|")
            for name, ds in sorted(report.datasets.items()):
                lines.append(f"| {name} | {ds.score:.1%} | {ds.total} | {ds.correct_count} |")
            lines.append("")

        lines.extend([
            "",
            "## By Question Type",
            "",
        ])

        # Collect all types
        all_types = set()
        for r in results:
            all_types.update(r["report"].overall.by_type.keys())

        # Table header
        header = ["Type"] + [
            f"i{r['config']['max_iterations']}/d{r['config']['max_depth']}"
            for r in results
        ]
        lines.append("| " + " | ".join(header) + " |")
        lines.append("|" + "|".join(["---"] * len(header)) + "|")

        for qtype in sorted(all_types):
            row = [qtype]
            for r in results:
                stats = r["report"].overall.by_type.get(qtype, {})
                score = stats.get("score", 0)
                row.append(f"{score:.1%}")
            lines.append("| " + " | ".join(row) + " |")

        with open(output_path, "w") as f:
            f.write("\n".join(lines))


async def run_benchmark(
    client: RLMModelClient,
    datasets: list[str] | None = None,
    subtasks: list[str] | None = None,
    limit: int | None = None,
    concurrency: int = 50,
    data_dir: Path | None = None,
) -> dict[str, Any]:
    """Convenience function to run benchmark with RLM client.

    Args:
        client: RLMModelClient instance
        datasets: Dataset names to run
        subtasks: Question types to include
        limit: Max items per dataset
        concurrency: Max concurrent requests
        data_dir: Path to benchmark data

    Returns:
        Dict with benchmark results
    """
    if not _HAS_BENCHMARK_CODE:
        raise RuntimeError(
            "Benchmark code not available. Please ensure benchmark/code exists."
        )

    runner = ParametricRunner(data_dir=data_dir)
    loader = runner._get_loader()
    evaluator = runner._get_evaluator()
    benchmark_runner = _BaseBenchmarkRunner(loader, evaluator)

    config = _ExternalRunConfig(
        datasets=datasets,
        subtasks=subtasks,
        limit=limit,
        concurrency=concurrency,
    )

    model_name = f"rlm_iter{client.max_iterations}_depth{client.max_depth}"

    async with client:
        report = await benchmark_runner.run(client, model_name, config)

    return {
        "config": client.get_config_summary(),
        "report": report,
        "cache": client.get_cache(),
    }


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Run RLM benchmark with configurable parameters"
    )

    # RLM parameters
    parser.add_argument(
        "--rlm-max-iterations",
        type=int,
        nargs="+",
        default=[30],
        help="RLM max iterations (can specify multiple for ablation)",
    )
    parser.add_argument(
        "--rlm-max-depth",
        type=int,
        nargs="+",
        default=[1],
        help="RLM recursive depth (can specify multiple for ablation)",
    )
    parser.add_argument(
        "--no-kg",
        action="store_true",
        help="Disable KG tools",
    )

    # Benchmark parameters
    parser.add_argument(
        "--datasets",
        type=str,
        nargs="+",
        default=None,
        help="Datasets to run (default: all)",
    )
    parser.add_argument(
        "--subtasks",
        type=str,
        nargs="+",
        default=None,
        help="Question types to include (default: all)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Max items per dataset",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=50,
        help="Max concurrent requests",
    )

    # Output
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("./results"),
        help="Output directory for results",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=None,
        help="Benchmark data directory",
    )

    args = parser.parse_args()

    # Run ablation
    runner = ParametricRunner(data_dir=args.data_dir)

    async def run():
        results = await runner.run_ablation(
            max_iterations_values=args.rlm_max_iterations,
            max_depth_values=args.rlm_max_depth,
            enable_kg_tools=not args.no_kg,
            datasets=args.datasets,
            subtasks=args.subtasks,
            limit=args.limit,
            concurrency=args.concurrency,
        )

        # Export results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = args.output_dir / timestamp
        runner.export_ablation_results(results, output_dir)

        return results

    asyncio.run(run())


if __name__ == "__main__":
    main()
