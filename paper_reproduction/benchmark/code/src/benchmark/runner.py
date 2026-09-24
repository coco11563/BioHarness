"""
Benchmark runner for executing evaluations.

Provides BenchmarkRunner class that orchestrates loading data,
calling model clients, evaluating results, and aggregating reports.
"""

import asyncio
import logging
import uuid
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from .client_protocol import ModelClient, ModelResponse
from .evaluator import MetricsEvaluator, aggregate_subtask_results, get_subtask_score
from .loader import BenchmarkLoader, QAItem
from .metrics import (
    BenchmarkReport,
    DatasetReport,
    OverallStats,
    ScoreResult,
    SingleResult,
    SubtaskReport,
)


@dataclass
class RunConfig:
    """Configuration for a benchmark run."""
    datasets: list[str] | None = None  # None = all available
    subtasks: list[str] | None = None  # None = all types
    limit: int | None = None  # Limit items per dataset
    concurrency: int = 50  # Max concurrent requests
    use_golden_context: bool = False  # Pass dataset context to model_client.generate
    thresholds: dict[str, float] = field(default_factory=dict)
    resume_dir: str | None = None  # Resume from existing run dir (skip completed items)
    item_allow: set[tuple[str, str]] | None = None  # If set, only run (dataset, item.id) in this set


class BenchmarkRunner:
    """Execute benchmark evaluations with configurable backends.

    Orchestrates the benchmark pipeline:
    1. Load items from datasets
    2. Send questions to model client
    3. Evaluate responses
    4. Aggregate results into reports

    Example:
        loader = BenchmarkLoader()
        evaluator = MetricsEvaluator()
        runner = BenchmarkRunner(loader, evaluator)

        async with LLMClient(backends=[...]) as client:
            report = await runner.run(
                model_client=client,
                model_name="qwen-30b",
                config=RunConfig(datasets=["bioasq"], concurrency=50),
            )

        print(f"Overall Score: {report.overall.score}")
    """

    def __init__(
        self,
        loader: BenchmarkLoader,
        evaluator: MetricsEvaluator,
        logger: Any = None,  # BenchmarkLogger, optional
    ):
        """Initialize runner.

        Args:
            loader: BenchmarkLoader instance
            evaluator: MetricsEvaluator instance
            logger: Optional BenchmarkLogger for detailed logging
        """
        self.loader = loader
        self.evaluator = evaluator
        self.logger = logger

    async def run(
        self,
        model_client: ModelClient,
        model_name: str,
        config: RunConfig | None = None,
    ) -> BenchmarkReport:
        """Run benchmark and return comprehensive report.

        Args:
            model_client: ModelClient instance (LLM or RAG)
            model_name: Name to identify this model in reports
            config: Run configuration

        Returns:
            BenchmarkReport with all results
        """
        config = config or RunConfig()
        run_id = datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + str(uuid.uuid4())[:8]
        start_time = datetime.now()

        # Log run start
        if self.logger:
            self.logger.log_run_start(model_name, {
                "run_id": run_id,
                "datasets": config.datasets,
                "subtasks": config.subtasks,
                "limit": config.limit,
                "concurrency": config.concurrency,
                "use_golden_context": config.use_golden_context,
            })

        # Load completed item IDs for resume
        completed_ids: set[str] = set()
        if config.resume_dir:
            import json as _rjson
            resume_path = Path(config.resume_dir)

            # Support two layouts:
            #   1. Single run.jsonl (old format: resume_dir/run.jsonl)
            #   2. Per-dataset dir (new format: resume_dir/<dataset>.jsonl)
            jsonl_files = []
            single = resume_path / "run.jsonl"
            if single.exists():
                jsonl_files.append(single)
            for f in sorted(resume_path.glob("*.jsonl")):
                if f.name != "run.jsonl":
                    jsonl_files.append(f)

            for jf in jsonl_files:
                with open(jf) as _rf:
                    for line in _rf:
                        try:
                            entry = _rjson.loads(line)
                            if entry.get("type") == "item" and entry.get("id"):
                                completed_ids.add(entry["id"])
                            elif entry.get("id") and "type" not in entry:
                                # Per-dataset jsonl: each line is an item record
                                completed_ids.add(entry["id"])
                        except _rjson.JSONDecodeError:
                            pass
            if completed_ids:
                print(f"  Resuming: {len(completed_ids)} items already completed, will skip them")

        # Collect items to process
        datasets_to_run = config.datasets or self.loader.list_datasets()
        items_by_dataset: dict[str, list[QAItem]] = defaultdict(list)

        for dataset in datasets_to_run:
            count = 0
            for item in self.loader.load_dataset(dataset):
                # Filter by subtask
                if config.subtasks and item.question_type not in config.subtasks:
                    continue
                # Filter by explicit allowlist (e.g., from disco router output)
                if config.item_allow is not None and (dataset, item.id) not in config.item_allow:
                    continue
                # Skip items without answers
                if not item.answer:
                    continue
                # Skip already completed items (resume mode)
                if item.id in completed_ids:
                    continue
                items_by_dataset[dataset].append(item)
                count += 1
                if config.limit and count >= config.limit:
                    break

        # Process all items with concurrency control
        all_results: dict[str, list[SingleResult]] = defaultdict(list)
        semaphore = asyncio.Semaphore(config.concurrency)

        async def process_item(item: QAItem) -> SingleResult:
            async with semaphore:
                try:
                    return await self.run_single(
                        item,
                        model_client,
                        config.thresholds,
                        use_golden_context=config.use_golden_context,
                    )
                except Exception as e:  # one failed item must not abort the whole run
                    logging.getLogger(__name__).error(
                        "item %s failed (%s: %s); recorded as incorrect", item.id, type(e).__name__, str(e)[:120]
                    )
                    return self._error_result(item, e, config.thresholds)

        # Create tasks for all items
        tasks = []
        item_list = []
        for dataset, items in items_by_dataset.items():
            for item in items:
                tasks.append(process_item(item))
                item_list.append((dataset, item))

        # Execute with progress tracking
        total = len(tasks)
        completed = 0
        print(f"\nRunning benchmark: {model_name}")
        print(f"  Datasets: {', '.join(datasets_to_run)}")
        print(f"  Total items: {total}")
        print()

        for coro in asyncio.as_completed(tasks):
            result = await coro
            completed += 1

            # Skip logging error results so they can be retried on resume
            if result.response_text and result.response_text.startswith("ERROR:"):
                print(f"  [SKIP] {result.id}: {result.response_text[:60]}")
                continue

            all_results[result.dataset].append(result)

            # Log individual result
            if self.logger:
                self.logger.log_item_result(result)

            # Progress update every 100 items
            if completed % 100 == 0 or completed == total:
                print(f"  Progress: {completed}/{total} ({100*completed/total:.1f}%)")

        # Aggregate results
        dataset_reports = self._aggregate_by_dataset(all_results)
        overall = self._compute_overall_stats(dataset_reports, start_time)

        report = BenchmarkReport(
            model_name=model_name,
            run_id=run_id,
            timestamp=start_time,
            datasets=dataset_reports,
            overall=overall,
            config={
                "datasets": config.datasets,
                "subtasks": config.subtasks,
                "limit": config.limit,
                "concurrency": config.concurrency,
                "use_golden_context": config.use_golden_context,
            },
        )

        # Log run end
        if self.logger:
            self.logger.log_run_end(report)

        return report

    async def run_single(
        self,
        item: QAItem,
        model_client: ModelClient,
        thresholds: dict[str, float] | None = None,
        use_golden_context: bool = False,
    ) -> SingleResult:
        """Run single question and evaluate.

        Args:
            item: QAItem to process
            model_client: ModelClient to use
            thresholds: Optional threshold overrides
            use_golden_context: Whether to pass item.context to model

        Returns:
            SingleResult with evaluation
        """
        # Generate response — pass item_key only to clients that accept it
        # (V14CascadeClient uses it for router_entities lookup)
        import inspect
        gen_kwargs = {
            'question': item.question,
            'question_type': item.question_type,
            'context': item.context if use_golden_context else None,
            'options': item.options,
        }
        sig = inspect.signature(model_client.generate)
        if 'item_key' in sig.parameters or any(
            p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()
        ):
            gen_kwargs['item_key'] = (item.dataset, item.id)
        response = await model_client.generate(**gen_kwargs)

        # Evaluate
        score_result = self.evaluator.evaluate(
            predicted=response.answer,
            ground_truth=item.answer or "",
            question_type=item.question_type,
            options=item.options,
            thresholds=thresholds,
        )

        return SingleResult(
            id=item.id,
            dataset=item.dataset,
            subtask=item.question_type,
            question=item.question,
            ground_truth=item.answer or "",
            predicted=response.answer,
            score_result=score_result,
            response_text=response.response_text,
            latency_ms=response.latency_ms,
            timestamp=datetime.now(),
            metadata=response.metadata,
        )

    def _error_result(self, item: QAItem, err: Exception, thresholds) -> SingleResult:
        """SingleResult for an item whose generation raised: empty prediction scored
        through the normal evaluator so per-dataset aggregates stay consistent."""
        score_result = self.evaluator.evaluate(
            predicted="",
            ground_truth=item.answer or "",
            question_type=item.question_type,
            options=item.options,
            thresholds=thresholds,
        )
        return SingleResult(
            id=item.id, dataset=item.dataset, subtask=item.question_type, question=item.question,
            ground_truth=item.answer or "", predicted="", score_result=score_result,
            response_text=f"ERROR: {type(err).__name__}: {str(err)[:200]}", latency_ms=0.0,
            timestamp=datetime.now(), metadata={"error": type(err).__name__},
        )

    def _aggregate_by_dataset(
        self,
        results: dict[str, list[SingleResult]],
    ) -> dict[str, DatasetReport]:
        """Aggregate results by dataset."""
        dataset_reports = {}

        for dataset, items in results.items():
            # Group by subtask
            by_subtask: dict[str, list[SingleResult]] = defaultdict(list)
            for item in items:
                by_subtask[item.subtask].append(item)

            # Aggregate each subtask
            subtask_reports = {}
            total_correct = 0
            total_items = 0

            for subtask, subtask_items in by_subtask.items():
                score_results = [item.score_result for item in subtask_items]
                aggregated = aggregate_subtask_results(score_results, subtask)
                score = get_subtask_score(aggregated, subtask)
                correct_count = sum(1 for item in subtask_items if item.score_result.correct)

                subtask_reports[subtask] = SubtaskReport(
                    name=subtask,
                    total=len(subtask_items),
                    correct_count=correct_count,
                    score=score,
                    raw_metrics_aggregate=aggregated,
                    results=subtask_items,
                )

                total_correct += correct_count
                total_items += len(subtask_items)

            # Compute dataset-level score (weighted average)
            dataset_score = total_correct / total_items if total_items > 0 else 0.0

            dataset_reports[dataset] = DatasetReport(
                name=dataset,
                total=total_items,
                correct_count=total_correct,
                score=dataset_score,
                subtasks=subtask_reports,
            )

        return dataset_reports

    def _compute_overall_stats(
        self,
        dataset_reports: dict[str, DatasetReport],
        start_time: datetime,
    ) -> OverallStats:
        """Compute overall statistics across all datasets."""
        total = 0
        correct = 0
        by_type: dict[str, dict[str, float]] = defaultdict(lambda: {"total": 0, "correct": 0})

        for report in dataset_reports.values():
            total += report.total
            correct += report.correct_count

            for subtask_name, subtask_report in report.subtasks.items():
                by_type[subtask_name]["total"] += subtask_report.total
                by_type[subtask_name]["correct"] += subtask_report.correct_count
                by_type[subtask_name]["score"] = (
                    by_type[subtask_name]["correct"] / by_type[subtask_name]["total"]
                    if by_type[subtask_name]["total"] > 0 else 0.0
                )

        duration = (datetime.now() - start_time).total_seconds()

        return OverallStats(
            total=total,
            correct_count=correct,
            score=correct / total if total > 0 else 0.0,
            by_type=dict(by_type),
            duration_seconds=duration,
        )


def print_report(report: BenchmarkReport) -> None:
    """Print benchmark report to console."""
    print("\n" + "=" * 60)
    print("BENCHMARK REPORT")
    print("=" * 60)
    print(f"Model: {report.model_name}")
    print(f"Run ID: {report.run_id}")
    print(f"Timestamp: {report.timestamp}")
    print(f"Duration: {report.overall.duration_seconds:.1f}s")
    print()

    print("OVERALL RESULTS")
    print("-" * 40)
    print(f"Total: {report.overall.total}")
    print(f"Correct: {report.overall.correct_count}")
    print(f"Accuracy: {report.overall.score:.1%}")
    print()

    print("BY DATASET")
    print("-" * 40)
    print(f"{'Dataset':<20} {'Total':>8} {'Correct':>8} {'Score':>8}")
    for name, ds_report in sorted(report.datasets.items()):
        print(f"{name:<20} {ds_report.total:>8} {ds_report.correct_count:>8} {ds_report.score:>7.1%}")
    print()

    print("BY QUESTION TYPE")
    print("-" * 40)
    print(f"{'Type':<15} {'Total':>8} {'Correct':>8} {'Score':>8}")
    for qtype, stats in sorted(report.overall.by_type.items()):
        print(f"{qtype:<15} {stats['total']:>8.0f} {stats['correct']:>8.0f} {stats['score']:>7.1%}")
    print()

    print("DETAILED BY DATASET/SUBTASK")
    print("-" * 40)
    for ds_name, ds_report in sorted(report.datasets.items()):
        print(f"\n[{ds_name}]")
        for st_name, st_report in sorted(ds_report.subtasks.items()):
            metrics_str = ""
            agg = st_report.raw_metrics_aggregate
            if "accuracy" in agg:
                metrics_str = f"acc={agg['accuracy']:.1%}"
            elif "f1" in agg:
                metrics_str = f"P={agg['precision']:.2f} R={agg['recall']:.2f} F1={agg['f1']:.2f}"
            elif "rouge_l" in agg:
                metrics_str = f"R1={agg['rouge_1']:.2f} R2={agg['rouge_2']:.2f} RL={agg['rouge_l']:.2f}"

            print(f"  {st_name:<12} {st_report.total:>6} items | {st_report.score:.1%} | {metrics_str}")

    print("\n" + "=" * 60)
