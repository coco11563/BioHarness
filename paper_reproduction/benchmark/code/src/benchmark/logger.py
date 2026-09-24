"""
Benchmark logging and export utilities.

Provides BenchmarkLogger for comprehensive logging of benchmark runs
with support for JSONL logs and various export formats.
"""

import csv
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any

from .metrics import BenchmarkReport, SingleResult


class BenchmarkLogger:
    """Comprehensive logging for benchmark runs.

    Logs run events to JSONL format for incremental writing.
    Supports export to JSON, CSV, and Markdown.

    Example:
        logger = BenchmarkLogger(log_dir=Path("output/logs"))

        # Automatic run ID
        logger.log_run_start("qwen-30b", {"concurrency": 50})

        for result in results:
            logger.log_item_result(result)

        logger.log_run_end(report)

        # Export
        logger.export_json(report)
        logger.export_csv(report)
        logger.export_markdown(report)
    """

    def __init__(
        self,
        log_dir: Path | str,
        run_id: str | None = None,
    ):
        """Initialize logger.

        Args:
            log_dir: Directory for log files
            run_id: Optional run ID (auto-generated if not provided)
        """
        self.log_dir = Path(log_dir)
        self.run_id = run_id or f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{os.getpid()}"
        self._run_dir = self.log_dir / self.run_id
        self._log_file: Path | None = None
        self._item_count = 0

    def _ensure_dir(self) -> None:
        """Create run directory if needed."""
        self._run_dir.mkdir(parents=True, exist_ok=True)

    def _get_log_file(self) -> Path:
        """Get path to log file."""
        if self._log_file is None:
            self._ensure_dir()
            self._log_file = self._run_dir / "run.jsonl"
        return self._log_file

    def _write_log_entry(self, entry: dict[str, Any]) -> None:
        """Write a single log entry to JSONL file."""
        log_file = self._get_log_file()
        with open(log_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    def log_run_start(self, model_name: str, config: dict[str, Any]) -> None:
        """Log run start event.

        Args:
            model_name: Name of model being evaluated
            config: Run configuration
        """
        self._ensure_dir()
        entry = {
            "type": "run_start",
            "model": model_name,
            "run_id": self.run_id,
            "timestamp": datetime.now().isoformat(),
            "config": config,
        }
        self._write_log_entry(entry)
        self._item_count = 0

    def log_item_result(self, result: SingleResult) -> None:
        """Log a single item result.

        Args:
            result: SingleResult to log
        """
        self._item_count += 1
        entry = {
            "type": "item",
            "seq": self._item_count,
            "id": result.id,
            "dataset": result.dataset,
            "subtask": result.subtask,
            "question": result.question,
            "ground_truth": result.ground_truth,
            "predicted": result.predicted,
            "response_text": result.response_text,
            "correct": result.score_result.correct,
            "score": result.score_result.score,
            "method": result.score_result.method,
            "latency_ms": result.latency_ms,
            "timestamp": result.timestamp.isoformat(),
            "metadata": result.metadata or {},
        }
        self._write_log_entry(entry)

    def log_run_end(self, report: BenchmarkReport) -> None:
        """Log run end event with summary.

        Args:
            report: BenchmarkReport with final results
        """
        entry = {
            "type": "run_end",
            "run_id": self.run_id,
            "timestamp": datetime.now().isoformat(),
            "total": report.overall.total,
            "correct": report.overall.correct_count,
            "score": report.overall.score,
            "duration_s": report.overall.duration_seconds,
            "by_type": report.overall.by_type,
        }
        self._write_log_entry(entry)

    def export_json(
        self,
        report: BenchmarkReport,
        include_results: bool = False,
    ) -> Path:
        """Export report to JSON file.

        Args:
            report: BenchmarkReport to export
            include_results: Include individual results (larger file)

        Returns:
            Path to exported file
        """
        self._ensure_dir()
        output_path = self._run_dir / "report.json"

        data = report.to_dict(include_results=include_results)

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

        return output_path

    def export_csv(self, report: BenchmarkReport) -> Path:
        """Export individual results to CSV.

        Args:
            report: BenchmarkReport to export

        Returns:
            Path to exported file
        """
        self._ensure_dir()
        output_path = self._run_dir / "results.csv"

        # Collect all results
        all_results = []
        for ds_report in report.datasets.values():
            for st_report in ds_report.subtasks.values():
                all_results.extend(st_report.results)

        if not all_results:
            return output_path

        # Write CSV
        fieldnames = [
            "id", "dataset", "subtask", "question", "ground_truth",
            "predicted", "correct", "score", "method", "latency_ms",
        ]

        with open(output_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()

            for result in all_results:
                writer.writerow({
                    "id": result.id,
                    "dataset": result.dataset,
                    "subtask": result.subtask,
                    "question": result.question,
                    "ground_truth": result.ground_truth,
                    "predicted": result.predicted,
                    "correct": result.score_result.correct,
                    "score": f"{result.score_result.score:.4f}",
                    "method": result.score_result.method,
                    "latency_ms": f"{result.latency_ms:.1f}",
                })

        return output_path

    def export_markdown(self, report: BenchmarkReport) -> Path:
        """Export summary report to Markdown.

        Args:
            report: BenchmarkReport to export

        Returns:
            Path to exported file
        """
        self._ensure_dir()
        output_path = self._run_dir / "report.md"

        lines = [
            "# Biomedical QA Benchmark Evaluation Report",
            "",
            f"**Model**: {report.model_name}",
            f"**Run ID**: {report.run_id}",
            f"**Timestamp**: {report.timestamp.strftime('%Y-%m-%d %H:%M:%S')}",
            f"**Duration**: {report.overall.duration_seconds:.1f} seconds",
            "",
            "## Overall Results",
            "",
            f"- **Total Questions**: {report.overall.total:,}",
            f"- **Correct**: {report.overall.correct_count:,}",
            f"- **Accuracy**: {report.overall.score:.2%}",
            "",
            "## By Dataset",
            "",
            "| Dataset | Total | Correct | Score | Subtasks |",
            "|---------|-------|---------|-------|----------|",
        ]

        for ds_name, ds_report in sorted(report.datasets.items()):
            subtask_str = ", ".join(
                f"{st}: {st_r.score:.1%}"
                for st, st_r in sorted(ds_report.subtasks.items())
            )
            lines.append(
                f"| {ds_name} | {ds_report.total} | {ds_report.correct_count} "
                f"| {ds_report.score:.1%} | {subtask_str} |"
            )

        lines.extend([
            "",
            "## By Question Type",
            "",
            "| Type | Total | Correct | Score | Metrics |",
            "|------|-------|---------|-------|---------|",
        ])

        # Aggregate metrics by type across all datasets
        type_metrics: dict[str, dict[str, Any]] = {}
        for ds_report in report.datasets.values():
            for st_name, st_report in ds_report.subtasks.items():
                if st_name not in type_metrics:
                    type_metrics[st_name] = {
                        "total": 0,
                        "correct": 0,
                        "agg": st_report.raw_metrics_aggregate.copy(),
                    }
                type_metrics[st_name]["total"] += st_report.total
                type_metrics[st_name]["correct"] += st_report.correct_count

        for qtype, data in sorted(type_metrics.items()):
            score = data["correct"] / data["total"] if data["total"] > 0 else 0
            agg = data["agg"]

            if "accuracy" in agg:
                metrics_str = f"Accuracy: {agg['accuracy']:.1%}"
            elif "f1" in agg:
                metrics_str = f"P={agg['precision']:.2f} R={agg['recall']:.2f} F1={agg['f1']:.2f}"
            elif "rouge_l" in agg:
                metrics_str = f"R1={agg['rouge_1']:.2f} R2={agg['rouge_2']:.2f} RL={agg['rouge_l']:.2f}"
            else:
                metrics_str = "-"

            lines.append(
                f"| {qtype} | {data['total']} | {data['correct']} "
                f"| {score:.1%} | {metrics_str} |"
            )

        lines.extend([
            "",
            "## Detailed Results",
            "",
        ])

        for ds_name, ds_report in sorted(report.datasets.items()):
            lines.append(f"### {ds_name}")
            lines.append("")
            lines.append("| Subtask | Total | Correct | Score | Metrics |")
            lines.append("|---------|-------|---------|-------|---------|")

            for st_name, st_report in sorted(ds_report.subtasks.items()):
                agg = st_report.raw_metrics_aggregate

                if "accuracy" in agg:
                    metrics_str = f"Accuracy: {agg['accuracy']:.1%}"
                elif "f1" in agg:
                    metrics_str = f"P={agg['precision']:.2f} R={agg['recall']:.2f} F1={agg['f1']:.2f}"
                elif "rouge_l" in agg:
                    metrics_str = f"R1={agg['rouge_1']:.2f} R2={agg['rouge_2']:.2f} RL={agg['rouge_l']:.2f}"
                else:
                    metrics_str = "-"

                lines.append(
                    f"| {st_name} | {st_report.total} | {st_report.correct_count} "
                    f"| {st_report.score:.1%} | {metrics_str} |"
                )

            lines.append("")

        with open(output_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))

        return output_path

    def get_run_dir(self) -> Path:
        """Get the run directory path."""
        self._ensure_dir()
        return self._run_dir
