"""
Core data structures for benchmark evaluation.

Defines metrics containers, score results, and report structures
for comprehensive benchmark tracking.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any


# =============================================================================
# Raw Metrics Types
# =============================================================================

@dataclass
class ClassificationMetrics:
    """Metrics for classification types (yesno, mcq).

    Used when evaluation is binary correct/incorrect.
    """
    correct: bool
    predicted: str
    ground_truth: str


@dataclass
class SetMetrics:
    """Metrics for set-based types (list, mcq_multi, expression).

    Captures precision, recall, and F1 for partial credit evaluation.
    """
    precision: float
    recall: float
    f1: float
    true_positives: int
    pred_count: int
    gt_count: int


@dataclass
class RougeMetrics:
    """Metrics for generative types (summary, factoid).

    Contains ROUGE-1, ROUGE-2, and ROUGE-L scores with P/R/F1.
    """
    rouge_1: dict[str, float]  # {precision, recall, fmeasure}
    rouge_2: dict[str, float]
    rouge_l: dict[str, float]


@dataclass
class RawMetrics:
    """Unified container for raw metrics based on question type.

    Only one of the metric types will be populated based on question_type:
    - yesno, mcq -> classification
    - list, mcq_multi, expression -> set_metrics
    - summary, factoid -> rouge
    """
    question_type: str
    classification: ClassificationMetrics | None = None
    set_metrics: SetMetrics | None = None
    rouge: RougeMetrics | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        result = {"question_type": self.question_type}
        if self.classification:
            result["classification"] = {
                "correct": self.classification.correct,
                "predicted": self.classification.predicted,
                "ground_truth": self.classification.ground_truth,
            }
        if self.set_metrics:
            result["set_metrics"] = {
                "precision": self.set_metrics.precision,
                "recall": self.set_metrics.recall,
                "f1": self.set_metrics.f1,
                "true_positives": self.set_metrics.true_positives,
                "pred_count": self.set_metrics.pred_count,
                "gt_count": self.set_metrics.gt_count,
            }
        if self.rouge:
            result["rouge"] = {
                "rouge_1": self.rouge.rouge_1,
                "rouge_2": self.rouge.rouge_2,
                "rouge_l": self.rouge.rouge_l,
            }
        return result


# =============================================================================
# Score Result
# =============================================================================

@dataclass
class ScoreResult:
    """Final evaluation result with score and method info.

    Attributes:
        score: Normalized score between 0.0 and 1.0
        correct: Binary pass/fail based on threshold
        method: Name of evaluation method used
        raw_metrics: Underlying raw metrics for detailed analysis
    """
    score: float
    correct: bool
    method: str
    raw_metrics: RawMetrics

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "score": self.score,
            "correct": self.correct,
            "method": self.method,
            "raw_metrics": self.raw_metrics.to_dict(),
        }


# =============================================================================
# Model Response
# =============================================================================

@dataclass
class ModelResponse:
    """Response from LLM or RAG system.

    Attributes:
        answer: Extracted answer for evaluation
        response_text: Full response text for logging
        latency_ms: Response time in milliseconds
        metadata: Optional extra info (e.g., RAG retrieval count)
    """
    answer: str
    response_text: str
    latency_ms: float
    metadata: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "answer": self.answer,
            "response_text": self.response_text,
            "latency_ms": self.latency_ms,
            "metadata": self.metadata,
        }


# =============================================================================
# Dataset Info
# =============================================================================

@dataclass
class DatasetInfo:
    """Information about a benchmark dataset.

    Attributes:
        name: Dataset identifier
        total: Total number of questions
        subtasks: Count per question type {type: count}
        answer_formats: Description of answer format per type
    """
    name: str
    total: int
    subtasks: dict[str, int]
    answer_formats: dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "name": self.name,
            "total": self.total,
            "subtasks": self.subtasks,
            "answer_formats": self.answer_formats,
        }


# =============================================================================
# Single Result
# =============================================================================

@dataclass
class SingleResult:
    """Result for a single question evaluation.

    Contains all information needed for logging and analysis.
    """
    id: str
    dataset: str
    subtask: str  # question_type
    question: str
    ground_truth: str
    predicted: str
    score_result: ScoreResult
    response_text: str
    latency_ms: float
    timestamp: datetime
    metadata: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "id": self.id,
            "dataset": self.dataset,
            "subtask": self.subtask,
            "question": self.question,
            "ground_truth": self.ground_truth,
            "predicted": self.predicted,
            "score_result": self.score_result.to_dict(),
            "response_text": self.response_text,
            "latency_ms": self.latency_ms,
            "timestamp": self.timestamp.isoformat(),
            "metadata": self.metadata,
        }


# =============================================================================
# Report Structures
# =============================================================================

@dataclass
class SubtaskReport:
    """Aggregated report for a single subtask (question type).

    Attributes:
        name: Question type name
        total: Number of questions
        correct_count: Number of correct answers
        score: Aggregated score (accuracy or avg F1/ROUGE-L)
        raw_metrics_aggregate: Aggregated raw metrics
        results: Individual results (optional, for detailed logging)
    """
    name: str
    total: int
    correct_count: int
    score: float
    raw_metrics_aggregate: dict[str, float]
    results: list[SingleResult] = field(default_factory=list)

    def to_dict(self, include_results: bool = False) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        data = {
            "name": self.name,
            "total": self.total,
            "correct_count": self.correct_count,
            "score": self.score,
            "raw_metrics_aggregate": self.raw_metrics_aggregate,
        }
        if include_results:
            data["results"] = [r.to_dict() for r in self.results]
        return data


@dataclass
class DatasetReport:
    """Aggregated report for a dataset.

    Attributes:
        name: Dataset name
        total: Total questions in dataset
        correct_count: Total correct across all subtasks
        score: Overall dataset score
        subtasks: Reports per question type
    """
    name: str
    total: int
    correct_count: int
    score: float
    subtasks: dict[str, SubtaskReport]

    def to_dict(self, include_results: bool = False) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "name": self.name,
            "total": self.total,
            "correct_count": self.correct_count,
            "score": self.score,
            "subtasks": {
                k: v.to_dict(include_results) for k, v in self.subtasks.items()
            },
        }


@dataclass
class OverallStats:
    """Overall benchmark statistics.

    Attributes:
        total: Total questions evaluated
        correct_count: Total correct answers
        score: Overall accuracy/score
        by_type: Aggregated scores per question type
        duration_seconds: Total runtime
    """
    total: int
    correct_count: int
    score: float
    by_type: dict[str, dict[str, float]]  # {type: {score, total, correct}}
    duration_seconds: float

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "total": self.total,
            "correct_count": self.correct_count,
            "score": self.score,
            "by_type": self.by_type,
            "duration_seconds": self.duration_seconds,
        }


@dataclass
class BenchmarkReport:
    """Complete benchmark run report.

    Top-level container for all benchmark results.
    """
    model_name: str
    run_id: str
    timestamp: datetime
    datasets: dict[str, DatasetReport]
    overall: OverallStats
    config: dict[str, Any] = field(default_factory=dict)

    def to_dict(self, include_results: bool = False) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "model_name": self.model_name,
            "run_id": self.run_id,
            "timestamp": self.timestamp.isoformat(),
            "datasets": {
                k: v.to_dict(include_results) for k, v in self.datasets.items()
            },
            "overall": self.overall.to_dict(),
            "config": self.config,
        }
