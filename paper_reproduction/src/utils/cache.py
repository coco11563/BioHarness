"""Query-level caching with execution traces for case study analysis.

DESIGN PRINCIPLE: Full Traceability
- Every query execution is recorded with complete trace
- Supports hard case / failure case analysis
- Caches embeddings and retrieval results for repeated runs
"""

import hashlib
import json
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Any


@dataclass
class QueryTrace:
    """Complete execution trace for case study analysis."""
    # Query identification
    query_id: str
    question: str
    question_type: str  # yesno, mcq, mcq_multi, factoid, list, summary, expression

    # RLM execution trace
    rlm_iterations: int = 0
    rlm_max_iterations: int = 30
    rlm_max_depth: int = 1
    rlm_trajectory: list[dict] = field(default_factory=list)
    rlm_sub_llm_calls: int = 0

    # Retrieval trace
    retrieval_results: list[dict] = field(default_factory=list)
    kg_tools_used: list[str] = field(default_factory=list)

    # Answer trace
    predicted_answer: str = ""
    ground_truth: str = ""
    is_correct: bool = False
    score: float = 0.0
    evaluation_method: str = ""

    # Timing (ms)
    total_latency_ms: float = 0.0
    retrieval_latency_ms: float = 0.0
    llm_latency_ms: float = 0.0

    # Case study
    error_type: str | None = None
    failure_reason: str | None = None

    # Metadata
    dataset: str = ""
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    run_config: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


class ErrorCategory:
    """Standard error categories for failure analysis."""
    RETRIEVAL_MISS = "retrieval_miss"
    WRONG_EXTRACTION = "wrong_extraction"
    REASONING_ERROR = "reasoning_error"
    INCOMPLETE_ANSWER = "incomplete_answer"
    TIMEOUT = "timeout"
    KG_TOOL_ERROR = "kg_tool_error"
    PARSING_ERROR = "parsing_error"
    SERVICE_ERROR = "service_error"


class BenchmarkCache:
    """Multi-level cache with execution traces for case study."""

    def __init__(self, cache_dir: Path | None = None):
        self.cache_dir = Path(cache_dir) if cache_dir else None
        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

        self.embedding_cache: dict[str, list[float]] = {}
        self.retrieval_cache: dict[str, dict] = {}
        self.trace_store: list[QueryTrace] = []
        self._stats = {"emb_hits": 0, "emb_misses": 0, "ret_hits": 0, "ret_misses": 0}

    # === Embedding Cache ===
    def get_embedding(self, text: str) -> list[float] | None:
        result = self.embedding_cache.get(text)
        self._stats["emb_hits" if result else "emb_misses"] += 1
        return result

    def set_embedding(self, text: str, embedding: list[float]) -> None:
        self.embedding_cache[text] = embedding

    # === Retrieval Cache ===
    @staticmethod
    def _hash_query(query: str, params: dict | None = None) -> str:
        key = query + (json.dumps(params, sort_keys=True) if params else "")
        return hashlib.sha256(key.encode()).hexdigest()[:16]

    def get_retrieval(self, query: str, params: dict | None = None) -> dict | None:
        key = self._hash_query(query, params)
        result = self.retrieval_cache.get(key)
        self._stats["ret_hits" if result else "ret_misses"] += 1
        return result

    def set_retrieval(self, query: str, results: dict, params: dict | None = None) -> None:
        self.retrieval_cache[self._hash_query(query, params)] = results

    # === Trace Operations ===
    def record_trace(self, trace: QueryTrace) -> None:
        self.trace_store.append(trace)

    def save_traces(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w', encoding='utf-8') as f:
            for t in self.trace_store:
                f.write(json.dumps(t.to_dict(), ensure_ascii=False) + '\n')

    def load_traces(self, path: Path) -> None:
        self.trace_store = []
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    self.trace_store.append(QueryTrace(**json.loads(line)))

    # === Case Study Analysis ===
    def get_failures(self) -> list[QueryTrace]:
        return [t for t in self.trace_store if not t.is_correct]

    def get_hard_cases(self, threshold: float = 0.5) -> list[QueryTrace]:
        return [t for t in self.trace_store if t.score < threshold]

    def get_by_error_type(self, error_type: str) -> list[QueryTrace]:
        return [t for t in self.trace_store if t.error_type == error_type]

    def get_by_question_type(self, qtype: str) -> list[QueryTrace]:
        return [t for t in self.trace_store if t.question_type == qtype]

    def get_by_dataset(self, dataset: str) -> list[QueryTrace]:
        return [t for t in self.trace_store if t.dataset == dataset]

    def get_timeout_cases(self) -> list[QueryTrace]:
        return [t for t in self.trace_store if t.rlm_iterations >= t.rlm_max_iterations]

    # === Export ===
    def export_case_study(self, output_dir: Path) -> dict:
        """Export structured case study data."""
        output_dir.mkdir(parents=True, exist_ok=True)

        self.save_traces(output_dir / "all_traces.jsonl")

        failures = self.get_failures()
        with open(output_dir / "failures.jsonl", 'w', encoding='utf-8') as f:
            for t in failures:
                f.write(json.dumps(t.to_dict(), ensure_ascii=False) + '\n')

        hard_cases = self.get_hard_cases(0.5)
        with open(output_dir / "hard_cases.jsonl", 'w', encoding='utf-8') as f:
            for t in hard_cases:
                f.write(json.dumps(t.to_dict(), ensure_ascii=False) + '\n')

        # Stats
        total = len(self.trace_store)
        stats = {
            "total": total,
            "correct": sum(1 for t in self.trace_store if t.is_correct),
            "accuracy": sum(1 for t in self.trace_store if t.is_correct) / total if total else 0,
            "failures": len(failures),
            "hard_cases": len(hard_cases),
            "by_question_type": {},
            "by_error_type": {},
            "by_dataset": {},
        }

        if total:
            stats["avg_iterations"] = sum(t.rlm_iterations for t in self.trace_store) / total
            stats["avg_latency_ms"] = sum(t.total_latency_ms for t in self.trace_store) / total

        for qtype in set(t.question_type for t in self.trace_store):
            cases = self.get_by_question_type(qtype)
            correct = sum(1 for t in cases if t.is_correct)
            stats["by_question_type"][qtype] = {
                "total": len(cases), "correct": correct,
                "accuracy": correct / len(cases) if cases else 0
            }

        for err in set(t.error_type for t in failures if t.error_type):
            stats["by_error_type"][err] = len(self.get_by_error_type(err))

        for ds in set(t.dataset for t in self.trace_store if t.dataset):
            cases = self.get_by_dataset(ds)
            correct = sum(1 for t in cases if t.is_correct)
            stats["by_dataset"][ds] = {
                "total": len(cases), "correct": correct,
                "accuracy": correct / len(cases) if cases else 0
            }

        with open(output_dir / "stats.json", 'w', encoding='utf-8') as f:
            json.dump(stats, f, indent=2, ensure_ascii=False)

        return stats

    def get_stats(self) -> dict:
        return {"cache": self._stats, "traces": len(self.trace_store)}

    def clear(self) -> None:
        self.embedding_cache.clear()
        self.retrieval_cache.clear()
        self.trace_store.clear()


def classify_error(trace: QueryTrace) -> str | None:
    """Classify error type for a failed trace."""
    if trace.is_correct:
        return None
    if trace.rlm_iterations >= trace.rlm_max_iterations:
        return ErrorCategory.TIMEOUT
    if trace.kg_tools_used and not trace.retrieval_results:
        return ErrorCategory.KG_TOOL_ERROR
    if not trace.retrieval_results:
        return ErrorCategory.RETRIEVAL_MISS
    if not trace.predicted_answer.strip():
        return ErrorCategory.WRONG_EXTRACTION
    return ErrorCategory.REASONING_ERROR
