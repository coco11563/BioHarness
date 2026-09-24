"""
Benchmark data loader.

Provides BenchmarkLoader class for loading and iterating benchmark datasets
with subtask filtering and dataset info capabilities.
"""

import json
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator


# Default unified directory (relative to benchmark/code/src/benchmark/loader.py)
UNIFIED_DIR = Path(__file__).resolve().parents[3] / "unified"

# Available datasets and their configurations
DATASET_CONFIG = {
    # LitQA2 (LAB-Bench). Answers are attested once in the primary literature and
    # are NOT recoverable from abstracts, so this is a direct test of
    # literature-grounded retrieval. metadata.key_passage carries the gold source
    # passage, which lets us score retrieval (passage recall) as well as accuracy.
    # Option lists include LAB-Bench's "Insufficient information" abstention
    # choice as the final letter (metadata.unsure_letter).
    "litqa2": {
        "description": "LitQA2 (LAB-Bench): literature-grounded MCQ, 199 items, 3-11 options",
        "question_types": ["mcq"],
        "answer_formats": {
            "mcq": "option text (not letter)",
        },
    },

    "litqa2_gold": {
        "description": "LitQA2 with the LAB-Bench key passage supplied as golden context (oracle ceiling)",
        "question_types": ["mcq"],
        "answer_formats": {
            "mcq": "option text (not letter)",
        },
    },

    "medxpertqa_text": {
        "description": "MedXpertQA (Text subset) — expert-level medical exam MCQ, 10 options (A-J)",
        "question_types": ["mcq"],
        "answer_formats": {
            "mcq": "option text (not letter)",
        },
    },

    # Option-order control: identical items with the A-J option list reversed.
    # Gold is stored as option TEXT, so the reference answer is unchanged and
    # scoring is identical; only the letter carrying it moves. Used to separate
    # a genuine positional preference from item difficulty.
    "medxpertqa_text_rot5": {
        "description": "MedXpertQA (Text) with options cyclically shifted by 5 — position control",
        "question_types": ["mcq"],
        "answer_formats": {
            "mcq": "option text (not letter)",
        },
    },

    "medxpertqa_text_rev": {
        "description": "MedXpertQA (Text) with reversed option order — position control",
        "question_types": ["mcq"],
        "answer_formats": {
            "mcq": "option text (not letter)",
        },
    },

    "bioasq": {
        "description": "BioASQ biomedical QA benchmark",
        "question_types": ["yesno", "factoid", "list", "summary"],
        "answer_formats": {
            "yesno": "yes/no/maybe",
            "factoid": "short text answer",
            "list": "JSON array of items",
            "summary": "long text summary",
        },
    },
    "geneturing": {
        "description": "GeneTuring genomics QA",
        "question_types": ["factoid"],
        "answer_formats": {
            "factoid": "gene/protein name",
        },
    },
    "medmcqa": {
        "description": "MedMCQA medical MCQ dataset",
        "question_types": ["mcq"],
        "answer_formats": {
            "mcq": "single letter A-D",
        },
    },
    "medqa_us": {
        "description": "MedQA USMLE-style questions",
        "question_types": ["mcq"],
        "answer_formats": {
            "mcq": "option text (not letter)",
        },
    },
    "medqa_taiwan": {
        "description": "MedQA Taiwan medical licensing exam (Traditional Chinese)",
        "question_types": ["mcq"],
        "answer_formats": {
            "mcq": "option text (not letter)",
        },
    },
    "medqa_mainland": {
        "description": "MedQA Mainland China medical exam (Simplified Chinese)",
        "question_types": ["mcq"],
        "answer_formats": {
            "mcq": "option text (not letter)",
        },
    },
    "pubmedqa_pqal_test": {
        "description": "PubMedQA labeled yes/no questions (test set)",
        "question_types": ["yesno"],
        "answer_formats": {
            "yesno": "yes/no/maybe",
        },
    },
    "scihorizon_hgkb": {
        "description": "Scihorizon genomics benchmark",
        "question_types": ["mcq", "mcq_multi", "expression", "list", "summary"],
        "answer_formats": {
            "mcq": "single letter A-E",
            "mcq_multi": "JSON array of letters",
            "expression": "JSON with tissue_list",
            "list": "JSON array of GO annotations",
            "summary": "text summary",
        },
    },
}


@dataclass
class QAItem:
    """A single QA item from the benchmark."""
    id: str
    dataset: str
    question: str
    question_type: str
    options: dict[str, str] | None
    context: list[str] | None
    answer: str | None
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "QAItem":
        """Create QAItem from dictionary."""
        return cls(
            id=data["id"],
            dataset=data["dataset"],
            question=data["question"],
            question_type=data["question_type"],
            options=data.get("options"),
            context=data.get("context"),
            answer=data.get("answer"),
            metadata=data.get("metadata", {}),
        )

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "dataset": self.dataset,
            "question": self.question,
            "question_type": self.question_type,
            "options": self.options,
            "context": self.context,
            "answer": self.answer,
            "metadata": self.metadata,
        }


@dataclass
class DatasetInfo:
    """Information about a benchmark dataset."""
    name: str
    total: int
    subtasks: dict[str, int]  # {question_type: count}
    answer_formats: dict[str, str] = field(default_factory=dict)
    description: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "total": self.total,
            "subtasks": self.subtasks,
            "answer_formats": self.answer_formats,
            "description": self.description,
        }


class BenchmarkLoader:
    """Loader for unified QA benchmark datasets.

    Supports loading datasets by name and filtering by subtask (question type).

    Example:
        loader = BenchmarkLoader()

        # List available datasets
        print(loader.list_datasets())

        # Get dataset info
        info = loader.get_dataset_info("bioasq")
        print(info.subtasks)

        # Load all items from a dataset
        for item in loader.load_dataset("bioasq"):
            print(item.question)

        # Load only specific subtask
        for item in loader.load_subtask("bioasq", "yesno"):
            print(item.question)
    """

    def __init__(self, unified_dir: Path | str = UNIFIED_DIR):
        """Initialize loader.

        Args:
            unified_dir: Path to directory containing unified JSONL files
        """
        self.unified_dir = Path(unified_dir)
        self._info_cache: dict[str, DatasetInfo] = {}

    def list_datasets(self) -> list[str]:
        """List available datasets in unified directory.

        Returns:
            List of dataset names that have corresponding JSONL files
        """
        available = []
        for name in DATASET_CONFIG:
            path = self.unified_dir / f"{name}.jsonl"
            if path.exists():
                available.append(name)
        return sorted(available)

    def list_subtasks(self, dataset: str) -> list[str]:
        """List subtasks (question types) for a dataset.

        Args:
            dataset: Dataset name

        Returns:
            List of question types in the dataset
        """
        info = self.get_dataset_info(dataset)
        return list(info.subtasks.keys())

    def get_dataset_path(self, name: str) -> Path:
        """Get path to dataset JSONL file.

        Args:
            name: Dataset name

        Returns:
            Path to JSONL file
        """
        if name == "all_with_answers":
            return self.unified_dir / "all_with_answers.jsonl"
        return self.unified_dir / f"{name}.jsonl"

    def get_dataset_info(self, dataset: str, force_refresh: bool = False) -> DatasetInfo:
        """Get information about a dataset.

        Scans the JSONL file to count items by question type.

        Args:
            dataset: Dataset name
            force_refresh: If True, rescan even if cached

        Returns:
            DatasetInfo with counts and format info
        """
        if dataset in self._info_cache and not force_refresh:
            return self._info_cache[dataset]

        path = self.get_dataset_path(dataset)
        if not path.exists():
            raise FileNotFoundError(f"Dataset not found: {path}")

        subtask_counts: dict[str, int] = defaultdict(int)
        total = 0

        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                data = json.loads(line)
                question_type = data.get("question_type", "unknown")
                subtask_counts[question_type] += 1
                total += 1

        config = DATASET_CONFIG.get(dataset, {})
        info = DatasetInfo(
            name=dataset,
            total=total,
            subtasks=dict(subtask_counts),
            answer_formats=config.get("answer_formats", {}),
            description=config.get("description", ""),
        )

        self._info_cache[dataset] = info
        return info

    def load_dataset(self, dataset: str) -> Iterator[QAItem]:
        """Load all items from a dataset.

        Args:
            dataset: Dataset name

        Yields:
            QAItem objects
        """
        path = self.get_dataset_path(dataset)
        if not path.exists():
            raise FileNotFoundError(f"Dataset not found: {path}")

        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                data = json.loads(line)
                yield QAItem.from_dict(data)

    def load_subtask(self, dataset: str, subtask: str) -> Iterator[QAItem]:
        """Load items from a specific subtask (question type).

        Args:
            dataset: Dataset name
            subtask: Question type to filter by

        Yields:
            QAItem objects matching the subtask
        """
        for item in self.load_dataset(dataset):
            if item.question_type == subtask:
                yield item

    def load_multiple(
        self,
        datasets: list[str] | None = None,
        subtasks: list[str] | None = None,
    ) -> Iterator[QAItem]:
        """Load items from multiple datasets with optional subtask filtering.

        Args:
            datasets: List of dataset names (None = all available)
            subtasks: List of question types to include (None = all)

        Yields:
            QAItem objects matching the criteria
        """
        if datasets is None:
            datasets = self.list_datasets()

        for dataset in datasets:
            for item in self.load_dataset(dataset):
                if subtasks is None or item.question_type in subtasks:
                    yield item

    def count_items(
        self,
        datasets: list[str] | None = None,
        subtasks: list[str] | None = None,
    ) -> dict[str, dict[str, int]]:
        """Count items by dataset and subtask.

        Args:
            datasets: List of dataset names (None = all available)
            subtasks: List of question types to count (None = all)

        Returns:
            Nested dict: {dataset: {subtask: count}}
        """
        if datasets is None:
            datasets = self.list_datasets()

        counts: dict[str, dict[str, int]] = {}
        for dataset in datasets:
            info = self.get_dataset_info(dataset)
            counts[dataset] = {}
            for st, count in info.subtasks.items():
                if subtasks is None or st in subtasks:
                    counts[dataset][st] = count

        return counts

    def get_summary(self) -> dict[str, Any]:
        """Get summary of all available datasets.

        Returns:
            Dict with dataset counts and totals
        """
        datasets = self.list_datasets()
        total_items = 0
        by_dataset = {}
        by_type: dict[str, int] = defaultdict(int)

        for ds in datasets:
            info = self.get_dataset_info(ds)
            by_dataset[ds] = {
                "total": info.total,
                "subtasks": info.subtasks,
                "description": info.description,
            }
            total_items += info.total
            for qtype, count in info.subtasks.items():
                by_type[qtype] += count

        return {
            "total_items": total_items,
            "dataset_count": len(datasets),
            "by_dataset": by_dataset,
            "by_type": dict(by_type),
        }
