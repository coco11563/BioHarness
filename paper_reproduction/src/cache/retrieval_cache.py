"""Multi-strategy retrieval cache storing PMID/score lists.

Design:
1. Store once per (question_id, strategy): top-N results + gt_pmids
2. Load with ablation: inject_gt controls whether GT is prepended
   - inject_gt=False → pure top-K (realistic)
   - inject_gt=True  → GT(x) + top-(K-x) (oracle)

Cache structure:
    .cache/retrieval/{strategy}/{dataset}_cache.jsonl

Each line:
    {
        "question_id": "bioasq_xxx",
        "params_hash": "abc123",
        "results": [
            {"pmid": "12345678", "score": 0.85, "rank": 1, "has_pmc": true},
            ...  # up to 100
        ],
        "gt_pmids": ["12345678", ...],  # extracted from metadata
        "cached_at": "2024-01-12T10:00:00"
    }

Usage:
    cache = RetrievalCache(cache_dir=".cache/retrieval")

    # Store (once per question per strategy)
    await cache.set(question_id, strategy="B6", results, gt_pmids=["123", "456"])

    # Load without GT (realistic)
    pmids = await cache.load(question_id, strategy="B6", top_k=50, inject_gt=False)

    # Load with GT (oracle) - GT takes priority slots
    pmids = await cache.load(question_id, strategy="B6", top_k=50, inject_gt=True)
"""

import hashlib
import json
import asyncio
import re
from dataclasses import dataclass, asdict
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Optional


class CacheMode(str, Enum):
    """Retrieval cache modes (for loading, not storage)."""
    WITHOUT_GT = "without_gt"  # Realistic: pure retrieval top-K
    WITH_GT = "with_gt"        # Oracle: GT + top-(K-x)


@dataclass
class CachedDoc:
    """A cached document reference."""
    pmid: str
    score: float
    rank: int
    has_pmc: bool = False
    paper_uuid: Optional[str] = None

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "CachedDoc":
        return cls(
            pmid=d["pmid"],
            score=d.get("score", 0.0),
            rank=d.get("rank", 0),
            has_pmc=d.get("has_pmc", False),
            paper_uuid=d.get("paper_uuid"),
        )


@dataclass
class CacheEntry:
    """A single cache entry."""
    question_id: str
    params_hash: str
    results: list[CachedDoc]        # Top-N retrieval results
    gt_pmids: list[str]             # Ground truth PMIDs from metadata
    cached_at: str = ""

    def __post_init__(self):
        if not self.cached_at:
            self.cached_at = datetime.now().isoformat()

    def to_dict(self) -> dict:
        return {
            "question_id": self.question_id,
            "params_hash": self.params_hash,
            "results": [r.to_dict() for r in self.results],
            "gt_pmids": self.gt_pmids,
            "cached_at": self.cached_at,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "CacheEntry":
        return cls(
            question_id=d["question_id"],
            params_hash=d.get("params_hash", ""),
            results=[CachedDoc.from_dict(r) for r in d.get("results", [])],
            gt_pmids=d.get("gt_pmids", []),
            cached_at=d.get("cached_at", ""),
        )


class RetrievalCache:
    """Multi-strategy retrieval cache with flexible GT injection at load time.

    Example:
        cache = RetrievalCache(cache_dir=".cache/retrieval")

        # Check if cached for strategy
        if not await cache.exists(question_id, strategy="B6"):
            results = await retriever.search(query, limit=100)
            gt_pmids = extract_pmids_from_metadata(metadata)
            await cache.set(question_id, strategy="B6", results, gt_pmids=gt_pmids)

        # Load for experiment (ablation via inject_gt)
        pmids = await cache.load(question_id, strategy="B6", top_k=50, inject_gt=False)  # realistic
        pmids = await cache.load(question_id, strategy="B6", top_k=50, inject_gt=True)   # oracle
    """

    def __init__(
        self,
        cache_dir: str | Path = ".cache/retrieval",
        max_results: int = 100,
    ):
        self._cache_dir = Path(cache_dir)
        self._cache_dir.mkdir(parents=True, exist_ok=True)
        self._max_results = max_results

        # In-memory index: {strategy: {question_id: CacheEntry}}
        self._index: dict[str, dict[str, CacheEntry]] = {}
        self._loaded_keys: set[tuple[str, str]] = set()  # (strategy, dataset)
        self._lock = asyncio.Lock()
        self._write_lock = asyncio.Lock()

    def _get_cache_file(self, strategy: str, dataset: str) -> Path:
        """Get cache file path: {cache_dir}/{strategy}/{dataset}_cache.jsonl"""
        strategy_dir = self._cache_dir / strategy
        strategy_dir.mkdir(parents=True, exist_ok=True)
        return strategy_dir / f"{dataset}_cache.jsonl"

    def _extract_dataset(self, question_id: str) -> str:
        """Extract dataset from question_id (e.g., bioasq_xxx -> bioasq)."""
        # Handle formats: bioasq_xxx, pubmedqa_pqal_xxx, pubmedqa_pqal_test_xxx
        if question_id.startswith("pubmedqa_pqal_test"):
            return "pubmedqa_pqal_test"
        elif question_id.startswith("pubmedqa_pqal"):
            return "pubmedqa_pqal"
        elif question_id.startswith("pubmedqa_pqaa"):
            return "pubmedqa_pqaa"
        elif question_id.startswith("pubmedqa_pqau"):
            return "pubmedqa_pqau"
        else:
            return question_id.split("_")[0]

    async def _load_dataset(self, strategy: str, dataset: str) -> None:
        """Load cache file into memory for a specific strategy and dataset."""
        key = (strategy, dataset)
        if key in self._loaded_keys:
            return

        cache_file = self._get_cache_file(strategy, dataset)
        if not cache_file.exists():
            self._loaded_keys.add(key)
            return

        async with self._lock:
            if key in self._loaded_keys:
                return

            if strategy not in self._index:
                self._index[strategy] = {}

            with open(cache_file, "r") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        data = json.loads(line)
                        entry = CacheEntry.from_dict(data)
                        self._index[strategy][entry.question_id] = entry
                    except (json.JSONDecodeError, KeyError) as e:
                        continue

            self._loaded_keys.add(key)

    async def exists(self, question_id: str, strategy: str) -> bool:
        """Check if question is cached for a specific strategy."""
        dataset = self._extract_dataset(question_id)
        await self._load_dataset(strategy, dataset)
        return strategy in self._index and question_id in self._index[strategy]

    async def get_entry(self, question_id: str, strategy: str) -> CacheEntry | None:
        """Get raw cache entry for a specific strategy."""
        dataset = self._extract_dataset(question_id)
        await self._load_dataset(strategy, dataset)
        if strategy not in self._index:
            return None
        return self._index[strategy].get(question_id)

    async def load(
        self,
        question_id: str,
        strategy: str,
        top_k: int = 50,
        inject_gt: bool = False,
    ) -> list[str]:
        """Load PMIDs from cache with optional GT injection.

        Args:
            question_id: Question identifier
            strategy: Retrieval strategy (A1-A6, B1-B6, C1-C2)
            top_k: Total number of PMIDs to return
            inject_gt: If True, prepend GT PMIDs (oracle mode)

        Returns:
            List of PMIDs (length = top_k or less), or empty list if not cached
        """
        entry = await self.get_entry(question_id, strategy)
        if not entry:
            return []

        result_pmids = [r.pmid for r in entry.results]

        if not inject_gt:
            # Realistic mode: pure retrieval
            return result_pmids[:top_k]
        else:
            # Oracle mode: GT + (top_k - len(GT)) from retrieval
            gt_set = set(entry.gt_pmids)

            # Filter out GT from retrieval results (avoid duplicates)
            non_gt_pmids = [p for p in result_pmids if p not in gt_set]

            # Combine: GT first, then fill remaining slots
            remaining_slots = max(0, top_k - len(entry.gt_pmids))
            combined = entry.gt_pmids + non_gt_pmids[:remaining_slots]

            return combined[:top_k]

    async def load_with_scores(
        self,
        question_id: str,
        strategy: str,
        top_k: int = 50,
        inject_gt: bool = False,
    ) -> list[tuple[str, float]]:
        """Load PMIDs with scores for a specific strategy."""
        entry = await self.get_entry(question_id, strategy)
        if not entry:
            return []

        if not inject_gt:
            return [(r.pmid, r.score) for r in entry.results[:top_k]]
        else:
            gt_set = set(entry.gt_pmids)

            # GT PMIDs get score 1.0
            gt_with_scores = [(p, 1.0) for p in entry.gt_pmids]

            # Non-GT from retrieval
            non_gt = [(r.pmid, r.score) for r in entry.results if r.pmid not in gt_set]

            remaining_slots = max(0, top_k - len(gt_with_scores))
            combined = gt_with_scores + non_gt[:remaining_slots]

            return combined[:top_k]

    async def set(
        self,
        question_id: str,
        strategy: str,
        results: list[dict | CachedDoc],
        gt_pmids: list[str] | None = None,
        params: dict | None = None,
    ) -> None:
        """Store retrieval results for a specific strategy.

        Args:
            question_id: Question identifier
            strategy: Retrieval strategy (A1-A6, B1-B6, C1-C2)
            results: List of {pmid, score, rank, has_pmc, paper_uuid}
            gt_pmids: Ground truth PMIDs from metadata
            params: Retrieval params (for hash)
        """
        dataset = self._extract_dataset(question_id)
        params_hash = hashlib.sha256(
            json.dumps(params or {}, sort_keys=True).encode()
        ).hexdigest()[:16]

        # Convert to CachedDoc
        cached_results = []
        for i, r in enumerate(results[:self._max_results]):
            if isinstance(r, CachedDoc):
                cached_results.append(r)
            else:
                cached_results.append(CachedDoc(
                    pmid=str(r.get("pmid", "")),
                    score=float(r.get("score", 0.0)),
                    rank=r.get("rank", i + 1),
                    has_pmc=bool(r.get("has_pmc", False)),
                    paper_uuid=r.get("paper_uuid"),
                ))

        entry = CacheEntry(
            question_id=question_id,
            params_hash=params_hash,
            results=cached_results,
            gt_pmids=gt_pmids or [],
        )

        # Update index
        if strategy not in self._index:
            self._index[strategy] = {}
        self._index[strategy][question_id] = entry

        # Append to file
        cache_file = self._get_cache_file(strategy, dataset)
        async with self._write_lock:
            with open(cache_file, "a") as f:
                f.write(json.dumps(entry.to_dict()) + "\n")

    def get_stats(self, strategy: str | None = None) -> dict:
        """Get cache statistics.

        Args:
            strategy: If specified, get stats for that strategy only.
                      If None, get stats for all strategies.
        """
        if strategy:
            strategies = [strategy] if strategy in self._index else []
        else:
            strategies = list(self._index.keys())

        total_entries = 0
        by_strategy: dict[str, dict[str, int]] = {}
        with_gt = 0
        total_gt_pmids = 0

        for s in strategies:
            by_strategy[s] = {}
            for qid, entry in self._index.get(s, {}).items():
                total_entries += 1
                dataset = self._extract_dataset(qid)
                by_strategy[s][dataset] = by_strategy[s].get(dataset, 0) + 1
                if entry.gt_pmids:
                    with_gt += 1
                    total_gt_pmids += len(entry.gt_pmids)

        return {
            "total_entries": total_entries,
            "by_strategy": by_strategy,
            "with_gt_pmids": with_gt,
            "total_gt_pmids": total_gt_pmids,
            "avg_gt_per_question": total_gt_pmids / with_gt if with_gt else 0,
        }

    def list_strategies(self) -> list[str]:
        """List all strategies that have cached data."""
        # Check both in-memory index and disk
        strategies = set(self._index.keys())

        # Also check disk
        if self._cache_dir.exists():
            for d in self._cache_dir.iterdir():
                if d.is_dir():
                    strategies.add(d.name)

        return sorted(strategies)


# =============================================================================
# PMID Extraction Utilities
# =============================================================================

def extract_pmids_from_urls(urls: list[str]) -> list[str]:
    """Extract PMIDs from PubMed URLs.

    Handles:
    - https://pubmed.ncbi.nlm.nih.gov/12345678/
    - http://www.ncbi.nlm.nih.gov/pubmed/12345678
    """
    pmids = []
    patterns = [
        r'pubmed\.ncbi\.nlm\.nih\.gov/(\d+)',
        r'ncbi\.nlm\.nih\.gov/pubmed/(\d+)',
        r'/pubmed/(\d+)',
    ]

    for url in urls:
        for pattern in patterns:
            match = re.search(pattern, url)
            if match:
                pmids.append(match.group(1))
                break

    return pmids


def extract_pmids_from_metadata(metadata: dict) -> list[str]:
    """Extract PMIDs from benchmark item metadata.

    Handles:
    - metadata.pubmed_id (PubMedQA)
    - metadata.documents (BioASQ PubMed URLs)
    """
    pmids = []

    # Direct PMID
    if "pubmed_id" in metadata:
        pmid = str(metadata["pubmed_id"])
        if pmid.isdigit():
            pmids.append(pmid)

    # PubMed URLs (BioASQ)
    if "documents" in metadata:
        urls = metadata["documents"]
        if isinstance(urls, list):
            pmids.extend(extract_pmids_from_urls(urls))

    return list(dict.fromkeys(pmids))  # Dedupe preserving order
