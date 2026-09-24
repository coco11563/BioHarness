#!/usr/bin/env python3
"""Build retrieval cache for all benchmark questions using specified strategies.

Usage:
    # Single strategy
    python scripts/build_strategy_cache.py \
        --strategy B6 \
        --datasets bioasq pubmedqa_pqal_test \
        --concurrency 20

    # All strategies (14 total)
    python scripts/build_strategy_cache.py \
        --all-strategies \
        --datasets bioasq pubmedqa_pqal_test \
        --concurrency 10

    # Single dataset with limit
    python scripts/build_strategy_cache.py \
        --strategy C2 \
        --datasets pubmedqa_pqal_test \
        --limit 100 \
        --concurrency 30
"""

import asyncio
import argparse
import json
import sys
import time
from pathlib import Path

# Add project root
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from cache.retrieval_cache import RetrievalCache, extract_pmids_from_metadata
from retriever.hybrid_retriever import HybridRetriever, get_config_by_strategy, STRATEGY_PRESETS


async def build_cache_for_strategy(
    strategy: str,
    dataset: str,
    questions: list[dict],
    cache: RetrievalCache,
    concurrency: int = 20,
    skip_existing: bool = True,
) -> dict:
    """Build cache for a single strategy and dataset.

    Args:
        strategy: Strategy name (A1-A6, B1-B6, C1-C2)
        dataset: Dataset name
        questions: List of question dicts
        cache: RetrievalCache instance
        concurrency: Max concurrent retrievals
        skip_existing: Skip already cached questions

    Returns:
        Stats dict
    """
    # Initialize retriever for this strategy
    config = get_config_by_strategy(strategy)
    retriever = HybridRetriever(config)
    await retriever.initialize()

    semaphore = asyncio.Semaphore(concurrency)
    stats = {
        "total": len(questions),
        "cached": 0,
        "skipped": 0,
        "errors": 0,
        "with_gt": 0,
    }

    async def process_one(q):
        async with semaphore:
            qid = q["id"]
            try:
                # Skip if already cached
                if skip_existing and await cache.exists(qid, strategy):
                    stats["skipped"] += 1
                    return

                # Run retrieval
                result = await retriever.retrieve(q["question"])

                # Convert to cache format
                results = []
                for i, ab in enumerate(result.abstracts):
                    results.append({
                        "pmid": ab.pmid,
                        "score": ab.score,
                        "rank": i + 1,
                        "has_pmc": ab.have_fulltext,
                        "paper_uuid": ab.paper_uuid,
                    })

                # Extract GT PMIDs
                gt_pmids = extract_pmids_from_metadata(q.get("metadata", {}))
                if gt_pmids:
                    stats["with_gt"] += 1

                # Store
                await cache.set(
                    qid,
                    strategy=strategy,
                    results=results,
                    gt_pmids=gt_pmids,
                    params={"strategy": strategy},
                )
                stats["cached"] += 1

            except Exception as e:
                stats["errors"] += 1
                print(f"    Error [{strategy}] {qid}: {e}")

    # Run all
    tasks = [process_one(q) for q in questions]
    done = 0
    for coro in asyncio.as_completed(tasks):
        await coro
        done += 1
        if done % 50 == 0 or done == len(tasks):
            print(f"    [{strategy}] Progress: {done}/{len(tasks)} "
                  f"(cached={stats['cached']}, skip={stats['skipped']}, err={stats['errors']})")

    await retriever.close()
    return stats


def load_questions(unified_dir: Path, dataset: str, limit: int | None = None) -> list[dict]:
    """Load questions from benchmark JSONL file."""
    jsonl_file = unified_dir / f"{dataset}.jsonl"
    if not jsonl_file.exists():
        print(f"  Warning: {jsonl_file} not found")
        return []

    questions = []
    with open(jsonl_file) as f:
        for i, line in enumerate(f):
            if limit and i >= limit:
                break
            questions.append(json.loads(line))

    return questions


async def main():
    parser = argparse.ArgumentParser(description="Build retrieval cache for benchmark questions")
    parser.add_argument("--strategy", type=str, help="Single strategy to build (A1-A6, B1-B6, C1-C2)")
    parser.add_argument("--all-strategies", action="store_true", help="Build all 14 strategies")
    parser.add_argument("--datasets", nargs="+", default=["bioasq", "pubmedqa_pqal_test"],
                        help="Datasets to cache")
    parser.add_argument("--limit", type=int, default=None, help="Limit questions per dataset")
    parser.add_argument("--concurrency", type=int, default=20, help="Concurrent retrievals")
    parser.add_argument("--cache-dir", type=str, default=".cache/retrieval", help="Cache directory")
    parser.add_argument("--unified-dir", type=str, default="benchmark/unified", help="Benchmark data dir")
    parser.add_argument("--skip-existing", action="store_true", default=True, help="Skip cached questions")
    args = parser.parse_args()

    print("=" * 60)
    print("Multi-Strategy Retrieval Cache Builder")
    print("=" * 60)

    # Determine strategies to build
    if args.all_strategies:
        strategies = list(STRATEGY_PRESETS.keys())
    elif args.strategy:
        strategies = [args.strategy.upper()]
    else:
        print("Error: Specify --strategy or --all-strategies")
        return

    print(f"Strategies: {strategies}")
    print(f"Datasets: {args.datasets}")
    print(f"Concurrency: {args.concurrency}")
    print(f"Cache dir: {args.cache_dir}")
    if args.limit:
        print(f"Limit: {args.limit} per dataset")
    print()

    # Initialize cache
    cache = RetrievalCache(cache_dir=args.cache_dir)

    unified_dir = Path(args.unified_dir)
    total_stats = {"strategies": 0, "questions": 0, "cached": 0, "skipped": 0, "errors": 0}

    # Process each strategy
    for strategy in strategies:
        print(f"\n{'='*40}")
        print(f"Strategy: {strategy}")
        print(f"{'='*40}")

        for dataset in args.datasets:
            print(f"\n[{strategy}] {dataset}")
            start = time.time()

            # Load questions
            questions = load_questions(unified_dir, dataset, args.limit)
            if not questions:
                print(f"  No questions found")
                continue

            print(f"  Loaded {len(questions)} questions")

            # Build cache
            stats = await build_cache_for_strategy(
                strategy=strategy,
                dataset=dataset,
                questions=questions,
                cache=cache,
                concurrency=args.concurrency,
                skip_existing=args.skip_existing,
            )

            elapsed = time.time() - start
            print(f"  Done in {elapsed:.1f}s: {stats}")

            # Update totals
            total_stats["questions"] += stats["total"]
            total_stats["cached"] += stats["cached"]
            total_stats["skipped"] += stats["skipped"]
            total_stats["errors"] += stats["errors"]

        total_stats["strategies"] += 1

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Strategies processed: {total_stats['strategies']}")
    print(f"Total questions: {total_stats['questions']}")
    print(f"Newly cached: {total_stats['cached']}")
    print(f"Skipped (existing): {total_stats['skipped']}")
    print(f"Errors: {total_stats['errors']}")

    print("\nCache Stats by Strategy:")
    for strategy in strategies:
        stats = cache.get_stats(strategy)
        if stats["total_entries"] > 0:
            print(f"  {strategy}: {stats['total_entries']} entries")


if __name__ == "__main__":
    asyncio.run(main())
