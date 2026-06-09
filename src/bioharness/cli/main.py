"""bioharness: doctor + config introspection.

The actual scoring entry point is ``framework-eval run --method
pipeline ...``; this CLI exists for infrastructure self-checks and
to print the resolved configuration.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from collections.abc import Sequence
from dataclasses import asdict

import httpx

from bioharness import __version__
from bioharness.config import (
    CASCADE_THRESHOLD,
    DENSE_COLLECTION,
    MAX_AGENT_ITERATIONS,
    RERANK_TOP_K,
    RETRIEVAL_TOP_K,
    ServiceConfig,
)


# ----------------------------------------------------------------------
# doctor
# ----------------------------------------------------------------------


async def _probe(url: str, *, expect_path: str = "") -> tuple[bool, str]:
    target = url.rstrip("/") + expect_path
    try:
        async with httpx.AsyncClient(timeout=5.0) as c:
            r = await c.get(target)
        if r.status_code in (200, 401, 403, 404):
            return True, f"HTTP {r.status_code}"
        return False, f"HTTP {r.status_code}"
    except Exception as exc:  # noqa: BLE001
        return False, str(exc)


async def _probe_postgres(url: str) -> tuple[bool, str]:
    try:
        import psycopg
    except ImportError:
        return False, "psycopg not installed"
    try:
        with psycopg.connect(url, connect_timeout=5) as conn:
            conn.execute("SELECT 1")
        return True, "ok"
    except Exception as exc:  # noqa: BLE001
        return False, str(exc)


async def _async_skip() -> tuple[bool, str]:
    return True, "not configured"


def _cmd_doctor(_args: argparse.Namespace) -> int:
    services = ServiceConfig.from_env()
    summary = services.reachable_summary()
    print(f"bioharness {__version__} — service connectivity check")
    fail = 0

    async def main_async() -> None:
        nonlocal fail
        results = await asyncio.gather(
            _probe(summary["llm"], expect_path="/models"),
            _probe(summary["embed"], expect_path="/models"),
            _probe(summary["rerank"], expect_path="/models"),
            _probe(summary["qdrant"], expect_path="/collections"),
            _probe_postgres(summary["pubmed_pg"]),
            _probe_postgres(summary["papergraph_pg"]),
        )
        labels = ["llm", "embed", "rerank", "qdrant", "pubmed_pg", "papergraph_pg"]
        for label, (ok, detail) in zip(labels, results, strict=False):
            url = summary.get(label, "(unset)")
            mark = "ok" if ok else "FAIL"
            if not ok:
                fail += 1
            print(f"  {label:<14}  {mark:<4}  {url}  ({detail})")

    asyncio.run(main_async())
    if services.force_agent:
        print("  force_agent ........ ON  (every non-yesno item escalates to the agent)")
    return 1 if fail else 0


# ----------------------------------------------------------------------
# config introspection
# ----------------------------------------------------------------------


def _cmd_config(_args: argparse.Namespace) -> int:
    """Print the resolved ServiceConfig + cascade constants as JSON."""
    services = asdict(ServiceConfig.from_env())
    payload = {
        "services": services,
        "cascade": {
            "cascade_threshold":    CASCADE_THRESHOLD,
            "retrieval_top_k":      RETRIEVAL_TOP_K,
            "rerank_top_k":         RERANK_TOP_K,
            "max_agent_iterations": MAX_AGENT_ITERATIONS,
            "dense_collection":     DENSE_COLLECTION,
        },
    }
    print(json.dumps(payload, indent=2))
    return 0


# ----------------------------------------------------------------------
# parser
# ----------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="bioharness",
        description="bioHarness — adaptive cascade utilities.",
    )
    parser.add_argument("--version", action="version", version=__version__)

    sub = parser.add_subparsers(dest="cmd", metavar="COMMAND")

    p = sub.add_parser(
        "doctor",
        help="probe LLM / embed / rerank / Qdrant / Postgres reachability",
    )
    p.set_defaults(func=_cmd_doctor)

    p = sub.add_parser(
        "config",
        help="print the resolved service URLs + cascade constants",
    )
    p.set_defaults(func=_cmd_config)

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    if not getattr(args, "cmd", None):
        parser.print_help()
        return 0
    return int(args.func(args))


if __name__ == "__main__":
    sys.exit(main())
