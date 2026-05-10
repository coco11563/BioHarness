"""framework-chi: doctor + thresholds + version utilities.

The actual scoring entry point is ``framework-eval run --method
v14-cascade-dual-rerank-grounded ...``; this CLI exists for
infrastructure self-checks and ablation flag introspection.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from collections.abc import Sequence

import httpx

from framework_chi import __version__
from framework_chi.config import CascadeOptions, ServiceConfig


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
    """Lightweight Postgres reachability check: import psycopg + try a connect."""
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


def _cmd_doctor(args: argparse.Namespace) -> int:
    services = ServiceConfig.from_env()
    summary = services.reachable_summary()
    print(f"framework-chi {__version__} — service connectivity check")
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
            _probe(summary["disco"], expect_path="/health") if "disco" in summary else _async_skip(),
            return_exceptions=False,
        )
        labels = [
            "llm", "embed", "rerank", "qdrant",
            "pubmed_pg", "papergraph_pg", "disco",
        ]
        for label, (ok, detail) in zip(labels, results, strict=False):
            url = summary.get(label, "(unset)")
            mark = "ok" if ok else "FAIL"
            if not ok and label != "disco":
                fail += 1
            print(f"  {label:<14}  {mark:<4}  {url}  ({detail})")

    asyncio.run(main_async())
    return 1 if fail else 0


async def _async_skip() -> tuple[bool, str]:
    return True, "not configured"


# ----------------------------------------------------------------------
# thresholds (proxy to framework-eval table) and ablations
# ----------------------------------------------------------------------


def _cmd_ablations(_args: argparse.Namespace) -> int:
    options = CascadeOptions()
    payload = {
        "cascade_threshold":    options.cascade_threshold,
        "enable_grounded_gate": options.enable_grounded_gate,
        "enable_dual_rerank":   options.enable_dual_rerank,
        "enable_disco":         options.enable_disco,
        "max_agent_iterations": options.max_agent_iterations,
        "no_tools":             options.no_tools,
        "retrieval_top_k":      options.retrieval_top_k,
        "rerank_top_k":         options.rerank_top_k,
    }
    print(json.dumps(payload, indent=2))
    return 0


# ----------------------------------------------------------------------
# parser
# ----------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="framework-chi",
        description="{framework}^χ — adaptive cascade method utilities.",
    )
    parser.add_argument("--version", action="version", version=__version__)

    sub = parser.add_subparsers(dest="cmd", metavar="COMMAND")

    p = sub.add_parser(
        "doctor",
        help="probe LLM / embed / rerank / Qdrant / Postgres reachability",
    )
    p.set_defaults(func=_cmd_doctor)

    p = sub.add_parser("ablations", help="print the default ablation flag table")
    p.set_defaults(func=_cmd_ablations)

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
