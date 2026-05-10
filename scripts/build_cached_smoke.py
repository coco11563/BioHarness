"""Record a 50-item cached smoke set against the live LLM stack.

Run once after standing up the user-provided infrastructure (see
``docs/infra.md``) to freeze a small replay tape. The output lands at
``golden/cached_smoke.jsonl`` together with a hash that must be pinned
in ``MANIFEST.toml`` ``[smoke_set].sha256``.

Usage::

    python scripts/build_cached_smoke.py --datasets bioasq scihorizon-gene \\
        --items-per-dataset 25 --output golden/cached_smoke.jsonl
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


async def _record(args: argparse.Namespace) -> int:
    from framework_eval.loader import load_from_hub

    from framework_chi.cascade.client import V14CascadeClient

    client = V14CascadeClient()
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, object]] = []
    for cfg in args.datasets:
        items = load_from_hub(cfg, allow_main=False)[: args.items_per_dataset]
        for it in items:
            try:
                pred = await client.generate(it)
            except Exception as exc:  # noqa: BLE001
                print(f"  {cfg}:{it.id} FAILED ({exc}); skipping", file=sys.stderr)
                continue
            rows.append(
                {
                    "id": it.id,
                    "dataset": it.dataset,
                    "subtask": it.question_type,
                    "predicted": pred.answer,
                    "ground_truth": it.answer,
                    "stage": pred.extras.get("stage"),
                }
            )

    await client.aclose()
    output.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n")
    sha = hashlib.sha256(output.read_bytes()).hexdigest()
    print(f"wrote {output} ({len(rows)} items)")
    print(f"sha256={sha}")
    print("Update MANIFEST.toml [smoke_set].sha256 with this value before committing.")
    return 0


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--datasets", nargs="+", default=["bioasq", "scihorizon-gene"])
    p.add_argument("--items-per-dataset", type=int, default=25)
    p.add_argument("--output", default="golden/cached_smoke.jsonl")
    args = p.parse_args()
    return asyncio.run(_record(args))


if __name__ == "__main__":
    sys.exit(main())
