"""Build the synthetic +D headline run by merging:
  - disco run output (for items where router said route_disco=true)
  - grounded run output (for items where router said route_disco=false / unknown)

Output: output/unified_benchmark/ablation/bioharness-headline/<dataset>.jsonl

This is what `v14-cascade-dual-rerank-grounded-disco` would produce on the full
19302 items, given that the router only flags ~1.35 % for atlas augmentation.
For non-routed items disco's behavior is identical to the grounded variant
(atlas fetch is skipped), so we splice grounded answers in directly without
re-running anything.

Reproducibility:
- Disco answers: union across all timestamp dirs that contain
  v14-cascade-dual-rerank-grounded-disco runs; latest per item id wins.
- Grounded answers: ablation/v14-cascade-dual-rerank-grounded/<dataset>.jsonl.
- Routed set: items where router_predictions_filtered.jsonl says
  route_disco == True.

Run: python scripts/merge_disco_into_headline.py
"""
from __future__ import annotations
import json
from pathlib import Path
from collections import defaultdict

ROOT = Path(__file__).resolve().parent.parent
ROUTER_FILE = ROOT / ".cache/disco_router/router_predictions_filtered.jsonl"
GROUNDED_DIR = ROOT / "output/unified_benchmark/ablation/v14-cascade-dual-rerank-grounded"
TIMESTAMP_GLOB = "output/unified_benchmark/2026*/run.jsonl"
OUT_DIR = ROOT / "output/unified_benchmark/ablation/bioharness-headline"
OUT_METHOD_NAME = "bioharness-headline"

DISCO_METHOD = "v14-cascade-dual-rerank-grounded-disco"


def load_routed_ids() -> set[tuple[str, str]]:
    """Set of (dataset, id) for items where router said route_disco=True."""
    routed: set[tuple[str, str]] = set()
    with open(ROUTER_FILE) as f:
        for line in f:
            try:
                d = json.loads(line)
                if d.get("route_disco") is True:
                    routed.add((d["dataset"], d["id"]))
            except Exception:
                continue
    return routed


def collect_disco_items() -> dict[tuple[str, str], dict]:
    """Walk all timestamp dirs that ran the disco method; latest per id wins.

    Identifies a disco run.jsonl by checking the run_start record's `model`.
    """
    by_id: dict[tuple[str, str], dict] = {}
    for f in sorted(ROOT.glob(TIMESTAMP_GLOB)):
        # Inspect first line for model tag.
        try:
            with open(f) as fh:
                first = fh.readline()
            d0 = json.loads(first)
            if d0.get("type") != "run_start":
                continue
            if DISCO_METHOD not in (d0.get("model") or ""):
                continue
        except Exception:
            continue
        # Stream all item lines; later runs overwrite earlier (same id).
        with open(f) as fh:
            for line in fh:
                try:
                    d = json.loads(line)
                except Exception:
                    continue
                if d.get("type") != "item":
                    continue
                ds, _id = d.get("dataset"), d.get("id")
                if not ds or not _id:
                    continue
                # Latest timestamp wins.
                cur = by_id.get((ds, _id))
                if cur is None or d.get("timestamp", "") > cur.get("timestamp", ""):
                    by_id[(ds, _id)] = d
    return by_id


def collect_grounded_items() -> dict[tuple[str, str], dict]:
    """Per-dataset ablation jsonl files. Each line is a single item (no run_start)."""
    by_id: dict[tuple[str, str], dict] = {}
    for f in sorted(GROUNDED_DIR.glob("*.jsonl")):
        with open(f) as fh:
            for line in fh:
                try:
                    d = json.loads(line)
                except Exception:
                    continue
                ds, _id = d.get("dataset"), d.get("id")
                if not ds or not _id:
                    continue
                by_id[(ds, _id)] = d
    return by_id


def main() -> None:
    routed = load_routed_ids()
    print(f"[router] {len(routed)} items flagged route_disco=True")

    disco = collect_disco_items()
    print(f"[disco]  {len(disco)} items (union across timestamp dirs)")
    print(f"[disco]  routed-and-have-disco: "
          f"{sum(1 for k in routed if k in disco)} / {len(routed)}")

    grounded = collect_grounded_items()
    print(f"[grounded] {len(grounded)} items in ablation/v14-cascade-dual-rerank-grounded/")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    by_dataset: dict[str, list[dict]] = defaultdict(list)
    n_disco_used = 0
    n_grounded_used = 0
    n_routed_no_disco = 0  # routed but disco run never produced an answer for it

    # Iterate the grounded set (= full 19302) so non-routed items are covered.
    for (ds, _id), gitem in grounded.items():
        if (ds, _id) in routed:
            ditem = disco.get((ds, _id))
            if ditem is not None:
                # Use disco's answer; relabel method tag for clarity.
                merged = dict(ditem)
                merged["_source"] = "disco"
                n_disco_used += 1
            else:
                # Routed but no disco answer — keep grounded as fallback.
                merged = dict(gitem)
                merged["_source"] = "grounded_fallback"
                n_routed_no_disco += 1
        else:
            merged = dict(gitem)
            merged["_source"] = "grounded"
            n_grounded_used += 1
        by_dataset[ds].append(merged)

    # Write per-dataset jsonl, mirroring the ablation/<method>/ layout.
    total = 0
    for ds, items in by_dataset.items():
        # Stable order by item id.
        items.sort(key=lambda x: x.get("id", ""))
        outf = OUT_DIR / f"{ds}.jsonl"
        with open(outf, "w") as fh:
            for it in items:
                fh.write(json.dumps(it, ensure_ascii=False) + "\n")
        print(f"[write] {ds}: {len(items)} items -> {outf}")
        total += len(items)

    print()
    print(f"[summary] {total} merged items across {len(by_dataset)} datasets")
    print(f"  disco answers used:        {n_disco_used}")
    print(f"  grounded answers used:     {n_grounded_used}")
    print(f"  routed-but-fallback:       {n_routed_no_disco}")
    if total != len(grounded):
        print(f"  WARNING: total {total} != grounded {len(grounded)}")


if __name__ == "__main__":
    main()
