"""Aggregate per-item continuous scores into figures/data_continuous/ CSVs.

For each `type:item` record:
  - factoid  -> score from *.tokenf1.jsonl sidecar (SQuAD token-F1)
  - else     -> existing `score` field already in run.jsonl
The pooled cell value is `100 * mean(score_i)` over items in that cell.

Reads:
  output/unified_benchmark/baselines/{method}/{dataset}.jsonl + .tokenf1.jsonl
  output/unified_benchmark/ablation/{method}/{dataset}.jsonl  + .tokenf1.jsonl
  output/unified_benchmark/scaling/{size}/{method}/run.jsonl + .tokenf1.jsonl
  benchmark/unified/sample_1-10_seed42.jsonl   (sample filter for scaling)

Writes:
  figures/data_continuous/overall.csv
  figures/data_continuous/main_results.csv
  figures/data_continuous/subtask_results.csv
  figures/data_continuous/scaling_results.csv
  figures/data_continuous/scaling_per_dataset.csv
"""
from __future__ import annotations

import argparse
import csv
import json
import statistics
from collections import defaultdict
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parent.parent
OUTPUT_ROOT = ROOT / "output" / "unified_benchmark"
SAMPLE_FILTER = ROOT / "benchmark" / "unified" / "sample_1-10_seed42.jsonl"
OUT_DIR = ROOT / "figures" / "data_continuous"


DATASET_TOTALS = {
    "pubmedqa_pqal": 500, "geneturing": 1178, "medqa_US": 1273,
    "medqa_Taiwan": 1413, "scihorizon_hgkb": 2610, "bioasq": 4719,
    "medmcqa": 4183, "medqa_Mainland": 3426,
}
FULL_SUITE_TOTAL = sum(DATASET_TOTALS.values())  # 19302

ACTIVE_PARAMS = {
    "0.8B": 0.8, "2B": 2.0, "4B": 4.0, "9B": 9.0,
    "27B": 27.0, "35B-A3B": 3.0, "122B-A10B": 10.0,
}


# ---------- I/O helpers ----------

def _load_sidecar(path: Path) -> dict[str, float]:
    """item_id -> score_factoid_tokenf1."""
    out: dict[str, float] = {}
    if not path.exists():
        return out
    with path.open() as fh:
        for line in fh:
            try:
                d = json.loads(line)
            except json.JSONDecodeError:
                continue
            iid = d.get("id")
            if iid is not None:
                out[iid] = float(d.get("score_factoid_tokenf1", 0.0))
    return out


def _stream_items(jsonl: Path):
    """Yield item dicts (`type:item`) from a run jsonl."""
    with jsonl.open() as fh:
        for line in fh:
            try:
                d = json.loads(line)
            except json.JSONDecodeError:
                continue
            if d.get("type") == "item":
                yield d


def _item_score(d: dict, sidecar: dict[str, float]) -> float:
    """Return continuous score s_i for this item."""
    if d.get("subtask") == "factoid":
        return sidecar.get(d.get("id"), float(d.get("score", 0.0)))
    return float(d.get("score", 0.0))


def _latest_run_jsonl(method_dir: Path) -> Path | None:
    cands = sorted(method_dir.rglob("run.jsonl"),
                   key=lambda p: p.stat().st_mtime, reverse=True)
    cands = [c for c in cands if not c.name.endswith(".tokenf1.jsonl")]
    return cands[0] if cands else None


def _bootstrap_ci(scores: list[float], n_resamples: int = 1000, seed: int = 42):
    if not scores:
        return float("nan"), float("nan"), float("nan")
    arr = np.array(scores, dtype=float)
    rng = np.random.default_rng(seed)
    n = len(arr)
    means = np.empty(n_resamples)
    for i in range(n_resamples):
        idx = rng.integers(0, n, size=n)
        means[i] = arr[idx].mean()
    return float(arr.mean()), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


# ---------- Per-method aggregation (baselines + ablation) ----------

def aggregate_methods(method_roots: list[Path]):
    """Walk method_roots = [baselines/, ablation/].

    Returns three lists of dicts:
      overall_rows  (one per method)
      main_rows     (one per method,dataset)
      subtask_rows  (one per method,dataset,subtask)
    """
    # method -> {dataset -> {subtask -> [scores]}}
    bucket: dict[str, dict[str, dict[str, list[float]]]] = defaultdict(
        lambda: defaultdict(lambda: defaultdict(list))
    )
    # method -> latency_ms list (per item where present)
    lat_bucket: dict[str, list[float]] = defaultdict(list)

    for root in method_roots:
        if not root.exists():
            continue
        for mdir in sorted(p for p in root.iterdir() if p.is_dir()):
            method = mdir.name
            for jsonl in sorted(mdir.glob("*.jsonl")):
                if jsonl.name.endswith(".tokenf1.jsonl"):
                    continue
                sidecar = _load_sidecar(jsonl.with_suffix(".tokenf1.jsonl"))
                for d in _stream_items(jsonl):
                    ds = d.get("dataset")
                    sub = d.get("subtask")
                    if ds is None or sub is None:
                        continue
                    s = _item_score(d, sidecar)
                    bucket[method][ds][sub].append(s)
                    lm = d.get("latency_ms")
                    if isinstance(lm, (int, float)) and lm > 0:
                        lat_bucket[method].append(float(lm))

    overall_rows: list[dict] = []
    main_rows: list[dict] = []
    subtask_rows: list[dict] = []
    for method, ds_map in bucket.items():
        all_scores: list[float] = []
        for ds, sub_map in ds_map.items():
            ds_scores: list[float] = []
            for sub, scores in sub_map.items():
                n = len(scores)
                if n == 0:
                    continue
                score_sum = float(sum(scores))
                subtask_rows.append({
                    "method": method, "dataset": ds, "subtask": sub,
                    "n_items": n,
                    "correct": round(score_sum, 6),
                    "accuracy": score_sum / n,
                })
                ds_scores.extend(scores)
            n_ds = len(ds_scores)
            if n_ds == 0:
                continue
            ds_sum = float(sum(ds_scores))
            total = DATASET_TOTALS.get(ds, n_ds)
            main_rows.append({
                "method": method, "dataset": ds,
                "n_items": n_ds,
                "correct": round(ds_sum, 6),
                "accuracy": ds_sum / n_ds,
                "total_in_dataset": total,
                "coverage_pct": n_ds / total if total else 1.0,
            })
            all_scores.extend(ds_scores)

        n = len(all_scores)
        if n == 0:
            continue
        s_sum = float(sum(all_scores))
        lat = lat_bucket.get(method, [])
        lat_med = statistics.median(lat) / 1000.0 if lat else float("nan")
        lat_mean = (sum(lat) / len(lat)) / 1000.0 if lat else float("nan")
        lat_p95 = float(np.percentile(lat, 95)) / 1000.0 if lat else float("nan")
        overall_rows.append({
            "method": method,
            "n_items": n,
            "correct": round(s_sum, 6),
            "accuracy": s_sum / n,
            "is_full_suite": n == FULL_SUITE_TOTAL,
            "latency_median_s": lat_med if lat else "",
            "latency_mean_s":   lat_mean if lat else "",
            "latency_p95_s":    lat_p95 if lat else "",
        })

    # Sort overall by accuracy desc (matches original overall.csv ordering)
    overall_rows.sort(key=lambda r: -r["accuracy"])
    main_rows.sort(key=lambda r: (r["method"], r["dataset"]))
    subtask_rows.sort(key=lambda r: (r["method"], r["dataset"], r["subtask"]))
    return overall_rows, main_rows, subtask_rows


# ---------- Scaling aggregation ----------

def _load_sample_ids() -> set[str]:
    out: set[str] = set()
    if not SAMPLE_FILTER.exists():
        raise SystemExit(f"sample filter not found: {SAMPLE_FILTER}")
    with SAMPLE_FILTER.open() as fh:
        for line in fh:
            try:
                d = json.loads(line)
            except json.JSONDecodeError:
                continue
            if d.get("route_disco") is True:
                out.add(d["id"])
    return out


def aggregate_scaling(scaling_root: Path, sample_ids: set[str], n_resamples: int = 1000):
    wide: list[dict] = []
    long: list[dict] = []
    if not scaling_root.exists():
        raise SystemExit(f"scaling root not found: {scaling_root}")
    for size_dir in sorted(p for p in scaling_root.iterdir() if p.is_dir()):
        size = size_dir.name
        for method_dir in sorted(p for p in size_dir.iterdir() if p.is_dir()):
            method = method_dir.name
            run_jsonl = _latest_run_jsonl(method_dir)
            if run_jsonl is None:
                wide.append({
                    "model_size": size, "method": method,
                    "n_items_expected": len(sample_ids),
                    "n_items_seen": 0,
                    "accuracy": float("nan"),
                    "accuracy_ci_lo": float("nan"),
                    "accuracy_ci_hi": float("nan"),
                    "latency_median_s": float("nan"),
                    "latency_p95_s": float("nan"),
                    "status": "NO_DATA",
                })
                continue
            sidecar = _load_sidecar(run_jsonl.with_suffix(".tokenf1.jsonl"))
            in_sample_scores: list[float] = []
            in_sample_lats: list[float] = []
            by_ds: dict[str, list[float]] = defaultdict(list)
            seen_ids: set[str] = set()
            for d in _stream_items(run_jsonl):
                iid = d.get("id")
                if iid is None:
                    continue
                seen_ids.add(iid)
                if iid not in sample_ids:
                    continue
                s = _item_score(d, sidecar)
                in_sample_scores.append(s)
                ds = d.get("dataset")
                if ds:
                    by_ds[ds].append(s)
                lm = d.get("latency_ms")
                if isinstance(lm, (int, float)) and lm > 0:
                    in_sample_lats.append(float(lm))
            only_sample = len(sample_ids - seen_ids)
            only_run = len(seen_ids - sample_ids)
            mean_acc, ci_lo, ci_hi = _bootstrap_ci(in_sample_scores, n_resamples)
            if in_sample_lats:
                lat_med = statistics.median(in_sample_lats) / 1000.0
                lat_p95 = float(np.percentile(in_sample_lats, 95)) / 1000.0
            else:
                lat_med = lat_p95 = float("nan")
            wide.append({
                "model_size": size, "method": method,
                "n_items_expected": len(sample_ids),
                "n_items_seen": len(in_sample_scores),
                "accuracy":         round(mean_acc * 100, 3),
                "accuracy_ci_lo":   round(ci_lo * 100, 3),
                "accuracy_ci_hi":   round(ci_hi * 100, 3),
                "latency_median_s": round(lat_med, 2) if lat_med == lat_med else float("nan"),
                "latency_p95_s":    round(lat_p95, 2) if lat_p95 == lat_p95 else float("nan"),
                "status": "OK" if (only_sample == 0 and only_run == 0) else "DRIFT",
            })
            for ds, ss in by_ds.items():
                long.append({
                    "model_size": size, "method": method,
                    "dataset": ds, "n_items": len(ss),
                    "accuracy": round(100 * sum(ss) / max(len(ss), 1), 3),
                })
    return wide, long


# ---------- Writers ----------

def _write_csv(rows: list[dict], path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as fh:
        if not rows:
            return
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        for r in rows:
            w.writerow(r)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    ap.add_argument("--out-dir",     type=Path, default=OUT_DIR)
    ap.add_argument("--n-resamples", type=int,  default=1000)
    args = ap.parse_args()

    method_roots = [args.output_root / "baselines", args.output_root / "ablation"]
    overall_rows, main_rows, subtask_rows = aggregate_methods(method_roots)
    _write_csv(overall_rows, args.out_dir / "overall.csv")
    _write_csv(main_rows,    args.out_dir / "main_results.csv")
    _write_csv(subtask_rows, args.out_dir / "subtask_results.csv")
    print(f"Wrote {len(overall_rows)} method rows -> overall.csv")
    print(f"Wrote {len(main_rows)} method,dataset rows -> main_results.csv")
    print(f"Wrote {len(subtask_rows)} method,dataset,subtask rows -> subtask_results.csv")

    # Scaling
    sample_ids = _load_sample_ids()
    print(f"Sample ids: {len(sample_ids)}")
    scaling_wide, scaling_long = aggregate_scaling(
        args.output_root / "scaling", sample_ids, args.n_resamples
    )
    _write_csv(scaling_wide, args.out_dir / "scaling_results.csv")
    _write_csv(scaling_long, args.out_dir / "scaling_per_dataset.csv")
    print(f"Wrote {len(scaling_wide)} scaling cells -> scaling_results.csv")
    print(f"Wrote {len(scaling_long)} scaling per-dataset rows -> scaling_per_dataset.csv")

    # Scaling matrix preview
    if scaling_wide:
        print("\nScaling preview (continuous):")
        sizes = sorted({r["model_size"] for r in scaling_wide},
                       key=lambda s: ACTIVE_PARAMS.get(s, 0))
        methods = sorted({r["method"] for r in scaling_wide})
        cell = {(r["model_size"], r["method"]): r["accuracy"] for r in scaling_wide}
        print(f"{'method':40s}  " + "  ".join(f"{s:>10s}" for s in sizes))
        for m in methods:
            cells = [f"{cell.get((s, m), float('nan')):>10.2f}" for s in sizes]
            print(f"{m:40s}  " + "  ".join(cells))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
