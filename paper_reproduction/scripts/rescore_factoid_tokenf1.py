"""Offline rescore of factoid items with SQuAD-style token-F1.

Walks every run.jsonl / per-dataset.jsonl in:
  output/unified_benchmark/baselines/{method}/{dataset}.jsonl
  output/unified_benchmark/ablation/{method}/{dataset}.jsonl
  output/unified_benchmark/scaling/{size}/{method}/run.jsonl

For each `type:item` record with `subtask == "factoid"`, computes
    token_f1(predicted, ground_truth)
using the SQuAD/BioASQ convention (lowercase, strip articles a/an/the,
strip punctuation, whitespace split, multiset-F1 over token bags).

If ground_truth parses as a JSON list (possibly nested for BioASQ
synonym groups), the score is max(token_f1(pred, variant)) over all
flattened gold strings.

Writes a sidecar `*.tokenf1.jsonl` next to each source file. Sidecar
lines: {id, dataset, score_factoid_tokenf1}. Source files are NOT
modified.

Run:
    python3 scripts/rescore_factoid_tokenf1.py
"""
from __future__ import annotations

import argparse
import json
import re
import string
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
OUTPUT_ROOT = ROOT / "output" / "unified_benchmark"


_ARTICLES = re.compile(r"\b(a|an|the)\b", re.IGNORECASE)
_PUNCT_TRANS = str.maketrans({c: " " for c in string.punctuation})


def _normalize(text: str) -> list[str]:
    """SQuAD-style normalization: lowercase, strip punctuation/articles, whitespace split."""
    if not text:
        return []
    s = text.lower()
    s = s.translate(_PUNCT_TRANS)
    s = _ARTICLES.sub(" ", s)
    return s.split()


def token_f1(pred: str, gold: str) -> float:
    """SQuAD token-F1: multiset-F1 over whitespace-tokenized, normalized strings."""
    p = _normalize(pred)
    g = _normalize(gold)
    if not p and not g:
        return 1.0
    if not p or not g:
        return 0.0
    common = Counter(p) & Counter(g)
    n_same = sum(common.values())
    if n_same == 0:
        return 0.0
    precision = n_same / len(p)
    recall = n_same / len(g)
    return 2 * precision * recall / (precision + recall)


def _parse_gold_variants(gt_raw: str) -> list[str]:
    """Return a flat list of gold-string variants.

    Plain string  ->  [gt_raw]
    JSON list     ->  flat list of items
    JSON nested   ->  flat list of all leaves (BioASQ synonym groups)
    """
    if gt_raw is None:
        return [""]
    gt_raw = gt_raw.strip()
    if not gt_raw:
        return [""]
    if gt_raw[0] in "[{\"":
        try:
            data = json.loads(gt_raw)
        except json.JSONDecodeError:
            return [gt_raw]
        flat: list[str] = []
        def _flatten(x):
            if isinstance(x, list):
                for y in x:
                    _flatten(y)
            elif isinstance(x, dict):
                for v in x.values():
                    _flatten(v)
            elif x is not None:
                flat.append(str(x))
        _flatten(data)
        return flat if flat else [gt_raw]
    return [gt_raw]


def score_factoid(predicted: str, ground_truth: str) -> float:
    variants = _parse_gold_variants(ground_truth)
    return max(token_f1(predicted or "", v) for v in variants)


def _rescore_file(src: Path) -> tuple[int, int]:
    """Write sidecar next to src; return (n_factoid_seen, n_lines_written)."""
    dst = src.with_suffix(".tokenf1.jsonl")
    n_seen = 0
    n_written = 0
    with src.open() as fin, dst.open("w") as fout:
        for line in fin:
            try:
                d = json.loads(line)
            except json.JSONDecodeError:
                continue
            if d.get("type") != "item":
                continue
            if d.get("subtask") != "factoid":
                continue
            n_seen += 1
            s = score_factoid(d.get("predicted", ""), d.get("ground_truth", ""))
            out = {
                "id": d.get("id"),
                "dataset": d.get("dataset"),
                "score_factoid_tokenf1": s,
            }
            fout.write(json.dumps(out) + "\n")
            n_written += 1
    return n_seen, n_written


def _collect_targets(output_root: Path) -> list[Path]:
    targets: list[Path] = []
    # Baselines: output/unified_benchmark/baselines/{method}/{dataset}.jsonl
    bdir = output_root / "baselines"
    if bdir.exists():
        for mdir in sorted(p for p in bdir.iterdir() if p.is_dir()):
            for jsonl in sorted(mdir.glob("*.jsonl")):
                if jsonl.name.endswith(".tokenf1.jsonl"):
                    continue
                targets.append(jsonl)
    # Ablation: output/unified_benchmark/ablation/{method}/{dataset}.jsonl
    adir = output_root / "ablation"
    if adir.exists():
        for mdir in sorted(p for p in adir.iterdir() if p.is_dir()):
            for jsonl in sorted(mdir.glob("*.jsonl")):
                if jsonl.name.endswith(".tokenf1.jsonl"):
                    continue
                targets.append(jsonl)
    # Scaling: output/unified_benchmark/scaling/{size}/{method}/.../run.jsonl
    sdir = output_root / "scaling"
    if sdir.exists():
        for size_dir in sorted(p for p in sdir.iterdir() if p.is_dir()):
            for mdir in sorted(p for p in size_dir.iterdir() if p.is_dir()):
                for jsonl in sorted(mdir.rglob("run.jsonl")):
                    if jsonl.name.endswith(".tokenf1.jsonl"):
                        continue
                    targets.append(jsonl)
    return targets


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    ap.add_argument("--restrict",
                    help="Comma-separated method filter (substring match on path).",
                    default=None)
    args = ap.parse_args()

    targets = _collect_targets(args.output_root)
    if args.restrict:
        keys = [k.strip() for k in args.restrict.split(",") if k.strip()]
        targets = [t for t in targets if any(k in str(t) for k in keys)]

    print(f"Found {len(targets)} source jsonl files.")
    total_factoid = 0
    n_files_with_factoid = 0
    for i, src in enumerate(targets, 1):
        try:
            n_seen, _ = _rescore_file(src)
        except Exception as e:
            print(f"[{i}/{len(targets)}] FAIL {src}: {e}")
            continue
        total_factoid += n_seen
        if n_seen > 0:
            n_files_with_factoid += 1
        if i % 50 == 0 or i == len(targets):
            print(f"  [{i}/{len(targets)}] processed; running total factoid={total_factoid}")
    print(f"Done. {total_factoid} factoid items rescored across "
          f"{n_files_with_factoid}/{len(targets)} files.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
