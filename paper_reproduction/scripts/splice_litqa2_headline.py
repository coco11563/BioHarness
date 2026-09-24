"""Rebuild the LitQA2 cell of the Ours row in Table 1 (61.8, footnote b).

The printed value is a splice of two runs of method
``v14-cascade-dual-rerank-grounded`` on the 199 LitQA2 items:

* ``--rerun``: the 142 items re-run on 2026-09-11 with ``runs/ours_litqa2_rerun142.sh``
  (the "defaultcaps" traced run, current V14 prompt V_cur);
* ``--base``: the full 199-item ``agentft`` run 1 of 2026-09-09
  (``runs/ours_litqa2_agentft.sh``); its items are used for the 57 ids that were
  not re-run.

Every id present in ``--rerun`` replaces the same id in ``--base``. The cell is the
mean of the per-item official score over the 199 base ids. This is a single spliced
run, not a held-out result; the three-run mean of the earlier configuration
(``agentft`` runs 1-3) is 57.8.

Official per-item score: token-F1 for ``subtask == "factoid"`` (SQuAD normalisation,
max over gold variants; ``scripts/rescore_factoid_tokenf1.py``), otherwise the stored
``score``. Records with a null dataset or subtask are skipped and the first record per
id is kept.
LitQA2 items are all ``mcq``, so only the stored score is used here.

Usage::

    python scripts/splice_litqa2_headline.py \\
        --rerun litqa2ft_v14-cascade-dual-rerank-grounded_traced_defaultcaps.jsonl \\
        --base  litqa2ft_v14-cascade-dual-rerank-grounded_agentft_run1.jsonl \\
        [--out spliced.jsonl]

Expected output with the research-run files: 142 re-run + 57 base = 199 items,
LitQA2 = 61.809.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from rescore_factoid_tokenf1 import score_factoid  # noqa: E402  (same scorer as the sidecars)


def item_score(rec: dict) -> float:
    if rec["subtask"] == "factoid":
        return score_factoid(rec.get("predicted") or "", rec.get("ground_truth"))
    return float(rec.get("score") or 0.0)


def load(path: str) -> dict[str, dict]:
    out: dict[str, dict] = {}
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            if rec.get("type") not in (None, "item"):
                continue
            if rec.get("dataset") is None or rec.get("subtask") is None:
                continue
            out.setdefault(rec["id"], rec)  # first occurrence wins
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--rerun", required=True, help="jsonl of the re-run items (142)")
    ap.add_argument("--base", required=True, help="jsonl of the full base run (199)")
    ap.add_argument("--out", help="optional path for the spliced per-item jsonl")
    args = ap.parse_args()

    rerun, base = load(args.rerun), load(args.base)
    extra = set(rerun) - set(base)
    if extra:
        raise SystemExit(f"{len(extra)} re-run ids are not in the base run")

    spliced = {i: rerun.get(i, rec) for i, rec in base.items()}
    n_rerun = sum(1 for i in base if i in rerun)
    mean = 100.0 * sum(item_score(r) for r in spliced.values()) / len(spliced)
    rerun_only = 100.0 * sum(item_score(r) for r in rerun.values()) / max(1, len(rerun))

    print(f"base items           : {len(base)}")
    print(f"from re-run          : {n_rerun}  (mean on those items {rerun_only:.3f})")
    print(f"from base            : {len(base) - n_rerun}")
    print(f"LitQA2 (spliced)     : {mean:.3f}")

    if args.out:
        with open(args.out, "w", encoding="utf-8") as fh:
            for rec in spliced.values():
                fh.write(json.dumps(rec, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
