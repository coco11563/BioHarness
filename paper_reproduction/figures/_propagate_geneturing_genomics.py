"""One-off: propagate the GeneTuring genomics-fix rerun into the headline cells.

Source: output/unified_benchmark/ablation/v14-geneturing-genomics/geneturing.jsonl
  (= the headline v14_cascade config rerun on GeneTuring with the genomics tool
   fix: BLAST / dbSNP / SNP-location / protein-coding resolvers).

All 1178 GeneTuring items are `factoid`; we score them with token-F1 (the paper's
continuous convention for factoid), NOT the stored ROUGE score. This updates ONLY
the headline (bioharness-headline). Ablation variants are left at their pre-fix
GeneTuring values by design (the table caption states this).

Cells updated:
  - data_continuous/main_results.csv   : bioharness-headline, geneturing
  - data_continuous/subtask_results.csv: bioharness-headline, geneturing, factoid
  - data_continuous/overall.csv        : bioharness-headline (pooled, n unchanged)
"""
from __future__ import annotations
from pathlib import Path
import json, re, string, csv
from collections import Counter

ROOT = Path(__file__).parent
DATA = ROOT / "data_continuous"
SRC = ROOT.parent / "output/unified_benchmark/ablation/v14-geneturing-genomics/geneturing.jsonl"
HEAD = "bioharness-headline"
OLD_GENE_CORRECT = 339.285714  # current headline geneturing correct-equiv (token-F1, pre-fix)

_ART = re.compile(r"\b(a|an|the)\b", re.I)
_P = re.compile(f"[{re.escape(string.punctuation)}]")


def norm(s):
    s = _P.sub(" ", s.lower()); s = _ART.sub(" ", s); return s.split()


def tf1(p, g):
    p = norm(p); g = norm(g)
    if not p and not g: return 1.0
    if not p or not g: return 0.0
    c = Counter(p) & Counter(g); ns = sum(c.values())
    if ns == 0: return 0.0
    pr = ns / len(p); rc = ns / len(g)
    return 2 * pr * rc / (pr + rc)


def new_geneturing_correct():
    rows = [json.loads(l) for l in open(SRC) if l.strip()]
    rows = [r for r in rows if r.get("type") == "item"]
    assert len(rows) == 1178, len(rows)
    assert all(r["subtask"] == "factoid" for r in rows)
    tot = 0.0
    for r in rows:
        gt = r["ground_truth"]
        try:
            gj = json.loads(gt) if isinstance(gt, str) and gt.strip().startswith("[") else [gt]
        except Exception:
            gj = [gt]
        tot += max(tf1(r.get("predicted", "") or "", str(g)) for g in gj)
    return tot, len(rows)


def patch_csv(path, match, new_correct, n):
    rows = list(csv.DictReader(open(path)))
    hdr = rows[0].keys()
    hit = 0
    for r in rows:
        if all(r[k] == v for k, v in match.items()):
            r["correct"] = f"{new_correct:.6f}"
            r["accuracy"] = f"{new_correct / n:.16f}"
            hit += 1
    assert hit == 1, f"{path.name}: expected 1 match, got {hit}"
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(hdr)); w.writeheader(); w.writerows(rows)


def patch_overall(path, new_gene_correct):
    rows = list(csv.DictReader(open(path)))
    hdr = rows[0].keys()
    hit = 0
    for r in rows:
        if r["method"] == HEAD:
            n = float(r["n_items"])
            new_c = float(r["correct"]) - OLD_GENE_CORRECT + new_gene_correct
            r["correct"] = f"{new_c:.10f}"
            r["accuracy"] = f"{new_c / n:.16f}"
            hit += 1
    assert hit == 1
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(hdr)); w.writeheader(); w.writerows(rows)


def main():
    new_c, n = new_geneturing_correct()
    print(f"GeneTuring headline (genomics fix, token-F1): correct={new_c:.4f}/{n} = {new_c/n*100:.2f}%")
    print(f"  was {OLD_GENE_CORRECT:.4f}/{n} = {OLD_GENE_CORRECT/n*100:.2f}%  (+{(new_c-OLD_GENE_CORRECT)/n*100:.2f} pp)")
    patch_csv(DATA / "main_results.csv", {"method": HEAD, "dataset": "geneturing"}, new_c, n)
    patch_csv(DATA / "subtask_results.csv",
              {"method": HEAD, "dataset": "geneturing", "subtask": "factoid"}, new_c, n)
    patch_overall(DATA / "overall.csv", new_c)
    print("Patched main_results.csv, subtask_results.csv, overall.csv (headline only).")


if __name__ == "__main__":
    main()
