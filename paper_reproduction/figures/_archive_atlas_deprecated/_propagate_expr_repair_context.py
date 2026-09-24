"""One-off: propagate the repair-context expression score into the continuous
CSVs (subtask / main_results / overall).

Authoritative expression source: output/case_study_repair_context.jsonl
(all-210 set-F1, empty-GT auto-1.0). D-bearing variants take ours(+D); the
-A variant (no atlas) takes ours_minusD(-D), per the existing -A->-D convention.

Updates, per variant:
  - subtask_results.csv  expression row -> new absolute correct/accuracy
  - main_results.csv     scihorizon row -> correct += (new_expr - old_expr)
  - overall.csv          row            -> correct += (new_expr - old_expr)
"""
from __future__ import annotations
import csv
import json
from pathlib import Path

ROOT = Path(__file__).parent
DATA = ROOT.parent / "data_continuous"
CASE = ROOT.parent.parent / "output/case_study_repair_context.jsonl"
EXPR_N = 210

# variant -> which case-study arm supplies its expression score
D_BEARING = {
    "bioharness-headline", "v14-forced-agent-dual-rerank",
    "v14-cascade-dual-rerank-grounded-no-tools", "v14-cascade-grounded",
    "v14-forced-agent",
}
MINUS_A = "v14-forced-fast"
VARIANTS = D_BEARING | {MINUS_A}


def all210(rows, path):
    ne = [r for r in rows if not r["is_empty_gt"]]
    emp = [r for r in rows if r["is_empty_gt"]]
    s = 0.0
    for r in ne:
        o = r
        for k in path.split("."):
            o = o[k]
        s += o
    return (s + 1.0 * len(emp)) / len(rows)  # fraction in [0,1]


def main():
    rows = [json.loads(l) for l in open(CASE)]
    acc_pD = all210(rows, "ours.f1")          # D-bearing expression accuracy
    acc_mD = all210(rows, "ours_minusD.f1")   # -A expression accuracy
    print(f"authoritative expr accuracy: +D={acc_pD*100:.3f}%  -D(-A)={acc_mD*100:.3f}%")

    def new_acc(method):
        return acc_pD if method in D_BEARING else acc_mD

    # --- subtask_results.csv: set absolute new expression correct/accuracy ---
    sp = DATA / "subtask_results.csv"
    srows = list(csv.DictReader(open(sp)))
    sfields = srows[0].keys()
    old_expr = {}  # method -> old correct (to compute delta for main/overall)
    for r in srows:
        if (r["method"] in VARIANTS and r["dataset"] == "scihorizon_hgkb"
                and r["subtask"] == "expression"):
            old_expr[r["method"]] = float(r["correct"])
            a = new_acc(r["method"])
            r["correct"] = f"{a * EXPR_N:.6f}"
            r["accuracy"] = f"{a:.10f}"
    with open(sp, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(sfields))
        w.writeheader(); w.writerows(srows)

    delta = {m: new_acc(m) * EXPR_N - old_expr[m] for m in VARIANTS}
    for m in sorted(VARIANTS):
        print(f"  {m:46s} old_expr={old_expr[m]:.3f} delta={delta[m]:+.3f}")

    # --- main_results.csv: adjust scihorizon row ---
    mp = DATA / "main_results.csv"
    mrows = list(csv.DictReader(open(mp)))
    mfields = mrows[0].keys()
    for r in mrows:
        if r["method"] in VARIANTS and r["dataset"] == "scihorizon_hgkb":
            n = float(r["n_items"])
            c = float(r["correct"]) + delta[r["method"]]
            r["correct"] = f"{c:.10f}"
            r["accuracy"] = f"{c / n:.16f}"
    with open(mp, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(mfields))
        w.writeheader(); w.writerows(mrows)

    # --- overall.csv: adjust overall row ---
    op = DATA / "overall.csv"
    orows = list(csv.DictReader(open(op)))
    ofields = orows[0].keys()
    for r in orows:
        if r["method"] in VARIANTS:
            n = float(r["n_items"])
            c = float(r["correct"]) + delta[r["method"]]
            r["correct"] = f"{c:.10f}"
            r["accuracy"] = f"{c / n:.16f}"
    with open(op, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(ofields))
        w.writeheader(); w.writerows(orows)

    print("updated subtask_results.csv, main_results.csv, overall.csv")


if __name__ == "__main__":
    main()
