#!/usr/bin/env bash
# LitQA2 column for a baseline row: three runs, reported as the mean (597 item-runs).
# Usage:  runs/baselines_litqa2_three_runs.sh <method-id>
# Protocol (pre-declared): full-text chunk retrieval over the shared index after the
# 46-paper ingest, official MCQ chain-of-thought format with an 8,192-token rationale,
# completion-template reranker (XC_RERANK_RAW=1), concurrency 6, answer-time evidence
# cap 8,000 characters.
#   rt-graphrag / rt-pathrag / rt-lightrag: same flags; with XC_FULLTEXT_CHUNKS=1 the RT-KG
#     adapter keys documents per chunk instead of per PMID (see retrieve_then_kg.py).
#   biorag-full: answers from PubMed abstracts found by its own tools (footnote c).
#   golden-context: runs on litqa2_gold (LAB-Bench key passage as context, 194 of 199
#     items); that file is not released: build it from LitQA2 with "context" set to the
#     item's key passage and "dataset" set to "litqa2_gold".
# Research runs: run 1 on 2026-09-07/08, runs 2-3 on 2026-09-08/09 (graph baselines
# 2026-09-14, FLARE 2026-09-13/14). Run 1 of methods that never rerank was made before
# XC_RERANK_RAW existed; it has no effect on them.
source "$(dirname "$0")/_common.sh"
M="${1:?usage: $0 <method-id>}"
export XC_MCQ_OFFICIAL=1 XC_MCQ_OFFICIAL_TOKENS=8192 XC_FULLTEXT_CHUNKS=1 XC_RERANK_RAW=1 XC_FAULTHANDLER=1
unset XC_EVIDENCE_DOC_CHARS XC_ANSWER_CONTEXT_CHARS XC_ESCALATE_ON_ABSTAIN XC_REJUDGE_AGENT_TEXT \
      XC_REJUDGE_AGENT_EVIDENCE XC_AGENT_FULLTEXT_TOOLS
DS=litqa2; EXTRA=(); GROUP=baselines
if [ "$M" = golden-context ]; then DS=litqa2_gold; EXTRA=(--use-golden-context); fi
F=output/unified_benchmark/$GROUP/$M/$DS.jsonl
mkdir -p output/litqa2_runs
for run in 1 2 3; do
  rm -f "$F"
  "$PY" "${RUNNER[@]}" --method "$M" --datasets "$DS" --concurrency 6 ${EXTRA[@]+"${EXTRA[@]}"}
  cp "$F" "output/litqa2_runs/${M}_run${run}.jsonl"
done
"$PY" - "$M" <<'PY'
import json, sys
m = sys.argv[1]
vals = []
for run in (1, 2, 3):
    recs = {}
    for line in open(f"output/litqa2_runs/{m}_run{run}.jsonl"):
        d = json.loads(line)
        if d.get("type", "item") == "item" and d.get("dataset") and d.get("subtask"):
            recs.setdefault(d["id"], float(d.get("score") or 0))
    vals.append(100 * sum(recs.values()) / len(recs))
print(m, " / ".join(f"{v:.2f}" for v in vals), f"mean {sum(vals)/3:.2f}")
PY
