#!/usr/bin/env bash
# Ours, LitQA2 base run ("agentft"): method v14-cascade-dual-rerank-grounded on the 199
# LitQA2 items over the PMC full-text chunk store (after the 46-paper ingest), official
# MCQ chain-of-thought protocol, 2,400-character evidence per chunk, agent full-text tools.
# Research runs: 2026-09-09 10:11-11:58, concurrency 6, prompt V_pre (all three runs were
# started before the 2026-09-09 audit edit).
#   RUNS=1 (default) produces run 1, whose answers fill the 57 non-re-run items of the
#   61.8 cell (ours_litqa2_rerun142.sh, then scripts/splice_litqa2_headline.py).
#   RUNS=3 produces the three runs behind footnote b's 57.8 (57.79 / 59.30 / 56.28).
source "$(dirname "$0")/_common.sh"
export XC_V14_PROMPT_FILE="$ROOT/prompts/V_pre.txt"
export XC_MCQ_OFFICIAL=1 XC_MCQ_OFFICIAL_TOKENS=8192 XC_FULLTEXT_CHUNKS=1 XC_RERANK_RAW=1 \
       XC_EVIDENCE_DOC_CHARS=2400 XC_AGENT_FULLTEXT_TOOLS=1 XC_FAULTHANDLER=1
unset XC_ESCALATE_ON_ABSTAIN XC_REJUDGE_AGENT_TEXT XC_REJUDGE_AGENT_EVIDENCE \
      XC_ANSWER_CONTEXT_CHARS XC_AGENT_MAX_ITERS
F=output/unified_benchmark/ablation/v14-cascade-dual-rerank-grounded/litqa2.jsonl
mkdir -p output/litqa2_runs
for run in $(seq 1 "${RUNS:-1}"); do
  rm -f "$F"
  "$PY" "${RUNNER[@]}" --method v14-cascade-dual-rerank-grounded --datasets litqa2 --concurrency 6
  cp "$F" "output/litqa2_runs/ours_agentft_run${run}.jsonl"
done
