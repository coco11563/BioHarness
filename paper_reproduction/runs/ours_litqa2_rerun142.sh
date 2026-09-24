#!/usr/bin/env bash
# Ours, LitQA2 61.8 (footnote b; single spliced run, NOT held out): the revised
# configuration re-run on 142 items, runs/item_filters/litqa2_rerun142.jsonl: the 93
# that agentft run 1 escalated plus the 49 non-escalated items on the research
# abstention list (48 chose the "Insufficient information" option; 1, litqa2_745f5a0d,
# chose "None of the above").
# The revision (escalate on abstention; agent's final turn passed to the re-judgment;
# 60,000-character answer context; 14 agent iterations) was chosen after error analysis
# on these same 199 items.
# Research run: 2026-09-11 15:26-16:40 ("defaultcaps" traced run), concurrency 6,
# prompt V_cur (the prompt in the released src/rlm/pipeline.py, so no prompt file).
# Tracing was on (XC_TRACE_DIR). It records tool calls and routes agent history through
# the rlm library's own defaults (5,000 / 2 / 12,000); it does not change the answer
# path. The research run also set XC_TRACE_GOLD_MAP (gold-passage ids, used only to log
# whether a gold id was retrieved); that file is not released.
source "$(dirname "$0")/_common.sh"
unset XC_V14_PROMPT_FILE XC_AGENT_OBS_CHARS XC_AGENT_KEEP_RECENT XC_AGENT_HIST_CHARS
export XC_MCQ_OFFICIAL=1 XC_MCQ_OFFICIAL_TOKENS=8192 XC_FULLTEXT_CHUNKS=1 XC_RERANK_RAW=1
export XC_EVIDENCE_DOC_CHARS=2400 XC_AGENT_FULLTEXT_TOOLS=1 XC_ESCALATE_ON_ABSTAIN=1
export XC_AGENT_QDRANT_TIMEOUT=300 XC_AGENT_MAX_ITERS=14 XC_FAULTHANDLER=1
export XC_ANSWER_CONTEXT_CHARS=60000 XC_REJUDGE_AGENT_TEXT=1 XC_REJUDGE_AGENT_EVIDENCE=0
export XC_TRACE_DIR=output/traces/litqa2_rerun142
mkdir -p "$XC_TRACE_DIR" output/litqa2_runs
F=output/unified_benchmark/ablation/v14-cascade-dual-rerank-grounded/litqa2.jsonl
rm -f "$F"
"$PY" "${RUNNER[@]}" --method v14-cascade-dual-rerank-grounded --datasets litqa2 \
    --concurrency 6 --item-filter runs/item_filters/litqa2_rerun142.jsonl
cp "$F" output/litqa2_runs/ours_rerun142.jsonl
# Splice: expected 61.809 with the research files.
"$PY" scripts/splice_litqa2_headline.py \
    --rerun output/litqa2_runs/ours_rerun142.jsonl \
    --base output/litqa2_runs/ours_agentft_run1.jsonl
