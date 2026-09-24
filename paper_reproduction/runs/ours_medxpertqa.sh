#!/usr/bin/env bash
# Ours, MedXpertQA 37.1 (909 / 2450): method v14-cascade-dual-rerank-grounded (no atlas)
# under the benchmark's official zero-shot chain-of-thought protocol
# (TsinghuaC3I/MedXpertQA eval: medical-assistant system role, inline answer choices,
# "Therefore, among A through J, the answer is" trigger), rationale budget 8,192 tokens,
# abstract retrieval, completion-template reranker (XC_RERANK_RAW=1).
# Research run: 2026-09-09 14:54-17:52 (second attempt; the first was rejected at 6.0 %
# stage-1 failures), concurrency 24, run alone on the LLM server. Prompt: V_audit.
# Quality gate used: 2,450 items and <= 5 % items with stage1_confidence == 0.
# Two later measurements of the same configuration gave 37.39 and 37.71.
source "$(dirname "$0")/_common.sh"
export XC_V14_PROMPT_FILE="$ROOT/prompts/V_audit.txt"
export XC_MCQ_OFFICIAL=1 XC_MCQ_OFFICIAL_TOKENS=8192 XC_RERANK_RAW=1 XC_FAULTHANDLER=1
unset XC_FULLTEXT_CHUNKS XC_EVIDENCE_DOC_CHARS XC_ANSWER_CONTEXT_CHARS XC_ESCALATE_ON_ABSTAIN \
      XC_REJUDGE_AGENT_TEXT XC_REJUDGE_AGENT_EVIDENCE XC_AGENT_FULLTEXT_TOOLS XC_AGENT_MAX_ITERS
# benchmark/unified/medxpertqa_text.jsonl: scripts/prepare_unified_data.py (from HF) or
# rebuttal_round1/convert_medxpertqa.py (from the upstream Text/test.jsonl).
rm -f output/unified_benchmark/ablation/v14-cascade-dual-rerank-grounded/medxpertqa_text.jsonl
"$PY" "${RUNNER[@]}" --method v14-cascade-dual-rerank-grounded --datasets medxpertqa_text \
    --concurrency 24
"$PY" - <<'PY'
import json
f = "output/unified_benchmark/ablation/v14-cascade-dual-rerank-grounded/medxpertqa_text.jsonl"
r = [json.loads(l) for l in open(f) if '"type": "item"' in l]
fail = sum(1 for d in r if not (d["metadata"].get("stage1_confidence") or 0))
print(f"{len(r)} items, accuracy {100*sum(d['score'] for d in r)/len(r):.2f}, "
      f"stage-1 failures {100*fail/len(r):.1f} % (gate: <= 5 %)")
PY
