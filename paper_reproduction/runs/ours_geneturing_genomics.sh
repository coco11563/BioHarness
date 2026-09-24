#!/usr/bin/env bash
# Ours, GeneTuring 54.6 (footnote a): GeneTuring-only re-run of the headline
# configuration with the genomics tool layer (BLAST, dbSNP, SNP-location and
# protein-coding resolvers), method key v14-geneturing-genomics (atlas off).
# Research run: 2026-06-15 13:05-14:02, top-k 20, concurrency 8, prompt V_pre.
# Scored with token-F1: 643.27 / 1178 = 54.61. Without this splice the cell is 28.8.
# The genomics tools call public NCBI services; set NCBI_API_KEY to raise the rate limit.
source "$(dirname "$0")/_common.sh"
export XC_V14_PROMPT_FILE="$ROOT/prompts/V_pre.txt"
unset XC_MCQ_OFFICIAL XC_MCQ_OFFICIAL_TOKENS XC_FULLTEXT_CHUNKS XC_EVIDENCE_DOC_CHARS \
      XC_ANSWER_CONTEXT_CHARS XC_ESCALATE_ON_ABSTAIN XC_REJUDGE_AGENT_TEXT \
      XC_REJUDGE_AGENT_EVIDENCE XC_AGENT_FULLTEXT_TOOLS XC_AGENT_MAX_ITERS
rm -f output/unified_benchmark/ablation/v14-geneturing-genomics/geneturing.jsonl
"$PY" "${RUNNER[@]}" --method v14-geneturing-genomics --datasets geneturing \
    --top-k 20 --concurrency 8
"$PY" scripts/rescore_factoid_tokenf1.py
# The paper wrote the spliced value into the aggregated CSVs (needs the CSVs from
# scripts/aggregate_continuous.py and the DISCO-merged headline):
#   "$PY" scripts/aggregate_continuous.py; "$PY" figures/_propagate_geneturing_genomics.py
# (aggregate_continuous.py writes the three main CSVs, then exits non-zero at its
#  scaling step because benchmark/unified/sample_1-10_seed42.jsonl is not released;
#  hence ";" rather than "&&".)
