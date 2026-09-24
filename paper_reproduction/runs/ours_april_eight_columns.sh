#!/usr/bin/env bash
# Ours, the eight original columns (PubMedQA, BioASQ, GeneTuring*, SciHorizon*, MedMCQA,
# MedQA-US/TW/CN): method v14-cascade-dual-rerank-grounded on the full eight datasets.
# Research run: 2026-04-21 to 2026-04-28, top-k 20, concurrency 8 (run_start record).
#   * PubMedQA 76.4, MedMCQA 74.9, MedQA-US 85.9, MedQA-TW 89.3, MedQA-CN 89.7 come from
#     this run (MedMCQA after the DISCO merge, which changes no MedMCQA answer).
#   * BioASQ 54.1 additionally needs the DISCO merge (ours_disco_merge.sh; 54.0 without).
#   * GeneTuring and SciHorizon are replaced by ours_geneturing_genomics.sh and
#     ours_scihorizon_expression.sh (footnote a); from this run alone they are 28.8 / 56.0.
# Prompt: V_apr (prompts/V_apr.txt). The released src/rlm/pipeline.py carries V_cur, so
# the April text is injected with XC_V14_PROMPT_FILE.
# No XC_* behaviour flags existed for these runs; all defaults apply.
# Reranker: XC_RERANK_RAW is left unset; the April runs used the default chat-completions
# scorer (see ../README.md, "Reranker").
source "$(dirname "$0")/_common.sh"
export XC_V14_PROMPT_FILE="$ROOT/prompts/V_apr.txt"
unset XC_MCQ_OFFICIAL XC_MCQ_OFFICIAL_TOKENS XC_FULLTEXT_CHUNKS XC_EVIDENCE_DOC_CHARS \
      XC_ANSWER_CONTEXT_CHARS XC_ESCALATE_ON_ABSTAIN XC_REJUDGE_AGENT_TEXT \
      XC_REJUDGE_AGENT_EVIDENCE XC_AGENT_FULLTEXT_TOOLS XC_AGENT_MAX_ITERS
"$PY" "${RUNNER[@]}" --method v14-cascade-dual-rerank-grounded \
    --datasets bioasq geneturing medmcqa medqa_us medqa_taiwan medqa_mainland \
               pubmedqa_pqal_test scihorizon_hgkb \
    --top-k 20 --concurrency 8
# Factoid items (BioASQ, GeneTuring) are scored with token-F1 sidecars:
"$PY" scripts/rescore_factoid_tokenf1.py
