#!/usr/bin/env bash
# MedXpertQA official protocol for a baseline row. Usage:
#   runs/baselines_medxpertqa_official.sh <method-id>
# Official zero-shot chain-of-thought format with retrieved evidence prepended. The
# rationale budget was NOT uniform across rows (Table 1 footnote):
#   16,384 tokens: no-context dense hyde chain-of-note self-rag rt-graphrag rt-pathrag
#                  rt-lightrag flare biorag genegpt-lite   (run with concurrency 64)
#    8,192 tokens: rankrag ircot-rerank biomedrag, and Ours (runs/ours_medxpertqa.sh)
#                  (run with concurrency 24, one method at a time, XC_RERANK_RAW=1)
# The BioRAG MedXpertQA cell uses the `biorag` variant (shared retrieval only), and the
# GeneGPT cell uses genegpt-lite. The 16,384-token methods never call the reranker, so
# XC_RERANK_RAW has no effect on them (their research runs did not set it).
# Override the budget with TOKENS=<n> to run a budget-matched comparison (not done for
# the paper; the effect of 8,192 vs 16,384 was not measured).
source "$(dirname "$0")/_common.sh"
M="${1:?usage: $0 <method-id>}"
case "$M" in
  rankrag|ircot-rerank|biomedrag) DEF_TOKENS=8192;  CONC=24 ;;
  *)                              DEF_TOKENS=16384; CONC=64 ;;
esac
export XC_MCQ_OFFICIAL=1 XC_MCQ_OFFICIAL_TOKENS="${TOKENS:-$DEF_TOKENS}" XC_RERANK_RAW=1 XC_FAULTHANDLER=1
unset XC_FULLTEXT_CHUNKS XC_EVIDENCE_DOC_CHARS XC_ANSWER_CONTEXT_CHARS XC_ESCALATE_ON_ABSTAIN \
      XC_REJUDGE_AGENT_TEXT XC_REJUDGE_AGENT_EVIDENCE XC_AGENT_FULLTEXT_TOOLS
"$PY" "${RUNNER[@]}" --method "$M" --datasets medxpertqa_text --concurrency "${CONCURRENCY:-$CONC}"
