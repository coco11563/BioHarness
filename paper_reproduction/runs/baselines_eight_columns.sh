#!/usr/bin/env bash
# Baseline rows, the eight original columns. Usage:  runs/baselines_eight_columns.sh <method-id>
# Method ids of the Table 1 rows:
#   No-Context no-context        DenseRAG dense            HyDE hyde
#   DenseRAG+Rerank rankrag      Chain-of-Note chain-of-note   Self-RAG self-rag
#   IRCoT ircot-rerank           GraphRAG rt-graphrag       PathRAG rt-pathrag
#   LightRAG rt-lightrag         FLARE flare                BiomedRAG biomedrag
#   BioRAG biorag-full           GeneGPT genegpt-lite (GeneTuring cell: genegpt-fmt, footnote h)
#   w/ Golden Context golden-context (PubMedQA and BioASQ only; adds --use-golden-context)
# Research runs: April-June 2026, top-k 20, no XC_* behaviour flags (none existed).
# Every baseline answers through shared_pipeline.generate_constrained_answer, which
# truncates evidence to its first 8,000 characters (XC_ANSWER_CONTEXT_CHARS default).
# Reranker: XC_RERANK_RAW is left unset, as in the research runs (see ../README.md).
# Baseline cells of the eight original columns come from scripts/aggregate_continuous.py
# over these runs.
source "$(dirname "$0")/_common.sh"
M="${1:?usage: $0 <method-id>}"
DATASETS=(bioasq geneturing medmcqa medqa_us medqa_taiwan medqa_mainland pubmedqa_pqal_test scihorizon_hgkb)
EXTRA=()
if [ "$M" = golden-context ]; then DATASETS=(bioasq pubmedqa_pqal_test); EXTRA=(--use-golden-context); fi
unset XC_MCQ_OFFICIAL XC_MCQ_OFFICIAL_TOKENS XC_FULLTEXT_CHUNKS XC_ANSWER_CONTEXT_CHARS
"$PY" "${RUNNER[@]}" --method "$M" --datasets "${DATASETS[@]}" --top-k 20 \
    --concurrency "${CONCURRENCY:-16}" ${EXTRA[@]+"${EXTRA[@]}"}
"$PY" scripts/rescore_factoid_tokenf1.py
