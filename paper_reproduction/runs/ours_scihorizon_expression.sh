#!/usr/bin/env bash
# Ours, SciHorizon 60.3 (footnote a): the 2,400 non-expression items of the April run
# (ours_april_eight_columns.sh, prompt V_apr; the DISCO merge changes none of them, since
# all 164 routed SciHorizon items are expression items) plus the 210 expression items
# replaced by the offline HPA-atlas "repair context" case study, arm `ours`:
#   literature = cached B6 retrieval (top 20 PMIDs, abstracts <= 1,200 chars, 8,000 total)
#   + HPA nTPM tissue rows from the atlas server, fixed 27-tissue vocabulary,
#   set-F1 against the gold tissue list; 35 items with empty gold are scored 1.0.
# Research run: output mtime 2026-06-10. Result: 60.29 (56.0 without the case study).
# The case study uses its own expression prompt (in the script), not the V14 prompt.
# Requires the atlas server (DISCO_SERVER_URL; not released, see ../README.md).
source "$(dirname "$0")/_common.sh"
# 1. B6 retrieval cache (MeSH + keyword + vector + rerank) for SciHorizon questions;
#    writes .cache/retrieval/B6/scihorizon_cache.jsonl
"$PY" scripts/build_strategy_cache.py --strategy B6 --datasets scihorizon_hgkb
# 2. The case study itself; writes output/case_study_repair_context.jsonl
"$PY" scripts/case_study_repair_context.py
# 3. The paper wrote the spliced value into the aggregated CSVs:
#   "$PY" scripts/aggregate_continuous.py; "$PY" figures/_archive_atlas_deprecated/_propagate_expr_repair_context.py
# (aggregate_continuous.py writes the three main CSVs, then exits non-zero at its
#  scaling step because benchmark/unified/sample_1-10_seed42.jsonl is not released;
#  hence ";" rather than "&&".)
