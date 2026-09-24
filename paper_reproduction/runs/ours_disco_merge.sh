#!/usr/bin/env bash
# DISCO router merge behind BioASQ 54.1 (87 routed items). It also touches the 164
# SciHorizon expression items (all replaced later by the case study, so the 60.3 cell
# does not depend on it; it matters only for the unspliced 56.0, 55.7 without it).
# Research run: router predictions 2026-04-28 02:09; DISCO run 2026-04-28 10:29 over the
# 261 routed items (164 SciHorizon, 87 BioASQ, 4 PubMedQA, 2 MedMCQA, 4 MedQA-CN);
# merge by scripts/merge_disco_into_headline.py. The MedMCQA answers are unchanged.
# Order: run after ours_april_eight_columns.sh and BEFORE ours_medxpertqa.sh or the
# LitQA2 scripts. merge_disco_into_headline.py copies every *.jsonl in
# output/unified_benchmark/ablation/v14-cascade-dual-rerank-grounded/, and those scripts
# write medxpertqa_text.jsonl / litqa2.jsonl there (move them away otherwise).
# Requires the atlas server (DISCO_SERVER_URL; not released, see ../README.md).
# Prompt: V_pre (prompts/V_pre.txt), i.e. V_apr plus the `atlas` block, which exists
# for exactly this configuration.
# The research router_predictions_filtered.jsonl that the run and the merge used holds
# 15,631 predictions, routed and non-routed; 261 have route_disco == true. Both the
# runner's --item-filter and the merge admit only those, so this script keeps every
# item with route_disco == true.
source "$(dirname "$0")/_common.sh"
export XC_V14_PROMPT_FILE="$ROOT/prompts/V_pre.txt"
"$PY" scripts/llm_disco_router.py \
    --datasets bioasq geneturing medmcqa medqa_us medqa_taiwan medqa_mainland \
               pubmedqa_pqal_test scihorizon_hgkb
"$PY" - <<'PY'
import json
src = ".cache/disco_router/router_predictions.jsonl"
dst = ".cache/disco_router/router_predictions_filtered.jsonl"
n = 0
with open(src) as fi, open(dst, "w") as fo:
    for line in fi:
        d = json.loads(line)
        if d.get("route_disco") is True:
            fo.write(line); n += 1
print(f"{n} routed items -> {dst} (research run: 261)")
PY
F=.cache/disco_router/router_predictions_filtered.jsonl
"$PY" "${RUNNER[@]}" --method v14-cascade-dual-rerank-grounded-disco \
    --datasets bioasq geneturing medmcqa medqa_us medqa_taiwan medqa_mainland \
               pubmedqa_pqal_test scihorizon_hgkb \
    --top-k 20 --concurrency 4 --item-filter "$F" --router-entities "$F"
# Merge into output/unified_benchmark/ablation/bioharness-headline/ and rescore factoids.
"$PY" scripts/merge_disco_into_headline.py
"$PY" scripts/rescore_factoid_tokenf1.py
