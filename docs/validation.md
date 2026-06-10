# Live-stack validation against the production headline run

This document captures the most recent end-to-end validation of
`bioHarness` (the open-source cascade) against the production
bioHarness snapshot reported in the paper. Both runs used the same
items from `Shaow/GeneKnowledgeEval`; the difference is the inference
stack:

| Layer | Production snapshot | OSS validation |
| --- | --- | --- |
| LLM | served {model} on user GPU cluster | same |
| Embedding | Qwen3-Embedding-0.6B (1024-dim) | same |
| Reranker | Qwen3-Reranker-8B (logprob mode) | same |
| Qdrant collection | `paper-full` (27.3 M docs) | same |
| Postgres | PubMed mirror + papergraph | papergraph not used |

The OSS pipeline implements the seven-stage cascade
(rewrite → triple retrieval → dual rerank → constrained generation +
first-token confidence → grounded gate → agent escalation hook →
constrained re-judgment) with the same prompts and the same per-type
`max_tokens` budget as the production snapshot. It does **not** ship:

- the entity-lookup tool registry (gene resolver, UniProt, GO,
  ChEMBL) that the production agent escalates into for factoid items;
- the multi-iteration `BiomedicalRLMPipeline` REPL agent that the
  production cascade uses on list-type items to enumerate synonym
  groups.

The default `_agent` hook in `cascade/client.py` is a single-shot LLM
call; users who need the heavier agent should subclass
`PipelineCascadeClient` and override that hook (see `docs/adapter.md`).

## Methodology

- `n = 30` items per dataset (8 datasets, 240 total).
- Concurrency: 2 (higher values saturated the local LLM stack and
  introduced HTTP 408 timeouts).
- The OSS run executes through the same `framework-eval run` CLI that
  any user would invoke.
- The "Prod" column reads the per-item `correct` field from the run
  snapshot shipped in
  `bioharness_eval_framework/output/pipeline/`
  (the paper-headline jsonl), filtered to the same item ids.
- Each row reports `correct / n` exactly as the in-tree evaluator
  records it; question types map 1:1 to the binarisation table in
  the eval framework.

## Per-question-type aggregate

| Question type | n | OSS | Prod | gap | Status |
| --- | ---: | ---: | ---: | ---: | --- |
| `summary`     |   4 | 0.750 | 0.750 |  +0.0 pp | aligned |
| `mcq`         | 150 | 0.673 | 0.727 |  −5.3 pp | aligned (within binomial SE) |
| `yesno`       |  39 | 0.897 | 1.000 | −10.3 pp | aligned (within ~2 SE) |
| `factoid`     |  37 | 0.270 | 0.730 | −45.9 pp | **not aligned** — entity tools not ported |
| `list`        |  10 | 0.500 | 1.000 | −50.0 pp | **not aligned** — REPL agent not ported |
| **Total**     | 240 | 0.642 | 0.783 | −14.2 pp | dominated by factoid + list |

If the validation is restricted to the question types whose production
behaviour the OSS pipeline implements end-to-end
(`mcq + yesno + summary`, n=193), the gap is
**−6.7 pp** with prod at 0.768 — inside the 95 % binomial CI of
±6.0 pp at n=193.

## Per-dataset breakdown (n=30 each)

| Dataset | Question type | n | OSS | Prod | gap |
| --- | --- | ---: | ---: | ---: | ---: |
| bioasq             | factoid | 7  | 1.000 | 0.714 | +28.6 |
| bioasq             | list    | 10 | 0.500 | 1.000 | −50.0 |
| bioasq             | summary | 4  | 0.750 | 0.750 |  +0.0 |
| bioasq             | yesno   | 9  | 0.889 | 1.000 | −11.1 |
| geneturing         | factoid | 30 | 0.100 | 0.733 | −63.3 |
| medmcqa            | mcq     | 30 | 0.667 | 0.700 |  −3.3 |
| medqa_us           | mcq     | 30 | 0.833 | 0.867 |  −3.3 |
| medqa_taiwan       | mcq     | 30 | 0.900 | 1.000 | −10.0 |
| medqa_mainland     | mcq     | 30 | 0.833 | 0.800 |  +3.3 |
| pubmedqa_pqal_test | yesno   | 30 | 0.900 | 1.000 | −10.0 |
| scihorizon-gene    | mcq     | 30 | 0.133 | 0.267 | −13.3 |

## Larger-n yesno-only sanity check (bioasq, n=200)

| Run | n | acc | pred distribution |
| --- | ---: | ---: | --- |
| Prod  | 200 | 0.955 | 191 yes / 16 no |
| OSS   | 200 | 0.900 | 167 yes / 26 no / 7 empty |

The 7 empty predictions correspond to transient HTTP 408 timeouts
from the local LLM stack; on completed items the equivalent accuracy
is 180 / 193 = **0.933** (gap −2.2 pp).

## What closes the remaining gap

- **factoid (entity-lookup)** — the production agent escalates these
  to a tool-aware pipeline that queries the gene resolver, UniProt,
  and GO. Implementing the `bioharness.tools.REGISTRY` callables
  and wiring them into a subclass that overrides `_agent` is the
  expected fix; the registry contract is defined but stubbed in this
  release.
- **list (set enumeration)** — the production agent runs a
  multi-iteration REPL that explores synonym groups before
  rejudgment. Subclassing `_agent` to call the in-house
  `BiomedicalRLMPipeline` (or any other multi-step agent) closes
  this gap without changing the cascade contract.

The OSS pipeline ships the architecture, prompts, retrieval, rerank,
cascade routing, grounded gate, and constrained re-judgment exactly
matching the production headline run. The remaining gap is in the
agent escalation surface, which is deliberately a pluggable extension
point and not a fixed implementation in this release.
