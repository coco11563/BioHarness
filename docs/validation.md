# Small-sample check: this package vs. the paper's run outputs

> **Read this first.** This package is a re-implementation of BioHarness
> and does not reproduce the revised Table 1 columns (see the README,
> "What this package does not reproduce"). The research code that
> produced the paper's numbers is in `paper_reproduction/`, and the
> per-item outputs that verify the BioHarness row offline are in
> [BioHarness_Eval_Framework](https://github.com/coco11563/BioHarness_Eval_Framework).
> The comparison below is kept as a record of how far the package is from
> those outputs. It predates the revision and has not been re-run.

Scope and caveats of this check:

- It covers the eight original datasets only (MedXpertQA and LitQA2 were
  added later) and uses **binary accuracy** (the per-item `correct`
  field), not the paper's official token-F1 metric.
- The items come from the benchmark now hosted as
  [`Shaow/BioHarness_Eval`](https://huggingface.co/datasets/Shaow/BioHarness_Eval).
- The "Prod" column is the paper's April run outputs for the same items,
  as they stood before the revision (see Methodology).
- The package run used the package's **earlier** reranker, a generic
  chat-completions yes/no scorer. It has since been replaced by the
  Qwen3-Reranker raw completion template (`src/bioharness/clients/rerank.py`),
  so a re-run would differ.

| Layer | Paper run ("Prod") | Package run ("OSS") |
| --- | --- | --- |
| LLM | Qwen3.5-35B-A3B | same |
| Embedding | Qwen3-Embedding-0.6B (1024-dim) | same |
| Reranker | Qwen3-Reranker-8B, research-code scorer | Qwen3-Reranker-8B, earlier package scorer (different prompt and scoring) |
| Qdrant collection | `paper-full` (27.3 M docs) | same |
| Agent | multi-iteration V14 REPL agent with tools | single LLM call over the stage-1 passages |
| Postgres | PubMed mirror + papergraph | not used |

The package implements the cascade shape (rewrite, triple retrieval,
rerank, constrained generation with first-token confidence, grounded gate,
agent escalation hook, constrained re-judgment), but its prompts, caps,
escalation conditions and agent differ from the paper's runs. The default
`_agent` hook in `cascade/client.py` is a single-shot LLM call; users who
need a heavier agent can subclass `PipelineCascadeClient` and override that
hook (see `docs/adapter.md`).

## Methodology

- `n = 30` items per dataset (8 datasets, 240 total).
- Concurrency: 2 (higher values saturated the local LLM stack and
  introduced HTTP 408 timeouts).
- The OSS run executes through the same `framework-eval run` CLI that
  any user would invoke.
- The "Prod" column was read from the per-item `correct` field of the
  April run snapshot then shipped in
  `BioHarness_Eval_Framework/output/pipeline/`, filtered to the same item
  ids. That snapshot has since been replaced by the revised outputs (GeneTuring genomics re-run,
  BioASQ with the DISCO merge, SciHorizon expression case study,
  MedXpertQA, LitQA2), so the Prod column cannot be regenerated from the
  current snapshot.
- Each row reports `correct / n` exactly as the in-tree evaluator
  records it; question types map 1:1 to the binarisation table in
  the eval framework.

## Per-question-type aggregate

| Question type | n | OSS | Prod | gap | Status |
| --- | ---: | ---: | ---: | ---: | --- |
| `summary`     |   4 | 0.750 | 0.750 |  +0.0 pp | no gap (n=4) |
| `mcq`         | 150 | 0.673 | 0.727 |  −5.3 pp | within binomial SE |
| `yesno`       |  39 | 0.897 | 1.000 | −10.3 pp | within ~2 SE |
| `factoid`     |  37 | 0.270 | 0.730 | −45.9 pp | **not aligned** — agent and tools differ |
| `list`        |  10 | 0.500 | 1.000 | −50.0 pp | **not aligned** — REPL agent not ported |
| **Total**     | 240 | 0.642 | 0.783 | −14.2 pp | dominated by factoid + list |

Restricted to the question types with the smallest gaps
(`mcq + yesno + summary`, n=193), the package scores 139/193 = 0.720 and
the paper run 151/193 = 0.782, a gap of **−6.2 pp** (computed from the
per-dataset table below).

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
| Prod  | 200 | 0.955 | not available |
| OSS   | 200 | 0.900 | 167 yes / 26 no / 7 empty |

The 7 empty predictions correspond to transient HTTP 408 timeouts
from the local LLM stack; on completed items the equivalent accuracy
is 180 / 193 = **0.933** (gap −2.2 pp).

## Where the gap comes from

- **factoid (entity lookup)** and **list (set enumeration)**: the paper
  escalates these into the multi-iteration V14 REPL agent with retrieval,
  MeSH, full-text and entity tools. The package escalates into a single
  LLM call with pre-called gene / genomics evidence.
- **mcq and yesno**: smaller gaps. Their causes were not isolated;
  prompts, context caps and the reranker all differ.

The full list of differences is in the README. None of the numbers above
should be read as the package reproducing the paper.
