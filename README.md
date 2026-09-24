# BioHarness

Code for **BioHarness: Substrate-Aware Evidence Assembly for Biomedical
Question Answering across Literature, Knowledge Bases, and Biological
Atlases**.

The repository has two parts. They are not the same code:

| Part | What it is | Reproduces the paper's Table 1? |
| --- | --- | --- |
| [`paper_reproduction/`](paper_reproduction/) | The research code behind the paper's numbers (September 2026 state), released without refactoring. | It is the research code (September 2026 state) behind the reported cells, with one run script per cell. |
| `src/bioharness/` (this package, method id `pipeline`) | A clean, installable **re-implementation** of the BioHarness cascade, packaged as a [framework-eval](https://github.com/coco11563/BioHarness_Eval_Framework) method plugin. | **No.** See [what the package does not reproduce](#what-this-package-does-not-reproduce). |

To check every number in the **BioHarness row** of Table 1 **offline,
without any inference infrastructure**, use the evaluation framework
[BioHarness_Eval_Framework](https://github.com/coco11563/BioHarness_Eval_Framework).
It ships the per-item outputs behind that row, and its `verify.py`
reproduces every BioHarness cell of Table 1 from the stored per-item
scores. Per-item outputs for the baseline and frontier rows are not
shipped.

---

## Paper results (BioHarness row of the revised Table 1)

Metric: per item, token-F1 for factoid items (SQuAD-style normalisation,
max over gold variants) and the stored type-specific score for every other
question type, averaged per dataset.

| PubMedQA | BioASQ | GeneTuring | SciHorizon | MedMCQA | MedQA-US | MedQA-TW | MedQA-CN | MedXpertQA | **Overall9** | LitQA2 (supplementary) |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 76.4 | 54.1 | 54.6 (a) | 60.3 (a) | 74.9 | 85.9 | 89.3 | 89.7 | 37.1 | **67.2** | 61.8 (b) |

- **Overall9** is the mean over the nine datasets other than LitQA2,
  weighted by full dataset size: 21,752 scored items, pooled per-item mean
  0.672131. The secondary binary accuracy on the same items is 0.741.
- (a) These two cells include post-hoc additions applied only to
  BioHarness: a GeneTuring re-run with a genomics tool layer, and an
  HPA-atlas context case study on the 210 SciHorizon expression items.
  Without them the cells are 28.8 and 56.0, and Overall9 is 65.3.
- (b) Single spliced run, not held out. The configuration was revised after
  error analysis on the same 199 items. The three-run mean of the earlier
  configuration is 57.8. LitQA2 is reported separately and is not part of
  Overall9.

## Benchmark data

Ten datasets. Nine are hosted on Hugging Face at
[`Shaow/BioHarness_Eval`](https://huggingface.co/datasets/Shaow/BioHarness_Eval)
(21,924 lines, 21,824 unique ids, 21,752 scored items). The tenth is
LitQA2 (199 items). It is **not** re-hosted: the dataset repo ships its item
ids and a build script that reconstructs it from a pinned LAB-Bench
revision. LitQA2 is CC BY-SA 4.0 and carries a canary string asking that
it never appear in training corpora. In the agent-harness experiment run
for the revised paper, web search also returned public LAB-Bench mirrors
that exposed labelled answers for 13 of the 199 items. With LitQA2 the
suite has 22,123 lines.

## Models and corpus used in the paper

- LLM backbone: **Qwen3.5-35B-A3B**, every call with `enable_thinking=False`.
- Embedding: Qwen3-Embedding-0.6B (1024-dim).
- Reranker: Qwen3-Reranker-8B.
- Corpus: PubMed (27.3 M abstracts) and PMC full text (6.6 M articles).
  For LitQA2 only, 46 open-access source papers of LitQA2 items were
  added, raising question-level coverage from 37.7 % to 64.8 %.

---

## What this package does not reproduce

The `pipeline` method in `src/bioharness/` follows the paper's cascade
(query rewrite, dense retrieval with a pseudo-answer pool and, for yes/no
items, a negative-evidence pool, cross-encoder rerank, constrained
generation with first-token confidence, a grounded gate, agent escalation
and constrained re-judgment). A live run of it will **not** reproduce the
Table 1 columns. The main gaps:

1. **Agent.** Escalated items go to a single LLM call over the top
   stage-1 passages. The paper escalates into the multi-iteration V14 REPL
   agent with its own system prompt and retrieval, MeSH, full-text and
   entity tools.
2. **MCQ protocol.** There is no official zero-shot chain-of-thought MCQ
   path. The package answers MCQ items with a letter-only constrained
   prompt; the paper's MedXpertQA and LitQA2 cells use the official
   zero-shot chain-of-thought protocol with an 8,192-token rationale
   budget.
3. **Retrieval.** Retrieval is abstract-only from the Qdrant `paper-full`
   collection. There is no PMC full-text chunk retrieval (used for LitQA2),
   no PostgreSQL abstract lookup and no MeSH or keyword path.
4. **Prompts and gates.** Prompts, context caps, escalation conditions
   (for example escalation on an "insufficient information" answer) and the
   re-judgment inputs differ from the configurations behind the Table 1
   cells, which themselves changed between the April runs and the
   September MedXpertQA/LitQA2 runs.
5. **Post-hoc cells.** The SciHorizon expression case study and the
   GeneTuring genomics re-run behind footnote (a) are separate runs. The
   atlas server used for the case study (an HPA tissue-expression
   endpoint) is documented but not released; the package's optional
   `+D` path needs such an endpoint that you provide.

The reranker's `mode="completion"` matches the research code's
`XC_RERANK_RAW=1` scorer: the official Qwen3-Reranker completion template
through `/v1/completions` (`P(yes) / (P(yes) + P(no))`, query cut to 500
characters, title plus abstract cut to 1,500 characters). The runs from
2026-09-08 on (MedXpertQA, LitQA2 and their reranking baselines) set
`XC_RERANK_RAW=1`; the April and June runs used the research code's default
chat-completions scorer (see
[`paper_reproduction/README.md`](paper_reproduction/README.md), "Reranker").
The default auto-detection uses a server's `/v1/rerank` endpoint instead when
one answers; that is not a scorer the paper used. The package's earlier
generic yes/no chat prompt has been removed.

For the research code (September 2026 state) and the run script behind
every cell, use [`paper_reproduction/`](paper_reproduction/). For the per-item outputs,
use [BioHarness_Eval_Framework](https://github.com/coco11563/BioHarness_Eval_Framework).

`docs/validation.md` records an earlier small-sample comparison between
this package and the paper's run outputs.

---

## Quickstart

### 1. Install from source

The packages are not on PyPI. Install the evaluation framework first,
then this package:

```bash
git clone https://github.com/coco11563/BioHarness.git
cd BioHarness
python -m venv .venv && source .venv/bin/activate
pip install "git+https://github.com/coco11563/BioHarness_Eval_Framework.git"
pip install -e ".[dev]"
```

### 2. Stand up the inference services

`bioharness` does not bundle the LLM, embedding, reranker or Qdrant
services it needs. The requirements are in `docs/infra.md`. Point the
package at them with environment variables:

```bash
export BIOHARNESS_LLM_URL=http://127.0.0.1:8000/v1
export BIOHARNESS_EMBED_URL=http://127.0.0.1:8002/v1
export BIOHARNESS_RERANK_URL=http://127.0.0.1:8001/v1
export BIOHARNESS_QDRANT_URL=http://127.0.0.1:13335
export BIOHARNESS_MODEL_NAME=Qwen3.5-35B-A3B   # the served model id
```

`bioharness doctor` checks that every service is reachable.

### 3. Run the method through the harness

```bash
framework-eval run \
  --method pipeline \
  --datasets bioasq scihorizon-gene geneturing medmcqa \
             medqa_us medqa_taiwan medqa_mainland pubmedqa_pqal_test \
             medxpertqa_text \
  --output runs/bioharness/

framework-eval score --run runs/bioharness/ --output runs/bioharness/scores/
```

Expect numbers that differ from Table 1 for the reasons listed above.

### 4. Check the paper's numbers (offline)

```bash
git clone https://github.com/coco11563/BioHarness_Eval_Framework.git
cd BioHarness_Eval_Framework
python verify.py
```

---

## Configuration

The cascade has one user-facing knob, `BIOHARNESS_FORCE_AGENT=1`, which
sends every non-yes/no item through agent escalation and re-judgment.
`BIOHARNESS_ENABLE_ATLAS=1` together with `BIOHARNESS_ATLAS_URL` turns on
the optional atlas context for SciHorizon expression items. Everything else
is fixed in `src/bioharness/config.py`.

The `pipeline-production` entry point and the `pipeline` agent hook
delegate to the research code only when `BIOHARNESS_PRODUCTION_SRC` is
set; point it at `paper_reproduction/` (there is no auto-detection). You
then also need the research code's own dependencies and `rlm`
(`paper_reproduction/requirements.txt`, `paper_reproduction/third_party/setup_rlm.sh`).
This path was not tested for the release, and it reproduces **no** Table 1
cell: `pipeline-production` builds the research cascade with no `XC_*`
flags and no `XC_V14_PROMPT_FILE`, so it runs the `V_cur` prompt, the
chat-completions reranker, no official MCQ protocol and no full-text
chunks. The per-cell configurations are the scripts in
[`paper_reproduction/runs/`](paper_reproduction/runs/).

---

## Repository layout

```
BioHarness/
├── README.md
├── LICENSE
├── CHANGELOG.md
├── CONTRIBUTING.md
├── MANIFEST.toml                # headline numbers + infra requirements
├── pyproject.toml
├── verify.py                    # offline plugin check; --live single-item smoke
├── paper_reproduction/          # research code behind the paper's numbers
├── src/bioharness/
│   ├── cascade/                 # PipelineCascadeClient (entry point `pipeline`)
│   ├── agent/                   # single-call agent + optional research-code adapter
│   ├── tools/                   # gene resolver, genomics (dbSNP / MyGene / BLAST)
│   ├── clients/                 # LLM / embedding / rerank / Qdrant / atlas clients
│   ├── data/                    # precomputed GeneTuring BLAST cache
│   └── cli/                     # `bioharness` CLI (doctor, config)
├── tests/                       # unit + integration tests
├── docs/
│   ├── infra.md                 # service requirements
│   ├── adapter.md               # how the QAClient maps to PipelineCascadeClient
│   └── validation.md            # earlier small-sample package-vs-paper check
├── scripts/
│   ├── scan_forbidden_strings.py
│   └── build_cached_smoke.py
└── .github/workflows/ci.yml
```

---

## Citation

The revised manuscript that these numbers follow is not yet on arXiv.
arXiv v1 (June 2026) reports the earlier eight-dataset, 19,302-item suite,
whose numbers do not match the tables here; the link will point to v2
once it is posted.

```
@article{xiao2026bioharness,
  title={BioHarness: Substrate-Aware Evidence Assembly for Biomedical Question Answering across Literature, Knowledge Bases, and Biological Atlases},
  author={Xiao, Meng and Qin, Chuan and Chen, Jinmiao and Cheng, Yihang and Zhou, Yuanchun and Zhu, Hengshu},
  journal={arXiv preprint arXiv:2606.19396},
  year={2026}
}
```

## License

Apache-2.0 (see `LICENSE`) for our own code. `paper_reproduction/`
contains parts ported from or modelled on third-party projects, some of
them under MIT and some without a published licence; see the
"Third-party code" and "Licence" sections of
[`paper_reproduction/README.md`](paper_reproduction/README.md) and
`paper_reproduction/THIRD_PARTY_NOTICES.txt`. Benchmark datasets keep
their own licences; see the dataset card on Hugging Face.

## Contact

Issues and pull requests: <https://github.com/coco11563/BioHarness/issues>.
