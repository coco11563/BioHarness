# Changelog

All notable changes to `BioHarness` are documented in this file.
Format: [Keep a Changelog](https://keepachangelog.com/en/1.1.0/);
versioning: [SemVer](https://semver.org/spec/v2.0.0.html).

## [0.2.0] - 2026-09-25

### Fixed
- **Reranker scorer.** The logprob mode of
  `bioharness.clients.rerank.RerankClient` sent a generic yes/no prompt
  through `/v1/chat/completions` and scored `exp(logprob("yes"))`. That
  path is removed. The client now scores the Qwen3-Reranker raw
  completion template (judge system prompt,
  `<Instruct>/<Query>/<Document>`, empty think block) through
  `/v1/completions` with `logprobs`, as `P(yes) / (P(yes) + P(no))`, with
  the query cut to 500 and the document to 1,500 characters, batched 64
  prompts per request. This matches the research code's `XC_RERANK_RAW=1`
  scorer (set by the runs from 2026-09-08 on: MedXpertQA, LitQA2) when
  `mode="completion"` is selected. The `/v1/rerank` API mode is kept (now
  with the same truncation, and sending `top_n` instead of `top_k`);
  auto-detection picks it whenever the endpoint answers, and it is not the
  paper's scorer.
- `dual_rerank` now scores each passage as `"{title}\n{text}"`, as the
  research pipeline does.
- Default `BIOHARNESS_MODEL_NAME` is `Qwen3.5-35B-A3B` instead of the
  literal placeholder `{model}`.

### Changed
- `pipeline-production` and the agent hook no longer auto-detect a sibling
  research checkout; they delegate only when `BIOHARNESS_PRODUCTION_SRC` is
  set.
- **Documentation reframed.** The README, `MANIFEST.toml`, `verify.py` and
  `docs/` now state that this package is a re-implementation of BioHarness
  that does not reproduce the revised Table 1 columns, list the main gaps
  (single-call agent instead of the V14 REPL agent, no official MCQ
  chain-of-thought protocol, abstract-only retrieval, different prompts
  and gates), and point to `paper_reproduction/` for the research code and
  to the eval framework for the per-item outputs.
- **Headline updated to the revised Table 1:** Overall9 67.2 (token-F1
  metric, 21,752 scored items over the nine datasets other than LitQA2;
  `continuous_mean=0.672131`, `binary_accuracy=0.740943`) and LitQA2 61.8
  as a separate, supplementary number (199 items). This supersedes the
  19,302-item headline below. The `[verify.live]` tolerance band is removed
  because the package does not reproduce the headline.
- Dataset renamed to `Shaow/BioHarness_Eval` in all docs and
  `pyproject.toml`; ten datasets (nine hosted, LitQA2 as ids plus a build
  script).
- Method name is **BioHarness** throughout.
- README: removed the PyPI install line (the packages are not on PyPI),
  the ablation-flag table (those flags were never implemented) and layout
  entries for files that do not exist (`CITATION.cff`, `golden/`,
  `docs/ablations.md`).
- **Rebranded to `BioHarness`** (package formerly `framework_chi`). **Breaking:**
  - Python package `framework_chi` → `bioharness` (import path:
    `from bioharness... import ...`).
  - Environment-variable prefix `FRAMEWORK_*` → `BIOHARNESS_*`. The old
    `FRAMEWORK_*` names are still read for one release with a deprecation
    warning; update your configs to `BIOHARNESS_*`.
  - CLI `framework-chi` → `bioharness`; repo URLs → `coco11563/BioHarness`.
  - Unchanged by the rename: the framework-eval method id (`pipeline`).
- History (superseded by the headline entry above): the companion eval
  framework
  ([BioHarness_Eval_Framework](https://github.com/coco11563/BioHarness_Eval_Framework))
  first added token-F1 scoring for factoid items as an additive protocol on
  the earlier 19,302-item suite. The revised Table 1 uses token-F1 for
  factoid items as its official metric.

### Added
- **`paper_reproduction/`**: the research code (September 2026 state)
  behind the revised Table 1 (benchmark runner and its import closure,
  scoring, aggregation and splice scripts, the four V14 prompt versions,
  per-cell run scripts in `runs/`, the pinned `rlm` commit plus local patch
  in `third_party/`, and `THIRD_PARTY_NOTICES.txt`). See its README for the
  release edits and known caveats.
- **Genomics tool layer** (`bioharness.tools.genomics`) for GeneTuring-style
  structured facts that literature retrieval cannot answer:
  - `snp_lookup(rsid)` — dbSNP rs ID → associated gene + chromosome (NCBI
    E-utilities).
  - `gene_genomic_info(symbol)` — gene → chromosome + protein-coding status
    (MyGene.info), with the answer pre-formatted as `TRUE`/`FALSE`.
  - `blast_align(sequence, mode=…)` — DNA sequence → human-genome coordinates
    (`chrN:start-end`) or source organism, via NCBI BLAST (first-HSP parse).
    Slow/rate-limited, so the headline pipeline precomputes these offline; the
    function is the self-contained live convenience.
  - `blast_lookup(...)` — **cache-first** wrapper that reads the shipped
    precomputed BLAST cache (`bioharness/data/blast_geneturing_cache.json`,
    114 GeneTuring DNA-alignment answers) and only falls back to live BLAST on a
    miss; wired into `precall_tools` so re-runs reproduce the alignment subtasks
    instantly without hammering NCBI for hours.
  - The dbSNP/MyGene lookups are wired into `tools.precall_tools`, and
    `precall_tools` now feeds the **fast path** (`constrained_generate(...,
    tool_evidence=...)`) in addition to the agent / re-judgment stage — so the
    authoritative answer is present before the model commits (gene-DB lookups
    otherwise answered "unknown" on the fast path). Targets the SNP-association
    / SNP-location / gene-location / protein-coding / DNA-alignment GeneTuring
    subtasks, whose answers live in NCBI databases rather than PubMed. Async +
    fail-soft; no API key required (NCBI calls share the gene-resolver throttle).
- **Atlas (D) component** (`bioharness.cascade.atlas`) for the SciHorizon
  `expression` subtask (gene → tissue list):
  - A recall + fixed 27-tissue-vocabulary + parseable-JSON prompt, replacing
    the free-text expression prompt that produced 0/210 parseable answers
    (~6% set-F1, a formatting artifact). This `-D` path ships and runs by
    default.
  - An opt-in `+D` *atlas-as-context* path: `bioharness.clients.atlas.AtlasClient`
    fetches a gene's HPA bulk tissue expression and injects it as a
    supplementary reference block. Enable with `BIOHARNESS_ENABLE_ATLAS=1`
    and a reachable `BIOHARNESS_ATLAS_URL` (off by default; fail-soft to `-D`).
  - The paper reports atlas context as a post-hoc case study applied to
    BioHarness alone (Table 1 footnote a, SciHorizon 60.3; 56.0 without it).
- Method-side manifest (`MANIFEST.toml`) pinning the framework-eval
  contract version, the paper's headline numbers (from the research runs)
  and the infra requirements.
- Project skeleton: `pyproject.toml`, `src/bioharness/`, tests
  layout, CI workflow, pre-commit config (mirrors the eval-framework
  scaffold).

### Notes
- Plugs into framework-eval as the `pipeline` method via the
  `framework_eval.methods` entry point.
- Superseded (see the headline entry above): the earlier headline was
  binary accuracy 0.766 on 19,302 items. The BioHarness row is
  verified offline by the eval framework's `python verify.py` from the
  stored per-item scores; this package re-implements the method and does
  not reproduce them.
