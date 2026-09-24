# paper_reproduction: the research code behind Table 1

This directory holds the research code (September 2026 state) behind the BioHarness
numbers in the revised paper's Table 1, with one run script per cell. It is released
without refactoring and contains the research tree's benchmark runner, its full import
closure, the scoring and table scripts, and the splice scripts. It is not a polished
package. Expect research-code conventions, legacy method ids and flags that
exist only for experiments outside the paper.

The installable package in `../src/bioharness/` is a separate re-implementation. It
does **not** reproduce the revised Table 1 (see `../README.md`). Use this directory to
see or re-run the code behind the paper's cells. To check the BioHarness row of
Table 1 offline, use the per-item outputs in the evaluation framework
([BioHarness_Eval_Framework](https://github.com/coco11563/BioHarness_Eval_Framework),
`output/pipeline/`); its `verify.py` reproduces every BioHarness cell of Table 1 from the
stored per-item scores and needs no infrastructure. Per-item outputs of the baseline and
frontier rows are not shipped, so those cells cannot be checked offline.

Contents:

| Path | What it is |
|---|---|
| `scripts/run_unified_benchmark.py` | The benchmark runner. Every method in Table 1 is a `--method` id in its `METHODS` table. |
| `src/`, `benchmark/code/src/` | Import closure of the runner, with the research tree's relative layout kept (`src/rlm/pipeline.py` = the BioHarness agent stage and V14 prompt; `src/utils/clients.py` = LLM, embedding and reranker clients; `benchmark/code/src/benchmark/shared_pipeline.py` = shared retrieval and answer stage). |
| `src/config.py` | Service configuration, rewritten for release (the other release edits are listed below the table): every endpoint and credential comes from an environment variable (see below). |
| `scripts/rescore_factoid_tokenf1.py` | Official factoid scorer (token-F1). Writes `*.tokenf1.jsonl` sidecars. |
| `scripts/aggregate_continuous.py` | Builds the per-dataset and overall CSVs from per-item results and sidecars. |
| `scripts/llm_disco_router.py`, `scripts/merge_disco_into_headline.py` | Atlas router and the DISCO merge behind the headline file. |
| `scripts/case_study_repair_context.py`, `figures/_archive_atlas_deprecated/_propagate_expr_repair_context.py` | SciHorizon expression case study (footnote a) and the script that wrote it into the CSVs. |
| `figures/_propagate_geneturing_genomics.py` | Wrote the GeneTuring genomics re-run (footnote a) into the CSVs. |
| `scripts/splice_litqa2_headline.py` | **New for this release.** Rebuilds the LitQA2 61.8 splice (footnote b). |
| `scripts/prepare_unified_data.py` | **New for this release.** Writes `benchmark/unified/*.jsonl` from the HF dataset and checks each file's SHA-256 against the research copy. |
| `scripts/build_strategy_cache.py` | Builds the cached B6 retrieval (MeSH + keyword + vector + rerank) used by the SciHorizon case study. |
| `rebuttal_round1/convert_medxpertqa.py` | Converts MedXpertQA Text/test into the runner's format. |
| `rebuttal_round1/ingest_litqa2_oa.py`, `rebuttal_round1/litqa2_ingested_papers.json` | Ingestion of the 46 open-access LitQA2 source papers into the full-text index, and their identifiers (DOI, PMCID, PMID). |
| `alembic/versions/20251128_000001_initial_schema.py` | Schema of the PMC full-text database (`papers`, `sections`, `chunks`). |
| `prompts/` | The four V14 agent system prompt versions and the cells each produced (`prompts/README.md`). |
| `runs/` | The run scripts behind the cells of the Ours row (one script can cover several cells; Overall9 is computed from the cells, not run), plus generic baseline scripts. |
| `third_party/` | Pinned upstream commit of the `rlm` library plus the local patch the runs used. |
| `requirements.txt`, `requirements-graspologic.txt` | Pinned third-party dependencies, derived from the imports in this directory (graspologic is installed with `--no-deps`; see the file header). |

Changes relative to the research tree, all made for release: `src/config.py` rewritten
around environment variables; hard-coded local database and Qdrant URLs in three
`src/retriever/` self-test helpers, `scripts/case_study_repair_context.py` and
`scripts/llm_disco_router.py` replaced by `src/config.py` look-ups or repository-relative
paths; private addresses in a `client_protocol.py` docstring and default replaced by
localhost; the coding-agent harness arm (`HarnessCLIClient`, `WebEvidenceClient`
and their six method ids) removed from the runner because it produced no Table 1 cell;
the headline output key renamed to `bioharness-headline`; the `DATA` and `CASE` paths
in `figures/_archive_atlas_deprecated/_propagate_expr_repair_context.py` corrected for
the archive folder it sits in (as placed in the research tree it raised
`FileNotFoundError`). No logic on the answer path was changed.

## Infrastructure required

Re-running needs the paper's retrieval substrate and three model services. None of it is
distributed here.

| Component | Paper setting |
|---|---|
| LLM | Qwen3.5-35B-A3B behind vLLM's OpenAI-compatible API; every call sends `chat_template_kwargs.enable_thinking=False` |
| Embedding | Qwen3-Embedding-0.6B (1024-d) behind vLLM; queries are embedded without an instruction prefix |
| Reranker | Qwen3-Reranker-8B behind vLLM (completion and chat-completion routes with logprobs) |
| PubMed abstracts | PostgreSQL database `paper-graph-pubmed` (27.3M abstracts; tables `articles`, `mesh_headings`) and Qdrant collection `paper-full` (1024-d vectors of the abstracts) |
| MeSH | Qdrant collection `mesh-term-only` (30,956 MeSH terms) |
| PMC full text | PostgreSQL database `papergraph` (6.6M articles; `papers`, `sections`, `chunks`, schema in `alembic/versions/`) and Qdrant collection `chunks` (chunks of at most 512 cl100k tokens, never crossing a section boundary; point id = `chunks.id`) |
| LitQA2 additions | 46 open-access source papers ingested into `papergraph` + `chunks` (`rebuttal_round1/ingest_litqa2_oa.py`; question-level corpus coverage 37.7 % to 64.8 %) |
| Atlas server | Only for the DISCO merge (BioASQ 54.1) and the SciHorizon expression case study. Not released; see "Atlas server" below. |
| Public web APIs | NCBI E-utilities / BLAST / dbSNP, MyGene, HGNC, UniProt, ChEMBL, ClinicalTrials.gov, the Gene Ontology OBO release, DuckDuckGo; called by tool pre-calls, agent tools and some baselines |

The scripts that built the PubMed and PMC indices are not in the research tree and are
not released. The chunking and schema rules above are those that
`ingest_litqa2_oa.py` reconstructs and checks against an already-indexed paper.

### Configuration

All endpoints come from environment variables; `runs/env.example.sh` lists them with
placeholder values. Copy it to `runs/env.sh` (git-ignored) and the run scripts source
it.

```text
XC_LLM_SERVERS  XC_LLM_MODEL  XC_LLM_KEY          chat LLM
XC_EMBED_SERVERS  XC_EMBED_MODEL                  embeddings
XC_RERANK_SERVERS  XC_RERANK_MODEL                reranker
XC_PG_HOST  XC_PG_PORTS  XC_PG_USER  XC_PG_PASSWORD  XC_PG_PUBMED_DB  XC_PG_PAPERGRAPH_DB
XC_QDRANT_URLS                                    Qdrant REST endpoints
DISCO_SERVER_URL                                  atlas server
NCBI_API_KEY (optional)                           NCBI rate limit
```

### Setup

```bash
cd paper_reproduction
python -m venv .venv && source .venv/bin/activate      # research runs: CPython 3.12
pip install -r requirements.txt
pip install --no-deps -r requirements-graspologic.txt   # GraphRAG port only
bash third_party/setup_rlm.sh                           # rlm 0.1.0 at the pinned commit + local patch
python scripts/prepare_unified_data.py --hf-dir <HF data/ folder> \
       --litqa2 <output of the HF repo's scripts/build_litqa2.py>
```

`prepare_unified_data.py` checks every benchmark file against the SHA-256 of the file
the paper runs read; all ten matched in our check: the nine hosted files of
`Shaow/BioHarness_Eval` at revision `f60c8fb2744dcfc048326226e6c0ecbbc110ffcb` and the
LitQA2 file written by that repository's `scripts/build_litqa2.py`. LitQA2 is not re-hosted: it is CC BY-SA 4.0 and
carries the LAB-Bench canary asking that it never appear in training corpora, and in our
coding-agent harness experiment web search returned public LAB-Bench mirrors exposing
labelled answers for 13 of the 199 items. The HF repository ships its item ids and a
build script pinned to a LAB-Bench revision.

**rlm.** The agent stage imports the Recursive Language Models library (`rlm`, version
0.1.0) from `.ref_project/rlm`. The research runs used upstream commit `db41d15`
(github.com/alexzhang13/rlm, 2026-01-05) with a local patch
(`third_party/rlm_db41d15_local.patch`): history pruning after every iteration (keep the
last 2 turns verbatim, compress older ones, 12,000-character cap), a 5,000-character cap
on REPL output shown to the model, `enable_thinking=False` and `max_tokens=4096` on every
call, and, from 2026-09-08, a 600 s request timeout with one retry. The PyPI
distribution `rlms==0.1.0` is a later, different code base and does not reproduce the
runs. The library must sit at `.ref_project/rlm` (as `setup_rlm.sh` places it), not only be
pip-installed: `src/rlm/` has the same package name and shadows an installed `rlm`
once `src/` is on `sys.path`; `src/rlm/pipeline.py` puts `.ref_project/rlm` first.

## Table 1 cells: runs, prompts and per-item outputs

Metric (official): per item, token-F1 for factoid items (`scripts/rescore_factoid_tokenf1.py`)
and the stored type-specific score otherwise; one record per id; mean per dataset.
Overall9 is the mean over the nine datasets other than LitQA2 weighted by full dataset
size (500 / 4,719 / 1,178 / 2,610 / 4,183 / 1,273 / 1,413 / 3,426 / 2,450 = 21,752 items):
67.213 from unrounded cells, printed 67.2.

| Cell (Ours) | Value | Run script(s) | Prompt | Per-item output in the framework snapshot |
|---|---|---|---|---|
| PubMedQA | 76.4 | `ours_april_eight_columns.sh` | `V_apr` | `output/pipeline/pubmedqa_pqal.jsonl` |
| BioASQ | 54.1 | `ours_april_eight_columns.sh` + `ours_disco_merge.sh` (87 routed items) | `V_apr`, routed items `V_pre` | `output/pipeline/bioasq.jsonl` |
| GeneTuring | 54.6 (a) | `ours_geneturing_genomics.sh` | `V_pre` | `output/pipeline/geneturing.jsonl` |
| SciHorizon | 60.3 (a) | `ours_april_eight_columns.sh` for the 2,400 non-expression items; `ours_scihorizon_expression.sh` for the 210 expression items | `V_apr`; case-study prompt | `output/pipeline/scihorizon_hgkb.jsonl` |
| MedMCQA | 74.9 | `ours_april_eight_columns.sh` | `V_apr` | `output/pipeline/medmcqa.jsonl` |
| MedQA-US | 85.9 | `ours_april_eight_columns.sh` | `V_apr` | `output/pipeline/medqa_US.jsonl` |
| MedQA-TW | 89.3 | `ours_april_eight_columns.sh` | `V_apr` | `output/pipeline/medqa_Taiwan.jsonl` |
| MedQA-CN | 89.7 | `ours_april_eight_columns.sh` | `V_apr` | `output/pipeline/medqa_Mainland.jsonl` |
| MedXpertQA | 37.1 | `ours_medxpertqa.sh` | `V_audit` | `output/pipeline/medxpertqa_text.jsonl` |
| LitQA2 | 61.8 (b) | `ours_litqa2_agentft.sh` (run 1) + `ours_litqa2_rerun142.sh` + `scripts/splice_litqa2_headline.py` | 57 base items `V_pre` (never escalated); 142 re-run items `V_cur` | `output/pipeline/litqa2.jsonl` |

The DISCO merge changes no SciHorizon non-expression item (all
164 routed SciHorizon items are expression items, and the case study replaces all 210),
so it matters for SciHorizon only in the unspliced 56.0 (55.7 without it).

Environment flags per cell (all other `XC_*` flags unset):

| Cell | Flags |
|---|---|
| Eight original columns (April 21-28) | none; top-k 20, concurrency 8 |
| GeneTuring genomics re-run (June 15) | none; method `v14-geneturing-genomics` |
| MedXpertQA (September 9) | `XC_MCQ_OFFICIAL=1 XC_MCQ_OFFICIAL_TOKENS=8192 XC_RERANK_RAW=1 XC_FAULTHANDLER=1`; concurrency 24 |
| LitQA2 base, agentft run 1 (September 9) | `XC_MCQ_OFFICIAL=1 XC_MCQ_OFFICIAL_TOKENS=8192 XC_FULLTEXT_CHUNKS=1 XC_RERANK_RAW=1 XC_EVIDENCE_DOC_CHARS=2400 XC_AGENT_FULLTEXT_TOOLS=1 XC_FAULTHANDLER=1`; concurrency 6 |
| LitQA2 142-item re-run (September 11) | the base flags plus `XC_ESCALATE_ON_ABSTAIN=1 XC_AGENT_QDRANT_TIMEOUT=300 XC_AGENT_MAX_ITERS=14 XC_ANSWER_CONTEXT_CHARS=60000 XC_REJUDGE_AGENT_TEXT=1 XC_REJUDGE_AGENT_EVIDENCE=0`, tracing on (`XC_TRACE_DIR`; the research run also set `XC_TRACE_GOLD_MAP`, see Known caveats), item filter `runs/item_filters/litqa2_rerun142.jsonl` |

The LitQA2 splice was re-checked for this release: `scripts/splice_litqa2_headline.py`
on the two research-run files gives 142 re-run items (53.52 on those items) + 57 base
items = **61.809**. The 142 ids are the 93 items agentft run 1 escalated plus the 49
non-escalated items on the research abstention list (48 chose the "Insufficient
information" option; 1, `litqa2_745f5a0d`, chose "None of the above").

Baselines: `runs/baselines_eight_columns.sh <method-id>` (eight original columns),
`runs/baselines_medxpertqa_official.sh <method-id>` (MedXpertQA; 16,384-token rationale
for No-Context, DenseRAG, HyDE, Chain-of-Note, Self-RAG, GraphRAG, PathRAG, LightRAG,
FLARE, BioRAG and GeneGPT-lite, 8,192 for DenseRAG+Rerank, IRCoT, BiomedRAG and Ours),
and `runs/baselines_litqa2_three_runs.sh <method-id>` (LitQA2, three runs, 597
item-runs per row). The script headers list the method id behind each Table 1 row.
The frontier block (API models without retrieval) is not covered by `runs/`.

## Reranker (`XC_RERANK_RAW`)

`XC_RERANK_RAW=1` scores with the official Qwen3-Reranker completion template (judge
system prompt, `<Instruct>/<Query>/<Document>` layout, empty think block) as
P(yes) / (P(yes) + P(no)) (`RerankClient._score_raw_template` in `src/utils/clients.py`).
The runs from 2026-09-08 on set it: MedXpertQA, the whole LitQA2 column, and the
MedXpertQA and LitQA2 baseline cells that use a reranker. The April and June runs used the
default chat-completions scorer (first-token log-probabilities of "yes" and "no").

## Known caveats

- **Answer-time evidence cap on baselines.** Every baseline answers through
  `shared_pipeline.generate_constrained_answer`, which truncates evidence to 8,000
  characters (`XC_ANSWER_CONTEXT_CHARS`), while Ours' first stage receives the full
  retrieved context. Measured once (DenseRAG+Rerank, LitQA2, three runs each): 58.1 capped
  against 60.0 uncapped, paired bootstrap +1.84 pp, 95 % CI [−0.50, +4.19].
- **Splices.** GeneTuring 54.6 and SciHorizon 60.3 are post-hoc, Ours-only additions
  (footnote a; 28.8 and 56.0 without them, Overall9 65.3). BioASQ 54.1 and the unspliced
  SciHorizon 56.0 include the DISCO router merge (54.0 and 55.7 without it). The April
  router file lists 15,631 predictions, 261 of them routed (164 SciHorizon, 87 BioASQ,
  4 PubMedQA, 2 MedMCQA, 4 MedQA-CN). `runs/ours_disco_merge.sh` keeps every item with
  `route_disco == true`, which is what the runner's item filter and the merge admit. The SciHorizon
  case study uses its own vocabulary-normalised set-F1 and scores the 35 items with empty
  gold as 1.0.
- **LitQA2 61.8 is not held out.** It is a single spliced run of a configuration chosen
  after error analysis on the same 199 items. The comparable number is footnote b's
  three-run mean of the earlier configuration, 57.8.
- **MedXpertQA budgets differ between rows** (8,192 vs 16,384 tokens, above); the effect
  was not measured. Three measurements of Ours' configuration gave 37.10, 37.39 and 37.71.
- **Research-only method ids and flags.** The runner registers many ablation and
  exploratory configurations; only those named in `runs/` produced Table 1 cells.
- **Gold-passage map not released.** The LitQA2 142-item research re-run also set
  `XC_TRACE_GOLD_MAP`, a file of gold-passage ids. `src/rlm/trace.py` uses it only to
  log whether a gold id was retrieved; it does not affect answers. The file is not
  released.
- **Golden-context LitQA2 baseline** runs on a `litqa2_gold` file (LAB-Bench key
  passage as context, 194 of 199 items) that is not released; the header of
  `runs/baselines_litqa2_three_runs.sh` says how to build it from LitQA2.
- **Coding-agent harness experiment not released.** The LitQA2 data policy cites the
  harness experiment in which web search exposed labelled answers for 13 of 199 items.
  That arm's code was removed from the runner (see the list of release edits above).
- **Run order.** Run `ours_april_eight_columns.sh` and `ours_disco_merge.sh` before the
  September scripts: `merge_disco_into_headline.py` copies every `*.jsonl` in the
  grounded method's output folder, where the MedXpertQA and LitQA2 scripts also write.
- **Atlas server not released** (next section).

## Atlas server (documented, not released)

The atlas component is served by a separate FastAPI service (`scdata_primitive_server`)
that is not part of this release. The code reaches it at `DISCO_SERVER_URL` (default
`http://127.0.0.1:8443`) with plain JSON POST requests:

| Endpoint | Request | Response `entries[]` fields | Data source |
|---|---|---|---|
| `/primitives/hpa/get_tissue_expression` | `{"gene": "<HGNC symbol>", "top_k": 30}` (the case study and the runner's expression shortcut send `gene` only) | `tissue`, `nx` (normalised expression, nTPM) | Human Protein Atlas tissue RNA expression table (bulk) |
| `/primitives/disco/get_tissue_expression` | `{"gene": ..., "top_k": 20}` | `tissue`, `nx`, `n_samples` | DISCO single-cell atlas, aggregated per tissue |
| `/primitives/disco/get_celltype_expression` | `{"gene": ..., "tissue": ..., "top_k": 15}` | `cell_type` (as `tissue::cell type`), `avg_expr`, `pct_cells`, `n_cells` | DISCO single-cell atlas, per cell type |

Two Table 1 cells depend on the server. BioASQ 54.1 includes 87 DISCO-routed items
(54.0 without them), answered by `v14-cascade-dual-rerank-grounded-disco`, which requests
HPA tissue rows, falls back to DISCO tissue rows when HPA returns nothing, and always
requests DISCO cell-type rows (`scripts/run_unified_benchmark.py`, `_fetch_one_gene`).
SciHorizon 60.3 depends on the HPA endpoint through the case study. In the SciHorizon case study
(`scripts/case_study_repair_context.py`) and in the atlas-routed expression path of the
runner, the gene symbol is taken from the question ("expression pattern of X gene"),
rows with `nx > 0` are mapped onto a fixed 27-tissue vocabulary (brain sub-regions to
`brain`, `adipose tissue` to `fat`, and so on), and the result is added to the prompt as
"Reference tissue expression (HPA nTPM ...): GENE: tissue:nx, ..." after the literature
evidence. If the server is unreachable or returns nothing, both paths fall back to the
literature-only prompt. Anyone with an HPA tissue expression table can stand up an
equivalent endpoint from this contract; results will differ with the HPA release.

## Third-party code

Each project below is an external dependency, the source of a port or the model for a
re-implementation. Two ports contain upstream text verbatim: the GraphRAG global-search
map/reduce prompts in `src/kg/graphrag/search.py` (and some clustering lines in
`community.py`), and the upstream context lines in `third_party/rlm_db41d15_local.patch`.
The MIT copyright and permission notices of GraphRAG, LightRAG and rlm are in
[`THIRD_PARTY_NOTICES.txt`](THIRD_PARTY_NOTICES.txt).

| Project | Upstream | Role | Licence |
|---|---|---|---|
| Recursive Language Models (`rlm`) | github.com/alexzhang13/rlm | agent REPL runtime; imported, pinned + patched (`third_party/`) | MIT, Copyright (c) 2025 Alex Zhang |
| Microsoft GraphRAG | github.com/microsoft/graphrag | ported into `src/kg/graphrag/`; map/reduce prompts copied verbatim | MIT, Copyright (c) Microsoft Corporation |
| LightRAG | github.com/HKUDS/LightRAG | ported into `src/kg/lightrag/` | MIT, Copyright (c) 2025 LightRAG Team |
| PathRAG | github.com/BUPT-GAMMA/PathRAG | re-implemented following the PathRAG paper and algorithm in `src/kg/pathrag/`; no upstream code is copied (the weighted-BFS scoring was rewritten and checked to give bit-identical scores) | upstream publishes no licence |
| FLARE | github.com/jzbjyb/FLARE | re-implemented as `FLAREClient` | MIT, Copyright (c) 2023 Zhengbao Jiang |
| Self-RAG | github.com/AkariAsai/self-rag | re-implemented as `SelfRAGClient` | MIT, Copyright (c) 2023 Akari Asai |
| IRCoT | github.com/StonyBrookNLP/ircot | re-implemented as `IRCoTClient` | Apache-2.0 |
| HyDE | github.com/texttron/hyde | re-implemented as `HyDEClient` | upstream publishes no licence |
| BIoMedRAG | github.com/ToneLi/BIoMedRAG | re-implemented as `BiomedRAGClient` | upstream publishes no licence |
| GeneGPT | github.com/ncbi/GeneGPT | re-implemented as `GeneGPTClient` / `GeneGPTLiteClient` | NCBI Public Domain Notice (US Government Work) |
| BioRAG | upstream URL not recorded in our tree | re-implemented in `benchmark/code/src/benchmark/biorag.py` | unknown |
| MedRAG | github.com/Teddy-XiongGZ/MedRAG | reference only | NCBI public domain notice |

## Licence

Our own code in this directory is released under the repository's licence (Apache-2.0,
`../LICENSE`). Text or code copied from the MIT projects above (the GraphRAG prompts and
clustering lines, the LightRAG port, the upstream context in
`third_party/rlm_db41d15_local.patch`) stays under MIT; the notices are in
`THIRD_PARTY_NOTICES.txt`. The PathRAG, HyDE, BIoMedRAG and BioRAG parts are our own
re-implementations from the published papers and algorithms; no upstream code from
projects without a licence is intended to be included, and we do not relicense any
upstream work. Benchmark data are not included; their licences are listed on the HF
dataset card.
