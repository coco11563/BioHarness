# XCompass_Chi

**The {framework}^χ headline method** —
``v14-cascade-dual-rerank-grounded`` — packaged as a registered
``framework_eval.methods`` plugin so the
[XCompass_Eval_Framework](https://github.com/coco11563/XCompass_Eval_Framework)
harness can score it on the
[Shaow/GeneKnowledgeEval](https://huggingface.co/datasets/Shaow/GeneKnowledgeEval)
benchmark (8 datasets, 19,474 items, 7 question types).

Headline reproduced by the {framework} paper:

| Method | Items | Binary acc | Continuous mean |
| --- | ---: | ---: | ---: |
| **{framework}^χ** (`v14-cascade-dual-rerank-grounded`) | **19,302** | **0.766** | **0.691** |

> **Placeholders.** This release uses the literal tokens `{framework}` and
> `{model}` wherever a paper-specific name would otherwise appear. Replace
> them with the names that match your own deployment when reading the docs.

---

## Architecture

`{framework}^χ` is a **context-assembly + RLM + adaptive cascade** pipeline:

1. **Adaptive context assembly.** Dense retrieval against a 27.3 M PubMed
   index, optional dual retrieval (positive + negative) for yes/no items,
   automatic tool calls (gene resolver, UniProt, GO).
2. **Constrained generation with logprob confidence.** A short
   benchmark-formatted answer (`max_tokens=4`) plus the routing logprob.
3. **Adaptive cascade.** High-confidence items return the fast-path answer
   immediately (~2 s). Low-confidence items escalate to the full RLM
   agent.
4. **RLM agent (escalation only).** REPL-based iterative reasoning with
   Python execution, `max_iterations=8`, access to PubMed / gene /
   UniProt / web / ChEMBL tools.
5. **Optional grounded gate.** When the constrained answer cannot be
   localised in the assembled context, escalate. Off by default; toggle
   with `--enable-grounded-gate`.
6. **Constrained re-judgment.** The agent's free-form answer goes back
   through a `max_tokens=4` constrained generation that emits the
   benchmark-compliant string.
7. **Optional Disco scRNA atlas.** Routed evidence tool for cell-type /
   expression questions; toggle with `--enable-disco`.

---

## Quickstart

### 1. Install

```bash
pip install framework-eval framework-chi
```

If you are working from source:

```bash
git clone https://github.com/coco11563/XCompass_Chi.git
cd XCompass_Chi
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
```

### 2. Stand up the inference services

`framework-chi` does NOT bundle the LLM, embedding, reranker, Qdrant, or
Postgres services it needs. The full requirements are documented in
`docs/infra.md`. Once they are reachable, point the framework at them via
environment variables:

```bash
export FRAMEWORK_LLM_URL=http://127.0.0.1:8000/v1
export FRAMEWORK_EMBED_URL=http://127.0.0.1:8002/v1
export FRAMEWORK_RERANK_URL=http://127.0.0.1:8001/v1
export FRAMEWORK_QDRANT_URL=http://127.0.0.1:13335
export FRAMEWORK_PUBMED_PG=postgresql://localhost:5432/paper-graph-pubmed
export FRAMEWORK_PAPERGRAPH_PG=postgresql://localhost:5432/papergraph
export FRAMEWORK_MODEL_NAME={model}
```

`framework-chi doctor` runs a pre-flight connectivity check against every
service and prints which are reachable and which are not.

### 3. Score the benchmark

`framework-chi` is registered as the
``v14-cascade-dual-rerank-grounded`` plugin under
``framework_eval.methods``. Run it through the harness:

```bash
framework-eval run \
  --method v14-cascade-dual-rerank-grounded \
  --datasets bioasq scihorizon-gene geneturing medmcqa \
             medqa_us medqa_taiwan medqa_mainland pubmedqa_pqal_test \
  --output runs/chi/

framework-eval score --run runs/chi/ --output runs/chi/scores/
```

Expected: binary accuracy within ±2.5 pp of the paper headline (0.766)
under the live tolerance band documented in `MANIFEST.toml`
(``[verify.live]``).

### 4. Reproduce the headline (offline, no infrastructure)

If you only need the headline numbers without re-running inference, the
shipped run snapshot in
[XCompass_Eval_Framework](https://github.com/coco11563/XCompass_Eval_Framework)
already reproduces them byte-for-byte:

```bash
git clone https://github.com/coco11563/XCompass_Eval_Framework.git
cd XCompass_Eval_Framework
python verify.py
```

Use this repo only when you want to **re-run** the method end-to-end on
your own infrastructure.

---

## Ablation flags

Several pipeline knobs are exposed as CLI flags so you can ablate the
headline configuration without forking the code. Pass them through
``framework-eval run --method-args``:

| Flag | Default | What it does |
| --- | :---: | --- |
| `--enable-grounded-gate` | off | Escalate when the fast-path answer cannot be localised in the assembled context. |
| `--enable-dual-rerank` | on | HyDE + cross-encoder rerank on the dense retrieval result. |
| `--enable-disco` | off | Route cell-type / expression questions through the Disco atlas tool. |
| `--cascade-threshold` | 0.7 | Logprob confidence below which items escalate to the agent. |
| `--max-agent-iterations` | 8 | REPL agent iteration cap. |
| `--no-tools` | off | Disable all tool pre-calls and agent tools (no-tools ablation). |

---

## Repository layout

```
XCompass_Chi/
├── MANIFEST.toml                # method-side reproducibility manifest
├── README.md
├── LICENSE
├── CHANGELOG.md
├── CITATION.cff
├── CONTRIBUTING.md
├── pyproject.toml
├── verify.py                    # offline cached smoke + --live re-run
├── src/framework_chi/
│   ├── cascade/                 # V14CascadeClient (entry point)
│   ├── agent/                   # REPL agent escalation path
│   ├── tools/                   # gene resolver, UniProt, GO, Disco wrappers
│   ├── clients/                 # OpenAI-compatible LLM/embed/rerank clients
│   └── cli/                     # framework-chi CLI (doctor, etc.)
├── tests/                       # unit + integration tests
├── docs/
│   ├── infra.md                 # service requirements + setup
│   ├── ablations.md             # what each --enable-* flag changes
│   └── adapter.md               # how the QAClient maps to V14CascadeClient
├── golden/
│   └── cached_smoke.jsonl       # 50-item recorded prompt/response set
├── scripts/
│   ├── scan_forbidden_strings.py
│   └── build_cached_smoke.py    # records the smoke set against live infra
└── .github/workflows/ci.yml
```

---

## License

Apache-2.0 (see `LICENSE`).

## Contact

Issues + PRs welcome at <https://github.com/coco11563/XCompass_Chi/issues>.
