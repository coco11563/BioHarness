# Ablation flags

Every flag below is exposed as a constructor argument on
`V14CascadeClient` (via `CascadeOptions`) and as a `--method-args`
key on the `framework-eval run` command line.

| Flag | Default | What it changes |
| --- | :---: | --- |
| `--cascade-threshold FLOAT` | `0.7` | Logprob confidence below which items escalate to the agent. |
| `--enable-grounded-gate` | off | Escalate when the fast-path answer cannot be localised in the assembled context. Adds ~1 % accuracy on factoid + summary at ~3 % latency. |
| `--enable-dual-rerank` | on | HyDE + cross-encoder rerank on the dense retrieval result. Disabling drops ~2 pp on bioasq factoid. |
| `--enable-disco` | off | Route cell-type / expression questions through the scRNA atlas tool. Lifts `expression` subset by ~5.5 pp; off elsewhere. |
| `--max-agent-iterations INT` | `8` | REPL agent iteration cap. |
| `--no-tools` | off | Disable all tool pre-calls and agent tools. Reproduces the `v14-no-tools` ablation row. |
| `--retrieval-top-k INT` | `50` | Dense retrieval top-K before rerank. |
| `--rerank-top-k INT` | `10` | Cross-encoder rerank top-K passed to the prompt. |

## Reproducing published ablation rows

| Paper row | Flag overrides |
| --- | --- |
| `v14-cascade-dual-rerank` | `--cascade-threshold 0.7` |
| `v14-cascade-dual-rerank-grounded` (headline) | _defaults_ |
| `v14-cascade-dual-rerank-grounded-no-tools` | `--no-tools` |
| `v14-cascade-grounded` | `--no-dual-rerank --enable-grounded-gate` |
| `v14-cascade-dual-rerank-grounded-disco` | `--enable-disco` |
