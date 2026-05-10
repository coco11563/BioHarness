# Ablation flags

Every flag below maps to a field of `CascadeOptions` and is honoured by
`V14CascadeClient` directly. Inject them by constructing the client with
a custom `CascadeOptions` and registering the result as your own
entry-point method (see `docs/adapter.md`); the bare
`framework-eval run --method v14-cascade-dual-rerank-grounded` invocation
constructs the headline configuration with the defaults below.

| Field | Default | What it changes |
| --- | :---: | --- |
| `cascade_threshold` | `0.7` | Logprob confidence below which items escalate to the agent. |
| `enable_grounded_gate` | **on** | Escalate when the fast-path answer cannot be localised in the assembled context. The `-grounded` suffix in the headline method id reflects this default. |
| `enable_dual_rerank` | on | HyDE + cross-encoder rerank on the dense retrieval result. Disabling drops ~2 pp on bioasq factoid. |
| `enable_disco` | off | Route cell-type / expression questions through the scRNA atlas tool. Lifts `expression` subset by ~5.5 pp; off elsewhere. |
| `max_agent_iterations` | `8` | Upper bound for REPL agent iterations (default `_agent` runs one). |
| `no_tools` | off | Disable all tool pre-calls and agent tools. Reproduces the `v14-no-tools` ablation row. |
| `retrieval_top_k` | `50` | Dense retrieval top-K before rerank. |
| `rerank_top_k` | `10` | Cross-encoder rerank top-K passed to the prompt. |

## Reproducing published ablation rows

To produce a row, instantiate `V14CascadeClient(options=CascadeOptions(...))`
and register it under a fresh entry point:

| Paper row | CascadeOptions overrides |
| --- | --- |
| `v14-cascade-dual-rerank` | `enable_grounded_gate=False` |
| `v14-cascade-dual-rerank-grounded` (headline) | _defaults_ |
| `v14-cascade-dual-rerank-grounded-no-tools` | `no_tools=True` |
| `v14-cascade-grounded` | `enable_dual_rerank=False` |
| `v14-cascade-dual-rerank-grounded-disco` | `enable_disco=True` |
