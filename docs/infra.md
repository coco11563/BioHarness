# Infrastructure requirements for `bioHarness`

`bioharness` does not bundle the inference services it relies on.
Every endpoint listed below must be reachable before
`framework-eval run --method pipeline ...`
can produce headline-grade numbers. Run `bioharness doctor` to
self-check connectivity.

## Required services

| Service | Default URL | Purpose | Suggested implementation |
| --- | --- | --- | --- |
| LLM | `http://127.0.0.1:8000/v1` | Constrained gen + agent escalation | vLLM serving `{model}` (32 B – 70 B parameters) |
| Embedding | `http://127.0.0.1:8002/v1` | 1024-dim dense passage retrieval | `Qwen3-Embedding-0.6B` or compatible |
| Reranker | `http://127.0.0.1:8001/v1` | Cross-encoder rerank | `Qwen3-Reranker-8B` or compatible |
| Qdrant | `http://127.0.0.1:13335` | Dense vector index over PubMed | Qdrant 1.9+; collection `paper-full` (27.3 M points, 1024-dim) |
| Postgres (PubMed) | `postgresql://localhost:5432/paper-graph-pubmed` | Article + MeSH metadata | Postgres 14+ with the `pgvector` extension |
| Postgres (papergraph) | `postgresql://localhost:5432/papergraph` | PMC full-text + chunks | Postgres 14+ |

## Optional services

| Service | Env var | Required when |
| --- | --- | --- |
| Disco scRNA atlas | `BIOHARNESS_DISCO_URL` | `--enable-disco` is set |

## Environment variable summary

```bash
export BIOHARNESS_LLM_URL=http://127.0.0.1:8000/v1
export BIOHARNESS_EMBED_URL=http://127.0.0.1:8002/v1
export BIOHARNESS_RERANK_URL=http://127.0.0.1:8001/v1
export BIOHARNESS_QDRANT_URL=http://127.0.0.1:13335
export BIOHARNESS_PUBMED_PG=postgresql://user:pass@localhost:5432/paper-graph-pubmed
export BIOHARNESS_PAPERGRAPH_PG=postgresql://user:pass@localhost:5432/papergraph
export BIOHARNESS_MODEL_NAME={model}
export BIOHARNESS_API_KEY=EMPTY            # vLLM accepts any non-empty value
# Optional:
export BIOHARNESS_DISCO_URL=http://127.0.0.1:8443
```

## Pre-flight check

```bash
bioharness doctor
```

Output is one line per service with reachability status; non-zero exit
indicates at least one required service is unreachable. The `disco` row
is informational.

## Building the Qdrant collection

The `paper-full` collection is built from the upstream PubMed dump.
Building it is out of scope for this repository; refer to the embedding
provider's documentation for an end-to-end recipe. As long as the
collection name, vector size (1024), and payload field `text` match the
expectations in `bioharness/clients/qdrant.py`, the cascade pipeline
will work.

## Resource sizing

The numbers below are what the bioHarness paper used; smaller
deployments work but may not reproduce the headline accuracy.

| Component | Memory | Disk |
| --- | --- | --- |
| LLM (`{model}`) | 80 GB GPU | 90 GB |
| Embedding | 4 GB GPU | 2 GB |
| Reranker | 16 GB GPU | 16 GB |
| Qdrant `paper-full` | 64 GB RAM | 250 GB SSD |
| Postgres mirrors | 32 GB RAM | 600 GB SSD |
