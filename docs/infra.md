# Infrastructure requirements for `BioHarness`

`bioharness` does not bundle the inference services it relies on.
The LLM, embedding, reranker and Qdrant endpoints below must be reachable
before `framework-eval run --method pipeline ...` can run; the two
Postgres databases are optional for `pipeline`. Even with all of them
in place, this package does not reproduce the paper's Table 1 columns (see
the README). Run `bioharness doctor` to self-check connectivity.

## Required services

| Service | Default URL | Purpose | Suggested implementation |
| --- | --- | --- | --- |
| LLM | `http://127.0.0.1:8000/v1` | Constrained gen + agent escalation | vLLM serving `Qwen3.5-35B-A3B` (the paper's backbone; every call sets `enable_thinking=False`) |
| Embedding | `http://127.0.0.1:8002/v1` | 1024-dim dense passage retrieval | `Qwen3-Embedding-0.6B` or compatible |
| Reranker | `http://127.0.0.1:8001/v1` | Cross-encoder rerank | `Qwen3-Reranker-8B` served for `/v1/completions` with logprobs (raw template), or any `/v1/rerank` endpoint |
| Qdrant | `http://127.0.0.1:13335` | Dense vector index over PubMed | Qdrant 1.9+; collection `paper-full` (27.3 M points, 1024-dim) |
| Postgres (PubMed), optional | `postgresql://localhost:5432/paper-graph-pubmed` | Article + MeSH metadata (probed by `bioharness doctor`; not used by the in-tree cascade) | Postgres 14+ with the `pgvector` extension |
| Postgres (papergraph), optional | `postgresql://localhost:5432/papergraph` | PMC full-text + chunks (probed by `bioharness doctor`; not used by the in-tree cascade) | Postgres 14+ |

## Optional services

| Service | Env var | Required when |
| --- | --- | --- |
| HPA tissue-expression endpoint (`POST /primitives/hpa/get_tissue_expression`) | `BIOHARNESS_ATLAS_URL` | `BIOHARNESS_ENABLE_ATLAS=1` is set |

The atlas server used in the paper is documented but not released; the
atlas path needs an endpoint with that contract that you provide, and
falls back to the no-atlas prompt when it is unreachable.

## Environment variable summary

```bash
export BIOHARNESS_LLM_URL=http://127.0.0.1:8000/v1
export BIOHARNESS_EMBED_URL=http://127.0.0.1:8002/v1
export BIOHARNESS_RERANK_URL=http://127.0.0.1:8001/v1
export BIOHARNESS_QDRANT_URL=http://127.0.0.1:13335
export BIOHARNESS_PUBMED_PG=postgresql://user:pass@localhost:5432/paper-graph-pubmed
export BIOHARNESS_PAPERGRAPH_PG=postgresql://user:pass@localhost:5432/papergraph
export BIOHARNESS_MODEL_NAME=Qwen3.5-35B-A3B   # the served model id
export BIOHARNESS_API_KEY=EMPTY            # vLLM accepts any non-empty value
# Optional:
export BIOHARNESS_ENABLE_ATLAS=1
export BIOHARNESS_ATLAS_URL=http://127.0.0.1:8443
```

## Pre-flight check

```bash
bioharness doctor
```

Output is one line per service with reachability status; non-zero exit
indicates at least one probed service is unreachable. `doctor` also probes
the two Postgres databases and exits 1 when they are unreachable, although
`pipeline` does not use them; ignore those two lines if you run only
`pipeline`.

## Building the Qdrant collection

The `paper-full` collection is built from the upstream PubMed dump.
Building it is out of scope for this repository; refer to the embedding
provider's documentation for an end-to-end recipe. As long as the
collection name and vector size (1024) match and each point's payload
carries the passage text as `text` (or `abstract`, read when `text` is
absent) and optionally `title` (prepended to the text for reranking), as
read by `bioharness/clients/qdrant.py` and `cascade/retrieval.py`, the
cascade pipeline will work.

## Resource sizing

The numbers below are indicative, not measured for this release.

| Component | Memory | Disk |
| --- | --- | --- |
| LLM (`Qwen3.5-35B-A3B`) | 80 GB GPU | 90 GB |
| Embedding | 4 GB GPU | 2 GB |
| Reranker | 16 GB GPU | 16 GB |
| Qdrant `paper-full` | 64 GB RAM | 250 GB SSD |
| Postgres mirrors | 32 GB RAM | 600 GB SSD |
