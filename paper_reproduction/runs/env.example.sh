# Service endpoints for the paper-reproduction runs. Copy to env.sh, edit, and the run
# scripts will source it. Every value below is a placeholder; see ../src/config.py.
export XC_LLM_SERVERS=http://127.0.0.1:8005/v1          # Qwen3.5-35B-A3B behind vLLM (OpenAI API)
export XC_LLM_MODEL=Qwen/Qwen3.5-35B-A3B                 # served model name
export XC_EMBED_SERVERS=http://127.0.0.1:8026/v1        # Qwen3-Embedding-0.6B (1024-d)
export XC_EMBED_MODEL=Qwen/Qwen3-Embedding-0.6B
export XC_RERANK_SERVERS=http://127.0.0.1:8015/v1       # Qwen3-Reranker-8B behind vLLM
export XC_RERANK_MODEL=Qwen/Qwen3-Reranker-8B
export XC_PG_HOST=127.0.0.1 XC_PG_PORTS=5432 XC_PG_USER=postgres XC_PG_PASSWORD=change-me
export XC_PG_PUBMED_DB=paper-graph-pubmed XC_PG_PAPERGRAPH_DB=papergraph
export XC_QDRANT_URLS=http://127.0.0.1:6333
# Atlas server (HPA/DISCO primitives; not released, see ../README.md). Only the
# DISCO-merge and SciHorizon expression steps call it.
export DISCO_SERVER_URL=http://127.0.0.1:8443
