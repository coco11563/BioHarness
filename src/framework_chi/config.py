"""Service configuration loaded from environment variables.

Every endpoint defaults to a localhost SSH-tunnel address that matches
the layout described in ``docs/infra.md``. Override any of them via the
documented environment variables before constructing a PipelineCascadeClient.

This release ships a single, fixed best configuration of the cascade.
The one user-facing knob is ``force_agent``: when True, every item
bypasses the constrained-generation fast path and is routed through the
agent escalation + re-judgment stages. Useful for ablation runs that
want to measure the agent's contribution in isolation; off by default
because the cascade fast-path is faster and (on the {framework}^χ
headline) more accurate on average.
"""

from __future__ import annotations

import os
from dataclasses import dataclass


def _env(name: str, default: str) -> str:
    return os.environ.get(name, default)


def _env_bool(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in ("1", "true", "yes", "on")


@dataclass(frozen=True)
class ServiceConfig:
    llm_url:           str
    embed_url:         str
    rerank_url:        str
    qdrant_url:        str
    pubmed_pg_url:     str
    papergraph_pg_url: str
    model_name:        str
    api_key:           str
    force_agent:       bool = False

    @classmethod
    def from_env(cls) -> "ServiceConfig":
        return cls(
            llm_url     = _env("FRAMEWORK_LLM_URL",     "http://127.0.0.1:8000/v1"),
            embed_url   = _env("FRAMEWORK_EMBED_URL",   "http://127.0.0.1:8002/v1"),
            rerank_url  = _env("FRAMEWORK_RERANK_URL",  "http://127.0.0.1:8001/v1"),
            qdrant_url  = _env("FRAMEWORK_QDRANT_URL",  "http://127.0.0.1:13335"),
            pubmed_pg_url = _env(
                "FRAMEWORK_PUBMED_PG",
                "postgresql://localhost:5432/paper-graph-pubmed",
            ),
            papergraph_pg_url = _env(
                "FRAMEWORK_PAPERGRAPH_PG",
                "postgresql://localhost:5432/papergraph",
            ),
            model_name  = _env("FRAMEWORK_MODEL_NAME", "{model}"),
            api_key     = _env("FRAMEWORK_API_KEY",    "EMPTY"),
            force_agent = _env_bool("FRAMEWORK_FORCE_AGENT", False),
        )

    def reachable_summary(self) -> dict[str, str]:
        return {
            "llm":            self.llm_url,
            "embed":          self.embed_url,
            "rerank":         self.rerank_url,
            "qdrant":         self.qdrant_url,
            "pubmed_pg":      self.pubmed_pg_url,
            "papergraph_pg":  self.papergraph_pg_url,
        }


# Fixed cascade constants used by the {framework}^χ headline configuration.
# These were tuned together with the prompt templates; do not edit without
# re-recording the cached smoke set.
CASCADE_THRESHOLD     = 0.7      # logprob-derived confidence below which we escalate
RETRIEVAL_TOP_K       = 20       # dense retrieval before rerank
RERANK_TOP_K          = 10       # passages handed to constrained generation
MAX_AGENT_ITERATIONS  = 8        # upper bound for the agent escalation hook
DENSE_COLLECTION      = "paper-full"
