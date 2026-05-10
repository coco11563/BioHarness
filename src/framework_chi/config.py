"""Service configuration loaded from environment variables.

Every endpoint defaults to a localhost SSH-tunnel address that matches
the layout described in ``docs/infra.md``. Override any of them via the
documented environment variables before constructing a V14CascadeClient.
"""

from __future__ import annotations

import os
from dataclasses import dataclass


def _env(name: str, default: str) -> str:
    return os.environ.get(name, default)


@dataclass(frozen=True)
class ServiceConfig:
    llm_url:        str
    embed_url:      str
    rerank_url:     str
    qdrant_url:     str
    pubmed_pg_url:  str
    papergraph_pg_url: str
    model_name:     str
    api_key:        str
    disco_url:      str | None = None

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
            disco_url   = os.environ.get("FRAMEWORK_DISCO_URL"),
        )

    def reachable_summary(self) -> dict[str, str]:
        """Return a {service: url} map for the doctor command."""
        out = {
            "llm":        self.llm_url,
            "embed":      self.embed_url,
            "rerank":     self.rerank_url,
            "qdrant":     self.qdrant_url,
            "pubmed_pg":  self.pubmed_pg_url,
            "papergraph_pg": self.papergraph_pg_url,
        }
        if self.disco_url:
            out["disco"] = self.disco_url
        return out


@dataclass(frozen=True)
class CascadeOptions:
    """Pipeline knobs exposed as CLI ablation flags.

    Defaults match the headline row ``v14-cascade-dual-rerank-grounded``;
    the ``-grounded`` suffix in the row name corresponds to
    ``enable_grounded_gate=True``.
    """

    cascade_threshold:    float = 0.7
    enable_grounded_gate: bool  = True
    enable_dual_rerank:   bool  = True
    enable_disco:         bool  = False
    max_agent_iterations: int   = 8
    no_tools:             bool  = False
    retrieval_top_k:      int   = 50
    rerank_top_k:         int   = 10
