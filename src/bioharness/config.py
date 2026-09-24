"""Service configuration loaded from environment variables.

Every endpoint defaults to a localhost SSH-tunnel address that matches
the layout described in ``docs/infra.md``. Override any of them via the
documented environment variables before constructing a PipelineCascadeClient.

This release ships a single, fixed best configuration of the cascade.
The one user-facing knob is ``force_agent``: when True, every item
bypasses the constrained-generation fast path and is routed through the
agent escalation + re-judgment stages. Useful for ablation runs that
want to measure the agent's contribution in isolation; off by default
because the cascade fast-path is faster and (on the BioHarness
headline) more accurate on average.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass

LOGGER = logging.getLogger(__name__)


def _legacy(name: str) -> str | None:
    """Deprecated ``FRAMEWORK_*`` alias of a ``BIOHARNESS_*`` var (warns once)."""
    if not name.startswith("BIOHARNESS_"):
        return None
    old = "FRAMEWORK_" + name[len("BIOHARNESS_"):]
    val = os.environ.get(old)
    if val is not None:
        LOGGER.warning("%s is deprecated; use %s instead", old, name)
    return val


def _env(name: str, default: str) -> str:
    val = os.environ.get(name)
    if val is None:
        val = _legacy(name)
    return default if val is None else val


def _env_bool(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        raw = _legacy(name)
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
    # Atlas component (D): off by default -> the cascade runs the -D expression
    # path. Set enable_atlas + a reachable atlas_url to activate +D.
    atlas_url:         str = "http://127.0.0.1:8443"
    enable_atlas:      bool = False

    @classmethod
    def from_env(cls) -> ServiceConfig:
        return cls(
            llm_url     = _env("BIOHARNESS_LLM_URL",     "http://127.0.0.1:8000/v1"),
            embed_url   = _env("BIOHARNESS_EMBED_URL",   "http://127.0.0.1:8002/v1"),
            rerank_url  = _env("BIOHARNESS_RERANK_URL",  "http://127.0.0.1:8001/v1"),
            qdrant_url  = _env("BIOHARNESS_QDRANT_URL",  "http://127.0.0.1:13335"),
            pubmed_pg_url = _env(
                "BIOHARNESS_PUBMED_PG",
                "postgresql://localhost:5432/paper-graph-pubmed",
            ),
            papergraph_pg_url = _env(
                "BIOHARNESS_PAPERGRAPH_PG",
                "postgresql://localhost:5432/papergraph",
            ),
            model_name  = _env("BIOHARNESS_MODEL_NAME", "Qwen3.5-35B-A3B"),
            api_key     = _env("BIOHARNESS_API_KEY",    "EMPTY"),
            force_agent = _env_bool("BIOHARNESS_FORCE_AGENT", False),
            atlas_url    = _env("BIOHARNESS_ATLAS_URL", "http://127.0.0.1:8443"),
            enable_atlas = _env_bool("BIOHARNESS_ENABLE_ATLAS", False),
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


# Fixed cascade constants of this re-implementation.
# These were tuned together with the prompt templates; do not edit without
# re-recording the cached smoke set.
CASCADE_THRESHOLD     = 0.7      # logprob-derived confidence below which we escalate
RETRIEVAL_TOP_K       = 20       # dense retrieval before rerank
RERANK_TOP_K          = 20       # passages handed to constrained generation (pipeline top_k=20)
MAX_AGENT_ITERATIONS  = 8        # upper bound for the agent escalation hook
DENSE_COLLECTION      = "paper-full"
