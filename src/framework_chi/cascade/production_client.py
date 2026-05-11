"""Full-delegation cascade client.

When the production source tree is available (``FRAMEWORK_PRODUCTION_SRC``
or sibling-detection), this class wraps the *entire* in-house production
cascade class from ``scripts/run_unified_benchmark.py`` and adapts it
to the ``framework_eval.plugins.QAClient`` protocol. Every stage —
retrieval, dual-rerank with the production-tuned rewrite+neg query
pool, constrained generation with logprob confidence, grounded gate,
agent escalation, and re-judgment — is the production code path, not a
re-implementation.

Use this client when "exactly aligned" matters. The simpler in-tree
``PipelineCascadeClient`` in :mod:`framework_chi.cascade.client` remains
the default for fully self-contained installs that do not have the
production source tree available.
"""

from __future__ import annotations

import importlib
import logging
import os
import sys
from typing import Any

from framework_eval.eval.types import Item, Prediction

from framework_chi.agent.production_adapter import _production_src_root

LOGGER = logging.getLogger(__name__)


def _load_production_pipeline_class() -> type[Any]:
    """Import the production cascade class from the upstream source tree.

    The class is referenced by its production name (``V14CascadeClient``)
    only inside the upstream module — this repo does not redefine it.
    """
    root = _production_src_root()
    if root is None:
        raise RuntimeError(
            "FRAMEWORK_PRODUCTION_SRC is not set and no sibling "
            "PaperAsKnowledgeGraph-RAG directory was found."
        )
    if root not in sys.path:
        sys.path.insert(0, root)
    # The class lives in scripts/run_unified_benchmark.py and is not in a
    # conventional importable module path; load it via importlib + spec.
    import importlib.util
    src_path = os.path.join(root, "scripts", "run_unified_benchmark.py")
    spec = importlib.util.spec_from_file_location("pipeline_production_runner", src_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load spec for {src_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    # Upstream class name is preserved as-is: this is the production identifier.
    return module.V14CascadeClient


class ProductionPipelineCascadeClient:
    """Thin wrapper around the production cascade class.

    Constructed with the same arguments as the ``pipeline`` headline:
    ``confidence_threshold=0.7, dual_rerank=True, grounded_gate=True``.

    Implements the ``framework_eval.plugins.QAClient`` protocol so it can
    be registered as a method and driven from ``framework-eval run``.
    """

    name = "pipeline"

    def __init__(self) -> None:
        cls = _load_production_pipeline_class()
        self._client = cls(
            top_k=20,
            confidence_threshold=0.7,
            version="v14",
            dual_rerank=True,
            grounded_gate=True,
            enable_tools=True,
        )
        self._entered = False

    async def _ensure_entered(self) -> None:
        if self._entered:
            return
        await self._client.__aenter__()
        self._entered = True

    async def generate(self, item: Item) -> Prediction:
        await self._ensure_entered()
        # Match the upstream runner.py contract: pass context=None unless
        # the use-golden-context flag is set. The headline cascade run does
        # NOT use the dataset's gold ideal_answer passages; the cascade
        # does its own dense retrieval inside.
        try:
            resp = await self._client.generate(
                item.question,
                item.question_type,
                context=None,
                options=item.options,
                item_key=(item.dataset, item.id),
            )
        except TypeError:
            resp = await self._client.generate(
                item.question,
                item.question_type,
                context=None,
                options=item.options,
            )

        meta: dict[str, Any] = {}
        if getattr(resp, "metadata", None):
            meta.update(resp.metadata)
        return Prediction(
            item_id=item.id,
            answer=(getattr(resp, "answer", None) or "").strip(),
            extras={
                "response_text": getattr(resp, "response_text", "") or "",
                "latency_ms":    getattr(resp, "latency_ms", 0.0),
                **meta,
            },
        )

    async def aclose(self) -> None:
        if self._entered:
            try:
                await self._client.__aexit__(None, None, None)
            finally:
                self._entered = False
