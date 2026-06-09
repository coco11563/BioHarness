"""Dense retrieval + dual rerank.

Both functions are fail-soft: if the service raises, they return an
empty result so the cascade falls through to the agent path rather than
crashing the whole run.
"""

from __future__ import annotations

import logging
from typing import Any

LOGGER = logging.getLogger(__name__)


def merge_passages(
    a: list[dict[str, Any]],
    b: list[dict[str, Any]],
    *,
    limit: int,
) -> list[dict[str, Any]]:
    """Union two passage lists, deduplicating by passage id, keeping the
    higher dense-retrieval score on a tie. Used by yesno dual retrieval."""
    seen: dict[Any, dict[str, Any]] = {}
    for src in (a, b):
        for p in src or []:
            pid = p.get("id")
            if pid is None:
                pid = (p.get("text") or "")[:64]
            existing = seen.get(pid)
            if existing is None or (p.get("score") or 0) > (existing.get("score") or 0):
                seen[pid] = p
    out = sorted(seen.values(), key=lambda p: -(p.get("score") or 0.0))
    return out[:limit]


async def dense_retrieve(
    embed_client: Any,
    qdrant_client: Any,
    question: str,
    *,
    top_k: int,
    collection: str = "paper-full",
) -> list[dict[str, Any]]:
    """Embed the question and search the dense vector index."""
    try:
        vec = await embed_client.embed_one(question)
    except Exception as exc:
        LOGGER.warning("dense embed failed: %s", exc)
        return []
    try:
        return await qdrant_client.search(collection, vec, top_k=top_k)
    except Exception as exc:
        LOGGER.warning("qdrant search failed: %s", exc)
        return []


async def dual_rerank(
    rerank_client: Any,
    question: str,
    passages: list[dict[str, Any]],
    *,
    top_k: int,
) -> list[float]:
    """Cross-encoder rerank; returns scores aligned with the input order."""
    if not passages:
        return []
    texts = [(p.get("text") or "") for p in passages]
    try:
        return await rerank_client.score(question, texts, top_k=top_k)
    except Exception as exc:
        LOGGER.warning("rerank failed: %s; falling back to identity ranking", exc)
        return [1.0 - i / max(1, len(passages)) for i in range(len(passages))]
