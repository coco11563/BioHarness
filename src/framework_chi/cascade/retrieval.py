"""Dense retrieval + dual rerank.

Both functions are fail-soft: if the service raises, they return an
empty result so the cascade falls through to the agent path rather than
crashing the whole run.
"""

from __future__ import annotations

import logging
from typing import Any

LOGGER = logging.getLogger(__name__)


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
    except Exception as exc:  # noqa: BLE001
        LOGGER.warning("dense embed failed: %s", exc)
        return []
    try:
        return await qdrant_client.search(collection, vec, top_k=top_k)
    except Exception as exc:  # noqa: BLE001
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
    except Exception as exc:  # noqa: BLE001
        LOGGER.warning("rerank failed: %s; falling back to identity ranking", exc)
        return [1.0 - i / max(1, len(passages)) for i in range(len(passages))]
