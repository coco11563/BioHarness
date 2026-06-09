"""Atlas client (D component, ``+D`` path).

Fetches a gene's Human Protein Atlas (HPA) bulk tissue expression from a
primitive server (the ``scdata_primitive_server`` endpoint
``POST /primitives/hpa/get_tissue_expression``) and renders it as the
supplementary reference block consumed by the expression prompt.

Fail-soft by design: any transport / HTTP / shape error returns ``None`` so the
cascade silently falls back to the ``-D`` (no-atlas) expression path. Enable it
with ``BIOHARNESS_ENABLE_ATLAS=1`` and point ``BIOHARNESS_ATLAS_URL`` at the
server (default ``http://127.0.0.1:8443``).
"""

from __future__ import annotations

import logging

import httpx

from bioharness.cascade.atlas import atlas_rows_from_hpa

LOGGER = logging.getLogger(__name__)


class AtlasClient:
    def __init__(self, base_url: str, *, timeout: float = 20.0) -> None:
        self._base_url = base_url.rstrip("/")
        self._client = httpx.AsyncClient(base_url=self._base_url, timeout=timeout)

    async def aclose(self) -> None:
        await self._client.aclose()

    async def tissue_rows(self, gene: str) -> str | None:
        """Return ``"GENE: tissue:nx, ..."`` for the prompt, or ``None`` on miss."""
        if not gene:
            return None
        try:
            resp = await self._client.post(
                "/primitives/hpa/get_tissue_expression", json={"gene": gene},
            )
            if resp.status_code != 200:
                return None
            entries = resp.json().get("entries") or []
        except Exception as exc:  # fail-soft: fall back to the -D path
            LOGGER.warning("atlas fetch failed for %s: %s", gene, exc)
            return None
        return atlas_rows_from_hpa(gene, entries)
