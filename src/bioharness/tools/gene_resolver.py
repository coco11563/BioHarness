"""NCBI gene resolver.

Looks up a gene symbol or alias and returns the canonical record
(official symbol, full name, chromosomal location, aliases) via the
public NCBI Entrez E-utilities. No API key required; the public
endpoint is rate-limited to ~3 requests per second per IP, so the
implementation includes a small in-process throttle.

This tool is the open-source equivalent of the entity-lookup primitive
used by the bioHarness headline agent on GeneTuring factoid items.
The output schema mirrors the upstream ``gene_resolve`` registry entry
so the rejudge stage can interpret it without further translation.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from typing import Any

import httpx

LOGGER = logging.getLogger(__name__)

_ESEARCH = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
_ESUMMARY = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi"

# E-utilities rate-limit without an API key is ~3 req/s.
_RATE_LIMIT_SLEEP = 0.34
_LAST_CALL: float = 0.0
_RATE_LOCK = asyncio.Lock()


@dataclass(frozen=True)
class GeneRecord:
    symbol:    str
    name:      str
    chromosome: str        # in "chrN" form when available
    cytoband:   str
    aliases:    tuple[str, ...]
    entrez_id:  str

    def to_dict(self) -> dict[str, Any]:
        return {
            "symbol":    self.symbol,
            "name":      self.name,
            "chromosome": self.chromosome,
            "cytoband":   self.cytoband,
            "aliases":    list(self.aliases),
            "entrez_id":  self.entrez_id,
        }


async def _throttle() -> None:
    """Cooperative throttle to keep total request rate under the NCBI cap."""
    global _LAST_CALL  # noqa: PLW0603
    async with _RATE_LOCK:
        loop = asyncio.get_running_loop()
        now = loop.time()
        wait = _RATE_LIMIT_SLEEP - (now - _LAST_CALL)
        if wait > 0:
            await asyncio.sleep(wait)
        _LAST_CALL = loop.time()


async def resolve_gene(
    query: str,
    *,
    organism: str = "Homo sapiens",
    client: httpx.AsyncClient | None = None,
) -> GeneRecord | None:
    """Resolve a gene symbol / alias / Entrez id / accession to a canonical
    NCBI Gene record. Returns None on miss or any upstream error.
    """
    owns_client = client is None
    if owns_client:
        client = httpx.AsyncClient(timeout=15.0)

    try:
        await _throttle()
        params: dict[str, str | int] = {
            "db": "gene",
            "term": f'{query}[Gene Name] AND "{organism}"[Organism]',
            "retmax": 1,
            "retmode": "json",
        }
        r = await client.get(_ESEARCH, params=params)
        r.raise_for_status()
        id_list = r.json().get("esearchresult", {}).get("idlist") or []
        if not id_list:
            # fallback: looser query (no organism filter, no field qualifier)
            await _throttle()
            r = await client.get(_ESEARCH, params={
                "db": "gene", "term": query, "retmax": 1, "retmode": "json",
            })
            r.raise_for_status()
            id_list = r.json().get("esearchresult", {}).get("idlist") or []
        if not id_list:
            return None
        gid = id_list[0]

        await _throttle()
        r = await client.get(_ESUMMARY, params={
            "db": "gene", "id": gid, "retmode": "json",
        })
        r.raise_for_status()
        summary = r.json().get("result", {}).get(gid) or {}
        if not summary:
            return None

        cytoband = (summary.get("maplocation") or "").strip()
        chrom_raw = (summary.get("chromosome") or "").strip()
        chromosome = f"chr{chrom_raw}" if chrom_raw else ""
        aliases_raw = (summary.get("otheraliases") or "").strip()
        aliases = tuple(a.strip() for a in aliases_raw.split(",") if a.strip())

        return GeneRecord(
            symbol=(summary.get("name") or "").strip(),
            name=(summary.get("description") or "").strip(),
            chromosome=chromosome,
            cytoband=cytoband,
            aliases=aliases,
            entrez_id=gid,
        )
    except Exception as exc:
        LOGGER.warning("resolve_gene(%r) failed: %s", query, exc)
        return None
    finally:
        if owns_client and client is not None:
            await client.aclose()
