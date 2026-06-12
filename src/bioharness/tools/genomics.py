"""Genomics database lookups (NCBI dbSNP + MyGene.info).

Open-source equivalent of the bioHarness headline agent's genomics primitives,
used on GeneTuring-style structured facts that literature retrieval cannot
answer:

- SNP (rs ID) -> associated gene / chromosome   (NCBI dbSNP E-utilities esummary)
- gene symbol -> chromosome / protein-coding?    (MyGene.info)

Async + fail-soft (returns ``None`` on miss/error) so the pre-call dispatcher
degrades gracefully. NCBI calls share the gene-resolver throttle to stay under
the public ~3 req/s cap.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Any

import httpx

from bioharness.tools.gene_resolver import _throttle

LOGGER = logging.getLogger(__name__)

_DBSNP_ESUMMARY = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi"
_MYGENE = "https://mygene.info/v3/query"
# Canonical human chromosomes (exclude alt/patch contigs like HSCHR16_1_CTG1).
_CANON_CHR = {str(i) for i in range(1, 23)} | {"X", "Y", "MT", "M"}


@dataclass(frozen=True)
class SNPRecord:
    rsid:       str
    gene:       str        # primary associated gene symbol ("" if none)
    chromosome: str        # "chrN" form when available
    position:   str        # e.g. "20:50298395"

    def to_dict(self) -> dict[str, Any]:
        return {"rsid": self.rsid, "gene": self.gene,
                "chromosome": self.chromosome, "position": self.position}


@dataclass(frozen=True)
class GeneGenomicRecord:
    symbol:            str
    chromosome:        str        # "chrN" form when available
    type_of_gene:      str        # e.g. "protein-coding", "ncRNA"
    is_protein_coding: bool | None
    protein_coding_answer: str    # "TRUE" / "FALSE" / ""

    def to_dict(self) -> dict[str, Any]:
        return {"symbol": self.symbol, "chromosome": self.chromosome,
                "type_of_gene": self.type_of_gene,
                "is_protein_coding": self.is_protein_coding,
                "protein_coding_answer": self.protein_coding_answer}


def _canon_chr(chrom: Any) -> str:
    if not chrom:
        return ""
    return str(chrom).replace("chr", "").replace("CHR", "").strip().upper()


def _pick_chr(genomic_pos: Any) -> str:
    """Pick the canonical chromosome from MyGene's genomic_pos (list/dict)."""
    if isinstance(genomic_pos, dict):
        return _canon_chr(genomic_pos.get("chr"))
    if isinstance(genomic_pos, list):
        cands = [_canon_chr(x.get("chr")) for x in genomic_pos if isinstance(x, dict)]
        return next((c for c in cands if c in _CANON_CHR), cands[0] if cands else "")
    return ""


async def snp_lookup(
    rsid: str, *, client: httpx.AsyncClient | None = None,
) -> SNPRecord | None:
    """Resolve a dbSNP rs ID to its associated gene and chromosome (NCBI dbSNP)."""
    m = re.search(r"rs(\d+)", str(rsid), re.IGNORECASE)
    sid = m.group(1) if m else re.sub(r"\D", "", str(rsid))
    if not sid:
        return None
    owns = client is None
    if owns:
        client = httpx.AsyncClient(timeout=15.0)
    try:
        await _throttle()
        r = await client.get(_DBSNP_ESUMMARY,
                             params={"db": "snp", "id": sid, "retmode": "json"})
        r.raise_for_status()
        rec = (r.json().get("result") or {}).get(sid) or {}
        if not rec:
            return None
        genes = [g.get("name") for g in (rec.get("genes") or []) if g.get("name")]
        chrom = _canon_chr(rec.get("chr"))
        return SNPRecord(
            rsid=f"rs{sid}",
            gene=genes[0] if genes else "",
            chromosome=f"chr{chrom}" if chrom else "",
            position=(rec.get("chrpos") or ""),
        )
    except Exception as exc:
        LOGGER.warning("snp_lookup(%r) failed: %s", rsid, exc)
        return None
    finally:
        if owns and client is not None:
            await client.aclose()


async def gene_genomic_info(
    symbol: str, *, client: httpx.AsyncClient | None = None,
) -> GeneGenomicRecord | None:
    """Resolve a gene symbol to chromosome + protein-coding status (MyGene.info)."""
    sym = str(symbol).strip()
    if not sym:
        return None
    owns = client is None
    if owns:
        client = httpx.AsyncClient(timeout=15.0)
    try:
        r = await client.get(_MYGENE, params={
            "q": sym, "species": "human",
            "fields": "symbol,type_of_gene,genomic_pos", "size": 1,
        })
        r.raise_for_status()
        hits = r.json().get("hits") or []
        if not hits:
            return None
        h = hits[0]
        tog = h.get("type_of_gene") or ""
        is_pc = (tog == "protein-coding") if tog else None
        chrom = _pick_chr(h.get("genomic_pos"))
        return GeneGenomicRecord(
            symbol=sym,
            chromosome=f"chr{chrom}" if chrom else "",
            type_of_gene=tog,
            is_protein_coding=is_pc,
            protein_coding_answer=("" if is_pc is None else ("TRUE" if is_pc else "FALSE")),
        )
    except Exception as exc:
        LOGGER.warning("gene_genomic_info(%r) failed: %s", symbol, exc)
        return None
    finally:
        if owns and client is not None:
            await client.aclose()
