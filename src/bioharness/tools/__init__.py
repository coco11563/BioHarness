"""Tool registry + pre-call dispatch.

The bioHarness headline pipeline pre-calls a small set of entity-lookup
tools before agent escalation so the agent (and the constrained
re-judgment stage) sees authoritative gene / protein / pathway data.
This module implements the open-source equivalent of that pre-call.
"""

from __future__ import annotations

import logging
import re
from typing import Any

from bioharness.tools.gene_resolver import GeneRecord, resolve_gene
from bioharness.tools.genomics import (
    GeneGenomicRecord,
    SNPRecord,
    blast_align,
    gene_genomic_info,
    snp_lookup,
)

LOGGER = logging.getLogger(__name__)


REGISTRY: dict[str, str] = {
    "gene_resolver": "Resolve a gene symbol/alias to its official symbol, "
                      "chromosomal location, name and aliases via NCBI Gene.",
    "snp_lookup": "Resolve a dbSNP rs ID to its associated gene and "
                  "chromosome via NCBI dbSNP.",
    "gene_genomic_info": "Resolve a gene symbol to its chromosome and "
                         "protein-coding status via MyGene.info.",
    "blast_align": "Align a DNA sequence to the human genome (-> chrN:start-end) "
                   "or identify its source organism via NCBI BLAST (slow; "
                   "precomputed offline in the headline pipeline).",
}


# Heuristics borrowed from the upstream pipeline; entities are gene-symbol-
# shaped tokens, with common English stopwords filtered.
_ENTITY_PHRASE_RE = re.compile(
    r"(?:gene symbol of |gene |protein |about |for |of |with )"
    r"([A-Z][A-Za-z0-9._-]{1,20})\b"
)
_ENTITY_BARE_RE = re.compile(r"\b([A-Z][A-Z0-9._-]{1,20})\b")
_STOPWORDS = {
    "THE", "AND", "FOR", "WITH", "FROM", "WHAT", "WHICH", "WHERE",
    "HOW", "NOT", "DOES", "WHO", "WHY", "WHEN", "CAN",
}


def extract_gene_entities(question: str, *, limit: int = 3) -> list[str]:
    """Pull up to ``limit`` gene-symbol-shaped tokens out of a question."""
    out: list[str] = []
    out += _ENTITY_PHRASE_RE.findall(question)
    out += _ENTITY_BARE_RE.findall(question)
    deduped: list[str] = []
    seen: set[str] = set()
    for token in out:
        upper = token.upper()
        if upper in _STOPWORDS or upper in seen:
            continue
        seen.add(upper)
        deduped.append(token)
        if len(deduped) >= limit:
            break
    return deduped


_GENE_TRIGGER_WORDS = (
    "gene", "symbol", "alias", "chromosome", "location",
    "cytoband", "snp", "official",
)


async def precall_tools(
    question: str,
    question_type: str,
    *,
    http_client: Any | None = None,
) -> str:
    """Return formatted tool evidence (or "" on no hits).

    For factoid items mentioning genes, this resolves each candidate
    symbol via NCBI and formats the result as a prompt-friendly block:

        ## Tool Results (pre-fetched)
        [gene_info(BRCA1)] symbol=BRCA1, name=..., chromosome=chr17, ...
    """
    q_lower = question.lower()
    if not any(w in q_lower for w in _GENE_TRIGGER_WORDS):
        return ""

    entities = extract_gene_entities(question)
    if not entities:
        return ""

    records: list[tuple[str, GeneRecord | None]] = []
    for ent in entities:
        rec = await resolve_gene(ent, client=http_client)
        records.append((ent, rec))

    parts: list[str] = []
    for ent, rec in records:
        if rec is None or not rec.symbol:
            continue
        bits = [f"symbol={rec.symbol}"]
        if rec.name:
            bits.append(f"name={rec.name}")
        if rec.chromosome:
            bits.append(
                f"chromosome={rec.chromosome} (use this exact format for "
                "chromosome answers)"
            )
        if rec.cytoband:
            bits.append(f"cytoband={rec.cytoband}")
        if rec.aliases:
            bits.append(f"aliases={list(rec.aliases)}")
        parts.append(f"[gene_info({ent})] {', '.join(bits)}")

    # Genomics lookups: dbSNP rs IDs -> gene/chromosome; and, for protein-coding
    # / chromosome questions, gene -> chromosome + protein-coding status.
    for rsid in re.findall(r"\brs\d{3,}\b", question, re.IGNORECASE)[:3]:
        snp = await snp_lookup(rsid, client=http_client)
        if snp is None:
            continue
        bits = []
        if snp.gene:
            bits.append(f"gene={snp.gene}")
        if snp.chromosome:
            bits.append(f"chromosome={snp.chromosome} (use this exact format)")
        if bits:
            parts.append(f"[snp_lookup({rsid})] {', '.join(bits)}")

    if any(w in q_lower for w in ("codes a protein", "protein-coding",
                                  "protein coding", "which chromosome", "located on")):
        for ent in entities:
            gi = await gene_genomic_info(ent, client=http_client)
            if gi is None:
                continue
            bits = []
            if gi.protein_coding_answer:
                bits.append(f"protein_coding={gi.protein_coding_answer} (answer TRUE or FALSE)")
            if gi.chromosome:
                bits.append(f"chromosome={gi.chromosome} (use this exact format)")
            if bits:
                parts.append(f"[gene_genomic_info({ent})] {', '.join(bits)}")

    if not parts:
        return ""
    return "## Tool Results (pre-fetched)\n" + "\n".join(parts)


__all__ = [
    "REGISTRY", "GeneRecord", "GeneGenomicRecord", "SNPRecord",
    "extract_gene_entities", "precall_tools", "resolve_gene",
    "snp_lookup", "gene_genomic_info", "blast_align",
]
