"""Offline tests for the genomics tools (dbSNP + MyGene) and pre-call wiring.

Network is stubbed with httpx.MockTransport so no live NCBI/MyGene calls are made.
"""

from __future__ import annotations

import json

import httpx
import pytest

from bioharness.tools import gene_genomic_info, precall_tools, snp_lookup
from bioharness.tools.genomics import _pick_chr


def _mock_client() -> httpx.AsyncClient:
    """An AsyncClient whose dbSNP/MyGene endpoints return canned payloads."""
    def handler(request: httpx.Request) -> httpx.Response:
        url = str(request.url)
        if "esummary.fcgi" in url and "db=snp" in url:
            # rs1217074595 -> LINC01270 on chr20
            return httpx.Response(200, json={"result": {"1217074595": {
                "genes": [{"name": "LINC01270"}], "chr": "20",
                "chrpos": "20:50298395"}}})
        if "mygene.info" in url:
            # NODAL -> protein-coding on chr10 (genomic_pos as list w/ patch)
            return httpx.Response(200, json={"hits": [{
                "symbol": "NODAL", "type_of_gene": "protein-coding",
                "genomic_pos": [{"chr": "10"}, {"chr": "HSCHR10_PATCH"}]}]})
        if "esearch.fcgi" in url:
            return httpx.Response(200, json={"esearchresult": {"idlist": []}})
        return httpx.Response(404)

    return httpx.AsyncClient(transport=httpx.MockTransport(handler))


def test_pick_chr_filters_patch_contigs():
    assert _pick_chr([{"chr": "HSCHR10_PATCH"}, {"chr": "10"}]) == "10"
    assert _pick_chr({"chr": "X"}) == "X"
    assert _pick_chr(None) == ""


@pytest.mark.asyncio
async def test_snp_lookup():
    async with _mock_client() as client:
        rec = await snp_lookup("rs1217074595", client=client)
    assert rec is not None
    assert rec.gene == "LINC01270"
    assert rec.chromosome == "chr20"


@pytest.mark.asyncio
async def test_gene_genomic_info_protein_coding():
    async with _mock_client() as client:
        rec = await gene_genomic_info("NODAL", client=client)
    assert rec is not None
    assert rec.is_protein_coding is True
    assert rec.protein_coding_answer == "TRUE"
    assert rec.chromosome == "chr10"


@pytest.mark.asyncio
async def test_precall_injects_snp_evidence():
    async with _mock_client() as client:
        ev = await precall_tools(
            "The name of the gene associated with SNP rs1217074595 is",
            "factoid", http_client=client)
    assert "snp_lookup(rs1217074595)" in ev
    assert "gene=LINC01270" in ev
    assert "chromosome=chr20" in ev


@pytest.mark.asyncio
async def test_precall_injects_protein_coding_evidence():
    async with _mock_client() as client:
        ev = await precall_tools(
            "Regarding if the gene codes a protein, NODAL is",
            "factoid", http_client=client)
    assert "protein_coding=TRUE" in ev
