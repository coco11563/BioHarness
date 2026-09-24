"""Genomics database lookup tool (NCBI dbSNP + MyGene.info).

Covers structured genomics facts that PubMed literature retrieval cannot answer
(e.g. the GeneTuring NCBI-database subtasks):

- SNP (rs ID) -> associated gene / chromosome   (NCBI dbSNP E-utilities esummary)
- gene symbol -> chromosome / protein-coding?    (MyGene.info)

APIs used:
- NCBI E-utilities esummary, db=snp
  (https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi)
- MyGene.info v3 query (https://mygene.info/v3/query)

Set ``NCBI_API_KEY`` to raise the NCBI rate limit (3 -> 10 req/s).

Example:
    resolver = GenomicsResolver()
    resolver.snp_lookup("rs1217074595")  # gene="LINC01270", chromosome="20"
    resolver.gene_info("NODAL")          # chromosome="10", is_protein_coding=True
"""

from __future__ import annotations

import json
import logging
import os
import re
import threading
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Optional

import httpx

logger = logging.getLogger(__name__)

_EUTILS = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
_MYGENE = "https://mygene.info/v3"
_BLAST_URL = "https://blast.ncbi.nlm.nih.gov/Blast.cgi"
# Persistent dbSNP/MyGene cache (pre-warmed once so parallel runs hit the cache
# instead of flooding NCBI past its rate limit).
_NCBI_CACHE_PATH = Path(__file__).resolve().parents[2] / ".cache/genomics/geneturing_ncbi.json"
# Canonical human chromosomes (exclude alt/patch contigs like HSCHR16_1_CTG1).
_CANON_CHR = {str(i) for i in range(1, 23)} | {"X", "Y", "MT", "M"}

# Scientific name -> GeneTuring organism vocabulary (multi-species alignment).
_ORGANISM_MAP = {
    "Homo sapiens": "human",
    "Mus musculus": "mouse",
    "Rattus norvegicus": "rat",
    "Danio rerio": "zebrafish",
    "Gallus gallus": "chicken",
    "Bos taurus": "cow",
    "Sus scrofa": "pig",
    "Caenorhabditis elegans": "worm",
    "Saccharomyces cerevisiae": "yeast",
    "Drosophila melanogaster": "fly",
    "Arabidopsis thaliana": "thale-cress",
}


@dataclass
class SNPInfo:
    """A dbSNP variant lookup result."""
    rsid: str
    gene: Optional[str] = None              # primary associated gene symbol
    genes: list[str] = field(default_factory=list)
    chromosome: Optional[str] = None        # canonical chromosome, e.g. "20"
    position: Optional[str] = None          # e.g. "20:50298395"


@dataclass
class GeneInfo:
    """A gene-symbol genomic lookup result."""
    symbol: str
    chromosome: Optional[str] = None        # canonical chromosome, e.g. "10"
    type_of_gene: Optional[str] = None      # e.g. "protein-coding", "ncRNA"
    is_protein_coding: Optional[bool] = None


class GenomicsResolver:
    """Thread-safe genomics lookups against NCBI dbSNP and MyGene.info.

    Results are cached in-process (GeneTuring repeats many rs IDs / symbols).
    All network errors fail soft to an empty record so the agent can fall back.
    """

    def __init__(self, timeout: float = 20.0) -> None:
        self._client = httpx.Client(
            timeout=timeout, headers={"User-Agent": "biomedical-rag/1.0"}
        )
        self._api_key = os.environ.get("NCBI_API_KEY") or None
        self._snp_cache: dict[str, SNPInfo] = {}
        self._gene_cache: dict[str, GeneInfo] = {}
        self._lock = threading.Lock()
        self._load_disk()

    def _load_disk(self) -> None:
        """Load the pre-warmed dbSNP/MyGene cache so parallel runs hit it."""
        try:
            d = json.loads(_NCBI_CACHE_PATH.read_text())
            for sid, v in (d.get("snp") or {}).items():
                self._snp_cache[sid] = SNPInfo(**v)
            for k, v in (d.get("gene") or {}).items():
                self._gene_cache[k] = GeneInfo(**v)
        except Exception:
            pass

    def save_disk(self) -> None:
        """Persist the in-process caches to the shared disk cache."""
        _NCBI_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
        with self._lock:
            d = {"snp": {k: asdict(v) for k, v in self._snp_cache.items()},
                 "gene": {k: asdict(v) for k, v in self._gene_cache.items()}}
        _NCBI_CACHE_PATH.write_text(json.dumps(d))

    def close(self) -> None:
        try:
            self._client.close()
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _norm_rsid(rsid: str) -> str:
        """Strip the ``rs`` prefix to the numeric dbSNP id."""
        m = re.search(r"rs(\d+)", str(rsid), re.IGNORECASE)
        return m.group(1) if m else re.sub(r"\D", "", str(rsid))

    @staticmethod
    def _canon_chr(chrom: Any) -> Optional[str]:
        if not chrom:
            return None
        c = str(chrom).replace("chr", "").replace("CHR", "").strip().upper()
        return c or None

    # ------------------------------------------------------------------
    # Lookups
    # ------------------------------------------------------------------

    def snp_lookup(self, rsid: str) -> SNPInfo:
        """Resolve a dbSNP rs ID to its associated gene and chromosome."""
        sid = self._norm_rsid(rsid)
        with self._lock:
            if sid in self._snp_cache:
                return self._snp_cache[sid]
        info = SNPInfo(rsid=f"rs{sid}")
        if sid:
            params = {"db": "snp", "id": sid, "retmode": "json"}
            if self._api_key:
                params["api_key"] = self._api_key
            try:
                resp = self._client.get(f"{_EUTILS}/esummary.fcgi", params=params)
                if resp.status_code == 200:
                    rec = (resp.json().get("result") or {}).get(sid) or {}
                    genes = [g.get("name") for g in (rec.get("genes") or []) if g.get("name")]
                    info.genes = genes
                    info.gene = genes[0] if genes else None
                    info.chromosome = self._canon_chr(rec.get("chr"))
                    info.position = rec.get("chrpos")
            except Exception as exc:  # fail soft
                logger.warning("snp_lookup(%s) failed: %s", rsid, exc)
        with self._lock:
            self._snp_cache[sid] = info
        return info

    def gene_info(self, symbol: str) -> GeneInfo:
        """Resolve a gene symbol to its chromosome and protein-coding status."""
        sym = str(symbol).strip()
        key = sym.upper()
        with self._lock:
            if key in self._gene_cache:
                return self._gene_cache[key]
        info = GeneInfo(symbol=sym)
        if sym:
            try:
                resp = self._client.get(
                    f"{_MYGENE}/query",
                    params={"q": sym, "species": "human",
                            "fields": "symbol,type_of_gene,genomic_pos", "size": 1},
                )
                if resp.status_code == 200:
                    hits = resp.json().get("hits") or []
                    if hits:
                        h = hits[0]
                        tog = h.get("type_of_gene")
                        info.type_of_gene = tog
                        info.is_protein_coding = (tog == "protein-coding") if tog else None
                        info.chromosome = self._extract_chr(h.get("genomic_pos"))
            except Exception as exc:  # fail soft
                logger.warning("gene_info(%s) failed: %s", symbol, exc)
        with self._lock:
            self._gene_cache[key] = info
        return info

    def _extract_chr(self, genomic_pos: Any) -> Optional[str]:
        """Pick the canonical chromosome from MyGene's genomic_pos (list/dict)."""
        if isinstance(genomic_pos, dict):
            return self._canon_chr(genomic_pos.get("chr"))
        if isinstance(genomic_pos, list):
            cands = [self._canon_chr(x.get("chr")) for x in genomic_pos if isinstance(x, dict)]
            canon = next((c for c in cands if c in _CANON_CHR), None)
            return canon or (cands[0] if cands else None)
        return None

    # ------------------------------------------------------------------
    # BLAST (DNA sequence alignment) — NCBI BLAST URL API
    # ------------------------------------------------------------------
    # Slow (submit -> poll -> fetch, ~30-90s each, rate-limited). Intended for
    # offline batch caching, not live in-benchmark calls. Covers the GeneTuring
    # DNA-alignment subtasks: sequence -> genome coordinates / source organism.

    def blast_submit(self, sequence: str, *, database: str, megablast: bool) -> Optional[str]:
        """Submit a blastn query; return the Request ID (RID) or None."""
        data = {
            "CMD": "Put", "PROGRAM": "blastn", "DATABASE": database,
            "QUERY": sequence,
        }
        if megablast:
            data["MEGABLAST"] = "on"
        try:
            r = self._client.post(_BLAST_URL, data=data, timeout=30)
            m = re.search(r"RID = (\w+)", r.text)
            return m.group(1) if m else None
        except Exception as exc:
            logger.warning("blast_submit failed: %s", exc)
            return None

    def blast_status(self, rid: str) -> str:
        """Return 'READY' | 'WAITING' | 'UNKNOWN'."""
        try:
            r = self._client.get(_BLAST_URL, params={
                "CMD": "Get", "FORMAT_OBJECT": "SearchInfo", "RID": rid}, timeout=30)
            m = re.search(r"Status=(\w+)", r.text)
            return m.group(1) if m else "UNKNOWN"
        except Exception:
            return "UNKNOWN"

    def blast_result_text(self, rid: str) -> str:
        """Fetch the plain-text BLAST report for a ready RID."""
        try:
            r = self._client.get(_BLAST_URL, params={
                "CMD": "Get", "FORMAT_TYPE": "Text", "RID": rid,
                "ALIGNMENTS": 1, "DESCRIPTIONS": 1}, timeout=45)
            return r.text
        except Exception as exc:
            logger.warning("blast_result_text(%s) failed: %s", rid, exc)
            return ""

    @staticmethod
    def parse_genome_hit(text: str) -> Optional[str]:
        """Top human-genome hit -> 'chrN:start-end' (GeneTuring format).

        Only the FIRST HSP of the top hit is used — a subject can have several
        HSPs far apart, and spanning across them yields a wrong (huge) end.
        """
        acc = re.search(r"^>(NC_0+(\d+)\.\d+)\s", text, re.MULTILINE)
        if not acc:
            return None
        num = int(acc.group(2))
        chrom = {23: "X", 24: "Y", 12920: "M"}.get(num, str(num) if 1 <= num <= 22 else None)
        if chrom is None:
            return None
        # Restrict to the first HSP: keep text up to the 2nd " Score =" marker.
        block = text[acc.end():]
        scores = [m.start() for m in re.finditer(r"\n Score =", block)]
        if len(scores) >= 2:
            block = block[:scores[1]]
        sbjcts = re.findall(r"^Sbjct\s+(\d+)\s+\S+\s+(\d+)", block, re.MULTILINE)
        if not sbjcts:
            return None
        start = int(sbjcts[0][0])
        end = int(sbjcts[-1][1])
        lo, hi = min(start, end), max(start, end)
        return f"chr{chrom}:{lo}-{hi}"

    @staticmethod
    def parse_organism_hit(text: str) -> Optional[str]:
        """Top hit's source organism -> GeneTuring organism vocabulary."""
        for sci, common in _ORGANISM_MAP.items():
            if sci.lower() in text.lower():
                return common
        return None
