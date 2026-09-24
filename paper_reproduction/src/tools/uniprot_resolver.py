"""UniProt protein lookup tool.

This module provides tools to look up protein information from UniProt.

APIs used:
- UniProt REST API (https://rest.uniprot.org)

Example:
    resolver = UniProtResolver()
    hits = await resolver.search_proteins("BRCA1")
    record = await resolver.get_protein("P38398")
"""

import logging
import threading
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any

import aiohttp

logger = logging.getLogger(__name__)

BASE_URL = "https://rest.uniprot.org"


@dataclass
class ProteinHit:
    """A protein search result."""
    accession: str
    protein_name: str = ""
    gene_names: List[str] = field(default_factory=list)
    organism_id: Optional[int] = None
    organism_name: str = ""


@dataclass
class ProteinFunction:
    """Protein function annotation."""
    text: str = ""
    evidences: List[str] = field(default_factory=list)  # ECO codes


@dataclass
class DiseaseAssociation:
    """Disease association for a protein."""
    name: str = ""
    acronym: str = ""
    description: str = ""
    xrefs: List[Dict[str, str]] = field(default_factory=list)  # {db, id}


@dataclass
class GOAnnotation:
    """Gene Ontology annotation."""
    go_id: str = ""  # e.g., GO:0006915
    term: str = ""   # e.g., "apoptotic process"
    aspect: str = "" # C (cellular component), F (function), P (process)
    evidence: str = ""  # e.g., ECO:0000314


@dataclass
class ProteinRecord:
    """Full protein record from UniProt."""
    accession: str
    protein_name: str = ""
    gene_names: List[str] = field(default_factory=list)
    organism_id: Optional[int] = None
    organism_name: str = ""
    function: Optional[ProteinFunction] = None
    diseases: List[DiseaseAssociation] = field(default_factory=list)
    go_annotations: List[GOAnnotation] = field(default_factory=list)


class UniProtResolver:
    """UniProt protein lookup client.

    Thread-safe singleton pattern. Uses aiohttp for async API access.
    All operations are fail-fast (no retries, exceptions propagate).
    """

    _instance = None
    _lock = threading.Lock()

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(self, timeout: float = 15.0, default_limit: int = 5):
        if self._initialized:
            return
        self.timeout = timeout
        self.default_limit = default_limit
        self._session: Optional[aiohttp.ClientSession] = None
        self._initialized = True

    async def _get_session(self) -> aiohttp.ClientSession:
        """Get or create aiohttp session."""
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=self.timeout)
            )
        return self._session

    async def close(self):
        """Close aiohttp session."""
        if self._session and not self._session.closed:
            await self._session.close()

    async def _get_json(self, path: str, params: Optional[dict] = None) -> dict:
        """GET request returning JSON. Fail-fast on errors."""
        session = await self._get_session()
        url = f"{BASE_URL}{path}"
        async with session.get(url, params=params or {}) as resp:
            resp.raise_for_status()  # fail-fast
            return await resp.json()

    @staticmethod
    def _extract_protein_name(data: dict) -> str:
        """Extract protein name from UniProt JSON."""
        desc = data.get("proteinDescription", {})
        # Try recommended name first
        rec_name = desc.get("recommendedName", {})
        if rec_name:
            full_name = rec_name.get("fullName", {})
            if full_name:
                return full_name.get("value", "")
        # Try submission names
        sub_names = desc.get("submissionNames", [])
        if sub_names:
            first = sub_names[0].get("fullName", {})
            if first:
                return first.get("value", "")
        return ""

    @staticmethod
    def _extract_gene_names(data: dict) -> List[str]:
        """Extract gene names from UniProt JSON."""
        genes = data.get("genes", [])
        names = []
        for gene in genes:
            # Primary gene name
            gene_name = gene.get("geneName", {})
            if gene_name:
                val = gene_name.get("value")
                if val:
                    names.append(val)
            # Synonyms
            for syn in gene.get("synonyms", []):
                val = syn.get("value")
                if val and val not in names:
                    names.append(val)
        return names

    @staticmethod
    def _extract_function(data: dict) -> Optional[ProteinFunction]:
        """Extract function annotation from UniProt JSON."""
        comments = data.get("comments", [])
        for comment in comments:
            if comment.get("commentType") == "FUNCTION":
                texts = comment.get("texts", [])
                if texts:
                    text_parts = []
                    evidences = []
                    for t in texts:
                        val = t.get("value", "")
                        if val:
                            text_parts.append(val)
                        for ev in t.get("evidences", []):
                            code = ev.get("evidenceCode", "")
                            if code:
                                evidences.append(code)
                    return ProteinFunction(
                        text=" ".join(text_parts),
                        evidences=list(set(evidences)),
                    )
        return None

    @staticmethod
    def _extract_diseases(data: dict) -> List[DiseaseAssociation]:
        """Extract disease associations from UniProt JSON."""
        comments = data.get("comments", [])
        diseases = []
        for comment in comments:
            if comment.get("commentType") == "DISEASE":
                disease = comment.get("disease", {})
                if disease:
                    xrefs = []
                    for xref in disease.get("diseaseCrossReferences", []):
                        xrefs.append({
                            "db": xref.get("database", ""),
                            "id": xref.get("id", ""),
                        })
                    diseases.append(DiseaseAssociation(
                        name=disease.get("diseaseId", ""),
                        acronym=disease.get("acronym", ""),
                        description=disease.get("description", ""),
                        xrefs=xrefs,
                    ))
        return diseases

    @staticmethod
    def _extract_go_annotations(data: dict) -> List[GOAnnotation]:
        """Extract GO annotations from UniProt JSON."""
        xrefs = data.get("uniProtKBCrossReferences", [])
        go_annots = []
        for xref in xrefs:
            if xref.get("database") == "GO":
                go_id = xref.get("id", "")
                properties = xref.get("properties", [])

                term = ""
                evidence = ""
                aspect = ""

                for prop in properties:
                    key = prop.get("key", "")
                    val = prop.get("value", "")
                    if key == "GoTerm":
                        # Format: "C:membrane" or "F:kinase activity"
                        if ":" in val:
                            aspect_code, term_text = val.split(":", 1)
                            aspect = aspect_code
                            term = term_text
                        else:
                            term = val
                    elif key == "GoEvidenceType":
                        evidence = val

                go_annots.append(GOAnnotation(
                    go_id=go_id,
                    term=term,
                    aspect=aspect,
                    evidence=evidence,
                ))
        return go_annots

    def _parse_hit(self, item: dict) -> ProteinHit:
        """Parse search result item to ProteinHit."""
        organism = item.get("organism", {})
        return ProteinHit(
            accession=item.get("primaryAccession", ""),
            protein_name=self._extract_protein_name(item),
            gene_names=self._extract_gene_names(item),
            organism_id=organism.get("taxonId"),
            organism_name=organism.get("scientificName", ""),
        )

    def _parse_record(self, data: dict) -> ProteinRecord:
        """Parse full protein record."""
        organism = data.get("organism", {})
        return ProteinRecord(
            accession=data.get("primaryAccession", ""),
            protein_name=self._extract_protein_name(data),
            gene_names=self._extract_gene_names(data),
            organism_id=organism.get("taxonId"),
            organism_name=organism.get("scientificName", ""),
            function=self._extract_function(data),
            diseases=self._extract_diseases(data),
            go_annotations=self._extract_go_annotations(data),
        )

    async def search_proteins(
        self,
        query: str,
        organism_id: int = 9606,  # Human
        limit: Optional[int] = None,
    ) -> List[ProteinHit]:
        """Search proteins by gene name, protein name, or accession.

        Args:
            query: Gene name, protein name, or UniProt accession
            organism_id: Organism taxon ID (default 9606 = Homo sapiens)
            limit: Maximum results (default 5)

        Returns:
            List of ProteinHit objects

        Raises:
            ValueError: If query is empty
            aiohttp.ClientError: On API errors (fail-fast)
        """
        if not query or not query.strip():
            raise ValueError("query must be non-empty")

        query = query.strip()
        limit = limit if limit and limit > 0 else self.default_limit

        # Build UniProt query: search gene or accession
        # Use simpler query - complex OR queries can cause 400 errors
        search_query = f"(gene:{query}) AND organism_id:{organism_id}"
        params = {
            "query": search_query,
            "format": "json",
            "size": limit,
        }

        data = await self._get_json("/uniprotkb/search", params=params)
        results = data.get("results", [])
        return [self._parse_hit(item) for item in results]

    async def get_protein(self, accession: str) -> Optional[ProteinRecord]:
        """Get full protein record by UniProt accession.

        Args:
            accession: UniProt accession (e.g., "P38398")

        Returns:
            ProteinRecord or None if not found

        Raises:
            ValueError: If accession is empty
            aiohttp.ClientError: On API errors (fail-fast)
        """
        if not accession or not accession.strip():
            raise ValueError("accession must be non-empty")

        accession = accession.strip()

        try:
            data = await self._get_json(f"/uniprotkb/{accession}", params={"format": "json"})
            return self._parse_record(data)
        except aiohttp.ClientResponseError as e:
            if e.status == 404:
                return None
            raise

    async def get_protein_function(self, accession: str) -> Optional[ProteinFunction]:
        """Get protein function annotation.

        Args:
            accession: UniProt accession

        Returns:
            ProteinFunction or None
        """
        record = await self.get_protein(accession)
        return record.function if record else None

    async def get_protein_diseases(self, accession: str) -> List[DiseaseAssociation]:
        """Get protein disease associations.

        Args:
            accession: UniProt accession

        Returns:
            List of DiseaseAssociation objects
        """
        record = await self.get_protein(accession)
        return record.diseases if record else []

    async def get_protein_go(self, accession: str) -> List[GOAnnotation]:
        """Get protein GO annotations.

        Args:
            accession: UniProt accession

        Returns:
            List of GOAnnotation objects
        """
        record = await self.get_protein(accession)
        return record.go_annotations if record else []


# Convenience async function
async def search_protein(query: str, limit: int = 5) -> List[ProteinHit]:
    """Convenience function to search proteins."""
    resolver = UniProtResolver()
    return await resolver.search_proteins(query, limit=limit)
