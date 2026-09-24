"""ChEMBL drug lookup tool using chembl_webresource_client.

This module provides tools to look up drug information from ChEMBL.

APIs used:
- chembl_webresource_client (local package, sync)
- ChEMBL REST API (https://www.ebi.ac.uk/chembl/api/data/)

Example:
    resolver = ChEMBLResolver()
    hits = resolver.lookup_drug("aspirin")
    mechanisms = resolver.get_mechanisms("CHEMBL25")  # aspirin
"""

import re
import threading
import logging
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any

logger = logging.getLogger(__name__)


@dataclass
class DrugHit:
    """A drug/molecule from ChEMBL."""
    chembl_id: str
    pref_name: Optional[str] = None
    molecule_type: Optional[str] = None
    max_phase: Optional[int] = None  # 4 = approved
    first_approval: Optional[int] = None  # year
    oral: Optional[bool] = None
    synonyms: List[str] = field(default_factory=list)


@dataclass
class MechanismOfAction:
    """Mechanism of action for a drug."""
    mechanism: Optional[str] = None
    action_type: Optional[str] = None  # e.g., "INHIBITOR", "AGONIST"
    target_chembl_id: Optional[str] = None
    target_name: Optional[str] = None
    binding_site_name: Optional[str] = None
    refs: List[Dict[str, str]] = field(default_factory=list)


@dataclass
class TargetInfo:
    """Target information from ChEMBL."""
    target_chembl_id: str
    pref_name: Optional[str] = None
    target_type: Optional[str] = None  # e.g., "SINGLE PROTEIN"
    organism: Optional[str] = None
    components: List[Dict[str, str]] = field(default_factory=list)


@dataclass
class Indication:
    """Drug indication (approved use)."""
    mesh_id: Optional[str] = None
    mesh_heading: Optional[str] = None
    efo_id: Optional[str] = None
    efo_term: Optional[str] = None
    max_phase_for_ind: Optional[int] = None


class ChEMBLResolver:
    """ChEMBL drug lookup client.

    Thread-safe singleton pattern. Uses chembl_webresource_client for API access.
    All operations are synchronous (chembl_webresource_client is sync).
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

    def __init__(self, limit_default: int = 5):
        if self._initialized:
            return
        self.limit_default = limit_default
        self._client = None
        self._molecule = None
        self._mechanism = None
        self._target = None
        self._drug_indication = None
        self._initialized = True

    def _get_client(self):
        """Lazy-load ChEMBL client. Fail-fast on import errors. Thread-safe."""
        if self._client is None:
            with self._lock:
                if self._client is None:
                    from chembl_webresource_client.new_client import new_client
                    self._client = new_client
                    self._molecule = self._client.molecule
                    self._mechanism = self._client.mechanism
                    self._target = self._client.target
                    self._drug_indication = self._client.drug_indication
        return self._client

    @staticmethod
    def _is_chembl_id(query: str) -> bool:
        """Check if query is a ChEMBL ID (e.g., CHEMBL25)."""
        return bool(re.fullmatch(r"CHEMBL\d+", query.strip().upper()))

    def _drug_from_record(self, rec: dict) -> DrugHit:
        """Convert ChEMBL API record to DrugHit."""
        synonyms = []
        for syn in rec.get("molecule_synonyms", []) or []:
            val = syn.get("molecule_synonym") or syn.get("synonym")
            if val:
                synonyms.append(val)
        return DrugHit(
            chembl_id=rec.get("molecule_chembl_id") or "",
            pref_name=rec.get("pref_name"),
            molecule_type=rec.get("molecule_type"),
            max_phase=rec.get("max_phase"),
            first_approval=rec.get("first_approval"),
            oral=rec.get("oral"),
            synonyms=sorted(set(synonyms)),
        )

    def get_drug(self, chembl_id: str) -> Optional[DrugHit]:
        """Get drug by ChEMBL ID.

        Args:
            chembl_id: ChEMBL ID (e.g., "CHEMBL25")

        Returns:
            DrugHit or None if not found
        """
        self._get_client()
        chembl_id = chembl_id.strip().upper()
        if not self._is_chembl_id(chembl_id):
            raise ValueError(f"Invalid ChEMBL ID format: {chembl_id}")

        try:
            rec = self._molecule.get(chembl_id)
            if not rec:
                return None
            return self._drug_from_record(rec)
        except Exception as e:
            logger.warning(f"ChEMBL get_drug failed for {chembl_id}: {e}")
            raise

    def lookup_drug(self, query: str, limit: Optional[int] = None) -> List[DrugHit]:
        """Look up drug by name, ChEMBL ID, or synonym.

        Search strategy:
        1. If query is ChEMBL ID, direct lookup
        2. Exact preferred name match
        3. Exact synonym match
        4. Partial name match

        Args:
            query: Drug name, ChEMBL ID, or synonym
            limit: Maximum results (default 5, must be > 0)

        Returns:
            List of DrugHit objects

        Raises:
            ValueError: If query is empty or limit <= 0
        """
        self._get_client()

        if not query or not query.strip():
            raise ValueError("query must be non-empty")

        query = query.strip()
        limit = limit if limit and limit > 0 else self.limit_default
        hits = []
        seen = set()

        # Direct ChEMBL ID lookup
        if self._is_chembl_id(query):
            hit = self.get_drug(query)
            return [hit] if hit else []

        # 1) Exact preferred name (fail-fast: let exceptions propagate)
        for rec in self._molecule.filter(pref_name__iexact=query):
            hit = self._drug_from_record(rec)
            if hit.chembl_id and hit.chembl_id not in seen:
                hits.append(hit)
                seen.add(hit.chembl_id)
            if len(hits) >= limit:
                return hits

        # 2) Exact synonym match
        for rec in self._molecule.filter(molecule_synonyms__molecule_synonym__iexact=query):
            hit = self._drug_from_record(rec)
            if hit.chembl_id and hit.chembl_id not in seen:
                hits.append(hit)
                seen.add(hit.chembl_id)
            if len(hits) >= limit:
                return hits

        # 3) Partial name match (fallback)
        if len(hits) < limit:
            for rec in self._molecule.filter(pref_name__icontains=query):
                hit = self._drug_from_record(rec)
                if hit.chembl_id and hit.chembl_id not in seen:
                    hits.append(hit)
                    seen.add(hit.chembl_id)
                if len(hits) >= limit:
                    break

        return hits

    def get_mechanisms(self, chembl_id: str) -> List[MechanismOfAction]:
        """Get mechanisms of action for a drug.

        Args:
            chembl_id: ChEMBL ID of the drug

        Returns:
            List of MechanismOfAction objects
        """
        self._get_client()
        chembl_id = chembl_id.strip().upper()
        if not self._is_chembl_id(chembl_id):
            raise ValueError(f"Invalid ChEMBL ID format: {chembl_id}")

        mechs = []
        try:
            for rec in self._mechanism.filter(molecule_chembl_id=chembl_id):
                refs = []
                for ref in rec.get("mechanism_refs", []) or []:
                    refs.append({
                        "ref_type": ref.get("ref_type"),
                        "ref_id": ref.get("ref_id"),
                        "ref_url": ref.get("ref_url"),
                    })
                mechs.append(MechanismOfAction(
                    mechanism=rec.get("mechanism_of_action"),
                    action_type=rec.get("action_type"),
                    target_chembl_id=rec.get("target_chembl_id"),
                    target_name=rec.get("target_name"),
                    binding_site_name=rec.get("binding_site_name"),
                    refs=refs,
                ))
        except Exception as e:
            logger.warning(f"ChEMBL get_mechanisms failed for {chembl_id}: {e}")
            raise

        return mechs

    def get_target(self, target_chembl_id: str) -> Optional[TargetInfo]:
        """Get target information.

        Args:
            target_chembl_id: ChEMBL target ID (e.g., "CHEMBL220")

        Returns:
            TargetInfo or None if not found
        """
        self._get_client()
        target_chembl_id = target_chembl_id.strip().upper()

        try:
            rec = self._target.get(target_chembl_id)
            if not rec:
                return None

            components = []
            for comp in rec.get("target_components", []) or []:
                components.append({
                    "accession": comp.get("accession"),
                    "description": comp.get("component_description"),
                    "synonyms": list(comp.get("component_synonyms", []) or []),
                })

            return TargetInfo(
                target_chembl_id=rec.get("target_chembl_id") or target_chembl_id,
                pref_name=rec.get("pref_name"),
                target_type=rec.get("target_type"),
                organism=rec.get("organism"),
                components=components,
            )
        except Exception as e:
            logger.warning(f"ChEMBL get_target failed for {target_chembl_id}: {e}")
            raise

    def get_indications(self, chembl_id: str) -> List[Indication]:
        """Get drug indications (approved uses).

        Args:
            chembl_id: ChEMBL ID of the drug

        Returns:
            List of Indication objects
        """
        self._get_client()
        chembl_id = chembl_id.strip().upper()
        if not self._is_chembl_id(chembl_id):
            raise ValueError(f"Invalid ChEMBL ID format: {chembl_id}")

        indications = []
        try:
            for rec in self._drug_indication.filter(molecule_chembl_id=chembl_id):
                indications.append(Indication(
                    mesh_id=rec.get("mesh_id"),
                    mesh_heading=rec.get("mesh_heading"),
                    efo_id=rec.get("efo_id"),
                    efo_term=rec.get("efo_term"),
                    max_phase_for_ind=rec.get("max_phase_for_ind"),
                ))
        except Exception as e:
            logger.warning(f"ChEMBL get_indications failed for {chembl_id}: {e}")
            raise

        return indications


# Convenience function
def lookup_drug(query: str, limit: int = 5) -> List[DrugHit]:
    """Convenience function to look up drugs."""
    resolver = ChEMBLResolver()
    return resolver.lookup_drug(query, limit=limit)
