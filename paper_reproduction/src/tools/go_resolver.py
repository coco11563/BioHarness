"""GO (Gene Ontology) DAG resolver using goatools.

This module provides tools to resolve GO terms and traverse the GO DAG.

APIs used:
- Local OBO file parsing (go-basic.obo)
- QuickGO API for online term lookup (optional fallback)

Example:
    resolver = GOResolver()
    await resolver.initialize()
    term = resolver.resolve("GO:0004672")  # protein kinase activity
    ancestors = resolver.get_ancestors("GO:0004672")
"""

import asyncio
import logging
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, List, Set, Dict
import aiohttp

logger = logging.getLogger(__name__)

# Default OBO file location
DEFAULT_OBO_PATH = Path(__file__).parent.parent.parent / "data" / "go-basic.obo"
OBO_URL = "http://purl.obolibrary.org/obo/go/go-basic.obo"


@dataclass
class GOTerm:
    """A GO term with its metadata."""
    id: str
    name: str
    namespace: str  # molecular_function, biological_process, cellular_component
    definition: str = ""
    alt_ids: List[str] = field(default_factory=list)
    is_obsolete: bool = False
    parents: Set[str] = field(default_factory=set)
    children: Set[str] = field(default_factory=set)


class GOResolver:
    """Resolve GO terms and traverse GO DAG.

    Uses goatools for OBO parsing and DAG operations.
    Falls back to QuickGO API if local resolution fails.
    """

    def __init__(self, obo_path: Optional[Path] = None, timeout: float = 10.0):
        """Initialize resolver.

        Args:
            obo_path: Path to go-basic.obo file. If None, downloads automatically.
            timeout: Request timeout for API calls in seconds
        """
        self.obo_path = obo_path or DEFAULT_OBO_PATH
        self.timeout = timeout
        self._godag = None
        self._name_to_id: Dict[str, str] = {}
        self._session: Optional[aiohttp.ClientSession] = None
        self._initialized = False

    async def _get_session(self) -> aiohttp.ClientSession:
        """Get or create aiohttp session."""
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=self.timeout)
            )
        return self._session

    async def _download_obo(self) -> bool:
        """Download go-basic.obo if not present."""
        if self.obo_path.exists():
            return True

        logger.info(f"Downloading GO OBO file to {self.obo_path}...")
        self.obo_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            session = await self._get_session()
            async with session.get(OBO_URL) as resp:
                if resp.status == 200:
                    content = await resp.read()
                    with open(self.obo_path, "wb") as f:
                        f.write(content)
                    logger.info(f"Downloaded {len(content)} bytes")
                    return True
                else:
                    logger.error(f"Failed to download OBO: {resp.status}")
                    return False
        except Exception as e:
            logger.error(f"Error downloading OBO: {e}")
            return False

    async def initialize(self) -> bool:
        """Initialize the GO DAG from OBO file.

        Downloads go-basic.obo if not present.
        """
        if self._initialized:
            return True

        # Ensure OBO file exists
        if not await self._download_obo():
            return False

        try:
            from goatools.obo_parser import GODag

            logger.info(f"Loading GO DAG from {self.obo_path}...")
            self._godag = GODag(str(self.obo_path), optional_attrs=["def"])

            # Build name-to-ID lookup
            for go_id, term in self._godag.items():
                self._name_to_id[term.name.lower()] = go_id

            logger.info(f"Loaded {len(self._godag)} GO terms")
            self._initialized = True
            return True

        except Exception as e:
            logger.error(f"Failed to load GO DAG: {e}")
            return False

    async def close(self):
        """Close aiohttp session."""
        if self._session and not self._session.closed:
            await self._session.close()

    def resolve(self, go_id_or_name: str) -> Optional[GOTerm]:
        """Resolve GO term by ID or name.

        Args:
            go_id_or_name: GO ID (e.g., "GO:0004672") or term name

        Returns:
            GOTerm object or None if not found
        """
        if not self._initialized or not self._godag:
            return None

        query = go_id_or_name.strip()

        # Try direct ID lookup
        if query.upper().startswith("GO:"):
            go_id = query.upper()
            if go_id in self._godag:
                return self._to_goterm(go_id)
            return None

        # Try name lookup
        name_lower = query.lower()
        if name_lower in self._name_to_id:
            go_id = self._name_to_id[name_lower]
            return self._to_goterm(go_id)

        # Try partial name match
        for name, go_id in self._name_to_id.items():
            if name_lower in name or name in name_lower:
                return self._to_goterm(go_id)

        return None

    def _to_goterm(self, go_id: str) -> GOTerm:
        """Convert goatools term to GOTerm dataclass."""
        term = self._godag[go_id]
        return GOTerm(
            id=go_id,
            name=term.name,
            namespace=term.namespace,
            definition=getattr(term, "defn", ""),
            alt_ids=list(term.alt_ids) if term.alt_ids else [],
            is_obsolete=term.is_obsolete,
            parents=set(p.id for p in term.parents) if hasattr(term, "parents") else set(),
            children=set(c.id for c in term.children) if hasattr(term, "children") else set(),
        )

    def get_ancestors(self, go_id_or_name: str, include_self: bool = False) -> Set[str]:
        """Get all ancestor GO terms.

        Args:
            go_id_or_name: GO ID or term name
            include_self: Include the term itself in result

        Returns:
            Set of ancestor GO IDs
        """
        term = self.resolve(go_id_or_name)
        if not term:
            return set()

        goatools_term = self._godag[term.id]
        ancestors = goatools_term.get_all_parents()

        if include_self:
            ancestors = ancestors | {term.id}

        return ancestors

    def get_descendants(self, go_id_or_name: str, include_self: bool = False) -> Set[str]:
        """Get all descendant GO terms.

        Args:
            go_id_or_name: GO ID or term name
            include_self: Include the term itself in result

        Returns:
            Set of descendant GO IDs
        """
        term = self.resolve(go_id_or_name)
        if not term:
            return set()

        goatools_term = self._godag[term.id]
        descendants = goatools_term.get_all_children()

        if include_self:
            descendants = descendants | {term.id}

        return descendants

    def get_related(self, go_id_or_name: str) -> Dict[str, Set[str]]:
        """Get related GO terms (parents, children, siblings).

        Args:
            go_id_or_name: GO ID or term name

        Returns:
            Dict with keys: ancestors, descendants, parents, children, siblings
        """
        term = self.resolve(go_id_or_name)
        if not term:
            return {}

        goatools_term = self._godag[term.id]

        parents = set(p.id for p in goatools_term.parents) if goatools_term.parents else set()
        children = set(c.id for c in goatools_term.children) if goatools_term.children else set()

        # Get siblings (same parents)
        siblings = set()
        for parent in goatools_term.parents:
            for sibling in parent.children:
                if sibling.id != term.id:
                    siblings.add(sibling.id)

        return {
            "parents": parents,
            "children": children,
            "siblings": siblings,
            "ancestors": goatools_term.get_all_parents(),
            "descendants": goatools_term.get_all_children(),
        }

    def get_namespace_terms(self, namespace: str) -> List[str]:
        """Get all terms in a namespace.

        Args:
            namespace: One of molecular_function, biological_process, cellular_component

        Returns:
            List of GO IDs
        """
        if not self._initialized or not self._godag:
            return []

        return [
            go_id for go_id, term in self._godag.items()
            if term.namespace == namespace and not term.is_obsolete
        ]

    async def search_quickgo(self, query: str, limit: int = 10) -> List[GOTerm]:
        """Search GO terms via QuickGO API.

        Args:
            query: Search query
            limit: Maximum results

        Returns:
            List of GOTerm objects
        """
        session = await self._get_session()

        try:
            url = "https://www.ebi.ac.uk/QuickGO/services/ontology/go/search"
            params = {"query": query, "limit": limit}
            headers = {"Accept": "application/json"}

            async with session.get(url, params=params, headers=headers) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    results = []
                    for item in data.get("results", []):
                        results.append(GOTerm(
                            id=item.get("id", ""),
                            name=item.get("name", ""),
                            namespace=item.get("aspect", "").lower().replace(" ", "_"),
                            definition=item.get("definition", {}).get("text", ""),
                        ))
                    return results
                else:
                    logger.warning(f"QuickGO search failed: {resp.status}")
                    return []
        except Exception as e:
            logger.warning(f"QuickGO search error: {e}")
            return []


# Synchronous convenience function
def resolve_go_term(go_id_or_name: str) -> Optional[GOTerm]:
    """Synchronous wrapper for GO term resolution.

    Args:
        go_id_or_name: GO ID or term name

    Returns:
        GOTerm object or None
    """
    async def _resolve():
        resolver = GOResolver()
        try:
            await resolver.initialize()
            return resolver.resolve(go_id_or_name)
        finally:
            await resolver.close()

    return asyncio.run(_resolve())
