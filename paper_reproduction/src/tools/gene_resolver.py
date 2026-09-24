"""Gene symbol resolver using HGNC REST and MyGene.info APIs.

This module provides tools to resolve gene names, aliases, and previous symbols
to official HGNC gene symbols.

APIs used (both free, no API key required):
- HGNC REST: https://www.genenames.org/help/rest-web-services/
- MyGene.info: https://mygene.info/

Example:
    resolver = GeneResolver()
    symbol = await resolver.resolve("C20orf195")  # Returns "FNDC11"
"""

import asyncio
import logging
import re
from dataclasses import dataclass
from typing import Optional, List
import aiohttp

logger = logging.getLogger(__name__)


@dataclass
class GeneHit:
    """A gene search hit."""
    symbol: str
    name: str = ""
    hgnc_id: str = ""
    aliases: List[str] = None
    location: str = ""  # Chromosomal location (e.g., "8q24.3")
    locus_group: str = ""  # e.g., "protein-coding gene"
    score: float = 1.0
    source: str = ""

    def __post_init__(self):
        if self.aliases is None:
            self.aliases = []


class GeneResolver:
    """Resolve gene names to official HGNC symbols.

    Uses a multi-tier strategy:
    1. HGNC REST API (highest precision)
    2. MyGene.info API (good coverage)
    3. Local cache for common mappings
    """

    # Common ORF to gene symbol mappings (manually curated)
    ORF_MAPPINGS = {
        "C20orf195": "FNDC11",
        "CXorf40B": "EOLA2",
        "LMP10": "PSMB10",
        "IMD20": "FCGR3A",
        # Hyphenated variants (common in literature)
        "DVL-1": "DVL1",
        "DVL-2": "DVL2",
        "DVL-3": "DVL3",
        "WNT-1": "WNT1",
        "WNT-3A": "WNT3A",
        "HER-2": "ERBB2",
        "HER2": "ERBB2",
        "C-MYC": "MYC",
        "N-MYC": "MYCN",
        "L-MYC": "MYCL",
        "BCL-2": "BCL2",
        "P-53": "TP53",
        # Add more as needed
    }

    def __init__(self, timeout: float = 10.0):
        """Initialize resolver.

        Args:
            timeout: Request timeout in seconds
        """
        self.timeout = timeout
        self._session: Optional[aiohttp.ClientSession] = None
        # Normalize ORF_MAPPINGS keys to uppercase for case-insensitive lookup
        self._orf_map = {k.upper(): v for k, v in self.ORF_MAPPINGS.items()}

    async def _get_session(self) -> aiohttp.ClientSession:
        """Get or create aiohttp session."""
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=self.timeout)
            )
        return self._session

    async def close(self):
        """Close the aiohttp session."""
        if self._session and not self._session.closed:
            await self._session.close()

    def _normalize_query(self, query: str) -> str:
        """Normalize query for searching."""
        # Remove common prefixes/suffixes
        query = query.strip()
        # Handle ORF naming conventions
        query = re.sub(r'^(C\d+orf\d+)$', r'\1', query, flags=re.IGNORECASE)
        return query

    async def _search_hgnc(self, query: str) -> List[GeneHit]:
        """Search HGNC REST API.

        HGNC REST API: https://rest.genenames.org/
        - No API key required
        - Rate limit: reasonable for single queries
        """
        session = await self._get_session()
        hits = []

        try:
            # Search by symbol, alias, and previous symbol
            url = f"https://rest.genenames.org/search/{query}"
            headers = {"Accept": "application/json"}

            async with session.get(url, headers=headers) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    for doc in data.get("response", {}).get("docs", []):
                        hit = GeneHit(
                            symbol=doc.get("symbol", ""),
                            name=doc.get("name", ""),
                            hgnc_id=doc.get("hgnc_id", ""),
                            aliases=doc.get("alias_symbol", []) or [],
                            location=doc.get("location", ""),
                            locus_group=doc.get("locus_group", ""),
                            score=doc.get("score", 1.0),
                            source="hgnc",
                        )
                        if hit.symbol:
                            hits.append(hit)
                else:
                    logger.warning(f"HGNC search failed: {resp.status}")

        except Exception as e:
            logger.warning(f"HGNC search error: {e}")

        return hits

    async def _search_mygene(self, query: str) -> List[GeneHit]:
        """Search MyGene.info API.

        MyGene.info: https://mygene.info/
        - No API key required
        - Good coverage of gene aliases
        """
        session = await self._get_session()
        hits = []

        try:
            url = "https://mygene.info/v3/query"
            params = {
                "q": query,
                "species": "human",
                "fields": "symbol,name,alias,entrezgene,HGNC",
                "size": 5,
            }

            async with session.get(url, params=params) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    for doc in data.get("hits", []):
                        aliases = doc.get("alias", [])
                        if isinstance(aliases, str):
                            aliases = [aliases]

                        hit = GeneHit(
                            symbol=doc.get("symbol", ""),
                            name=doc.get("name", ""),
                            hgnc_id=doc.get("HGNC", ""),
                            aliases=aliases,
                            score=doc.get("_score", 1.0),
                            source="mygene",
                        )
                        if hit.symbol:
                            hits.append(hit)
                else:
                    logger.warning(f"MyGene search failed: {resp.status}")

        except Exception as e:
            logger.warning(f"MyGene search error: {e}")

        return hits

    async def resolve(self, query: str) -> Optional[str]:
        """Resolve a gene name/alias to official HGNC symbol.

        Args:
            query: Gene name, alias, or previous symbol

        Returns:
            Official HGNC symbol or None if not found
        """
        if not query:
            return None

        query = self._normalize_query(query)
        query_upper = query.upper()

        # Step 0: Check local mappings first (for known ORF patterns)
        if query_upper in self._orf_map:
            logger.info(f"Gene resolved from local cache: {query} -> {self._orf_map[query_upper]}")
            return self._orf_map[query_upper]

        # Step 1: Search HGNC
        hgnc_hits = await self._search_hgnc(query)
        if hgnc_hits:
            # Prefer exact symbol match
            for hit in hgnc_hits:
                if hit.symbol.upper() == query_upper:
                    logger.info(f"Gene resolved via HGNC (exact): {query} -> {hit.symbol}")
                    return hit.symbol
            # Otherwise return highest score
            best = max(hgnc_hits, key=lambda x: x.score)
            logger.info(f"Gene resolved via HGNC (best): {query} -> {best.symbol}")
            return best.symbol

        # Step 2: Search MyGene.info
        mygene_hits = await self._search_mygene(query)
        if mygene_hits:
            # Prefer exact symbol match
            for hit in mygene_hits:
                if hit.symbol.upper() == query_upper:
                    logger.info(f"Gene resolved via MyGene (exact): {query} -> {hit.symbol}")
                    return hit.symbol
                # Check aliases
                for alias in hit.aliases:
                    if alias.upper() == query_upper:
                        logger.info(f"Gene resolved via MyGene (alias): {query} -> {hit.symbol}")
                        return hit.symbol
            # Otherwise return highest score
            best = max(mygene_hits, key=lambda x: x.score)
            logger.info(f"Gene resolved via MyGene (best): {query} -> {best.symbol}")
            return best.symbol

        logger.warning(f"Gene not found: {query}")
        return None

    async def resolve_full(self, query: str) -> Optional[dict]:
        """Resolve gene and return full info including location.

        Uses HGNC fetch API (not search) to get location, aliases, name etc.

        Returns:
            Dict with symbol, name, location, aliases, or None if not found.
        """
        if not query:
            return None

        query = self._normalize_query(query)

        # Step 1: Resolve to official symbol first
        symbol = await self.resolve(query)
        if not symbol:
            return None

        # Step 2: Fetch full info from HGNC using /fetch/symbol/
        session = await self._get_session()
        try:
            url = f"https://rest.genenames.org/fetch/symbol/{symbol}"
            headers = {"Accept": "application/json"}
            async with session.get(url, headers=headers) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    docs = data.get("response", {}).get("docs", [])
                    if docs:
                        doc = docs[0]
                        return {
                            "symbol": doc.get("symbol", symbol),
                            "name": doc.get("name", ""),
                            "location": doc.get("location", ""),
                            "aliases": doc.get("alias_symbol", []) or [],
                            "locus_group": doc.get("locus_group", ""),
                        }
        except Exception as e:
            logger.warning(f"HGNC fetch error for {symbol}: {e}")

        # Fallback: return symbol-only result
        return {"symbol": symbol, "name": "", "location": "", "aliases": [], "locus_group": ""}

    async def resolve_batch(self, queries: List[str]) -> dict[str, Optional[str]]:
        """Resolve multiple gene names in parallel.

        Args:
            queries: List of gene names to resolve

        Returns:
            Dict mapping query to resolved symbol (or None)
        """
        tasks = [self.resolve(q) for q in queries]
        results = await asyncio.gather(*tasks)
        return dict(zip(queries, results))


# Convenience function for synchronous usage
def resolve_gene_symbol(query: str) -> Optional[str]:
    """Synchronous wrapper for gene resolution.

    Args:
        query: Gene name to resolve

    Returns:
        Official HGNC symbol or None
    """
    async def _resolve():
        resolver = GeneResolver()
        try:
            return await resolver.resolve(query)
        finally:
            await resolver.close()

    return asyncio.run(_resolve())
