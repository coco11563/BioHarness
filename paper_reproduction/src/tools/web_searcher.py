"""Web search tool with DuckDuckGo (default) and Serper (Google) backends.

Backends:
- DuckDuckGo: Free, no API key required (default)
- Serper: Requires SERPER_API_KEY, higher quality Google results

Fail-fast: raise on errors; no silent retries

Result schema:
{
  "title": str,
  "url": str,
  "snippet": str,
  "date": str | None,
  "domain": str,
  "position": int,
  "is_guideline": bool,
  "is_medical_source": bool,
}
"""

import os
import logging
from dataclasses import dataclass, field
from typing import Any, Optional, List
from urllib.parse import urlparse

import aiohttp

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Constants and Helpers
# ---------------------------------------------------------------------------

# Medical/scientific authority domains (whitelisted)
MEDICAL_ALLOWLIST = {
    # US Government / NIH
    "nih.gov",
    "ncbi.nlm.nih.gov",
    "cdc.gov",
    "fda.gov",
    "clinicaltrials.gov",
    # International
    "who.int",
    "nice.org.uk",
    "ema.europa.eu",
    # Major Journals
    "nejm.org",
    "jamanetwork.com",
    "thelancet.com",
    "bmj.com",
    "nature.com",
    "sciencedirect.com",
    "springer.com",
    "wiley.com",
    "cell.com",
    # Academic / Research
    "pubmed.ncbi.nlm.nih.gov",
    "pmc.ncbi.nlm.nih.gov",
    "cochranelibrary.com",
    "uptodate.com",
}

# Content farms and low-quality sources (blocklisted)
DEFAULT_BLOCKLIST = {
    "medium.com",
    "quora.com",
    "reddit.com",
    "facebook.com",
    "twitter.com",
    "x.com",
    "pinterest.com",
    "wikihow.com",
    "chegg.com",
    "coursehero.com",
    "scribd.com",
    "healthline.com",  # Often low quality
    "webmd.com",       # Often low quality
    "verywellhealth.com",
}


def _extract_domain(url: str) -> str:
    """Extract domain from URL, removing www prefix."""
    try:
        parsed = urlparse(url)
        host = parsed.netloc.lower()
        if host.startswith("www."):
            host = host[4:]
        return host
    except Exception:
        return ""


def _is_domain_match(domain: str, domain_set: set[str]) -> bool:
    """Check if domain matches any in set (including subdomains)."""
    return any(domain == d or domain.endswith("." + d) for d in domain_set)


def _is_guideline(domain: str, title: str, snippet: str) -> bool:
    """Detect if result is likely a clinical guideline."""
    # Domain-based signal
    guideline_domains = {"nice.org.uk", "who.int", "cdc.gov", "nih.gov", "cochranelibrary.com"}
    if _is_domain_match(domain, guideline_domains):
        return True

    # Keyword-based signal
    text = f"{title} {snippet}".lower()
    guideline_keywords = [
        "guideline", "recommendation", "practice guideline",
        "consensus statement", "clinical practice", "treatment protocol"
    ]
    return any(k in text for k in guideline_keywords)


# ---------------------------------------------------------------------------
# DuckDuckGo Backend (default, no API key required)
# ---------------------------------------------------------------------------

@dataclass
class DuckDuckGoSearcher:
    """DuckDuckGo Web Search client.

    Free, no API key required. Uses duckduckgo-search library.

    Example:
        searcher = DuckDuckGoSearcher()
        results = await searcher.search("diabetes treatment guidelines", limit=10)
    """

    timeout: float = 15.0
    allowlist: set[str] = field(default_factory=lambda: set(MEDICAL_ALLOWLIST))
    blocklist: set[str] = field(default_factory=lambda: set(DEFAULT_BLOCKLIST))
    filter_to_allowlist: bool = False

    async def search(
        self,
        query: str,
        limit: int = 10,
        freshness_days: int | None = None,
        filter_to_allowlist: bool | None = None,
    ) -> List[dict[str, Any]]:
        """Search via DuckDuckGo.

        Args:
            query: search query string
            limit: max results to return
            freshness_days: restrict to recent results (1=day, 7=week, 30=month)
            filter_to_allowlist: if True, only return results from allowlist domains

        Returns:
            List of structured dicts with: title, url, snippet, date, domain, position, is_guideline
        """
        if not query or not query.strip():
            raise ValueError("query must be non-empty")
        if limit <= 0:
            raise ValueError("limit must be > 0")

        if filter_to_allowlist is None:
            filter_to_allowlist = self.filter_to_allowlist

        # DuckDuckGo timelimit parameter
        timelimit = None
        if freshness_days is not None:
            if freshness_days <= 1:
                timelimit = "d"  # Past day
            elif freshness_days <= 7:
                timelimit = "w"  # Past week
            elif freshness_days <= 31:
                timelimit = "m"  # Past month
            else:
                timelimit = "y"  # Past year

        # Use sync DDGS in executor to avoid blocking
        import asyncio
        from concurrent.futures import ThreadPoolExecutor
        from ddgs import DDGS

        def _do_search():
            with DDGS(timeout=int(self.timeout)) as ddgs:
                # Fetch more results to compensate for filtering
                raw_results = list(ddgs.text(
                    query.strip(),
                    max_results=min(max(limit * 2, 10), 30),
                    timelimit=timelimit,
                ))
            return raw_results

        loop = asyncio.get_event_loop()
        with ThreadPoolExecutor(max_workers=1) as executor:
            raw_results = await loop.run_in_executor(executor, _do_search)

        results = []
        for idx, item in enumerate(raw_results):
            title = item.get("title", "") or ""
            item_url = item.get("href", "") or item.get("link", "") or ""
            snippet = item.get("body", "") or item.get("snippet", "") or ""

            if not item_url:
                continue

            domain = _extract_domain(item_url)
            if not domain:
                continue

            # Apply blocklist filter
            if _is_domain_match(domain, self.blocklist):
                continue

            # Apply allowlist filter if enabled
            if filter_to_allowlist and not _is_domain_match(domain, self.allowlist):
                continue

            is_medical_source = _is_domain_match(domain, self.allowlist)

            results.append({
                "title": title,
                "url": item_url,
                "snippet": snippet,
                "date": None,  # DuckDuckGo doesn't provide date
                "domain": domain,
                "position": idx + 1,
                "is_guideline": _is_guideline(domain, title, snippet),
                "is_medical_source": is_medical_source,
            })

            if len(results) >= limit:
                break

        return results

    async def search_medical(
        self,
        query: str,
        limit: int = 10,
        freshness_days: int | None = None,
    ) -> List[dict[str, Any]]:
        """Search restricted to medical authority domains only."""
        return await self.search(
            query=query,
            limit=limit,
            freshness_days=freshness_days,
            filter_to_allowlist=True,
        )


# ---------------------------------------------------------------------------
# Serper (Google) Backend (requires API key)
# ---------------------------------------------------------------------------

@dataclass
class WebSearcher:
    """Serper (Google) Web Search client.

    Uses Serper.dev API for Google search results.
    Supports domain filtering and medical source prioritization.

    Example:
        searcher = WebSearcher()
        results = await searcher.search("diabetes treatment guidelines", limit=10)
    """

    api_key: Optional[str] = None
    timeout: float = 15.0
    allowlist: set[str] = field(default_factory=lambda: set(MEDICAL_ALLOWLIST))
    blocklist: set[str] = field(default_factory=lambda: set(DEFAULT_BLOCKLIST))
    filter_to_allowlist: bool = False  # If True, only return allowlist domains

    def __post_init__(self):
        if self.api_key is None:
            self.api_key = os.getenv("SERPER_API_KEY")
        if not self.api_key:
            raise RuntimeError(
                "SERPER_API_KEY is required for WebSearcher. "
                "Get a free API key at https://serper.dev (2500 searches/month free)"
            )

    async def search(
        self,
        query: str,
        limit: int = 10,
        freshness_days: int | None = None,
        filter_to_allowlist: bool | None = None,
    ) -> List[dict[str, Any]]:
        """Search via Serper API.

        Args:
            query: search query string
            limit: max results to return
            freshness_days: restrict to recent results (1=day, 7=week, 30=month)
            filter_to_allowlist: if True, only return results from allowlist domains

        Returns:
            List of structured dicts with: title, url, snippet, date, domain, position, is_guideline

        Raises:
            ValueError: if query is empty
            RuntimeError: if API call fails
        """
        if not query or not query.strip():
            raise ValueError("query must be non-empty")
        if limit <= 0:
            raise ValueError("limit must be > 0")

        # Use instance default if not specified
        if filter_to_allowlist is None:
            filter_to_allowlist = self.filter_to_allowlist

        payload = {
            "q": query.strip(),
            "num": min(max(limit * 2, 10), 30),  # Fetch more to compensate for filtering
        }

        # Serper time filter using tbs parameter
        if freshness_days is not None:
            if freshness_days <= 1:
                payload["tbs"] = "qdr:d"  # Past day
            elif freshness_days <= 7:
                payload["tbs"] = "qdr:w"  # Past week
            elif freshness_days <= 31:
                payload["tbs"] = "qdr:m"  # Past month
            else:
                payload["tbs"] = "qdr:y"  # Past year

        headers = {
            "X-API-KEY": self.api_key,
            "Content-Type": "application/json",
        }

        url = "https://google.serper.dev/search"

        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=self.timeout)) as session:
            async with session.post(url, headers=headers, json=payload) as resp:
                if resp.status != 200:
                    text = await resp.text()
                    raise RuntimeError(f"Serper API error: {resp.status} - {text[:200]}")

                data = await resp.json()

        results = []
        organic = data.get("organic", [])

        for item in organic:
            title = item.get("title", "") or ""
            item_url = item.get("link", "") or ""
            snippet = item.get("snippet", "") or ""
            date = item.get("date")
            position = item.get("position", 0) or 0

            if not item_url:
                continue

            domain = _extract_domain(item_url)
            if not domain:
                continue

            # Apply blocklist filter
            if _is_domain_match(domain, self.blocklist):
                continue

            # Apply allowlist filter if enabled
            if filter_to_allowlist and not _is_domain_match(domain, self.allowlist):
                continue

            is_medical_source = _is_domain_match(domain, self.allowlist)

            results.append({
                "title": title,
                "url": item_url,
                "snippet": snippet,
                "date": date,
                "domain": domain,
                "position": position,
                "is_guideline": _is_guideline(domain, title, snippet),
                "is_medical_source": is_medical_source,
            })

            if len(results) >= limit:
                break

        return results

    async def search_medical(
        self,
        query: str,
        limit: int = 10,
        freshness_days: int | None = None,
    ) -> List[dict[str, Any]]:
        """Search restricted to medical authority domains only.

        Convenience method that forces filter_to_allowlist=True.
        """
        return await self.search(
            query=query,
            limit=limit,
            freshness_days=freshness_days,
            filter_to_allowlist=True,
        )


async def close_searcher(searcher):
    """No-op for API compatibility. Both searchers use short-lived sessions."""
    pass
