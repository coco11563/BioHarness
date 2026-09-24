"""Retrieval tools for BiomedicalREPL.

RLM Philosophy:
- RLM executes code to manage long-context retrieval and reasoning
- Tools are sync functions injected into REPL globals
- Each tool performs ONE clear operation (single responsibility)
- Tools return structured data that RLM can process in code

Tool Categories:
1. Search tools: Find papers/chunks in the corpus
2. Filter tools: Narrow down results using MeSH/keywords
3. Ranking tools: Reorder results by relevance
4. Structure tools: Access paper hierarchy (sections/chunks)
5. Helper tools: Format and extract information
6. External DBs: ChEMBL, UniProt, ClinicalTrials, Web Search
7. Entity resolution: GO, Gene symbol normalization

DESIGN: Fail-Fast
- All errors propagate immediately
- No fallbacks or silent retries
- Thread-safe lazy initialization

TOOL TRACING:
- Set TOOL_TRACE=1 to enable tool call logging
- Logs written to _tool_trace list (accessible via get_tool_trace())
"""

import functools
import logging
import os
import threading
import time

logger = logging.getLogger(__name__)
from dataclasses import dataclass, field
from typing import Any, Callable, List, Optional

# Use our improved async helper
from ..utils.async_helper import run_async


# =============================================================================
# Tool Call Tracing (for validation experiments)
# =============================================================================

@dataclass
class ToolCallRecord:
    """Record of a single tool call."""
    tool_name: str
    args: dict
    result_summary: str
    latency_ms: float
    success: bool
    error: Optional[str] = None
    timestamp: float = field(default_factory=time.time)


_tool_trace: List[ToolCallRecord] = []
_trace_enabled = os.getenv("TOOL_TRACE", "0") == "1"
_trace_lock = threading.Lock()


def enable_tool_trace():
    """Enable tool call tracing."""
    global _trace_enabled
    _trace_enabled = True


def disable_tool_trace():
    """Disable tool call tracing."""
    global _trace_enabled
    _trace_enabled = False


def get_tool_trace() -> List[ToolCallRecord]:
    """Get recorded tool calls."""
    with _trace_lock:
        return list(_tool_trace)


def clear_tool_trace():
    """Clear recorded tool calls."""
    global _tool_trace
    with _trace_lock:
        _tool_trace = []


def _summarize_result(result: Any, max_len: int = 100) -> str:
    """Create a summary of tool result for logging."""
    if result is None:
        return "None"
    if isinstance(result, list):
        return f"list[{len(result)}]"
    if isinstance(result, dict):
        keys = list(result.keys())[:5]
        return f"dict{{{', '.join(keys)}{'...' if len(result) > 5 else ''}}}"
    s = str(result)
    return s[:max_len] + "..." if len(s) > max_len else s


def traced_tool(tool_name: str):
    """Decorator to trace tool calls."""
    def decorator(fn: Callable) -> Callable:
        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            if not _trace_enabled:
                return fn(*args, **kwargs)

            start = time.time()
            try:
                result = fn(*args, **kwargs)
                latency = (time.time() - start) * 1000

                record = ToolCallRecord(
                    tool_name=tool_name,
                    args={"args": args[:2], "kwargs": {k: str(v)[:50] for k, v in list(kwargs.items())[:3]}},
                    result_summary=_summarize_result(result),
                    latency_ms=latency,
                    success=True,
                )
                with _trace_lock:
                    _tool_trace.append(record)
                return result
            except Exception as e:
                latency = (time.time() - start) * 1000
                record = ToolCallRecord(
                    tool_name=tool_name,
                    args={"args": args[:2], "kwargs": {k: str(v)[:50] for k, v in list(kwargs.items())[:3]}},
                    result_summary="ERROR",
                    latency_ms=latency,
                    success=False,
                    error=str(e)[:200],
                )
                with _trace_lock:
                    _tool_trace.append(record)
                raise
        return wrapper
    return decorator

# =============================================================================
# Thread-Safe Lazy Singletons
# =============================================================================

_lock = threading.Lock()
_retriever = None
_chunk_search = None
_section_aware_chunk_search = None
_chunk_qdrant = None  # Keep reference for cleanup
_keyword_filter = None
_db_pool = None
_go_resolver = None
_gene_resolver = None
_web_searcher = None
_chembl_resolver = None
_uniprot_resolver = None
_clinicaltrials_resolver = None
_genomics_resolver = None


def _get_retriever():
    """Thread-safe lazy-load of hybrid retriever."""
    global _retriever
    if _retriever is None:
        with _lock:
            if _retriever is None:
                from ..retriever.hybrid_retriever import HybridRetriever, HybridConfig
                config = HybridConfig(enable_rerank=False)
                _retriever = HybridRetriever(config)
    return _retriever


def _get_chunk_search():
    """Thread-safe lazy-load of chunk search."""
    global _chunk_search, _chunk_qdrant
    if _chunk_search is None:
        with _lock:
            if _chunk_search is None:
                from qdrant_client import AsyncQdrantClient
                from ..retriever.chunk_search import ChunkSearch
                from ..config import get_config
                cfg = get_config()
                # 60 s is not enough for this collection: a cold query over the
                # 128M-point on-disk chunk index takes ~15 s alone, and the agent
                # runs while the cascade holds its own client, so both compete.
                # Every agent search_chunks call observed in the 2026-09-09 probe
                # failed at ~22 s with "All connection attempts failed"; the same
                # queries succeed 8/8 in isolation.
                _chunk_qdrant = AsyncQdrantClient(
                    cfg.qdrant.url, check_compatibility=False,
                    timeout=int(os.environ.get("XC_AGENT_QDRANT_TIMEOUT", "300")),
                )
                _chunk_search = ChunkSearch(_chunk_qdrant)
    return _chunk_search


def _get_section_aware_chunk_search():
    """Thread-safe lazy-load of section-aware chunk search."""
    global _section_aware_chunk_search
    if _section_aware_chunk_search is None:
        with _lock:
            if _section_aware_chunk_search is None:
                from ..retriever.section_aware_chunk_search import SectionAwareChunkSearch
                _section_aware_chunk_search = SectionAwareChunkSearch(_get_chunk_search())
    return _section_aware_chunk_search


_db_pools: dict[int, Any] = {}


async def _get_db_pool_async():
    """Return an asyncpg pool bound to the CURRENTLY running event loop.

    asyncpg connections belong to the loop that created them. A single global pool
    built on the AsyncExecutor's loop raised "another operation is in progress" as
    soon as a nested tool call ran the coroutine on a different loop, which is how
    get_paper_abstracts failed inside the agent. Keying the pool by loop keeps each
    loop's connections to itself.
    """
    import asyncio
    import asyncpg
    from ..config import get_config

    key = id(asyncio.get_running_loop())
    pool = _db_pools.get(key)
    if pool is None:
        pool = await asyncpg.create_pool(
            get_config().postgres.pubmed_url, min_size=1, max_size=10,
        )
        _db_pools[key] = pool
    return pool


def _get_db_pool():
    """Thread-safe lazy-load of database pool (sync callers only)."""
    global _db_pool
    if _db_pool is None:
        with _lock:
            if _db_pool is None:
                import asyncpg
                from ..config import get_config
                cfg = get_config()

                async def create_pool():
                    return await asyncpg.create_pool(
                        cfg.postgres.pubmed_url,
                        min_size=2,
                        max_size=10,
                    )

                _db_pool = run_async(create_pool())
    return _db_pool


def _get_go_resolver():
    """Thread-safe lazy-load of GOResolver.

    Note: Does NOT call initialize() here to avoid event loop issues.
    Each async tool function must ensure initialization.
    """
    global _go_resolver
    if _go_resolver is None:
        with _lock:
            if _go_resolver is None:
                from ..tools.go_resolver import GOResolver
                _go_resolver = GOResolver()
    return _go_resolver


def _get_gene_resolver():
    """Thread-safe lazy-load of GeneResolver."""
    global _gene_resolver
    if _gene_resolver is None:
        with _lock:
            if _gene_resolver is None:
                from ..tools.gene_resolver import GeneResolver
                _gene_resolver = GeneResolver()
    return _gene_resolver


def _get_web_searcher():
    """Thread-safe lazy-load of DuckDuckGoSearcher.

    Uses DuckDuckGo by default (free, no API key required).
    For Google results, set SERPER_API_KEY and use WebSearcher directly.
    """
    global _web_searcher
    if _web_searcher is None:
        with _lock:
            if _web_searcher is None:
                from ..tools.web_searcher import DuckDuckGoSearcher
                _web_searcher = DuckDuckGoSearcher()
    return _web_searcher


def _get_chembl_resolver():
    """Thread-safe lazy-load of ChEMBLResolver.

    Note: ChEMBL client is sync, so no async operations needed.
    """
    global _chembl_resolver
    if _chembl_resolver is None:
        with _lock:
            if _chembl_resolver is None:
                from ..tools.chembl_resolver import ChEMBLResolver
                _chembl_resolver = ChEMBLResolver()
    return _chembl_resolver


def _get_uniprot_resolver():
    """Thread-safe lazy-load of UniProtResolver."""
    global _uniprot_resolver
    if _uniprot_resolver is None:
        with _lock:
            if _uniprot_resolver is None:
                from ..tools.uniprot_resolver import UniProtResolver
                _uniprot_resolver = UniProtResolver()
    return _uniprot_resolver


def _get_clinicaltrials_resolver():
    """Thread-safe lazy-load of ClinicalTrialsResolver."""
    global _clinicaltrials_resolver
    if _clinicaltrials_resolver is None:
        with _lock:
            if _clinicaltrials_resolver is None:
                from ..tools.clinicaltrials_resolver import ClinicalTrialsResolver
                _clinicaltrials_resolver = ClinicalTrialsResolver()
    return _clinicaltrials_resolver


def _get_genomics_resolver():
    """Thread-safe lazy-load of GenomicsResolver (NCBI dbSNP + MyGene, sync)."""
    global _genomics_resolver
    if _genomics_resolver is None:
        with _lock:
            if _genomics_resolver is None:
                from ..tools.genomics_resolver import GenomicsResolver
                _genomics_resolver = GenomicsResolver()
    return _genomics_resolver


def shutdown_tools():
    """Shutdown all tool singletons and release resources.

    Call this when done with REPL/pipeline to prevent resource leaks.
    Safe to call multiple times.
    """
    global _retriever, _chunk_search, _section_aware_chunk_search, _chunk_qdrant, _db_pool, _go_resolver, _gene_resolver, _web_searcher, _chembl_resolver, _uniprot_resolver, _clinicaltrials_resolver, _genomics_resolver

    async def _shutdown():
        # Close each resource independently so one failure doesn't skip the rest
        resources = [
            ("retriever", _retriever),
            ("chunk_qdrant", _chunk_qdrant),
            ("section_aware_chunk_search", _section_aware_chunk_search),
            ("db_pool", _db_pool),
            ("go_resolver", _go_resolver),
            ("gene_resolver", _gene_resolver),
            ("uniprot_resolver", _uniprot_resolver),
            ("clinicaltrials_resolver", _clinicaltrials_resolver),
        ]
        for name, resource in resources:
            if resource is not None:
                try:
                    await resource.close()
                except Exception as e:
                    logger.warning(f"Failed to close {name}: {e}")

    with _lock:
        if any([_retriever, _chunk_search, _section_aware_chunk_search, _db_pool, _go_resolver, _gene_resolver, _uniprot_resolver, _clinicaltrials_resolver]):
            try:
                run_async(_shutdown())
            except Exception as e:
                logger.warning(f"shutdown_tools error: {e}")

        # GenomicsResolver uses a sync httpx.Client; close it directly.
        if _genomics_resolver is not None:
            try:
                _genomics_resolver.close()
            except Exception as e:
                logger.warning(f"Failed to close genomics_resolver: {e}")

        _retriever = None
        _chunk_search = None
        _section_aware_chunk_search = None
        _chunk_qdrant = None
        _db_pool = None
        _go_resolver = None
        _gene_resolver = None
        _web_searcher = None
        _chembl_resolver = None
        _uniprot_resolver = None
        _clinicaltrials_resolver = None
        _genomics_resolver = None


# =============================================================================
# 1. SEARCH TOOLS - Find papers and chunks
# =============================================================================

def search_papers(query: str, limit: int = 50) -> list[dict[str, Any]]:
    """Search PubMed papers (27.3M) using hybrid MeSH+keyword+vector retrieval.

    This is the primary search tool. Uses:
    - MeSH term matching for precision
    - Keyword filtering for recall
    - Vector similarity for semantic matching
    - RRF fusion to combine results

    Args:
        query: Natural language biomedical query
        limit: Maximum results to return (default 50)

    Returns:
        List of paper dicts with: pmid, title, abstract, score, rank, have_fulltext

    Example:
        papers = search_papers("Does metformin help diabetes?", limit=20)
        for p in papers[:5]:
            print(f"{p['pmid']}: {p['title'][:60]}...")
    """
    async def _search():
        retriever = _get_retriever()
        await retriever.initialize()
        result = await retriever.retrieve(query)
        return [
            {
                "pmid": abs.pmid,
                "title": abs.title,
                "abstract": abs.abstract,
                "score": abs.score,
                "rank": abs.rank,
                "have_fulltext": abs.have_fulltext,
            }
            for abs in result.abstracts[:limit]
        ]

    return run_async(_search())


def search_chunks(
    query: str,
    limit: int = 50,
    paper_ids: list[str] | None = None,
) -> list[dict[str, Any]]:
    """Search full-text chunks from PMC papers (129M chunks).

    Use this for fine-grained evidence when abstracts aren't enough.
    Can filter to specific papers for focused search.

    Args:
        query: Natural language query
        limit: Maximum chunks to return (default 50)
        paper_ids: Optional PMIDs to filter (search only these papers)

    Returns:
        List of chunk dicts with: chunk_id, pmid, pmcid, text, section_id, score, rank

    Example:
        # Get chunks from top papers
        papers = search_papers("BRCA1 cancer treatment")
        pmids = [p['pmid'] for p in papers[:10]]
        chunks = search_chunks("treatment response", paper_ids=pmids)
    """
    async def _search():
        chunk_search = _get_chunk_search()
        pmid_filter = set(paper_ids) if paper_ids else None
        chunks, _ = await chunk_search.search(query, pmid_filter, limit)
        return [
            {
                "chunk_id": c.chunk_id,
                "pmid": c.pmid,
                "pmcid": c.pmcid,
                "text": c.text,
                "section_id": c.section_id,
                "score": c.score,
                "rank": c.rank,
                "token_count": c.token_count,
            }
            for c in chunks
        ]

    return run_async(_search())


def search_section_aware_chunks(
    query: str,
    limit: int = 20,
    paper_ids: list[str] | None = None,
    preferred_section_types: list[str] | None = None,
    seed_limit: int | None = None,
) -> list[dict[str, Any]]:
    """Search full-text with section-aware re-ranking and intra-paper expansion.

    Unlike flat chunk search, this tool downweights introduction/background hits
    and expands toward higher-value sections such as results/discussion/conclusion.

    Args:
        query: Natural language query
        limit: Maximum chunks to return
        paper_ids: Optional PMIDs to restrict search to
        preferred_section_types: Optional preferred section roles
        seed_limit: Optional number of seed chunks before section-aware reranking

    Returns:
        List of chunk dicts with section metadata and adjusted_score
    """
    async def _search():
        searcher = _get_section_aware_chunk_search()
        chunks = await searcher.search(
            query=query,
            paper_ids=paper_ids,
            preferred_section_types=preferred_section_types,
            limit=limit,
            seed_limit=seed_limit,
        )
        return [
            {
                "chunk_id": c["chunk_id"],
                "paper_id": c["paper_id"],
                "pmid": c["pmid"],
                "pmcid": c["pmcid"],
                "text": c["text"],
                "section_id": c["section_id"],
                "section_title": c["section_title"],
                "section_type": c["section_type"],
                "section_role": c["section_role"],
                "hierarchy_level": c["hierarchy_level"],
                "score": c["score"],
                "adjusted_score": c["adjusted_score"],
                "rank": c["rank"],
                "retrieval_stage": c["retrieval_stage"],
                "token_count": c["token_count"],
            }
            for c in chunks
        ]

    return run_async(_search())


def search_by_entities(entities: list[str], limit: int = 30) -> list[dict[str, Any]]:
    """Search papers by biomedical entities (genes, drugs, diseases, etc.).

    Constructs optimized query from entities for targeted retrieval.

    Args:
        entities: List of biomedical entities
        limit: Maximum results

    Returns:
        List of paper dicts

    Example:
        papers = search_by_entities(["BRCA1", "breast cancer", "PARP inhibitor"])
    """
    query = " ".join(entities)
    return search_papers(query, limit)


# =============================================================================
# 2. FILTER TOOLS - Narrow down using MeSH/keywords
# =============================================================================

def search_mesh_terms(query: str, entities: list[str] | None = None) -> list[dict[str, Any]]:
    """Find relevant MeSH (Medical Subject Headings) terms for a query.

    MeSH terms are controlled vocabulary for indexing biomedical literature.
    Use this to understand the domain terminology for your query.

    Args:
        query: Natural language query
        entities: Optional entities for better matching

    Returns:
        List of MeSH dicts with: descriptor_ui, descriptor_name, score, tree_numbers

    Example:
        mesh = search_mesh_terms("heart attack treatment")
        # Returns: [{"descriptor_name": "Myocardial Infarction", ...}, ...]
    """
    async def _search():
        retriever = _get_retriever()
        await retriever.initialize()
        mesh_filter = retriever._mesh_filter
        matches, _ = await mesh_filter.match_mesh_terms(query, entities=entities)
        return [
            {
                "descriptor_ui": m.descriptor_ui,
                "descriptor_name": m.descriptor_name,
                "score": m.score,
                "tree_numbers": m.tree_numbers,
            }
            for m in matches
        ]

    return run_async(_search())


def get_pmids_by_mesh(mesh_terms: list[str], limit: int = 1000) -> list[str]:
    """Get PMIDs indexed with specific MeSH terms.

    Useful for precise filtering based on controlled vocabulary.

    Args:
        mesh_terms: List of MeSH descriptor names or UIs
        limit: Maximum PMIDs to return

    Returns:
        List of PMID strings

    Example:
        pmids = get_pmids_by_mesh(["Diabetes Mellitus, Type 2", "Metformin"])
    """
    async def _get_pmids():
        retriever = _get_retriever()
        await retriever.initialize()
        mesh_filter = retriever._mesh_filter
        matches = []

        for term in mesh_terms:
            term_matches, _ = await mesh_filter.match_mesh_terms(term, entities=[term])
            matches.extend(term_matches[:2])

        if not matches:
            return []

        pmids, _ = await mesh_filter.get_pmids_by_mesh(matches, limit=limit)
        return list(pmids)

    return run_async(_get_pmids())


def keyword_search(
    keywords: list[str],
    limit: int = 100,
    match_all: bool = False,
) -> list[str]:
    """Search PMIDs using full-text keyword matching.

    Uses PostgreSQL full-text search for fast keyword lookup.

    Args:
        keywords: List of keywords to search
        limit: Maximum PMIDs to return
        match_all: If True, papers must contain ALL keywords (AND); else ANY (OR)

    Returns:
        List of PMID strings

    Example:
        pmids = keyword_search(["metformin", "diabetes"], match_all=True)
    """
    async def _search():
        retriever = _get_retriever()
        await retriever.initialize()
        keyword_filter = retriever._keyword_filter
        pmids, _ = await keyword_filter.get_pmids_by_keywords(
            keywords,
            limit=limit,
            match_any=not match_all,
        )
        return list(pmids)

    return run_async(_search())


# =============================================================================
# 3. RANKING TOOLS - Reorder by relevance
# =============================================================================

def rerank_papers(
    query: str,
    papers: list[dict[str, Any]],
    top_k: int = 10,
) -> list[dict[str, Any]]:
    """Rerank papers by relevance using Qwen3-Reranker-8B.

    Cross-encoder reranking provides more accurate relevance scoring
    than initial retrieval. Use after search_papers() for best results.

    Args:
        query: Query to rank against
        papers: List of paper dicts (must have 'abstract' or 'title')
        top_k: Number of top results to return

    Returns:
        Reranked papers with 'rerank_score' added

    Example:
        papers = search_papers("metformin diabetes", limit=50)
        top_papers = rerank_papers("Does metformin help diabetes?", papers, top_k=10)
    """
    async def _rerank():
        if not papers:
            return []

        from ..utils.clients import rerank_client

        documents = [
            (p.get("abstract", "") or p.get("title", ""))[:1500]
            for p in papers
        ]

        results = await rerank_client.rerank(query, documents, top_k=top_k)

        reranked = []
        for idx, score in results:
            paper = papers[idx].copy()
            paper["rerank_score"] = score
            reranked.append(paper)

        return reranked

    return run_async(_rerank())


def rerank_chunks(
    query: str,
    chunks: list[dict[str, Any]],
    top_k: int = 10,
) -> list[dict[str, Any]]:
    """Rerank chunks by relevance using cross-encoder.

    Args:
        query: Query to rank against
        chunks: List of chunk dicts (must have 'text')
        top_k: Number of top results to return

    Returns:
        Reranked chunks with 'rerank_score' added
    """
    async def _rerank():
        if not chunks:
            return []

        from ..utils.clients import rerank_client

        documents = [c.get("text", "")[:1500] for c in chunks]
        results = await rerank_client.rerank(query, documents, top_k=top_k)

        reranked = []
        for idx, score in results:
            chunk = chunks[idx].copy()
            chunk["rerank_score"] = score
            reranked.append(chunk)

        return reranked

    return run_async(_rerank())


# =============================================================================
# 4. STRUCTURE TOOLS - Access paper hierarchy
# =============================================================================

def get_paper_abstracts(pmids: list[str]) -> dict[str, dict[str, str]]:
    """Fetch abstracts for specific PMIDs.

    Args:
        pmids: List of PubMed IDs

    Returns:
        Dict mapping pmid to {title, abstract}

    Example:
        abstracts = get_paper_abstracts(["12345678", "87654321"])
        for pmid, data in abstracts.items():
            print(f"{pmid}: {data['title']}")
    """
    async def _fetch():
        pool = await _get_db_pool_async()
        async with pool.acquire() as conn:
            placeholders = ", ".join(f"${i+1}" for i in range(len(pmids)))
            rows = await conn.fetch(f"""
                SELECT pmid::text, title, abstract
                FROM articles
                WHERE pmid IN ({placeholders})
            """, *[int(p) for p in pmids])

        return {
            row["pmid"]: {
                "title": row["title"] or "",
                "abstract": row["abstract"] or "",
            }
            for row in rows
        }

    return run_async(_fetch())


def get_paper_sections(pmid: str) -> dict[str, Any]:
    """Get hierarchical section structure of a PMC full-text paper.

    Returns the paper's sections with subsections and chunks,
    organized as a tree. Only works for papers with full-text (have_fulltext=True).

    Args:
        pmid: PubMed ID of the paper

    Returns:
        Dict with: pmid, pmcid, title, abstract, sections (tree), total_sections, total_chunks
        Each section has: id, title, type, level, content, subsections, chunks

    Example:
        structure = get_paper_sections("12345678")
        for section in structure['sections']:
            print(f"- {section['title']} ({len(section['chunks'])} chunks)")
    """
    async def _get_structure():
        import asyncpg
        from ..config import get_config

        cfg = get_config()
        conn = await asyncpg.connect(cfg.postgres.papergraph_url)

        try:
            paper = await conn.fetchrow("""
                SELECT id, pmid, pmcid, title, abstract
                FROM papers WHERE pmid = $1
            """, pmid)

            if not paper:
                return {"error": f"Paper {pmid} not found in PMC full-text database"}

            paper_id = paper["id"]

            sections = await conn.fetch("""
                SELECT id, title, section_type, hierarchy_level, sequence_order, content_text
                FROM sections WHERE paper_id = $1
                ORDER BY hierarchy_level, sequence_order
            """, paper_id)

            rels = await conn.fetch("""
                SELECT child_section_id, parent_section_id
                FROM section_parent_rels spr
                JOIN sections s ON spr.child_section_id = s.id
                WHERE s.paper_id = $1
            """, paper_id)

            chunks = await conn.fetch("""
                SELECT c.id, c.section_id, c.text_content, c.sequence_order, c.token_count
                FROM chunks c JOIN sections s ON c.section_id = s.id
                WHERE s.paper_id = $1 ORDER BY c.sequence_order
            """, paper_id)

        finally:
            await conn.close()

        # Build hierarchy with proper ordering
        parent_map = {str(r["child_section_id"]): str(r["parent_section_id"]) for r in rels}
        children_map: dict[str, list] = {}
        for r in rels:
            pid = str(r["parent_section_id"])
            children_map.setdefault(pid, []).append(str(r["child_section_id"]))

        chunks_by_section: dict[str, list] = {}
        for c in chunks:
            sid = str(c["section_id"])
            chunks_by_section.setdefault(sid, []).append({
                "chunk_id": str(c["id"]),
                "text": c["text_content"] or "",
                "sequence": c["sequence_order"],
                "token_count": c["token_count"],
            })

        # Build section dict with sequence info for sorting
        section_dict = {}
        section_sequence = {}  # Track sequence_order for sorting
        for s in sections:
            sid = str(s["id"])
            section_dict[sid] = {
                "id": sid,
                "title": s["title"] or "",
                "type": s["section_type"] or "",
                "level": s["hierarchy_level"],
                "content": s["content_text"] or "",
                "subsections": [],
                "chunks": chunks_by_section.get(sid, []),
            }
            section_sequence[sid] = (s["hierarchy_level"], s["sequence_order"] or 0)

        # Attach children to parents, sorted by sequence_order
        for sid, sec in section_dict.items():
            child_ids = children_map.get(sid, [])
            # Sort children by (level, sequence_order)
            child_ids_sorted = sorted(child_ids, key=lambda cid: section_sequence.get(cid, (999, 999)))
            for child_id in child_ids_sorted:
                if child_id in section_dict:
                    sec["subsections"].append(section_dict[child_id])

        # Get top-level sections (no parent), sorted by (level, sequence_order)
        top_sections = [
            sec for sid, sec in section_dict.items()
            if sid not in parent_map
        ]
        top_sections.sort(key=lambda x: section_sequence.get(x["id"], (999, 999)))

        return {
            "pmid": paper["pmid"],
            "pmcid": paper["pmcid"],
            "title": paper["title"],
            "abstract": paper["abstract"],
            "sections": top_sections,
            "total_sections": len(sections),
            "total_chunks": len(chunks),
        }

    return run_async(_get_structure())


# =============================================================================
# 5. HYBRID/ADVANCED TOOLS
# =============================================================================

def hybrid_search(
    query: str,
    limit: int = 50,
    use_mesh: bool = True,
    use_keywords: bool = True,
    use_vector: bool = True,
) -> list[dict[str, Any]]:
    """Configurable hybrid search with path selection.

    Allows fine-tuning which retrieval paths to use.

    Args:
        query: Natural language query
        limit: Maximum results
        use_mesh: Enable MeSH path
        use_keywords: Enable keyword path
        use_vector: Enable dense vector path

    Returns:
        List of paper dicts with scores from enabled paths
    """
    async def _search():
        from ..retriever.hybrid_retriever import HybridRetriever, HybridConfig

        config = HybridConfig(
            enable_mesh_path=use_mesh,
            enable_keyword_path=use_keywords,
            enable_dense_path=use_vector,
            enable_rerank=False,
        )
        retriever = HybridRetriever(config)
        await retriever.initialize()

        try:
            result = await retriever.retrieve(query)
            return [
                {
                    "pmid": abs.pmid,
                    "title": abs.title,
                    "abstract": abs.abstract,
                    "score": abs.score,
                    "rank": abs.rank,
                    "have_fulltext": abs.have_fulltext,
                }
                for abs in result.abstracts[:limit]
            ]
        finally:
            await retriever.close()

    return run_async(_search())


def iterative_search(
    query: str,
    rounds: int = 2,
    papers_per_round: int = 20,
) -> list[dict[str, Any]]:
    """Iterative search with result-informed expansion.

    Each round:
    1. Search for papers
    2. Extract key entities from top results
    3. Expand query with new entities
    4. Search again

    Args:
        query: Initial query
        rounds: Number of search rounds
        papers_per_round: Papers to retrieve each round

    Returns:
        Accumulated unique papers across all rounds
    """
    all_papers: dict[str, dict] = {}
    current_query = query

    for round_num in range(rounds):
        papers = search_papers(current_query, limit=papers_per_round)

        for p in papers:
            if p["pmid"] not in all_papers:
                all_papers[p["pmid"]] = p

        if round_num < rounds - 1 and papers:
            # Extract entities from top abstracts for expansion
            top_abstracts = " ".join(p["abstract"][:500] for p in papers[:5])
            # Simple entity extraction: capitalized terms
            import re
            entities = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', top_abstracts)
            unique_entities = list(set(entities))[:5]
            if unique_entities:
                current_query = f"{query} {' '.join(unique_entities)}"

    return list(all_papers.values())


# =============================================================================
# 6. HELPER TOOLS - Format and extract
# =============================================================================

def format_papers(papers: list[dict], max_papers: int = 5) -> str:
    """Format papers for LLM consumption.

    Creates a readable summary of papers with key information.

    Args:
        papers: List of paper dicts from search
        max_papers: Maximum papers to include

    Returns:
        Formatted string with paper summaries

    Example:
        papers = search_papers("metformin diabetes")
        context = format_papers(papers, max_papers=10)
        # Use context in LLM prompt
    """
    if not papers:
        return "No papers found."

    lines = []
    for i, p in enumerate(papers[:max_papers], 1):
        title = p.get("title", "No title")
        abstract = p.get("abstract", "")[:500]
        pmid = p.get("pmid", "?")
        score = p.get("rerank_score") or p.get("score", 0)

        lines.append(f"[{i}] PMID: {pmid} (score: {score:.3f})")
        lines.append(f"    Title: {title}")
        if abstract:
            lines.append(f"    Abstract: {abstract}...")
        lines.append("")

    return "\n".join(lines)


def format_chunks(chunks: list[dict], max_chunks: int = 10) -> str:
    """Format chunks for LLM consumption.

    Args:
        chunks: List of chunk dicts from search
        max_chunks: Maximum chunks to include

    Returns:
        Formatted string with chunk summaries
    """
    if not chunks:
        return "No chunks found."

    lines = []
    for i, c in enumerate(chunks[:max_chunks], 1):
        text = c.get("text", "")[:400]
        pmid = c.get("pmid", "?")
        score = c.get("rerank_score") or c.get("adjusted_score") or c.get("score", 0)
        section_role = c.get("section_role")
        section_title = c.get("section_title")
        retrieval_stage = c.get("retrieval_stage")

        header = f"[{i}] PMID: {pmid} (score: {score:.3f})"
        if section_role:
            header += f" [{section_role}]"
        if retrieval_stage:
            header += f" [{retrieval_stage}]"

        lines.append(header)
        if section_title:
            lines.append(f"    Section: {section_title}")
        lines.append(f"    {text}...")
        lines.append("")

    return "\n".join(lines)


# =============================================================================
# TOOL REGISTRY
# =============================================================================

# =============================================================================
# Adaptive Evidence Tools
# =============================================================================

def judge_evidence(
    question: str,
    evidence: list[str],
    question_type: str = "yesno",
    gaps: list[str] | None = None,
) -> dict[str, Any]:
    """Let LLM judge if collected evidence is sufficient to answer the question.

    This is a key tool for adaptive evidence collection. The model calls this
    after collecting evidence to decide whether to:
    - Answer now (status="enough")
    - Expand search (status="need_more")
    - Search by entities (status="needs_entities")
    - Drill into fulltext (status="needs_fulltext")

    Args:
        question: The biomedical question
        evidence: List of evidence texts collected so far
        question_type: Type of question (yesno, mcq, factoid, etc.)
        gaps: Known gaps from previous iterations

    Returns:
        Dict with: status, gaps, suggested_actions, confidence
        - status: "enough" | "need_more" | "needs_entities" | "needs_fulltext"
        - gaps: List of missing information
        - suggested_actions: List of suggested next steps
        - confidence: 0.0-1.0 confidence in having enough evidence

    Example:
        judgment = judge_evidence(question, [abstract1, abstract2], "yesno")
        if judgment['status'] == 'need_more':
            # Expand to more papers
        elif judgment['status'] == 'needs_fulltext':
            # Drill into fulltext
    """
    from ..utils.clients import llm_client

    # Format evidence
    evidence_text = "\n---\n".join(evidence[:10])  # Limit to 10 snippets
    gap_text = ", ".join(gaps) if gaps else "None identified yet"

    prompt = f"""You are evaluating whether the collected evidence is sufficient to answer a biomedical question.

Question: {question}
Question Type: {question_type}

Previously identified gaps: {gap_text}

Collected Evidence:
{evidence_text}

Evaluate the evidence: choose a status, give a confidence (0-1), list the missing
information (gaps), suggest next steps, and explain briefly.

Guidelines:
- "enough": Evidence clearly supports a definitive answer
- "need_more": Need more papers/abstracts from search
- "needs_entities": Should search by specific entities (genes, drugs, diseases)
- "needs_fulltext": Should look at full paper content for details

Be conservative - only say "enough" if evidence is conclusive."""

    # Schema-constrained decoding (vLLM structured outputs): the server guarantees
    # the reply parses and carries exactly these fields, so no regex extraction.
    schema = {
        "type": "object",
        "properties": {
            "status": {"enum": ["enough", "need_more", "needs_entities", "needs_fulltext"]},
            "confidence": {"type": "number", "minimum": 0, "maximum": 1},
            "gaps": {"type": "array", "items": {"type": "string"}},
            "suggested_actions": {"type": "array", "items": {"type": "string"}},
            "reasoning": {"type": "string"},
        },
        "required": ["status", "confidence", "gaps", "suggested_actions", "reasoning"],
        "additionalProperties": False,
    }
    resp = run_async(llm_client.chat_raw(
        [{"role": "user", "content": prompt}],
        max_tokens=500,
        extra_body={"response_format": {
            "type": "json_schema",
            "json_schema": {"name": "evidence_judgment", "schema": schema},
        }},
    ))

    import json
    try:
        return json.loads(resp.choices[0].message.content or "")
    except (json.JSONDecodeError, AttributeError, IndexError):
        pass

    # Default if the server returned nothing parseable
    return {
        "status": "need_more",
        "gaps": ["Could not parse LLM response"],
        "suggested_actions": ["Expand search"],
        "confidence": 0.3,
    }


def get_paper_meta(paper_id: str) -> dict[str, Any]:
    """Get paper metadata including fulltext availability.

    Args:
        paper_id: PMID or paper UUID

    Returns:
        Dict with: id, pmid, title, abstract, have_fulltext, pmcid

    Example:
        meta = get_paper_meta("12345678")
        if meta['have_fulltext']:
            sections = list_sections(meta['id'])
    """
    async def _get_meta():
        import asyncpg
        from ..config import get_config

        cfg = get_config()
        conn = await asyncpg.connect(cfg.postgres.papergraph_url)

        try:
            # Try by PMID first
            paper = await conn.fetchrow("""
                SELECT id, pmid, pmcid, title, abstract,
                       CASE WHEN pmcid IS NOT NULL THEN true ELSE false END as have_fulltext
                FROM papers WHERE pmid = $1
            """, paper_id)

            if not paper:
                # Try by UUID
                try:
                    import uuid
                    uuid_val = uuid.UUID(paper_id)
                    paper = await conn.fetchrow("""
                        SELECT id, pmid, pmcid, title, abstract,
                               CASE WHEN pmcid IS NOT NULL THEN true ELSE false END as have_fulltext
                        FROM papers WHERE id = $1
                    """, uuid_val)
                except ValueError:
                    pass

            if not paper:
                return {"error": f"Paper {paper_id} not found"}

            return dict(paper)
        finally:
            await conn.close()

    return run_async(_get_meta())


def list_sections(
    paper_id: str,
    focus: str | None = None,
    section_types: list[str] | None = None,
    limit: int = 10,
) -> list[dict[str, Any]]:
    """List sections of a fulltext paper, optionally filtered by focus query.

    Args:
        paper_id: PMID or paper UUID
        focus: Optional query to filter sections (matches title/content)
        section_types: Optional list of types to include (e.g., ["methods", "results"])
        limit: Maximum sections to return

    Returns:
        List of dicts with: id, title, section_type, level, content_preview

    Example:
        # Get all sections
        sections = list_sections("12345678")

        # Get sections relevant to a specific topic
        sections = list_sections("12345678", focus="treatment efficacy")

        # Get only methods and results
        sections = list_sections("12345678", section_types=["methods", "results"])
    """
    async def _list():
        import asyncpg
        from ..config import get_config

        cfg = get_config()
        conn = await asyncpg.connect(cfg.postgres.papergraph_url)

        try:
            # First get the paper UUID
            paper = await conn.fetchrow("""
                SELECT id FROM papers WHERE pmid = $1
            """, paper_id)

            if not paper:
                try:
                    import uuid
                    paper = {"id": uuid.UUID(paper_id)}
                except ValueError:
                    return []

            paper_uuid = paper["id"]

            # Build query
            query = """
                SELECT id, title, section_type, hierarchy_level as level,
                       LEFT(content_text, 500) as content_preview
                FROM sections WHERE paper_id = $1
            """
            params = [paper_uuid]
            param_idx = 2

            if focus:
                query += f" AND (title ILIKE ${param_idx} OR content_text ILIKE ${param_idx})"
                params.append(f"%{focus}%")
                param_idx += 1

            if section_types:
                query += f" AND section_type = ANY(${param_idx})"
                params.append(section_types)
                param_idx += 1

            query += f" ORDER BY hierarchy_level, sequence_order LIMIT ${param_idx}"
            params.append(limit)

            rows = await conn.fetch(query, *params)
            return [dict(r) for r in rows]
        finally:
            await conn.close()

    return run_async(_list())


def search_fulltext_chunks(
    paper_id: str,
    query: str,
    section_ids: list[str] | None = None,
    top_k: int = 10,
) -> list[dict[str, Any]]:
    """Search chunks within a fulltext paper, optionally within specific sections.

    This is for drilling into paper details when abstract isn't enough.

    Args:
        paper_id: PMID or paper UUID
        query: Search query
        section_ids: Optional list of section IDs to search within
        top_k: Maximum chunks to return

    Returns:
        List of dicts with: chunk_id, section_id, content, score, section_title

    Example:
        # Search all chunks in a paper
        chunks = search_fulltext_chunks("12345678", "treatment outcome")

        # Search within specific sections
        sections = list_sections("12345678", focus="results")
        sec_ids = [s['id'] for s in sections]
        chunks = search_fulltext_chunks("12345678", "efficacy", section_ids=sec_ids)
    """
    async def _search():
        chunk_search = _get_chunk_search()

        # Use PMID-based filtering (not paper UUID)
        pmid_filter = {str(paper_id)} if paper_id else None
        chunks, _time_ms = await chunk_search.search(
            query=query,
            pmid_filter=pmid_filter,
            limit=top_k,
        )

        return [
            {
                "chunk_id": getattr(c, "chunk_id", ""),
                "section_id": getattr(c, "section_id", ""),
                "content": getattr(c, "text", ""),
                "score": getattr(c, "score", 0.0),
                "section_title": getattr(c, "section_title", ""),
            }
            for c in chunks
        ]

    return run_async(_search())


def expand_section_evidence(
    paper_id: str,
    seed_section_ids: list[str],
    query: str,
    top_k: int = 10,
    preferred_section_types: list[str] | None = None,
) -> list[dict[str, Any]]:
    """Expand evidence inside one paper from seed sections to better sections.

    Use this when an initial hit lands in introduction/background and you want
    to follow the same topic into results/discussion/conclusion.

    Args:
        paper_id: PMID or paper UUID
        seed_section_ids: Section IDs from the initial retrieval
        query: Search query used for expansion
        top_k: Maximum chunks to return
        preferred_section_types: Optional preferred section roles

    Returns:
        List of chunk dicts with section metadata and adjusted_score
    """
    async def _expand():
        searcher = _get_section_aware_chunk_search()
        chunks = await searcher.expand_from_seed(
            paper_id=paper_id,
            seed_section_ids=seed_section_ids,
            query=query,
            top_k=top_k,
            preferred_section_types=preferred_section_types,
        )
        return [
            {
                "chunk_id": c["chunk_id"],
                "paper_id": c["paper_id"],
                "pmid": c["pmid"],
                "pmcid": c["pmcid"],
                "text": c["text"],
                "section_id": c["section_id"],
                "section_title": c["section_title"],
                "section_type": c["section_type"],
                "section_role": c["section_role"],
                "hierarchy_level": c["hierarchy_level"],
                "score": c["score"],
                "adjusted_score": c["adjusted_score"],
                "rank": c["rank"],
                "retrieval_stage": c["retrieval_stage"],
                "token_count": c["token_count"],
            }
            for c in chunks
        ]

    return run_async(_expand())


def entity_expand(
    text: str,
    entity_types: list[str] | None = None,
) -> list[dict[str, Any]]:
    """Extract biomedical entities from text for targeted search.

    Use this when judge_evidence suggests "needs_entities".

    Args:
        text: Text to extract entities from (question + evidence)
        entity_types: Optional filter for types: ["gene", "disease", "drug", "chemical"]

    Returns:
        List of dicts with: text, type, confidence

    Example:
        entities = entity_expand("Does metformin help with type 2 diabetes?")
        # [{"text": "metformin", "type": "drug", "confidence": 0.95},
        #  {"text": "type 2 diabetes", "type": "disease", "confidence": 0.92}]

        # Use entities for targeted search
        for ent in entities:
            hits = search_papers(ent['text'], limit=20)
    """
    from ..utils.clients import LLMClient
    from ..config import get_config

    cfg = get_config()

    prompt = f"""Extract biomedical entities from this text. Focus on:
- Genes/proteins
- Diseases/conditions
- Drugs/chemicals
- Biological processes

Text: {text[:2000]}

Return JSON array:
[{{"text": "entity name", "type": "gene|disease|drug|chemical|process", "confidence": 0.0-1.0}}]

Only include entities with confidence >= 0.7."""

    llm = LLMClient(base_urls=cfg.llm.servers, model=cfg.llm.model)
    response = run_async(llm.chat(prompt, max_tokens=500))

    import json
    import re

    # Extract JSON array
    json_match = re.search(r'\[.*\]', response, re.DOTALL)
    if json_match:
        try:
            entities = json.loads(json_match.group())
            if entity_types:
                entities = [e for e in entities if e.get("type") in entity_types]
            return entities
        except json.JSONDecodeError:
            pass

    return []


# Import EvidenceState for REPL access
from .evidence_state import EvidenceState, EvidenceSnippet, create_evidence_state


# =============================================================================
# 7. GO TOOLS - Gene Ontology resolution and hierarchy
# =============================================================================

def go_resolve(term: str) -> dict[str, Any] | None:
    """Resolve GO term by ID or name.

    Args:
        term: GO ID (e.g., "GO:0004672") or term name (e.g., "protein kinase activity")

    Returns:
        Dict with: id, name, namespace, definition, alt_ids, is_obsolete, parents, children
        None if not found

    Example:
        term = go_resolve("GO:0004672")
        print(term['name'])  # "protein kinase activity"

        term = go_resolve("apoptosis")
        print(term['id'])  # "GO:0006915"
    """
    async def _resolve():
        resolver = _get_go_resolver()
        # Ensure initialization (fail-fast on error)
        if not resolver._initialized:
            ok = await resolver.initialize()
            if not ok:
                raise RuntimeError("GOResolver initialization failed (OBO file not available)")
        return resolver.resolve(term)

    goterm = run_async(_resolve())
    if goterm is None:
        return None
    return {
        "id": goterm.id,
        "name": goterm.name,
        "namespace": goterm.namespace,
        "definition": goterm.definition,
        "alt_ids": goterm.alt_ids,
        "is_obsolete": goterm.is_obsolete,
        "parents": sorted(goterm.parents),
        "children": sorted(goterm.children),
    }


def go_ancestors(term: str, include_self: bool = False) -> list[str]:
    """Get all ancestor GO terms (more general terms).

    Args:
        term: GO ID or term name
        include_self: Include the term itself in result

    Returns:
        List of ancestor GO IDs, sorted

    Example:
        ancestors = go_ancestors("GO:0004672")  # protein kinase activity
        # Returns ancestors up to root (molecular_function)
    """
    async def _get():
        resolver = _get_go_resolver()
        if not resolver._initialized:
            ok = await resolver.initialize()
            if not ok:
                raise RuntimeError("GOResolver initialization failed")
        return resolver.get_ancestors(term, include_self=include_self)

    return sorted(run_async(_get()))


def go_descendants(term: str, include_self: bool = False) -> list[str]:
    """Get all descendant GO terms (more specific terms).

    Args:
        term: GO ID or term name
        include_self: Include the term itself in result

    Returns:
        List of descendant GO IDs, sorted

    Example:
        descendants = go_descendants("GO:0016301")  # kinase activity
        # Returns all specific kinase types
    """
    async def _get():
        resolver = _get_go_resolver()
        if not resolver._initialized:
            ok = await resolver.initialize()
            if not ok:
                raise RuntimeError("GOResolver initialization failed")
        return resolver.get_descendants(term, include_self=include_self)

    return sorted(run_async(_get()))


def go_related(term: str) -> dict[str, list[str]]:
    """Get related GO terms (parents, children, siblings).

    Args:
        term: GO ID or term name

    Returns:
        Dict with: parents, children, siblings (each a list of GO IDs)

    Example:
        related = go_related("GO:0004672")
        print(related['parents'])   # Direct parent terms
        print(related['children'])  # Direct child terms
        print(related['siblings'])  # Terms with same parents
    """
    async def _get():
        resolver = _get_go_resolver()
        if not resolver._initialized:
            ok = await resolver.initialize()
            if not ok:
                raise RuntimeError("GOResolver initialization failed")
        return resolver.get_related(term)

    result = run_async(_get())
    if not result:
        return {"parents": [], "children": [], "siblings": []}
    return {
        "parents": sorted(result.get("parents", set())),
        "children": sorted(result.get("children", set())),
        "siblings": sorted(result.get("siblings", set())),
    }


def go_search(query: str, limit: int = 10) -> list[dict[str, Any]]:
    """Search GO terms via QuickGO API.

    Use this for fuzzy/partial name matching when exact lookup fails.

    Args:
        query: Search query (partial name or keywords)
        limit: Maximum results

    Returns:
        List of dicts with: id, name, namespace, definition

    Example:
        results = go_search("kinase")
        for r in results[:5]:
            print(f"{r['id']}: {r['name']}")
    """
    async def _search():
        resolver = _get_go_resolver()
        # QuickGO API doesn't require local OBO, but init for consistency
        if not resolver._initialized:
            await resolver.initialize()  # Best effort, API fallback works without OBO
        results = await resolver.search_quickgo(query, limit=limit)
        if results is None:
            raise RuntimeError("QuickGO API search failed")
        return results

    results = run_async(_search())
    return [
        {
            "id": r.id,
            "name": r.name,
            "namespace": r.namespace,
            "definition": r.definition,
        }
        for r in results
    ]


# =============================================================================
# 8. GENE TOOLS - Gene symbol resolution
# =============================================================================

def gene_resolve(query: str) -> dict[str, Any]:
    """Resolve gene name/alias to official HGNC symbol.

    Uses multi-tier strategy:
    1. Local cache (common ORF mappings)
    2. HGNC REST API
    3. MyGene.info API

    Args:
        query: Gene name, alias, or previous symbol

    Returns:
        Dict with: query, symbol (None if not found)

    Example:
        result = gene_resolve("C20orf195")
        print(result['symbol'])  # "FNDC11"

        result = gene_resolve("HER2")
        print(result['symbol'])  # "ERBB2"
    """
    async def _resolve():
        resolver = _get_gene_resolver()
        full_info = await resolver.resolve_full(query)
        if full_info:
            return {"query": query, **full_info}
        # resolve_full already calls resolve() internally, no need to retry
        return {"query": query, "symbol": None}

    return run_async(_resolve())


def gene_resolve_batch(queries: list[str]) -> dict[str, str | None]:
    """Resolve multiple gene names/aliases in batch.

    Args:
        queries: List of gene names to resolve

    Returns:
        Dict mapping query to resolved symbol (or None)

    Example:
        mapping = gene_resolve_batch(["LMP10", "C20orf195", "HER2"])
        # {"LMP10": "PSMB10", "C20orf195": "FNDC11", "HER2": "ERBB2"}
    """
    async def _resolve():
        resolver = _get_gene_resolver()
        return await resolver.resolve_batch(queries)

    return run_async(_resolve())


# =============================================================================
# 8b. GENOMICS TOOLS - NCBI dbSNP + MyGene structured lookups
# =============================================================================

def snp_lookup(rsid: str) -> dict[str, Any]:
    """Look up a dbSNP variant (rs ID) -> associated gene and chromosome.

    Resolves structured genomics facts that are NOT in PubMed abstracts
    (NCBI dbSNP via E-utilities). Use for "gene associated with SNP rsX" and
    "which chromosome is SNP rsX on" questions.

    Args:
        rsid: dbSNP identifier, e.g. "rs1217074595" (the "rs" prefix is optional)

    Returns:
        Dict with: rsid, gene (primary symbol or None), genes (all),
        chromosome (e.g. "chr20" or None), position (e.g. "20:50298395")

    Example:
        snp_lookup("rs1217074595")
        # {"rsid": "rs1217074595", "gene": "LINC01270",
        #  "genes": ["LINC01270"], "chromosome": "chr20", "position": "20:50298395"}
    """
    r = _get_genomics_resolver().snp_lookup(rsid)
    return {
        "rsid": r.rsid,
        "gene": r.gene,
        "genes": r.genes,
        "chromosome": (f"chr{r.chromosome}" if r.chromosome else None),
        "position": r.position,
    }


def gene_genomic_info(symbol: str) -> dict[str, Any]:
    """Gene symbol -> chromosome and protein-coding status (MyGene.info).

    Use for "which chromosome is GENE on" and "does GENE code a protein"
    questions. ``protein_coding_answer`` is pre-formatted as "TRUE"/"FALSE".

    Args:
        symbol: Official gene symbol, e.g. "NODAL"

    Returns:
        Dict with: symbol, chromosome (e.g. "chr10" or None), type_of_gene
        (e.g. "protein-coding", "ncRNA"), is_protein_coding (bool or None),
        protein_coding_answer ("TRUE"/"FALSE"/None)

    Example:
        gene_genomic_info("NODAL")
        # {"symbol": "NODAL", "chromosome": "chr10", "type_of_gene": "protein-coding",
        #  "is_protein_coding": True, "protein_coding_answer": "TRUE"}
    """
    r = _get_genomics_resolver().gene_info(symbol)
    pc = (None if r.is_protein_coding is None
          else ("TRUE" if r.is_protein_coding else "FALSE"))
    return {
        "symbol": r.symbol,
        "chromosome": (f"chr{r.chromosome}" if r.chromosome else None),
        "type_of_gene": r.type_of_gene,
        "is_protein_coding": r.is_protein_coding,
        "protein_coding_answer": pc,
    }


def blast_dna_lookup(sequence: str) -> dict[str, Any]:
    """Align a DNA sequence -> human-genome coordinates or source organism.

    Reads a precomputed NCBI BLAST cache (built offline by
    ``scripts/build_blast_cache.py``; BLAST is too slow for live calls). Covers
    the GeneTuring DNA-alignment subtasks.

    Args:
        sequence: a raw DNA sequence (A/C/G/T...)

    Returns:
        Dict with: sequence_sha, mode ("genome"/"organism"/None), answer
        (e.g. "chr15:91950805-91950932" or "worm", or None if not cached)

    Example:
        blast_dna_lookup("ATTCTGCC...")  # {"mode": "genome", "answer": "chr15:91950805-91950932"}
    """
    import hashlib
    import json as _json
    from pathlib import Path as _Path
    seq = (sequence or "").strip()
    h = hashlib.sha1(seq.encode()).hexdigest()[:16]
    out = {"sequence_sha": h, "mode": None, "answer": None}
    try:
        cache_path = _Path(__file__).resolve().parents[2] / ".cache/blast/geneturing.json"
        hit = _json.loads(cache_path.read_text()).get(h)
        if hit and hit.get("answer"):
            out["mode"] = hit.get("mode")
            out["answer"] = hit.get("answer")
    except Exception:
        pass
    return out


# =============================================================================
# 9. WEB SEARCH TOOLS - External medical literature search
# =============================================================================

def web_search(
    query: str,
    limit: int = 10,
    freshness_days: int | None = None,
    medical_only: bool = False,
) -> list[dict[str, Any]]:
    """Search the web via Serper (Google) API.

    Useful for finding recent guidelines, news, and sources not in PubMed.
    Results are filtered to exclude low-quality sources (content farms, social media).

    Args:
        query: Search query string
        limit: Maximum results to return (default 10)
        freshness_days: Restrict to recent results (1=day, 7=week, 30=month)
        medical_only: If True, only return results from medical authority domains

    Returns:
        List of dicts with: title, url, snippet, date, domain, position, is_guideline, is_medical_source

    Raises:
        RuntimeError: If SERPER_API_KEY not set or API call fails

    Example:
        # General medical search
        results = web_search("metformin diabetes guidelines 2024")

        # Recent results only
        results = web_search("COVID treatment", freshness_days=30)

        # Only medical sources (NIH, WHO, major journals)
        results = web_search("cancer immunotherapy", medical_only=True)

        for r in results[:5]:
            print(f"{r['title'][:50]}... ({r['domain']})")
            if r['is_guideline']:
                print("  ^ CLINICAL GUIDELINE")
    """
    async def _search():
        searcher = _get_web_searcher()
        return await searcher.search(
            query=query,
            limit=limit,
            freshness_days=freshness_days,
            filter_to_allowlist=medical_only,
        )

    return run_async(_search())


def web_search_medical(
    query: str,
    limit: int = 10,
    freshness_days: int | None = None,
) -> list[dict[str, Any]]:
    """Search the web restricted to medical authority domains only.

    Returns results only from trusted sources:
    - Government: NIH, CDC, FDA, WHO
    - Journals: NEJM, JAMA, Lancet, BMJ, Nature, Cell
    - Databases: PubMed, Cochrane, UpToDate

    Args:
        query: Search query string
        limit: Maximum results to return
        freshness_days: Restrict to recent results

    Returns:
        List of search results (same schema as web_search)

    Example:
        # Find official guidelines
        results = web_search_medical("diabetes treatment guidelines")

        # Recent clinical updates
        results = web_search_medical("SGLT2 inhibitors heart failure", freshness_days=30)
    """
    return web_search(query, limit=limit, freshness_days=freshness_days, medical_only=True)


# =============================================================================
# 10. CHEMBL TOOLS - Drug information lookup
# =============================================================================

def chembl_drug_lookup(query: str, limit: int = 5) -> list[dict[str, Any]]:
    """Look up drug by name, ChEMBL ID, or synonym.

    Uses ChEMBL database for comprehensive drug information.

    Args:
        query: Drug name (e.g., "aspirin"), ChEMBL ID (e.g., "CHEMBL25"), or synonym
        limit: Maximum results to return

    Returns:
        List of dicts with: chembl_id, pref_name, molecule_type, max_phase,
        first_approval, oral, synonyms

    Example:
        # Look up by name
        drugs = chembl_drug_lookup("metformin")
        print(drugs[0]['chembl_id'])  # "CHEMBL1431"

        # Look up by ChEMBL ID
        drugs = chembl_drug_lookup("CHEMBL25")  # aspirin
        print(drugs[0]['pref_name'])  # "ASPIRIN"

        # max_phase: 4 = approved drug
    """
    resolver = _get_chembl_resolver()
    hits = resolver.lookup_drug(query, limit=limit)
    return [
        {
            "chembl_id": h.chembl_id,
            "pref_name": h.pref_name,
            "molecule_type": h.molecule_type,
            "max_phase": h.max_phase,
            "first_approval": h.first_approval,
            "oral": h.oral,
            "synonyms": h.synonyms,
        }
        for h in hits
    ]


def chembl_mechanism(chembl_id: str) -> list[dict[str, Any]]:
    """Get mechanism of action for a drug.

    Args:
        chembl_id: ChEMBL ID (e.g., "CHEMBL25" for aspirin)

    Returns:
        List of dicts with: mechanism, action_type, target_chembl_id,
        target_name, binding_site_name, refs

    Example:
        mechs = chembl_mechanism("CHEMBL25")
        for m in mechs:
            print(f"{m['action_type']}: {m['mechanism']}")
            print(f"  Target: {m['target_name']}")
    """
    resolver = _get_chembl_resolver()
    mechs = resolver.get_mechanisms(chembl_id)
    return [
        {
            "mechanism": m.mechanism,
            "action_type": m.action_type,
            "target_chembl_id": m.target_chembl_id,
            "target_name": m.target_name,
            "binding_site_name": m.binding_site_name,
            "refs": m.refs,
        }
        for m in mechs
    ]


def chembl_target(target_chembl_id: str) -> dict[str, Any] | None:
    """Get target information from ChEMBL.

    Args:
        target_chembl_id: ChEMBL target ID (e.g., "CHEMBL220" for COX-1)

    Returns:
        Dict with: target_chembl_id, pref_name, target_type, organism, components
        None if not found

    Example:
        target = chembl_target("CHEMBL220")
        print(target['pref_name'])  # "Cyclooxygenase-1"
        print(target['organism'])   # "Homo sapiens"
    """
    resolver = _get_chembl_resolver()
    target = resolver.get_target(target_chembl_id)
    if target is None:
        return None
    return {
        "target_chembl_id": target.target_chembl_id,
        "pref_name": target.pref_name,
        "target_type": target.target_type,
        "organism": target.organism,
        "components": target.components,
    }


def chembl_indications(chembl_id: str) -> list[dict[str, Any]]:
    """Get drug indications (approved uses).

    Args:
        chembl_id: ChEMBL ID of the drug

    Returns:
        List of dicts with: mesh_id, mesh_heading, efo_id, efo_term, max_phase_for_ind

    Example:
        inds = chembl_indications("CHEMBL1431")  # metformin
        for ind in inds:
            print(f"{ind['mesh_heading']} (phase {ind['max_phase_for_ind']})")
    """
    resolver = _get_chembl_resolver()
    inds = resolver.get_indications(chembl_id)
    return [
        {
            "mesh_id": i.mesh_id,
            "mesh_heading": i.mesh_heading,
            "efo_id": i.efo_id,
            "efo_term": i.efo_term,
            "max_phase_for_ind": i.max_phase_for_ind,
        }
        for i in inds
    ]


# =============================================================================
# 11. UNIPROT TOOLS - Protein information lookup
# =============================================================================

def uniprot_search(
    query: str,
    organism_id: int = 9606,
    limit: int = 5,
) -> list[dict[str, Any]]:
    """Search proteins by gene name, protein name, or UniProt accession.

    Args:
        query: Gene name (e.g., "BRCA1"), protein name, or accession (e.g., "P38398")
        organism_id: Organism taxon ID (default 9606 = Homo sapiens)
        limit: Maximum results to return

    Returns:
        List of dicts with: accession, protein_name, gene_names, organism_id, organism_name

    Example:
        # Search by gene name
        proteins = uniprot_search("BRCA1")
        print(proteins[0]['accession'])  # "P38398"

        # Search by protein name
        proteins = uniprot_search("insulin")
    """
    async def _search():
        resolver = _get_uniprot_resolver()
        hits = await resolver.search_proteins(query, organism_id=organism_id, limit=limit)
        return [
            {
                "accession": h.accession,
                "protein_name": h.protein_name,
                "gene_names": h.gene_names,
                "organism_id": h.organism_id,
                "organism_name": h.organism_name,
            }
            for h in hits
        ]

    return run_async(_search())


def uniprot_get_protein(accession: str) -> dict[str, Any] | None:
    """Get full protein record by UniProt accession.

    Args:
        accession: UniProt accession (e.g., "P38398")

    Returns:
        Dict with: accession, protein_name, gene_names, organism_id, organism_name,
        function, diseases, go_annotations
        None if not found

    Example:
        protein = uniprot_get_protein("P38398")  # BRCA1
        print(protein['protein_name'])
        print(protein['function']['text'])
    """
    async def _get():
        resolver = _get_uniprot_resolver()
        record = await resolver.get_protein(accession)
        if record is None:
            return None
        return {
            "accession": record.accession,
            "protein_name": record.protein_name,
            "gene_names": record.gene_names,
            "organism_id": record.organism_id,
            "organism_name": record.organism_name,
            "function": {
                "text": record.function.text,
                "evidences": record.function.evidences,
            } if record.function else None,
            "diseases": [
                {
                    "name": d.name,
                    "acronym": d.acronym,
                    "description": d.description,
                    "xrefs": d.xrefs,
                }
                for d in record.diseases
            ],
            "go_annotations": [
                {
                    "go_id": g.go_id,
                    "term": g.term,
                    "aspect": g.aspect,
                    "evidence": g.evidence,
                }
                for g in record.go_annotations
            ],
        }

    return run_async(_get())


def uniprot_function(accession: str) -> dict[str, Any] | None:
    """Get protein function annotation.

    Args:
        accession: UniProt accession

    Returns:
        Dict with: text, evidences
        None if not found

    Example:
        func = uniprot_function("P38398")
        print(func['text'][:200])
    """
    async def _get():
        resolver = _get_uniprot_resolver()
        func = await resolver.get_protein_function(accession)
        if func is None:
            return None
        return {
            "text": func.text,
            "evidences": func.evidences,
        }

    return run_async(_get())


def uniprot_diseases(accession: str) -> list[dict[str, Any]]:
    """Get protein disease associations.

    Args:
        accession: UniProt accession

    Returns:
        List of dicts with: name, acronym, description, xrefs

    Example:
        diseases = uniprot_diseases("P38398")  # BRCA1
        for d in diseases:
            print(f"{d['name']}: {d['description'][:100]}...")
    """
    async def _get():
        resolver = _get_uniprot_resolver()
        diseases = await resolver.get_protein_diseases(accession)
        return [
            {
                "name": d.name,
                "acronym": d.acronym,
                "description": d.description,
                "xrefs": d.xrefs,
            }
            for d in diseases
        ]

    return run_async(_get())


def uniprot_go(accession: str) -> list[dict[str, Any]]:
    """Get protein GO annotations.

    Args:
        accession: UniProt accession

    Returns:
        List of dicts with: go_id, term, aspect (C/F/P), evidence

    Example:
        go_terms = uniprot_go("P38398")  # BRCA1
        for g in go_terms[:5]:
            print(f"{g['go_id']}: {g['term']} ({g['aspect']})")
    """
    async def _get():
        resolver = _get_uniprot_resolver()
        annotations = await resolver.get_protein_go(accession)
        return [
            {
                "go_id": g.go_id,
                "term": g.term,
                "aspect": g.aspect,
                "evidence": g.evidence,
            }
            for g in annotations
        ]

    return run_async(_get())


# =============================================================================
# 12. CLINICALTRIALS TOOLS - Clinical trial search
# =============================================================================

def clinicaltrials_search(
    condition: str | None = None,
    intervention: str | None = None,
    keyword: str | None = None,
    limit: int = 10,
) -> list[dict[str, Any]]:
    """Search clinical trials by condition, intervention, or keyword.

    Args:
        condition: Disease/condition (e.g., "diabetes")
        intervention: Drug/treatment (e.g., "metformin")
        keyword: General search term
        limit: Maximum results to return

    Returns:
        List of dicts with: nct_id, title, status, phases, enrollment, conditions, interventions

    Example:
        # Search by condition
        trials = clinicaltrials_search(condition="breast cancer", limit=5)
        for t in trials:
            print(f"{t['nct_id']}: {t['title'][:50]}... (status: {t['status']})")

        # Search by intervention
        trials = clinicaltrials_search(intervention="metformin", condition="diabetes")
    """
    async def _search():
        resolver = _get_clinicaltrials_resolver()
        results = await resolver.search_studies(
            condition=condition,
            intervention=intervention,
            keyword=keyword,
            page_size=limit,
        )
        return [
            {
                "nct_id": t.nct_id,
                "title": t.title,
                "status": t.status,
                "phases": t.phases,
                "enrollment": t.enrollment,
                "conditions": t.conditions,
                "interventions": t.interventions,
            }
            for t in results
        ]

    return run_async(_search())


def clinicaltrials_get_study(nct_id: str) -> dict[str, Any] | None:
    """Get full clinical trial details by NCT ID.

    Args:
        nct_id: ClinicalTrials.gov NCT ID (e.g., "NCT00000102")

    Returns:
        Dict with: nct_id, title, status, phases, enrollment, start_date, completion_date,
        conditions, interventions, eligibility, outcomes, sponsor
        None if not found

    Example:
        study = clinicaltrials_get_study("NCT00000102")
        if study:
            print(study['title'])
            print(f"Status: {study['status']}, Phases: {study['phases']}")
    """
    async def _get():
        resolver = _get_clinicaltrials_resolver()
        study = await resolver.get_study(nct_id)
        if study is None:
            return None
        return {
            "nct_id": study.nct_id,
            "title": study.title,
            "status": study.status,
            "phases": study.phases,
            "enrollment": study.enrollment,
            "start_date": study.start_date,
            "completion_date": study.completion_date,
            "conditions": study.conditions,
            "interventions": study.interventions,
            "eligibility": {
                "criteria_text": study.eligibility.criteria_text,
                "sex": study.eligibility.sex,
                "minimum_age": study.eligibility.minimum_age,
                "maximum_age": study.eligibility.maximum_age,
                "healthy_volunteers": study.eligibility.healthy_volunteers,
            } if study.eligibility else None,
            "outcomes": [
                {
                    "type": o.type,
                    "measure": o.measure,
                    "time_frame": o.time_frame,
                    "description": o.description,
                }
                for o in study.outcomes
            ],
            "sponsor": study.sponsor,
        }

    return run_async(_get())


def clinicaltrials_eligibility(nct_id: str) -> dict[str, Any] | None:
    """Get eligibility criteria for a clinical trial.

    Args:
        nct_id: NCT ID

    Returns:
        Dict with: criteria_text, sex, minimum_age, maximum_age, healthy_volunteers
        None if not found

    Example:
        elig = clinicaltrials_eligibility("NCT00000102")
        if elig:
            print(f"Age: {elig['minimum_age']} - {elig['maximum_age']}")
    """
    async def _get():
        resolver = _get_clinicaltrials_resolver()
        elig = await resolver.get_study_eligibility(nct_id)
        if elig is None:
            return None
        return {
            "criteria_text": elig.criteria_text,
            "sex": elig.sex,
            "minimum_age": elig.minimum_age,
            "maximum_age": elig.maximum_age,
            "healthy_volunteers": elig.healthy_volunteers,
        }

    return run_async(_get())


def clinicaltrials_outcomes(nct_id: str) -> list[dict[str, Any]]:
    """Get primary and secondary outcomes for a clinical trial.

    Args:
        nct_id: NCT ID

    Returns:
        List of dicts with: type (primary/secondary), measure, time_frame, description

    Example:
        outcomes = clinicaltrials_outcomes("NCT00000102")
        for o in outcomes:
            print(f"[{o['type']}] {o['measure']}")
    """
    async def _get():
        resolver = _get_clinicaltrials_resolver()
        outcomes = await resolver.get_study_outcomes(nct_id)
        return [
            {
                "type": o.type,
                "measure": o.measure,
                "time_frame": o.time_frame,
                "description": o.description,
            }
            for o in outcomes
        ]

    return run_async(_get())


RETRIEVAL_TOOLS = {
    # Search
    "search_papers": search_papers,
    "search_chunks": search_chunks,
    "search_section_aware_chunks": search_section_aware_chunks,
    "search_by_entities": search_by_entities,

    # Filter
    "search_mesh_terms": search_mesh_terms,
    "get_pmids_by_mesh": get_pmids_by_mesh,
    "keyword_search": keyword_search,

    # Ranking
    "rerank_papers": rerank_papers,
    "rerank_chunks": rerank_chunks,

    # Structure
    "get_paper_abstracts": get_paper_abstracts,
    "get_paper_sections": get_paper_sections,

    # Hybrid/Advanced
    "hybrid_search": hybrid_search,
    "iterative_search": iterative_search,

    # Helpers
    "format_papers": format_papers,
    "format_chunks": format_chunks,

    # Adaptive Evidence Collection
    "judge_evidence": judge_evidence,
    "get_paper_meta": get_paper_meta,
    "list_sections": list_sections,
    "search_fulltext_chunks": search_fulltext_chunks,
    "expand_section_evidence": expand_section_evidence,
    "entity_expand": entity_expand,

    # State Management
    "EvidenceState": EvidenceState,
    "EvidenceSnippet": EvidenceSnippet,
    "create_evidence_state": create_evidence_state,

    # GO Tools (Gene Ontology)
    "go_resolve": go_resolve,
    "go_ancestors": go_ancestors,
    "go_descendants": go_descendants,
    "go_related": go_related,
    "go_search": go_search,

    # Gene Tools (Symbol Resolution)
    "gene_resolve": gene_resolve,
    "gene_resolve_batch": gene_resolve_batch,

    # Genomics Tools (NCBI dbSNP + MyGene: SNP/gene -> chromosome, protein-coding)
    "snp_lookup": snp_lookup,
    "gene_genomic_info": gene_genomic_info,
    "blast_dna_lookup": blast_dna_lookup,

    # Web Search Tools
    "web_search": web_search,
    "web_search_medical": web_search_medical,

    # ChEMBL Tools (Drug Information)
    "chembl_drug_lookup": chembl_drug_lookup,
    "chembl_mechanism": chembl_mechanism,
    "chembl_target": chembl_target,
    "chembl_indications": chembl_indications,

    # UniProt Tools (Protein Information)
    "uniprot_search": uniprot_search,
    "uniprot_get_protein": uniprot_get_protein,
    "uniprot_function": uniprot_function,
    "uniprot_diseases": uniprot_diseases,
    "uniprot_go": uniprot_go,

    # ClinicalTrials Tools (Clinical Trial Search)
    "clinicaltrials_search": clinicaltrials_search,
    "clinicaltrials_get_study": clinicaltrials_get_study,
    "clinicaltrials_eligibility": clinicaltrials_eligibility,
    "clinicaltrials_outcomes": clinicaltrials_outcomes,

    # Lifecycle
    "shutdown_tools": shutdown_tools,

    # Tracing utilities (exposed for experiments)
    "enable_tool_trace": enable_tool_trace,
    "disable_tool_trace": disable_tool_trace,
    "get_tool_trace": get_tool_trace,
    "clear_tool_trace": clear_tool_trace,
}


# Apply tracing to key external tools
_TRACED_TOOLS = {
    "chembl_drug_lookup", "chembl_mechanism", "chembl_target", "chembl_indications",
    "uniprot_search", "uniprot_get_protein", "uniprot_function", "uniprot_diseases", "uniprot_go",
    "clinicaltrials_search", "clinicaltrials_get_study", "clinicaltrials_eligibility", "clinicaltrials_outcomes",
    "web_search", "web_search_medical",
    "go_resolve", "go_ancestors", "go_descendants", "go_search",
    "gene_resolve", "gene_resolve_batch",
}

for _tool_name in _TRACED_TOOLS:
    if _tool_name in RETRIEVAL_TOOLS:
        _original = RETRIEVAL_TOOLS[_tool_name]
        RETRIEVAL_TOOLS[_tool_name] = traced_tool(_tool_name)(_original)


# =============================================================================
# TOOL REGISTRY INTEGRATION
# =============================================================================
