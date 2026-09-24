"""Knowledge Graph tools for BiomedicalREPL.

RLM Philosophy:
- KG tools provide optional graph-based retrieval capabilities
- Tools wrap our native implementations in src/kg/
- Each tool returns structured data that RLM can process in code

Supports three styles:
- LightRAG: Entity/relation focused hybrid search
- PathRAG: Graph path based retrieval
- GraphRAG: Community report aggregation (MS-style)

DESIGN: Fail-Fast
- All errors propagate immediately
- No fallbacks or mock data
"""

from typing import Any, Literal

try:
    from kg import LightRAGClient, PathRAGClient, GraphRAGClient
    from utils.async_helper import run_async
except ImportError:
    from src.kg import LightRAGClient, PathRAGClient, GraphRAGClient
    from src.utils.async_helper import run_async

# Global client instances (lazy initialization)
_lightrag_client: LightRAGClient | None = None
_pathrag_client: PathRAGClient | None = None
_graphrag_client: GraphRAGClient | None = None


def _get_lightrag_client() -> LightRAGClient:
    """Get or create LightRAG client singleton."""
    global _lightrag_client
    if _lightrag_client is None:
        _lightrag_client = LightRAGClient(
            storage_dir="data/kg/lightrag",
            namespace="kg_lightrag",
        )
    return _lightrag_client


def _get_pathrag_client() -> PathRAGClient:
    """Get or create PathRAG client singleton."""
    global _pathrag_client
    if _pathrag_client is None:
        _pathrag_client = PathRAGClient(
            storage_dir="data/kg/pathrag",
            namespace="kg_pathrag",
        )
    return _pathrag_client


def _get_graphrag_client() -> GraphRAGClient:
    """Get or create GraphRAG client singleton."""
    global _graphrag_client
    if _graphrag_client is None:
        _graphrag_client = GraphRAGClient(
            storage_dir="data/kg/graphrag",
            namespace="kg_graphrag",
        )
    return _graphrag_client


# =============================================================================
# LightRAG Style Tools
# =============================================================================

def kg_lightrag_query(
    query: str,
    mode: Literal["local", "global", "hybrid", "naive"] = "hybrid",
    top_k: int = 20,
) -> dict[str, Any]:
    """Query using LightRAG-style entity/relation retrieval.

    LightRAG modes:
    - local: Entity-centric search (find specific entities)
    - global: Relation-centric, high-level knowledge aggregation
    - hybrid: Combines local + global (recommended)
    - naive: Simple vector search (baseline)

    Args:
        query: Natural language query
        mode: Search mode
        top_k: Number of results

    Returns:
        Dict with: answer, mode, query, entities, relations

    Example:
        result = kg_lightrag_query("How does BRCA1 affect cancer?", mode="hybrid")
        print(result['answer'])
    """
    async def _query():
        client = _get_lightrag_client()

        # Try to load existing KG, fail silently if not available
        try:
            await client.load()
        except (FileNotFoundError, Exception):
            raise RuntimeError(
                "LightRAG KG not indexed. Run indexing first with client.index(chunks)"
            )

        result = await client.query(query, mode=mode, top_k=top_k)

        return {
            "answer": result.answer,
            "mode": mode,
            "query": query,
            "entities": [e.name for e in result.entities[:10]],
            "relations": [
                f"{r.source} --[{r.type}]--> {r.target}"
                for r in result.relations[:10]
            ],
            "latency_ms": result.latency_ms,
        }

    return run_async(_query())


def kg_lightrag_get_data(
    query: str,
    mode: Literal["local", "global", "hybrid"] = "hybrid",
) -> dict[str, Any]:
    """Get raw retrieval data from LightRAG without LLM generation.

    Useful for inspecting what entities/relations are retrieved.

    Args:
        query: Natural language query
        mode: Search mode

    Returns:
        Dict with: entities, relations (raw data)
    """
    async def _get_data():
        client = _get_lightrag_client()

        try:
            await client.load()
        except (FileNotFoundError, Exception):
            raise RuntimeError("LightRAG KG not indexed")

        result = await client.query(query, mode=mode, top_k=30)

        return {
            "entities": [
                {
                    "name": e.name,
                    "type": e.type,
                    "description": e.description[:200],
                }
                for e in result.entities
            ],
            "relations": [
                {
                    "source": r.source,
                    "target": r.target,
                    "type": r.type,
                    "description": r.description[:200],
                }
                for r in result.relations
            ],
            "mode": mode,
        }

    return run_async(_get_data())


# =============================================================================
# PathRAG Style Tools
# =============================================================================

def kg_pathrag_query(
    query: str,
    top_k: int = 40,
    path_depth: int = 3,
) -> dict[str, Any]:
    """Query using PathRAG-style graph path retrieval.

    PathRAG finds relevant paths in the knowledge graph connecting
    entities mentioned in the query.

    Args:
        query: Natural language query
        top_k: Number of paths to retrieve
        path_depth: Maximum path length in graph hops

    Returns:
        Dict with: answer, top_k, path_depth, query, paths

    Example:
        result = kg_pathrag_query("Relationship between BRCA1 and breast cancer")
        print(result['answer'])
    """
    async def _query():
        client = _get_pathrag_client()

        try:
            await client.load()
        except (FileNotFoundError, Exception):
            raise RuntimeError("PathRAG KG not indexed")

        result = await client.query(query, top_k=top_k, max_hops=path_depth)

        return {
            "answer": result.answer,
            "top_k": top_k,
            "path_depth": path_depth,
            "query": query,
            "paths": [p.to_string() for p in result.paths[:10]],
            "num_paths_found": len(result.paths),
            "latency_ms": result.latency_ms,
        }

    return run_async(_query())


# =============================================================================
# MS-GraphRAG Style Tools
# =============================================================================

def kg_graphrag_query(
    query: str,
    search_type: Literal["local", "global"] = "local",
    community_level: int = 1,
) -> dict[str, Any]:
    """Query using MS-GraphRAG style community-based retrieval.

    GraphRAG modes:
    - local: Detailed entity/relationship search
    - global: High-level community report aggregation

    Args:
        query: Natural language query
        search_type: "local" for entity details, "global" for summaries
        community_level: Community hierarchy level (0=coarsest)

    Returns:
        Dict with: answer, search_type, query, communities

    Example:
        result = kg_graphrag_query("Overview of cancer treatment", search_type="global")
        print(result['answer'])
    """
    async def _query():
        client = _get_graphrag_client()

        try:
            await client.load()
        except (FileNotFoundError, Exception):
            raise RuntimeError("GraphRAG KG not indexed")

        result = await client.query(
            query,
            mode=search_type,
            community_level=community_level,
        )

        return {
            "answer": result.answer,
            "search_type": search_type,
            "query": query,
            "communities": [
                {
                    "title": c.title,
                    "size": len(c.member_entities),
                    "summary": c.summary[:200] if c.summary else "",
                }
                for c in result.communities[:5]
            ],
            "num_entities": len(result.entities),
            "latency_ms": result.latency_ms,
        }

    return run_async(_query())


# =============================================================================
# Unified KG Interface
# =============================================================================

def kg_query(
    query: str,
    style: Literal["lightrag", "pathrag", "graphrag"] = "lightrag",
    **kwargs,
) -> dict[str, Any]:
    """Unified KG query interface.

    Dispatches to the appropriate KG implementation based on style.

    Args:
        query: Natural language query
        style: KG implementation to use
        **kwargs: Additional arguments passed to the specific implementation

    Returns:
        Dict with query results
    """
    if style == "lightrag":
        return kg_lightrag_query(query, **kwargs)
    elif style == "pathrag":
        return kg_pathrag_query(query, **kwargs)
    elif style == "graphrag":
        return kg_graphrag_query(query, **kwargs)
    else:
        raise ValueError(f"Unknown KG style: {style}")


def kg_get_stats(style: Literal["lightrag", "pathrag", "graphrag"] = "lightrag") -> dict[str, Any]:
    """Get statistics for a KG implementation.

    Args:
        style: Which KG implementation to get stats for

    Returns:
        Dict with entity/relation/community counts
    """
    if style == "lightrag":
        client = _get_lightrag_client()
    elif style == "pathrag":
        client = _get_pathrag_client()
    elif style == "graphrag":
        client = _get_graphrag_client()
    else:
        raise ValueError(f"Unknown KG style: {style}")

    stats = client.get_stats()
    return {
        "num_entities": stats.num_entities,
        "num_relations": stats.num_relations,
        "num_communities": stats.num_communities,
        "num_chunks": stats.num_chunks,
        "entity_types": stats.entity_types,
        "relation_types": stats.relation_types,
    }


# =============================================================================
# Tool Registry
# =============================================================================

KG_TOOLS = {
    "kg_lightrag_query": kg_lightrag_query,
    "kg_lightrag_get_data": kg_lightrag_get_data,
    "kg_pathrag_query": kg_pathrag_query,
    "kg_graphrag_query": kg_graphrag_query,
    "kg_query": kg_query,
    "kg_get_stats": kg_get_stats,
}
