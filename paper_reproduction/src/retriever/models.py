"""Type definitions for the PubMed Abstract Retriever.

This module defines all data classes and enums used by the retriever system.
Configuration defaults are loaded from src/config.py for centralized management.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

from config import get_config as _get_config

# Get config for defaults
_cfg = _get_config()


class FilterMode(str, Enum):
    """Filter combination modes for PMID filtering.

    Controls how MeSH and keyword filters are combined:
    - NONE: No filtering, vanilla vector search on all 27.3M papers
    - MESH_ONLY: Only MeSH-based filtering (ablation)
    - KEYWORD_ONLY: Only keyword-based FTS filtering (ablation)
    - MESH_OR_KEYWORD: Union of MeSH and keyword PMIDs (default, broader coverage)
    - MESH_AND_KEYWORD: Intersection of MeSH and keyword PMIDs (higher precision)
    """
    NONE = "none"
    MESH_ONLY = "mesh"
    KEYWORD_ONLY = "keyword"
    MESH_OR_KEYWORD = "mesh_or_keyword"
    MESH_AND_KEYWORD = "mesh_and_keyword"


@dataclass
class ParsedQuery:
    """Result of LLM-based query parsing.

    Attributes:
        original_query: The original query string
        entities: Extracted biomedical entities (drugs, diseases, genes, etc.)
        keywords: Keywords for full-text search
        mesh_candidates: Suggested MeSH descriptors from the query
        parse_time_ms: Time taken for parsing in milliseconds
    """
    original_query: str
    entities: list[str] = field(default_factory=list)
    keywords: list[str] = field(default_factory=list)
    mesh_candidates: list[str] = field(default_factory=list)
    parse_time_ms: float = 0.0


@dataclass
class MeSHMatch:
    """A matched MeSH term from vector search.

    Attributes:
        descriptor_ui: MeSH Descriptor Unique Identifier (e.g., "D008687")
                      This is the indexed column in mesh_headings table!
        descriptor_name: Human-readable MeSH term (e.g., "Metformin")
        score: Similarity score from vector search (0-1)
        tree_numbers: Hierarchical tree numbers for expansion
    """
    descriptor_ui: str
    descriptor_name: str
    score: float = 0.0
    tree_numbers: list[str] = field(default_factory=list)


@dataclass
class RetrievedAbstract:
    """A retrieved PubMed abstract with metadata.

    Attributes:
        pmid: PubMed ID (primary key in articles table)
        title: Article title
        abstract: Article abstract text
        score: Vector similarity score
        rank: Position in results (1-indexed)
        have_fulltext: Whether PMC fulltext is available
        paper_uuid: UUID linking to papergraph.papers.id (if exists)
        pmc: PMC ID if available (e.g., "PMC1234567")
        matched_mesh: List of MeSH terms that matched this paper
    """
    pmid: str
    title: str
    abstract: str
    score: float
    rank: int
    have_fulltext: bool = False
    paper_uuid: Optional[str] = None
    pmc: Optional[str] = None
    matched_mesh: list[str] = field(default_factory=list)


@dataclass
class RetrievedChunk:
    """A retrieved text chunk from PMC full-text papers.

    Attributes:
        chunk_id: Unique chunk identifier (UUID)
        pmid: PubMed ID of the source paper
        pmcid: PMC ID of the source paper
        text: Chunk text content
        section_id: UUID of the parent section
        score: Vector similarity score
        rank: Position in results (1-indexed)
        token_count: Number of tokens in the chunk
        sequence_order: Position within the section
    """
    chunk_id: str
    pmid: str
    pmcid: str
    text: str
    section_id: str
    score: float
    rank: int
    token_count: int = 0
    sequence_order: int = 0


@dataclass
class FilterStats:
    """Statistics from the filtering step.

    Attributes:
        mesh_pmid_count: Number of PMIDs from MeSH filter
        keyword_pmid_count: Number of PMIDs from keyword filter
        combined_pmid_count: Number of PMIDs after combining filters
        gt_in_mesh: Whether GT PMID is in MeSH results (for tracking)
        gt_in_keyword: Whether GT PMID is in keyword results (for tracking)
        gt_in_combined: Whether GT PMID is in combined results (for tracking)
    """
    mesh_pmid_count: int = 0
    keyword_pmid_count: int = 0
    combined_pmid_count: int = 0
    gt_in_mesh: bool = False
    gt_in_keyword: bool = False
    gt_in_combined: bool = False


@dataclass
class RetrievalResult:
    """Complete retrieval result with metadata and GT tracking.

    Attributes:
        query: Original query string
        abstracts: List of retrieved abstracts (ordered by rank)
        parsed_query: Result of query parsing
        mesh_matches: Matched MeSH terms used for filtering
        filter_stats: Statistics from filtering step
        filter_mode: The filter mode used

        # Timing
        total_time_ms: Total retrieval time
        parse_time_ms: Query parsing time
        mesh_time_ms: MeSH matching and filtering time
        keyword_time_ms: Keyword filtering time
        vector_time_ms: Vector search time

        # Ground Truth tracking
        gt_pmids: List of ground truth PMIDs (for evaluation)
        gt_ranks: Mapping of GT PMID to its rank (-1 if not found)
    """
    query: str
    abstracts: list[RetrievedAbstract] = field(default_factory=list)
    parsed_query: Optional[ParsedQuery] = None
    mesh_matches: list[MeSHMatch] = field(default_factory=list)
    filter_stats: Optional[FilterStats] = None
    filter_mode: FilterMode = FilterMode.MESH_OR_KEYWORD

    # Timing
    total_time_ms: float = 0.0
    parse_time_ms: float = 0.0
    mesh_time_ms: float = 0.0
    keyword_time_ms: float = 0.0
    vector_time_ms: float = 0.0

    # Ground Truth tracking
    gt_pmids: list[str] = field(default_factory=list)
    gt_ranks: dict[str, int] = field(default_factory=dict)


@dataclass
class RetrieverConfig:
    """Configuration for the PubMed Retriever.

    Attributes:
        # Feature toggles
        enable_mesh_filter: Enable MeSH-based filtering
        enable_keyword_filter: Enable keyword-based FTS filtering
        filter_mode: How to combine MeSH and keyword filters

        # MeSH matching parameters
        mesh_top_k: Number of MeSH candidates from vector search
        mesh_llm_select: Max MeSH terms after LLM refinement
        mesh_pmid_limit: Max PMIDs to retrieve from MeSH filter

        # Keyword filter parameters
        keyword_limit: Max PMIDs to retrieve from keyword filter

        # Vector search parameters
        vector_limit: Max results from initial vector search
        final_limit: Final number of abstracts to return

        # Database configuration
        pubmed_db_url: PostgreSQL connection URL for paper-graph-pubmed
        qdrant_url: Qdrant server URL
        mesh_collection: Qdrant collection for MeSH terms
        paper_collection: Qdrant collection for paper embeddings

        # LLM configuration
        llm_base_url: Base URL for LLM API
        embedding_base_url: Base URL for embedding API
        llm_model: Model name for chat completions
        embedding_model: Model name for embeddings
    """
    # Feature toggles
    enable_mesh_filter: bool = True
    enable_keyword_filter: bool = True
    filter_mode: FilterMode = FilterMode.MESH_OR_KEYWORD

    # MeSH matching parameters
    mesh_top_k: int = 20
    mesh_llm_select: int = 20  # Increased from 8 to include more relevant terms
    mesh_pmid_limit: int = 50000  # Increased from 10000 for better recall

    # Keyword filter parameters
    keyword_limit: int = 10000

    # Vector search parameters
    vector_limit: int = 100
    final_limit: int = 50

    # Database configuration (from config.py)
    pubmed_db_url: str = field(default_factory=lambda: _cfg.postgres.pubmed_url)
    qdrant_url: str = field(default_factory=lambda: _cfg.qdrant.url)
    mesh_collection: str = field(default_factory=lambda: _cfg.qdrant.mesh_collection)
    paper_collection: str = field(default_factory=lambda: _cfg.qdrant.paper_collection)

    # LLM configuration (from config.py)
    llm_base_url: str = field(default_factory=lambda: _cfg.llm.primary)
    embedding_base_url: str = field(default_factory=lambda: _cfg.embedding.servers[0])
    rerank_base_url: str = field(default_factory=lambda: _cfg.rerank.primary)
    llm_model: str = field(default_factory=lambda: _cfg.llm.model)
    embedding_model: str = field(default_factory=lambda: _cfg.embedding.model)
    rerank_model: str = field(default_factory=lambda: _cfg.rerank.model)
