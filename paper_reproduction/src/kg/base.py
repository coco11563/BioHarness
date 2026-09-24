"""Base classes and data models for KG-based RAG implementations.

This module defines:
- Data models: Entity, Relation, Community, Path
- KGClient protocol: Interface for LightRAG, PathRAG, GraphRAG
- Common types and utilities

Design Philosophy:
- Fail-fast: No fallbacks, errors propagate immediately
- Shared models: All three implementations use these data structures
- Protocol-based: Duck typing for flexibility
"""

from dataclasses import dataclass, field
from typing import Any, Literal, Protocol, runtime_checkable
from enum import Enum


# =============================================================================
# Enums
# =============================================================================

class EntityType(str, Enum):
    """Standard entity types for biomedical domain."""
    GENE = "gene"
    PROTEIN = "protein"
    DISEASE = "disease"
    DRUG = "drug"
    CHEMICAL = "chemical"
    PATHWAY = "pathway"
    CELL_TYPE = "cell_type"
    ORGANISM = "organism"
    ANATOMY = "anatomy"
    PROCESS = "biological_process"
    CONCEPT = "concept"
    METHOD = "method"
    OTHER = "other"


class RelationType(str, Enum):
    """Standard relation types for biomedical domain."""
    TREATS = "treats"
    CAUSES = "causes"
    ASSOCIATED_WITH = "associated_with"
    INHIBITS = "inhibits"
    ACTIVATES = "activates"
    REGULATES = "regulates"
    EXPRESSED_IN = "expressed_in"
    INTERACTS_WITH = "interacts_with"
    PART_OF = "part_of"
    LOCATED_IN = "located_in"
    RELATED_TO = "related_to"
    OTHER = "other"


class QueryMode(str, Enum):
    """Query modes for KG retrieval."""
    # LightRAG modes
    NAIVE = "naive"       # Pure vector search
    LOCAL = "local"       # Entity-centric
    GLOBAL = "global"     # Relation/community-centric
    HYBRID = "hybrid"     # Combined local + global
    # GraphRAG specific
    MAP_REDUCE = "map_reduce"  # Community report aggregation


# =============================================================================
# Data Models
# =============================================================================

@dataclass
class Entity:
    """A named entity extracted from text.

    Represents a concept, object, or thing mentioned in the corpus.
    Used by all three KG implementations.

    Attributes:
        id: Unique identifier (usually hash of name)
        name: Canonical entity name
        type: Entity type (gene, disease, drug, etc.)
        description: LLM-generated description of the entity
        mentions: Number of times mentioned in corpus
        source_chunks: IDs of chunks where entity was found
        embedding: Vector embedding of entity description
        attributes: Additional metadata
    """
    id: str
    name: str
    type: str
    description: str
    mentions: int = 1
    source_chunks: list[str] = field(default_factory=list)
    embedding: list[float] | None = None
    attributes: dict[str, Any] = field(default_factory=dict)

    def __hash__(self) -> int:
        return hash(self.id)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Entity):
            return False
        return self.id == other.id


@dataclass
class Relation:
    """A relationship between two entities.

    Represents a directed edge in the knowledge graph.
    Used by all three KG implementations.

    Attributes:
        id: Unique identifier
        source: Source entity name
        target: Target entity name
        type: Relation type (treats, causes, etc.)
        description: LLM-generated description of the relationship
        weight: Edge weight (default 1.0, higher = stronger)
        keywords: Keywords describing the relationship
        source_chunks: IDs of chunks where relation was found
        embedding: Vector embedding of relation description
        attributes: Additional metadata
    """
    id: str
    source: str
    target: str
    type: str
    description: str
    weight: float = 1.0
    keywords: list[str] = field(default_factory=list)
    source_chunks: list[str] = field(default_factory=list)
    embedding: list[float] | None = None
    attributes: dict[str, Any] = field(default_factory=dict)

    def __hash__(self) -> int:
        return hash(self.id)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Relation):
            return False
        return self.id == other.id


@dataclass
class Community:
    """A community of related entities (for MS GraphRAG).

    Represents a cluster of entities detected by community detection
    algorithm (Louvain/Leiden). Used primarily by GraphRAG.

    Attributes:
        id: Unique identifier
        level: Hierarchy level (0 = finest granularity)
        member_entities: List of entity IDs in this community
        member_relations: List of relation IDs within community
        title: Short title for the community
        summary: LLM-generated summary of the community
        rank: Importance ranking
        parent_id: ID of parent community (higher level)
        child_ids: IDs of child communities (lower level)
        embedding: Vector embedding of summary
    """
    id: str
    level: int
    member_entities: list[str] = field(default_factory=list)
    member_relations: list[str] = field(default_factory=list)
    title: str = ""
    summary: str = ""
    rank: int = 0
    parent_id: str | None = None
    child_ids: list[str] = field(default_factory=list)
    embedding: list[float] | None = None

    @property
    def size(self) -> int:
        """Number of entities in community."""
        return len(self.member_entities)


@dataclass
class Path:
    """A path through the knowledge graph (for PathRAG).

    Represents a sequence of entities connected by relations.
    Used primarily by PathRAG for multi-hop reasoning.

    Attributes:
        nodes: List of entity names in path order
        edges: List of relation types connecting nodes
        edge_ids: List of relation IDs
        score: Path relevance score
        hops: Number of hops (edges) in path
        evidence: Natural language description of path
        source_chunks: Chunks supporting this path
    """
    nodes: list[str]
    edges: list[str]
    edge_ids: list[str] = field(default_factory=list)
    score: float = 0.0
    evidence: str = ""
    source_chunks: list[str] = field(default_factory=list)

    @property
    def hops(self) -> int:
        """Number of hops in path."""
        return len(self.edges)

    def to_string(self) -> str:
        """Convert path to readable string format."""
        if not self.nodes:
            return ""
        parts = [self.nodes[0]]
        for i, (edge, node) in enumerate(zip(self.edges, self.nodes[1:])):
            parts.append(f" --[{edge}]--> {node}")
        return "".join(parts)


@dataclass
class KGStats:
    """Statistics about a knowledge graph."""
    num_entities: int = 0
    num_relations: int = 0
    num_communities: int = 0
    num_chunks: int = 0
    entity_types: dict[str, int] = field(default_factory=dict)
    relation_types: dict[str, int] = field(default_factory=dict)
    avg_entity_degree: float = 0.0
    max_community_level: int = 0


@dataclass
class QueryResult:
    """Result from a KG query.

    Contains both the answer and supporting context.

    Attributes:
        answer: Final answer text
        entities: Relevant entities retrieved
        relations: Relevant relations retrieved
        paths: Relevant paths (PathRAG)
        communities: Relevant communities (GraphRAG)
        context_text: Formatted context for LLM
        latency_ms: Query latency in milliseconds
        mode: Query mode used
        metadata: Additional query metadata
    """
    answer: str
    entities: list[Entity] = field(default_factory=list)
    relations: list[Relation] = field(default_factory=list)
    paths: list[Path] = field(default_factory=list)
    communities: list[Community] = field(default_factory=list)
    context_text: str = ""
    latency_ms: float = 0.0
    mode: str = "hybrid"
    metadata: dict[str, Any] = field(default_factory=dict)


# =============================================================================
# Protocols
# =============================================================================

@runtime_checkable
class KGClient(Protocol):
    """Protocol for KG-based RAG clients.

    All three implementations (LightRAG, PathRAG, GraphRAG) must
    implement this interface for benchmark integration.

    Methods:
        index: Build/update KG from text chunks
        query: Answer a question using KG retrieval
        get_stats: Return KG statistics
        save: Persist KG to storage
        load: Load KG from storage
    """

    async def index(
        self,
        chunks: list[str],
        chunk_ids: list[str] | None = None,
        **kwargs,
    ) -> None:
        """Build or update knowledge graph from text chunks.

        Args:
            chunks: List of text chunks to process
            chunk_ids: Optional IDs for chunks (auto-generated if not provided)
            **kwargs: Implementation-specific options
        """
        ...

    async def query(
        self,
        question: str,
        mode: str = "hybrid",
        top_k: int = 20,
        **kwargs,
    ) -> QueryResult:
        """Query the knowledge graph.

        Args:
            question: Natural language question
            mode: Query mode (naive, local, global, hybrid)
            top_k: Number of results to retrieve
            **kwargs: Implementation-specific options

        Returns:
            QueryResult with answer and supporting context
        """
        ...

    def get_stats(self) -> KGStats:
        """Get knowledge graph statistics.

        Returns:
            KGStats with entity/relation/community counts
        """
        ...

    async def save(self, path: str) -> None:
        """Save knowledge graph to disk.

        Args:
            path: Directory path to save to
        """
        ...

    async def load(self, path: str) -> None:
        """Load knowledge graph from disk.

        Args:
            path: Directory path to load from
        """
        ...


# =============================================================================
# Benchmark Integration
# =============================================================================

@dataclass
class ModelResponse:
    """Response format for benchmark integration.

    Matches the benchmark ModelResponse protocol.
    """
    answer: str
    response_text: str
    latency_ms: float
    metadata: dict[str, Any] | None = None


@runtime_checkable
class BenchmarkModelClient(Protocol):
    """Protocol matching benchmark ModelClient interface."""

    async def generate(
        self,
        question: str,
        question_type: str,
        context: list[str] | None = None,
        options: dict[str, str] | None = None,
    ) -> ModelResponse:
        """Generate answer for benchmark question.

        Args:
            question: Question text
            question_type: Type (yesno, mcq, factoid, list, summary)
            context: Optional context passages
            options: MCQ options dict

        Returns:
            ModelResponse with answer and metadata
        """
        ...
