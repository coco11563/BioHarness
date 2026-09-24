"""NetworkX-based graph storage for KG.

Provides in-memory graph operations with serialization support.
"""

from collections import defaultdict
from typing import Iterator

import networkx as nx

from ..base import Entity, Relation, Community, Path, KGStats


class GraphStore:
    """In-memory graph storage using NetworkX.

    Stores entities as nodes and relations as edges.
    Supports efficient neighbor queries, path finding, and community detection.

    Example:
        >>> store = GraphStore()
        >>> store.add_entity(Entity(id="1", name="BRCA1", type="gene", description="..."))
        >>> store.add_relation(Relation(id="r1", source="BRCA1", target="Breast Cancer", ...))
        >>> neighbors = store.get_neighbors("BRCA1", max_hops=2)
    """

    def __init__(self):
        """Initialize empty graph store."""
        self._graph: nx.DiGraph = nx.DiGraph()
        self._entities: dict[str, Entity] = {}  # name -> Entity
        self._relations: dict[str, Relation] = {}  # id -> Relation
        self._communities: dict[str, Community] = {}  # id -> Community

    @property
    def num_entities(self) -> int:
        """Number of entities in graph."""
        return len(self._entities)

    @property
    def num_relations(self) -> int:
        """Number of relations in graph."""
        return len(self._relations)

    @property
    def num_communities(self) -> int:
        """Number of communities."""
        return len(self._communities)

    def add_entity(self, entity: Entity) -> None:
        """Add or update an entity in the graph.

        If entity with same name exists, merges the information.

        Args:
            entity: Entity to add
        """
        existing = self._entities.get(entity.name)
        if existing:
            # Merge: update mentions, extend source_chunks
            existing.mentions += entity.mentions
            existing.source_chunks = list(
                set(existing.source_chunks + entity.source_chunks)
            )
            # Keep longer description
            if len(entity.description) > len(existing.description):
                existing.description = entity.description
            # Update embedding if provided
            if entity.embedding is not None:
                existing.embedding = entity.embedding
        else:
            self._entities[entity.name] = entity
            self._graph.add_node(
                entity.name,
                entity_id=entity.id,
                entity_type=entity.type,
            )

    def add_relation(self, relation: Relation) -> None:
        """Add or update a relation in the graph.

        If relation between same source/target exists, merges information.

        Args:
            relation: Relation to add
        """
        # Ensure source and target nodes exist
        if relation.source not in self._graph:
            self._graph.add_node(relation.source)
        if relation.target not in self._graph:
            self._graph.add_node(relation.target)

        # Check for existing relation
        existing = self._relations.get(relation.id)
        if existing:
            # Merge: increase weight, extend source_chunks
            existing.weight += relation.weight
            existing.source_chunks = list(
                set(existing.source_chunks + relation.source_chunks)
            )
            existing.keywords = list(set(existing.keywords + relation.keywords))[:10]
            # Keep longer description
            if len(relation.description) > len(existing.description):
                existing.description = relation.description
            # Update the canonical edge. Relation ids are normalized to lowercase,
            # so repeated extractions may arrive with different surface casing
            # (for example "Breast Cancer" vs "breast cancer"). Reuse the
            # original edge endpoints instead of indexing with the new casing.
            if existing.source not in self._graph:
                self._graph.add_node(existing.source)
            if existing.target not in self._graph:
                self._graph.add_node(existing.target)

            if self._graph.has_edge(existing.source, existing.target):
                self._graph[existing.source][existing.target]["weight"] = existing.weight
            else:
                self._graph.add_edge(
                    existing.source,
                    existing.target,
                    relation_id=existing.id,
                    relation_type=existing.type,
                    weight=existing.weight,
                )
        else:
            self._relations[relation.id] = relation
            self._graph.add_edge(
                relation.source,
                relation.target,
                relation_id=relation.id,
                relation_type=relation.type,
                weight=relation.weight,
            )

    def add_community(self, community: Community) -> None:
        """Add a community to the store.

        Args:
            community: Community to add
        """
        self._communities[community.id] = community

    def get_entity(self, name: str) -> Entity | None:
        """Get entity by name.

        Args:
            name: Entity name

        Returns:
            Entity or None if not found
        """
        return self._entities.get(name)

    def get_relation(self, relation_id: str) -> Relation | None:
        """Get relation by ID.

        Args:
            relation_id: Relation ID

        Returns:
            Relation or None if not found
        """
        return self._relations.get(relation_id)

    def get_community(self, community_id: str) -> Community | None:
        """Get community by ID.

        Args:
            community_id: Community ID

        Returns:
            Community or None if not found
        """
        return self._communities.get(community_id)

    def get_relations_for_entity(self, entity_name: str) -> list[Relation]:
        """Get all relations involving an entity.

        Args:
            entity_name: Entity name

        Returns:
            List of relations where entity is source or target
        """
        relations = []
        for rel in self._relations.values():
            if rel.source == entity_name or rel.target == entity_name:
                relations.append(rel)
        return relations

    def get_neighbors(
        self,
        entity_name: str,
        max_hops: int = 1,
        direction: str = "both",
    ) -> dict[str, list[Entity]]:
        """Get neighboring entities within hop distance.

        Args:
            entity_name: Starting entity name
            max_hops: Maximum number of hops
            direction: "out", "in", or "both"

        Returns:
            Dict mapping hop distance to list of entities at that distance
        """
        if entity_name not in self._graph:
            return {}

        neighbors_by_hop: dict[str, list[Entity]] = defaultdict(list)
        visited = {entity_name}
        current_level = {entity_name}

        for hop in range(1, max_hops + 1):
            next_level = set()

            for node in current_level:
                if direction in ("out", "both"):
                    next_level.update(self._graph.successors(node))
                if direction in ("in", "both"):
                    next_level.update(self._graph.predecessors(node))

            # Remove already visited
            next_level -= visited

            # Add entities at this hop
            for name in next_level:
                entity = self._entities.get(name)
                if entity:
                    neighbors_by_hop[str(hop)].append(entity)

            visited.update(next_level)
            current_level = next_level

            if not current_level:
                break

        return dict(neighbors_by_hop)

    def find_paths(
        self,
        source: str,
        target: str,
        max_hops: int = 3,
        max_paths: int = 10,
    ) -> list[Path]:
        """Find paths between two entities.

        Args:
            source: Source entity name
            target: Target entity name
            max_hops: Maximum path length
            max_paths: Maximum number of paths to return

        Returns:
            List of Path objects
        """
        if source not in self._graph or target not in self._graph:
            return []

        paths = []
        try:
            # Use undirected view for path finding
            undirected = self._graph.to_undirected()

            for i, path_nodes in enumerate(
                nx.all_simple_paths(undirected, source, target, cutoff=max_hops)
            ):
                if i >= max_paths:
                    break

                # Build path with edge types
                edges = []
                edge_ids = []
                source_chunks = set()

                for j in range(len(path_nodes) - 1):
                    src, tgt = path_nodes[j], path_nodes[j + 1]

                    # Find relation (check both directions)
                    rel = None
                    for r in self._relations.values():
                        if (r.source == src and r.target == tgt) or (
                            r.source == tgt and r.target == src
                        ):
                            rel = r
                            break

                    if rel:
                        edges.append(rel.type)
                        edge_ids.append(rel.id)
                        source_chunks.update(rel.source_chunks)
                    else:
                        edges.append("unknown")

                # Calculate path score (inverse of length)
                score = 1.0 / len(path_nodes)

                paths.append(
                    Path(
                        nodes=path_nodes,
                        edges=edges,
                        edge_ids=edge_ids,
                        score=score,
                        source_chunks=list(source_chunks),
                    )
                )

        except nx.NetworkXNoPath:
            pass

        return paths

    def iter_entities(self) -> Iterator[Entity]:
        """Iterate over all entities.

        Yields:
            Entity objects
        """
        yield from self._entities.values()

    def iter_relations(self) -> Iterator[Relation]:
        """Iterate over all relations.

        Yields:
            Relation objects
        """
        yield from self._relations.values()

    def iter_communities(self) -> Iterator[Community]:
        """Iterate over all communities.

        Yields:
            Community objects
        """
        yield from self._communities.values()

    def get_stats(self) -> KGStats:
        """Get graph statistics.

        Returns:
            KGStats with counts and distributions
        """
        entity_types: dict[str, int] = defaultdict(int)
        for e in self._entities.values():
            entity_types[e.type] += 1

        relation_types: dict[str, int] = defaultdict(int)
        for r in self._relations.values():
            relation_types[r.type] += 1

        # Calculate average degree
        avg_degree = 0.0
        if self._graph.number_of_nodes() > 0:
            avg_degree = self._graph.number_of_edges() / self._graph.number_of_nodes()

        # Get max community level
        max_level = 0
        for c in self._communities.values():
            max_level = max(max_level, c.level)

        # Collect all source chunks
        all_chunks = set()
        for e in self._entities.values():
            all_chunks.update(e.source_chunks)
        for r in self._relations.values():
            all_chunks.update(r.source_chunks)

        return KGStats(
            num_entities=len(self._entities),
            num_relations=len(self._relations),
            num_communities=len(self._communities),
            num_chunks=len(all_chunks),
            entity_types=dict(entity_types),
            relation_types=dict(relation_types),
            avg_entity_degree=avg_degree,
            max_community_level=max_level,
        )

    def get_subgraph(self, entity_names: list[str]) -> "GraphStore":
        """Extract a subgraph containing only specified entities.

        Args:
            entity_names: List of entity names to include

        Returns:
            New GraphStore with the subgraph
        """
        subgraph = GraphStore()

        entity_set = set(entity_names)

        # Add entities
        for name in entity_names:
            entity = self._entities.get(name)
            if entity:
                subgraph.add_entity(entity)

        # Add relations where both endpoints are in the subgraph
        for relation in self._relations.values():
            if relation.source in entity_set and relation.target in entity_set:
                subgraph.add_relation(relation)

        return subgraph

    def clear(self) -> None:
        """Clear all data from the store."""
        self._graph.clear()
        self._entities.clear()
        self._relations.clear()
        self._communities.clear()

    def to_networkx(self) -> nx.DiGraph:
        """Get the underlying NetworkX graph.

        Returns:
            NetworkX DiGraph
        """
        return self._graph

    def to_dict(self) -> dict:
        """Serialize graph to dictionary.

        Returns:
            Dictionary representation
        """
        return {
            "entities": [
                {
                    "id": e.id,
                    "name": e.name,
                    "type": e.type,
                    "description": e.description,
                    "mentions": e.mentions,
                    "source_chunks": e.source_chunks,
                    "attributes": e.attributes,
                }
                for e in self._entities.values()
            ],
            "relations": [
                {
                    "id": r.id,
                    "source": r.source,
                    "target": r.target,
                    "type": r.type,
                    "description": r.description,
                    "weight": r.weight,
                    "keywords": r.keywords,
                    "source_chunks": r.source_chunks,
                    "attributes": r.attributes,
                }
                for r in self._relations.values()
            ],
            "communities": [
                {
                    "id": c.id,
                    "level": c.level,
                    "member_entities": c.member_entities,
                    "member_relations": c.member_relations,
                    "title": c.title,
                    "summary": c.summary,
                    "rank": c.rank,
                    "parent_id": c.parent_id,
                    "child_ids": c.child_ids,
                }
                for c in self._communities.values()
            ],
        }

    @classmethod
    def from_dict(cls, data: dict) -> "GraphStore":
        """Deserialize graph from dictionary.

        Args:
            data: Dictionary representation

        Returns:
            GraphStore instance
        """
        store = cls()

        for e_data in data.get("entities", []):
            entity = Entity(
                id=e_data["id"],
                name=e_data["name"],
                type=e_data["type"],
                description=e_data["description"],
                mentions=e_data.get("mentions", 1),
                source_chunks=e_data.get("source_chunks", []),
                attributes=e_data.get("attributes", {}),
            )
            store.add_entity(entity)

        for r_data in data.get("relations", []):
            relation = Relation(
                id=r_data["id"],
                source=r_data["source"],
                target=r_data["target"],
                type=r_data["type"],
                description=r_data["description"],
                weight=r_data.get("weight", 1.0),
                keywords=r_data.get("keywords", []),
                source_chunks=r_data.get("source_chunks", []),
                attributes=r_data.get("attributes", {}),
            )
            store.add_relation(relation)

        for c_data in data.get("communities", []):
            community = Community(
                id=c_data["id"],
                level=c_data["level"],
                member_entities=c_data.get("member_entities", []),
                member_relations=c_data.get("member_relations", []),
                title=c_data.get("title", ""),
                summary=c_data.get("summary", ""),
                rank=c_data.get("rank", 0),
                parent_id=c_data.get("parent_id"),
                child_ids=c_data.get("child_ids", []),
            )
            store.add_community(community)

        return store
