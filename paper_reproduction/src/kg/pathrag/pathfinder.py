"""Path finding algorithms for PathRAG.

Implements multi-hop path discovery with weighted BFS scoring.

Re-implemented following the PathRAG paper and algorithm
(Reference: PathRAG/operate.py:1014-1240). The upstream repository
(github.com/BUPT-GAMMA/PathRAG) publishes no licence.
"""

import tiktoken
from collections import defaultdict
from dataclasses import dataclass, field

import networkx as nx

from ..base import Entity, Path, Relation
from ..storage.graph_store import GraphStore


# Tokenizer for evidence truncation
_tokenizer = None


def _get_tokenizer():
    global _tokenizer
    if _tokenizer is None:
        _tokenizer = tiktoken.get_encoding("cl100k_base")
    return _tokenizer


def _count_tokens(text: str) -> int:
    """Count tokens in text."""
    return len(_get_tokenizer().encode(text))


@dataclass
class PathScore:
    """Scoring result for a path."""
    path: Path
    score: float
    decay_factor: float
    hop_count: int


@dataclass
class PathFindingResult:
    """Result from path finding with hop tiers."""
    one_hop: list[list[str]] = field(default_factory=list)
    two_hop: list[list[str]] = field(default_factory=list)
    three_hop: list[list[str]] = field(default_factory=list)
    all_paths: dict[tuple[str, str], dict] = field(default_factory=dict)


class PathFinder:
    """Multi-hop path finding with weighted BFS scoring.

    Implements reference PathRAG algorithm:
    1. DFS to find all paths up to 3 hops
    2. Weighted BFS scoring with alpha decay
    3. Threshold pruning for edge weights
    4. Hop-tier quota selection

    Reference: PathRAG/operate.py:1014-1240

    Example:
        >>> finder = PathFinder(graph_store, alpha=0.8, threshold=0.3)
        >>> paths = await finder.find_paths(
        ...     source_entities=["BRCA1", "TP53"],
        ...     max_hops=3
        ... )
    """

    def __init__(
        self,
        graph_store: GraphStore,
        alpha: float = 0.8,
        threshold: float = 0.3,
        max_total_edges: int = 15,
        max_evidence_tokens: int = 4000,
    ):
        """Initialize path finder.

        Args:
            graph_store: Graph storage with entities and relations
            alpha: Decay factor per hop (0-1)
            threshold: Minimum edge weight to continue traversal
            max_total_edges: Maximum total paths/edges to return
            max_evidence_tokens: Maximum tokens for evidence text
        """
        self._graph = graph_store
        self._alpha = alpha
        self._threshold = threshold
        self._max_total_edges = max_total_edges
        self._max_evidence_tokens = max_evidence_tokens

    async def find_paths(
        self,
        source_entities: list[str],
        max_hops: int = 3,
    ) -> list[PathScore]:
        """Find paths between source entities with weighted scoring.

        Reference: operate.py:1014-1053, 1107-1171

        1. DFS to find all paths between entity pairs
        2. Apply weighted BFS scoring with alpha decay
        3. Select paths with hop-tier quotas
        4. Limit to max_total_edges

        Args:
            source_entities: Entity names to find paths between
            max_hops: Maximum path length (1-3)

        Returns:
            List of PathScore sorted by score descending
        """
        # Get underlying NetworkX graph
        nx_graph = self._graph.to_networkx()

        # Filter to entities that exist in graph
        target_nodes = [e for e in source_entities if e in nx_graph]

        if len(target_nodes) < 2:
            return []

        # Step 1: DFS to find all paths with hop tiers
        result = await self._find_paths_with_stats(nx_graph, target_nodes, max_hops)

        # Step 2: Apply weighted BFS scoring
        all_weighted_paths: list[tuple[list[str], float]] = []

        for node1 in target_nodes:
            for node2 in target_nodes:
                if node1 != node2 and (node1, node2) in result.all_paths:
                    paths = result.all_paths[(node1, node2)]["paths"]
                    if paths:
                        weighted = self._bfs_weighted_paths(
                            nx_graph, paths, node1, node2
                        )
                        all_weighted_paths.extend(weighted)

        # Sort by weight descending
        all_weighted_paths.sort(key=lambda x: x[1], reverse=True)

        # Deduplicate by sorted path tuple
        seen = set()
        unique_paths: list[tuple[list[str], float]] = []
        for path, weight in all_weighted_paths:
            sorted_path = tuple(sorted(path))
            if sorted_path not in seen:
                seen.add(sorted_path)
                unique_paths.append((path, weight))

        # Step 3: Apply hop-tier quotas
        # Reference: Take half of each tier
        quota_1hop = len(result.one_hop) // 2
        quota_2hop = len(result.two_hop) // 2
        quota_3hop = len(result.three_hop) // 2

        total_quota = quota_1hop + quota_2hop + quota_3hop
        if total_quota > self._max_total_edges:
            total_quota = self._max_total_edges

        # Select top paths up to quota
        selected_paths = unique_paths[:total_quota] if unique_paths else []

        # Step 4: Build PathScore objects
        path_scores = []
        for path_nodes, weight in selected_paths:
            path_obj = self._build_path(nx_graph, path_nodes)
            hop_count = len(path_nodes) - 1

            path_scores.append(
                PathScore(
                    path=path_obj,
                    score=weight,
                    decay_factor=self._alpha ** hop_count,
                    hop_count=hop_count,
                )
            )

        return path_scores

    async def _find_paths_with_stats(
        self,
        graph: nx.DiGraph,
        target_nodes: list[str],
        max_hops: int,
    ) -> PathFindingResult:
        """Find all paths using DFS with hop tier tracking.

        Reference: operate.py:1014-1053

        Args:
            graph: NetworkX graph
            target_nodes: Nodes to find paths between
            max_hops: Maximum path length

        Returns:
            PathFindingResult with paths organized by hop count
        """
        result = PathFindingResult()
        result.all_paths = defaultdict(lambda: {"paths": [], "edges": set()})

        # Use undirected view
        undirected = graph.to_undirected()

        def dfs(current: str, target: str, path: list[str], depth: int):
            """DFS to find paths."""
            if depth > max_hops:
                return

            if current == target:
                result.all_paths[(path[0], target)]["paths"].append(list(path))
                for u, v in zip(path[:-1], path[1:]):
                    result.all_paths[(path[0], target)]["edges"].add(
                        tuple(sorted((u, v)))
                    )

                if depth == 1:
                    result.one_hop.append(list(path))
                elif depth == 2:
                    result.two_hop.append(list(path))
                elif depth == 3:
                    result.three_hop.append(list(path))
                return

            for neighbor in undirected.neighbors(current):
                if neighbor not in path:
                    dfs(neighbor, target, path + [neighbor], depth + 1)

        # Find paths between all pairs
        for node1 in target_nodes:
            for node2 in target_nodes:
                if node1 != node2:
                    dfs(node1, node2, [node1], 0)

        # Convert edges to lists
        for key in result.all_paths:
            result.all_paths[key]["edges"] = list(result.all_paths[key]["edges"])

        return result

    def _bfs_weighted_paths(
        self,
        graph: nx.Graph,
        paths: list[list[str]],
        source: str,
        target: str,
    ) -> list[tuple[list[str], float]]:
        """Calculate weighted scores for paths using BFS with alpha decay.

        Reference: operate.py:1054-1106

        Edge weights are propagated from source with:
        - Initial weight: 1 / num_neighbors
        - Decay: weight * alpha / num_next_neighbors
        - Pruning: only continue if weight > threshold

        Args:
            graph: NetworkX graph
            paths: List of paths (node sequences)
            source: Source node
            target: Target node

        Returns:
            List of (path, weight) tuples
        """
        edge_weights: dict[tuple[str, str], float] = defaultdict(float)

        # successors[n]: every node that directly follows n in some candidate path
        successors: dict[str, set[str]] = defaultdict(set)
        for p in paths:
            for a, b in zip(p, p[1:]):
                successors[a].add(b)

        if source not in successors:
            return []

        def spread(u: str, v: str) -> float:
            # Weight handed to each edge leaving v: the accumulated (u, v) weight,
            # decayed by alpha and split evenly over v's successors.
            return edge_weights[(u, v)] * self._alpha / len(successors[v])

        # Propagate from `source` for at most three hops. An edge is expanded only
        # while its accumulated weight is above the threshold, never past `target`,
        # and only if its head node has successors.
        for hop1 in successors[source]:
            edge_weights[(source, hop1)] += 1 / len(successors[source])
            if (hop1 == target or not edge_weights[(source, hop1)] > self._threshold
                    or hop1 not in successors):
                continue
            for hop2 in successors[hop1]:
                edge_weights[(hop1, hop2)] += spread(source, hop1)
                if (hop2 == target or not edge_weights[(hop1, hop2)] > self._threshold
                        or hop2 not in successors):
                    continue
                for hop3 in successors[hop2]:
                    edge_weights[(hop2, hop3)] += spread(hop1, hop2)

        # Path score = mean accumulated weight of its edges
        path_weights: list[tuple[list[str], float]] = []
        for p in paths:
            path_weight = 0.0
            for edge in zip(p, p[1:]):
                path_weight += edge_weights.get(edge, 0.0)
            if len(p) > 1:
                path_weight /= len(p) - 1
            path_weights.append((p, path_weight))

        return path_weights

    def _build_path(self, graph: nx.DiGraph, nodes: list[str]) -> Path:
        """Build Path object from node sequence.

        Args:
            graph: NetworkX graph
            nodes: List of node names in path order

        Returns:
            Path object
        """
        edges = []
        edge_ids = []
        source_chunks: set[str] = set()

        for i in range(len(nodes) - 1):
            src, tgt = nodes[i], nodes[i + 1]

            # Get edge data (try both directions)
            if graph.has_edge(src, tgt):
                edge_data = graph.edges[src, tgt]
            elif graph.has_edge(tgt, src):
                edge_data = graph.edges[tgt, src]
            else:
                edge_data = {}

            rel_type = edge_data.get("relation_type", "related_to")
            rel_id = edge_data.get("relation_id", "")

            edges.append(rel_type)
            if rel_id:
                edge_ids.append(rel_id)

            # Get relation for source chunks
            relation = self._graph.get_relation(rel_id) if rel_id else None
            if relation:
                source_chunks.update(relation.source_chunks)

        return Path(
            nodes=nodes,
            edges=edges,
            edge_ids=edge_ids,
            source_chunks=list(source_chunks),
        )

    async def find_connecting_subgraph(
        self,
        entities: list[str],
        max_hops: int = 2,
    ) -> tuple[list[Entity], list[Path]]:
        """Find subgraph connecting given entities.

        Args:
            entities: Entity names to connect
            max_hops: Maximum distance between any pair

        Returns:
            Tuple of (entities in subgraph, paths connecting them)
        """
        # Find paths between all pairs
        path_scores = await self.find_paths(
            source_entities=entities,
            max_hops=max_hops,
        )

        # Collect all entities in paths
        entity_names: set[str] = set(entities)
        for ps in path_scores:
            entity_names.update(ps.path.nodes)

        # Get entity objects
        entities_list = []
        for name in entity_names:
            entity = self._graph.get_entity(name)
            if entity:
                entities_list.append(entity)

        # Get paths
        paths = [ps.path for ps in path_scores]

        return entities_list, paths

    async def generate_path_evidence(
        self,
        paths: list[Path],
        max_tokens: int | None = None,
    ) -> str:
        """Generate natural language evidence from paths.

        Reference: operate.py:1176-1238

        Creates narrative descriptions using:
        - Entity descriptions: "The entity X is a TYPE with the description(...)"
        - Edge keywords: "through edge (KEYWORDS) to connect to X and Y"

        Args:
            paths: List of paths to describe
            max_tokens: Maximum tokens for evidence (uses config default if None)

        Returns:
            Natural language description of paths
        """
        if not paths:
            return "No connecting paths found."

        max_tokens = max_tokens or self._max_evidence_tokens
        evidences: list[str] = []
        total_tokens = 0

        for path in paths:
            evidence = await self._generate_single_path_evidence(path)
            evidence_tokens = _count_tokens(evidence)

            if total_tokens + evidence_tokens > max_tokens:
                break

            evidences.append(evidence)
            total_tokens += evidence_tokens

        if not evidences:
            return "No connecting paths found."

        lines = ["## Path-based Evidence\n"]
        for i, evidence in enumerate(evidences, 1):
            lines.append(f"**Path {i}**: {evidence}\n")

        return "\n".join(lines)

    async def _generate_single_path_evidence(self, path: Path) -> str:
        """Generate evidence for a single path.

        Reference: operate.py:1176-1229

        Formats:
        - 1-hop: "Entity A (type: desc) --[keywords]--> Entity B (type: desc)"
        - 2-hop: A → B → C with entity descriptions and edge keywords
        - 3-hop: A → B → C → D with entity descriptions and edge keywords

        Args:
            path: Path to describe

        Returns:
            Natural language path description
        """
        nodes = path.nodes
        num_hops = len(nodes) - 1

        if num_hops < 1:
            return ""

        parts: list[str] = []

        for i, node_name in enumerate(nodes):
            # Get entity info
            entity = self._graph.get_entity(node_name)
            if entity:
                entity_desc = f"The entity {node_name} is a {entity.type}"
                if entity.description:
                    # Truncate description
                    desc = entity.description[:200]
                    entity_desc += f" with the description ({desc})"
            else:
                entity_desc = f"Entity {node_name}"

            parts.append(entity_desc)

            # Add edge description (if not last node)
            if i < num_hops:
                edge_id = path.edge_ids[i] if i < len(path.edge_ids) else None
                relation = self._graph.get_relation(edge_id) if edge_id else None

                if relation and relation.keywords:
                    keywords_str = ", ".join(relation.keywords[:3])
                    edge_desc = f"through edge ({keywords_str}) to connect to {nodes[i]} and {nodes[i + 1]}"
                else:
                    edge_type = path.edges[i] if i < len(path.edges) else "related_to"
                    edge_desc = f"through edge ({edge_type}) to connect to {nodes[i]} and {nodes[i + 1]}"

                parts.append(edge_desc)

        # Join alternating entity and edge descriptions
        evidence = ". ".join(parts) + "."

        return evidence

    def generate_simple_path_evidence(self, paths: list[Path]) -> str:
        """Generate simple path evidence without entity descriptions.

        Fallback for when detailed evidence is not needed.

        Args:
            paths: List of paths to describe

        Returns:
            Simple path strings
        """
        if not paths:
            return "No connecting paths found."

        lines = ["## Path-based Evidence\n"]

        for i, path in enumerate(paths[:10], 1):
            path_str = path.to_string()
            lines.append(f"**Path {i}**: {path_str}")

            if path.evidence:
                lines.append(f"  Evidence: {path.evidence}")

            lines.append("")

        return "\n".join(lines)
