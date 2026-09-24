"""Community detection and report generation for GraphRAG.

Uses hierarchical Leiden algorithm for community detection with
max cluster size constraints and LCC filtering.

Reference: graphrag/index/operations/cluster_graph.py
Parts of the clustering code follow Microsoft GraphRAG
(https://github.com/microsoft/graphrag) line for line; MIT licence, Copyright (c)
Microsoft Corporation; see ../../../THIRD_PARTY_NOTICES.txt.
"""

import hashlib
import json
import re
import tiktoken
from collections import defaultdict
from dataclasses import dataclass
from typing import Any

import networkx as nx

from ..base import Entity, Relation, Community
from ..storage.graph_store import GraphStore
from ..extraction.prompts import COMMUNITY_SUMMARY_PROMPT

try:
    from utils.clients import llm_client
except ImportError:
    from src.utils.clients import llm_client


# Try to import graspologic for hierarchical Leiden
try:
    from graspologic.partition import hierarchical_leiden
    HAS_GRASPOLOGIC = True
except ImportError:
    HAS_GRASPOLOGIC = False


# Tokenizer for token-aware report generation
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
class HierarchicalCommunity:
    """Community with hierarchy information."""
    level: int
    cluster_id: int
    parent_id: int  # -1 if no parent
    members: list[str]


def stable_largest_connected_component(graph: nx.Graph) -> nx.Graph:
    """Get largest connected component with stable ordering.

    Reference: graphrag/index/utils/stable_lcc.py
    """
    if graph.number_of_nodes() == 0:
        return graph

    # Get connected components sorted by size (descending)
    if graph.is_directed():
        components = list(nx.weakly_connected_components(graph))
    else:
        components = list(nx.connected_components(graph))

    if not components:
        return graph

    # Sort by size, then by sorted node names for stability
    components.sort(key=lambda c: (-len(c), sorted(c)))
    largest = components[0]

    return graph.subgraph(largest).copy()


class CommunityDetector:
    """Detect hierarchical communities in knowledge graph.

    Uses hierarchical Leiden algorithm (if available) or falls back
    to Louvain with max cluster size constraints.

    Reference: graphrag/index/operations/cluster_graph.py

    Example:
        >>> detector = CommunityDetector(graph_store)
        >>> communities = await detector.detect(max_cluster_size=10)
        >>> print(f"Found {len(communities)} communities")
    """

    def __init__(self, graph_store: GraphStore):
        """Initialize community detector.

        Args:
            graph_store: Graph storage with entities and relations
        """
        self._graph = graph_store

    async def detect(
        self,
        max_cluster_size: int = 10,
        use_lcc: bool = True,
        seed: int = 42,
        levels: int = 3,
    ) -> list[Community]:
        """Detect hierarchical communities.

        Reference: graphrag/index/operations/cluster_graph.py:19-53

        Args:
            max_cluster_size: Maximum cluster size for Leiden algorithm
            use_lcc: Whether to filter to largest connected component
            seed: Random seed for reproducibility
            levels: Number of hierarchy levels (for Louvain fallback)

        Returns:
            List of Community objects across all levels
        """
        nx_graph = self._graph.to_networkx()

        if nx_graph.number_of_nodes() < 2 or nx_graph.number_of_edges() == 0:
            return []

        # Convert to undirected for community detection
        undirected = nx_graph.to_undirected()

        try:
            # Use hierarchical Leiden if available
            if HAS_GRASPOLOGIC:
                hierarchical_communities = self._detect_leiden(
                    undirected, max_cluster_size, use_lcc, seed
                )
            else:
                # Fallback to Louvain with max cluster size constraint
                hierarchical_communities = self._detect_louvain_hierarchical(
                    undirected, max_cluster_size, use_lcc, seed, levels
                )
        except Exception:
            # Degenerate mini-KGs are common in retrieve-then-KG mode. When the
            # clustering backend rejects an empty/ill-formed network, treat it
            # as "no communities" so GraphRAG can still answer from entities
            # and relations instead of failing the whole sample.
            return []

        # Convert to Community objects
        all_communities: list[Community] = []
        for hc in hierarchical_communities:
            if len(hc.members) < 2:
                continue

            community_id = self._generate_community_id(
                hc.level, hc.cluster_id, set(hc.members)
            )

            # Find relations within community
            member_set = set(hc.members)
            member_relations = []
            for rel in self._graph.iter_relations():
                if rel.source in member_set and rel.target in member_set:
                    member_relations.append(rel.id)

            community = Community(
                id=community_id,
                level=hc.level,
                member_entities=hc.members,
                member_relations=member_relations,
            )

            # Set parent_id based on hierarchy
            if hc.parent_id >= 0:
                # Find parent community by matching cluster_id
                for other_hc in hierarchical_communities:
                    if other_hc.cluster_id == hc.parent_id and other_hc.level == hc.level + 1:
                        parent_comm_id = self._generate_community_id(
                            other_hc.level, other_hc.cluster_id, set(other_hc.members)
                        )
                        community.parent_id = parent_comm_id
                        break

            all_communities.append(community)

        # Build child_ids references
        id_to_community = {c.id: c for c in all_communities}
        for c in all_communities:
            if c.parent_id and c.parent_id in id_to_community:
                id_to_community[c.parent_id].child_ids.append(c.id)

        return all_communities

    def _detect_leiden(
        self,
        graph: nx.Graph,
        max_cluster_size: int,
        use_lcc: bool,
        seed: int,
    ) -> list[HierarchicalCommunity]:
        """Detect communities using hierarchical Leiden.

        Reference: graphrag/index/operations/cluster_graph.py:57-80
        """
        if use_lcc:
            graph = stable_largest_connected_component(graph)

        if graph.number_of_nodes() < 2 or graph.number_of_edges() == 0:
            return []

        community_mapping = hierarchical_leiden(
            graph, max_cluster_size=max_cluster_size, random_seed=seed
        )

        # Build results from partition objects
        results: dict[int, dict[str, int]] = {}
        hierarchy: dict[int, int] = {}

        for partition in community_mapping:
            if partition.level not in results:
                results[partition.level] = {}
            results[partition.level][partition.node] = partition.cluster

            hierarchy[partition.cluster] = (
                partition.parent_cluster if partition.parent_cluster is not None else -1
            )

        # Convert to HierarchicalCommunity objects
        communities = []
        for level, node_clusters in results.items():
            # Group nodes by cluster
            clusters: dict[int, list[str]] = defaultdict(list)
            for node, cluster_id in node_clusters.items():
                clusters[cluster_id].append(node)

            for cluster_id, members in clusters.items():
                communities.append(HierarchicalCommunity(
                    level=level,
                    cluster_id=cluster_id,
                    parent_id=hierarchy.get(cluster_id, -1),
                    members=members,
                ))

        return communities

    def _detect_louvain_hierarchical(
        self,
        graph: nx.Graph,
        max_cluster_size: int,
        use_lcc: bool,
        seed: int,
        levels: int,
    ) -> list[HierarchicalCommunity]:
        """Detect communities using Louvain with max cluster size constraint.

        Fallback when graspologic is not available.
        """
        if use_lcc:
            graph = stable_largest_connected_component(graph)

        if graph.number_of_nodes() == 0:
            return []

        all_communities: list[HierarchicalCommunity] = []
        cluster_counter = 0

        for level in range(levels):
            # Vary resolution to create hierarchy (higher = finer granularity)
            resolution = 1.0 + level * 0.5

            try:
                partition = list(nx.community.louvain_communities(
                    graph,
                    resolution=resolution,
                    seed=seed,
                ))
            except Exception:
                try:
                    partition = list(nx.community.greedy_modularity_communities(graph))
                except Exception:
                    continue

            for members_set in partition:
                members = list(members_set)

                # Split large clusters if needed
                if len(members) > max_cluster_size and level < levels - 1:
                    # Create subgraph and detect sub-communities
                    subgraph = graph.subgraph(members)
                    try:
                        sub_partition = list(nx.community.louvain_communities(
                            subgraph.to_undirected(),
                            resolution=resolution * 1.5,
                            seed=seed,
                        ))
                        for sub_members_set in sub_partition:
                            sub_members = list(sub_members_set)
                            all_communities.append(HierarchicalCommunity(
                                level=level,
                                cluster_id=cluster_counter,
                                parent_id=-1,  # Will be resolved later
                                members=sub_members,
                            ))
                            cluster_counter += 1
                    except Exception:
                        # Keep original if subdivision fails
                        all_communities.append(HierarchicalCommunity(
                            level=level,
                            cluster_id=cluster_counter,
                            parent_id=-1,
                            members=members,
                        ))
                        cluster_counter += 1
                else:
                    all_communities.append(HierarchicalCommunity(
                        level=level,
                        cluster_id=cluster_counter,
                        parent_id=-1,
                        members=members,
                    ))
                    cluster_counter += 1

        # Build hierarchy based on member overlap
        for i, child in enumerate(all_communities):
            if child.level == 0:
                continue

            child_members = set(child.members)
            best_parent_idx = -1
            best_overlap = 0

            for j, parent in enumerate(all_communities):
                if parent.level == child.level - 1:
                    overlap = len(child_members & set(parent.members))
                    if overlap > best_overlap:
                        best_overlap = overlap
                        best_parent_idx = parent.cluster_id

            if best_parent_idx >= 0:
                child.parent_id = best_parent_idx

        return all_communities

    def _generate_community_id(
        self, level: int, index: int, members: set[str]
    ) -> str:
        """Generate deterministic community ID."""
        sorted_members = sorted(members)[:5]  # Use first 5 for stability
        key = f"L{level}_{index}_{','.join(sorted_members)}"
        return hashlib.sha256(key.encode()).hexdigest()[:16]


class CommunityReportGenerator:
    """Generate LLM summaries for communities with token-aware truncation.

    Creates natural language reports describing each community's
    theme, key entities, and important relationships.

    Reference: graphrag/index/operations/summarize_communities/
    """

    def __init__(
        self,
        graph_store: GraphStore,
        max_tokens_per_report: int = 2000,
    ):
        """Initialize report generator.

        Args:
            graph_store: Graph storage for entity/relation lookup
            max_tokens_per_report: Maximum tokens for report input
        """
        self._graph = graph_store
        self._max_tokens = max_tokens_per_report

    async def generate_reports(
        self,
        communities: list[Community],
        max_entities_per_report: int = 15,
        max_relations_per_report: int = 15,
        concurrency: int = 5,
    ) -> list[Community]:
        """Generate reports for all communities with token-aware truncation.

        Reference: graphrag/index/operations/summarize_communities/

        Args:
            communities: Communities to generate reports for
            max_entities_per_report: Max entities to include in prompt
            max_relations_per_report: Max relations to include in prompt
            concurrency: Number of concurrent LLM calls

        Returns:
            Communities with title and summary populated
        """
        import asyncio

        semaphore = asyncio.Semaphore(concurrency)

        async def generate_single(community: Community) -> Community:
            async with semaphore:
                return await self._generate_single_report(
                    community,
                    max_entities_per_report,
                    max_relations_per_report,
                )

        tasks = [generate_single(c) for c in communities]
        return await asyncio.gather(*tasks)

    async def _generate_single_report(
        self,
        community: Community,
        max_entities: int,
        max_relations: int,
    ) -> Community:
        """Generate report for a single community with token-aware truncation.

        Args:
            community: Community to summarize
            max_entities: Max entities in prompt
            max_relations: Max relations in prompt

        Returns:
            Community with title and summary populated
        """
        # Gather entity information with token tracking
        entity_info = []
        entity_tokens = 0
        max_entity_tokens = self._max_tokens // 2

        for name in community.member_entities[:max_entities]:
            entity = self._graph.get_entity(name)
            if entity:
                line = f"- {entity.name} ({entity.type}): {entity.description[:150]}"
                line_tokens = _count_tokens(line)
                if entity_tokens + line_tokens > max_entity_tokens:
                    break
                entity_info.append(line)
                entity_tokens += line_tokens

        # Gather relation information with token tracking
        relation_info = []
        relation_tokens = 0
        max_relation_tokens = self._max_tokens // 2

        for rel_id in community.member_relations[:max_relations]:
            relation = self._graph.get_relation(rel_id)
            if relation:
                line = f"- {relation.source} --[{relation.type}]--> {relation.target}"
                if relation.description:
                    line += f": {relation.description[:80]}"
                line_tokens = _count_tokens(line)
                if relation_tokens + line_tokens > max_relation_tokens:
                    break
                relation_info.append(line)
                relation_tokens += line_tokens

        # Build prompt
        prompt = COMMUNITY_SUMMARY_PROMPT.format(
            entities="\n".join(entity_info) or "No entity details available",
            relations="\n".join(relation_info) or "No relation details available",
        )

        try:
            response = await llm_client.chat(prompt)

            # Parse JSON response
            json_match = re.search(r"\{[\s\S]*\}", response)
            if json_match:
                data = json.loads(json_match.group())
                community.title = data.get("title", f"Community {community.id[:8]}")
                community.summary = data.get("summary", response)
            else:
                # Use raw response
                community.title = f"Community {community.id[:8]}"
                community.summary = response[:500]

        except Exception as e:
            # Fallback title/summary
            community.title = f"Community L{community.level}"
            community.summary = f"A community of {len(community.member_entities)} entities. Error: {e}"

        return community
