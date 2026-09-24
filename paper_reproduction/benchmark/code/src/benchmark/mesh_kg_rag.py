"""MeSH-KG-RAG: Knowledge Graph RAG using pre-existing MeSH hierarchy.

Unlike RT-KG which dynamically extracts entities/relations from documents,
MeSH-KG-RAG uses the standardized MeSH (Medical Subject Headings) hierarchy
as a pre-built knowledge graph.

Key differences from RT-KG:
- No LLM extraction overhead (uses curated MeSH terms)
- Standardized, high-quality knowledge structure
- Hierarchical relationships via tree_numbers
- Lower noise but limited to MeSH vocabulary

Pipeline:
1. Query → MeSH term matching (vector search in Qdrant)
2. MeSH hierarchy expansion (parent/child/sibling via tree_numbers)
3. PMID retrieval (mesh_headings table)
4. Document retrieval (abstracts from PostgreSQL)
5. Structured context generation
6. LLM answer generation
"""

from __future__ import annotations

import asyncio
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

# Add project root to path
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent.parent
_SRC_PATH = _PROJECT_ROOT / "src"

if str(_SRC_PATH) not in sys.path:
    sys.path.insert(0, str(_SRC_PATH))
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import asyncpg
from qdrant_client import AsyncQdrantClient

from .client_protocol import ModelClient, ModelResponse
from .shared_pipeline import generate_constrained_answer
from config import get_config
from utils.clients import embed_client


@dataclass
class MeSHTerm:
    """Represents a MeSH term with hierarchy info."""
    descriptor_ui: str
    descriptor_name: str
    tree_numbers: list[str]
    scope_note: str = ""
    entry_terms: list[str] = field(default_factory=list)
    score: float = 1.0
    hop: int = 0  # 0 = direct match, 1 = parent/child, 2 = sibling


@dataclass
class MeSHSubgraph:
    """A subgraph of MeSH terms extracted for a query."""
    seed_terms: list[MeSHTerm]
    expanded_terms: list[MeSHTerm]
    pmids: list[str]

    @property
    def all_terms(self) -> list[MeSHTerm]:
        return self.seed_terms + self.expanded_terms

    def to_context(self, include_scope: bool = True) -> str:
        """Convert subgraph to structured text context."""
        lines = ["## MeSH Knowledge Graph Context\n"]

        # Seed terms (direct matches)
        if self.seed_terms:
            lines.append("### Primary Concepts (Direct Match)")
            for term in self.seed_terms:
                lines.append(f"- **{term.descriptor_name}** ({term.descriptor_ui})")
                if include_scope and term.scope_note:
                    # Truncate long scope notes
                    scope = term.scope_note[:300] + "..." if len(term.scope_note) > 300 else term.scope_note
                    lines.append(f"  Definition: {scope}")
                if term.entry_terms:
                    lines.append(f"  Synonyms: {', '.join(term.entry_terms[:5])}")
            lines.append("")

        # Expanded terms (hierarchy)
        if self.expanded_terms:
            lines.append("### Related Concepts (Hierarchy)")
            for term in self.expanded_terms:
                relation = "parent" if term.hop == 1 else "sibling" if term.hop == 2 else "related"
                lines.append(f"- {term.descriptor_name} [{relation}]")
            lines.append("")

        return "\n".join(lines)


class MeSHKGRAG(ModelClient):
    """MeSH-KG-RAG: Uses pre-existing MeSH hierarchy for structured RAG.

    Compared to RT-KG:
    - No LLM extraction (uses curated MeSH)
    - Standardized hierarchy via tree_numbers
    - Lower computational cost
    - Limited to MeSH vocabulary coverage

    Example:
        async with MeSHKGRAG(expansion_hops=1) as client:
            response = await client.generate(
                question="Is metformin effective for diabetes?",
                question_type="yesno",
            )
    """

    def __init__(
        self,
        expansion_hops: int = 1,
        max_mesh_terms: int = 10,
        max_expanded_terms: int = 20,
        max_pmids: int = 100,
        max_abstracts: int = 20,
        mesh_score_threshold: float = 0.55,
        include_scope_notes: bool = True,
        qdrant_url: str | None = None,
        pg_dsn: str | None = None,
    ):
        """Initialize MeSH-KG-RAG client.

        Args:
            expansion_hops: How many hops to expand (0=none, 1=parent/child, 2=+siblings)
            max_mesh_terms: Max seed MeSH terms from query
            max_expanded_terms: Max terms after expansion
            max_pmids: Max PMIDs to retrieve
            max_abstracts: Max abstracts to include in context
            mesh_score_threshold: Min similarity score for MeSH matching
            include_scope_notes: Include MeSH definitions in context
            qdrant_url: Qdrant URL (defaults to config)
            pg_dsn: PostgreSQL DSN (defaults to config)
        """
        self._expansion_hops = expansion_hops
        self._max_mesh_terms = max_mesh_terms
        self._max_expanded_terms = max_expanded_terms
        self._max_pmids = max_pmids
        self._max_abstracts = max_abstracts
        self._mesh_score_threshold = mesh_score_threshold
        self._include_scope_notes = include_scope_notes

        config = get_config()
        self._qdrant_url = qdrant_url or config.qdrant.url
        self._pg_dsn = pg_dsn or config.postgres.pubmed_url

        self._qdrant: AsyncQdrantClient | None = None
        self._pool: asyncpg.Pool | None = None
        self._mesh_hierarchy: dict[str, MeSHTerm] = {}  # Cache for hierarchy lookup
        self._initialized = False

    async def __aenter__(self):
        """Initialize async resources."""
        self._qdrant = AsyncQdrantClient(self._qdrant_url, timeout=60)
        self._pool = await asyncpg.create_pool(
            self._pg_dsn,
            min_size=2,
            max_size=10,
        )
        self._initialized = True
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Close async resources."""
        if self._pool:
            await self._pool.close()
        if self._qdrant:
            await self._qdrant.close()
        self._initialized = False

    async def generate(
        self,
        question: str,
        question_type: str,
        context: list[str] | None = None,
        options: dict[str, str] | None = None,
    ) -> ModelResponse:
        """Generate answer using MeSH-KG-RAG pipeline.

        Args:
            question: Question text
            question_type: Type (yesno, mcq, factoid, list, summary)
            context: Optional golden context (ignored in realistic mode)
            options: MCQ options dict

        Returns:
            ModelResponse with answer and metadata
        """
        if not self._initialized:
            raise RuntimeError("Client not initialized. Use async context manager.")

        start_time = time.perf_counter()

        # Stage 1: Match query to MeSH terms
        mesh_start = time.perf_counter()
        seed_terms = await self._match_mesh_terms(question)
        mesh_match_ms = (time.perf_counter() - mesh_start) * 1000

        if not seed_terms:
            return ModelResponse(
                answer="No relevant MeSH terms found.",
                response_text="Could not match query to MeSH vocabulary.",
                latency_ms=(time.perf_counter() - start_time) * 1000,
                metadata={"error": "no_mesh_match"},
            )

        # Stage 2: Expand MeSH hierarchy
        expand_start = time.perf_counter()
        expanded_terms = await self._expand_hierarchy(seed_terms)
        expand_ms = (time.perf_counter() - expand_start) * 1000

        # Stage 3: Retrieve PMIDs via MeSH
        pmid_start = time.perf_counter()
        all_terms = seed_terms + expanded_terms
        pmids = await self._get_pmids_by_mesh(all_terms)
        pmid_ms = (time.perf_counter() - pmid_start) * 1000

        if not pmids:
            return ModelResponse(
                answer="No documents found for matched MeSH terms.",
                response_text="MeSH terms matched but no documents retrieved.",
                latency_ms=(time.perf_counter() - start_time) * 1000,
                metadata={
                    "error": "no_documents",
                    "mesh_terms": [t.descriptor_name for t in seed_terms],
                },
            )

        # Stage 4: Retrieve abstracts
        abstract_start = time.perf_counter()
        abstracts = await self._get_abstracts(pmids[:self._max_abstracts])
        abstract_ms = (time.perf_counter() - abstract_start) * 1000

        # Stage 5: Build structured context
        subgraph = MeSHSubgraph(
            seed_terms=seed_terms,
            expanded_terms=expanded_terms,
            pmids=pmids,
        )

        mesh_context = subgraph.to_context(include_scope=self._include_scope_notes)

        # Add document context
        doc_context = "\n## Retrieved Documents\n\n"
        for i, (pmid, title, abstract) in enumerate(abstracts[:self._max_abstracts], 1):
            # Truncate long abstracts
            abs_text = abstract[:500] + "..." if len(abstract) > 500 else abstract
            doc_context += f"### [{i}] PMID {pmid}\n**{title}**\n{abs_text}\n\n"

        full_context = mesh_context + doc_context

        # Stage 6: Generate answer
        llm_start = time.perf_counter()
        answer, answer_text = await generate_constrained_answer(
            question=question,
            question_type=question_type,
            context=full_context,
            options=options,
            temperature=0.1,
        )
        llm_ms = (time.perf_counter() - llm_start) * 1000

        total_ms = (time.perf_counter() - start_time) * 1000

        return ModelResponse(
            answer=answer,
            response_text=answer_text,
            latency_ms=total_ms,
            metadata={
                "strategy": "mesh_kg_rag",
                "expansion_hops": self._expansion_hops,
                "mesh_match_ms": mesh_match_ms,
                "expand_ms": expand_ms,
                "pmid_ms": pmid_ms,
                "abstract_ms": abstract_ms,
                "llm_ms": llm_ms,
                "num_seed_terms": len(seed_terms),
                "num_expanded_terms": len(expanded_terms),
                "num_pmids": len(pmids),
                "num_abstracts": len(abstracts),
                "seed_terms": [t.descriptor_name for t in seed_terms[:5]],
            },
        )

    async def _match_mesh_terms(self, query: str) -> list[MeSHTerm]:
        """Match query to MeSH terms via vector search."""
        # Embed query
        query_vector = await embed_client.embed_single(query)

        # Search MeSH collection
        results = await self._qdrant.query_points(
            collection_name="mesh-term-only",
            query=query_vector,
            limit=self._max_mesh_terms * 2,  # Get more, filter by score
            with_payload=True,
        )

        terms = []
        for point in results.points:
            if point.score < self._mesh_score_threshold:
                continue

            term = MeSHTerm(
                descriptor_ui=point.payload["descriptor_ui"],
                descriptor_name=point.payload["descriptor_name"],
                tree_numbers=point.payload.get("tree_numbers", []),
                scope_note=point.payload.get("scope_note", ""),
                entry_terms=point.payload.get("entry_terms", []),
                score=point.score,
                hop=0,
            )
            terms.append(term)

            # Cache for hierarchy lookup
            self._mesh_hierarchy[term.descriptor_ui] = term

            if len(terms) >= self._max_mesh_terms:
                break

        return terms

    async def _expand_hierarchy(self, seed_terms: list[MeSHTerm]) -> list[MeSHTerm]:
        """Expand MeSH terms via hierarchy (parent/child/sibling)."""
        if self._expansion_hops == 0:
            return []

        expanded = []
        seen_uis = {t.descriptor_ui for t in seed_terms}

        # Collect all tree numbers from seed terms
        all_tree_numbers = []
        for term in seed_terms:
            all_tree_numbers.extend(term.tree_numbers)

        if not all_tree_numbers:
            return []

        # Find related terms by tree number prefix
        # Parent: remove last segment (D02.355.291 → D02.355)
        # Child: terms with longer prefix
        # Sibling: same parent prefix

        parent_prefixes = set()
        sibling_prefixes = set()

        for tn in all_tree_numbers:
            parts = tn.split(".")
            if len(parts) > 1:
                parent_prefix = ".".join(parts[:-1])
                parent_prefixes.add(parent_prefix)
                sibling_prefixes.add(parent_prefix)  # Siblings share parent

        # Query Qdrant for terms matching these prefixes
        # We'll do a scroll to find terms with matching tree_numbers
        # This is a simplified approach - in production, you'd build an index

        # For efficiency, we search by embedding the parent concept names
        # and find similar terms
        if parent_prefixes:
            # Get parent terms by searching for tree_number prefixes
            expanded_terms = await self._find_terms_by_tree_prefix(
                list(parent_prefixes),
                seen_uis,
                hop=1,
            )
            expanded.extend(expanded_terms[:self._max_expanded_terms // 2])
            seen_uis.update(t.descriptor_ui for t in expanded_terms)

        # For hop=2, also get siblings
        if self._expansion_hops >= 2 and sibling_prefixes:
            sibling_terms = await self._find_sibling_terms(
                list(sibling_prefixes),
                seen_uis,
            )
            expanded.extend(sibling_terms[:self._max_expanded_terms // 2])

        return expanded[:self._max_expanded_terms]

    async def _find_terms_by_tree_prefix(
        self,
        prefixes: list[str],
        exclude_uis: set[str],
        hop: int,
    ) -> list[MeSHTerm]:
        """Find MeSH terms by tree number prefix."""
        # Use PostgreSQL to find terms with matching tree_numbers
        # This is more efficient than scanning Qdrant

        # For simplicity, we'll search Qdrant by embedding the prefix concepts
        # A full implementation would use a tree_number index

        terms = []

        # Scroll through mesh collection and filter by tree_number prefix
        # Limited implementation - in production use proper indexing
        offset = None
        batch_size = 100
        max_batches = 5

        for _ in range(max_batches):
            result = await self._qdrant.scroll(
                collection_name="mesh-term-only",
                limit=batch_size,
                offset=offset,
                with_payload=True,
                with_vectors=False,
            )

            points, next_offset = result

            for point in points:
                ui = point.payload["descriptor_ui"]
                if ui in exclude_uis:
                    continue

                tree_nums = point.payload.get("tree_numbers", [])

                # Check if any tree_number matches our prefixes
                for tn in tree_nums:
                    for prefix in prefixes:
                        if tn.startswith(prefix) or prefix.startswith(tn):
                            term = MeSHTerm(
                                descriptor_ui=ui,
                                descriptor_name=point.payload["descriptor_name"],
                                tree_numbers=tree_nums,
                                scope_note=point.payload.get("scope_note", ""),
                                entry_terms=point.payload.get("entry_terms", []),
                                score=0.8,  # Hierarchy match score
                                hop=hop,
                            )
                            terms.append(term)
                            exclude_uis.add(ui)
                            break
                    else:
                        continue
                    break

            if next_offset is None or len(terms) >= self._max_expanded_terms:
                break
            offset = next_offset

        return terms

    async def _find_sibling_terms(
        self,
        parent_prefixes: list[str],
        exclude_uis: set[str],
    ) -> list[MeSHTerm]:
        """Find sibling terms (same parent in hierarchy)."""
        return await self._find_terms_by_tree_prefix(
            parent_prefixes,
            exclude_uis,
            hop=2,
        )

    async def _get_pmids_by_mesh(self, terms: list[MeSHTerm]) -> list[str]:
        """Get PMIDs that have any of the given MeSH terms."""
        if not terms:
            return []

        descriptor_uis = [t.descriptor_ui for t in terms]
        placeholders = ", ".join(f"${i+1}" for i in range(len(descriptor_uis)))

        # Weight by hop distance and match count
        query = f"""
            WITH term_weights AS (
                SELECT unnest($1::text[]) as descriptor_ui,
                       unnest($2::float[]) as weight
            ),
            paper_scores AS (
                SELECT mh.pmid,
                       SUM(tw.weight) as score,
                       COUNT(DISTINCT mh.descriptor_ui) as match_count
                FROM mesh_headings mh
                JOIN term_weights tw ON mh.descriptor_ui = tw.descriptor_ui
                GROUP BY mh.pmid
            )
            SELECT pmid::text
            FROM paper_scores
            ORDER BY score DESC, match_count DESC
            LIMIT $3
        """

        # Assign weights based on hop distance
        weights = [1.0 / (1 + t.hop) for t in terms]  # hop=0: 1.0, hop=1: 0.5, hop=2: 0.33

        async with self._pool.acquire() as conn:
            rows = await conn.fetch(
                query,
                descriptor_uis,
                weights,
                self._max_pmids,
            )

        return [row["pmid"] for row in rows]

    async def _get_abstracts(
        self,
        pmids: list[str],
    ) -> list[tuple[str, str, str]]:
        """Get abstracts for given PMIDs."""
        if not pmids:
            return []

        # Convert to integers for query
        pmid_ints = [int(p) for p in pmids]

        query = """
            SELECT pmid::text, title, abstract
            FROM articles
            WHERE pmid = ANY($1)
            AND abstract IS NOT NULL
            AND abstract != ''
        """

        async with self._pool.acquire() as conn:
            rows = await conn.fetch(query, pmid_ints)

        # Maintain order from input
        pmid_to_row = {row["pmid"]: row for row in rows}
        results = []
        for pmid in pmids:
            if pmid in pmid_to_row:
                row = pmid_to_row[pmid]
                results.append((row["pmid"], row["title"], row["abstract"]))

        return results


# Factory function for consistency with retrieve_then_kg.py
def create_mesh_kg_rag_client(
    expansion_hops: int = 1,
    **kwargs,
) -> MeSHKGRAG:
    """Create MeSH-KG-RAG client.

    Args:
        expansion_hops: 0=no expansion, 1=parent/child, 2=+siblings
        **kwargs: Additional options

    Returns:
        MeSHKGRAG instance
    """
    return MeSHKGRAG(expansion_hops=expansion_hops, **kwargs)


# Register strategy names
MESH_KG_RAG_STRATEGIES = [
    "mesh_kg_rag_0hop",  # No hierarchy expansion
    "mesh_kg_rag_1hop",  # Parent/child expansion
    "mesh_kg_rag_2hop",  # +Sibling expansion
]


def create_mesh_kg_client_by_strategy(strategy: str, **kwargs) -> MeSHKGRAG:
    """Create MeSH-KG-RAG client by strategy name."""
    if strategy == "mesh_kg_rag_0hop":
        return MeSHKGRAG(expansion_hops=0, **kwargs)
    elif strategy == "mesh_kg_rag_1hop":
        return MeSHKGRAG(expansion_hops=1, **kwargs)
    elif strategy == "mesh_kg_rag_2hop":
        return MeSHKGRAG(expansion_hops=2, **kwargs)
    else:
        raise ValueError(f"Unknown MeSH-KG-RAG strategy: {strategy}")
