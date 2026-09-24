"""Query execution for LightRAG.

Implements 4 query modes aligned with reference implementation:
- naive: Chunk-vector retrieval (baseline)
- local: Entity-centric with low-level keywords
- global: Relation-centric with high-level keywords
- hybrid: Combined local + global with round-robin merge

Reference: LightRAG/lightrag/operate.py
"""

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Literal, Protocol
import tiktoken

from ..base import Entity, Relation, QueryResult
from ..storage.graph_store import GraphStore
from ..storage.vector_store import KGVectorStore
from ..extraction.keywords import extract_keywords, ExtractedKeywords

try:
    from config import get_config
    from utils.clients import llm_client
except ImportError:
    from src.config import get_config
    from src.utils.clients import llm_client


class QueryMode(str, Enum):
    """LightRAG query modes."""
    NAIVE = "naive"
    LOCAL = "local"
    GLOBAL = "global"
    HYBRID = "hybrid"


class ChunkSearcher(Protocol):
    """Protocol for chunk-based vector search.

    Reference: LightRAG naive mode uses chunk vectors, not entity vectors.
    Implementations should return (chunk_text, score) tuples.
    """

    async def search(self, query: str, limit: int) -> list[tuple[str, float]]:
        """Search chunks by semantic similarity.

        Args:
            query: Search query
            limit: Maximum results

        Returns:
            List of (chunk_text, score) tuples
        """
        ...


@dataclass
class QueryConfig:
    """Configuration for query execution."""
    top_k: int = 20
    max_entity_tokens: int = 4000
    max_relation_tokens: int = 4000
    max_chunk_tokens: int = 4000
    max_total_tokens: int = 12000
    include_chunk_evidence: bool = True


@dataclass
class QueryContext:
    """Context assembled for LLM generation."""
    entities: list[Entity] = field(default_factory=list)
    relations: list[Relation] = field(default_factory=list)
    chunks: list[dict] = field(default_factory=list)  # {id, text, score}
    context_text: str = ""
    mode: str = "hybrid"
    keywords: ExtractedKeywords | None = None


# Tokenizer for context truncation
_tokenizer = None

def _get_tokenizer():
    global _tokenizer
    if _tokenizer is None:
        _tokenizer = tiktoken.get_encoding("cl100k_base")
    return _tokenizer


def _count_tokens(text: str) -> int:
    """Count tokens in text."""
    return len(_get_tokenizer().encode(text))


def _truncate_by_tokens(text: str, max_tokens: int) -> str:
    """Truncate text to fit within token limit."""
    tokens = _get_tokenizer().encode(text)
    if len(tokens) <= max_tokens:
        return text
    return _get_tokenizer().decode(tokens[:max_tokens])


def _truncate_list_by_tokens(
    items: list[tuple[str, float]],  # (text, score)
    max_tokens: int,
) -> list[tuple[str, float]]:
    """Truncate list of items to fit within token limit."""
    result = []
    total_tokens = 0

    for text, score in items:
        item_tokens = _count_tokens(text)
        if total_tokens + item_tokens > max_tokens:
            # Truncate this item to fit
            remaining = max_tokens - total_tokens
            if remaining > 50:  # Minimum useful size
                truncated = _truncate_by_tokens(text, remaining)
                result.append((truncated, score))
            break
        result.append((text, score))
        total_tokens += item_tokens

    return result


RAG_RESPONSE_PROMPT = """---Role---
You are an expert AI assistant answering questions using the provided knowledge graph context.

---Context---
{context}

---Question---
{question}

---Instructions---
1. Use ONLY information from the context to answer
2. If the context doesn't contain enough information, say so clearly
3. Be concise and accurate
4. For biomedical questions, cite specific entities and relationships when relevant

---Answer---
"""


NAIVE_RESPONSE_PROMPT = """---Role---
You are an expert AI assistant synthesizing information from document chunks.

---Document Chunks---
{chunks}

---Question---
{question}

---Instructions---
1. Use ONLY information from the document chunks
2. Cite specific chunks when relevant
3. If insufficient information, acknowledge gaps
4. Be concise and accurate

---Answer---
"""


class QueryExecutor:
    """Execute LightRAG queries across different modes.

    Implements the reference query pipeline:
    1. Extract keywords (high-level and low-level)
    2. Search entities/relations based on mode
    3. Optionally retrieve supporting chunks
    4. Build token-aware context
    5. Generate answer with LLM

    Example:
        >>> executor = QueryExecutor(graph_store, vector_store)
        >>> result = await executor.query("What treats diabetes?", mode="hybrid")
        >>> print(result.answer)
    """

    def __init__(
        self,
        graph_store: GraphStore,
        vector_store: KGVectorStore,
        config: QueryConfig | None = None,
        chunk_searcher: ChunkSearcher | None = None,
    ):
        """Initialize query executor.

        Args:
            graph_store: Graph storage with entities/relations
            vector_store: Vector storage for semantic search
            config: Query configuration
            chunk_searcher: Optional chunk searcher for naive mode (aligned with reference)
        """
        self._graph = graph_store
        self._vectors = vector_store
        self._config = config or QueryConfig()
        self._chunk_searcher = chunk_searcher

    async def query(
        self,
        question: str,
        mode: Literal["naive", "local", "global", "hybrid"] = "hybrid",
        top_k: int | None = None,
        question_type: str | None = None,
        options: dict[str, str] | None = None,
        context_only: bool = False,
    ) -> QueryResult:
        """Execute query in specified mode.

        Args:
            question: Natural language question
            mode: Query mode
            top_k: Override default top_k
            question_type: Question type for constrained output (yesno/mcq/factoid/list)
            options: MCQ options dict (e.g., {"A": "option1", "B": "option2"})
            context_only: If True, only build context and skip final LLM answer

        Returns:
            QueryResult with answer and supporting context
        """
        start_time = time.perf_counter()
        config_top_k = top_k or self._config.top_k

        if mode == "naive":
            context = await self._naive_query(question, config_top_k)
        elif mode == "local":
            context = await self._local_query(question, config_top_k)
        elif mode == "global":
            context = await self._global_query(question, config_top_k)
        elif mode == "hybrid":
            context = await self._hybrid_query(question, config_top_k)
        else:
            raise ValueError(f"Unknown query mode: {mode}")

        answer = ""
        if not context_only:
            # Generate answer with constrained output if question_type provided
            answer = await self._generate_answer(
                question, context, question_type=question_type, options=options
            )

        latency_ms = (time.perf_counter() - start_time) * 1000

        return QueryResult(
            answer=answer,
            entities=context.entities,
            relations=context.relations,
            context_text=context.context_text,
            latency_ms=latency_ms,
            mode=mode,
            metadata={
                "keywords": {
                    "high_level": context.keywords.high_level if context.keywords else [],
                    "low_level": context.keywords.low_level if context.keywords else [],
                },
                "num_chunks": len(context.chunks),
                "context_only": context_only,
            },
        )

    async def _naive_query(self, question: str, top_k: int) -> QueryContext:
        """Chunk-vector retrieval (baseline).

        Reference: operate.py:4735-4910
        Simply retrieves most similar chunks without entity/relation search.

        Aligned with reference: Uses chunk vectors if chunk_searcher is provided,
        otherwise falls back to entity embeddings as proxy.
        """
        chunks = []
        entities = []

        # Use chunk_searcher if available (aligned with reference implementation)
        if self._chunk_searcher is not None:
            chunk_results = await self._chunk_searcher.search(question, limit=top_k)
            for chunk_text, score in chunk_results:
                chunks.append({
                    "id": f"chunk_{len(chunks)}",
                    "text": chunk_text,
                    "score": score,
                })
        else:
            # Fallback: use entity embeddings as proxy (original behavior)
            entity_results = await self._vectors.search_entities(question, top_k=top_k)
            entities = [e for e, _ in entity_results]

            for entity, score in entity_results:
                chunk_text = f"**{entity.name}** ({entity.type}): {entity.description}"
                chunks.append({
                    "id": entity.id,
                    "text": chunk_text,
                    "score": score,
                })

        # Truncate chunks by token limit
        chunk_items = [(c["text"], c["score"]) for c in chunks]
        truncated = _truncate_list_by_tokens(chunk_items, self._config.max_chunk_tokens)
        chunks = [{"text": t, "score": s} for t, s in truncated]

        context_text = self._format_naive_context(chunks)

        return QueryContext(
            entities=entities,
            chunks=chunks,
            context_text=context_text,
            mode="naive",
        )

    async def _local_query(self, question: str, top_k: int) -> QueryContext:
        """Entity-centric query with low-level keywords.

        Reference: operate.py:3424-3535
        1. Extract low-level keywords (entities, specific terms)
        2. Search entities matching keywords
        3. Expand with 1-hop neighbors
        4. Get relations between entities
        5. Build token-aware context
        """
        # Step 1: Extract keywords
        keywords = await extract_keywords(question)

        # Step 2: Search entities using low-level keywords
        search_query = " ".join(keywords.low_level) if keywords.low_level else question
        entity_results = await self._vectors.search_entities(search_query, top_k=top_k)

        seed_entities = [e for e, _ in entity_results]
        seed_names = {e.name for e in seed_entities}

        # Step 3: Expand with 1-hop neighbors
        expanded_names = set(seed_names)
        for entity in seed_entities[:10]:  # Limit expansion
            neighbors = self._graph.get_neighbors(entity.name, max_hops=1)
            for hop_entities in neighbors.values():
                for neighbor in hop_entities:
                    expanded_names.add(neighbor.name)

        # Limit total entities
        expanded_names = set(list(expanded_names)[:top_k * 2])

        # Step 4: Get full entity data
        entities = []
        for name in expanded_names:
            entity = self._graph.get_entity(name)
            if entity:
                entities.append(entity)

        # Step 5: Get relations between these entities
        # Rank by edge weight/degree (reference uses degree)
        relations = []
        for rel in self._graph.iter_relations():
            if rel.source in expanded_names and rel.target in expanded_names:
                relations.append(rel)

        # Sort by weight (proxy for importance)
        relations.sort(key=lambda r: r.weight, reverse=True)
        relations = relations[:top_k]

        # Step 6: Build token-aware context
        context_text = self._build_kg_context(
            entities,
            relations,
            self._config.max_entity_tokens,
            self._config.max_relation_tokens,
        )

        return QueryContext(
            entities=entities,
            relations=relations,
            context_text=context_text,
            mode="local",
            keywords=keywords,
        )

    async def _global_query(self, question: str, top_k: int) -> QueryContext:
        """Relation-centric query with high-level keywords.

        Reference: operate.py:3424-3535 (global path)
        1. Extract high-level keywords (concepts, themes)
        2. Search relations matching keywords
        3. Extract entities from relations
        4. Build context emphasizing relationships
        """
        # Step 1: Extract keywords
        keywords = await extract_keywords(question)

        # Step 2: Search relations using high-level keywords
        search_query = " ".join(keywords.high_level) if keywords.high_level else question
        relation_results = await self._vectors.search_relations(search_query, top_k=top_k)

        relations = [r for r, _ in relation_results]

        # Step 3: Extract entities from relations
        entity_names = set()
        for rel in relations:
            entity_names.add(rel.source)
            entity_names.add(rel.target)

        entities = []
        for name in entity_names:
            entity = self._graph.get_entity(name)
            if entity:
                entities.append(entity)

        # Step 4: Build context emphasizing relationships
        context_text = self._build_kg_context(
            entities,
            relations,
            self._config.max_entity_tokens // 2,  # Less space for entities
            self._config.max_relation_tokens * 2,  # More space for relations
        )

        return QueryContext(
            entities=entities,
            relations=relations,
            context_text=context_text,
            mode="global",
            keywords=keywords,
        )

    async def _hybrid_query(self, question: str, top_k: int) -> QueryContext:
        """Combined local + global with round-robin merge.

        Reference: operate.py:3489-3515
        1. Run both local and global searches with full top_k
        2. Merge results with round-robin selection
        3. Deduplicate entities/relations
        4. Build combined context

        Aligned with reference: Both sub-queries use full top_k, then
        round-robin merge limits final results to top_k.
        """
        # Run both queries with full top_k (aligned with reference)
        local_ctx = await self._local_query(question, top_k)
        global_ctx = await self._global_query(question, top_k)

        # Round-robin merge entities
        entities = self._round_robin_merge(
            local_ctx.entities,
            global_ctx.entities,
            key=lambda e: e.name,
            max_items=top_k,
        )

        # Round-robin merge relations
        relations = self._round_robin_merge(
            local_ctx.relations,
            global_ctx.relations,
            key=lambda r: r.id,
            max_items=top_k,
        )

        # Build hybrid context
        context_text = self._build_kg_context(
            entities,
            relations,
            self._config.max_entity_tokens,
            self._config.max_relation_tokens,
        )

        return QueryContext(
            entities=entities,
            relations=relations,
            context_text=context_text,
            mode="hybrid",
            keywords=local_ctx.keywords,  # Use local keywords
        )

    def _round_robin_merge(
        self,
        list1: list,
        list2: list,
        key,
        max_items: int,
    ) -> list:
        """Merge two lists with round-robin selection and deduplication.

        Reference: operate.py uses this pattern for combining local/global results.
        """
        seen = set()
        result = []
        i1, i2 = 0, 0

        while len(result) < max_items and (i1 < len(list1) or i2 < len(list2)):
            # Alternate between lists
            if i1 < len(list1):
                item = list1[i1]
                item_key = key(item)
                if item_key not in seen:
                    seen.add(item_key)
                    result.append(item)
                i1 += 1

            if len(result) >= max_items:
                break

            if i2 < len(list2):
                item = list2[i2]
                item_key = key(item)
                if item_key not in seen:
                    seen.add(item_key)
                    result.append(item)
                i2 += 1

        return result

    def _build_kg_context(
        self,
        entities: list[Entity],
        relations: list[Relation],
        max_entity_tokens: int,
        max_relation_tokens: int,
    ) -> str:
        """Build token-aware KG context.

        Reference: operate.py:4048-4235
        Structures context with entities and relations, truncating to fit limits.
        """
        lines = ["## Knowledge Graph Context\n"]

        # Format entities with token tracking
        if entities:
            lines.append("### Entities\n")
            entity_items = []
            for e in entities:
                text = f"- **{e.name}** ({e.type}): {e.description[:200]}"
                entity_items.append((text, 1.0))

            truncated = _truncate_list_by_tokens(entity_items, max_entity_tokens)
            for text, _ in truncated:
                lines.append(text)

        # Format relations with token tracking
        if relations:
            lines.append("\n### Relationships\n")
            relation_items = []
            for r in relations:
                keywords_str = ", ".join(r.keywords[:3]) if r.keywords else ""
                text = f"- {r.source} --[{r.type}]--> {r.target}"
                if r.description:
                    text += f": {r.description[:150]}"
                if keywords_str:
                    text += f" (Keywords: {keywords_str})"
                relation_items.append((text, r.weight))

            truncated = _truncate_list_by_tokens(relation_items, max_relation_tokens)
            for text, _ in truncated:
                lines.append(text)

        return "\n".join(lines)

    def _format_naive_context(self, chunks: list[dict]) -> str:
        """Format naive mode context."""
        lines = ["## Retrieved Information\n"]
        for i, chunk in enumerate(chunks, 1):
            lines.append(f"[{i}] {chunk['text']}")
        return "\n".join(lines)

    async def _generate_answer(
        self,
        question: str,
        context: QueryContext,
        question_type: str | None = None,
        options: dict[str, str] | None = None,
    ) -> str:
        """Generate answer using LLM.

        Args:
            question: User question
            context: Assembled context
            question_type: Question type for constrained output (yesno/mcq/factoid/list)
            options: MCQ options dict

        Returns:
            Generated answer text
        """
        # Use constrained prompts if question_type is provided
        if question_type:
            from ..answer_prompts import build_answer_messages, get_max_tokens

            system_prompt, user_prompt = build_answer_messages(
                question_type=question_type,
                question=question,
                context=context.context_text,
                options=options,
            )
            max_tokens = get_max_tokens(question_type)

            return await llm_client.chat(
                prompt=user_prompt,
                system=system_prompt,
                max_tokens=max_tokens,
                temperature=0.1,  # Low temperature for constrained output
            )

        # Fallback to original prompts for unconstrained generation
        if context.mode == "naive":
            prompt = NAIVE_RESPONSE_PROMPT.format(
                chunks=context.context_text,
                question=question,
            )
        else:
            prompt = RAG_RESPONSE_PROMPT.format(
                context=context.context_text,
                question=question,
            )

        # Truncate prompt if needed
        total_tokens = _count_tokens(prompt)
        if total_tokens > self._config.max_total_tokens:
            # Truncate context portion
            max_context_tokens = self._config.max_total_tokens - 500  # Reserve for question/instructions
            context.context_text = _truncate_by_tokens(context.context_text, max_context_tokens)

            if context.mode == "naive":
                prompt = NAIVE_RESPONSE_PROMPT.format(
                    chunks=context.context_text,
                    question=question,
                )
            else:
                prompt = RAG_RESPONSE_PROMPT.format(
                    context=context.context_text,
                    question=question,
                )

        return await llm_client.chat(prompt)
