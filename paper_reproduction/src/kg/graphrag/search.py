"""Search implementations for GraphRAG.

Two search modes aligned with reference implementation:
- LocalSearch: Entity/relation focused with token-aware context
- GlobalSearch: Community report aggregation via map-reduce with JSON parsing

Reference: graphrag/query/structured_search/global_search/search.py
The GlobalSearch map/reduce prompt text is copied verbatim from Microsoft GraphRAG
(https://github.com/microsoft/graphrag), MIT licence, Copyright (c) Microsoft
Corporation; see ../../../THIRD_PARTY_NOTICES.txt.
"""

import asyncio
import json
import re
import time
import tiktoken
from dataclasses import dataclass, field
from typing import Any

from ..base import Entity, Relation, Community, QueryResult
from ..storage.graph_store import GraphStore
from ..storage.vector_store import KGVectorStore

try:
    from utils.clients import llm_client
except ImportError:
    from src.utils.clients import llm_client


# Tokenizer for token-aware context building
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
class SearchContext:
    """Context assembled for search."""
    entities: list[Entity]
    relations: list[Relation]
    communities: list[Community]
    context_text: str


@dataclass
class MapResponse:
    """Response from map phase with scoring.

    Reference: graphrag search returns structured points with scores.
    """
    points: list[dict[str, Any]] = field(default_factory=list)
    community_title: str = ""
    llm_calls: int = 0
    prompt_tokens: int = 0
    output_tokens: int = 0


@dataclass
class SearchMetrics:
    """Metrics for search operation.

    Reference: graphrag tracks llm_calls and token usage.
    """
    llm_calls: int = 0
    prompt_tokens: int = 0
    output_tokens: int = 0
    completion_time: float = 0.0


LOCAL_SEARCH_PROMPT = """Answer the question using the knowledge graph context provided.

## Community Reports
{community_reports}

## Entities
{entities}

## Relationships
{relations}

## Question
{question}

## Instructions
1. Focus on specific entities and relationships mentioned
2. Use community reports for high-level context
3. Use the detailed information provided
4. If the information is insufficient, say so

## Answer
"""


# Reference: graphrag/prompts/query/global_search_map_system_prompt.py
GLOBAL_MAP_PROMPT = """---Role---
You are a helpful assistant responding to questions about data in the tables provided.

---Goal---
Generate a response consisting of a list of key points that responds to the user's question, summarizing all relevant information in the input data tables.

You should use the data provided in the data tables below as the primary context for generating the response.
If you don't know the answer or if the input data tables do not contain sufficient information to provide an answer, just say so. Do not make anything up.

Each key point in the response should have the following element:
- Description: A comprehensive description of the point.
- Importance Score: An integer score between 0-100 that indicates how important the point is in answering the user's question. An 'I don't know' type of response should have a score of 0.

The response should be JSON formatted as follows:
{{
    "points": [
        {{"description": "Description of point 1 [Data: Reports (report ids)]", "score": score_value}},
        {{"description": "Description of point 2 [Data: Reports (report ids)]", "score": score_value}}
    ]
}}

---Data tables---
{context_data}

---Goal---
Generate a response consisting of a list of key points that responds to the user's question, summarizing all relevant information in the input data tables.

You should use the data provided in the data tables below as the primary context for generating the response.
If you don't know the answer or if the input data tables do not contain sufficient information to provide an answer, just say so. Do not make anything up.

The response shall be JSON formatted with a maximum of {max_length} words per point description.
"""


GLOBAL_REDUCE_PROMPT = """---Role---
You are a helpful assistant responding to questions about a dataset by synthesizing perspectives from multiple analysts.

---Goal---
Generate a response of the target length and format that responds to the user's question, summarize all the reports from the analysts who focused on different parts of the dataset.

Note that the analysts' reports provided below are ranked in the **descending order of importance**.

If you don't know the answer or if the provided reports do not contain sufficient information to provide an answer, just say so. Do not make anything up.

The final response should remove all irrelevant information from the analysts' reports and merge the cleaned information into a comprehensive answer that provides explanations of all the key points and implications appropriate for the response length and format.

Add sections and commentary to the response as appropriate for the length and format. Style the response in markdown.

The response shall be {response_type} with a maximum of {max_length} words.

---Target response length and format---
{response_type}

---Analyst Reports---
{report_data}

---Goal---
Generate a response of the target length and format that responds to the user's question, summarize all the reports from the analysts who focused on different parts of the dataset.

Note that the analysts' reports provided below are ranked in the **descending order of importance**.

If you don't know the answer or if the provided reports do not contain sufficient information to provide an answer, just say so. Do not make anything up.

The final response should remove all irrelevant information from the analysts' reports and merge the cleaned information into a comprehensive answer that provides explanations of all the key points and implications appropriate for the response length and format.

Add sections and commentary to the response as appropriate for the length and format. Style the response in markdown.
"""


NO_DATA_ANSWER = "I am sorry but I am unable to answer this question given the provided data."


class LocalSearch:
    """Entity/relation-focused local search with token-aware context.

    Reference: graphrag/query/structured_search/local_search/search.py
    """

    def __init__(
        self,
        graph_store: GraphStore,
        vector_store: KGVectorStore,
        max_context_tokens: int = 4000,
        community_prop: float = 0.4,
    ):
        """Initialize local search.

        Args:
            graph_store: Graph storage
            vector_store: Vector storage for semantic search
            max_context_tokens: Maximum tokens for context
            community_prop: Proportion of context tokens for community reports (default 0.4)
                           Aligned with reference: mixed context = community + entities + text units
        """
        self._graph = graph_store
        self._vectors = vector_store
        self._max_context_tokens = max_context_tokens
        self._community_prop = community_prop

    async def search(
        self,
        question: str,
        top_k: int = 20,
        question_type: str | None = None,
        options: dict[str, str] | None = None,
        context_only: bool = False,
    ) -> QueryResult:
        """Execute local search with token-aware context building.

        Args:
            question: User question
            top_k: Number of results
            question_type: Question type for constrained output (yesno/mcq/factoid/list)
            options: MCQ options dict
            context_only: If True, only build search context and skip LLM answer

        Returns:
            QueryResult with answer
        """
        start_time = time.perf_counter()
        metrics = SearchMetrics()

        # Find relevant entities
        entity_results = await self._vectors.search_entities(question, top_k=top_k)
        entities = [e for e, _ in entity_results]
        entity_names = {e.name for e in entities}

        # Get relations involving these entities
        relations = []
        for rel in self._graph.iter_relations():
            if rel.source in entity_names or rel.target in entity_names:
                relations.append(rel)
        relations = relations[:top_k]

        # Find communities containing these entities
        communities = []
        for community in self._graph.iter_communities():
            overlap = len(set(community.member_entities) & entity_names)
            if overlap > 0:
                communities.append((community, overlap))

        communities.sort(key=lambda x: x[1], reverse=True)
        communities = [c for c, _ in communities[:5]]

        # Build token-aware context with mixed context (aligned with reference)
        context = self._build_token_aware_context(
            entities, relations, communities
        )

        # Generate answer with constrained output if question_type provided
        # Mixed context includes community reports, entities, and relations
        context_text = f"{context['community_reports']}\n\n{context['entities']}\n\n{context['relations']}"

        if context_only:
            latency_ms = (time.perf_counter() - start_time) * 1000
            return QueryResult(
                answer="",
                entities=entities,
                relations=relations,
                communities=communities,
                context_text=context_text,
                latency_ms=latency_ms,
                mode="local",
                metadata={
                    "llm_calls": 0,
                    "prompt_tokens": 0,
                    "output_tokens": 0,
                    "context_only": True,
                },
            )

        if question_type:
            from ..answer_prompts import build_answer_messages, get_max_tokens

            system_prompt, user_prompt = build_answer_messages(
                question_type=question_type,
                question=question,
                context=context_text,
                options=options,
            )
            max_tokens = get_max_tokens(question_type)

            metrics.prompt_tokens = _count_tokens(user_prompt)
            answer = await llm_client.chat(
                prompt=user_prompt,
                system=system_prompt,
                max_tokens=max_tokens,
                temperature=0.1,
            )
        else:
            prompt = LOCAL_SEARCH_PROMPT.format(
                community_reports=context["community_reports"],
                entities=context["entities"],
                relations=context["relations"],
                question=question,
            )
            metrics.prompt_tokens = _count_tokens(prompt)
            answer = await llm_client.chat(prompt)

        metrics.output_tokens = _count_tokens(answer)
        metrics.llm_calls = 1

        latency_ms = (time.perf_counter() - start_time) * 1000

        return QueryResult(
            answer=answer,
            entities=entities,
            relations=relations,
            communities=communities,
            context_text=context_text,
            latency_ms=latency_ms,
            mode="local",
            metadata={
                "llm_calls": metrics.llm_calls,
                "prompt_tokens": metrics.prompt_tokens,
                "output_tokens": metrics.output_tokens,
                "context_only": False,
            },
        )

    def _build_token_aware_context(
        self,
        entities: list[Entity],
        relations: list[Relation],
        communities: list[Community],
    ) -> dict[str, str]:
        """Build context strings with token budget.

        Reference: graphrag uses token-budgeted context builders with mixed context.
        Aligned with reference: community_prop portion for community reports,
        remaining split between entities and relations.
        """
        # Allocate tokens: community_prop for communities, rest split between entities/relations
        max_community_tokens = int(self._max_context_tokens * self._community_prop)
        remaining_tokens = self._max_context_tokens - max_community_tokens
        max_entity_tokens = remaining_tokens // 2
        max_relation_tokens = remaining_tokens // 2

        # Build community context (aligned with reference mixed context)
        community_lines = []
        community_tokens = 0
        for c in communities:
            if c.summary:
                line = f"### {c.title}\n{c.summary[:300]}"
                line_tokens = _count_tokens(line)
                if community_tokens + line_tokens > max_community_tokens:
                    break
                community_lines.append(line)
                community_tokens += line_tokens

        # Build entity context with token tracking
        entity_lines = []
        entity_tokens = 0
        for e in entities:
            line = f"- **{e.name}** ({e.type}): {e.description[:200]}"
            line_tokens = _count_tokens(line)
            if entity_tokens + line_tokens > max_entity_tokens:
                break
            entity_lines.append(line)
            entity_tokens += line_tokens

        # Build relation context with token tracking
        relation_lines = []
        relation_tokens = 0
        for r in relations:
            line = f"- {r.source} --[{r.type}]--> {r.target}"
            if r.description:
                line += f": {r.description[:100]}"
            line_tokens = _count_tokens(line)
            if relation_tokens + line_tokens > max_relation_tokens:
                break
            relation_lines.append(line)
            relation_tokens += line_tokens

        return {
            "community_reports": "\n\n".join(community_lines) or "No community reports available",
            "entities": "\n".join(entity_lines) or "No entities found",
            "relations": "\n".join(relation_lines) or "No relations found",
        }


class GlobalSearch:
    """Community-based global search using map-reduce with JSON responses.

    Reference: graphrag/query/structured_search/global_search/search.py

    1. Map: Extract scored key points from each community report (JSON format)
    2. Reduce: Synthesize key points into final answer with token budget
    """

    def __init__(
        self,
        graph_store: GraphStore,
        vector_store: KGVectorStore,
        max_data_tokens: int = 8000,
        map_max_length: int = 1000,
        reduce_max_length: int = 2000,
        concurrent_coroutines: int = 32,
    ):
        """Initialize global search.

        Args:
            graph_store: Graph storage with communities
            vector_store: Vector storage for semantic search
            max_data_tokens: Maximum tokens for reduce context
            map_max_length: Max words per map response point
            reduce_max_length: Max words for reduce response
            concurrent_coroutines: Concurrent map calls
        """
        self._graph = graph_store
        self._vectors = vector_store
        self._max_data_tokens = max_data_tokens
        self._map_max_length = map_max_length
        self._reduce_max_length = reduce_max_length
        self._semaphore = asyncio.Semaphore(concurrent_coroutines)

    async def search(
        self,
        question: str,
        community_level: int = 1,
        max_communities: int = 10,
        response_type: str = "multiple paragraphs",
        question_type: str | None = None,
        options: dict[str, str] | None = None,
        context_only: bool = False,
    ) -> QueryResult:
        """Execute global search via map-reduce with JSON parsing.

        Reference: graphrag/query/structured_search/global_search/search.py:135-207

        Args:
            question: User question
            community_level: Which hierarchy level to use
            max_communities: Max communities to process
            response_type: Type of response to generate
            question_type: Question type for constrained output (yesno/mcq/factoid/list)
            options: MCQ options dict
            context_only: If True, only build reduce context and skip final answer

        Returns:
            QueryResult with synthesized answer
        """
        start_time = time.perf_counter()
        metrics = SearchMetrics()

        # Get communities at specified level
        communities = [
            c for c in self._graph.iter_communities()
            if c.level == community_level
        ]

        # Rank communities by potential relevance
        entity_results = await self._vectors.search_entities(question, top_k=20)
        query_entities = {e.name for e, _ in entity_results}

        ranked_communities = []
        for community in communities:
            overlap = len(set(community.member_entities) & query_entities)
            if overlap > 0 or community.summary:
                ranked_communities.append((community, overlap))

        ranked_communities.sort(
            key=lambda x: (x[1], len(x[0].member_entities)), reverse=True
        )
        selected = [c for c, _ in ranked_communities[:max_communities]]

        if not selected:
            selected = sorted(
                communities,
                key=lambda c: len(c.member_entities),
                reverse=True
            )[:max_communities]

        # Map phase: Extract key points with scores (parallel)
        map_tasks = [
            self._map_response_single(community, question)
            for community in selected
            if community.summary
        ]
        map_responses: list[MapResponse] = await asyncio.gather(*map_tasks)

        # Aggregate metrics from map phase
        for mr in map_responses:
            metrics.llm_calls += mr.llm_calls
            metrics.prompt_tokens += mr.prompt_tokens
            metrics.output_tokens += mr.output_tokens

        # Collect and filter key points
        all_points = []
        for i, mr in enumerate(map_responses):
            for point in mr.points:
                if point.get("score", 0) > 0:
                    all_points.append({
                        "analyst": i,
                        "description": point.get("description", ""),
                        "score": point.get("score", 0),
                        "community": mr.community_title,
                    })

        # Sort by score descending
        all_points.sort(key=lambda x: x["score"], reverse=True)

        if not all_points:
            return QueryResult(
                answer="" if context_only else NO_DATA_ANSWER,
                communities=selected,
                context_text="",
                latency_ms=(time.perf_counter() - start_time) * 1000,
                mode="global",
                metadata={
                    "llm_calls": metrics.llm_calls,
                    "prompt_tokens": metrics.prompt_tokens,
                    "output_tokens": metrics.output_tokens,
                    "context_only": context_only,
                },
            )

        # Build reduce context with token budget
        reduce_context = self._build_reduce_context(all_points)

        # Collect entities from selected communities
        entity_names = set()
        for c in selected:
            entity_names.update(c.member_entities[:10])

        entities = []
        for name in list(entity_names)[:30]:
            entity = self._graph.get_entity(name)
            if entity:
                entities.append(entity)

        if context_only:
            return QueryResult(
                answer="",
                entities=entities,
                communities=selected,
                context_text=reduce_context,
                latency_ms=(time.perf_counter() - start_time) * 1000,
                mode="global",
                metadata={
                    "num_communities_processed": len(selected),
                    "num_key_points": len(all_points),
                    "community_level": community_level,
                    "llm_calls": metrics.llm_calls,
                    "prompt_tokens": metrics.prompt_tokens,
                    "output_tokens": metrics.output_tokens,
                    "context_only": True,
                },
            )

        # Reduce phase: Synthesize answer with constrained output if question_type provided
        if question_type:
            from ..answer_prompts import build_answer_messages, get_max_tokens

            system_prompt, user_prompt = build_answer_messages(
                question_type=question_type,
                question=question,
                context=reduce_context,
                options=options,
            )
            max_tokens = get_max_tokens(question_type)

            reduce_prompt_tokens = _count_tokens(user_prompt)
            answer = await llm_client.chat(
                prompt=user_prompt,
                system=system_prompt,
                max_tokens=max_tokens,
                temperature=0.1,
            )
        else:
            reduce_prompt = GLOBAL_REDUCE_PROMPT.format(
                report_data=reduce_context,
                response_type=response_type,
                max_length=self._reduce_max_length,
            )
            reduce_prompt_tokens = _count_tokens(reduce_prompt)
            answer = await llm_client.chat(f"{reduce_prompt}\n\nQuestion: {question}")

        reduce_output_tokens = _count_tokens(answer)

        metrics.llm_calls += 1
        metrics.prompt_tokens += reduce_prompt_tokens
        metrics.output_tokens += reduce_output_tokens

        latency_ms = (time.perf_counter() - start_time) * 1000

        return QueryResult(
            answer=answer,
            entities=entities,
            communities=selected,
            context_text=reduce_context,
            latency_ms=latency_ms,
            mode="global",
            metadata={
                "num_communities_processed": len(selected),
                "num_key_points": len(all_points),
                "community_level": community_level,
                "llm_calls": metrics.llm_calls,
                "prompt_tokens": metrics.prompt_tokens,
                "output_tokens": metrics.output_tokens,
                "context_only": False,
            },
        )

    async def _map_response_single(
        self,
        community: Community,
        question: str,
    ) -> MapResponse:
        """Generate map response for a single community.

        Reference: graphrag/query/structured_search/global_search/search.py:209-264

        Returns JSON-formatted points with scores.
        """
        async with self._semaphore:
            context_data = f"Title: {community.title}\n\n{community.summary}"

            prompt = GLOBAL_MAP_PROMPT.format(
                context_data=context_data,
                max_length=self._map_max_length,
            )

            prompt_tokens = _count_tokens(prompt)

            try:
                response = await llm_client.chat(
                    f"{prompt}\n\nQuestion: {question}"
                )
                output_tokens = _count_tokens(response)

                # Parse JSON response
                points = self._parse_map_response(response)

                return MapResponse(
                    points=points,
                    community_title=community.title,
                    llm_calls=1,
                    prompt_tokens=prompt_tokens,
                    output_tokens=output_tokens,
                )

            except Exception:
                return MapResponse(
                    points=[{"description": "", "score": 0}],
                    community_title=community.title,
                    llm_calls=1,
                    prompt_tokens=prompt_tokens,
                    output_tokens=0,
                )

    def _parse_map_response(self, response: str) -> list[dict[str, Any]]:
        """Parse JSON map response.

        Reference: graphrag/query/structured_search/global_search/search.py:266-294
        """
        # Try to extract JSON from response
        json_match = re.search(r"\{[\s\S]*\}", response)
        if not json_match:
            return [{"description": "", "score": 0}]

        try:
            data = json.loads(json_match.group())
            points = data.get("points", [])

            if not isinstance(points, list):
                return [{"description": "", "score": 0}]

            return [
                {
                    "description": p.get("description", ""),
                    "score": int(p.get("score", 0)),
                }
                for p in points
                if isinstance(p, dict) and "description" in p and "score" in p
            ]

        except (json.JSONDecodeError, ValueError):
            return [{"description": "", "score": 0}]

    def _build_reduce_context(self, points: list[dict]) -> str:
        """Build reduce context with token budget.

        Reference: graphrag/query/structured_search/global_search/search.py:351-370
        """
        data = []
        total_tokens = 0

        for point in points:
            formatted_lines = [
                f"----Analyst {point['analyst'] + 1}----",
                f"Importance Score: {point['score']}",
                point["description"],
            ]
            formatted_text = "\n".join(formatted_lines)
            formatted_tokens = _count_tokens(formatted_text)

            if total_tokens + formatted_tokens > self._max_data_tokens:
                break

            data.append(formatted_text)
            total_tokens += formatted_tokens

        return "\n\n".join(data)
