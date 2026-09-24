"""BioRAG benchmark adapter.

This adapter follows the structure of the reference BioRAG code under
`.ref_project/biorag`:

1. Planner creates a multi-step tool-use plan
2. Router classifies each step into a tool
3. Tool-specific query rewriting
4. Tool execution
5. Final response generation from accumulated tool outputs

Differences from the reference implementation:
- Replaces LangChain chains with direct calls to the project's load-balanced
  `llm_client`
- Replaces BioPython Entrez dependencies with local PubMed retrieval plus
  in-repo gene/web tools that do not depend on NCBI
- Reuses in-repo retrieval utilities for PubMed/MeSH, gene normalization,
  UniProt lookup, and web search

The goal is to stay faithful to the original control flow while making it
benchmark-compatible in this repository.
"""

from __future__ import annotations

import asyncio
import json
import re
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import asyncpg
from qdrant_client import AsyncQdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchAny, SearchParams

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent.parent
_SRC_PATH = _PROJECT_ROOT / "src"

if str(_SRC_PATH) not in sys.path:
    sys.path.insert(0, str(_SRC_PATH))
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from .client_protocol import ModelClient, ModelResponse
from .shared_pipeline import (
    DenseEvidenceDoc,
    format_pubmed_evidence,
    generate_constrained_answer,
    retrieve_dense_evidence,
)
from config import get_config
from tools.gene_resolver import GeneResolver
from tools.uniprot_resolver import UniProtResolver
from tools.web_searcher import DuckDuckGoSearcher
from utils.clients import embed_client, llm_client


PLANNER_SYSTEM = """
Before we begin, break the biomedical QA problem into a concise plan.
Available tools:
- Web Search
- PubMed Search
- Gene
- dbSNP
- Response

Return ONLY a JSON object inside a markdown code block:
{
  "step_1": {"tool": "Suggested tool", "task": "Task description"},
  "step_2": {"tool": "Suggested tool", "task": "Task description"}
}

Use Response exactly once as the final step.
"""

ROUTER_SYSTEM = """
Classify the task into exactly one category:
- Web Search
- PubMed Search
- Gene
- dbSNP
- Response

Return the category only, with no explanation.
"""

PUBMED_REWRITE_SYSTEM = """
Rewrite the user task into a concise, retrieval-friendly PubMed query.
Keep the biomedical intent intact. Return only the rewritten query.
"""

WEB_REWRITE_SYSTEM = """
Rewrite the user task into concise web search keywords.
Return only the rewritten query.
"""

GENE_REWRITE_SYSTEM = """
Extract the gene symbol or gene name needed for lookup.
Return only the gene query string.
"""

DBSNP_REWRITE_SYSTEM = """
Extract the variant identifier needed for lookup.
Prefer an rsID when present, e.g. rs12345.
Return only the variant query string.
"""

ROUTE_LABELS = {
    "web search": "Web Search",
    "pubmed search": "PubMed Search",
    "gene": "Gene",
    "dbsnp": "dbSNP",
    "response": "Response",
}

_TRANSIENT_TOOL_ERROR_PATTERNS = (
    re.compile(r"\b408\b"),
    re.compile(r"\b409\b"),
    re.compile(r"\b425\b"),
    re.compile(r"\b429\b"),
    re.compile(r"\b500\b"),
    re.compile(r"\b502\b"),
    re.compile(r"\b503\b"),
    re.compile(r"\b504\b"),
    re.compile(r"request timeout", re.IGNORECASE),
    re.compile(r"item timeout", re.IGNORECASE),
    re.compile(r"timed out", re.IGNORECASE),
    re.compile(r"timeout", re.IGNORECASE),
    re.compile(r"connection error", re.IGNORECASE),
    re.compile(r"connection reset", re.IGNORECASE),
    re.compile(r"network is unreachable", re.IGNORECASE),
    re.compile(r"temporarily unavailable", re.IGNORECASE),
    re.compile(r"broken pipe", re.IGNORECASE),
    re.compile(r"service unavailable", re.IGNORECASE),
    re.compile(r"too many requests", re.IGNORECASE),
)


def _is_transient_tool_error(message: str) -> bool:
    if not message:
        return False
    return any(pattern.search(message) for pattern in _TRANSIENT_TOOL_ERROR_PATTERNS)

SYSTEM_PROMPTS = {
    "yesno": (
        "You are a strict answer formatter.\n"
        "Output EXACTLY one word: yes or no.\n"
        "Lowercase only. No punctuation, no explanation, no extra words."
    ),
    "mcq": (
        "You are a strict answer formatter.\n"
        "Output EXACTLY one uppercase letter from the provided options (A, B, C, D, etc.).\n"
        "No punctuation, no explanation, no extra words."
    ),
    "factoid": (
        "You are a strict answer formatter.\n"
        "Output a single short biomedical entity or phrase (max 6 words).\n"
        "No complete sentence, no explanation, no extra punctuation.\n"
        "If evidence is insufficient, output: unknown"
    ),
    "list": (
        "You are a strict answer formatter.\n"
        "Output a comma-separated list of biomedical entities.\n"
        "No brackets, no bullets, no numbering, no explanation.\n"
        "Each item should be 1-5 words.\n"
        "If no items are supported by evidence, output: none"
    ),
}

USER_PROMPTS = {
    "yesno": (
        "Based on the evidence below, answer the question.\n\n"
        "Question: {question}\n\n"
        "Evidence:\n{context}\n\n"
        "Answer (yes or no):"
    ),
    "mcq": (
        "Based on the evidence below, select the correct option.\n\n"
        "Question: {question}\n\n"
        "Options:\n{options}\n\n"
        "Evidence:\n{context}\n\n"
        "Answer (single letter):"
    ),
    "factoid": (
        "Based on the evidence below, answer the question with a short phrase.\n\n"
        "Question: {question}\n\n"
        "Evidence:\n{context}\n\n"
        "Answer:"
    ),
    "list": (
        "Based on the evidence below, list all relevant items.\n\n"
        "Question: {question}\n\n"
        "Evidence:\n{context}\n\n"
        "Answer (comma-separated):"
    ),
}

MAX_TOKENS = {
    "yesno": 4,
    "mcq": 4,
    "factoid": 32,
    "list": 128,
}


@dataclass
class MeSHMatch:
    descriptor_ui: str
    descriptor_name: str
    score: float = 0.0
    tree_numbers: list[str] = field(default_factory=list)


@dataclass
class RetrievedAbstract:
    pmid: str
    title: str
    abstract: str
    score: float
    rank: int
    have_fulltext: bool = False
    paper_uuid: str | None = None
    pmc: str | None = None
    matched_mesh: list[str] = field(default_factory=list)


class MeSHFilter:
    """Minimal MeSH filter copied from src/retriever/mesh_filter.py.

    It is kept local here to avoid importing the full retriever package,
    whose package-level side effects pull in GraphRAG dependencies.
    """

    def __init__(
        self,
        qdrant_client: AsyncQdrantClient,
        db_pool: asyncpg.Pool,
        mesh_collection: str = "mesh-term-only",
        top_k: int = 20,
        llm_select: int = 8,
    ):
        self._qdrant = qdrant_client
        self._pool = db_pool
        self._collection = mesh_collection
        self._top_k = top_k
        self._llm_select = llm_select

    async def match_mesh_terms(
        self,
        query: str,
        entities: list[str] | None = None,
    ) -> tuple[list[MeSHMatch], float]:
        start_time = time.perf_counter()
        search_terms = entities if entities else self._extract_search_terms(query)
        if not search_terms:
            return [], (time.perf_counter() - start_time) * 1000

        embeddings = await embed_client.embed(search_terms)

        seen_uis: set[str] = set()
        all_candidates: list[tuple[float, dict[str, Any]]] = []
        for term_embedding in embeddings:
            search_response = await self._qdrant.query_points(
                collection_name=self._collection,
                query=term_embedding,
                limit=5,
                with_payload=True,
                search_params=SearchParams(hnsw_ef=128),
            )

            for point in search_response.points:
                if not point.payload:
                    continue
                ui = point.payload["descriptor_ui"]
                if point.score >= 0.75 and ui not in seen_uis:
                    seen_uis.add(ui)
                    all_candidates.append((point.score, point.payload))

        if not all_candidates:
            return await self._match_mesh_terms_fullquery(query, start_time)

        all_candidates.sort(key=lambda x: x[0], reverse=True)
        matches = [
            MeSHMatch(
                descriptor_ui=payload["descriptor_ui"],
                descriptor_name=payload["descriptor_name"],
                score=score,
                tree_numbers=payload.get("tree_numbers", []),
            )
            for score, payload in all_candidates[: self._llm_select]
        ]
        return matches, (time.perf_counter() - start_time) * 1000

    async def _match_mesh_terms_fullquery(
        self,
        query: str,
        start_time: float,
    ) -> tuple[list[MeSHMatch], float]:
        query_vector = await embed_client.embed_single(query)
        search_response = await self._qdrant.query_points(
            collection_name=self._collection,
            query=query_vector,
            limit=self._top_k,
            with_payload=True,
            search_params=SearchParams(hnsw_ef=128),
        )

        matches = []
        for point in search_response.points[: self._llm_select]:
            if not point.payload:
                continue
            matches.append(
                MeSHMatch(
                    descriptor_ui=point.payload["descriptor_ui"],
                    descriptor_name=point.payload["descriptor_name"],
                    score=point.score,
                    tree_numbers=point.payload.get("tree_numbers", []),
                )
            )
        return matches, (time.perf_counter() - start_time) * 1000

    def _extract_search_terms(self, query: str) -> list[str]:
        stopwords = {
            "are", "is", "do", "does", "the", "a", "an", "in", "of", "with",
            "or", "and", "to", "for", "by", "on", "at", "from", "as",
        }
        words = re.findall(r"\b\w+\b", query.lower())
        return [w for w in words if len(w) > 2 and w not in stopwords][:10]

    async def get_pmids_by_mesh(
        self,
        matches: list[MeSHMatch],
        limit: int = 10000,
        max_term_papers: int = 500000,
        return_ordered: bool = False,
    ) -> tuple[set[str] | list[str], float]:
        if not matches:
            return [] if return_ordered else set(), 0.0

        start_time = time.perf_counter()
        descriptor_uis = [m.descriptor_ui for m in matches]
        placeholders = ", ".join(f"${i+1}" for i in range(len(descriptor_uis)))

        query = f"""
            WITH term_counts AS (
                SELECT descriptor_ui, COUNT(DISTINCT pmid) as cnt
                FROM mesh_headings
                WHERE descriptor_ui IN ({placeholders})
                GROUP BY descriptor_ui
                HAVING COUNT(DISTINCT pmid) <= {max_term_papers}
            ),
            paper_scores AS (
                SELECT mh.pmid, COUNT(DISTINCT mh.descriptor_ui) as match_count
                FROM mesh_headings mh
                JOIN term_counts tc ON mh.descriptor_ui = tc.descriptor_ui
                GROUP BY mh.pmid
            )
            SELECT pmid::text
            FROM paper_scores
            ORDER BY match_count DESC, pmid DESC
            LIMIT {limit}
        """

        async with self._pool.acquire() as conn:
            rows = await conn.fetch(query, *descriptor_uis)

        pmids = [row["pmid"] for row in rows] if return_ordered else {row["pmid"] for row in rows}
        return pmids, (time.perf_counter() - start_time) * 1000


class VectorSearch:
    """Minimal vector search copied from src/retriever/vector_search.py."""

    def __init__(
        self,
        qdrant_client: AsyncQdrantClient,
        collection: str = "paper-full",
    ):
        self._qdrant = qdrant_client
        self._collection = collection

    async def search(
        self,
        query: str,
        pmid_filter: set[str] | None = None,
        limit: int = 50,
    ) -> tuple[list[RetrievedAbstract], float]:
        start_time = time.perf_counter()
        query_vector = await embed_client.embed_single(query)

        qdrant_filter = None
        if pmid_filter:
            qdrant_filter = Filter(
                must=[
                    FieldCondition(
                        key="pmid",
                        match=MatchAny(any=list(pmid_filter)),
                    )
                ]
            )

        search_response = await self._qdrant.query_points(
            collection_name=self._collection,
            query=query_vector,
            query_filter=qdrant_filter,
            limit=limit,
            with_payload=True,
            search_params=SearchParams(hnsw_ef=128),
        )

        abstracts = []
        for rank, point in enumerate(search_response.points, start=1):
            payload = point.payload or {}
            abstracts.append(
                RetrievedAbstract(
                    pmid=str(payload.get("pmid", "")),
                    title=payload.get("title") or "",
                    abstract=payload.get("abstract") or "",
                    score=point.score,
                    rank=rank,
                    have_fulltext=bool(payload.get("pmc")),
                    paper_uuid=None,
                    pmc=payload.get("pmc"),
                    matched_mesh=[],
                )
            )
        return abstracts, (time.perf_counter() - start_time) * 1000


def _format_options(options: dict[str, str] | None) -> str:
    if not options:
        return ""
    return "\n".join(f"{k}. {options[k]}" for k in sorted(options))


def _build_answer_messages(
    question_type: str,
    question: str,
    context: str,
    options: dict[str, str] | None = None,
) -> tuple[str, str]:
    qt = question_type if question_type in SYSTEM_PROMPTS else "factoid"
    system = SYSTEM_PROMPTS[qt]
    user = USER_PROMPTS[qt].format(
        question=question,
        context=context[:8000],
        options=_format_options(options),
    )
    return system, user


def _get_max_tokens(question_type: str) -> int:
    return MAX_TOKENS.get(question_type, 64)


def _extract_constrained_answer(
    response: str,
    question_type: str,
    options: dict[str, str] | None = None,
) -> str:
    response = response.strip()

    if question_type == "yesno":
        lowered = response.lower()
        if "yes" in lowered:
            return "yes"
        if "no" in lowered:
            return "no"
        first = lowered.split()[0] if lowered else ""
        return "yes" if first.startswith("y") else "no"

    if question_type == "mcq":
        response_upper = response.upper()
        valid_keys = set(options.keys()) if options else set("ABCDE")
        for char in response_upper:
            if char in valid_keys:
                return char
        return sorted(valid_keys)[0] if valid_keys else "A"

    if question_type == "factoid":
        first_line = response.split("\n")[0].strip()
        for prefix in ["Answer:", "The answer is", "It is"]:
            if first_line.lower().startswith(prefix.lower()):
                first_line = first_line[len(prefix):].strip()
        return first_line[:100]

    if question_type == "list":
        cleaned = response.replace("\n", ", ")
        cleaned = re.sub(r"^\s*[-*\d.)\]]+\s*", "", cleaned, flags=re.MULTILINE)
        return cleaned[:200]

    return response[:500]


def _extract_code_block(text: str) -> str:
    match = re.search(r"```(?:json)?\s*(.*?)```", text, re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1).strip()
    return text.strip()


def _json_from_text(text: str) -> dict[str, Any] | None:
    raw = _extract_code_block(text)
    try:
        data = json.loads(raw)
        return data if isinstance(data, dict) else None
    except json.JSONDecodeError:
        return None


def _strip_fences(text: str) -> str:
    return _extract_code_block(text).strip().strip("`").strip()


class BioRAGClient(ModelClient):
    """Benchmark-compatible BioRAG adapter."""

    def __init__(
        self,
        max_steps: int = 4,
        max_mesh_terms: int = 8,
        max_pmids: int = 5000,
        max_abstracts: int = 4,
        evidence_top_k: int = 20,
        qdrant_url: str | None = None,
        pg_dsn: str | None = None,
        ncbi_timeout: float = 20.0,
        tool_retries: int = 2,
        tool_backoff_base: float = 1.5,
        tool_timeout: float = 60.0,
        shared_retrieval_only: bool = True,
    ):
        config = get_config()
        self._max_steps = max_steps
        self._max_pmids = max_pmids
        self._max_abstracts = max_abstracts
        self._evidence_top_k = max(1, evidence_top_k)
        self._qdrant_url = qdrant_url or config.qdrant.url
        self._pg_dsn = pg_dsn or config.postgres.pubmed_url
        self._ncbi_timeout = ncbi_timeout
        self._max_mesh_terms = max_mesh_terms
        self._tool_retries = max(0, tool_retries)
        self._tool_backoff_base = max(1.0, tool_backoff_base)
        self._tool_timeout = max(1.0, tool_timeout)
        self._shared_retrieval_only = shared_retrieval_only

        self._qdrant: AsyncQdrantClient | None = None
        self._pool: asyncpg.Pool | None = None
        self._mesh_filter: MeSHFilter | None = None
        self._vector_search: VectorSearch | None = None
        self._gene_resolver: GeneResolver | None = None
        self._uniprot_resolver: UniProtResolver | None = None
        self._web_searcher: DuckDuckGoSearcher | None = None
        self._initialized = False

    async def _call_tool_with_retries(
        self,
        route: str,
        query: str,
        tool_fn,
    ) -> tuple[str, dict[str, Any]]:
        max_attempts = max(1, self._tool_retries + 1)
        last_error_type = "RuntimeError"
        last_error_message = "unknown error"

        for attempt in range(1, max_attempts + 1):
            started = time.perf_counter()
            try:
                content, meta = await asyncio.wait_for(tool_fn(query), timeout=self._tool_timeout)
                elapsed_ms = (time.perf_counter() - started) * 1000
                meta = dict(meta or {})
                meta["route"] = route
                meta["attempt"] = attempt
                meta["max_attempts"] = max_attempts
                meta["tool_latency_ms"] = elapsed_ms
                return content, meta
            except Exception as exc:
                elapsed_ms = (time.perf_counter() - started) * 1000
                last_error_type = exc.__class__.__name__
                message = str(exc).strip() or repr(exc)
                last_error_message = message
                transient = _is_transient_tool_error(message)

                if transient and attempt < max_attempts:
                    backoff = self._tool_backoff_base ** (attempt - 1)
                    await asyncio.sleep(backoff)
                    continue

                return "", {
                    "route": route,
                    "attempt": attempt,
                    "max_attempts": max_attempts,
                    "tool_latency_ms": elapsed_ms,
                    "tool_failed": True,
                    "tool_error_type": last_error_type,
                    "tool_error": last_error_message[:300],
                    "tool_error_transient": transient,
                }

        return "", {
            "route": route,
            "attempt": max_attempts,
            "max_attempts": max_attempts,
            "tool_failed": True,
            "tool_error_type": last_error_type,
            "tool_error": last_error_message[:300],
            "tool_error_transient": _is_transient_tool_error(last_error_message),
        }

    @staticmethod
    def _rank_shared_docs(query: str, docs: list[DenseEvidenceDoc]) -> list[DenseEvidenceDoc]:
        """Lightweight lexical re-ranking within fixed shared evidence pool."""
        tokens = {tok for tok in re.findall(r"[a-z0-9]+", query.lower()) if len(tok) >= 3}
        if not tokens:
            return docs

        def score(doc: DenseEvidenceDoc) -> float:
            text = f"{doc.title} {doc.abstract}".lower()
            overlap = sum(1 for tok in tokens if tok in text)
            # keep dense rank order influence via original score
            return overlap * 10.0 + doc.score

        return sorted(docs, key=score, reverse=True)

    async def _tool_shared_evidence(
        self,
        query: str,
        shared_docs: list[DenseEvidenceDoc],
    ) -> tuple[str, dict[str, Any]]:
        ranked = self._rank_shared_docs(query, shared_docs)
        selected = ranked[: self._max_abstracts]
        content = format_pubmed_evidence(selected, max_docs=self._max_abstracts)
        return content, {
            "source": "shared_dense_evidence",
            "num_abstracts": len(selected),
            "candidate_pool": len(shared_docs),
            "evidence_top_k": self._evidence_top_k,
        }

    async def __aenter__(self):
        self._qdrant = AsyncQdrantClient(self._qdrant_url, timeout=60)
        self._pool = await asyncpg.create_pool(
            self._pg_dsn,
            min_size=2,
            max_size=10,
        )
        self._mesh_filter = MeSHFilter(
            self._qdrant,
            self._pool,
            llm_select=self._max_mesh_terms,
        )
        self._vector_search = VectorSearch(self._qdrant, collection="paper-full")
        self._gene_resolver = GeneResolver(timeout=self._ncbi_timeout)
        self._uniprot_resolver = UniProtResolver(timeout=self._ncbi_timeout)
        self._web_searcher = DuckDuckGoSearcher(timeout=self._ncbi_timeout)
        self._initialized = True
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self._gene_resolver:
            await self._gene_resolver.close()
        if self._uniprot_resolver:
            await self._uniprot_resolver.close()
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
        if not self._initialized:
            raise RuntimeError("Client not initialized. Use async context manager.")

        start_time = time.perf_counter()
        retrieval_meta: dict[str, Any] = {
            "retrieval_ms": 0.0,
            "embed_ms": 0.0,
            "search_ms": 0.0,
            "abstract_ms": 0.0,
            "source": "none",
            "docs_retrieved": 0,
            "docs_loaded": 0,
        }
        shared_docs: list[DenseEvidenceDoc] = []

        if self._shared_retrieval_only:
            if context:
                shared_docs = [
                    DenseEvidenceDoc(
                        pmid=f"golden_context_{i}",
                        title=f"Golden Context {i + 1}",
                        abstract=ctx,
                        score=2.0,
                        rank=i,
                    )
                    for i, ctx in enumerate(context)
                ]
                retrieval_meta = {
                    "retrieval_ms": 0.0,
                    "embed_ms": 0.0,
                    "search_ms": 0.0,
                    "abstract_ms": 0.0,
                    "source": "golden_context",
                    "docs_retrieved": len(shared_docs),
                    "docs_loaded": len(shared_docs),
                }
            else:
                shared_docs, retrieval_meta = await retrieve_dense_evidence(
                    question,
                    qdrant=self._qdrant,
                    pool=self._pool,
                    top_k=self._evidence_top_k,
                    collection="paper-full",
                )
                retrieval_meta = dict(retrieval_meta)
                retrieval_meta["source"] = "shared_dense_stage1"

        plan = await self._plan(question)
        steps = self._normalize_plan(plan, question)

        tool_contexts: list[str] = []
        tool_runs: list[dict[str, Any]] = []

        async def _shared_tool(query_text: str) -> tuple[str, dict[str, Any]]:
            return await self._tool_shared_evidence(query_text, shared_docs)

        for idx, step in enumerate(steps[: self._max_steps], start=1):
            route = await self._route(step.get("task", question), step.get("tool"))
            if route == "Response":
                break

            rewritten = await self._rewrite_query(step["task"], route)
            if self._shared_retrieval_only:
                content, meta = await self._call_tool_with_retries(
                    route=route,
                    query=rewritten,
                    tool_fn=_shared_tool,
                )
            elif route == "PubMed Search":
                content, meta = await self._call_tool_with_retries(
                    route=route,
                    query=rewritten,
                    tool_fn=self._tool_pubmed_search,
                )
            elif route == "Gene":
                content, meta = await self._call_tool_with_retries(
                    route=route,
                    query=rewritten,
                    tool_fn=self._tool_gene,
                )
            elif route == "dbSNP":
                content, meta = await self._call_tool_with_retries(
                    route=route,
                    query=rewritten,
                    tool_fn=self._tool_dbsnp,
                )
            elif route == "Web Search":
                content, meta = await self._call_tool_with_retries(
                    route=route,
                    query=rewritten,
                    tool_fn=self._tool_web_search,
                )
            else:
                continue

            tool_runs.append(
                {
                    "step": idx,
                    "route": route,
                    "task": step.get("task", ""),
                    "query": rewritten,
                    **meta,
                }
            )
            if content.strip():
                tool_contexts.append(f"### Step {idx}: {route}\n{content}")

        if not tool_contexts:
            if self._shared_retrieval_only and shared_docs:
                fallback_content, meta = await self._tool_shared_evidence(question, shared_docs)
                meta = dict(meta)
                meta["attempt"] = 1
                meta["max_attempts"] = 1
                meta["tool_latency_ms"] = 0.0
            else:
                fallback_content, meta = await self._call_tool_with_retries(
                    route="PubMed Search",
                    query=question,
                    tool_fn=self._tool_pubmed_search,
                )
            tool_runs.append({"step": 1, "route": "PubMed Search", "task": question, "query": question, **meta})
            if fallback_content.strip():
                tool_contexts.append(f"### Step 1: PubMed Search\n{fallback_content}")

        if not tool_contexts:
            tool_contexts.append("No external evidence retrieved due to transient tool errors.")

        full_context = "\n\n".join(tool_contexts)

        answer, answer_text = await generate_constrained_answer(
            question=question,
            question_type=question_type,
            context=full_context,
            options=options,
            temperature=0.1,
        )

        total_ms = (time.perf_counter() - start_time) * 1000
        return ModelResponse(
            answer=answer,
            response_text=answer_text,
            latency_ms=total_ms,
            metadata={
                "strategy": "biorag",
                "shared_retrieval_only": self._shared_retrieval_only,
                "evidence_top_k": self._evidence_top_k,
                "num_steps": len(tool_runs),
                "routes": [r["route"] for r in tool_runs],
                "tool_runs": tool_runs,
                "retrieval_ms": retrieval_meta.get("retrieval_ms"),
                "embed_ms": retrieval_meta.get("embed_ms"),
                "search_ms": retrieval_meta.get("search_ms"),
                "abstract_ms": retrieval_meta.get("abstract_ms"),
                "retrieval_source": retrieval_meta.get("source"),
                "docs_retrieved": retrieval_meta.get("docs_retrieved"),
                "docs_loaded": retrieval_meta.get("docs_loaded"),
            },
        )

    async def _plan(self, question: str) -> dict[str, Any] | None:
        try:
            text = await llm_client.chat(
                prompt=question,
                system=PLANNER_SYSTEM,
                max_tokens=512,
                temperature=0.0,
            )
            return _json_from_text(text)
        except Exception:
            return None

    def _normalize_plan(
        self,
        plan: dict[str, Any] | None,
        question: str,
    ) -> list[dict[str, str]]:
        if not plan:
            return [
                {"tool": "PubMed Search", "task": question},
                {"tool": "Response", "task": "Answer the user question from the gathered context."},
            ]

        steps: list[tuple[int, dict[str, str]]] = []
        for key, value in plan.items():
            if not isinstance(value, dict):
                continue
            match = re.search(r"(\d+)", key)
            order = int(match.group(1)) if match else 999
            tool = str(value.get("tool", "")).strip() or "PubMed Search"
            task = str(value.get("task", "")).strip() or question
            steps.append((order, {"tool": tool, "task": task}))

        steps.sort(key=lambda x: x[0])
        normalized = [s for _, s in steps]
        if not normalized:
            normalized = [{"tool": "PubMed Search", "task": question}]
        if normalized[-1]["tool"].lower() != "response":
            normalized.append(
                {"tool": "Response", "task": "Answer the user question from the gathered context."}
            )
        return normalized

    async def _route(self, task: str, suggested_tool: str | None = None) -> str:
        suggested = (suggested_tool or "").strip().lower()
        if suggested in ROUTE_LABELS:
            return ROUTE_LABELS[suggested]

        lowered = task.lower()
        if re.search(r"\brs\d+\b", lowered):
            return "dbSNP"
        if any(tok in lowered for tok in ["guideline", "guidelines", "recommendation", "website", "web search"]):
            return "Web Search"

        try:
            route = await llm_client.chat(
                prompt=task,
                system=ROUTER_SYSTEM,
                max_tokens=16,
                temperature=0.0,
            )
            route = _strip_fences(route)
            for label in ROUTE_LABELS.values():
                if label.lower() in route.lower():
                    return label
        except Exception:
            pass

        if any(tok in lowered for tok in ["gene", "protein", "symbol", "alias"]):
            return "Gene"
        return "PubMed Search"

    async def _rewrite_query(self, task: str, route: str) -> str:
        system = PUBMED_REWRITE_SYSTEM
        if route == "Web Search":
            system = WEB_REWRITE_SYSTEM
        elif route == "Gene":
            system = GENE_REWRITE_SYSTEM
        elif route == "dbSNP":
            system = DBSNP_REWRITE_SYSTEM

        try:
            rewritten = await llm_client.chat(
                prompt=task,
                system=system,
                max_tokens=128,
                temperature=0.0,
            )
            rewritten = _strip_fences(rewritten)
            return rewritten or task
        except Exception:
            return task

    async def _tool_pubmed_search(self, query: str) -> tuple[str, dict[str, Any]]:
        assert self._mesh_filter is not None
        assert self._vector_search is not None

        mesh_matches, mesh_ms = await self._mesh_filter.match_mesh_terms(query)

        pmids: set[str] | list[str]
        pmids = set()
        pmid_ms = 0.0
        if mesh_matches:
            pmids, pmid_ms = await self._mesh_filter.get_pmids_by_mesh(
                mesh_matches,
                limit=self._max_pmids,
            )

        abstracts, vector_ms = await self._vector_search.search(
            query,
            pmid_filter=set(pmids) if pmids else None,
            limit=self._max_abstracts,
        )

        lines = []
        if mesh_matches:
            lines.append("Matched MeSH Terms:")
            for m in mesh_matches[: self._max_mesh_terms]:
                lines.append(f"- {m.descriptor_name} ({m.descriptor_ui}, score={m.score:.3f})")
            lines.append("")

        lines.append("PubMed Evidence:")
        for i, abs_ in enumerate(abstracts, start=1):
            abstract = abs_.abstract[:1200]
            lines.append(f"[{i}] PMID {abs_.pmid}")
            lines.append(f"Title: {abs_.title}")
            lines.append(f"Abstract: {abstract}")
            lines.append("")

        if not abstracts:
            lines.append("No PubMed evidence found.")

        return (
            "\n".join(lines).strip(),
            {
                "mesh_terms": [m.descriptor_name for m in mesh_matches[: self._max_mesh_terms]],
                "num_pmids": len(pmids) if pmids else 0,
                "num_abstracts": len(abstracts),
                "mesh_ms": mesh_ms,
                "pmid_ms": pmid_ms,
                "vector_ms": vector_ms,
            },
        )

    async def _tool_gene(self, query: str) -> tuple[str, dict[str, Any]]:
        assert self._gene_resolver is not None
        assert self._uniprot_resolver is not None

        symbol = await self._gene_resolver.resolve(query)
        search_term = symbol or query.strip()
        hits = await self._uniprot_resolver.search_proteins(search_term, organism_id=9606, limit=3)
        record = await self._uniprot_resolver.get_protein(hits[0].accession) if hits else None

        pubmed_content = ""
        pubmed_meta: dict[str, Any] = {}
        try:
            pubmed_content, pubmed_meta = await self._tool_pubmed_search(f"{search_term} gene")
        except Exception:
            pubmed_content = ""

        lines = []
        if symbol and symbol.upper() != query.upper():
            lines.append(f"Resolved gene symbol: {symbol}")
            lines.append("")

        if hits:
            lines.append("UniProt hits:")
            for hit in hits:
                genes = ", ".join(hit.gene_names[:5])
                lines.append(
                    f"- {hit.accession}: {hit.protein_name} | genes={genes} | organism={hit.organism_name}"
                )
            lines.append("")

        if record and record.function and record.function.text:
            lines.append("UniProt function:")
            lines.append(record.function.text[:1600].strip())
            lines.append("")

        if record and record.diseases:
            lines.append("UniProt diseases:")
            for disease in record.diseases[:5]:
                desc = (disease.description or "")[:220]
                lines.append(f"- {disease.name}: {desc}")
            lines.append("")

        if record and record.go_annotations:
            lines.append("UniProt GO annotations:")
            for go in record.go_annotations[:8]:
                lines.append(f"- {go.go_id} [{go.aspect}] {go.term}")
            lines.append("")

        if pubmed_content.strip():
            lines.append("Local PubMed evidence:")
            lines.append(pubmed_content[:2200].strip())
        elif not lines:
            lines.append("No useful gene information found.")

        result_text = "\n".join(lines).strip()
        return result_text, {
            "resolved_symbol": symbol,
            "uniprot_accessions": [hit.accession for hit in hits],
            "pubmed_abstracts": pubmed_meta.get("num_abstracts", 0),
        }

    async def _tool_dbsnp(self, query: str) -> tuple[str, dict[str, Any]]:
        rs_match = re.search(r"\brs(\d+)\b", query, re.IGNORECASE)
        rsid = f"rs{rs_match.group(1)}" if rs_match else query.strip()

        pubmed_content = ""
        pubmed_meta: dict[str, Any] = {}
        try:
            pubmed_content, pubmed_meta = await self._tool_pubmed_search(f"{rsid} variant polymorphism")
        except Exception:
            pubmed_content = ""

        web_content = ""
        web_meta: dict[str, Any] = {}
        try:
            web_content, web_meta = await self._tool_web_search(f"{rsid} SNP variant")
        except Exception:
            web_content = ""

        lines = [f"Variant query: {rsid}", ""]
        if pubmed_content.strip():
            lines.append("Local PubMed evidence:")
            lines.append(pubmed_content[:2200].strip())
            lines.append("")
        if web_content.strip():
            lines.append("Web evidence:")
            lines.append(web_content[:1800].strip())
        if len(lines) <= 2:
            lines.append("No useful variant information found.")

        return "\n".join(lines).strip(), {
            "rs_query": rsid,
            "pubmed_abstracts": pubmed_meta.get("num_abstracts", 0),
            "web_results": web_meta.get("num_results", 0),
            "source": "local_pubmed_plus_web",
        }

    async def _tool_web_search(self, query: str) -> tuple[str, dict[str, Any]]:
        if self._web_searcher is None:
            raise RuntimeError("Web searcher not initialized")

        results = await self._web_searcher.search(query, limit=10)
        lines = []
        for r in results:
            lines.append(f"- {r.get('title', '')}")
            lines.append(f"  URL: {r.get('url', '')}")
            lines.append(f"  Snippet: {r.get('snippet', '')}")
        if not lines:
            lines.append("No useful web results found.")
        return "\n".join(lines), {"num_results": len(results)}


def create_biorag_client(strategy: str = "biorag", **kwargs) -> ModelClient:
    """Factory for BioRAG benchmark client."""
    if strategy not in {"biorag", "biorag_pubmed"}:
        raise ValueError(f"Unknown BioRAG strategy: {strategy}")
    return BioRAGClient(**kwargs)
