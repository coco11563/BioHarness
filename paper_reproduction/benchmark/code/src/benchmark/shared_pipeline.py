"""Shared benchmark pipeline components.

This module provides:
1. Dense evidence retrieval (shared stage-1 across methods)
2. Shared constrained answer generation

Goal: make baseline methods comparable by sharing the same retrieval corpus
and final answer formatting path, while allowing method-specific context
processing in the middle.
"""

from __future__ import annotations

import asyncio
import os
import logging
import time
from dataclasses import dataclass
from typing import Any

import asyncpg
from qdrant_client import AsyncQdrantClient, models

from config import get_config
from kg.answer_prompts import (
    build_answer_messages,
    extract_constrained_answer,
    get_max_tokens,
)
from utils.clients import embed_client, llm_client

# Character cap on the evidence block handed to the constrained answerer. 8000 was
# sized for abstracts; with 20 full-text chunks the evidence block is ~25k chars, so
# the rejudge that produces the FINAL answer saw ~6 of 20 documents while stage 1 saw
# all of them uncapped. XC_ANSWER_CONTEXT_CHARS lifts it; default unchanged.
_ANSWER_CONTEXT_CHARS = int(os.environ.get("XC_ANSWER_CONTEXT_CHARS", "8000"))

_logger = logging.getLogger(__name__)

# Shared failover pool: one AsyncQdrantClient per configured tunnel URL.
# On transport error, rotate to the next client so a single dropped SSH
# forward doesn't kill the whole benchmark run.
_qdrant_failover_clients: list[AsyncQdrantClient] | None = None


def _get_failover_qdrant_clients() -> list[AsyncQdrantClient]:
    global _qdrant_failover_clients
    if _qdrant_failover_clients is None:
        cfg = get_config()
        _qdrant_failover_clients = [
            AsyncQdrantClient(u, timeout=60, check_compatibility=False)
            for u in cfg.qdrant.urls
        ]
    return _qdrant_failover_clients


_DEFAULT_CONTEXT_TOKENS = int(os.environ.get("XC_CONTEXT_TOKENS", "4000"))


@dataclass
class DenseEvidenceDoc:
    pmid: str
    title: str
    abstract: str
    score: float
    rank: int


def _payload_has_abstract(payload: dict[str, Any]) -> bool:
    """Return whether a Qdrant payload contains a usable abstract."""
    abstract = payload.get("abstract")
    return isinstance(abstract, str) and bool(abstract.strip())


def _title_only_fallback(title: str, pmid: str) -> str:
    """Construct a minimal text payload when only a title is available."""
    clean_title = (title or "").strip()
    if not clean_title:
        return f"Title-only PubMed record for PMID {pmid}."
    return f"Title-only PubMed record for PMID {pmid}: {clean_title}"


def _candidate_search_limits(top_k: int) -> list[int]:
    """Search wider only when top-k hits are dominated by title-only records."""
    limits = [max(top_k, 1)]
    for candidate in (max(top_k * 5, 100), max(top_k * 10, 200)):
        if candidate > limits[-1]:
            limits.append(candidate)
    return limits


async def _query_candidate_points(
    *,
    qdrant: AsyncQdrantClient,
    collection: str,
    query_vector: list[float],
    top_k: int,
) -> tuple[list[Any], float, int, int]:
    """Query Qdrant and widen the search window if abstracts are sparse."""
    total_search_ms = 0.0
    last_points: list[Any] = []
    limits = _candidate_search_limits(top_k)

    # Build failover chain: primary qdrant first, then the shared pool clients.
    failover = [qdrant] + [c for c in _get_failover_qdrant_clients() if c is not qdrant]

    for expansion_idx, limit in enumerate(limits, start=1):
        search_start = time.perf_counter()
        results = None
        last_err: Exception | None = None
        # All failover clients ride the same SSH tunnel, so a tunnel blip fails
        # every one of them at once; retry the whole chain with backoff before
        # giving up (one such blip killed a 2-hour MedXpertQA run, 2026-09-08).
        for attempt in range(4):
            for client in failover:
                try:
                    results = await client.query_points(
                        collection_name=collection,
                        query=query_vector,
                        limit=limit,
                        with_payload=["pmid", "title", "abstract"],
                    )
                    break
                except Exception as e:
                    last_err = e
                    _logger.warning(
                        "Qdrant query_points failed on %s (%s); failing over",
                        getattr(client, "_client", client),
                        type(e).__name__,
                    )
                    await asyncio.sleep(0.5)
            if results is not None:
                break
            await asyncio.sleep(min(30.0, 3.0 * (2 ** attempt)))
        if results is None:
            raise last_err  # all failovers exhausted
        total_search_ms += (time.perf_counter() - search_start) * 1000

        last_points = list(results.points) if results and results.points else []
        abstract_count = sum(1 for point in last_points if _payload_has_abstract(point.payload or {}))
        if abstract_count >= top_k:
            return last_points, total_search_ms, limit, expansion_idx

    return last_points, total_search_ms, limits[-1], len(limits)


async def _fetch_pg_abstracts(
    pool: asyncpg.Pool,
    pmids: list[str],
) -> list[tuple[str, str, str, float]]:
    """Fetch abstracts from PostgreSQL while preserving input PMID order.

    The score semantics intentionally match benchmark/dense_rag.py historical
    behavior (rank-based pseudo score) to avoid changing old Dense formatting.
    """
    if not pmids:
        return []

    pmid_scores = {pmid: 1.0 - i * 0.01 for i, pmid in enumerate(pmids)}
    pmid_ints = [int(p) for p in pmids if p.isdigit()]
    if not pmid_ints:
        return []

    query = """
        SELECT pmid::text, title, abstract
        FROM articles
        WHERE pmid = ANY($1)
        AND abstract IS NOT NULL
        AND abstract != ''
    """
    async with pool.acquire() as conn:
        rows = await conn.fetch(query, pmid_ints)

    pmid_to_row = {row["pmid"]: row for row in rows}
    results: list[tuple[str, str, str, float]] = []
    for pmid in pmids:
        if pmid in pmid_to_row:
            row = pmid_to_row[pmid]
            results.append((row["pmid"], row["title"], row["abstract"], pmid_scores.get(pmid, 0.5)))
    return results


_FT_POOL = None


async def retrieve_dense_evidence(
    question: str,
    *,
    qdrant: AsyncQdrantClient,
    pool: asyncpg.Pool,
    top_k: int,
    collection: str = "paper-full",
    query_vector: list[float] | None = None,
) -> tuple[list[DenseEvidenceDoc], dict[str, Any]]:
    """Shared dense stage-1 retrieval.

    When ``query_vector`` is provided it is used as-is (e.g. HyDE). Otherwise
    the question string is embedded with ``embed_client``.

    XC_FULLTEXT_CHUNKS=1 swaps the abstract index for the PMC full-text chunk
    store. LitQA2 answers are attested once in the primary literature and are by
    construction absent from abstracts; on the 75 LitQA2 items whose source paper
    is in the chunk store, direct chunk search recovers the gold passage for
    62.7% of them versus 36.0% for abstract-retrieval-then-PMID-filter (abstract
    vectors recall the source paper for only 48% even at k=200). Chunks are
    adapted to DenseEvidenceDoc so every downstream consumer is unchanged.

    Returns:
        (evidence_docs, retrieval_meta)
    """
    import os as _os
    if _os.environ.get("XC_FULLTEXT_CHUNKS") == "1":
        from src.retriever.chunk_search import ChunkSearch
        # The caller's pool points at paper-graph-pubmed (abstracts); chunk text
        # lives in the separate papergraph database, so open (and cache) our own.
        global _FT_POOL
        try:
            _FT_POOL
        except NameError:
            _FT_POOL = None
        if _FT_POOL is None:
            import asyncpg as _apg
            from src.config import get_config as _gc
            _FT_POOL = await _apg.create_pool(
                _gc().postgres.papergraph_url, min_size=1, max_size=8, timeout=60)
        cs = ChunkSearch(qdrant_client=qdrant, db_pool=_FT_POOL, collection="chunks")
        chunks, ms = await cs.search(question, limit=top_k)
        docs = [
            DenseEvidenceDoc(
                pmid=str(getattr(c, "pmid", "") or ""),
                title=f"Full-text excerpt (PMC{getattr(c, 'pmcid', '') or '?'})",
                abstract=(getattr(c, "text", "") or ""),
                score=float(getattr(c, "score", 0.0) or 0.0),
                rank=i + 1,
            )
            for i, c in enumerate(chunks)
        ]
        return docs, {"source": "chunks", "latency_ms": ms, "n": len(docs)}

    embed_start = time.perf_counter()
    if query_vector is None:
        query_vector = await embed_client.embed_single(question)
    embed_ms = (time.perf_counter() - embed_start) * 1000

    points, search_ms, search_limit, search_expansions = await _query_candidate_points(
        qdrant=qdrant,
        collection=collection,
        query_vector=query_vector,
        top_k=top_k,
    )
    if not points:
        return [], {
            "embed_ms": embed_ms,
            "search_ms": search_ms,
            "abstract_ms": 0.0,
            "retrieval_ms": embed_ms + search_ms,
            "docs_retrieved": 0,
            "docs_loaded": 0,
            "docs_from_payload": 0,
            "docs_from_pg": 0,
            "docs_from_title_only": 0,
            "docs_with_real_abstracts": 0,
            "search_limit": search_limit,
            "search_expansions": search_expansions,
            "top_k": top_k,
        }

    # Build docs from Qdrant payload first.
    # Keep rank/score from vector retrieval so PG fallback docs stay in the same
    # relevance ordering when possible.
    abstract_start = time.perf_counter()
    docs: list[DenseEvidenceDoc] = []
    title_only_docs: list[DenseEvidenceDoc] = []
    need_pg_pmids: list[str] = []  # PMIDs missing abstract in payload
    payload_pmids: set[str] = set()
    pg_pmids: set[str] = set()
    pmid_rank: dict[str, int] = {}
    pmid_score: dict[str, float] = {}

    for rank, p in enumerate(points, start=1):
        payload = p.payload or {}
        pmid = str(payload.get("pmid", p.id))

        # Deduplicate by PMID while preserving first (best) rank.
        if pmid in pmid_rank:
            continue

        pmid_rank[pmid] = rank
        pmid_score[pmid] = float(p.score) if p.score is not None else 0.0

        title = payload.get("title") or ""
        abstract_val = payload.get("abstract")
        abstract = abstract_val if isinstance(abstract_val, str) else ""

        if abstract.strip():
            payload_pmids.add(pmid)
            docs.append(DenseEvidenceDoc(
                pmid=pmid, title=title, abstract=abstract,
                score=pmid_score[pmid], rank=rank,
            ))
        else:
            need_pg_pmids.append(pmid)
            title_only_docs.append(DenseEvidenceDoc(
                pmid=pmid,
                title=title,
                abstract=_title_only_fallback(title, pmid),
                score=pmid_score[pmid],
                rank=rank,
            ))

    # Fallback: fetch missing abstracts from PostgreSQL only if payload docs
    # are still insufficient.
    if len(docs) < top_k and need_pg_pmids and pool:
        pg_rows = await _fetch_pg_abstracts(pool, need_pg_pmids)
        for pmid, title, abstract, _score in pg_rows:
            if not abstract:
                continue
            if pmid in payload_pmids:
                continue
            pg_pmids.add(pmid)
            docs.append(DenseEvidenceDoc(
                pmid=pmid, title=title, abstract=abstract,
                score=pmid_score.get(pmid, 0.0),
                rank=pmid_rank.get(pmid, 10**9),
            ))

    # Sort real-abstract docs first; if still insufficient, fall back to
    # title-only records rather than returning zero evidence.
    docs.sort(key=lambda d: (-d.score, d.rank))
    dedup_docs: list[DenseEvidenceDoc] = []
    seen_pmids: set[str] = set()
    for doc in docs:
        if doc.pmid in seen_pmids:
            continue
        seen_pmids.add(doc.pmid)
        dedup_docs.append(doc)
        if len(dedup_docs) >= top_k:
            break

    if len(dedup_docs) < top_k:
        title_only_docs.sort(key=lambda d: (-d.score, d.rank))
        for doc in title_only_docs:
            if doc.pmid in seen_pmids:
                continue
            seen_pmids.add(doc.pmid)
            dedup_docs.append(doc)
            if len(dedup_docs) >= top_k:
                break
    docs = dedup_docs

    for i, d in enumerate(docs, 1):
        d.rank = i

    abstract_ms = (time.perf_counter() - abstract_start) * 1000
    docs_from_payload = sum(1 for d in docs if d.pmid in payload_pmids)
    docs_from_pg = sum(1 for d in docs if d.pmid in pg_pmids)
    docs_from_title_only = len(docs) - docs_from_payload - docs_from_pg

    return docs, {
        "embed_ms": embed_ms,
        "search_ms": search_ms,
        "abstract_ms": abstract_ms,
        "retrieval_ms": embed_ms + search_ms + abstract_ms,
        "docs_retrieved": len(points),
        "docs_loaded": len(docs),
        "docs_from_payload": docs_from_payload,
        "docs_from_pg": docs_from_pg,
        "docs_from_title_only": docs_from_title_only,
        "docs_with_real_abstracts": docs_from_payload + docs_from_pg,
        "search_limit": search_limit,
        "search_expansions": search_expansions,
        "missing_abstract_candidates": len(need_pg_pmids),
        "top_k": top_k,
    }


def build_dense_context(
    docs: list[DenseEvidenceDoc],
    *,
    max_context_tokens: int | None = None,
) -> tuple[str, int, int]:
    """Build DenseRAG-style concatenated context.

    Returns:
        (context_text, docs_included, context_chars)
    """
    if max_context_tokens is None:
        # Abstracts fit 4000; PMC full-text chunks run ~7.4k tokens per 20 docs,
        # so XC_CONTEXT_TOKENS lets a full-text run widen the window for every method.
        max_context_tokens = _DEFAULT_CONTEXT_TOKENS
    context_parts = ["## Retrieved Documents\n"]
    total_chars = 0
    docs_included = 0

    for i, doc in enumerate(docs, start=1):
        doc_text = (
            f"### [{i}] PMID {doc.pmid} (score: {doc.score:.3f})\n"
            f"**{doc.title}**\n{doc.abstract}\n\n"
        )
        if total_chars + len(doc_text) > max_context_tokens * 4:
            break
        context_parts.append(doc_text)
        total_chars += len(doc_text)
        docs_included += 1

    return "".join(context_parts), docs_included, total_chars


def format_pubmed_evidence(
    docs: list[DenseEvidenceDoc],
    *,
    max_docs: int,
    max_abstract_chars: int = 1200,
) -> str:
    """Format shared dense evidence into a tool-friendly text block."""
    lines = ["PubMed Evidence:"]
    for i, doc in enumerate(docs[:max_docs], start=1):
        lines.append(f"[{i}] PMID {doc.pmid}")
        lines.append(f"Title: {doc.title}")
        lines.append(f"Abstract: {doc.abstract[:max_abstract_chars]}")
        lines.append("")
    if len(lines) == 1:
        lines.append("No PubMed evidence found.")
    return "\n".join(lines).strip()


async def generate_constrained_answer(
    *,
    question: str,
    question_type: str,
    context: str,
    options: dict[str, str] | None,
    temperature: float = 0.1,
) -> tuple[str, str]:
    """Shared final answer generation + extraction."""
    # Official MedXpertQA protocol (XC_MCQ_OFFICIAL=1), applied here so EVERY
    # method that answers through this function -- all baselines -- gets it.
    # It was previously implemented only inside V14CascadeClient, which meant a
    # "protocol" comparison actually compared our CoT arm against baselines
    # still on the 4-token single-letter prompt.
    # Reproduces TsinghuaC3I/MedXpertQA eval/: medical-assistant system role,
    # inline "Answer Choices: (A) ...", zero-shot CoT, then the
    # "Therefore, among A through <end>, the answer is" trigger continued on the
    # assistant turn, and word-boundary letter extraction over exactly the
    # offered letters (on RAW text -- upper-casing first would let a stray "a"
    # or "i" in prose match as an option).
    import os as _os
    if _os.environ.get("XC_MCQ_OFFICIAL") == "1" and question_type == "mcq" and options:
        import re as _re
        keys = sorted(options)
        end = keys[-1]
        choices = " ".join(f"({k}) {options[k]}" for k in keys)
        body = (f"Evidence:\n{context[:_ANSWER_CONTEXT_CHARS]}\n\n"
                f"Q: {question}\nAnswer Choices: {choices}\n"
                f"A: Let's think step by step.")
        msgs = [{"role": "system", "content": "You are a helpful medical assistant."},
                {"role": "user", "content": body}]
        try:
            cot = await llm_client.chat_raw(
                msgs, max_tokens=int(_os.environ.get("XC_MCQ_OFFICIAL_TOKENS", "512")),
                temperature=temperature)
            rationale = (cot.choices[0].message.content or "").strip()
        except Exception:
            rationale = ""
        trigger = f"Therefore, among A through {end}, the answer is"
        if _os.environ.get("XC_LLM_PORTABLE") == "1":
            # Cloud APIs cannot continue the assistant's own turn (no
            # continue_final_message), so the official trigger is delivered as a
            # short follow-up user turn. Documented deviation for frontier runs.
            msgs2 = msgs + [{"role": "assistant", "content": rationale or "(no rationale)"},
                            {"role": "user", "content": f"{trigger} ... ? Reply with the single option letter in parentheses and nothing else."}]
            # 256 tokens: costs nothing when the reply is "(B)", but survives a model
            # that restates the question before answering (a 64-token cap truncated
            # the local smoke test before any letter appeared).
            resp = await llm_client.chat_raw(msgs2, max_tokens=max(256, int(_os.environ.get("XC_ANSWER_MAX_TOKENS_FLOOR", "0"))), temperature=temperature)
        else:
            msgs2 = msgs + [{"role": "assistant", "content": f"{rationale}\n{trigger}"}]
            resp = await llm_client.chat_raw(
                msgs2, max_tokens=8, temperature=temperature,
                extra_body={"continue_final_message": True, "add_generation_prompt": False})
        text = resp.choices[0].message.content or ""
        for junk in ("I understand", f"A through {end}"):
            text = text.replace(junk, "")
        hits = _re.findall(r"\b(" + "|".join(keys) + r")\b", text)
        if _os.environ.get("XC_LLM_PORTABLE") == "1":
            # The follow-up turn is a complete reply, not a continuation, so the
            # answer is at the END: prefer the last "(X)" we asked for, else the
            # last bare letter. hits[0] would pick a letter mentioned mid-reasoning.
            paren = _re.findall(r"\((" + "|".join(keys) + r")\)", text)
            return (paren[-1] if paren else (hits[-1] if hits else "")), text
        return (hits[0] if hits else ""), text

    system_prompt, user_prompt = build_answer_messages(
        question_type=question_type,
        question=question,
        context=context,
        options=options,
    )
    # XC_ANSWER_MAX_TOKENS_FLOOR: cloud models whose hidden reasoning is billed inside
    # max_tokens (Gemini cannot disable thinking; the reasoning-on arm) need room above
    # the 4-32-token constrained caps or they return empty content. Default 0 = unchanged.
    answer_text = await llm_client.chat(
        prompt=user_prompt,
        system=system_prompt,
        max_tokens=max(get_max_tokens(question_type), int(_os.environ.get("XC_ANSWER_MAX_TOKENS_FLOOR", "0"))),
        temperature=temperature,
    )
    answer = extract_constrained_answer(answer_text, question_type, options)
    return answer, answer_text
