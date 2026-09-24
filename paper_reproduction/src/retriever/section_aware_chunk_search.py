"""Section-aware full-text retrieval for biomedical papers.

This module builds on top of flat chunk vector search and adds a lightweight
intra-paper traversal layer:
1. Retrieve seed chunks semantically
2. Inspect their section roles
3. If seeds come from low-signal sections (e.g. introduction), expand within
   the same paper toward higher-value sections such as results/discussion
4. Re-rank chunks with section-aware priors
"""

from __future__ import annotations

import re
from typing import Any, Iterable, Sequence

import asyncpg

try:
    from ..config import get_config
except ImportError:  # Support PYTHONPATH=src runtime
    from config import get_config
from .chunk_search import ChunkSearch


DEFAULT_PREFERRED_SECTION_TYPES = ("results", "discussion", "conclusion")
LOW_SIGNAL_SECTION_TYPES = {"introduction", "background", "methods"}


def _normalize_section_role(section_type: str | None, section_title: str | None) -> str:
    """Map raw section metadata to a small set of retrieval-relevant roles."""
    raw = f"{section_type or ''} {section_title or ''}".lower()

    if re.search(r"\babstract\b", raw):
        return "abstract"
    if re.search(r"\b(introduction|intro|background)\b", raw):
        return "introduction" if "background" not in raw else "background"
    if re.search(r"\b(materials?\s+and\s+methods?|methods?|methodology|patients?\s+and\s+methods?)\b", raw):
        return "methods"
    if re.search(r"\bresults?\b", raw):
        return "results"
    if re.search(r"\bdiscussion\b", raw):
        return "discussion"
    if re.search(r"\b(conclusion|conclusions|concluding remarks?)\b", raw):
        return "conclusion"
    return (section_type or "other").lower().strip() or "other"


def _normalize_preferred_types(preferred_section_types: Sequence[str] | None) -> tuple[str, ...]:
    if not preferred_section_types:
        return DEFAULT_PREFERRED_SECTION_TYPES
    return tuple(_normalize_section_role(item, item) for item in preferred_section_types)


def _section_weight(role: str, preferred_roles: Sequence[str]) -> float:
    base = {
        "results": 1.18,
        "conclusion": 1.14,
        "discussion": 1.10,
        "abstract": 0.95,
        "methods": 0.78,
        "background": 0.68,
        "introduction": 0.65,
        "other": 1.00,
    }.get(role, 1.00)
    if role in preferred_roles:
        base += 0.05
    return base


class SectionAwareChunkSearch:
    """Layer section-aware traversal on top of the existing ChunkSearch."""

    def __init__(
        self,
        base_chunk_search: ChunkSearch,
        db_pool: asyncpg.Pool | None = None,
    ):
        self._base = base_chunk_search
        self._db_pool = db_pool
        self._config = get_config()

    async def _ensure_db_pool(self) -> asyncpg.Pool:
        if self._db_pool is None:
            self._db_pool = await asyncpg.create_pool(
                self._config.postgres.papergraph_url,
                min_size=2,
                max_size=10,
            )
        return self._db_pool

    async def _fetch_chunk_metadata(self, chunk_ids: Sequence[str]) -> dict[str, dict[str, Any]]:
        if not chunk_ids:
            return {}

        pool = await self._ensure_db_pool()
        async with pool.acquire() as conn:
            placeholders = ", ".join(f"${i+1}" for i in range(len(chunk_ids)))
            rows = await conn.fetch(f"""
                SELECT
                    c.id::text AS chunk_id,
                    c.text_content,
                    c.section_id::text,
                    c.sequence_order,
                    c.token_count,
                    s.id::text AS section_id_text,
                    s.title AS section_title,
                    s.section_type,
                    s.hierarchy_level,
                    s.sequence_order AS section_sequence_order,
                    p.id::text AS paper_id,
                    p.pmid::text AS pmid,
                    p.pmcid
                FROM chunks c
                JOIN sections s ON c.section_id = s.id
                JOIN papers p ON s.paper_id = p.id
                WHERE c.id IN ({placeholders})
            """, *chunk_ids)

        result = {}
        for row in rows:
            role = _normalize_section_role(row["section_type"], row["section_title"])
            result[row["chunk_id"]] = {
                "chunk_id": row["chunk_id"],
                "paper_id": row["paper_id"],
                "pmid": row["pmid"] or "",
                "pmcid": row["pmcid"] or "",
                "text": row["text_content"] or "",
                "section_id": row["section_id_text"] or "",
                "section_title": row["section_title"] or "",
                "section_type": row["section_type"] or "",
                "section_role": role,
                "hierarchy_level": row["hierarchy_level"],
                "sequence_order": row["sequence_order"] or 0,
                "section_sequence_order": row["section_sequence_order"] or 0,
                "token_count": row["token_count"] or 0,
            }
        return result

    async def _resolve_paper(self, paper_identifier: str) -> dict[str, str] | None:
        pool = await self._ensure_db_pool()
        async with pool.acquire() as conn:
            paper = await conn.fetchrow("""
                SELECT id::text AS paper_id, pmid::text AS pmid, pmcid
                FROM papers
                WHERE pmid::text = $1
            """, str(paper_identifier))

            if paper:
                return dict(paper)

            paper = await conn.fetchrow("""
                SELECT id::text AS paper_id, pmid::text AS pmid, pmcid
                FROM papers
                WHERE id::text = $1
            """, str(paper_identifier))

            return dict(paper) if paper else None

    async def _get_candidate_section_ids(
        self,
        paper_id: str,
        seed_section_ids: Iterable[str],
        preferred_roles: Sequence[str],
    ) -> list[str]:
        pool = await self._ensure_db_pool()
        async with pool.acquire() as conn:
            rows = await conn.fetch("""
                SELECT id::text AS section_id, title, section_type, hierarchy_level, sequence_order
                FROM sections
                WHERE paper_id::text = $1
                ORDER BY hierarchy_level, sequence_order
            """, paper_id)

        seed_set = {str(section_id) for section_id in seed_section_ids if section_id}
        preferred = []
        neutral = []
        for row in rows:
            role = _normalize_section_role(row["section_type"], row["title"])
            section_id = row["section_id"]
            if section_id in seed_set:
                continue
            if role in preferred_roles:
                preferred.append(section_id)
            elif role not in LOW_SIGNAL_SECTION_TYPES:
                neutral.append(section_id)

        return preferred or neutral or list(seed_set)

    async def _filter_ranked_chunks(
        self,
        query: str,
        pmid: str,
        allowed_section_ids: set[str],
        preferred_roles: Sequence[str],
        limit: int,
        retrieval_stage: str,
    ) -> list[dict[str, Any]]:
        if not allowed_section_ids:
            return []

        raw_chunks, _ = await self._base.search(
            query=query,
            pmid_filter={str(pmid)},
            limit=max(limit * 8, 60),
        )
        metadata = await self._fetch_chunk_metadata([chunk.chunk_id for chunk in raw_chunks])

        ranked = []
        for chunk in raw_chunks:
            meta = metadata.get(chunk.chunk_id)
            if not meta or meta["section_id"] not in allowed_section_ids:
                continue
            adjusted_score = chunk.score * _section_weight(meta["section_role"], preferred_roles)
            ranked.append({
                **meta,
                "score": chunk.score,
                "rank": chunk.rank,
                "adjusted_score": adjusted_score,
                "retrieval_stage": retrieval_stage,
            })

        ranked.sort(key=lambda item: item["adjusted_score"], reverse=True)
        return ranked[:limit]

    async def search(
        self,
        query: str,
        paper_ids: Sequence[str] | None = None,
        preferred_section_types: Sequence[str] | None = None,
        limit: int = 20,
        seed_limit: int | None = None,
    ) -> list[dict[str, Any]]:
        """Search chunks, then expand low-signal hits toward higher-value sections."""
        preferred_roles = _normalize_preferred_types(preferred_section_types)
        pmid_filter = {str(paper_id) for paper_id in paper_ids} if paper_ids else None
        raw_chunks, _ = await self._base.search(
            query=query,
            pmid_filter=pmid_filter,
            limit=seed_limit or max(limit * 4, 40),
        )
        if not raw_chunks:
            return []

        metadata = await self._fetch_chunk_metadata([chunk.chunk_id for chunk in raw_chunks])
        seed_ranked = []
        paper_seed_sections: dict[str, set[str]] = {}
        ordered_papers: list[tuple[str, str]] = []

        for chunk in raw_chunks:
            meta = metadata.get(chunk.chunk_id)
            if not meta:
                continue
            adjusted_score = chunk.score * _section_weight(meta["section_role"], preferred_roles)
            seed_ranked.append({
                **meta,
                "score": chunk.score,
                "rank": chunk.rank,
                "adjusted_score": adjusted_score,
                "retrieval_stage": "seed",
            })
            paper_seed_sections.setdefault(meta["paper_id"], set()).add(meta["section_id"])
            if meta["paper_id"] not in {paper_id for paper_id, _ in ordered_papers}:
                ordered_papers.append((meta["paper_id"], meta["pmid"]))

        seed_ranked.sort(key=lambda item: item["adjusted_score"], reverse=True)

        expansions: list[dict[str, Any]] = []
        for paper_id, pmid in ordered_papers[: max(3, min(limit, 6))]:
            paper_seeds = [item for item in seed_ranked if item["paper_id"] == paper_id]
            if not paper_seeds:
                continue
            best_role = paper_seeds[0]["section_role"]
            if best_role not in LOW_SIGNAL_SECTION_TYPES and best_role in preferred_roles:
                continue

            candidate_sections = await self._get_candidate_section_ids(
                paper_id=paper_id,
                seed_section_ids=paper_seed_sections.get(paper_id, set()),
                preferred_roles=preferred_roles,
            )
            expanded = await self._filter_ranked_chunks(
                query=query,
                pmid=pmid,
                allowed_section_ids=set(candidate_sections),
                preferred_roles=preferred_roles,
                limit=max(4, limit // 2),
                retrieval_stage="expanded",
            )
            expansions.extend(expanded)

        merged: dict[str, dict[str, Any]] = {}
        for item in seed_ranked + expansions:
            current = merged.get(item["chunk_id"])
            if current is None or item["adjusted_score"] > current["adjusted_score"]:
                merged[item["chunk_id"]] = item

        ranked = sorted(merged.values(), key=lambda item: item["adjusted_score"], reverse=True)
        return ranked[:limit]

    async def expand_from_seed(
        self,
        paper_id: str,
        seed_section_ids: Sequence[str],
        query: str,
        top_k: int = 10,
        preferred_section_types: Sequence[str] | None = None,
    ) -> list[dict[str, Any]]:
        """Expand evidence inside one paper from seed sections toward better sections."""
        paper = await self._resolve_paper(paper_id)
        if not paper or not paper.get("pmid"):
            return []

        preferred_roles = _normalize_preferred_types(preferred_section_types)
        candidate_sections = await self._get_candidate_section_ids(
            paper_id=paper["paper_id"],
            seed_section_ids=seed_section_ids,
            preferred_roles=preferred_roles,
        )
        if not candidate_sections:
            candidate_sections = [str(section_id) for section_id in seed_section_ids if section_id]

        return await self._filter_ranked_chunks(
            query=query,
            pmid=paper["pmid"],
            allowed_section_ids=set(candidate_sections),
            preferred_roles=preferred_roles,
            limit=top_k,
            retrieval_stage="expanded",
        )

    async def close(self):
        if self._db_pool:
            await self._db_pool.close()
            self._db_pool = None
