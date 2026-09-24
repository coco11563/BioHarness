#!/usr/bin/env python3
"""Unified benchmark runner for all methods.

Single entry point for running ALL benchmark methods on 8 datasets.

Methods:
  Baselines:
    no-context       Pure LLM, no retrieval
    dense            Dense RAG (retrieve → concat → LLM)
    dual-dense       Dual retrieval (pos+neg evidence)
    golden-context   Dense RAG with dataset-provided golden context

  KG-RAG:
    rt-lightrag      Retrieve-then-LightRAG (hybrid mode)
    rt-pathrag       Retrieve-then-PathRAG
    rt-graphrag      Retrieve-then-GraphRAG (local mode)
    mesh-kg          MeSH hierarchy as knowledge graph
    biorag           BioRAG (planner→router→tools pipeline)

  Our System:
    v14              V14 RLM Agent (REPL reasoning + constrained re-judgment)
    v14.1            V14.1 RLM Agent (section-aware full-text policy)
    v14.2            V14.2 RLM Agent (section-aware retrieval + budgeted full-text assembly)
    v14.3            V14.3 RLM Agent (v14 + selective full-text escalation)
    v14-dh           V14 + Dual Hypothesis NLI

Usage:
    PYTHONPATH=src python scripts/run_unified_benchmark.py --method dense --datasets bioasq --limit 100
    PYTHONPATH=src python scripts/run_unified_benchmark.py --method rt-lightrag --datasets bioasq --limit 100
    PYTHONPATH=src python scripts/run_unified_benchmark.py --method v14 --datasets bioasq --limit 100
    PYTHONPATH=src python scripts/run_unified_benchmark.py --method v14.1 --datasets pubmedqa_pqal_test --limit 20
    PYTHONPATH=src python scripts/run_unified_benchmark.py --method v14.2 --datasets pubmedqa_pqal_test --limit 20
    PYTHONPATH=src python scripts/run_unified_benchmark.py --method v14.3 --datasets bioasq --limit 20
"""

import asyncio
import logging
import os
import re
import sys
import time
from pathlib import Path

_atlas_logger = logging.getLogger("v14_atlas")
LOGGER = logging.getLogger(__name__)

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent.parent / "benchmark" / "code" / "src"))

from benchmark import BenchmarkLoader, MetricsEvaluator, BenchmarkRunner, BenchmarkLogger, RunConfig, print_report
from benchmark.client_protocol import ModelClient, ModelResponse
from benchmark.shared_pipeline import (
    retrieve_dense_evidence,
    build_dense_context,
    format_pubmed_evidence,
    DenseEvidenceDoc,
    generate_constrained_answer,
)
from benchmark.retrieve_then_kg import create_retrieve_then_kg_client
from benchmark.mesh_kg_rag import MeSHKGRAG
from benchmark.biorag import BioRAGClient
from src.rlm.pipeline import BiomedicalRLMPipeline, PipelineConfig
from src.rlm import trace as _trace  # evidence-cut tracing; no-op unless XC_TRACE_DIR is set
from src.retriever.chunk_search import ChunkSearch
from src.retriever.fulltext_escalation_gate import FulltextEscalationGate
from src.retriever.fulltext_evidence_assembler import FulltextEvidenceAssembler
from src.retriever.section_aware_chunk_search import SectionAwareChunkSearch
from src.rlm.tools import shutdown_tools
from src.utils.answer_extraction import extract_answer

import asyncpg
from qdrant_client import AsyncQdrantClient
from config import get_config


class UnifiedEvidenceClient(ModelClient):
    """Base client with shared dense retrieval. Subclass for each method."""

    def __init__(self, top_k: int = 20, collection: str = "paper-full"):
        self._top_k = top_k
        self._collection = collection
        self._qdrant = None
        self._pool = None
        self._papergraph_pool = None
        cfg = get_config()
        self._qdrant_url = cfg.qdrant.url
        self._pg_dsn = cfg.postgres.pubmed_url
        self._papergraph_dsn = cfg.postgres.papergraph_url

    async def __aenter__(self):
        self._qdrant = AsyncQdrantClient(self._qdrant_url, timeout=300)
        self._pool = await asyncpg.create_pool(self._pg_dsn, min_size=2, max_size=10)
        return self

    async def __aexit__(self, *args):
        if self._pool:
            await self._pool.close()
        if self._papergraph_pool:
            await self._papergraph_pool.close()
        if self._qdrant:
            await self._qdrant.close()

    async def _ensure_papergraph_pool(self):
        if self._papergraph_pool is None:
            self._papergraph_pool = await asyncpg.create_pool(
                self._papergraph_dsn,
                min_size=2,
                max_size=10,
            )
        return self._papergraph_pool

    async def _retrieve_shared_evidence(self, question: str):
        """Shared retrieval — same for ALL methods."""
        return await retrieve_dense_evidence(
            question,
            qdrant=self._qdrant,
            pool=self._pool,
            top_k=self._top_k,
            collection=self._collection,
        )


class DenseRAGClient(UnifiedEvidenceClient):
    """Dense RAG: retrieve → concat → LLM answer."""

    async def generate(self, question, question_type, context=None, options=None):
        start = time.perf_counter()
        docs, meta = await self._retrieve_shared_evidence(question)
        if not docs:
            return ModelResponse(answer="", response_text="No evidence found.",
                                latency_ms=(time.perf_counter() - start) * 1000)

        full_context, _, _ = build_dense_context(docs)
        answer, answer_text = await generate_constrained_answer(
            question=question, question_type=question_type,
            context=full_context, options=options,
        )
        return ModelResponse(
            answer=answer, response_text=answer_text,
            latency_ms=(time.perf_counter() - start) * 1000,
            metadata={"method": "dense_rag", "docs": len(docs)},
        )


class V14AgentClient(UnifiedEvidenceClient):
    """V14-family Agent: retrieve shared evidence → agent analyzes it in REPL."""

    def __init__(self, top_k=20, dual_hypothesis=False, version="v14",
                 enable_tools=True, enable_rejudge=True):
        super().__init__(top_k=top_k)
        self._dual_hypothesis = dual_hypothesis
        self._version = version
        self._pipeline_version = "v14" if version == "v14.3" else version
        self._enable_tools = enable_tools
        self._enable_rejudge = enable_rejudge
        self._pipeline = None
        self._section_search = None
        self._assembler = None
        self._gate = None

    async def __aenter__(self):
        await super().__aenter__()
        if self._version in ("v14.1", "v14.2", "v14.3"):
            papergraph_pool = await self._ensure_papergraph_pool()
            self._section_search = SectionAwareChunkSearch(
                ChunkSearch(self._qdrant, db_pool=papergraph_pool),
                db_pool=papergraph_pool,
            )
        if self._version == "v14.2":
            self._assembler = FulltextEvidenceAssembler(evidence_token_budget=6000)
            self._gate = FulltextEscalationGate()
        elif self._version == "v14.3":
            self._assembler = FulltextEvidenceAssembler(evidence_token_budget=2800)
            self._gate = FulltextEscalationGate()
        config = PipelineConfig(
            version=self._pipeline_version,
            max_iterations=8,
            use_gold_context=True,  # Agent receives pre-retrieved evidence
            yesno_dual_hypothesis=self._dual_hypothesis,
            enable_kg_tools=self._enable_tools,
        )
        self._pipeline = BiomedicalRLMPipeline(config=config)
        return self

    async def __aexit__(self, *args):
        self._pipeline = None
        self._section_search = None
        self._assembler = None
        self._gate = None
        shutdown_tools()
        await super().__aexit__(*args)

    @staticmethod
    def _merge_section_chunks(*groups):
        merged = {}
        for group in groups:
            for chunk in group:
                chunk_id = chunk.get("chunk_id")
                if not chunk_id:
                    continue
                prev = merged.get(chunk_id)
                score = chunk.get("adjusted_score", chunk.get("score", 0.0))
                prev_score = prev.get("adjusted_score", prev.get("score", 0.0)) if prev else None
                if prev is None or score > prev_score:
                    merged[chunk_id] = chunk
        return sorted(
            merged.values(),
            key=lambda c: c.get("adjusted_score", c.get("score", 0.0)),
            reverse=True,
        )

    @staticmethod
    def _format_section_evidence(chunks, max_docs=20):
        blocks = []
        for idx, chunk in enumerate(chunks[:max_docs], 1):
            pmid = chunk.get("pmid") or "unknown"
            role = chunk.get("section_role") or chunk.get("section_type") or "other"
            title = chunk.get("section_title") or ""
            stage = chunk.get("retrieval_stage") or "seed"
            header = f"[Chunk {idx} | PMID:{pmid} | role:{role}"
            if title:
                header += f" | title:{title}"
            header += f" | stage:{stage}]"
            text = (chunk.get("text") or "").strip()
            if text:
                blocks.append(f"{header}\n{text}")
        return "\n\n".join(blocks)

    @staticmethod
    def _estimate_tokens(text: str) -> int:
        return max(1, (len(text or "") + 3) // 4)

    @classmethod
    def _build_budgeted_abstract_evidence(cls, docs, *, max_docs_cap: int, token_budget: int):
        if not docs:
            return "", 0

        upper = min(len(docs), max_docs_cap)
        for doc_count in range(upper, 0, -1):
            text = format_pubmed_evidence(docs[:doc_count], max_docs=doc_count)
            if cls._estimate_tokens(text) <= token_budget:
                return text, doc_count

        text = format_pubmed_evidence(docs[:1], max_docs=1)
        return text[: token_budget * 4], 1

    @staticmethod
    def _combine_v143_evidence(abstract_evidence_text: str, fulltext_evidence_text: str) -> str:
        blocks = []
        if abstract_evidence_text:
            blocks.append(f"## Abstract Evidence\n{abstract_evidence_text}")
        if fulltext_evidence_text:
            blocks.append(fulltext_evidence_text)
        return "\n\n".join(blocks)

    @staticmethod
    def _route_label(escalate: bool) -> str:
        return "abstract_plus_section_fulltext" if escalate else "abstract_only"

    def _preview_v143_route(
        self,
        *,
        question: str,
        question_type: str,
        fulltext_summary: dict,
        distinct_pmids_in_section_chunks: int,
        section_aware_roles: list[str],
        assembly_mode: str,
        assembled_units: int,
        assembled_support_units: int,
        assembled_refute_units: int,
        assembled_pmids: int,
    ) -> dict[str, object]:
        if not self._gate:
            return {
                "v143_route_preview_gate": False,
                "v143_route_preview_attempted": False,
                "v143_route_preview": "abstract_only",
                "v143_route_preview_reason": "gate_unavailable",
                "v143_route_preview_pre_reason": "gate_unavailable",
                "v143_route_preview_post_reason": "",
                "v143_route_preview_mode": "abstract_only",
            }

        precheck = self._gate.precheck(
            question=question,
            question_type=question_type,
            candidate_have_pmc_in_articles=fulltext_summary.get("candidate_have_pmc_in_articles", 0),
            candidate_in_papergraph=fulltext_summary.get("candidate_in_papergraph", 0),
        )
        if not precheck.escalate:
            return {
                "v143_route_preview_gate": False,
                "v143_route_preview_attempted": False,
                "v143_route_preview": self._route_label(False),
                "v143_route_preview_reason": precheck.reason,
                "v143_route_preview_pre_reason": precheck.reason,
                "v143_route_preview_post_reason": "",
                "v143_route_preview_mode": precheck.mode,
            }

        postcheck = self._gate.postcheck(
            question_type=question_type,
            distinct_pmids_in_section_chunks=distinct_pmids_in_section_chunks,
            section_aware_roles=section_aware_roles,
            assembly_mode=assembly_mode,
            assembled_units=assembled_units,
            assembled_support_units=assembled_support_units,
            assembled_refute_units=assembled_refute_units,
            assembled_pmids=assembled_pmids,
        )
        return {
            "v143_route_preview_gate": postcheck.escalate,
            "v143_route_preview_attempted": True,
            "v143_route_preview": self._route_label(postcheck.escalate),
            "v143_route_preview_reason": postcheck.reason,
            "v143_route_preview_pre_reason": precheck.reason,
            "v143_route_preview_post_reason": postcheck.reason,
            "v143_route_preview_mode": postcheck.mode,
        }

    async def _summarize_candidate_fulltext(self, candidate_pmids):
        if not candidate_pmids:
            return (
                {
                    "candidate_pmids": 0,
                    "candidate_have_pmc_in_articles": 0,
                    "candidate_in_papergraph": 0,
                },
                set(),
            )

        pmids = [str(pmid) for pmid in candidate_pmids if pmid]
        article_fulltext = set()
        papergraph_fulltext = set()

        if self._pool is not None:
            async with self._pool.acquire() as conn:
                rows = await conn.fetch(
                    """
                    SELECT pmid::text AS pmid, pmc
                    FROM articles
                    WHERE pmid::text = ANY($1::text[])
                    """,
                    pmids,
                )
            article_fulltext = {row["pmid"] for row in rows if row["pmc"]}

        papergraph_pool = await self._ensure_papergraph_pool()
        async with papergraph_pool.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT pmid::text AS pmid
                FROM papers
                WHERE pmid::text = ANY($1::text[])
                """,
                pmids,
            )
        papergraph_fulltext = {row["pmid"] for row in rows}

        return (
            {
                "candidate_pmids": len(pmids),
                "candidate_have_pmc_in_articles": len(article_fulltext),
                "candidate_in_papergraph": len(papergraph_fulltext),
            },
            papergraph_fulltext,
        )

    async def _build_v141_evidence(self, question: str, question_type: str):
        pos_docs, _ = await self._retrieve_shared_evidence(question)
        candidate_pmids = [doc.pmid for doc in pos_docs if getattr(doc, "pmid", None)]

        neg_docs = []
        if question_type == "yesno":
            neg_terms = "no effect OR not effective OR no association OR failed OR no benefit"
            neg_query = f"{question} {neg_terms}"
            neg_docs, _ = await retrieve_dense_evidence(
                neg_query,
                qdrant=self._qdrant,
                pool=self._pool,
                top_k=self._top_k,
                collection=self._collection,
            )
            for doc in neg_docs:
                if getattr(doc, "pmid", None) and doc.pmid not in candidate_pmids:
                    candidate_pmids.append(doc.pmid)

        if question_type == "yesno":
            seen = {d.pmid for d in pos_docs}
            abstract_docs = list(pos_docs)
            for doc in neg_docs:
                if doc.pmid not in seen:
                    seen.add(doc.pmid)
                    abstract_docs.append(doc)
            abstract_docs.sort(key=lambda d: -d.score)
        else:
            abstract_docs = pos_docs

        abstract_evidence_text = format_pubmed_evidence(abstract_docs[:self._top_k], max_docs=20)

        if not self._section_search:
            return abstract_evidence_text, {
                "section_aware_retrieval": False,
                "coverage_miss": False,
                "section_aware_chunks": 0,
                "distinct_pmids_in_section_chunks": 0,
                "section_aware_roles": [],
                "section_aware_stages": [],
                "candidate_pmids": len(candidate_pmids),
                "candidate_have_pmc_in_articles": 0,
                "candidate_in_papergraph": 0,
            }

        fulltext_summary, fulltext_pmids = await self._summarize_candidate_fulltext(candidate_pmids)
        if not candidate_pmids or not fulltext_pmids:
            return abstract_evidence_text, {
                "section_aware_retrieval": False,
                "coverage_miss": True,
                "section_aware_chunks": 0,
                "distinct_pmids_in_section_chunks": 0,
                "section_aware_roles": [],
                "section_aware_stages": [],
                **fulltext_summary,
            }

        search_kwargs = {
            "limit": self._top_k,
            "seed_limit": max(self._top_k * 4, 40),
        }
        pos_chunks = await self._section_search.search(
            query=question,
            paper_ids=sorted(fulltext_pmids),
            **search_kwargs,
        )

        neg_chunks = []
        if question_type == "yesno":
            neg_terms = "no effect OR not effective OR no association OR failed OR no benefit"
            neg_query = f"{question} {neg_terms}"
            neg_chunks = await self._section_search.search(
                query=neg_query,
                paper_ids=sorted(fulltext_pmids),
                **search_kwargs,
            )

        merged_chunks = self._merge_section_chunks(pos_chunks, neg_chunks)[:self._top_k]
        if not merged_chunks:
            return abstract_evidence_text, {
                "section_aware_retrieval": False,
                "coverage_miss": False,
                "section_aware_chunks": 0,
                "distinct_pmids_in_section_chunks": 0,
                "section_aware_roles": [],
                "section_aware_stages": [],
                **fulltext_summary,
            }

        evidence_text = self._format_section_evidence(merged_chunks, max_docs=self._top_k)
        metadata = {
            "section_aware_retrieval": True,
            "coverage_miss": False,
            "section_aware_chunks": len(merged_chunks),
            "distinct_pmids_in_section_chunks": len({c.get("pmid") for c in merged_chunks if c.get("pmid")}),
            "section_aware_roles": sorted({c.get("section_role", "") for c in merged_chunks if c.get("section_role")}),
            "section_aware_stages": sorted({c.get("retrieval_stage", "") for c in merged_chunks if c.get("retrieval_stage")}),
            **fulltext_summary,
        }
        return evidence_text, metadata

    async def _build_v142_evidence(self, question: str, question_type: str):
        pos_docs, _ = await self._retrieve_shared_evidence(question)
        candidate_pmids = [doc.pmid for doc in pos_docs if getattr(doc, "pmid", None)]

        neg_docs = []
        if question_type == "yesno":
            neg_terms = "no effect OR not effective OR no association OR failed OR no benefit"
            neg_query = f"{question} {neg_terms}"
            neg_docs, _ = await retrieve_dense_evidence(
                neg_query,
                qdrant=self._qdrant,
                pool=self._pool,
                top_k=self._top_k,
                collection=self._collection,
            )
            for doc in neg_docs:
                if getattr(doc, "pmid", None) and doc.pmid not in candidate_pmids:
                    candidate_pmids.append(doc.pmid)

        if question_type == "yesno":
            seen = {d.pmid for d in pos_docs}
            abstract_docs = list(pos_docs)
            for doc in neg_docs:
                if doc.pmid not in seen:
                    seen.add(doc.pmid)
                    abstract_docs.append(doc)
            abstract_docs.sort(key=lambda d: -d.score)
        else:
            abstract_docs = pos_docs

        abstract_evidence_text = format_pubmed_evidence(abstract_docs[:self._top_k], max_docs=20)

        if not self._section_search or not self._assembler:
            preview_meta = self._preview_v143_route(
                question=question,
                question_type=question_type,
                fulltext_summary={
                    "candidate_pmids": len(candidate_pmids),
                    "candidate_have_pmc_in_articles": 0,
                    "candidate_in_papergraph": 0,
                },
                distinct_pmids_in_section_chunks=0,
                section_aware_roles=[],
                assembly_mode="unavailable",
                assembled_units=0,
                assembled_support_units=0,
                assembled_refute_units=0,
                assembled_pmids=0,
            )
            return abstract_evidence_text, {
                "section_aware_retrieval": False,
                "coverage_miss": False,
                "section_aware_chunks": 0,
                "distinct_pmids_in_section_chunks": 0,
                "section_aware_roles": [],
                "section_aware_stages": [],
                "candidate_pmids": len(candidate_pmids),
                "candidate_have_pmc_in_articles": 0,
                "candidate_in_papergraph": 0,
                "assembly_mode": "unavailable",
                "assembly_budget_tokens": 0,
                "assembly_estimated_tokens": 0,
                "assembled_units": 0,
                "assembled_support_units": 0,
                "assembled_refute_units": 0,
                "assembled_pmids": 0,
                **preview_meta,
            }

        fulltext_summary, fulltext_pmids = await self._summarize_candidate_fulltext(candidate_pmids)
        if not candidate_pmids or not fulltext_pmids:
            preview_meta = self._preview_v143_route(
                question=question,
                question_type=question_type,
                fulltext_summary=fulltext_summary,
                distinct_pmids_in_section_chunks=0,
                section_aware_roles=[],
                assembly_mode="coverage_miss",
                assembled_units=0,
                assembled_support_units=0,
                assembled_refute_units=0,
                assembled_pmids=0,
            )
            return abstract_evidence_text, {
                "section_aware_retrieval": False,
                "coverage_miss": True,
                "section_aware_chunks": 0,
                "distinct_pmids_in_section_chunks": 0,
                "section_aware_roles": [],
                "section_aware_stages": [],
                **fulltext_summary,
                "assembly_mode": "coverage_miss",
                "assembly_budget_tokens": self._assembler.evidence_token_budget,
                "assembly_estimated_tokens": 0,
                "assembled_units": 0,
                "assembled_support_units": 0,
                "assembled_refute_units": 0,
                "assembled_pmids": 0,
                **preview_meta,
            }

        search_kwargs = {
            "limit": self._top_k,
            "seed_limit": max(self._top_k * 4, 40),
        }
        pos_chunks = await self._section_search.search(
            query=question,
            paper_ids=sorted(fulltext_pmids),
            **search_kwargs,
        )

        neg_chunks = []
        if question_type == "yesno":
            neg_terms = "no effect OR not effective OR no association OR failed OR no benefit"
            neg_query = f"{question} {neg_terms}"
            neg_chunks = await self._section_search.search(
                query=neg_query,
                paper_ids=sorted(fulltext_pmids),
                **search_kwargs,
            )

        raw_merged = self._merge_section_chunks(pos_chunks, neg_chunks)[: self._top_k]
        if not raw_merged:
            preview_meta = self._preview_v143_route(
                question=question,
                question_type=question_type,
                fulltext_summary=fulltext_summary,
                distinct_pmids_in_section_chunks=0,
                section_aware_roles=[],
                assembly_mode="retrieval_miss",
                assembled_units=0,
                assembled_support_units=0,
                assembled_refute_units=0,
                assembled_pmids=0,
            )
            return abstract_evidence_text, {
                "section_aware_retrieval": False,
                "coverage_miss": False,
                "section_aware_chunks": 0,
                "distinct_pmids_in_section_chunks": 0,
                "section_aware_roles": [],
                "section_aware_stages": [],
                **fulltext_summary,
                "assembly_mode": "retrieval_miss",
                "assembly_budget_tokens": self._assembler.evidence_token_budget,
                "assembly_estimated_tokens": 0,
                "assembled_units": 0,
                "assembled_support_units": 0,
                "assembled_refute_units": 0,
                "assembled_pmids": 0,
                **preview_meta,
            }

        evidence_text, assembly_meta = self._assembler.assemble(
            question_type=question_type,
            pos_chunks=pos_chunks[: self._top_k],
            neg_chunks=neg_chunks[: self._top_k],
            abstract_evidence_text=abstract_evidence_text,
        )

        preview_meta = self._preview_v143_route(
            question=question,
            question_type=question_type,
            fulltext_summary=fulltext_summary,
            distinct_pmids_in_section_chunks=len({c.get("pmid") for c in raw_merged if c.get("pmid")}),
            section_aware_roles=sorted({c.get("section_role", "") for c in raw_merged if c.get("section_role")}),
            assembly_mode=assembly_meta.get("assembly_mode", "retrieval_miss"),
            assembled_units=assembly_meta.get("assembled_units", 0),
            assembled_support_units=assembly_meta.get("assembled_support_units", 0),
            assembled_refute_units=assembly_meta.get("assembled_refute_units", 0),
            assembled_pmids=assembly_meta.get("assembled_pmids", 0),
        )

        metadata = {
            "section_aware_retrieval": assembly_meta.get("assembly_mode") != "abstract_fallback",
            "coverage_miss": False,
            "section_aware_chunks": len(raw_merged),
            "distinct_pmids_in_section_chunks": len({c.get("pmid") for c in raw_merged if c.get("pmid")}),
            "section_aware_roles": sorted({c.get("section_role", "") for c in raw_merged if c.get("section_role")}),
            "section_aware_stages": sorted({c.get("retrieval_stage", "") for c in raw_merged if c.get("retrieval_stage")}),
            **fulltext_summary,
            **assembly_meta,
            **preview_meta,
        }
        return evidence_text, metadata

    async def _build_v143_evidence(self, question: str, question_type: str):
        pos_docs, _ = await self._retrieve_shared_evidence(question)
        candidate_pmids = [doc.pmid for doc in pos_docs if getattr(doc, "pmid", None)]

        neg_docs = []
        if question_type == "yesno":
            neg_terms = "no effect OR not effective OR no association OR failed OR no benefit"
            neg_query = f"{question} {neg_terms}"
            neg_docs, _ = await retrieve_dense_evidence(
                neg_query,
                qdrant=self._qdrant,
                pool=self._pool,
                top_k=self._top_k,
                collection=self._collection,
            )
            for doc in neg_docs:
                if getattr(doc, "pmid", None) and doc.pmid not in candidate_pmids:
                    candidate_pmids.append(doc.pmid)

        if question_type == "yesno":
            seen = {d.pmid for d in pos_docs}
            abstract_docs = list(pos_docs)
            for doc in neg_docs:
                if doc.pmid not in seen:
                    seen.add(doc.pmid)
                    abstract_docs.append(doc)
            abstract_docs.sort(key=lambda d: -d.score)
        else:
            abstract_docs = pos_docs

        abstract_evidence_text = format_pubmed_evidence(abstract_docs[:self._top_k], max_docs=20)

        if not self._section_search or not self._assembler or not self._gate:
            return abstract_evidence_text, {
                "section_aware_retrieval": False,
                "coverage_miss": False,
                "section_aware_chunks": 0,
                "distinct_pmids_in_section_chunks": 0,
                "section_aware_roles": [],
                "section_aware_stages": [],
                "candidate_pmids": len(candidate_pmids),
                "candidate_have_pmc_in_articles": 0,
                "candidate_in_papergraph": 0,
                "fulltext_gate_attempted": False,
                "fulltext_gate": False,
                "fulltext_gate_reason": "gate_unavailable",
                "fulltext_gate_mode": "abstract_only",
                "evidence_route": "abstract_only",
                "evidence_route_reason": "gate_unavailable",
                "evidence_route_mode": "abstract_only",
                "final_evidence_source": "abstract_only",
            }

        fulltext_summary, fulltext_pmids = await self._summarize_candidate_fulltext(candidate_pmids)
        precheck = self._gate.precheck(
            question=question,
            question_type=question_type,
            candidate_have_pmc_in_articles=fulltext_summary["candidate_have_pmc_in_articles"],
            candidate_in_papergraph=fulltext_summary["candidate_in_papergraph"],
        )
        if not precheck.escalate:
            return abstract_evidence_text, {
                "section_aware_retrieval": False,
                "coverage_miss": precheck.reason == "coverage_miss",
                "section_aware_chunks": 0,
                "distinct_pmids_in_section_chunks": 0,
                "section_aware_roles": [],
                "section_aware_stages": [],
                **fulltext_summary,
                "fulltext_gate_attempted": False,
                "fulltext_gate": False,
                "fulltext_gate_reason": precheck.reason,
                "fulltext_gate_pre_reason": precheck.reason,
                "fulltext_gate_post_reason": "",
                "fulltext_gate_mode": precheck.mode,
                "evidence_route": "abstract_only",
                "evidence_route_reason": precheck.reason,
                "evidence_route_mode": precheck.mode,
                "assembly_mode": "abstract_only",
                "assembly_budget_tokens": self._assembler.evidence_token_budget,
                "assembly_estimated_tokens": 0,
                "assembled_units": 0,
                "assembled_support_units": 0,
                "assembled_refute_units": 0,
                "assembled_pmids": 0,
                "abstract_docs_in_context": min(len(abstract_docs), self._top_k),
                "final_evidence_source": "abstract_only",
            }

        search_kwargs = {
            "limit": self._top_k,
            "seed_limit": max(self._top_k * 4, 40),
        }
        pos_chunks = await self._section_search.search(
            query=question,
            paper_ids=sorted(fulltext_pmids),
            **search_kwargs,
        )

        neg_chunks = []
        if question_type == "yesno":
            neg_terms = "no effect OR not effective OR no association OR failed OR no benefit"
            neg_query = f"{question} {neg_terms}"
            neg_chunks = await self._section_search.search(
                query=neg_query,
                paper_ids=sorted(fulltext_pmids),
                **search_kwargs,
            )

        raw_merged = self._merge_section_chunks(pos_chunks, neg_chunks)[: self._top_k]
        section_roles = sorted({c.get("section_role", "") for c in raw_merged if c.get("section_role")})
        section_stages = sorted({c.get("retrieval_stage", "") for c in raw_merged if c.get("retrieval_stage")})

        fulltext_evidence_text, assembly_meta = self._assembler.assemble(
            question_type=question_type,
            pos_chunks=pos_chunks[: self._top_k],
            neg_chunks=neg_chunks[: self._top_k],
            abstract_evidence_text="",
        )
        postcheck = self._gate.postcheck(
            question_type=question_type,
            distinct_pmids_in_section_chunks=len({c.get("pmid") for c in raw_merged if c.get("pmid")}),
            section_aware_roles=section_roles,
            assembly_mode=assembly_meta.get("assembly_mode", "retrieval_miss"),
            assembled_units=assembly_meta.get("assembled_units", 0),
            assembled_support_units=assembly_meta.get("assembled_support_units", 0),
            assembled_refute_units=assembly_meta.get("assembled_refute_units", 0),
            assembled_pmids=assembly_meta.get("assembled_pmids", 0),
        )

        if not postcheck.escalate:
            return abstract_evidence_text, {
                "section_aware_retrieval": False,
                "coverage_miss": False,
                "section_aware_chunks": len(raw_merged),
                "distinct_pmids_in_section_chunks": len({c.get("pmid") for c in raw_merged if c.get("pmid")}),
                "section_aware_roles": section_roles,
                "section_aware_stages": section_stages,
                **fulltext_summary,
                **assembly_meta,
                "fulltext_gate_attempted": True,
                "fulltext_gate": False,
                "fulltext_gate_reason": postcheck.reason,
                "fulltext_gate_pre_reason": precheck.reason,
                "fulltext_gate_post_reason": postcheck.reason,
                "fulltext_gate_mode": postcheck.mode,
                "evidence_route": "abstract_only",
                "evidence_route_reason": postcheck.reason,
                "evidence_route_mode": postcheck.mode,
                "abstract_docs_in_context": min(len(abstract_docs), self._top_k),
                "final_evidence_source": "abstract_only",
            }

        abstract_token_budget = 6500 - self._assembler.evidence_token_budget
        abstract_cap = 8 if question_type == "yesno" else 6 if question_type in {"list", "summary"} else 8
        abstract_context_text, abstract_docs_used = self._build_budgeted_abstract_evidence(
            abstract_docs,
            max_docs_cap=min(self._top_k, abstract_cap),
            token_budget=abstract_token_budget,
        )
        combined_evidence = self._combine_v143_evidence(abstract_context_text, fulltext_evidence_text)

        return combined_evidence, {
            "section_aware_retrieval": True,
            "coverage_miss": False,
            "section_aware_chunks": len(raw_merged),
            "distinct_pmids_in_section_chunks": len({c.get("pmid") for c in raw_merged if c.get("pmid")}),
            "section_aware_roles": section_roles,
            "section_aware_stages": section_stages,
            **fulltext_summary,
            **assembly_meta,
            "fulltext_gate_attempted": True,
            "fulltext_gate": True,
            "fulltext_gate_reason": postcheck.reason,
            "fulltext_gate_pre_reason": precheck.reason,
            "fulltext_gate_post_reason": postcheck.reason,
            "fulltext_gate_mode": postcheck.mode,
            "evidence_route": "abstract_plus_section_fulltext",
            "evidence_route_reason": postcheck.reason,
            "evidence_route_mode": postcheck.mode,
            "abstract_docs_in_context": abstract_docs_used,
            "final_evidence_source": "abstract_plus_section_fulltext",
        }

    async def _precall_tools(self, question: str, question_type: str,
                             router_genes: list[str] | None = None) -> str:
        """Auto-invoke specialized tools based on question content.

        Returns tool results as formatted evidence text, or empty string.
        Ensures agent sees authoritative tool data even if it doesn't
        write REPL code to call tools itself.
        """
        import re as _re
        import os as _os
        results = []

        try:
            from src.rlm.tools import RETRIEVAL_TOOLS
            gene_resolve = RETRIEVAL_TOOLS.get('gene_resolve')
            uniprot_search = RETRIEVAL_TOOLS.get('uniprot_search')
            go_search = RETRIEVAL_TOOLS.get('go_search')

            q_lower = question.lower()

            # Extract entity names from question (support gene names like RP11-17A4.3, RNU6-1143P)
            entities = _re.findall(
                r'(?:gene symbol of |gene |protein |about |for |of |with )([A-Z][A-Za-z0-9._-]{1,20})\b',
                question
            )
            entities += _re.findall(r'\b([A-Z][A-Z0-9._-]{1,20})\b', question)
            # Filter out common English words captured by regex
            stopwords = {'THE', 'AND', 'FOR', 'WITH', 'FROM', 'WHAT', 'WHICH', 'WHERE', 'HOW', 'NOT', 'DOES'}
            entities = [e for e in entities if e.upper() not in stopwords]
            entities = list(dict.fromkeys(entities))[:5]  # dedup, max 5
            # XC_TOOLS_ROUTER=1: prefer the router's LLM-extracted genes over the
            # regex. The regex matches any 3+ char uppercase token, so on clinical
            # text it yields BMI/CT/HCO3/ECG/ASIS (575 distinct junk symbols on
            # MedXpertQA); the router extracted 24 items' worth of real genes with
            # no observed false positive. PREFERENCE, not union — unioning is what
            # let the junk through in _fetch_atlas_dict.
            if _os.environ.get("XC_TOOLS_ROUTER") == "1":
                _rg = list(router_genes or [])
                entities = _rg[:5] if _rg else []

            # Gene resolution (full info including location)
            if entities and any(
                w in q_lower for w in ['gene', 'symbol', 'alias', 'chromosome', 'location', 'snp']
            ):
                from src.rlm.tools import gene_resolve as _gene_resolve
                for entity in entities[:3]:
                    try:
                        info = _gene_resolve(entity)
                        if info and info.get('symbol'):
                            # Normalize location to chr format for GeneTuring compatibility
                            loc = info.get('location', '')
                            if loc:
                                chrom = loc.split('q')[0].split('p')[0]  # "8q13.1" -> "8"
                                info['chromosome'] = f'chr{chrom}'
                            parts = [f"symbol={info['symbol']}"]
                            if info.get('name'): parts.append(f"name={info['name']}")
                            if info.get('chromosome'): parts.append(f"chromosome={info['chromosome']} (use this exact format for chromosome answers)")
                            if loc: parts.append(f"cytoband={loc}")
                            if info.get('aliases'): parts.append(f"aliases={info['aliases']}")
                            results.append(f"[gene_info({entity})] {', '.join(parts)}")
                    except Exception as e:
                        import logging
                        logging.getLogger(__name__).warning(f"gene_resolve({entity}) failed: {e}")

            # Genomics structured lookups (NCBI dbSNP + MyGene): facts not in
            # literature — SNP->gene/chromosome, gene->chromosome/protein-coding.
            from src.rlm.tools import (
                snp_lookup as _snp_lookup,
                gene_genomic_info as _gene_genomic_info,
            )
            for rsid in _re.findall(r'\brs\d{3,}\b', question, _re.IGNORECASE)[:3]:
                try:
                    s = _snp_lookup(rsid)
                    parts = []
                    if s.get('gene'):
                        parts.append(f"gene={s['gene']}")
                    if s.get('chromosome'):
                        parts.append(f"chromosome={s['chromosome']} (use this exact format)")
                    if parts:
                        results.append(f"[snp_lookup({rsid})] {', '.join(parts)}")
                except Exception as e:
                    import logging
                    logging.getLogger(__name__).warning(f"snp_lookup({rsid}) failed: {e}")

            if entities and any(
                w in q_lower for w in ['codes a protein', 'protein-coding', 'protein coding',
                                       'which chromosome', 'located on']
            ):
                for entity in entities[:3]:
                    try:
                        g = _gene_genomic_info(entity)
                        parts = []
                        if g.get('protein_coding_answer'):
                            parts.append(f"protein_coding={g['protein_coding_answer']} (answer TRUE or FALSE)")
                        if g.get('chromosome'):
                            parts.append(f"chromosome={g['chromosome']} (use this exact format)")
                        if parts:
                            results.append(f"[gene_genomic_info({entity})] {', '.join(parts)}")
                    except Exception as e:
                        import logging
                        logging.getLogger(__name__).warning(f"gene_genomic_info({entity}) failed: {e}")

            # DNA alignment (NCBI BLAST cache): sequence -> genome coordinates
            # ("align ... to the human genome") or source organism.
            if any(w in q_lower for w in ['align the dna', 'which organism does the dna',
                                          'dna sequence to the human genome']):
                from src.rlm.tools import blast_dna_lookup as _blast
                seq = question.split(':')[-1].strip()
                if _re.fullmatch(r'[ACGTNacgtn]{20,}', seq):
                    try:
                        b = _blast(seq)
                        if b.get('answer'):
                            label = ('genome_coordinates' if b.get('mode') == 'genome'
                                     else 'source_organism')
                            results.append(f"[blast_dna_lookup] {label}={b['answer']} (use this exact answer)")
                    except Exception as e:
                        import logging
                        logging.getLogger(__name__).warning(f"blast_dna_lookup failed: {e}")

            # UniProt
            if entities and uniprot_search and any(
                w in q_lower for w in ['protein', 'function', 'disease', 'regulation']
            ):
                for entity in entities[:2]:
                    try:
                        info = uniprot_search(entity, limit=1)
                        if info:
                            results.append(f"[uniprot_search({entity})] {str(info)[:300]}")
                    except Exception as e:
                        import logging
                        logging.getLogger(__name__).warning(f"uniprot_search({entity}) failed: {e}")

            # GO search
            if go_search and any(
                w in q_lower for w in ['ontology', 'go term', 'biological process', 'molecular function']
            ):
                try:
                    genes = _re.findall(r'\b([A-Z][A-Z0-9]{1,10})\b', question)
                    if genes:
                        info = go_search(', '.join(genes[:5]))
                        if info:
                            results.append(f"[go_search] {str(info)[:300]}")
                except Exception as e:
                    import logging
                    logging.getLogger(__name__).warning(f"go_search failed: {e}")

        except Exception as e:
            import logging
            logging.getLogger(__name__).warning(f"_precall_tools failed: {e}")

        if results:
            return "## Tool Results (pre-fetched)\n" + "\n".join(results)
        return ""

    async def generate(self, question, question_type, context=None, options=None):
        start = time.perf_counter()
        pos_docs = []

        section_meta = {}
        if self._version == "v14.2":
            evidence_text, section_meta = await self._build_v142_evidence(question, question_type)
        elif self._version == "v14.3":
            evidence_text, section_meta = await self._build_v143_evidence(question, question_type)
        elif self._version == "v14.1":
            evidence_text, section_meta = await self._build_v141_evidence(question, question_type)
        else:
            pos_docs, _ = await self._retrieve_shared_evidence(question)
            evidence_text = format_pubmed_evidence(pos_docs, max_docs=20)

        # Stage 2: Tool pre-calling (auto-invoke specialized tools, skip if tools disabled)
        tool_evidence = await self._precall_tools(question, question_type) if self._enable_tools else ""
        if tool_evidence:
            if self._version in ("v14.2", "v14.3"):
                tool_evidence = tool_evidence[:2400].rstrip()
            evidence_text = tool_evidence + "\n\n" + evidence_text

        # Stage 3: RLM Agent handles ALL question types
        # - v14/v14-dh: yesno gets dual abstract evidence
        # - v14.1/v14.2/v14.3: evidence has already been assembled upstream
        # - factoid/list/summary: pre-fetched evidence + tools as starting point
        if question_type == "yesno" and self._version not in ("v14.1", "v14.2", "v14.3"):
            # YesNo: provide DUAL-retrieved evidence (pos + neg) so agent sees both sides
            neg_terms = "no effect OR not effective OR no association OR failed OR no benefit"
            neg_query = f"{question} {neg_terms}"
            neg_docs, _ = await retrieve_dense_evidence(
                neg_query, qdrant=self._qdrant, pool=self._pool,
                top_k=self._top_k, collection=self._collection,
            )
            seen = {d.pmid for d in pos_docs}
            all_docs = list(pos_docs)
            for d in neg_docs:
                if d.pmid not in seen:
                    seen.add(d.pmid)
                    all_docs.append(d)
            all_docs.sort(key=lambda d: -d.score)
            evidence_text = format_pubmed_evidence(all_docs[:self._top_k], max_docs=20)
            ctx = {"evidence": evidence_text}
        else:
            ctx = {"evidence": evidence_text, "options": options} if options else {"evidence": evidence_text}

        result = await self._pipeline.answer_async(
            question=question,
            question_type=question_type,
            context=ctx if ctx else None,
        )

        # Stage 4: Constrained re-judgment for ALL question types
        if self._enable_rejudge:
            agent_reasoning = result.answer or ""
            rejudge_context = f"Agent analysis:\n{agent_reasoning}\n\nOriginal evidence:\n{evidence_text}"
            answer, _ = await generate_constrained_answer(
                question=question, question_type=question_type,
                context=rejudge_context, options=options,
            )
        else:
            # Ablation: use agent's raw answer directly
            answer = extract_answer(result.answer, question_type, options)

        # Extract token usage from RLM completion
        token_meta = {}
        if result.raw_completion and result.raw_completion.usage_summary:
            for model_id, usage in result.raw_completion.usage_summary.model_usage_summaries.items():
                token_meta["input_tokens"] = token_meta.get("input_tokens", 0) + usage.total_input_tokens
                token_meta["output_tokens"] = token_meta.get("output_tokens", 0) + usage.total_output_tokens
                token_meta["llm_calls"] = token_meta.get("llm_calls", 0) + usage.total_calls

        return ModelResponse(
            answer=answer, response_text=result.answer,
            latency_ms=(time.perf_counter() - start) * 1000,
            metadata={"method": f"{self._version}_agent_primed", "docs": len(pos_docs) if pos_docs else section_meta.get("candidate_pmids", 0),
                       "iterations": result.iterations_used, **token_meta, **section_meta},
        )


class NoContextClient(UnifiedEvidenceClient):
    """No-Context baseline: just LLM, no retrieval at all."""

    async def generate(self, question, question_type, context=None, options=None):
        start = time.perf_counter()
        answer, answer_text = await generate_constrained_answer(
            question=question, question_type=question_type,
            context="No evidence available. Answer based on your knowledge.",
            options=options,
        )
        return ModelResponse(
            answer=answer, response_text=answer_text,
            latency_ms=(time.perf_counter() - start) * 1000,
            metadata={"method": "no_context"},
        )


# [paper_reproduction] The coding-agent-CLI harness arm (HarnessCLIClient) and its
# evidence-transplant follow-up (WebEvidenceClient) are removed from this release: neither
# produced a Table 1 cell; the first drives third-party coding-agent CLIs with local account
# state, and the second only consumes that arm's fetched web pages.


class GoldenContextDenseClient(ModelClient):
    """Golden context baseline: use dataset-provided context + constrained answer.

    For datasets with golden context (pubmedqa, bioasq), this shows the ceiling
    performance when the retrieval is perfect.
    """

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        pass

    async def generate(self, question, question_type, context=None, options=None):
        start = time.perf_counter()
        if context:
            ctx_text = "\n".join(context) if isinstance(context, list) else str(context)
        else:
            ctx_text = "No context provided."
        answer, answer_text = await generate_constrained_answer(
            question=question, question_type=question_type,
            context=ctx_text, options=options,
        )
        return ModelResponse(
            answer=answer, response_text=answer_text,
            latency_ms=(time.perf_counter() - start) * 1000,
            metadata={"method": "golden_context"},
        )


HYDE_PROMPT = (
    "Please write a short scientific PubMed-style abstract that answers the "
    "question below. Write in one paragraph, 4-6 sentences, focused on the "
    "biomedical facts directly relevant to the question.\n\n"
    "Question: {question}\n\nAbstract:"
)


class HyDEClient(UnifiedEvidenceClient):
    """HyDE (Gao et al. 2022): LLM writes a hypothetical passage, we embed
    that passage alone, then dense-retrieve with that vector.

    Paper-faithful variant: retrieval uses ONLY the hypothesis embedding.
    When ``mix_query=True`` we instead average [question, hypothesis]
    embeddings (our earlier hybrid variant, kept for comparison).
    """

    def __init__(self, top_k: int = 20, collection: str = "paper-full",
                 mix_query: bool = False):
        super().__init__(top_k=top_k, collection=collection)
        self._mix_query = mix_query

    async def generate(self, question, question_type, context=None, options=None):
        from utils.clients import embed_client, llm_client

        start = time.perf_counter()

        try:
            hyp = await llm_client.chat(
                HYDE_PROMPT.format(question=question),
                max_tokens=256,
                temperature=0.1,
            )
            hyp = (hyp or "").strip()
        except Exception:
            hyp = ""

        if self._mix_query:
            import numpy as np
            texts = [question] + ([hyp] if hyp else [])
            embs = await embed_client.embed(texts)
            vec = np.mean(np.array(embs), axis=0).tolist()
        else:
            # Paper-faithful HyDE: embed hypothesis alone (fall back to
            # question only when the LLM fails to produce a passage).
            hyde_text = hyp or question
            vec = await embed_client.embed_single(hyde_text)

        docs, _ = await retrieve_dense_evidence(
            question,
            qdrant=self._qdrant, pool=self._pool,
            top_k=self._top_k, collection=self._collection,
            query_vector=vec,
        )

        if not docs:
            return ModelResponse(answer="", response_text="No evidence found.",
                                latency_ms=(time.perf_counter() - start) * 1000)

        full_context, _, _ = build_dense_context(docs)
        answer, answer_text = await generate_constrained_answer(
            question=question, question_type=question_type,
            context=full_context, options=options,
        )
        return ModelResponse(
            answer=answer, response_text=answer_text,
            latency_ms=(time.perf_counter() - start) * 1000,
            metadata={"method": "hyde_mixed" if self._mix_query else "hyde",
                      "docs": len(docs), "hyp_chars": len(hyp)},
        )


class RankRAGClient(UnifiedEvidenceClient):
    """RankRAG-style baseline: retrieve a wider pool, rerank with the
    cross-encoder reranker (Qwen3-Reranker-8B), keep top-k for answering.

    Differs from Dense by surfacing docs whose dense score is low but whose
    query-document relevance (as judged by the reranker) is high.
    """

    def __init__(self, top_k: int = 20, rerank_pool: int = 50, collection: str = "paper-full"):
        super().__init__(top_k=top_k, collection=collection)
        self._rerank_pool = max(rerank_pool, top_k)

    async def generate(self, question, question_type, context=None, options=None):
        from utils.clients import rerank_client

        start = time.perf_counter()

        # Retrieve a wider candidate pool via shared dense retrieval.
        docs, _ = await retrieve_dense_evidence(
            question,
            qdrant=self._qdrant, pool=self._pool,
            top_k=self._rerank_pool, collection=self._collection,
        )
        if not docs:
            return ModelResponse(answer="", response_text="No evidence found.",
                                latency_ms=(time.perf_counter() - start) * 1000)

        # Build (title + abstract) as documents for the reranker.
        passages = [f"{d.title}\n{d.abstract}".strip() for d in docs]
        try:
            scored = await rerank_client.rerank(
                question, passages,
                top_k=self._top_k, max_concurrent=8,
            )
        except Exception:
            scored = None

        if scored:
            # scored is list of (original_idx, score) sorted desc.
            ordered = []
            for rank, (idx, score) in enumerate(scored, start=1):
                if 0 <= idx < len(docs):
                    d = docs[idx]
                    d.score = float(score)
                    d.rank = rank
                    ordered.append(d)
            docs = ordered[: self._top_k]
        else:
            docs = docs[: self._top_k]

        full_context, _, _ = build_dense_context(docs, max_context_tokens=4000)
        answer, answer_text = await generate_constrained_answer(
            question=question, question_type=question_type,
            context=full_context, options=options,
        )
        return ModelResponse(
            answer=answer, response_text=answer_text,
            latency_ms=(time.perf_counter() - start) * 1000,
            metadata={"method": "rankrag", "docs": len(docs), "pool": self._rerank_pool},
        )


# =============================================================================
# FLARE (Jiang et al. 2023, EMNLP) — FAITHFUL: forward-looking active
# retrieval. Generate a look-ahead sentence; if it contains a low-confidence
# token, mask the low-confidence tokens, use the masked sentence as the
# retrieval query, then REGENERATE that sentence conditioned on the newly
# retrieved evidence. Backbone is the shared LLM (GPT-3.5/4 -> {model} swap);
# the active-retrieval mechanism is preserved (FLARE-direct variant).
# =============================================================================

import re as _re_flare


class FLAREClient(UnifiedEvidenceClient):
    """Faithful FLARE-direct (Jiang et al. 2023) with the shared LLM backbone.

    Per the paper: generate the upcoming sentence as a "look-ahead"; if any
    token's probability is below ``trigger_prob`` the model is uncertain, so we
    form a retrieval query by masking the low-confidence tokens (keeping the
    high-confidence ones, ``mask_prob``), retrieve, and regenerate the sentence
    with the new evidence. Confident sentences are accepted without new
    retrieval. Only the backbone differs from the paper; the look-ahead +
    confidence-triggered active retrieval loop is intact.
    """

    def __init__(self, top_k: int = 20, collection: str = "paper-full",
                 trigger_prob: float = 0.8, mask_prob: float = 0.4,
                 look_ahead_tokens: int = 64, max_steps: int = 8,
                 ret_topk: int = 2):
        super().__init__(top_k=top_k, collection=collection)
        self._trigger_prob = trigger_prob   # token prob <= this -> low confidence -> retrieve
        self._mask_prob = mask_prob         # token prob <= this -> mask out of query
        self._look_ahead_tokens = look_ahead_tokens
        self._max_steps = max_steps         # max sentences in the CoT
        self._ret_topk = ret_topk           # paper: topk=2 docs per retrieval
        # Round-robin across all configured LLM endpoints (e.g. the two Qwen
        # instances on 8004/8005) so FLARE's many look-ahead calls spread the
        # load. Honours FLARE_LLM_URL as a single-endpoint override.
        _cfg = get_config()
        self._fixed_url = os.environ.get("FLARE_LLM_URL")
        self._llm_cfg = _cfg.llm
        self._api_key = _cfg.llm.api_key
        self._model = _cfg.llm.model
        self._http = None

    def _next_url(self) -> str:
        if self._fixed_url:
            return self._fixed_url
        return self._llm_cfg.get_server()  # round-robin endpoint

    async def __aenter__(self):
        await super().__aenter__()
        import httpx
        self._http = httpx.AsyncClient(timeout=60.0)
        return self

    async def __aexit__(self, *args):
        if self._http is not None:
            await self._http.aclose()
        await super().__aexit__(*args)

    async def _complete_lp(self, prompt: str, max_tokens: int):
        """Chat-completions call returning (text, tokens, token_probs).

        Uses the chat endpoint with thinking disabled (Qwen-3 emits <think>
        spans on the raw completions endpoint); per-token logprobs come from
        ``logprobs.content`` and drive FLARE's confidence trigger.
        """
        import math as _m
        payload = {
            "model": self._model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens, "temperature": 0.0,
            "logprobs": True, "top_logprobs": 1,
            "chat_template_kwargs": {"enable_thinking": False},
        }
        base = self._next_url()
        r = await self._http.post(
            f"{base.rstrip('/')}/chat/completions", json=payload,
            headers={"Authorization": f"Bearer {self._api_key}"},
        )
        r.raise_for_status()
        ch = r.json()["choices"][0]
        text = (ch.get("message") or {}).get("content", "") or ""
        content_lp = ((ch.get("logprobs") or {}).get("content")) or []
        toks = [c.get("token", "") for c in content_lp]
        probs = [_m.exp(c["logprob"]) if c.get("logprob") is not None else 1.0
                 for c in content_lp]
        return text, toks, probs

    @staticmethod
    def _first_sentence(text: str) -> str:
        m = _re_flare.search(r"^(.*?[.!?])(\s|$)", text.strip(), _re_flare.S)
        return (m.group(1) if m else text).strip()

    def _mask_query(self, toks: list, probs: list, sent_text: str) -> str:
        """Keep high-confidence tokens (prob > mask_prob) as the query."""
        if not toks:
            return sent_text
        # only over the span covering the first sentence
        kept = []
        acc = ""
        for t, p in zip(toks, probs):
            acc += t
            kept.append(t if p > self._mask_prob else " ")
            if acc.strip().endswith((".", "!", "?")) and len(acc.strip()) >= 5:
                break
        return "".join(kept).strip() or sent_text

    _FLARE_FMT = {
        "yesno": "Begin your answer with 'yes' or 'no', then briefly justify.",
        "mcq": "Begin your answer with the single correct option letter, then justify.",
        "mcq_multi": "Begin with the correct option letters (comma-separated), then justify.",
        "factoid": "Begin with the short answer entity/phrase, then justify.",
        "list": "Begin with the comma-separated list of items, then justify.",
        "summary": "Write a concise 2-4 sentence summary of the key findings.",
        "expression": "Begin with the comma-separated tissue names, then justify.",
    }

    def _build_prompt(self, question, question_type, options, ctx_docs):
        # FLARE uses topk=2 documents per the paper's config.
        ctx, _, _ = build_dense_context(ctx_docs[: self._ret_topk], max_context_tokens=3000)
        opt = ""
        if options:
            opt = "\nOptions:\n" + "\n".join(f"{k}. {options[k]}" for k in sorted(options))
        fmt = self._FLARE_FMT.get(question_type, "Begin with the direct answer, then justify.")
        return (
            "Answer the biomedical question using the evidence. "
            f"{fmt}\n\n"
            f"Evidence:\n{ctx}\n\nQuestion: {question}{opt}\n\nAnswer:"
        )

    @staticmethod
    def _first_sentence_span(toks: list, probs: list) -> list:
        """Probabilities of the tokens making up the first sentence."""
        span, acc = [], ""
        for t, p in zip(toks, probs):
            acc += t
            span.append(p)
            if acc.strip().endswith((".", "!", "?")) and len(acc.strip()) >= 5:
                break
        return span

    async def generate(self, question, question_type, context=None, options=None):
        """Faithful FLARE-direct (answer-first). Generate the answer attempt as
        a look-ahead; if any look-ahead token is low-confidence, mask the
        low-confidence tokens into a retrieval query, retrieve (topk=2,
        only_use_look_ahead), and REGENERATE the whole answer with the new
        evidence; repeat until confident. The model's own generation is the
        final answer (no shared constrained step); the answer-first format lets
        the leading yes/no/letter/entity be extracted.
        """
        start = time.perf_counter()
        docs, _ = await self._retrieve_shared_evidence(question)   # pre-retrieval
        budget = 256 if question_type == "summary" else 96
        gen = ""
        n_retr = 1
        n_trig = 0
        for _ in range(self._max_steps):
            prompt = self._build_prompt(question, question_type, options, docs)
            try:
                text, toks, probs = await self._complete_lp(prompt, budget)
            except Exception as exc:  # noqa: BLE001
                LOGGER.warning("flare generate failed: %s", exc)
                break
            if not text.strip():
                break
            gen = text.strip()
            # confidence trigger over the look-ahead's first sentence
            span = self._first_sentence_span(toks, probs)
            low_conf = any(p <= self._trigger_prob for p in span) if span else False
            if not low_conf:
                break   # confident -> accept
            n_trig += 1
            query = self._mask_query(toks, probs, self._first_sentence(text))
            if not query.strip():
                break
            new_docs, _ = await self._retrieve_shared_evidence(query)
            n_retr += 1
            docs = new_docs   # topk=2, only_use_look_ahead: replace context

        # The model's own answer-first generation IS the answer (faithful FLARE,
        # no shared constrained step). Strict extraction reads the leading
        # yes/no/letter/entity; lenient is a fallback for verbose phrasing.
        answer = extract_answer(gen, question_type, options, mode="strict")
        if not answer:
            answer = extract_answer(gen, question_type, options, mode="lenient")
        return ModelResponse(
            answer=answer, response_text=gen.strip(),
            latency_ms=(time.perf_counter() - start) * 1000,
            metadata={"method": "flare", "retrievals": n_retr, "triggers": n_trig,
                      "ret_topk": self._ret_topk},
        )


CHAIN_OF_NOTE_INSTR = (
    "Task: You are given a biomedical question and a set of retrieved PubMed "
    "documents. Before answering, write a short reading note for each "
    "document that classifies it as one of:\n"
    "  [RELEVANT]  directly supports/answers the question\n"
    "  [PARTIAL]   related but does not fully answer\n"
    "  [IRRELEVANT] unrelated to the question\n"
    "For RELEVANT/PARTIAL notes, extract the 1-2 key facts in one sentence.\n"
    "After all notes, synthesize the evidence and produce a concise final "
    "answer. If no document is useful, rely on your own biomedical knowledge "
    "and say so.\n\n"
)


class ChainOfNoteClient(UnifiedEvidenceClient):
    """Chain-of-Note (Yu et al. 2023): retrieve → LLM writes per-document
    reading notes (RELEVANT / PARTIAL / IRRELEVANT + key facts) → LLM answers
    from the notes. Improves robustness when retrieval is noisy.

    We run the note-writing and synthesis as one LLM call (matching the
    original paper's single-pass inference) and then pass the resulting
    synthesis as context to the shared constrained-answer path so the
    evaluation format stays identical to Dense.
    """

    async def generate(self, question, question_type, context=None, options=None):
        from utils.clients import llm_client

        start = time.perf_counter()
        docs, _ = await self._retrieve_shared_evidence(question)
        if not docs:
            return ModelResponse(answer="", response_text="No evidence found.",
                                latency_ms=(time.perf_counter() - start) * 1000)

        # Build numbered doc block (truncate to keep prompt sane).
        doc_lines = []
        for i, d in enumerate(docs[: self._top_k], start=1):
            abs_text = (d.abstract or "").strip()
            if len(abs_text) > 800:
                abs_text = abs_text[:800] + "..."
            doc_lines.append(
                f"[{i}] PMID:{d.pmid} {d.title}\n{abs_text}"
            )
        docs_block = "\n\n".join(doc_lines)

        note_prompt = (
            CHAIN_OF_NOTE_INSTR
            + f"Question: {question}\n\nRetrieved documents:\n{docs_block}\n\n"
            "Reading notes (one per document):\n"
        )

        try:
            notes_text = await llm_client.chat(
                note_prompt, max_tokens=1024, temperature=0.1,
            )
        except Exception:
            notes_text = ""

        # Feed the note synthesis as the context for the constrained answerer.
        full_context, _, _ = build_dense_context(docs[: self._top_k], max_context_tokens=2000)
        merged_context = (
            "## Chain-of-Note synthesis\n"
            f"{notes_text.strip()}\n\n"
            "## Source documents (reference)\n"
            f"{full_context}"
        )

        answer, answer_text = await generate_constrained_answer(
            question=question, question_type=question_type,
            context=merged_context, options=options,
        )
        return ModelResponse(
            answer=answer, response_text=answer_text,
            latency_ms=(time.perf_counter() - start) * 1000,
            metadata={"method": "chain_of_note", "docs": len(docs), "notes_chars": len(notes_text)},
        )


IRCOT_STEP_INSTR = (
    "You are answering a biomedical question with the help of retrieved "
    "PubMed abstracts. Use the documents and the reasoning written so far to "
    "write the NEXT ONE reasoning sentence toward the answer. "
    "Be concrete and grounded in the documents. When you reach the final "
    "answer, end the sentence with 'So the answer is: <answer>.' and stop.\n\n"
)


def _ircot_is_terminal(sentence: str) -> bool:
    """Accept only explicit 'the answer is: X.' style terminators."""
    import re
    s = (sentence or "").strip()
    return bool(re.search(r"(?:so\s+)?the answer is[:\s]+\S.*[.!?]\s*$", s, re.IGNORECASE))


class IRCoTClient(UnifiedEvidenceClient):
    """IRCoT (Trivedi et al. 2023): interleave retrieval with chain-of-thought.

    Loop for up to ``max_steps``:
      1. Show current docs + prior CoT, ask LLM for the next CoT sentence.
      2. If sentence signals the answer, stop.
      3. Otherwise retrieve more docs using that sentence as the query and
         accumulate them.
    Final step: rerank the accumulated pool with Qwen3-Reranker and keep
    ``top_k`` docs for the answer (Option C — preserves IRCoT interleaving
    semantics while guaranteeing step-retrieved docs can compete for the
    answerer's token budget).
    """

    def __init__(self, top_k: int = 20, max_steps: int = 3, step_top_k: int = 5,
                 collection: str = "paper-full", rerank_final: bool = True):
        super().__init__(top_k=top_k, collection=collection)
        self._max_steps = max_steps
        self._step_top_k = step_top_k
        self._rerank_final = rerank_final

    @staticmethod
    def _first_sentence(text: str) -> str:
        text = (text or "").strip()
        if not text:
            return ""
        import re
        parts = re.split(r"(?<=[.!?])\s", text, maxsplit=1)
        return parts[0].strip()

    def _merge_docs(self, pool, new_docs):
        seen = {(d.pmid, d.abstract) for d in pool}  # (pmid, text): full-text chunks share a PMID
        for d in new_docs:
            if (d.pmid, d.abstract) not in seen:
                pool.append(d)
                seen.add((d.pmid, d.abstract))
        return pool

    async def generate(self, question, question_type, context=None, options=None):
        from utils.clients import llm_client, rerank_client

        start = time.perf_counter()
        docs, _ = await retrieve_dense_evidence(
            question,
            qdrant=self._qdrant, pool=self._pool,
            top_k=self._top_k, collection=self._collection,
        )
        if not docs:
            return ModelResponse(answer="", response_text="No evidence found.",
                                latency_ms=(time.perf_counter() - start) * 1000)

        cot_sentences: list[str] = []
        steps_run = 0
        for step in range(self._max_steps):
            steps_run += 1
            # Step-time context: current top-20 from the accumulated pool, so
            # the LLM can see recently-retrieved docs without blowing budget.
            step_ctx, _, _ = build_dense_context(docs[: self._top_k],
                                                 max_context_tokens=3500)
            prior_cot = " ".join(cot_sentences).strip() or "(none yet)"
            prompt = (
                IRCOT_STEP_INSTR
                + f"Question: {question}\n\n"
                + f"Documents:\n{step_ctx}\n\n"
                + f"Reasoning so far: {prior_cot}\n\n"
                + "Next sentence:"
            )

            try:
                raw = await llm_client.chat(
                    prompt, max_tokens=120, temperature=0.1,
                )
            except Exception:
                break

            sentence = self._first_sentence(raw)
            if not sentence:
                break
            cot_sentences.append(sentence)

            if _ircot_is_terminal(sentence):
                break

            try:
                more, _ = await retrieve_dense_evidence(
                    sentence,
                    qdrant=self._qdrant, pool=self._pool,
                    top_k=self._step_top_k, collection=self._collection,
                )
                self._merge_docs(docs, more)
            except Exception:
                pass

        # Option C: rerank the accumulated pool and keep top_k for the answer.
        if len(docs) > self._top_k:
            passages = [f"{d.title}\n{d.abstract}".strip() for d in docs]
            try:
                scored = await rerank_client.rerank(
                    question, passages,
                    top_k=self._top_k, max_concurrent=8,
                )
                if scored:
                    ordered = []
                    for rank, (idx, score) in enumerate(scored, start=1):
                        if 0 <= idx < len(docs):
                            d = docs[idx]
                            d.score = float(score)
                            d.rank = rank
                            ordered.append(d)
                    docs = ordered[: self._top_k]
                else:
                    docs = docs[: self._top_k]
            except Exception:
                docs = docs[: self._top_k]

        final_context, _, _ = build_dense_context(docs, max_context_tokens=4000)
        cot_text = " ".join(cot_sentences).strip()
        merged_context = (
            "## Iterative reasoning (IRCoT)\n"
            f"{cot_text or '(no CoT produced)'}\n\n"
            "## Evidence documents\n"
            f"{final_context}"
        )
        answer, answer_text = await generate_constrained_answer(
            question=question, question_type=question_type,
            context=merged_context, options=options,
        )
        return ModelResponse(
            answer=answer, response_text=answer_text,
            latency_ms=(time.perf_counter() - start) * 1000,
            metadata={"method": "ircot", "docs": len(docs),
                      "steps": steps_run, "cot_chars": len(cot_text)},
        )


# =============================================================================
# Self-RAG (Asai et al. 2023, prompt-based variant)
# =============================================================================

SELFRAG_RETRIEVE_PROMPT = (
    "You are deciding whether external retrieval is needed to answer a "
    "biomedical question. Respond with exactly YES or NO.\n"
    "Retrieve (YES) when the question asks about specific biomedical facts, "
    "drug-disease relations, gene function, clinical evidence, or named "
    "entities that require up-to-date knowledge. Do NOT retrieve (NO) when "
    "the question is a pure definition or a closed-domain choice where the "
    "options fully determine the answer.\n\n"
    "Question: {question}\n\nAnswer (YES or NO):"
)

SELFRAG_FILTER_PROMPT = (
    "Classify the document's relevance to the biomedical question.\n"
    "Output exactly one label and nothing else:\n"
    "REL = directly answers/supports the question\n"
    "PARTIAL = related but incomplete/background evidence\n"
    "IRREL = off-topic or not useful\n\n"
    "Question: {question}\n\n"
    "Document: {doc}\n\nRelevance:"
)


class SelfRAGClient(UnifiedEvidenceClient):
    """Self-RAG (Asai et al. 2023) — prompt-based zero-training variant.

    Original Self-RAG trains Llama-2 to emit reflection tokens
    (Retrieve?, IsRel, IsSup, IsUse). We approximate it with three LLM gates:

      1. Retrieve gate: LLM decides YES/NO whether retrieval is needed.
      2. Per-doc relevance filter: LLM labels each retrieved doc as
         REL / PARTIAL / IRREL. REL + PARTIAL docs are kept.
      3. Answer: standard shared constrained-answer path on filtered docs.

    When the gate says NO we answer without retrieval (no-context path),
    mirroring Self-RAG's "only retrieve when useful" behavior.
    """

    def __init__(self, top_k: int = 20, collection: str = "paper-full",
                 keep_partial: bool = True, filter_max_concurrent: int = 8):
        super().__init__(top_k=top_k, collection=collection)
        self._keep_partial = keep_partial
        self._filter_max_concurrent = max(1, filter_max_concurrent)

    @staticmethod
    def _parse_binary_gate(raw: str | None) -> bool | None:
        import re
        match = re.search(r"\b(YES|NO)\b", (raw or "").strip().upper())
        if not match:
            return None
        return match.group(1) == "YES"

    @staticmethod
    def _parse_filter_label(raw: str | None) -> str | None:
        import re
        match = re.search(r"\b(REL|PARTIAL|IRREL)\b", (raw or "").strip().upper())
        return match.group(1) if match else None

    async def _decide_retrieve(self, question: str) -> bool:
        from utils.clients import llm_client
        try:
            raw = await llm_client.chat(
                SELFRAG_RETRIEVE_PROMPT.format(question=question),
                max_tokens=4, temperature=0.0,
            )
            parsed = self._parse_binary_gate(raw)
            return True if parsed is None else parsed
        except Exception:
            return True  # default to retrieval on failure

    async def _filter_one(self, question: str, doc) -> str:
        from utils.clients import llm_client
        doc_text = f"{doc.title}\n{(doc.abstract or '')[:600]}".strip()
        try:
            raw = await llm_client.chat(
                SELFRAG_FILTER_PROMPT.format(question=question, doc=doc_text),
                max_tokens=4, temperature=0.0,
            )
            label = self._parse_filter_label(raw)
            return label or "PARTIAL"
        except Exception:
            return "PARTIAL"  # conservative fallback: keep but flag

    async def _filter_docs(self, question: str, docs) -> list[str]:
        import asyncio
        semaphore = asyncio.Semaphore(self._filter_max_concurrent)

        async def run_one(doc):
            async with semaphore:
                return await self._filter_one(question, doc)

        return await asyncio.gather(*(run_one(doc) for doc in docs))

    async def generate(self, question, question_type, context=None, options=None):
        start = time.perf_counter()

        # Stage 1: retrieval decision
        need_retrieve = await self._decide_retrieve(question)

        if not need_retrieve:
            answer, answer_text = await generate_constrained_answer(
                question=question, question_type=question_type,
                context="No evidence available. Answer based on your knowledge.",
                options=options,
            )
            return ModelResponse(
                answer=answer, response_text=answer_text,
                latency_ms=(time.perf_counter() - start) * 1000,
                metadata={"method": "self_rag", "retrieved": False, "docs": 0},
            )

        # Stage 2: retrieve + per-doc relevance filter (bounded concurrency)
        docs, _ = await self._retrieve_shared_evidence(question)
        if not docs:
            return ModelResponse(answer="", response_text="No evidence found.",
                                latency_ms=(time.perf_counter() - start) * 1000)

        labels = await self._filter_docs(question, docs)
        kept = {"REL"} if not self._keep_partial else {"REL", "PARTIAL"}
        filtered = [d for d, lab in zip(docs, labels) if lab in kept]

        # Fallback: if filter keeps nothing, use unfiltered docs.
        if not filtered:
            filtered = docs[: self._top_k]

        full_context, _, _ = build_dense_context(filtered[: self._top_k], max_context_tokens=4000)
        answer, answer_text = await generate_constrained_answer(
            question=question, question_type=question_type,
            context=full_context, options=options,
        )
        return ModelResponse(
            answer=answer, response_text=answer_text,
            latency_ms=(time.perf_counter() - start) * 1000,
            metadata={
                "method": "self_rag",
                "retrieved": True,
                "docs_before": len(docs),
                "docs_after": len(filtered),
                "rel_count": sum(1 for l in labels if l == "REL"),
                "partial_count": sum(1 for l in labels if l == "PARTIAL"),
                "irrel_count": sum(1 for l in labels if l == "IRREL"),
            },
        )


# =============================================================================
# Self-RAG (Asai et al. 2023) — TRAINED checkpoint, native reflection tokens
# =============================================================================
# Unlike SelfRAGClient (prompt-based approximation on the shared backbone),
# this client runs the released `selfrag/selfrag_llama2_13b` checkpoint and
# uses its native reflection-token decoding (Retrieve? / IsRel / IsSup /
# IsUse), faithfully reproducing the paper's adaptive-retrieval + per-passage
# critic-reranking inference (run_short_form.call_model_rerank_w_scores_batch).
#
# It talks to a dedicated vLLM instance serving the Self-RAG checkpoint
# (default http://127.0.0.1:8009/v1, override via SELFRAG_URL). Retrieval
# reuses the SAME shared dense stage-1 as every other method, so the only
# difference vs the rest of the table is the answerer + its reflection logic.

SELFRAG_CONTROL_TOKENS = [
    "[Fully supported]", "[Partially supported]", "[No support / Contradictory]",
    "[No Retrieval]", "[Retrieval]", "[Continue to Use Evidence]",
    "[Irrelevant]", "[Relevant]", "<paragraph>", "</paragraph>",
    "[Utility:1]", "[Utility:2]", "[Utility:3]", "[Utility:4]", "[Utility:5]",
]
SELFRAG_RET_TOKENS = ["[No Retrieval]", "[Retrieval]", "[Continue to Use Evidence]"]
SELFRAG_REL_TOKENS = ["[Irrelevant]", "[Relevant]"]
SELFRAG_GRD_TOKENS = ["[Fully supported]", "[Partially supported]", "[No support / Contradictory]"]
SELFRAG_UT_TOKENS = ["[Utility:1]", "[Utility:2]", "[Utility:3]", "[Utility:4]", "[Utility:5]"]

# Per-type task instruction so the trained model emits a benchmark-shaped
# answer. Mirrors the paper's TASK_INST style (short, format-directing).
SELFRAG_TASK_INST = {
    "yesno": "Answer the question with yes or no.",
    "mcq": "Given the answer candidates, choose the single best option and respond with its letter only.",
    "mcq_multi": "Given the answer candidates, choose all correct options and respond with their letters.",
    "factoid": "Answer the question with a short factual phrase.",
    "list": "Answer the question with a comma-separated list of items.",
    "summary": "Answer the question with a concise summary.",
    "expression": "Answer the question with a comma-separated list of tissues.",
}


def selfrag_postprocess(answer: str) -> str:
    """Strip Self-RAG control tokens + EOS markers from a generation."""
    for token in SELFRAG_CONTROL_TOKENS:
        answer = answer.replace(token, "")
    for eos in ("</s>", "<|endoftext|>", "<unk>", "<s>"):
        answer = answer.replace(eos, "")
    return answer.replace("\n", " ").strip()


class SelfRAGTrainedClient(UnifiedEvidenceClient):
    """Self-RAG (Asai et al. 2023) using the released 13B checkpoint.

    Faithful reproduction of the short-form adaptive-retrieval algorithm:

      1. Decode the base (no-paragraph) prompt; read the first-step
         retrieval-reflection logprobs and decide retrieval when
         P([Retrieval]) / (P([Retrieval]) + P([No Retrieval])) > threshold.
      2. If retrieving, condition the prompt on each retrieved passage in
         turn, decode, and score the path by
         w_rel * IsRel + w_sup * IsSup + w_use * IsUse (+ optional seqscore),
         then keep the highest-scoring passage's answer.
      3. Otherwise decode the [No Retrieval] continuation directly.

    The typed benchmark answer is extracted from the cleaned generation with
    the shared ``extract_answer`` (no second LLM call, so the baseline stays
    Self-RAG-only and does not depend on the main backbone).
    """

    def __init__(self, top_k: int = 10, threshold: float = 0.2,
                 w_rel: float = 1.0, w_sup: float = 1.0, w_use: float = 0.5,
                 use_seqscore: bool = True, collection: str = "paper-full",
                 max_new_tokens: int = 100, n_logprobs: int = 20):
        super().__init__(top_k=top_k, collection=collection)
        self._threshold = threshold
        self._w_rel = w_rel
        self._w_sup = w_sup
        self._w_use = w_use
        self._use_seqscore = use_seqscore
        self._max_new_tokens = max_new_tokens
        self._n_logprobs = n_logprobs
        self._base_url = os.environ.get("SELFRAG_URL", "http://127.0.0.1:8009/v1").rstrip("/")
        self._api_key = os.environ.get("SELFRAG_API_KEY", "EMPTY")
        self._model = os.environ.get("SELFRAG_MODEL", "")
        self._http = None

    async def __aenter__(self):
        await super().__aenter__()
        import httpx
        self._http = httpx.AsyncClient(timeout=120.0)
        if not self._model:
            self._model = await self._discover_model()
        return self

    async def __aexit__(self, *args):
        if self._http is not None:
            await self._http.aclose()
        await super().__aexit__(*args)

    async def _discover_model(self) -> str:
        r = await self._http.get(
            f"{self._base_url}/models",
            headers={"Authorization": f"Bearer {self._api_key}"},
        )
        r.raise_for_status()
        data = r.json().get("data") or []
        if not data:
            raise RuntimeError(f"no model served at {self._base_url}")
        return data[0]["id"]

    async def _complete(self, prompt: str, max_tokens: int | None = None):
        """One /v1/completions call; returns (text, tokens, top_logprobs).

        ``top_logprobs`` is a per-position list of {token_str: logprob}.
        """
        payload = {
            "model": self._model,
            "prompt": prompt,
            "temperature": 0.0,
            "top_p": 1.0,
            "max_tokens": max_tokens or self._max_new_tokens,
            "logprobs": self._n_logprobs,
        }
        r = await self._http.post(
            f"{self._base_url}/completions",
            json=payload,
            headers={"Authorization": f"Bearer {self._api_key}"},
        )
        r.raise_for_status()
        choice = r.json()["choices"][0]
        text = choice.get("text", "")
        lp = choice.get("logprobs") or {}
        tokens = lp.get("tokens") or []
        top = lp.get("top_logprobs") or []
        return text, tokens, top

    @staticmethod
    def _score_from_position(pos_logprobs: dict, names: list[str]) -> dict:
        """Map a {token: logprob} dict to {name: prob} over the given names."""
        import math
        out = {}
        for name in names:
            if name in pos_logprobs:
                out[name] = math.exp(float(pos_logprobs[name]))
        return out

    def _build_prompt(self, question: str, question_type: str,
                      options: dict | None) -> str:
        inst = SELFRAG_TASK_INST.get(question_type, "Answer the question.")
        q = question
        if options:
            opt_lines = "\n".join(f"{k}: {options[k]}" for k in sorted(options))
            q = f"{question}\n{opt_lines}"
        instruction = f"{inst}\n\n## Input:\n\n{q}"
        return f"### Instruction:\n{instruction}\n\n### Response:\n"

    async def generate(self, question, question_type, context=None, options=None):
        import math
        start = time.perf_counter()
        prompt = self._build_prompt(question, question_type, options)
        # Per-type answer-generation budget, mirroring the shared
        # get_max_tokens() used by the other baselines (yesno/mcq 4,
        # factoid 32, list 128, summary 256) plus headroom for Self-RAG's
        # interleaved reflection tokens, which the plain baselines do not emit.
        budget = {
            "yesno": 32, "mcq": 32, "mcq_multi": 48, "factoid": 64,
            "list": 192, "summary": 320, "expression": 96,
        }.get(question_type, 96)

        # --- Step 1: retrieval-reflection gate on the base prompt ----------
        # Only the first decoded token (the retrieval reflection) is needed
        # here, so keep this call short.
        try:
            _, _, base_top = await self._complete(prompt, max_tokens=8)
        except Exception as exc:
            return ModelResponse(answer="", response_text=f"selfrag error: {exc}",
                                 latency_ms=(time.perf_counter() - start) * 1000,
                                 metadata={"method": "self_rag_13b", "error": str(exc)})

        do_retrieve = True
        if base_top:
            ret = self._score_from_position(base_top[0], SELFRAG_RET_TOKENS)
            p_ret = ret.get("[Retrieval]", 0.0)
            p_no = ret.get("[No Retrieval]", 0.0)
            if (p_ret + p_no) > 0:
                do_retrieve = (p_ret / (p_ret + p_no)) > self._threshold

        # --- Step 2: no-retrieval path -------------------------------------
        if not do_retrieve:
            text, _, _ = await self._complete(prompt + "[No Retrieval]", max_tokens=budget)
            clean = selfrag_postprocess(text)
            answer = extract_answer(clean, question_type, options, mode="strict")
            return ModelResponse(
                answer=answer, response_text=clean,
                latency_ms=(time.perf_counter() - start) * 1000,
                metadata={"method": "self_rag_13b", "retrieved": False, "docs": 0},
            )

        # --- Step 3: retrieve + per-passage critic reranking ---------------
        docs, _ = await self._retrieve_shared_evidence(question)
        if not docs:
            text, _, _ = await self._complete(prompt + "[No Retrieval]", max_tokens=budget)
            clean = selfrag_postprocess(text)
            answer = extract_answer(clean, question_type, options, mode="strict")
            return ModelResponse(
                answer=answer, response_text=clean,
                latency_ms=(time.perf_counter() - start) * 1000,
                metadata={"method": "self_rag_13b", "retrieved": True, "docs": 0},
            )

        # Score each retrieved passage. The Self-RAG vLLM server handles
        # concurrent requests fine; only cross-ITEM concurrency in a single
        # event loop deadlocks here, so we parallelise WITHIN the item and run
        # the benchmark itself at concurrency=1 (one item / one retrieval at a
        # time). Cap the passage pool at 10 (Self-RAG short-form ndocs).
        async def _score_passage(idx, d):
            para = f"{d.title}\n{(d.abstract or '').strip()}"
            aug = prompt + f"[Retrieval]<paragraph>{para}</paragraph>"
            try:
                text, tokens, top = await self._complete(aug, max_tokens=budget)
            except Exception:
                return None
            if not top:
                return None
            rel = self._score_from_position(top[0], SELFRAG_REL_TOKENS)
            rel_sum = sum(rel.values())
            relevance = (rel.get("[Relevant]", 0.0) / rel_sum) if rel_sum > 0 else 0.0
            ground = 0.0
            for pos_lp in top:
                grd = self._score_from_position(pos_lp, SELFRAG_GRD_TOKENS)
                if len(grd) == len(SELFRAG_GRD_TOKENS):
                    g_sum = sum(grd.values())
                    if g_sum > 0:
                        ground = (grd["[Fully supported]"] / g_sum) + 0.5 * (
                            grd["[Partially supported]"] / g_sum)
                    break
            utility = 0.0
            for pos_lp in top:
                ut = self._score_from_position(pos_lp, SELFRAG_UT_TOKENS)
                if len(ut) == len(SELFRAG_UT_TOKENS):
                    u_sum = sum(ut.values())
                    if u_sum > 0:
                        weights = [-1, -0.5, 0, 0.5, 1]
                        utility = sum(
                            weights[i] * (ut[f"[Utility:{i+1}]"] / u_sum)
                            for i in range(5))
                    break
            score = self._w_rel * relevance + self._w_sup * ground + self._w_use * utility
            if self._use_seqscore and tokens:
                seq = 0.0
                n = 0
                for pos_lp, tok in zip(top, tokens):
                    if tok in pos_lp:
                        seq += float(pos_lp[tok])
                        n += 1
                if n:
                    score += math.exp(seq / n)
            return (idx, text, score)

        n_pass = min(len(docs), self._top_k, 10)
        scored = await asyncio.gather(
            *[_score_passage(idx, docs[idx]) for idx in range(n_pass)]
        )
        best_answer_text = ""
        best_score = -1e9
        best_idx = -1
        for r in scored:
            if r is None:
                continue
            idx, text, score = r
            if score > best_score:
                best_score = score
                best_answer_text = text
                best_idx = idx

        clean = selfrag_postprocess(best_answer_text)
        answer = extract_answer(clean, question_type, options, mode="strict")
        return ModelResponse(
            answer=answer, response_text=clean,
            latency_ms=(time.perf_counter() - start) * 1000,
            metadata={
                "method": "self_rag_13b",
                "retrieved": True,
                "docs": min(len(docs), self._top_k),
                "best_passage": best_idx,
                "best_score": round(best_score, 4),
            },
        )


# =============================================================================
# GeneGPT-lite (Jin et al. 2024) — non-agent tool-augmented variant
# =============================================================================

GENEGPT_EXTRACT_PROMPT = (
    "Extract gene or protein entity names from the question that would benefit "
    "from canonical-ID lookup via HGNC or UniProt. Output a comma-separated "
    "list of up to 5 entity names, in the original surface form. If no "
    "gene/protein entity is present, output NONE.\n\n"
    "Question: {question}\n\nEntities:"
)


class GeneGPTLiteClient(UnifiedEvidenceClient):
    """GeneGPT-lite (Jin et al. 2024) — non-agent tool-augmented variant.

    Original GeneGPT lets GPT-4 call NCBI E-utilities via Codex-generated
    API calls. We approximate it by:

      1. LLM extracts entity mentions (gene/protein/disease/drug).
      2. Resolve each via gene_resolver and uniprot_resolver to get
         canonical symbols + brief metadata (reusing V14's tool wrappers).
      3. Dense-retrieve PubMed abstracts with the original question.
      4. Prepend the resolved-entity block to the retrieved context and
         answer via the shared constrained-answer path.
    """

    def __init__(self, top_k: int = 20, collection: str = "paper-full",
                 max_entities: int = 5, resolver_timeout_s: float = 12.0,
                 resolver_max_concurrent: int = 4):
        super().__init__(top_k=top_k, collection=collection)
        self._max_entities = max_entities
        self._resolver_timeout_s = resolver_timeout_s
        self._resolver_max_concurrent = max(1, resolver_max_concurrent)

    async def _extract_entities(self, question: str) -> list[str]:
        from utils.clients import llm_client
        try:
            raw = await llm_client.chat(
                GENEGPT_EXTRACT_PROMPT.format(question=question),
                max_tokens=64, temperature=0.0,
            )
        except Exception:
            return []
        raw = (raw or "").strip()
        if not raw or raw.upper().startswith("NONE"):
            return []
        parts = [p.strip() for p in raw.replace("\n", ",").split(",")]
        out = []
        seen = set()
        for p in parts:
            if not p or p.upper() == "NONE":
                continue
            # trim punctuation and long tail
            p = p.strip(" .;:[]()\"'")
            if 1 <= len(p) <= 80 and p.lower() not in seen:
                seen.add(p.lower())
                out.append(p)
            if len(out) >= self._max_entities:
                break
        return out

    async def _resolve_entities(self, entities: list[str]) -> str:
        """Return markdown block of resolved entity metadata."""
        if not entities:
            return ""
        import asyncio
        from src.rlm.tools import _get_gene_resolver, _get_uniprot_resolver

        gene = _get_gene_resolver()
        uni = _get_uniprot_resolver()
        semaphore = asyncio.Semaphore(self._resolver_max_concurrent)

        async def lookup(name: str) -> str | None:
            async with semaphore:
                async def resolve_gene():
                    try:
                        return await asyncio.wait_for(
                            gene.resolve_full(name),
                            timeout=self._resolver_timeout_s,
                        )
                    except Exception:
                        return None

                async def resolve_uniprot():
                    try:
                        return await asyncio.wait_for(
                            uni.search_proteins(name, limit=1),
                            timeout=self._resolver_timeout_s,
                        )
                    except Exception:
                        return []

                g, hits = await asyncio.gather(resolve_gene(), resolve_uniprot())
                lines = []
                if g and g.get("symbol"):
                    parts = [f"Gene: {g['symbol']}"]
                    if g.get("name"):
                        parts.append(g["name"])
                    if g.get("location"):
                        parts.append(f"location {g['location']}")
                    lines.append(" — ".join(parts))
                if hits:
                    h = hits[0]
                    desc = h.protein_name or ""
                    lines.append(f"UniProt {h.accession}" + (f": {desc[:150]}" if desc else ""))
                if lines:
                    return f"**{name}**: " + "; ".join(lines)
                return None

        results = await asyncio.gather(*(lookup(e) for e in entities))
        lines = [r for r in results if r]
        if not lines:
            return ""
        return "## Resolved biomedical entities\n" + "\n".join(lines) + "\n\n"

    async def generate(self, question, question_type, context=None, options=None):
        start = time.perf_counter()

        # Stage 1+2: extract & resolve entities (background) + Stage 3 retrieve in parallel
        import asyncio
        entities_task = asyncio.create_task(self._extract_entities(question))
        retrieve_task = asyncio.create_task(self._retrieve_shared_evidence(question))
        entities = await entities_task
        entity_block = await self._resolve_entities(entities) if entities else ""
        docs, _ = await retrieve_task

        if not docs and not entity_block:
            return ModelResponse(answer="", response_text="No evidence found.",
                                latency_ms=(time.perf_counter() - start) * 1000)

        retrieved_ctx, _, _ = build_dense_context(docs[: self._top_k], max_context_tokens=3500)
        merged_context = (entity_block + retrieved_ctx).strip() or retrieved_ctx

        answer, answer_text = await generate_constrained_answer(
            question=question, question_type=question_type,
            context=merged_context, options=options,
        )
        return ModelResponse(
            answer=answer, response_text=answer_text,
            latency_ms=(time.perf_counter() - start) * 1000,
            metadata={
                "method": "genegpt_lite",
                "entities": entities,
                "entity_chars": len(entity_block),
                "docs": len(docs),
            },
        )


# =============================================================================
# GeneGPT (Jin et al. 2024) — FAITHFUL: LLM-generated NCBI E-utilities API
# call chains, executed live, with the result fed back (ReAct-style loop).
# Backbone is the shared LLM (the GPT-4 -> {model} swap); the defining
# mechanism — the model *writing* esearch/esummary/efetch/BLAST calls and
# chaining them — is preserved.
# =============================================================================

NCBI_EUTILS = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
NCBI_BLAST = "https://blast.ncbi.nlm.nih.gov/Blast.cgi"

# Faithful reproduction of the GeneGPT prompt (ncbi/GeneGPT main.py
# get_prompt_header with full mask 111111): system line + Eutils doc +
# BLAST doc + the four cross-task demonstrations (gene symbol, SNP->gene,
# disease->gene, BLAST alignment), using the paper's exact "[url]->[result]"
# call format. Demo result snippets are abbreviated real returns (kept short
# so the in-context prompt does not blow the context budget).
GENEGPT_SYSTEM = (
    "Hello. Your task is to use NCBI Web APIs to answer genomic questions.\n"
    "You can call Eutils by: "
    "\"[https://eutils.ncbi.nlm.nih.gov/entrez/eutils/{esearch|efetch|esummary}.fcgi"
    "?db={gene|snp|omim}&retmax={}&{term|id}={term|id}]\".\n"
    "esearch: input is a search term and output is database id(s).\n"
    "efetch/esummary: input is database id(s) and output is full records or "
    "summaries that contain name, chromosome location, and other information.\n"
    "Normally, you need to first call esearch to get the database id(s) of the "
    "search term, and then call efetch/esummary to get the information with the "
    "database id(s).\n"
    "Database: gene is for genes, snp is for SNPs, and omim is for genetic diseases.\n\n"
    "For DNA sequences, you can use BLAST by: "
    "\"[https://blast.ncbi.nlm.nih.gov/blast/Blast.cgi?CMD={Put|Get}&PROGRAM=blastn"
    "&MEGABLAST=on&DATABASE=nt&FORMAT_TYPE={XML|Text}&QUERY={sequence}"
    "&HITLIST_SIZE={max_hit_size}]\".\n"
    "BLAST maps a specific DNA {sequence} to its chromosome location among "
    "different species.\n"
    "You need to first PUT the BLAST request and then GET the results using the "
    "RID returned by PUT.\n\n"
    "PROTOCOL: respond with EXACTLY ONE line each turn.\n"
    "  To call an API:  [<full https URL>]\n"
    "  When you can answer:  Answer: <concise final answer, no explanation>\n"
    "Issue one API call, read the result I return inside ->[...], then continue. "
    "Do not invent results. Keep terms URL-safe (use + for spaces)."
)

GENEGPT_DEMO = (
    "Here are some examples:\n\n"
    # Example 1 — gene symbol (esearch gene -> efetch)
    "Question: What is the official gene symbol of LMP10?\n"
    f"[{NCBI_EUTILS}/esearch.fcgi?db=gene&retmax=5&retmode=json&sort=relevance&term=LMP10]"
    "->[{\"esearchresult\":{\"idlist\":[\"19171\",\"5699\",\"8138\"]}}]\n"
    f"[{NCBI_EUTILS}/efetch.fcgi?db=gene&retmax=5&retmode=json&id=19171,5699,8138]"
    "->[id 5699: Official Symbol PSMB10 and Name: proteasome 20S subunit beta 10 (human)]\n"
    "Answer: PSMB10\n\n"
    # Example 2 — SNP -> gene (esummary snp)
    "Question: Which gene is SNP rs1217074595 associated with?\n"
    f"[{NCBI_EUTILS}/esummary.fcgi?db=snp&retmax=10&retmode=json&id=1217074595]"
    "->[{\"result\":{\"1217074595\":{\"genes\":[{\"name\":\"LINC01270\",\"gene_id\":\"284751\"}]}}}]\n"
    "Answer: LINC01270\n\n"
    # Example 3 — disease -> genes (esearch omim -> esummary omim)
    "Question: What are genes related to Meesmann corneal dystrophy?\n"
    f"[{NCBI_EUTILS}/esearch.fcgi?db=omim&retmax=20&retmode=json&sort=relevance&term=Meesmann+corneal+dystrophy]"
    "->[{\"esearchresult\":{\"idlist\":[\"618767\",\"601687\",\"300778\",\"148043\",\"122100\"]}}]\n"
    f"[{NCBI_EUTILS}/esummary.fcgi?db=omim&retmax=20&retmode=json&id=618767,601687,300778,148043,122100]"
    "->[records list gene maps KRT12 and KRT3 for Meesmann corneal dystrophy]\n"
    "Answer: KRT12, KRT3\n\n"
    # Example 4 — BLAST sequence alignment (PUT -> GET via RID)
    "Question: Align the DNA sequence to the human genome:"
    "ATTCTGCCTTTAGTAATTTGATGACAGAGACTTCTTGGGAACCACAGCCAGGGAGCCACCCTTTACTCCACCAACAGGTGGCTTATATCCAATCTGAGAAAGAAAGAAAAAAAAAAAAGTATTTCTCT\n"
    f"[{NCBI_BLAST}?CMD=Put&PROGRAM=blastn&MEGABLAST=on&DATABASE=nt&FORMAT_TYPE=XML&QUERY=ATTCTGCC...TCT&HITLIST_SIZE=5]"
    "->[RID = ABCDEFG123]\n"
    f"[{NCBI_BLAST}?CMD=Get&FORMAT_TYPE=Text&RID=ABCDEFG123]"
    "->[Homo sapiens chromosome 15, alignment at 91950805-91950932]\n"
    "Answer: chr15:91950805-91950932\n\n"
)


class GeneGPTClient(UnifiedEvidenceClient):
    """Faithful GeneGPT (Jin et al. 2024) with the shared LLM backbone.

    The model generates real NCBI E-utilities / BLAST URLs, which are executed
    live; raw responses are appended back into the dialog so the model can
    chain calls (esearch -> esummary -> efetch ...) and reason over actual
    database records — the mechanism that defines GeneGPT. Only the backbone
    differs from the paper (GPT-4 -> {model}); the API-calling loop is intact.

    Retrieval base class is reused only for interface parity; GeneGPT does not
    use dense PubMed retrieval (it pulls structured data via NCBI APIs).
    """

    def __init__(self, top_k: int = 20, collection: str = "paper-full",
                 max_calls: int = 10, http_timeout_s: float = 40.0,
                 max_result_chars: int = 10000, cut_length: int = 18000,
                 format_enhanced: bool = False):
        super().__init__(top_k=top_k, collection=collection)
        # format_enhanced: keep the faithful API-chaining loop UNCHANGED, but when
        # the instruct/thinking backbone leaves the answer "off-format" (dialog
        # filler / verbose prose with no terse answer), run one extra forced
        # terse-extraction call over the accumulated REAL API results. This
        # adapts GeneGPT's completion-style decoding to a chat model; it does not
        # add any new evidence (only re-reads results the model already fetched).
        self._fmt = format_enhanced
        self._max_calls = max_calls            # paper: 10
        self._http_timeout_s = http_timeout_s
        self._max_result_chars = max_result_chars  # paper: 10000
        self._cut_length = cut_length          # paper: 18000 (truncate from start)
        _cfg = get_config()
        self._llm_cfg = _cfg.llm
        self._llm_key = _cfg.llm.api_key
        self._llm_model = _cfg.llm.model
        self._http = None      # NCBI API client
        self._llm = None       # LLM client (round-robin over the Qwen endpoints)

    def _next_llm_url(self) -> str:
        return self._llm_cfg.get_server()  # round-robin (8004/8005)

    async def __aenter__(self):
        await super().__aenter__()
        import httpx
        self._http = httpx.AsyncClient(
            timeout=self._http_timeout_s,
            headers={"User-Agent": "genegpt-benchmark/1.0 (mailto:research@example.org)"},
            follow_redirects=True,
        )
        self._llm = httpx.AsyncClient(timeout=120.0)
        return self

    async def __aexit__(self, *args):
        if self._http is not None:
            await self._http.aclose()
        if self._llm is not None:
            await self._llm.aclose()
        await super().__aexit__(*args)

    async def _complete_stop(self, prompt: str) -> str:
        """Single completion that STOPS at '->' or a new Question, faithfully
        reproducing GeneGPT's augmented decoding: the model writes one
        ``[url]`` and is cut at ``->`` so it never fabricates the API result.

        Uses chat/completions with thinking disabled (Qwen-3) but a single
        accumulating user message to mimic the paper's prompt-continuation.
        """
        base = self._next_llm_url()
        payload = {
            "model": self._llm_model,
            "messages": [
                {"role": "system", "content": GENEGPT_SYSTEM},
                {"role": "user", "content": prompt},
            ],
            "max_tokens": 512, "temperature": 0.0,
            "stop": ["->", "\n\nQuestion"],
            "chat_template_kwargs": {"enable_thinking": False},
        }
        r = await self._llm.post(
            f"{base.rstrip('/')}/chat/completions", json=payload,
            headers={"Authorization": f"Bearer {self._llm_key}"},
        )
        r.raise_for_status()
        return ((r.json()["choices"][0].get("message") or {}).get("content", "") or "")

    @staticmethod
    def _find_directive(text: str) -> tuple[str, str]:
        """Return (kind, payload): kind in {'call','answer',''}.

        Matches the GeneGPT prompt format: an API call is written as
        ``[<https url>]`` (optionally ``[url]->...`` if the model echoes a
        result), and the final answer as ``Answer: ...``. Falls back to a
        bare eutils/blast URL or a legacy ``CALL:`` line.
        """
        import re
        # answer line takes priority only if no pending call precedes it
        # within the same chunk; scan line by line.
        for line in text.splitlines():
            s = line.strip()
            # API call in bracket form: [https://eutils...]  (stop at ] or ->)
            m = re.search(r"\[\s*(https?://(?:eutils|blast)\.ncbi\.nlm\.nih\.gov\S*?)\s*\]", s)
            if m:
                return "call", m.group(1).strip().strip('`<>"')
            m = re.match(r"(?i)^answer:\s*(.+)", s)
            if m:
                return "answer", m.group(1).strip()
            m = re.match(r"(?i)^CALL:\s*(\S+)", s)  # legacy
            if m:
                return "call", m.group(1).strip().strip('`<>"')
        # fallback: a bare eutils/blast URL anywhere
        m = re.search(r"https?://(?:eutils|blast)\.ncbi\.nlm\.nih\.gov\S+", text)
        if m:
            return "call", m.group(0).strip().strip('`<>"[]')
        return "", ""

    async def _exec_call(self, url: str) -> str:
        import asyncio
        if not url.lower().startswith(("https://eutils.ncbi.nlm.nih.gov",
                                       "https://blast.ncbi.nlm.nih.gov",
                                       "http://eutils.ncbi.nlm.nih.gov")):
            return f"(rejected: only NCBI E-utilities/BLAST URLs are allowed; got {url[:80]})"
        try:
            r = await self._http.get(url)
            await asyncio.sleep(0.34)  # NCBI courtesy limit ~3 req/s
            txt = r.text
        except Exception as exc:  # noqa: BLE001
            return f"(API error: {exc})"
        if len(txt) > self._max_result_chars:
            txt = txt[: self._max_result_chars] + " ...[truncated]"
        return txt

    # --- format-enhanced variant only (genegpt-fmt) ---------------------------
    @staticmethod
    def _needs_rescue(final_raw: str, final: str) -> bool:
        """True if the model ended off-format: dialog filler, demo-question echo,
        empty, or verbose prose with no explicit 'Answer:' marker."""
        import re
        f = (final or "").strip().lower()
        if not f:
            return True
        FILLER = [
            r"\bunderstood\b", r"provide (your |the )?(genomic )?question",
            r"how (can|may) i (help|assist)", r"please provide",
            r"what is the (official gene symbol of lmp10|next question)",
            r"i'?m (ready|here) to (help|assist)",
        ]
        if any(re.search(p, f) for p in FILLER):
            return True
        # `final` is already past 'Answer:' extraction; it should now be a TERSE
        # answer. Prose (long, or >=5 words) means the backbone emitted a
        # sentence ("NAXE is a protein-coding gene") instead of the canonical
        # token ("TRUE") — even if it carried an 'Answer:' marker — so terse
        # extraction may recover it. Concise correct answers (gene symbol,
        # coordinates, TRUE/FALSE, organism name, short gene list) are <=4 words
        # and never fire here, so already-correct rows are untouched.
        if len(f) > 100 or len(f.split()) >= 5:
            return True
        return False

    async def _force_answer(self, accumulated: str, q: str,
                            question_type: str, options=None,
                            prior: str = "") -> str:
        """One extra call that turns the model's own draft + the real API results
        already in ``accumulated`` into a terse, benchmark-formatted answer. No
        new evidence is fetched. It first tries to NORMALIZE the draft conclusion
        (so a correct-but-verbose answer like "X is a protein-coding gene" -> TRUE
        is not re-litigated); only if the draft is empty/filler does it derive the
        answer from the API results. Mirrors what a completion-style backbone (the
        paper's code-davinci/GPT-4) would emit as ``Answer: <terse>``."""
        FMT_SYSTEM = (
            "You extract the final answer to a GeneTuring question from real NCBI "
            "API results that are already provided. Output ONLY the answer — no "
            "explanation, no restating the question, no 'Answer:' prefix, no markdown. "
            "Formatting rules:\n"
            "- Gene symbol/name: only the official symbol(s), e.g. PSMB10 (comma-separate multiple).\n"
            "- Is-it-protein-coding questions: only TRUE or FALSE.\n"
            "- Organism questions: only the common name, e.g. yeast, mouse, rat, human, zebrafish, fly.\n"
            "- Genomic location/alignment: only chrN:start-end (e.g. chr15:91950805-91950932).\n"
            "- Yes/no questions: only yes or no."
        )
        base = self._next_llm_url()
        draft = (prior or "").strip()
        user = (
            f"{accumulated}\nQuestion: {q}\n"
            f'A draft answer was: "{draft[:300]}".\n'
            "If the draft already states the answer, reformat it to the required terse form "
            "(do NOT change its meaning — e.g. 'X is a protein-coding gene' -> TRUE). "
            "If the draft is empty or just filler (e.g. 'Understood', 'please provide'), "
            "derive the answer from the real API results above. "
            "Output ONLY the final answer, nothing else.\n"
            "Answer:"
        )
        payload = {
            "model": self._llm_model,
            "messages": [
                {"role": "system", "content": FMT_SYSTEM},
                {"role": "user", "content": user},
            ],
            "max_tokens": 48, "temperature": 0.0,
            "stop": ["\n\n", "\nQuestion", "->"],
            "chat_template_kwargs": {"enable_thinking": False},
        }
        try:
            r = await self._llm.post(
                f"{base.rstrip('/')}/chat/completions", json=payload,
                headers={"Authorization": f"Bearer {self._llm_key}"},
            )
            r.raise_for_status()
            out = ((r.json()["choices"][0].get("message") or {}).get("content", "") or "")
        except Exception as exc:  # noqa: BLE001
            LOGGER.warning("genegpt-fmt force_answer failed: %s", exc)
            return ""
        import re
        out = re.sub(r"(?is)<think>.*?</think>", "", out).strip()
        out = re.sub(r"(?i)^answer:\s*", "", out).strip().strip("*` ")
        return out

    async def generate(self, question, question_type, context=None, options=None):
        """Faithful reproduction of GeneGPT main.py's augmented-decoding loop.

        Single accumulating prompt; generate STOPS at '->'; a ``[url]`` is
        executed against NCBI and the *real* result is appended as
        ``{text}->[{call}]``; the loop ends when the model emits no further
        URL (its text is then the final answer). Mirrors the paper's control
        flow incl. the 18k-char left-truncation, 10-call cap, BLAST PUT/GET,
        and 10k-char result cap.
        """
        import re
        import asyncio
        start = time.perf_counter()

        q = question
        if options:
            q = question + "\nOptions:\n" + "\n".join(
                f"{k}. {options[k]}" for k in sorted(options))
        q_prompt = f"{GENEGPT_DEMO}Question: {q}\n"

        url_regex = r"\[(https?://[^\[\]]+)\]"
        calls: list[str] = []
        final = ""
        num_calls = 0
        while True:
            if len(q_prompt) > self._cut_length:
                q_prompt = q_prompt[len(q_prompt) - self._cut_length:]
            try:
                text = await self._complete_stop(q_prompt)
            except Exception as exc:  # noqa: BLE001
                LOGGER.warning("genegpt llm failed: %s", exc)
                final = ""
                break
            num_calls += 1
            m = re.findall(url_regex, text)
            if m:
                url = m[0]
                # BLAST GET: wait for the job; PUT: extract the RID
                if "blast" in url.lower() and "Get" in url:
                    await asyncio.sleep(30)
                call = await self._exec_call(url)
                if "blast" in url.lower() and "Put" in url:
                    rid = re.search(r"RID = (.*)\n", call)
                    call = rid.group(1) if rid else call
                if len(call) > self._max_result_chars:
                    call = call[: self._max_result_chars]
                calls.append(url)
                # append the model's text (up to the [url]) + the real result,
                # closing the '->' the stop-token cut off
                q_prompt = f"{q_prompt}{text}->[{call}]\n"
            else:
                final = text.strip()
                break
            if num_calls >= self._max_calls:
                final = text.strip()
                break

        # The model emits its conclusion as "Answer: <X>" (per the demos); take
        # the text after the last such marker so the question echo / reasoning
        # preamble does not leak into the parsed answer.
        final_raw = final
        if final:
            mm = re.findall(r"(?im)^answer:\s*(.+)$", final)
            if mm:
                final = mm[-1].strip()

        # format-enhanced variant: rescue off-format endings with one terse
        # extraction over the already-fetched API results (no new evidence).
        rescued = False
        if self._fmt and self._needs_rescue(final_raw, final):
            forced = await self._force_answer(q_prompt, q, question_type, options,
                                              prior=final)
            if forced:
                final = forced
                rescued = True

        answer = extract_answer(final, question_type, options, mode="strict")
        return ModelResponse(
            answer=answer, response_text=final.strip(),
            latency_ms=(time.perf_counter() - start) * 1000,
            metadata={
                "method": "genegpt-fmt" if self._fmt else "genegpt",
                "api_calls": calls,
                "n_calls": len(calls),
                "rescued": rescued,
            },
        )


# =============================================================================
# BiomedRAG-chunk (Li et al. 2024) — zero-training chunk-level adaptation
# =============================================================================

BIOMEDRAG_INSTRUCTION_TEMPLATE = (
    "Below is an instruction that describes a biomedical task, paired with an "
    "input that provides further context. Write a response that appropriately "
    "completes the request.\n\n"
    "### Instruction:\n{question}\n\n"
    "### Input:\n{context}\n\n"
    "### Response:\n"
)


def _sentence_chunks(text: str, max_len: int = 280, min_len: int = 30) -> list[str]:
    """Split text into sentence-level chunks."""
    import re
    sents = re.split(r"(?<=[.!?])\s+", (text or "").strip())
    out = []
    for s in sents:
        s = s.strip()
        if len(s) < min_len:
            # merge short fragments with previous chunk
            if out and len(out[-1]) + len(s) + 1 <= max_len:
                out[-1] = (out[-1] + " " + s).strip()
            elif s:
                out.append(s)
            continue
        if len(s) <= max_len:
            out.append(s)
        else:
            # long sentence — hard-cut
            for i in range(0, len(s), max_len):
                out.append(s[i : i + max_len])
    return [c for c in out if len(c) >= min_len // 2]


class BiomedRAGClient(UnifiedEvidenceClient):
    """BiomedRAG-chunk (Li et al. 2024) — zero-training chunk-level adaptation.

    Original BiomedRAG trains a BERT chunk scorer + LoRA-tunes Llama2-13B
    on biomedical instructions. We cannot retrain, so we approximate the
    key ideas zero-shot:

      1. Dense-retrieve top-K abstracts.
      2. Split each abstract into sentence chunks (~280 chars).
      3. Score all chunks with Qwen3-Reranker (the "tailored chunk scorer").
      4. Keep top-N chunks (finer granularity than abstracts).
      5. Answer using BiomedRAG's Alpaca-style instruction template.
    """

    def __init__(self, top_k: int = 20, chunk_top_k: int = 25,
                 collection: str = "paper-full", max_chunks: int = 256):
        super().__init__(top_k=top_k, collection=collection)
        self._chunk_top_k = chunk_top_k
        self._max_chunks = max(chunk_top_k, max_chunks)

    async def generate(self, question, question_type, context=None, options=None):
        from utils.clients import rerank_client

        start = time.perf_counter()
        docs, _ = await self._retrieve_shared_evidence(question)
        if not docs:
            return ModelResponse(answer="", response_text="No evidence found.",
                                latency_ms=(time.perf_counter() - start) * 1000)

        # Build chunks, remembering their source doc rank for attribution.
        chunks: list[tuple[str, str, int]] = []  # (text, pmid, src_rank)
        for d in docs[: self._top_k]:
            full = f"{d.title}. {d.abstract or ''}".strip()
            for c in _sentence_chunks(full, max_len=280):
                chunks.append((c, d.pmid, d.rank))
                if len(chunks) >= self._max_chunks:
                    break
            if len(chunks) >= self._max_chunks:
                break
        if not chunks:
            # fallback to abstract-level
            full_context, _, _ = build_dense_context(docs, max_context_tokens=4000)
            answer, answer_text = await generate_constrained_answer(
                question=question, question_type=question_type,
                context=full_context, options=options,
            )
            return ModelResponse(
                answer=answer, response_text=answer_text,
                latency_ms=(time.perf_counter() - start) * 1000,
                metadata={"method": "biomedrag", "chunks": 0, "fallback": True},
            )

        passages = [c[0] for c in chunks]
        try:
            scored = await rerank_client.rerank(
                question, passages,
                top_k=self._chunk_top_k, max_concurrent=8,
            )
        except Exception:
            scored = None

        selected = []
        if scored:
            seen_idx = set()
            for idx, _ in scored:
                if 0 <= idx < len(chunks) and idx not in seen_idx:
                    selected.append(chunks[idx])
                    seen_idx.add(idx)
        if not selected:
            selected = chunks[: self._chunk_top_k]

        # Alpaca-style instruction template (BiomedRAG inference format).
        ctx_lines = []
        for i, (text, pmid, rank) in enumerate(selected, 1):
            ctx_lines.append(f"[{i}] PMID:{pmid} (rank {rank}): {text}")
        chunk_block = BIOMEDRAG_INSTRUCTION_TEMPLATE.format(
            question=question,
            context="\n".join(ctx_lines),
        )

        answer, answer_text = await generate_constrained_answer(
            question=question, question_type=question_type,
            context=chunk_block, options=options,
        )
        return ModelResponse(
            answer=answer, response_text=answer_text,
            latency_ms=(time.perf_counter() - start) * 1000,
            metadata={
                "method": "biomedrag",
                "docs": len(docs),
                "chunks_total": len(chunks),
                "chunks_kept": len(selected),
                "chunk_cap": self._max_chunks,
            },
        )


class DualRetrievalDenseClient(UnifiedEvidenceClient):
    """Dual retrieval + Dense RAG answer generation.

    Searches for BOTH supporting and refuting evidence, merges top-k,
    then uses the same generate_constrained_answer as Dense RAG.
    No NLI, no RLM — just broader evidence + direct LLM judgment.
    """

    async def generate(self, question, question_type, context=None, options=None):
        start = time.perf_counter()

        # Positive retrieval (standard)
        pos_docs, _ = await self._retrieve_shared_evidence(question)

        # Negative retrieval (for yesno: add negation terms)
        if question_type == "yesno":
            neg_terms = "no effect OR not effective OR no association OR failed OR no benefit OR no difference"
            neg_query = f"{question} {neg_terms}"
            neg_docs, _ = await retrieve_dense_evidence(
                neg_query,
                qdrant=self._qdrant, pool=self._pool,
                top_k=self._top_k, collection=self._collection,
            )
            # Merge and deduplicate, keep top-k by score
            seen = {d.pmid for d in pos_docs}
            for d in neg_docs:
                if d.pmid not in seen:
                    seen.add(d.pmid)
                    pos_docs.append(d)
            pos_docs.sort(key=lambda d: -d.score)
            pos_docs = pos_docs[:self._top_k]
        docs = pos_docs

        if not docs:
            return ModelResponse(answer="", response_text="No evidence found.",
                                latency_ms=(time.perf_counter() - start) * 1000)

        full_context, _, _ = build_dense_context(docs, max_context_tokens=4000)
        answer, answer_text = await generate_constrained_answer(
            question=question, question_type=question_type,
            context=full_context, options=options,
        )
        return ModelResponse(
            answer=answer, response_text=answer_text,
            latency_ms=(time.perf_counter() - start) * 1000,
            metadata={"method": "dual_retrieval_dense", "docs": len(docs)},
        )


# Upper bound for one agent escalation (seconds); see the wait_for in V14CascadeClient.generate.
logger = logging.getLogger(__name__)
_AGENT_TIMEOUT = float(os.environ.get("XC_AGENT_TIMEOUT", "900"))

# Escalate when stage 1 answered "insufficient information": that answer names its
# own remedy (more evidence), and the escalated agent can re-retrieve. Matched on
# the option TEXT so the rule stays dataset-independent.
_ESCALATE_ON_ABSTAIN = os.environ.get("XC_ESCALATE_ON_ABSTAIN") == "1"

# Shallow half of the agent-output fix: hand the rejudge the agent's full final turn
# rather than the single letter extract_answer() distilled from it.
_REJUDGE_AGENT_TEXT = os.environ.get("XC_REJUDGE_AGENT_TEXT") == "1"
_REJUDGE_AGENT_TEXT_CHARS = int(os.environ.get("XC_REJUDGE_AGENT_TEXT_CHARS", "6000"))

# Deep half: give the rejudge the text every REPL block printed during the agent run
# (PipelineResult.repl_stdout), i.e. the evidence the agent actually fetched.
_REJUDGE_AGENT_EVIDENCE = os.environ.get("XC_REJUDGE_AGENT_EVIDENCE") == "1"
_REJUDGE_AGENT_EVIDENCE_CHARS = int(os.environ.get("XC_REJUDGE_AGENT_EVIDENCE_CHARS", "12000"))

# When the pipeline still lands on "insufficient information", re-ask with no
# evidence rather than scoring a guaranteed zero (LitQA2: no-context answers 31%
# of those same items correctly).
_ANSWER_NO_CTX_ON_ABSTAIN = os.environ.get("XC_ANSWER_WITHOUT_CONTEXT_ON_ABSTAIN") == "1"
_ABSTAIN_RE = re.compile(
    r"insufficient information|not enough information|cannot be determined"
    r"|unable to determine|none of the above|no answer",
    re.IGNORECASE,
)


class V14CascadeClient(UnifiedEvidenceClient):
    """Adaptive cascade: constrained gen first, escalate to agent if low confidence.

    Stage 1: Dense retrieval + constrained generation (fast, like dual-dense)
             → get answer + logprob confidence
    Stage 2: If confidence < threshold → V14 agent deep reasoning (slow but thorough)

    For yesno: uses dual retrieval (pos + neg evidence).
    For other types: uses standard retrieval.

    This gives dual-dense quality on easy questions and V14 depth on hard ones.
    """

    _GENE_RE = re.compile(r'\b[A-Z][A-Z0-9]{2,}[a-z0-9]*\b')
    _LOC_RE = re.compile(r'\bLOC\d+\b')
    _ACCESSION_RE = re.compile(r'\b(NM_|NP_|NR_|NG_|GCF_)[\d.]+\b')
    _EN_CLINICAL_RE = re.compile(
        r'\b(year|month|week|day)-old\b|newborn|pediatrician|emergency department'
        r'|brought to the|presents with|comes to the', re.IGNORECASE)
    _COMMON_NON_GENE = frozenset({
        'DNA','RNA','PCR','COVID','MRI','CT','EEG','ECG','NMR','FDA','AIDS','HIV',
        'BMI','BP','PhD','MD','US','EU','WHO','ICU','ER','CNS','PNS','GI','GU',
        'MRSA','TSH','LDL','HDL','BUN','CBC','CSF','IV','IM','PO',
    })

    @classmethod
    def _should_rewrite(cls, question: str, question_type: str) -> bool:
        ql = question.lower()
        if any(p in ql for p in ['chromosome location of', 'subcellular localization of',
                                  'biological function of', 'genomic location of']):
            return True
        if any(p in ql for p in ['gene symbol of', 'official symbol of',
                                  'accession', 'identifier', 'alias of']):
            return False
        if cls._EN_CLINICAL_RE.search(question):
            return False
        if cls._LOC_RE.search(question) or cls._ACCESSION_RE.search(question):
            return False
        genes = [g for g in cls._GENE_RE.findall(question) if g not in cls._COMMON_NON_GENE]
        if genes and question_type in ('factoid',):
            return False
        return True

    REWRITE_TYPES = frozenset({'yesno', 'summary', 'list'})
    # Skip HyDE-based methods for question types where the answer requires
    # entity-database lookup (e.g., geneturing factoid asks "official symbol of X").
    # Rerank can't save us when relevant docs simply don't exist in PubMed.
    SKIP_HYDE_TYPES = frozenset({'factoid'})
    # Types where Stage-1-answer-in-retrieved-docs is a reliable signal of
    # non-hallucination. Pilot validated: geneturing/factoid grounded rate
    # 63% (acc 28.6 grounded vs 8.1 ungrounded); bioasq/factoid grounded 99%.
    GROUNDED_CHECK_TYPES = frozenset({'factoid', 'list'})

    @staticmethod
    def _is_answer_grounded(answer: str, docs) -> bool:
        """True if the Stage-1 answer can be located in retrieved documents.
        Ungrounded answers on factoid/list are treated as likely hallucinations.
        """
        ans = (answer or "").lower().strip()
        if not ans or len(ans) <= 3:
            return True
        if ans.upper() in {"A", "B", "C", "D", "E"} and len(ans) == 1:
            return True
        doc_text = " ".join(
            ((d.abstract or "") + " " + (d.title or "")).lower() for d in docs
        )
        if ans in doc_text:
            return True
        tokens = [t for t in ans.split() if len(t) > 2]
        if not tokens:
            return True
        hits = sum(1 for t in tokens if t in doc_text)
        return hits / len(tokens) >= 0.5

    def __init__(self, top_k=20, confidence_threshold=0.7, version="v14",
                 rewrite_query=False, selective_rewrite=False, type_rewrite=False,
                 dual_rerank=False, grounded_gate=False, enable_tools=True,
                 enable_disco=False, disco_url=None, router_entities=None):
        super().__init__(top_k=top_k)
        self._threshold = confidence_threshold
        self._version = version
        self._rewrite_query = rewrite_query
        self._selective_rewrite = selective_rewrite
        self._type_rewrite = type_rewrite
        self._dual_rerank = dual_rerank
        self._grounded_gate = grounded_gate
        self._enable_tools = enable_tools
        self._enable_disco = enable_disco
        # Prefer DISCO_SERVER_URL; SCDATA_SERVER_URL kept for backward compat.
        self._disco_url = (
            disco_url
            or os.getenv('DISCO_SERVER_URL')
            or os.getenv('SCDATA_SERVER_URL')
            or 'http://127.0.0.1:8443'
        )
        # Router-extracted entities keyed by (dataset, item_id) → {extracted_genes,
        # extracted_tissues, extracted_celltypes, route_disco}.
        self._router_entities: dict[tuple[str, str], dict] = router_entities or {}
        # Per-instance atlas-fetch counters for telemetry.
        self._atlas_calls_ok = 0
        self._atlas_calls_fail = 0
        self._pipeline = None
        # Lazy OpenAI client for logprobs
        self._logprob_client = None

    # Keywords triggering disco lookup. Conservative — must look like an scRNA
    # / cell-type / expression / marker question.
    _DISCO_KEYWORDS = (
        'cell type', 'celltype', 'cell-type', 'marker gene', 'markers of',
        'specific marker', 'top marker', 'expression in', 'expressed in',
        'expression of', 'expression pattern', 'expression profile',
        'tissue expression', 'expression level',
        'scRNA', 'single cell', 'single-cell',
        'pseudobulk', 'cell composition', 'cell fraction', 'cell proportion',
        'differentially expressed', 'DE gene', 'cluster marker',
    )
    _DISCO_TISSUE_HINTS = (
        'lung', 'blood', 'brain', 'kidney', 'heart', 'liver', 'pancreas',
        'intestine', 'thymus', 'tonsil', 'bone marrow', 'adipose', 'breast',
        'ovary', 'testis', 'placenta', 'skeletal muscle', 'stomach',
    )

    # HPA → SciHorizon GT canonical tissue vocabulary alignment.
    # Without this, atlas returns "adipose tissue"/"adrenal gland"/"amygdala"
    # which never string-matches GT ["fat", "adrenal", "brain"].
    # Brain regions aggregate to "brain" (max NX). Heart muscle → heart, etc.
    _TISSUE_NORMALIZE = {
        'adrenal gland': 'adrenal',
        'adipose tissue': 'fat',
        'gallbladder': 'gall bladder',
        'heart muscle': 'heart',
        'thyroid gland': 'thyroid',
        'amygdala': 'brain',
        'basal ganglia': 'brain',
        'cerebellum': 'brain',
        'cerebral cortex': 'brain',
        'choroid plexus': 'brain',
        'hippocampal formation': 'brain',
        'hypothalamus': 'brain',
        'midbrain': 'brain',
        'spinal cord': 'brain',
        # Disco lab-style names → standard
        'ad frontal cortex parenchyma': 'brain',
        'pdac pancreas': 'pancreas',
        'type 1 diabetes pancreas': 'pancreas',
        'type 2 diabetes pancreas': 'pancreas',
        'pancreas cell': 'pancreas',
        'liver cell': 'liver',
        'liver nucleus': 'liver',
        'hiv blood': 'blood',
        'hnscc blood': 'blood',
        'sarcoidosis blood': 'blood',
        'adipose cell': 'fat',
        'adipose nucleus': 'fat',
        'basophil mast cell': 'blood',
        'hiv cerebrospinal fluid': 'brain',
    }

    @classmethod
    def _normalize_atlas_tissues(cls, entries: list, value_key: str = 'nx') -> list:
        """Map raw HPA/Disco tissue names to benchmark-aligned canonical names.
        Aggregates duplicates by max value so multiple brain sub-regions collapse
        to a single 'brain' entry with the highest NX.
        """
        agg: dict[str, dict] = {}
        for e in entries:
            raw = (e.get('tissue') or '').lower().strip()
            canon = cls._TISSUE_NORMALIZE.get(raw, raw)
            cur = agg.get(canon)
            if cur is None or e.get(value_key, 0) > cur.get(value_key, 0):
                new = dict(e)
                new['tissue'] = canon
                if cur and 'raw_tissue' not in new:
                    new['raw_tissue'] = raw
                agg[canon] = new
        return sorted(agg.values(), key=lambda r: -r.get(value_key, 0))

    # ── Expression-subtask answer path (recall + fixed tissue vocab + JSON) ──
    # SciHorizon expression is a structured-DB fact task: the exact-string set-F1
    # scorer demands answers in HPA's fixed ~27-tissue vocabulary, and literature
    # retrieval / agent over-analysis hurt it. This path gives a fair recall prompt
    # with the allowed vocabulary; with +D the gene's HPA tissue expression is
    # injected as supplementary context (single-flag −D/+D on enable_disco).
    _EXPR_VOCAB = [
        'adrenal', 'appendix', 'bone marrow', 'brain', 'colon', 'duodenum',
        'endometrium', 'esophagus', 'fat', 'gall bladder', 'heart', 'kidney',
        'liver', 'lung', 'lymph node', 'ovary', 'pancreas', 'placenta', 'prostate',
        'salivary gland', 'skin', 'small intestine', 'spleen', 'stomach', 'testis',
        'thyroid', 'urinary bladder',
    ]
    _EXPR_PROMPT = (
        "You are an expert on human gene/protein tissue expression. "
        "List ALL tissues from the ALLOWED list where the gene is expressed — be comprehensive, "
        "include every tissue with meaningful expression, not just the top one. "
        "Choose ONLY from the allowed tissues. Output a JSON array of tissue names, nothing else.\n"
        "ALLOWED TISSUES: {vocab}\n"
    )

    @classmethod
    def _parse_expr_tissues(cls, txt: str) -> list:
        import json as _json
        txt = (txt or "").strip()
        cand = []
        try:
            a = _json.loads(txt)
            if isinstance(a, list):
                cand = [str(x).lower().strip() for x in a]
            elif isinstance(a, dict) and 'tissue_list' in a:
                cand = [str(x).lower().strip() for x in a['tissue_list']]
        except Exception:
            cand = [t.lower().strip() for t in re.sub(r'[\[\]"{}]', ' ', txt).split(',') if t.strip()]
        allowed = set(cls._EXPR_VOCAB) | {'low expression'}
        seen, out = set(), []
        for t in cand:
            if t in allowed and t not in seen:
                seen.add(t); out.append(t)
        return out

    async def _answer_expression(self, question, options=None, evidence_text="", n_docs=0):
        """Repair-context expression answerer.

        Literature evidence is the base context; the atlas REPAIRS it by adding the
        gene's structured HPA tissue expression as a supplementary block the
        literature lacks. The combined context feeds the final (recall + fixed
        tissue vocab + parseable JSON) generation. When the atlas returns nothing
        (gene absent / +D disabled), it degrades to the literature-only answer.
        """
        import json as _json
        start = time.perf_counter()
        vocab = ", ".join(self._EXPR_VOCAB)
        lit_block = (f"Literature evidence:\n{evidence_text}\n" if evidence_text
                     else "Literature evidence: (none retrieved)\n")
        atlas_block, atlas_used, gene_count = "", False, 0
        if self._enable_disco:
            # Tissue-expression questions need only HPA bulk tissue expression;
            # fetch it directly (skip the heavy disco cell-type primitive).
            m = re.search(r'expression pattern of ([A-Za-z0-9\-\._]+) gene', question or '')
            gene = m.group(1) if m else None
            if gene:
                try:
                    import httpx
                    async with httpx.AsyncClient(base_url=self._disco_url, timeout=20) as _c:
                        resp = await _c.post('/primitives/hpa/get_tissue_expression',
                                             json={'gene': gene})
                    if resp.status_code == 200:
                        raw = [{'tissue': e.get('tissue'), 'nx': float(e.get('nx') or 0)}
                               for e in (resp.json().get('entries') or []) if (e.get('nx') or 0) > 0]
                        te = self._normalize_atlas_tissues(raw)
                        allowed = set(self._EXPR_VOCAB)
                        pairs = [(e['tissue'], e['nx']) for e in te if e.get('tissue') in allowed]
                        pairs.sort(key=lambda x: -(x[1] or 0))
                        if pairs:
                            atlas_block = ("Reference tissue expression (HPA nTPM; higher = more "
                                           "expressed) — supplementary structured evidence the "
                                           "literature above does not contain:\n" + gene + ": "
                                           + ", ".join(f"{t}:{nx:.0f}" for t, nx in pairs) + "\n")
                            atlas_used = True
                            gene_count = 1
                except Exception:
                    pass
        # Repair-context prompt: literature base + atlas repair (if any). Falls
        # back to literature-only when the atlas returns nothing.
        context_variant = "literature+atlas" if atlas_used else "literature"
        prompt = (self._EXPR_PROMPT.format(vocab=vocab)
                  + "\n" + lit_block
                  + (("\n" + atlas_block) if atlas_block else "")
                  + f"\nGene question: {question}\nAnswer (JSON array):")
        try:
            resp = await self._logprob_client.chat.completions.create(
                model=self._logprob_model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=300, temperature=0.0,
                extra_body={"chat_template_kwargs": {"enable_thinking": False}},
            )
            txt = resp.choices[0].message.content or "[]"
        except Exception:
            txt = "[]"
        return ModelResponse(
            answer=_json.dumps(self._parse_expr_tissues(txt)),
            response_text=txt,
            latency_ms=(time.perf_counter() - start) * 1000,
            metadata={"method": "v14_cascade", "question_type": "expression",
                      "context_variant": context_variant,
                      "atlas_used": atlas_used, "atlas_gene_count": gene_count,
                      "atlas_fallback_to_literature": not atlas_used,
                      "docs": n_docs, "enable_disco": self._enable_disco},
        )

    @staticmethod
    def _summarize_atlas_for_rejudge(atlas: dict, max_per_gene: int = 8) -> str:
        """Render atlas dict as compact human-readable text for rejudge stage.

        The agent may have used Python on `atlas` to derive its answer; the rejudge
        LLM doesn't have REPL access, so we serialize the most salient atlas rows
        as plaintext evidence so rejudge can corroborate the agent's claim.
        """
        if not atlas or not atlas.get('genes'):
            return ""
        lines = []
        for gene, gd in atlas['genes'].items():
            te = gd.get('tissue_expression', [])
            if te:
                tsrc = gd.get('tissue_source', 'unknown')
                parts = [f"{r.get('tissue')}(NX={r.get('nx', 0):.2f})" for r in te[:max_per_gene]]
                lines.append(f"  {gene} [tissue:{tsrc}] tissues: {', '.join(parts)}")
            ce = gd.get('celltype_expression', [])
            if ce:
                csrc = gd.get('celltype_source') or 'disco_scrna'
                parts = [f"{r.get('cell_type')}@{r.get('tissue') or '?'}(avg={r.get('avg_expr', 0):.2f})"
                         for r in ce[:max_per_gene]]
                lines.append(f"  {gene} [celltype:{csrc}] cell-types: {', '.join(parts)}")
        if not lines:
            return ""
        return "## Atlas evidence (Disco scRNA + HPA bulk fallback)\n" + "\n".join(lines)

    async def _fetch_atlas_dict(self, question: str, question_type: str,
                                  preextracted_genes: list[str] | None = None,
                                  preextracted_tissues: list[str] | None = None) -> dict:
        """Fetch atlas context as a structured dict to inject into REPL.

        Tries Disco scRNA first; falls back to HPA bulk for genes Disco doesn't cover.
        Returns dict with shape {genes: {SYMBOL: {tissue_expression, celltype_expression, source}}}.
        Empty dict if no relevant gene/no data.

        Per design: only escalation path calls this. Markers excluded (wrong direction
        for gene→tissue queries; would also explode payload size).
        """
        try:
            import httpx
        except Exception:
            return {}

        # Hybrid gene extraction: union of router-extracted (high-precision LLM)
        # + regex-extracted (high-recall, may catch genes router missed). Router
        # genes preserved at the front so they're prioritised when capped at 5.
        router_genes = list(dict.fromkeys(g.upper() for g in (preextracted_genes or []) if g))
        # XC_TOOLS_ROUTER=1: router genes REPLACE the regex rather than being
        # unioned with it. The union is what let HCO3/AST/WBC reach the atlas.
        if os.environ.get("XC_TOOLS_ROUTER") == "1":
            regex_genes = []
        else:
            regex_genes = [g for g in self._GENE_RE.findall(question)
                           if g not in self._COMMON_NON_GENE]
        regex_genes = list(dict.fromkeys(regex_genes))
        seen = set(router_genes)
        merged = list(router_genes)
        for g in regex_genes:
            if g not in seen:
                merged.append(g); seen.add(g)
        genes = merged[:5]
        if not genes:
            return {}

        # Tissue hint: prefer router-extracted (handles synonyms like
        # "pulmonary"→lung), fall back to keyword regex.
        tissue_hint = None
        if preextracted_tissues:
            tissue_hint = (preextracted_tissues[0] or '').lower().strip() or None
        if not tissue_hint:
            ql = question.lower()
            tissue_hint = next((t for t in self._DISCO_TISSUE_HINTS if t in ql), None)
        ql = question.lower()

        atlas: dict = {
            'queried_genes': genes,
            'queried_tissue': tissue_hint,
            'genes': {},
            'atlas_source_note': 'Disco scRNA atlas (33 tissues, 955 cell types) + HPA bulk fallback',
        }

        # Resolve gene aliases / official symbol / descriptive name (L2 alias inject)
        gene_aliases: dict[str, dict] = {}
        try:
            from src.rlm.tools import gene_resolve as _gene_resolve
            for g in genes:
                try:
                    info = _gene_resolve(g) or {}
                    gene_aliases[g] = {
                        'symbol_official': info.get('symbol', g),
                        'name': info.get('name', ''),
                        'aliases': [a.strip() for a in (info.get('aliases', '') or '').split(',') if a.strip()],
                    }
                except Exception:
                    gene_aliases[g] = {'symbol_official': g, 'name': '', 'aliases': []}
        except Exception:
            for g in genes:
                gene_aliases[g] = {'symbol_official': g, 'name': '', 'aliases': []}

        # NOTE: atlas['genes'] is keyed by UPPERCASE official symbol.
        # When using this dict in REPL or tests, always upper-case the
        # symbol before lookup: atlas["genes"][gene.upper()].
        # HPA top_k=30 captures the full ~32 normal-tissue distribution before
        # NX>0 filter; Disco top_k=20 caps payload after lab-name normalization.

        async def _fetch_one_gene(client, g):
            gene_data: dict = {}
            tissue_source = None
            celltype_source = None
            # 1) HPA tissue (standard names; preferred)
            try:
                resp = await client.post(
                    '/primitives/hpa/get_tissue_expression',
                    json={'gene': g, 'top_k': 30},
                )
                if resp.status_code == 200:
                    rows = [r for r in (resp.json().get('entries') or [])
                            if r.get('nx', 0) > 0]
                    if rows:
                        raw = [{'tissue': r.get('tissue'),
                                'nx': float(r.get('nx', 0))} for r in rows]
                        gene_data['tissue_expression'] = self._normalize_atlas_tissues(raw)[:25]
                        tissue_source = 'hpa_bulk'
                self._atlas_calls_ok += 1
            except Exception as e:
                self._atlas_calls_fail += 1
                _atlas_logger.warning(
                    f"atlas-fetch-failed gene={g} endpoint=hpa_tissue "
                    f"err={type(e).__name__}: {str(e)[:120]}"
                )
            # 2) Disco tissue fallback if HPA missed
            if not gene_data.get('tissue_expression'):
                try:
                    resp = await client.post(
                        '/primitives/disco/get_tissue_expression',
                        json={'gene': g, 'top_k': 20},
                    )
                    if resp.status_code == 200:
                        rows = [r for r in (resp.json().get('entries') or [])
                                if r.get('nx', 0) > 0]
                        if rows:
                            raw = [{'tissue': r.get('tissue'),
                                    'nx': float(r.get('nx', 0)),
                                    'n_samples': r.get('n_samples')} for r in rows]
                            gene_data['tissue_expression'] = self._normalize_atlas_tissues(raw)[:20]
                            tissue_source = 'disco_scrna'
                    self._atlas_calls_ok += 1
                except Exception as e:
                    self._atlas_calls_fail += 1
                    _atlas_logger.warning(
                        f"atlas-fetch-failed gene={g} endpoint=disco_tissue "
                        f"err={type(e).__name__}: {str(e)[:120]}"
                    )
            # 3) Disco celltype (always added when available)
            try:
                resp = await client.post(
                    '/primitives/disco/get_celltype_expression',
                    json={'gene': g, 'tissue': tissue_hint, 'top_k': 15},
                )
                if resp.status_code == 200:
                    rows = [r for r in (resp.json().get('entries') or [])
                            if r.get('avg_expr', 0) > 0]
                    if rows:
                        ce = []
                        for r in rows[:15]:
                            ct = r.get('cell_type', '')
                            if '::' in ct:
                                t, ct = ct.split('::', 1)
                            else:
                                t = tissue_hint
                            ce.append({
                                'tissue': t, 'cell_type': ct,
                                'avg_expr': float(r.get('avg_expr', 0)),
                                'pct_cells': float(r.get('pct_cells', 0)),
                                'n_cells': r.get('n_cells', 0),
                            })
                        gene_data['celltype_expression'] = ce
                        celltype_source = 'disco_scrna'
                self._atlas_calls_ok += 1
            except Exception as e:
                self._atlas_calls_fail += 1
                _atlas_logger.warning(
                    f"atlas-fetch-failed gene={g} endpoint=disco_celltype "
                    f"err={type(e).__name__}: {str(e)[:120]}"
                )
            # Always attach alias info even if data sparse — agent may still
            # need to align gene name to question vocabulary (e.g., "Type 1 deiodinase")
            aliases = gene_aliases.get(g, {})
            if gene_data or aliases.get('aliases') or aliases.get('name'):
                gene_data['symbol_official'] = aliases.get('symbol_official', g)
                gene_data['name'] = aliases.get('name', '')
                gene_data['aliases'] = aliases.get('aliases', [])
                # Split source attribution: separate fields for tissue vs cell-type
                gene_data['tissue_source'] = tissue_source or 'unknown'
                gene_data['celltype_source'] = celltype_source
                return g, gene_data
            return g, None

        # Cap total atlas-fetch wall time per question; gather across genes
        # to overlap network IO. Past 30s = give up, agent runs on PubMed only.
        timeout = httpx.Timeout(10.0, connect=3.0)
        try:
            async with httpx.AsyncClient(base_url=self._disco_url, timeout=timeout) as client:
                results = await asyncio.wait_for(
                    asyncio.gather(*[_fetch_one_gene(client, g) for g in genes]),
                    timeout=30.0,
                )
            for g, gd in results:
                if gd:
                    atlas['genes'][g] = gd
        except asyncio.TimeoutError:
            _atlas_logger.warning(
                f"atlas-fetch-timeout total>30s genes={genes}"
            )
        return atlas

    async def _hyde_retrieve(self, question: str):
        """HyDE-style retrieval: LLM writes hypothesis, embed it, retrieve."""
        from utils.clients import embed_client, llm_client
        try:
            hyp = await llm_client.chat(
                HYDE_PROMPT.format(question=question),
                max_tokens=256, temperature=0.1,
            )
            hyp = (hyp or "").strip()
        except Exception:
            hyp = ""
        hyde_text = hyp or question
        vec = await embed_client.embed_single(hyde_text)
        return await retrieve_dense_evidence(
            question, qdrant=self._qdrant, pool=self._pool,
            top_k=self._top_k, collection=self._collection,
            query_vector=vec,
        )

    async def _dual_retrieve_rerank(self, question: str, neg_query: str | None = None,
                                     pool_each: int = 30):
        """Dual retrieval (standard + HyDE [+ neg]) with cross-encoder rerank.

        Robust against HyDE hallucination: standard retrieval guarantees
        entity-specific docs, HyDE adds concept-level candidates, optional
        neg_query adds counter-evidence (for yesno). Rerank by original
        question unifies scores and filters drift.
        """
        from utils.clients import rerank_client
        tasks = [
            retrieve_dense_evidence(
                question, qdrant=self._qdrant, pool=self._pool,
                top_k=pool_each, collection=self._collection,
            ),
            self._hyde_retrieve_pool(question, pool_each),
        ]
        if neg_query:
            tasks.append(retrieve_dense_evidence(
                neg_query, qdrant=self._qdrant, pool=self._pool,
                top_k=pool_each, collection=self._collection,
            ))
        results = await asyncio.gather(*tasks)
        retrievals = [docs for docs, _ in results]

        # Union + dedup (order: standard, HyDE, neg). Key on (pmid, text), not
        # pmid alone: with XC_FULLTEXT_CHUNKS every chunk of a paper shares its
        # PMID, and pmid-only dedup kept just the first chunk per paper -- which
        # silently swapped the gold passage for a weaker same-paper chunk. For
        # abstract retrieval one pmid has one text, so behaviour is unchanged.
        seen = set()
        merged = []
        for docs in retrievals:
            for d in docs:
                key = (d.pmid, d.abstract)
                if key not in seen:
                    seen.add(key); merged.append(d)

        if not merged:
            return [], {}

        # Rerank by ORIGINAL question (unifies scores from different queries)
        passages = [f"{d.title}\n{d.abstract}".strip() for d in merged]
        try:
            scored = await rerank_client.rerank(
                question, passages, top_k=self._top_k, max_concurrent=8,
            )
        except Exception:
            scored = None

        tag = "dual_rerank_yesno" if neg_query else "dual_rerank"
        if scored:
            ordered = []
            for rank, (idx, score) in enumerate(scored, start=1):
                if 0 <= idx < len(merged):
                    d = merged[idx]; d.score = float(score); d.rank = rank
                    ordered.append(d)
            return ordered[:self._top_k], {"method": tag, "pool": len(merged)}
        return merged[:self._top_k], {"method": f"{tag}_fallback", "pool": len(merged)}

    async def _hyde_retrieve_pool(self, question: str, pool_size: int):
        """HyDE retrieval with custom pool size."""
        from utils.clients import embed_client, llm_client
        try:
            hyp = await llm_client.chat(
                HYDE_PROMPT.format(question=question),
                max_tokens=256, temperature=0.1,
            )
            hyp = (hyp or "").strip()
        except Exception:
            hyp = ""
        hyde_text = hyp or question
        vec = await embed_client.embed_single(hyde_text)
        return await retrieve_dense_evidence(
            question, qdrant=self._qdrant, pool=self._pool,
            top_k=pool_size, collection=self._collection,
            query_vector=vec,
        )

    async def __aenter__(self):
        await super().__aenter__()
        from openai import AsyncOpenAI
        cfg = get_config()
        ep = cfg.llm.endpoints[0]
        # Explicit long timeout: the SDK default (600 s) was silently
        # timing out 34% of stage-1 calls once the CoT budget grew, and the
        # caller's except-handler turns that into answer="" / confidence=0.0,
        # which force-escalates the item. 61% escalation was mostly failures.
        self._logprob_client = AsyncOpenAI(base_url=ep.url, api_key=cfg.llm.api_key,
                                           timeout=1800.0, max_retries=2)
        self._logprob_model = ep.model
        # Lazy-init agent only when needed (Stage 2).
        # max_iterations=8 was too tight once the agent has retrieval tools: in the
        # 2026-09-09 probe 3 of 5 escalations used 9-11 LLM calls, hit the cap and
        # returned an empty answer, so the rejudge received "Agent analysis:\n" and
        # nothing else. XC_AGENT_MAX_ITERS raises it; default unchanged.
        self._agent_config = PipelineConfig(
            version=self._version,
            max_iterations=int(os.environ.get("XC_AGENT_MAX_ITERS", "8")),
            use_gold_context=True,
            yesno_dual_hypothesis=False,
            enable_kg_tools=self._enable_tools,
        )
        return self

    async def __aexit__(self, *args):
        if self._pipeline:
            self._pipeline = None
            shutdown_tools()
        if self._logprob_client:
            await self._logprob_client.close()
        await super().__aexit__(*args)

    async def _constrained_with_confidence(
        self, question, question_type, context, options
    ):
        """Stage 1: constrained generation with logprob confidence score."""
        from kg.answer_prompts import build_answer_messages, extract_constrained_answer, get_max_tokens

        system_prompt, user_prompt = build_answer_messages(
            question_type=question_type,
            question=question,
            context=context,
            options=options,
        )
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": user_prompt})

        # Official MedXpertQA protocol (XC_MCQ_OFFICIAL=1).
        # Reproduces TsinghuaC3I/MedXpertQA eval/: system role, inline
        # "Answer Choices: (A) ...", zero-shot CoT, then the
        # "Therefore, among A through <end>, the answer is" trigger, and
        # word-boundary letter extraction over exactly the offered letters.
        # The only deviation is the retrieved-evidence block, which the
        # zero-shot official harness has no slot for.
        if (os.environ.get("XC_MCQ_OFFICIAL") == "1"
                and question_type == "mcq" and options):
            keys = sorted(options)
            end = keys[-1]
            choices = " ".join(f"({k}) {options[k]}" for k in keys)
            q = f"{question}\nAnswer Choices: {choices}"
            body = f"Evidence:\n{context}\n\nQ: {q}\nA: Let's think step by step."
            msgs = [{"role": "system", "content": "You are a helpful medical assistant."},
                    {"role": "user", "content": body}]
            try:
                cot = await self._logprob_client.chat.completions.create(
                    model=self._logprob_model, messages=msgs,
                    max_tokens=int(os.environ.get("XC_MCQ_OFFICIAL_TOKENS", "512")),
                    temperature=0.1,
                    extra_body={"chat_template_kwargs": {"enable_thinking": False}},
                )
                rationale = (cot.choices[0].message.content or "").strip()
            except Exception:
                rationale = ""
            trigger = f"Therefore, among A through {end}, the answer is"
            # The official harness is text completion: the trigger is a prefix of
            # the model's OWN turn, which it continues. Sending it as a separate
            # user turn makes a chat model reply in prose instead of a letter, so
            # continue the assistant message instead.
            msgs2 = msgs + [{"role": "assistant",
                             "content": f"{rationale}\n{trigger}"}]
            resp = await self._logprob_client.chat.completions.create(
                model=self._logprob_model, messages=msgs2, max_tokens=8,
                temperature=0.1, logprobs=True, top_logprobs=5,
                extra_body={"chat_template_kwargs": {"enable_thinking": False},
                            "continue_final_message": True,
                            "add_generation_prompt": False},
            )
            text = resp.choices[0].message.content or ""
            for junk in ("I understand", f"A through {end}"):
                text = text.replace(junk, "")
            # Match on the raw text exactly like the official cleaner: upper-casing
            # first would let a stray standalone "a"/"i" in prose match as an option.
            hits = re.findall(r"\b(" + "|".join(keys) + r")\b", text)
            ans = hits[0] if hits else ""
            # Confidence must be the probability of the ANSWER token, not of the
            # first token emitted: the model opens with markup (" **(") before
            # the letter, so lp.content[0] would score the formatting instead.
            conf = 0.5
            lp = resp.choices[0].logprobs
            if lp and lp.content:
                import math
                for tok in lp.content:
                    if tok.token.strip(" *()[].:\n").upper() == ans and ans:
                        if tok.logprob is not None:
                            conf = math.exp(tok.logprob)
                        break
            return ans, text, conf

        # Optional two-pass MCQ reasoning (XC_MCQ_REASONING=1).
        # The default single-pass path is unchanged: stage-1 emits one letter,
        # so the first-token logprob below IS the answer probability. Letting
        # the model reason inline would make that first token a word like
        # "The", silently destroying the escalation signal, so reasoning is
        # done in a separate pass and the constrained letter call is kept.
        if os.environ.get("XC_MCQ_REASONING") == "1" and question_type == "mcq":
            reason_msgs = list(messages)
            reason_msgs[0] = {
                "role": "system",
                "content": ("You are a biomedical expert. Reason step by step, "
                            "concisely, about which option is correct. "
                            "Do NOT state a final letter yet."),
            }
            think = await self._logprob_client.chat.completions.create(
                model=self._logprob_model,
                messages=reason_msgs,
                max_tokens=int(os.environ.get("XC_MCQ_REASONING_TOKENS", "512")),
                temperature=0.1,
                extra_body={"chat_template_kwargs": {"enable_thinking": False}},
            )
            rationale = (think.choices[0].message.content or "").strip()
            if rationale:
                messages = messages + [
                    {"role": "assistant", "content": rationale},
                    {"role": "user", "content":
                     "Now output ONLY the single uppercase option letter."},
                ]

        response = await self._logprob_client.chat.completions.create(
            model=self._logprob_model,
            messages=messages,
            max_tokens=get_max_tokens(question_type),
            temperature=0.1,
            logprobs=True,
            top_logprobs=5,
            extra_body={"chat_template_kwargs": {"enable_thinking": False}},
        )

        answer_text = response.choices[0].message.content or ""
        answer = extract_constrained_answer(answer_text, question_type, options)

        # Extract confidence from first token logprob
        confidence = 0.5  # default
        lp_content = response.choices[0].logprobs
        if lp_content and lp_content.content:
            first_token = lp_content.content[0]
            if first_token.logprob is not None:
                import math
                confidence = math.exp(first_token.logprob)  # convert logprob to probability

        return answer, answer_text, confidence

    async def _debate_yesno(self, question, docs):
        """Dual-agent debate for low-confidence yesno questions.

        Two advocates argue opposite sides, then an adjudicator judges.
        This avoids the single-agent NO bias on ambiguous evidence.
        """
        from utils.clients import llm_client

        # Per-document cap for the agent/rejudge evidence. 1200 chars fits an abstract;
        # PMC full-text chunks run ~1900 chars, so the agent saw only the head of each
        # chunk while stage 1 saw the whole chunk (35% of LitQA2 key passages fell past
        # 1200 chars). XC_EVIDENCE_DOC_CHARS widens it; default unchanged.
        evidence_text = format_pubmed_evidence(
            docs, max_docs=20,
            max_abstract_chars=int(os.environ.get("XC_EVIDENCE_DOC_CHARS", "1200")),
        )

        # Advocate YES: argue why the answer should be yes
        yes_prompt = (
            f"You are an advocate arguing that the answer to this biomedical question is YES.\n\n"
            f"Question: {question}\n\n"
            f"Evidence:\n{evidence_text[:6000]}\n\n"
            f"Make the strongest possible case for YES in 2-3 sentences. "
            f"Focus on supporting evidence, positive findings, and confirmed associations."
        )

        # Advocate NO: argue why the answer should be no
        no_prompt = (
            f"You are an advocate arguing that the answer to this biomedical question is NO.\n\n"
            f"Question: {question}\n\n"
            f"Evidence:\n{evidence_text[:6000]}\n\n"
            f"Make the strongest possible case for NO in 2-3 sentences. "
            f"Focus on negative findings, lack of evidence, failed associations, and limitations."
        )

        # Run both advocates concurrently
        yes_arg, no_arg = await asyncio.gather(
            llm_client.chat(yes_prompt, max_tokens=200, temperature=0.1),
            llm_client.chat(no_prompt, max_tokens=200, temperature=0.1),
        )

        # Adjudicator: weigh both arguments
        adjudicator_system = (
            "You are an impartial scientific adjudicator.\n"
            "Two advocates have argued opposite sides of a biomedical question.\n"
            "Judge which side has stronger evidence.\n"
            "Output EXACTLY one word: yes, no, or maybe.\n"
            "Output maybe ONLY if both sides are equally strong and the evidence is truly inconclusive.\n"
            "Lowercase only. No explanation."
        )
        adjudicator_prompt = (
            f"Question: {question}\n\n"
            f"Advocate for YES:\n{yes_arg}\n\n"
            f"Advocate for NO:\n{no_arg}\n\n"
            f"Based on the strength of evidence presented by each side, the answer is:"
        )

        verdict = await llm_client.chat(
            adjudicator_prompt, system=adjudicator_system,
            max_tokens=4, temperature=0.1,
        )

        # Extract yes/no/maybe
        verdict_lower = verdict.strip().lower().split()[0] if verdict.strip() else ""
        if verdict_lower in ("yes", "no", "maybe"):
            return verdict_lower
        if "yes" in verdict.lower():
            return "yes"
        if "no" in verdict.lower():
            return "no"
        return "maybe"

    async def generate(self, question, question_type, context=None, options=None,
                       *, item_key: tuple[str, str] | None = None):
        # Evidence-cut tracing wrapper (src/rlm/trace.py). Opens one trace per item,
        # keyed by (dataset, item_id), so the stage-1 / REPL / history / rejudge
        # recorders below all land in the same file. No-op unless XC_TRACE_DIR is set.
        _tok = _trace.begin(item_key, {"question_type": question_type, "question": str(question)[:500]})
        try:
            resp = await self._generate_inner(question, question_type, context=context,
                                              options=options, item_key=item_key)
        except BaseException as exc:
            _trace.end(_tok, error=f"{type(exc).__name__}: {exc}"[:300])
            raise
        _trace.end(_tok, answer=getattr(resp, "answer", None), metadata=getattr(resp, "metadata", None))
        return resp

    async def _generate_inner(self, question, question_type, context=None, options=None,
                              *, item_key: tuple[str, str] | None = None):
        start = time.perf_counter()

        # Expression: repair-context design. Literature retrieval is the base
        # context; the atlas REPAIRS it by adding structured tissue expression the
        # literature lacks, and the combined context feeds final generation. When
        # the atlas returns nothing, +D degrades to the literature-only answer
        # (single flag −D/+D = enable_disco).
        if question_type == "expression":
            skip_hyde = question_type in self.SKIP_HYDE_TYPES
            use_rewrite = self._rewrite_query or (
                self._selective_rewrite and self._should_rewrite(question, question_type)) or (
                self._type_rewrite and question_type in self.REWRITE_TYPES)
            if self._dual_rerank and not skip_hyde:
                pos_docs, _ = await self._dual_retrieve_rerank(question)
                expr_docs = pos_docs[:self._top_k]
            elif use_rewrite and not skip_hyde:
                pos_docs, _ = await self._hyde_retrieve(question)
                expr_docs = pos_docs
            else:
                pos_docs, _ = await self._retrieve_shared_evidence(question)
                expr_docs = pos_docs
            expr_evidence = format_pubmed_evidence(expr_docs, max_docs=20)
            return await self._answer_expression(
                question, options, evidence_text=expr_evidence, n_docs=len(expr_docs))

        # Retrieve evidence (dual for yesno, standard for others)
        use_rewrite = self._rewrite_query or (
            self._selective_rewrite and self._should_rewrite(question, question_type)) or (
            self._type_rewrite and question_type in self.REWRITE_TYPES)
        skip_hyde = question_type in self.SKIP_HYDE_TYPES
        neg_terms = "no effect OR not effective OR no association OR failed OR no benefit"
        neg_query = f"{question} {neg_terms}" if question_type == "yesno" else None

        if self._dual_rerank and not skip_hyde:
            # Unified rerank over (standard + HyDE [+ neg]); reranker normalizes scores.
            pos_docs, _ = await self._dual_retrieve_rerank(question, neg_query=neg_query)
            docs = pos_docs[:self._top_k]
        else:
            if use_rewrite and not skip_hyde:
                pos_docs, _ = await self._hyde_retrieve(question)
            else:
                pos_docs, _ = await self._retrieve_shared_evidence(question)

            if question_type == "yesno":
                neg_docs, _ = await retrieve_dense_evidence(
                    neg_query, qdrant=self._qdrant, pool=self._pool,
                    top_k=self._top_k, collection=self._collection,
                )
                seen = {d.pmid for d in pos_docs}
                all_docs = list(pos_docs)
                for d in neg_docs:
                    if d.pmid not in seen:
                        seen.add(d.pmid)
                        all_docs.append(d)
                all_docs.sort(key=lambda d: -d.score)
                docs = all_docs[:self._top_k]
            else:
                docs = pos_docs

        evidence_text = format_pubmed_evidence(  # see XC_EVIDENCE_DOC_CHARS note above
            docs, max_docs=20,
            max_abstract_chars=int(os.environ.get("XC_EVIDENCE_DOC_CHARS", "1200")),
        )

        # Pre-fetch authoritative tool data (gene / SNP / genomics / BLAST) up
        # front so BOTH the fast path and the agent see it. Previously it was
        # injected only on escalation, so fast-path GeneTuring lookups (e.g. DNA
        # alignment / organism) answered "unknown" despite a cached answer.
        _rh = self._router_entities.get(item_key, {}) if item_key else {}
        tool_evidence = (
            await V14AgentClient._precall_tools(
                self, question, question_type,
                router_genes=_rh.get('extracted_genes') or [])
            if self._enable_tools else ""
        )

        # Stage 1: Fast constrained generation with confidence
        dense_ctx = build_dense_context(docs)[0] if docs else "No evidence."
        fast_ctx = (tool_evidence + "\n\n" + dense_ctx) if tool_evidence else dense_ctx
        try:
            answer, answer_text, confidence = await self._constrained_with_confidence(
                question, question_type,
                context=fast_ctx,
                options=options,
            )
        except Exception:
            answer, confidence = "", 0.0
            answer_text = ""

        escalated = False
        agent_timed_out = False
        no_ctx_fallback = False
        ungrounded = False
        atlas_context: dict = {}
        atlas_route_decision: str = 'not-attempted'  # not-attempted | router-skipped | fetched
        if self._grounded_gate and question_type in self.GROUNDED_CHECK_TYPES:
            ungrounded = not self._is_answer_grounded(answer, docs)

        # Order-consistency escalation trigger (XC_ESCALATE_ON_DISAGREE=1).
        # For MCQ, replaces the logprob threshold: stage 1 is re-run with the
        # option list reversed and the item escalates when the two orders select
        # different option TEXT. Measured on MedXpertQA, agreement separates
        # 36.4% from ~23% accuracy, against AUC 0.582 for the logprob signal,
        # which additionally differs run-to-run on 96% of items.
        order_disagree = None
        if (os.environ.get("XC_ESCALATE_ON_DISAGREE") == "1"
                and question_type == "mcq" and options):
            _keys = sorted(options)
            _rev = {k: options[k2] for k, k2 in zip(_keys, list(reversed(_keys)))}
            try:
                _rev_answer, _, _ = await self._constrained_with_confidence(
                    question, question_type, context=fast_ctx, options=_rev,
                )
            except Exception:
                _rev_answer = ""
            order_disagree = options.get(answer) != _rev.get(_rev_answer)

        _low_conf = (order_disagree if order_disagree is not None
                     else confidence < self._threshold)

        # An "insufficient information" answer is the one case where more evidence
        # is certainly what is missing, yet it never triggered escalation: picking
        # that option is an easy, high-probability choice, so its confidence is high
        # (LitQA2 stage 1: abstentions score a median 0.990 vs 0.460 for substantive
        # answers). The 0.7 threshold therefore escalated the items the model was
        # already answering well (72.3% correct) and left the abstentions (42.9%)
        # alone. XC_ESCALATE_ON_ABSTAIN=1 escalates them too; default unchanged.
        _abstained = False
        if _ESCALATE_ON_ABSTAIN and answer and options:
            _chosen = str(options.get(answer, "") or "")
            _abstained = bool(_ABSTAIN_RE.search(_chosen))

        if _trace.enabled():
            _s1 = _trace.summarize_docs([{"pmid": getattr(d, "pmid", None), "text": getattr(d, "abstract", "") or ""} for d in (docs or [])])
            _trace.event("stage1", n_docs=len(docs or []), evidence_chars=len(evidence_text or ""),
                         gold_rank=_s1["gold_rank"], gold_hits=_s1["gold_hits"],
                         gold_in_evidence=_trace.gold_in_text(evidence_text or ""),
                         answer=answer, confidence=round(float(confidence or 0.0), 3),
                         low_conf=bool(_low_conf), ungrounded=bool(ungrounded), abstained=bool(_abstained))

        # Stage 2: Escalate if the trigger fires OR ungrounded answer (skip yesno)
        if ((_low_conf or not answer or ungrounded or _abstained)
                and question_type != "yesno"):
            escalated = True

            if False:  # debate path disabled — kept for reference
                answer = await self._debate_yesno(question, docs)
            else:
                # Complex types: full V14 agent reasoning
                if self._pipeline is None:
                    self._pipeline = BiomedicalRLMPipeline(config=self._agent_config)

                # tool_evidence already pre-fetched before the fast path; reuse it.
                if tool_evidence:
                    evidence_text = tool_evidence + "\n\n" + evidence_text

                # Atlas dict (Disco + HPA fallback) — passed as REPL variable, not text.
                # Use router-extracted entities (gene/tissue) when available; gate fetch
                # on the offline router's route_disco decision so non-relevant items
                # don't pay HTTP latency.
                if self._enable_disco:
                    router_hit = self._router_entities.get(item_key, {}) if item_key else {}
                    # Honor route_disco flag when present. Default FALSE: the
                    # loader already defaults a missing field to False, and the
                    # old True here was fail-OPEN — a dataset with no router
                    # data routed 100% of escalated items. On MedXpertQA that
                    # meant 778 atlas fetches, only 45 of which returned data,
                    # burning 5.2h on clinical abbreviations (HCO3/AST/WBC)
                    # that the regex fallback mistook for gene symbols.
                    if router_hit.get('route_disco', False):
                        pre_genes = router_hit.get('extracted_genes') or None
                        pre_tissues = router_hit.get('extracted_tissues') or None
                        atlas_context = await self._fetch_atlas_dict(
                            question, question_type,
                            preextracted_genes=pre_genes,
                            preextracted_tissues=pre_tissues,
                        )
                        atlas_route_decision = 'fetched'
                    else:
                        atlas_route_decision = 'router-skipped'

                ctx = {"evidence": evidence_text}
                if atlas_context.get('genes'):
                    ctx["atlas_context"] = atlas_context
                if options:
                    ctx["options"] = options

                # Wall-clock guard: the RLM agent runs synchronous LLM calls in
                # socketserver threads; a dropped request left them blocked for hours
                # and froze the whole benchmark (2026-09-08). On timeout keep the
                # stage-1 answer instead of waiting forever.
                try:
                    result = await asyncio.wait_for(
                        self._pipeline.answer_async(
                            question=question,
                            question_type=question_type,
                            context=ctx,
                        ),
                        timeout=_AGENT_TIMEOUT,
                    )
                except asyncio.TimeoutError:
                    logger.warning("agent escalation exceeded %.0fs; keeping stage-1 answer", _AGENT_TIMEOUT)
                    result = None
                    agent_timed_out = True

                if result is not None:
                    # `result.answer` is what extract_answer() distilled out of the
                    # agent's last turn — for MCQ that is a single letter, so the
                    # rejudge prompt read "Agent analysis:\nE" with no reasoning and
                    # no trace of anything the agent retrieved. Everything the agent
                    # fetched (abstracts, new chunks) was discarded before this point.
                    # XC_REJUDGE_AGENT_TEXT=1 passes the agent's full final turn
                    # instead; default keeps the letter-only behaviour.
                    agent_reasoning = result.answer or ""
                    if _REJUDGE_AGENT_TEXT:
                        # raw_completion.response is only what FINAL(...) wrapped, i.e. a
                        # letter on 10/12 traced items. The agent's reasoning is the full
                        # text of its last turn, kept by the pipeline as agent_turns.
                        _turns = [t for t in (getattr(result, "agent_turns", None) or []) if t.strip()]
                        _raw = _turns[-1] if _turns else (getattr(getattr(result, "raw_completion", None), "response", "") or "")
                        if _raw.strip():
                            agent_reasoning = _raw.strip()[:_REJUDGE_AGENT_TEXT_CHARS]
                    # Append atlas summary to rejudge context — the rejudge LLM has no
                    # REPL access to inspect `atlas`, so we serialize the salient rows
                    # as plaintext so it can corroborate the agent's atlas-grounded claim.
                    atlas_block = self._summarize_atlas_for_rejudge(atlas_context)
                    rejudge_pieces = [f"Agent analysis:\n{agent_reasoning}"]
                    if atlas_block:
                        rejudge_pieces.append(atlas_block)
                    # Deep half of the fix: hand the rejudge what the agent actually
                    # retrieved. Without this the rejudge only ever saw the stage-1
                    # evidence, so a successful agent retrieval could not change the
                    # answer -- which is why fixing the retrieval tools moved latency
                    # 6x and accuracy 0.
                    if _REJUDGE_AGENT_EVIDENCE:
                        _repl = [t for t in (getattr(result, "repl_stdout", None) or []) if t.strip()]
                        if _repl:
                            _block = "\n\n".join(_repl)[-_REJUDGE_AGENT_EVIDENCE_CHARS:]
                            rejudge_pieces.append(f"Evidence the agent retrieved:\n{_block}")
                    rejudge_pieces.append(f"Original evidence:\n{evidence_text}")
                    rejudge_context = "\n\n".join(rejudge_pieces)
                    _pre_rejudge = answer
                    answer, _ = await generate_constrained_answer(
                        question=question, question_type=question_type,
                        context=rejudge_context, options=options,
                    )
                    if _trace.enabled():
                        # How much of each piece survives generate_constrained_answer's
                        # context[:XC_ANSWER_CONTEXT_CHARS] slice, and whether a gold id does.
                        _cap = int(os.environ.get("XC_ANSWER_CONTEXT_CHARS", "8000"))
                        _off, _pieces = 0, []
                        for _p in rejudge_pieces:
                            _kept = max(0, min(len(_p), _cap - _off))
                            _pieces.append({"label": _p.split("\n", 1)[0][:40], "chars": len(_p), "kept": _kept})
                            _off += len(_p) + 2
                        _trace.event("rejudge", cap=_cap, context_chars=len(rejudge_context), pieces=_pieces,
                                     gold_in_context=_trace.gold_in_text(rejudge_context),
                                     gold_in_capped=_trace.gold_in_text(rejudge_context[:_cap]),
                                     agent_answer=getattr(result, "answer", None), before=_pre_rejudge, after=answer)

        # Last resort when the pipeline still ends on "insufficient information".
        # Abstaining scores zero, while answering the same items with no context at
        # all scores 31% on LitQA2 (the model's own knowledge) -- retrieved but
        # irrelevant evidence talks it out of a guess it would otherwise get right.
        # XC_ANSWER_WITHOUT_CONTEXT_ON_ABSTAIN=1 re-asks with no evidence; default off.
        if (_ANSWER_NO_CTX_ON_ABSTAIN and answer and options
                and question_type != "yesno"
                and _ABSTAIN_RE.search(str(options.get(answer, "") or ""))):
            _fallback, _ = await generate_constrained_answer(
                question=question, question_type=question_type,
                context="", options=options,
            )
            _pre_fallback = answer
            if _fallback and not _ABSTAIN_RE.search(str(options.get(_fallback, "") or "")):
                answer = _fallback
                no_ctx_fallback = True
            _trace.event("fallback", before=_pre_fallback, proposed=_fallback, accepted=no_ctx_fallback)

        return ModelResponse(
            answer=answer,
            response_text=answer_text,
            latency_ms=(time.perf_counter() - start) * 1000,
            metadata={
                "method": "v14_cascade",
                "escalated": escalated,
                "agent_timeout": agent_timed_out,
                "no_ctx_fallback": no_ctx_fallback,
                "stage1_confidence": round(confidence, 3),
                "order_disagree": order_disagree,
                "docs": len(docs),
                "atlas_route": atlas_route_decision,
                "atlas_gene_count": len(atlas_context.get('genes') or {}),
                "atlas_tissue_sources": sorted({
                    g.get('tissue_source', 'unknown')
                    for g in (atlas_context.get('genes') or {}).values()
                    if g.get('tissue_source')
                }),
            },
        )


METHODS = {
    # Baselines
    "no-context": NoContextClient,
    # Same client, separate name: frontier-API runs merge into
    # baselines/no-context-frontier/ instead of overwriting the paper's local run.
    "no-context-frontier": NoContextClient,
    "dense": DenseRAGClient,
    "dual-dense": DualRetrievalDenseClient,
    "hyde": HyDEClient,
    "hyde-mixed": lambda: HyDEClient(mix_query=True),
    "rankrag": RankRAGClient,
    "flare": FLAREClient,
    "chain-of-note": ChainOfNoteClient,
    "ircot": IRCoTClient,
    "ircot-rerank": IRCoTClient,  # Option C: rerank accumulated pool (new)
    "self-rag": SelfRAGClient,
    "self-rag-13b": SelfRAGTrainedClient,
    "genegpt-lite": GeneGPTLiteClient,
    "genegpt": GeneGPTClient,
    "genegpt-fmt": lambda: GeneGPTClient(format_enhanced=True),  # terse-extraction rescue variant
    "biomedrag": BiomedRAGClient,
    "golden-context": GoldenContextDenseClient,
    # KG-RAG baselines
    "rt-lightrag": lambda top_k=20: create_retrieve_then_kg_client("rt_lightrag_hybrid", retrieval_top_k=top_k),
    "rt-pathrag": lambda top_k=20: create_retrieve_then_kg_client("rt_pathrag", retrieval_top_k=top_k),
    "rt-graphrag": lambda top_k=20: create_retrieve_then_kg_client("rt_graphrag_local", retrieval_top_k=top_k),
    "mesh-kg": MeSHKGRAG,
    "biorag": BioRAGClient,
    "biorag-full": lambda: BioRAGClient(shared_retrieval_only=False),
    # Our system
    "v14": lambda: V14AgentClient(dual_hypothesis=False, version="v14"),
    "v14.1": lambda: V14AgentClient(dual_hypothesis=False, version="v14.1"),
    "v14.2": lambda: V14AgentClient(dual_hypothesis=False, version="v14.2"),
    "v14.3": lambda: V14AgentClient(dual_hypothesis=False, version="v14.3"),
    "v14-dh": lambda: V14AgentClient(dual_hypothesis=True),
    "v14-cascade": lambda: V14CascadeClient(confidence_threshold=0.7, version="v14"),
    "v14-cascade-rewrite": lambda: V14CascadeClient(confidence_threshold=0.7, version="v14", rewrite_query=True),
    "v14-cascade-selective-rewrite": lambda: V14CascadeClient(confidence_threshold=0.7, version="v14", selective_rewrite=True),
    "v14-cascade-type-rewrite": lambda: V14CascadeClient(confidence_threshold=0.7, version="v14", type_rewrite=True),
    "v14-forced-agent-trw": lambda: V14CascadeClient(confidence_threshold=1.0, version="v14", type_rewrite=True),
    "v14-cascade-dual-rerank": lambda: V14CascadeClient(confidence_threshold=0.7, version="v14", dual_rerank=True),
    "v14-forced-agent-dual-rerank": lambda: V14CascadeClient(confidence_threshold=1.0, version="v14", dual_rerank=True),
    "v14-cascade-dual-rerank-grounded": lambda: V14CascadeClient(confidence_threshold=0.7, version="v14", dual_rerank=True, grounded_gate=True),
    # Same client, separate result directories: the prompt-audit equivalence check must not merge
    # its 10 % sample into the -D ablation column, whose cells are the April full-set run.
    "v14-promptcheck-old": lambda: V14CascadeClient(confidence_threshold=0.7, version="v14", dual_rerank=True, grounded_gate=True),
    "v14-promptcheck-new": lambda: V14CascadeClient(confidence_threshold=0.7, version="v14", dual_rerank=True, grounded_gate=True),
    "v14-cascade-dual-rerank-grounded-disco": lambda: V14CascadeClient(confidence_threshold=0.7, version="v14", dual_rerank=True, grounded_gate=True, enable_disco=True),
    # Headline config for the GeneTuring genomics-tool before/after run; isolated
    # key so canonical dirs are untouched. disco off: the atlas contributes
    # nothing to GeneTuring (no atlas_route in baseline metadata) and its server
    # is down — disabling it gives identical results without the timeout latency.
    "v14-geneturing-genomics": lambda: V14CascadeClient(confidence_threshold=0.7, version="v14", dual_rerank=True, grounded_gate=True, enable_disco=False),
    # Ablation variants re-run on GeneTuring WITH the genomics tools (honest
    # ablation: each keeps the tools, so the GeneTuring column isolates the
    # ablated component instead of "only headline has genomics"). Isolated keys
    # mirror the canonical ablation configs; canonical dirs untouched.
    "abl-C-gt":  lambda: V14CascadeClient(confidence_threshold=1.0, version="v14", dual_rerank=True),                              # -C (forced-agent-dual-rerank)
    "abl-R-gt":  lambda: V14CascadeClient(confidence_threshold=0.7, version="v14", dual_rerank=False, grounded_gate=True),         # -R (cascade-grounded)
    "abl-RC-gt": lambda: V14CascadeClient(confidence_threshold=1.0),                                                              # -R,-C (v14-forced-agent: always-escalate, no dual-rerank)
    "abl-A-gt":  lambda: V14CascadeClient(confidence_threshold=0.0),                                                              # -A (forced-fast)
    "v14-cascade-grounded": lambda: V14CascadeClient(confidence_threshold=0.7, version="v14", dual_rerank=False, grounded_gate=True),
    "v14-cascade-dual-rerank-grounded-no-tools": lambda: V14CascadeClient(confidence_threshold=0.7, version="v14", dual_rerank=True, grounded_gate=True, enable_tools=False),
    # Ablations
    "v14-no-tools": lambda: V14AgentClient(enable_tools=False),
    # Single-flag counterpart of the headline config: identical to
    # "v14-cascade-dual-rerank-grounded" except confidence_threshold 0.7 -> 0.0,
    # i.e. the agent path never fires. "v14-forced-fast" below differs from the
    # headline by THREE flags (no dual_rerank, no grounded_gate), so it cannot
    # isolate the agent path's contribution.
    "v14-cascade-dual-rerank-grounded-forcedfast": lambda: V14CascadeClient(
        confidence_threshold=0.0, version="v14", dual_rerank=True, grounded_gate=True),
    "v14-forced-fast": lambda: V14CascadeClient(confidence_threshold=0.0),  # never escalate (confidence always >= 0)
    "v14-forced-agent": lambda: V14CascadeClient(confidence_threshold=1.0),  # always escalate (confidence always < 1)
    "v14-no-rejudge": lambda: V14AgentClient(enable_rejudge=False),
}


async def main():
    import argparse
    parser = argparse.ArgumentParser(description="Unified benchmark with shared evidence")
    parser.add_argument("--method", required=True, choices=list(METHODS.keys()))
    parser.add_argument("--datasets", nargs="+", default=["bioasq"])
    parser.add_argument("--limit", type=int, default=None, help="0 or omit for full")
    parser.add_argument("--top-k", type=int, default=20, help="Shared retrieval top-k")
    parser.add_argument("--concurrency", type=int, default=16)
    parser.add_argument("--use-golden-context", action="store_true",
                        help="Pass dataset golden context to model (for golden-context method)")
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--question-types", nargs="+", default=None,
                        help="Filter to these question_types only (e.g. expression mcq)")
    parser.add_argument("--item-filter", type=str, default=None,
                        help="Path to JSONL with {dataset, id, route_disco} — only run items where route_disco==true")
    parser.add_argument("--router-entities", type=str, default=None,
                        help="Path to router JSONL with extracted_genes/tissues/celltypes per question — used by disco fetch")
    args = parser.parse_args()
    if args.limit == 0:
        args.limit = None

    unified_dir = Path(__file__).parent.parent / "benchmark" / "unified"
    output_root = Path(__file__).parent.parent / "output" / "unified_benchmark"
    output_dir = output_root / args.method if args.method in {"v14.1", "v14.2", "v14.3"} else output_root

    loader = BenchmarkLoader(unified_dir=unified_dir)
    evaluator = MetricsEvaluator()

    # Auto-resolve resume: check baselines/ or ablation/ for existing results
    if not args.resume:
        for check_dir in [output_root / "ablation" / args.method, output_root / "baselines" / args.method]:
            if check_dir.exists() and any(check_dir.glob("*.jsonl")):
                args.resume = str(check_dir)
                print(f"Auto-resume from: {check_dir}")
                break

    if args.resume:
        resume_path = Path(args.resume)
        if "baselines" in str(resume_path) or "ablation" in str(resume_path):
            logger = BenchmarkLogger(log_dir=output_dir)
        else:
            logger = BenchmarkLogger(log_dir=resume_path.parent, run_id=resume_path.name)
    else:
        logger = BenchmarkLogger(log_dir=output_dir)

    # Create method client
    factory = METHODS[args.method]
    if isinstance(factory, type):
        # Class — try passing top_k if it accepts it
        import inspect
        sig = inspect.signature(factory.__init__)
        if 'top_k' in sig.parameters or 'retrieval_top_k' in sig.parameters:
            client = factory(top_k=args.top_k) if 'top_k' in sig.parameters else factory(retrieval_top_k=args.top_k)
        else:
            client = factory()
    elif callable(factory):
        # Lambda/function — try passing top_k
        import inspect
        sig = inspect.signature(factory)
        if sig.parameters:
            client = factory(top_k=args.top_k)
        else:
            client = factory()
    else:
        client = factory

    if hasattr(client, '_top_k'):
        client._top_k = args.top_k

    runner = BenchmarkRunner(loader, evaluator, logger)

    item_allow_set = None
    if args.item_filter:
        import json as _json
        item_allow_set = set()
        with open(args.item_filter) as _f:
            for _line in _f:
                try:
                    _d = _json.loads(_line)
                    if _d.get('route_disco'):
                        item_allow_set.add((_d['dataset'], _d['id']))
                except Exception:
                    pass
        print(f"Item filter: {len(item_allow_set)} items allowed (from {args.item_filter})")

    # Optional router entities — feed extracted_genes/tissues to disco fetch.
    # Keyed by (dataset, item_id) to avoid silent miss from question-text drift.
    if args.router_entities and hasattr(client, '_router_entities'):
        import json as _json
        _router_lookup: dict = {}
        with open(args.router_entities) as _f:
            for _line in _f:
                try:
                    _d = _json.loads(_line)
                    _router_lookup[(_d['dataset'], _d['id'])] = {
                        'extracted_genes': _d.get('extracted_genes', []),
                        'extracted_tissues': _d.get('extracted_tissues', []),
                        'extracted_celltypes': _d.get('extracted_celltypes', []),
                        'route_disco': _d.get('route_disco', False),
                    }
                except Exception:
                    pass
        client._router_entities = _router_lookup
        print(f"Router entities: loaded {len(_router_lookup)} (dataset, id)→entities lookups")

    model_name = f"unified-{args.method}-top{args.top_k}"
    print("=" * 60)
    print(f"Unified Benchmark: {args.method}")
    print(f"Datasets: {args.datasets} | Top-k: {args.top_k} | Concurrency: {args.concurrency}")
    print("=" * 60)

    start = time.perf_counter()
    async with client:
        report = await runner.run(
            model_client=client,
            model_name=model_name,
            config=RunConfig(
                datasets=args.datasets, limit=args.limit,
                concurrency=args.concurrency,
                use_golden_context=args.use_golden_context or args.method == "golden-context",
                resume_dir=args.resume,
                subtasks=args.question_types,
                item_allow=item_allow_set,
            ),
        )
    elapsed = time.perf_counter() - start

    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    print_report(report)
    print(f"\nTotal time: {elapsed / 60:.1f} min")
    print(f"Saved: {logger.get_run_dir()}")

    # Merge new results into baselines/ or ablation/ per method
    ABLATION_METHODS = {"v14-no-tools", "v14-forced-fast", "v14-forced-agent", "v14-no-rejudge", "v14-cascade", "v14-cascade-rewrite", "v14-cascade-selective-rewrite", "v14-cascade-type-rewrite", "v14-forced-agent-trw", "v14-cascade-dual-rerank", "v14-forced-agent-dual-rerank", "v14-cascade-dual-rerank-grounded", "v14-cascade-dual-rerank-grounded-forcedfast", "v14-cascade-dual-rerank-grounded-disco", "v14-cascade-grounded", "v14-cascade-dual-rerank-grounded-no-tools", "v14-geneturing-genomics", "abl-C-gt", "abl-R-gt", "abl-RC-gt", "abl-A-gt"}
    V14_MAIN = {"v14", "v14.1", "v14.2", "v14.3", "v14-dh"}
    if args.method not in V14_MAIN:
        import json as _mjson
        if args.method in ABLATION_METHODS:
            results_dir = output_root / "ablation" / args.method
        else:
            results_dir = output_root / "baselines" / args.method
        results_dir.mkdir(parents=True, exist_ok=True)
        run_jsonl = logger.get_run_dir() / "run.jsonl"
        if run_jsonl.exists():
            # Load existing items
            existing = {}
            for f in results_dir.glob("*.jsonl"):
                # Skip the token-F1 sidecars (scripts/rescore_factoid_tokenf1.py). They share
                # item ids with the result files, so reading them here replaced each real record
                # with a {id, dataset, score_factoid_tokenf1} stub and the rewrite below then
                # wrote the stubs back as the result file. Found 2026-09-13: 15 geneturing.jsonl
                # files had lost every prediction this way.
                if f.name.endswith(".tokenf1.jsonl"):
                    continue
                for line in open(f):
                    try:
                        r = _mjson.loads(line)
                        if r.get("id"):
                            existing[r["id"]] = r
                    except: pass
            # Read new items
            new_count = 0
            for line in open(run_jsonl):
                try:
                    r = _mjson.loads(line)
                    if r.get("type") == "item" and r.get("id"):
                        existing[r["id"]] = r
                        new_count += 1
                except: pass
            # Rewrite per-dataset files
            from collections import defaultdict as _dd
            by_ds = _dd(list)
            for item in existing.values():
                by_ds[item.get("dataset", "unknown")].append(item)
            total = 0
            for ds in sorted(by_ds):
                items = sorted(by_ds[ds], key=lambda x: x.get("id", ""))
                with open(results_dir / f"{ds}.jsonl", "w") as f:
                    for item in items:
                        f.write(_mjson.dumps(item, ensure_ascii=False) + "\n")
                total += len(items)
            correct = sum(1 for v in existing.values() if v.get("correct"))
            acc = f"{correct/total*100:.1f}%" if total else "—"
            _mjson.dump({"method": args.method, "total": total, "correct": correct, "accuracy": acc},
                        open(results_dir / "summary.json", "w"), indent=2)
            subdir = "ablation" if args.method in ABLATION_METHODS else "baselines"
            print(f"Merged {new_count} new items → {subdir}/{args.method}/ ({total} total, {acc})")


if __name__ == "__main__":
    # Diagnostic only, off by default: with XC_FAULTHANDLER=1, `kill -USR1 <pid>`
    # dumps every thread's Python stack to stderr without stopping the run. A
    # 12k-context LitQA2 run froze for 3 h in a synchronous socket recv inside
    # some __init__ and macOS offers no root-less py-spy, so this is the way to
    # see the Python frames next time.
    if os.environ.get("XC_FAULTHANDLER"):
        import faulthandler, signal
        faulthandler.register(signal.SIGUSR1, all_threads=True, chain=False)
    asyncio.run(main())
