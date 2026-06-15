"""PipelineCascadeClient: the bioHarness headline method.

Pipeline shape (single best-config configuration, no ablation flags):

    item -> rewrite query (yesno also produces a paired negative-hypothesis query)
         -> dense retrieve + dual rerank (yesno: union of pos/neg retrievals)
         -> constrained generation (per-type prompt, per-type max_tokens,
            mean-logprob confidence)
         -> if confidence >= 0.7 AND answer is grounded AND not force_agent
                AND item is yesno  -> return fast-path answer
            else                    -> agent escalation + constrained re-judgment
                                       (yesno still uses the fast path because
                                        the agent over-analyses; this matches
                                        the paper §5.2 finding)

The single user-facing knob is ``ServiceConfig.force_agent``: when True,
every non-yesno item bypasses the fast path. yesno always takes the fast
path even with ``force_agent`` because the agent introduces a documented
'no' bias for yes/no questions.

Every stage is a stable extension point: subclass ``PipelineCascadeClient`` and
override ``_retrieve``, ``_fast_path``, ``_agent``, or ``_rejudge`` to plug
in a heavier or differently-tuned implementation while keeping the cascade
routing intact. To reach the published headline numbers on the full
19,302-item benchmark you must connect a live LLM stack (see
``docs/infra.md``).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

from framework_eval.eval.types import Item, Prediction

from bioharness.config import (
    CASCADE_THRESHOLD,
    DENSE_COLLECTION,
    RERANK_TOP_K,
    ServiceConfig,
)

LOGGER = logging.getLogger(__name__)


# ----------------------------------------------------------------------
# Stage records
# ----------------------------------------------------------------------


@dataclass
class RetrievalContext:
    passages: list[dict[str, Any]] = field(default_factory=list)
    rerank_scores: list[float]     = field(default_factory=list)
    rewritten_query: str           = ""
    negative_query: str            = ""


@dataclass
class FastPathOutcome:
    answer: str
    response_text: str
    logprob: float
    grounded: bool
    tool_evidence: str = ""


@dataclass
class AgentOutcome:
    answer: str
    response_text: str
    iterations: int
    tool_calls: list[str]
    tool_evidence: str = ""


# ----------------------------------------------------------------------
# Client
# ----------------------------------------------------------------------


class PipelineCascadeClient:
    """Headline bioHarness method registered as ``pipeline``."""

    name = "pipeline"

    def __init__(self, *, services: ServiceConfig | None = None) -> None:
        self.services = services or ServiceConfig.from_env()

        # Lazy-imported clients; created on first use.
        self._llm     = None
        self._embed   = None
        self._rerank  = None
        self._qdrant  = None
        self._atlas   = None
        self._closed  = False

    # ------------------------------------------------------------------
    # QAClient protocol
    # ------------------------------------------------------------------

    async def generate(self, item: Item) -> Prediction:
        retrieval = await self._retrieve(item)
        fast_path = await self._fast_path(item, retrieval)

        # Grounded gate escalates only factoid/list answers (mirrors the
        # pipeline's GROUNDED_CHECK_TYPES); substring grounding is meaningless
        # for one-token labels (mcq/yesno) and structured answers (expression).
        grounded_fail = (
            item.question_type in ("factoid", "list") and not fast_path.grounded
        )
        should_escalate = (
            self.services.force_agent
            or fast_path.logprob < CASCADE_THRESHOLD
            or not fast_path.answer
            or grounded_fail
        )

        # yesno and expression always return the fast-path answer:
        #  - yesno: the agent over-analyses and introduces a documented 'no'
        #    bias (paper §5.2);
        #  - expression: the repair-context fast path (literature base + atlas
        #    repair) IS the complete answer. Escalating would discard the +D
        #    atlas signal and diverge from the production pipeline, which
        #    handles expression outside the cascade entirely.
        if not should_escalate or item.question_type in ("yesno", "expression"):
            return Prediction(
                item_id=item.id,
                answer=fast_path.answer,
                extras={
                    "response_text":   fast_path.response_text,
                    "logprob":         fast_path.logprob,
                    "grounded":        fast_path.grounded,
                    "rewritten_query": retrieval.rewritten_query,
                    "negative_query":  retrieval.negative_query,
                    "stage":           "fast_path",
                },
            )

        agent = await self._agent(item, retrieval, fast_path)
        # Reuse the fast-path pre-fetched tool evidence in re-judgment (avoids a
        # second NCBI round-trip) if the agent did not collect its own.
        if not agent.tool_evidence:
            agent.tool_evidence = fast_path.tool_evidence
        rejudged = await self._rejudge(item, agent, retrieval=retrieval)
        return Prediction(
            item_id=item.id,
            answer=rejudged,
            extras={
                "response_text":     agent.response_text,
                "fast_path_logprob": fast_path.logprob,
                "rewritten_query":   retrieval.rewritten_query,
                "agent_iterations":  agent.iterations,
                "agent_tool_calls":  agent.tool_calls,
                "stage":             "agent_rejudged",
            },
        )

    async def aclose(self) -> None:
        for client in (self._llm, self._embed, self._rerank, self._qdrant, self._atlas):
            if client is None:
                continue
            close = getattr(client, "aclose", None) or getattr(client, "close", None)
            if close is not None:
                result = close()
                if hasattr(result, "__await__"):
                    await result
        self._closed = True

    # ------------------------------------------------------------------
    # Stage hooks (override / monkey-patch points)
    # ------------------------------------------------------------------

    async def _retrieve(self, item: Item) -> RetrievalContext:
        """Multi-source retrieval: question + pseudo-answer + (yesno) negative.

        Three retrieval pools are issued in parallel:

        1. Literal question against the dense index.
        2. Embedding of an LLM-drafted pseudo-answer paragraph (skipped on
           ``factoid``; falls back to the literal question on failure).
        3. For yesno items only: a verbatim "no effect / not effective /
           …" augmentation of the question.

        The three pools are unioned on passage id and reranked by the
        *original* question text, returning the top ``RERANK_TOP_K``
        passages.
        """
        import asyncio

        from bioharness.cascade.retrieval import dense_retrieve, dual_rerank, merge_passages
        from bioharness.cascade.rewrite import (
            SKIP_REWRITE_TYPES,
            negative_evidence_query,
            pseudo_answer_text,
        )

        llm = self._llm_client()
        embed = self._embed_client()
        qdrant = self._qdrant_client()

        pool_each = 30  # matches upstream `_dual_retrieve_rerank` pool size

        # Pool 1: literal question
        tasks = [
            dense_retrieve(
                embed, qdrant, item.question,
                top_k=pool_each, collection=DENSE_COLLECTION,
            ),
        ]

        # Pool 2: pseudo-answer retrieval (skipped on factoid)
        pseudo = ""
        if item.question_type not in SKIP_REWRITE_TYPES:
            pseudo = await pseudo_answer_text(
                llm, model_name=self.services.model_name, item=item,
            )
            tasks.append(
                dense_retrieve(
                    embed, qdrant, pseudo,
                    top_k=pool_each, collection=DENSE_COLLECTION,
                )
            )

        # Pool 3: negative-evidence (yesno only)
        neg_query = negative_evidence_query(item)
        if neg_query:
            tasks.append(
                dense_retrieve(
                    embed, qdrant, neg_query,
                    top_k=pool_each, collection=DENSE_COLLECTION,
                )
            )

        pools = await asyncio.gather(*tasks)
        merged = pools[0]
        for p in pools[1:]:
            merged = merge_passages(merged, p, limit=pool_each * 3)

        # Rerank the merged pool by the ORIGINAL question (not the
        # pseudo-answer) so the top-K reflects question-relevance.
        if merged:
            scores = await dual_rerank(
                self._rerank_client(), item.question, merged,
                top_k=RERANK_TOP_K,
            )
            order = sorted(
                range(len(merged)),
                key=lambda i: -scores[i] if i < len(scores) else 0.0,
            )
            ranked = [(merged[i], scores[i] if i < len(scores) else 0.0) for i in order]
            ranked = ranked[:RERANK_TOP_K]
            passages = [p for p, _ in ranked]
            scores = [s for _, s in ranked]
        else:
            passages, scores = [], []

        return RetrievalContext(
            passages=passages,
            rerank_scores=scores,
            rewritten_query=pseudo,
            negative_query=neg_query,
        )

    async def _fast_path(self, item: Item, ctx: RetrievalContext) -> FastPathOutcome:
        """Constrained generation + logprob + grounded check."""
        from bioharness.cascade.constrained import (
            constrained_generate,
            extract_constrained_answer,
        )

        # Atlas component (D): for expression questions, fetch the gene's HPA
        # tissue expression and inject it as supplementary context (+D). No-op
        # unless enable_atlas is set; fails soft to the -D path.
        atlas_rows = None
        if item.question_type == "expression":
            atlas = self._atlas_client()
            if atlas is not None:
                from bioharness.cascade.atlas import gene_from_expression_question

                gene = gene_from_expression_question(item.question)
                if gene:
                    atlas_rows = await atlas.tissue_rows(gene)

        # Pre-fetch authoritative tool data (gene / SNP / genomics / BLAST) so
        # BOTH the fast path and the agent see it — gene-DB lookups (e.g. SNP,
        # DNA alignment) otherwise answer "unknown" on the fast path.
        from bioharness.tools import precall_tools

        tool_evidence = await precall_tools(item.question, item.question_type)

        text, logprob = await constrained_generate(
            self._llm_client(),
            model_name=self.services.model_name,
            item=item,
            passages=ctx.passages,
            atlas_rows=atlas_rows,
            tool_evidence=tool_evidence,
        )
        normalised = extract_constrained_answer(text, item.question_type, item.options)
        grounded = self._answer_is_grounded(normalised, ctx, item) if normalised else False
        return FastPathOutcome(
            answer=normalised, response_text=text,
            logprob=logprob, grounded=grounded, tool_evidence=tool_evidence,
        )

    async def _agent(
        self, item: Item, ctx: RetrievalContext, fast_path: FastPathOutcome,
    ) -> AgentOutcome:
        """REPL agent escalation hook.

        Resolution order:
          1. If ``BIOHARNESS_PRODUCTION_SRC`` points at a working
             PaperAsKnowledgeGraph-RAG tree, delegate to the full
             production ``BiomedicalRLMPipeline`` (multi-iteration REPL +
             tool dispatch). This is the "exactly aligned" path.
          2. Otherwise fall back to the in-tree single-pass agent that
             ships with this package.
        """
        from bioharness.agent.production_adapter import (
            production_agent_available,
            run_production_agent,
        )
        from bioharness.agent.repl_agent import run_agent

        if production_agent_available():
            return await run_production_agent(
                llm=self._llm_client(),
                services=self.services,
                item=item,
                retrieval=ctx,
                fast_path=fast_path,
            )
        return await run_agent(
            llm=self._llm_client(),
            services=self.services,
            item=item,
            retrieval=ctx,
            fast_path=fast_path,
        )

    async def _rejudge(
        self, item: Item, agent: AgentOutcome,
        retrieval: RetrievalContext | None = None,
    ) -> str:
        """Constrained re-judgment that sees the agent's free-form answer,
        the original retrieved evidence, AND any tool evidence collected
        during escalation. Mirrors the upstream rejudge composition."""
        from bioharness.cascade.constrained import extract_constrained_answer, rejudge

        text = await rejudge(
            self._llm_client(),
            model_name=self.services.model_name,
            item=item,
            agent_text=agent.response_text or agent.answer,
            tool_evidence=agent.tool_evidence,
            retrieval=retrieval,
        )
        return extract_constrained_answer(text, item.question_type, item.options)

    # ------------------------------------------------------------------
    # Grounded gate
    # ------------------------------------------------------------------

    def _answer_is_grounded(
        self, answer: str, ctx: RetrievalContext, item: Item | None = None,
    ) -> bool:
        """Substring-grounded check (relaxed for short labels).

        yesno labels are always considered grounded because the substring
        check is meaningless for one-token answers. mcq labels are
        grounded if the *option text* appears in retrieved evidence.
        Longer answers must literally appear in at least one passage.
        """
        if not answer:
            return False
        qt = item.question_type if item else ""
        if qt == "yesno" or len(answer) <= 3:
            return True
        if not ctx.passages:
            return False
        needle = answer.lower()
        if (
            item and item.options and len(answer) == 1
            and answer.upper() in item.options
        ):
            needle = item.options[answer.upper()].lower()
        for p in ctx.passages:
            text = (p.get("text") or "").lower()
            if needle and needle in text:
                return True
        return False

    # ------------------------------------------------------------------
    # Lazy client constructors
    # ------------------------------------------------------------------

    def _llm_client(self):
        if self._llm is None:
            from bioharness.clients.llm import LLMClient

            self._llm = LLMClient(self.services.llm_url, self.services.api_key)
        return self._llm

    def _embed_client(self):
        if self._embed is None:
            from bioharness.clients.embed import EmbedClient

            self._embed = EmbedClient(self.services.embed_url, self.services.api_key)
        return self._embed

    def _rerank_client(self):
        if self._rerank is None:
            from bioharness.clients.rerank import RerankClient

            self._rerank = RerankClient(self.services.rerank_url, self.services.api_key)
        return self._rerank

    def _qdrant_client(self):
        if self._qdrant is None:
            from bioharness.clients.qdrant import QdrantClient

            self._qdrant = QdrantClient(self.services.qdrant_url)
        return self._qdrant

    def _atlas_client(self):
        """Atlas (D) client, or None when +D is disabled (the default)."""
        if not self.services.enable_atlas:
            return None
        if self._atlas is None:
            from bioharness.clients.atlas import AtlasClient

            self._atlas = AtlasClient(self.services.atlas_url)
        return self._atlas


# ----------------------------------------------------------------------
# Stub for offline / CI smoke
# ----------------------------------------------------------------------


class StubPipelineCascadeClient(PipelineCascadeClient):
    """PipelineCascadeClient that returns a fixed answer.

    Useful for CI runs that exercise the framework-eval → Chi integration
    without touching any live service.
    """

    def __init__(self, *, fixed_answer: str = "yes", **kw: Any) -> None:
        super().__init__(**kw)
        self._fixed_answer = fixed_answer

    async def generate(self, item: Item) -> Prediction:
        from bioharness.cascade.constrained import extract_constrained_answer

        return Prediction(
            item_id=item.id,
            answer=extract_constrained_answer(
                self._fixed_answer, item.question_type, item.options,
            ),
            extras={"stage": "stub", "response_text": self._fixed_answer},
        )

    async def aclose(self) -> None:
        return
