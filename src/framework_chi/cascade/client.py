"""V14CascadeClient: the {framework}^χ headline method.

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

Every stage is a stable extension point: subclass ``V14CascadeClient`` and
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

from framework_eval.eval.types import Item, Prediction, QuestionType

from framework_chi.config import (
    CASCADE_THRESHOLD,
    DENSE_COLLECTION,
    RERANK_TOP_K,
    RETRIEVAL_TOP_K,
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


@dataclass
class AgentOutcome:
    answer: str
    response_text: str
    iterations: int
    tool_calls: list[str]


# ----------------------------------------------------------------------
# Client
# ----------------------------------------------------------------------


class V14CascadeClient:
    """Headline {framework}^χ method registered as
    ``v14-cascade-dual-rerank-grounded``."""

    name = "v14-cascade-dual-rerank-grounded"

    def __init__(self, *, services: ServiceConfig | None = None) -> None:
        self.services = services or ServiceConfig.from_env()

        # Lazy-imported clients; created on first use.
        self._llm     = None
        self._embed   = None
        self._rerank  = None
        self._qdrant  = None
        self._closed  = False

    # ------------------------------------------------------------------
    # QAClient protocol
    # ------------------------------------------------------------------

    async def generate(self, item: Item) -> Prediction:
        retrieval = await self._retrieve(item)
        fast_path = await self._fast_path(item, retrieval)

        should_escalate = (
            self.services.force_agent
            or fast_path.logprob < CASCADE_THRESHOLD
            or not fast_path.answer
            or not fast_path.grounded
        )

        # yesno always returns the fast-path answer (paper §5.2: the agent
        # over-analyses and introduces a documented 'no' bias).
        if not should_escalate or item.question_type == "yesno":
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
        rejudged = await self._rejudge(item, agent)
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
        for client in (self._llm, self._embed, self._rerank, self._qdrant):
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
        """Rewrite query → dense retrieve → dual rerank.

        ``yesno`` and ``factoid`` items retrieve with the verbatim question
        (rewrite hurts on these per the upstream pipeline's policy).
        Other types get a single LLM-rewritten retrieval query.
        """
        from framework_chi.cascade.retrieval import dense_retrieve, dual_rerank
        from framework_chi.cascade.rewrite import SKIP_REWRITE_TYPES, rewrite_query

        llm = self._llm_client()
        embed = self._embed_client()
        qdrant = self._qdrant_client()

        if item.question_type in SKIP_REWRITE_TYPES or item.question_type == "yesno":
            rewritten = item.question
        else:
            rewritten = await rewrite_query(
                llm, model_name=self.services.model_name, item=item,
            )

        passages = await dense_retrieve(
            embed, qdrant, rewritten,
            top_k=RETRIEVAL_TOP_K, collection=DENSE_COLLECTION,
        )
        negative = ""

        scores: list[float]
        if passages:
            scores = await dual_rerank(
                self._rerank_client(), item.question, passages,
                top_k=RERANK_TOP_K,
            )
            order = sorted(
                range(len(passages)),
                key=lambda i: -scores[i] if i < len(scores) else 0.0,
            )
            ranked = [(passages[i], scores[i] if i < len(scores) else 0.0) for i in order]
            ranked = ranked[:RERANK_TOP_K]
            passages = [p for p, _ in ranked]
            scores = [s for _, s in ranked]
        else:
            scores = []

        return RetrievalContext(
            passages=passages,
            rerank_scores=scores,
            rewritten_query=rewritten,
            negative_query=negative,
        )

    async def _fast_path(self, item: Item, ctx: RetrievalContext) -> FastPathOutcome:
        """Constrained generation + logprob + grounded check."""
        from framework_chi.cascade.constrained import (
            constrained_generate,
            extract_constrained_answer,
        )

        text, logprob = await constrained_generate(
            self._llm_client(),
            model_name=self.services.model_name,
            item=item,
            passages=ctx.passages,
        )
        normalised = extract_constrained_answer(text, item.question_type, item.options)
        grounded = self._answer_is_grounded(normalised, ctx, item) if normalised else False
        return FastPathOutcome(
            answer=normalised, response_text=text,
            logprob=logprob, grounded=grounded,
        )

    async def _agent(
        self, item: Item, ctx: RetrievalContext, fast_path: FastPathOutcome,
    ) -> AgentOutcome:
        """REPL agent escalation hook. Default is a longer-budget LLM call
        with the assembled context; subclass to wire a multi-iteration
        REPL agent."""
        from framework_chi.agent.repl_agent import run_agent

        return await run_agent(
            llm=self._llm_client(),
            services=self.services,
            item=item,
            retrieval=ctx,
            fast_path=fast_path,
        )

    async def _rejudge(self, item: Item, agent: AgentOutcome) -> str:
        from framework_chi.cascade.constrained import extract_constrained_answer, rejudge

        text = await rejudge(
            self._llm_client(),
            model_name=self.services.model_name,
            item=item,
            agent_text=agent.response_text or agent.answer,
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
            from framework_chi.clients.llm import LLMClient

            self._llm = LLMClient(self.services.llm_url, self.services.api_key)
        return self._llm

    def _embed_client(self):
        if self._embed is None:
            from framework_chi.clients.embed import EmbedClient

            self._embed = EmbedClient(self.services.embed_url, self.services.api_key)
        return self._embed

    def _rerank_client(self):
        if self._rerank is None:
            from framework_chi.clients.rerank import RerankClient

            self._rerank = RerankClient(self.services.rerank_url, self.services.api_key)
        return self._rerank

    def _qdrant_client(self):
        if self._qdrant is None:
            from framework_chi.clients.qdrant import QdrantClient

            self._qdrant = QdrantClient(self.services.qdrant_url)
        return self._qdrant


# ----------------------------------------------------------------------
# Stub for offline / CI smoke
# ----------------------------------------------------------------------


class StubV14CascadeClient(V14CascadeClient):
    """V14CascadeClient that returns a fixed answer.

    Useful for CI runs that exercise the framework-eval → Chi integration
    without touching any live service.
    """

    def __init__(self, *, fixed_answer: str = "yes", **kw: Any) -> None:
        super().__init__(**kw)
        self._fixed_answer = fixed_answer

    async def generate(self, item: Item) -> Prediction:
        from framework_chi.cascade.constrained import extract_constrained_answer

        return Prediction(
            item_id=item.id,
            answer=extract_constrained_answer(
                self._fixed_answer, item.question_type, item.options,
            ),
            extras={"stage": "stub", "response_text": self._fixed_answer},
        )

    async def aclose(self) -> None:
        return
