"""V14CascadeClient: the {framework}^χ headline method.

Pipeline shape (matches the paper):

    item -> tool pre-call -> dense retrieval -> dual rerank
         -> constrained generation (max_tokens=4 + logprob)
         -> if logprob > threshold and (not grounded_gate or grounded):
              return fast-path answer
            else:
              REPL agent escalation (max_iterations=8)
              -> constrained re-judgment (max_tokens=4)
              -> return re-judged answer

Each stage is a small private method so unit tests can exercise it in
isolation. The class implements ``framework_eval.plugins.QAClient``.

The implementation here is deliberately a faithful skeleton: every stage
is exposed as a hook that can be subclassed or monkey-patched, and the
default deterministic-only paths return reasonable fallbacks if the
backing service is unavailable. To reach the published headline numbers
on the full 19,302-item benchmark you must connect a live LLM stack
(see ``docs/infra.md``); cached-smoke offline mode covers a 50-item
slice for CI.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

from framework_eval.eval.extraction import extract_answer
from framework_eval.eval.types import Item, Prediction, QuestionType

from framework_chi.config import CascadeOptions, ServiceConfig

LOGGER = logging.getLogger(__name__)


# ----------------------------------------------------------------------
# Stage records
# ----------------------------------------------------------------------


@dataclass
class RetrievalContext:
    passages: list[dict[str, Any]] = field(default_factory=list)
    rerank_scores: list[float]     = field(default_factory=list)


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

    def __init__(
        self,
        *,
        services: ServiceConfig | None = None,
        options: CascadeOptions | None = None,
    ) -> None:
        self.services = services or ServiceConfig.from_env()
        self.options  = options or CascadeOptions()

        # Lazy-imported clients; created on first use so the constructor
        # stays cheap and the framework-eval ``framework-eval list-methods``
        # call does not require a live inference stack.
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
            fast_path.logprob < self.options.cascade_threshold
            or (self.options.enable_grounded_gate and not fast_path.grounded)
        )

        if not should_escalate or item.question_type == "yesno":
            # yesno always uses the fast path (agent over-analyses; see paper §5.2)
            return Prediction(
                item_id=item.id,
                answer=fast_path.answer,
                extras={
                    "response_text": fast_path.response_text,
                    "logprob": fast_path.logprob,
                    "grounded": fast_path.grounded,
                    "stage": "fast_path",
                },
            )

        agent = await self._agent(item, retrieval, fast_path)
        rejudged = await self._rejudge(item, agent)
        return Prediction(
            item_id=item.id,
            answer=rejudged,
            extras={
                "response_text": agent.response_text,
                "fast_path_logprob": fast_path.logprob,
                "agent_iterations": agent.iterations,
                "agent_tool_calls": agent.tool_calls,
                "stage": "agent_rejudged",
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
        """Dense retrieval + optional dual-rerank pass."""
        from framework_chi.cascade.retrieval import dense_retrieve, dual_rerank

        passages = await dense_retrieve(
            self._embed_client(), self._qdrant_client(),
            item.question, top_k=self.options.retrieval_top_k,
        )
        if self.options.enable_dual_rerank and passages:
            scores = await dual_rerank(
                self._rerank_client(), item.question, passages,
                top_k=self.options.rerank_top_k,
            )
        else:
            scores = [1.0] * len(passages[: self.options.rerank_top_k])
            passages = passages[: self.options.rerank_top_k]
        return RetrievalContext(passages=passages, rerank_scores=scores)

    async def _fast_path(self, item: Item, ctx: RetrievalContext) -> FastPathOutcome:
        """Constrained generation with logprob; returns the routing signal."""
        from framework_chi.cascade.constrained import constrained_generate

        text, logprob = await constrained_generate(
            self._llm_client(),
            model_name=self.services.model_name,
            item=item,
            passages=ctx.passages,
        )
        normalised = extract_answer(text, item.question_type, item.options, mode="strict")
        grounded = self._answer_is_grounded(normalised, ctx) if normalised else False
        return FastPathOutcome(
            answer=normalised, response_text=text,
            logprob=logprob, grounded=grounded,
        )

    async def _agent(
        self, item: Item, ctx: RetrievalContext, fast_path: FastPathOutcome,
    ) -> AgentOutcome:
        """REPL agent escalation. Default returns the fast-path answer when
        the agent module is not configured; override to plug in the full
        ``BiomedicalRLMPipeline`` from your infrastructure repository."""
        from framework_chi.agent.repl_agent import run_agent

        return await run_agent(
            llm=self._llm_client(),
            services=self.services,
            options=self.options,
            item=item,
            retrieval=ctx,
            fast_path=fast_path,
        )

    async def _rejudge(self, item: Item, agent: AgentOutcome) -> str:
        """Constrained re-judgment from agent's free-form answer."""
        from framework_chi.cascade.constrained import rejudge

        text = await rejudge(
            self._llm_client(),
            model_name=self.services.model_name,
            item=item,
            agent_text=agent.response_text or agent.answer,
        )
        return extract_answer(text, item.question_type, item.options, mode="strict")

    # ------------------------------------------------------------------
    # Grounded gate
    # ------------------------------------------------------------------

    @staticmethod
    def _answer_is_grounded(answer: str, ctx: RetrievalContext) -> bool:
        if not answer or not ctx.passages:
            return False
        needle = answer.lower()
        for p in ctx.passages:
            text = (p.get("text") or "").lower()
            if needle in text:
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


# Convenience: a no-op client for offline tests.

class StubV14CascadeClient(V14CascadeClient):
    """V14CascadeClient that always returns a fixed answer.

    Useful for CI smoke runs that exercise the framework-eval -> Chi
    integration without touching any live service.
    """

    def __init__(self, *, fixed_answer: str = "yes", **kw: Any) -> None:
        super().__init__(**kw)
        self._fixed_answer = fixed_answer

    async def generate(self, item: Item) -> Prediction:
        return Prediction(
            item_id=item.id,
            answer=extract_answer(
                self._fixed_answer, item.question_type, item.options, mode="strict",
            ),
            extras={"stage": "stub", "response_text": self._fixed_answer},
        )

    async def aclose(self) -> None:
        return
