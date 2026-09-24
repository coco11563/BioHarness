"""RLMModelClient - ModelClient implementation for RLM-based QA.

Wraps BiomedicalRLMPipeline to conform to the existing benchmark
ModelClient protocol, enabling use with BenchmarkRunner.

Benchmark Protocol (from benchmark/code/README.md):
- ModelClient.generate() returns ModelResponse
- answer: Extracted answer in benchmark-compliant format
- response_text: Full response for logging
- latency_ms: Response time
- metadata: Optional RAG-specific info

DESIGN: Fail-Fast
- Errors propagate immediately
- No mock data or fallbacks
"""

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any


@dataclass
class ModelResponse:
    """Response from a model backend."""
    answer: str
    response_text: str
    latency_ms: float
    metadata: dict[str, Any] | None = None


class ModelClient(ABC):
    """Abstract interface for LLM/RAG backends."""

    @abstractmethod
    async def generate(
        self,
        question: str,
        question_type: str,
        context: list[str] | None = None,
        options: dict[str, str] | None = None,
    ) -> ModelResponse:
        """Generate answer for a question."""
        ...

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        pass


from ..config import get_config
from ..rlm.pipeline import BiomedicalRLMPipeline, PipelineConfig
from ..rlm.tools import shutdown_tools
from ..utils.cache import BenchmarkCache
from ..utils.answer_extraction import extract_answer


class RLMModelClient(ModelClient):
    """ModelClient implementation using RLM + BiomedicalREPL.

    Wraps BiomedicalRLMPipeline for use with the existing benchmark
    infrastructure. Supports configurable max_iterations and max_depth
    for ablation experiments.

    Usage:
        client = RLMModelClient(max_iterations=30, max_depth=1)
        async with client:
            response = await client.generate(
                question="Does metformin help with diabetes?",
                question_type="yesno",
            )
            print(response.answer)  # "yes"

    For ablation:
        # Test with different iteration limits
        for iters in [5, 10, 20, 30]:
            client = RLMModelClient(max_iterations=iters)
            # ... run benchmark
    """

    def __init__(
        self,
        max_iterations: int = 30,
        max_depth: int = 1,
        enable_kg_tools: bool = True,
        model_name: str = "qwen",
        base_url: str | None = None,
        cache: BenchmarkCache | None = None,
    ):
        """Initialize RLM model client.

        Args:
            max_iterations: RLM max iterations (for ablation)
            max_depth: RLM recursive depth (for ablation)
            enable_kg_tools: Whether to enable KG tools
            model_name: LLM model name
            base_url: LLM API base URL
            cache: Optional shared cache for traces
        """
        self.max_iterations = max_iterations
        self.max_depth = max_depth
        self.enable_kg_tools = enable_kg_tools
        self.model_name = model_name
        self.base_url = base_url or get_config().llm.endpoints[0].url

        self._cache = cache or BenchmarkCache()
        self._pipeline: BiomedicalRLMPipeline | None = None

    async def __aenter__(self):
        """Initialize pipeline on context entry."""
        config = PipelineConfig(
            max_iterations=self.max_iterations,
            max_depth=self.max_depth,
            model_name=self.model_name,
            base_url=self.base_url,
            enable_kg_tools=self.enable_kg_tools,
            enable_cache=True,
        )
        self._pipeline = BiomedicalRLMPipeline(config=config, cache=self._cache)
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Cleanup on context exit - release all resources."""
        self._pipeline = None
        # Shutdown tool singletons (DB pools, Qdrant clients, etc.)
        shutdown_tools()

    async def generate(
        self,
        question: str,
        question_type: str,
        context: list[str] | None = None,
        options: dict[str, str] | None = None,
    ) -> ModelResponse:
        """Generate answer using RLM pipeline.

        Args:
            question: The question text
            question_type: Type of question (yesno, mcq, etc.)
            context: Optional context passages (added to prompt)
            options: MCQ options dict

        Returns:
            ModelResponse with answer and metadata
        """
        if self._pipeline is None:
            raise RuntimeError("RLMModelClient not initialized. Use 'async with' context.")

        # Build context dict if provided
        extra_context = None
        if context or options:
            extra_context = {}
            if context:
                extra_context["context_passages"] = context
            if options:
                extra_context["options"] = options

        start_time = time.perf_counter()

        # Run pipeline (sync, will be run in executor)
        result = await self._pipeline.answer_async(
            question=question,
            question_type=question_type,
            context=extra_context,
        )

        latency_ms = (time.perf_counter() - start_time) * 1000

        # Extract answer using centralized extraction (benchmark compliant)
        answer = extract_answer(result.answer, question_type, options)

        return ModelResponse(
            answer=answer,
            response_text=result.answer,
            latency_ms=latency_ms,
            metadata={
                "iterations_used": result.iterations_used,
                "max_iterations": self.max_iterations,
                "max_depth": self.max_depth,
                "enable_kg_tools": self.enable_kg_tools,
            },
        )

    def get_cache(self) -> BenchmarkCache:
        """Get the cache for trace analysis."""
        return self._cache

    def get_config_summary(self) -> dict[str, Any]:
        """Get configuration summary for reporting."""
        return {
            "client_type": "rlm",
            "max_iterations": self.max_iterations,
            "max_depth": self.max_depth,
            "enable_kg_tools": self.enable_kg_tools,
            "model_name": self.model_name,
            "base_url": self.base_url,
        }


# =============================================================================
# Direct Answer Client - No Retrieval (for MCQ where LLM knowledge is sufficient)
# =============================================================================

# MCQ-specific prompt template
MCQ_DIRECT_PROMPT = """You are a medical expert answering multiple choice questions.

Question: {question}

Options:
{options_text}

Instructions:
- Use your medical knowledge to select the best answer
- Consider all options carefully before deciding
- Output ONLY the letter of your answer (A, B, C, D, or E)
- Do not explain your reasoning

Answer:"""


class DirectAnswerClient(ModelClient):
    """ModelClient that uses direct LLM without retrieval.

    Optimal for MCQ questions where:
    - Model's parametric knowledge is sufficient
    - Retrieval introduces noise that hurts accuracy

    Based on experimental findings:
    - medmcqa: -3~5% with retrieval → direct is better
    - medqa_us: -3~5% with retrieval → direct is better

    Usage:
        client = DirectAnswerClient()
        async with client:
            response = await client.generate(
                question="Which drug is a selective COX-2 inhibitor?",
                question_type="mcq",
                options={"A": "Aspirin", "B": "Celecoxib", "C": "Ibuprofen", "D": "Naproxen"}
            )
            print(response.answer)  # "B"
    """

    def __init__(
        self,
        model_name: str = "qwen",
        base_url: str | None = None,
        temperature: float = 0.0,
    ):
        self.model_name = model_name
        self.base_url = base_url or get_config().llm.endpoints[0].url
        self.temperature = temperature
        self._client = None

    async def __aenter__(self):
        import httpx
        from openai import AsyncOpenAI

        self._client = AsyncOpenAI(
            base_url=self.base_url,
            api_key="EMPTY",
            http_client=httpx.AsyncClient(timeout=60.0),
        )
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self._client:
            await self._client.close()
            self._client = None

    async def generate(
        self,
        question: str,
        question_type: str,
        context: list[str] | None = None,
        options: dict[str, str] | None = None,
    ) -> ModelResponse:
        """Generate answer using direct LLM (no retrieval)."""
        if self._client is None:
            raise RuntimeError("DirectAnswerClient not initialized. Use 'async with' context.")

        start_time = time.perf_counter()

        # Format options for MCQ
        if options:
            options_text = "\n".join(f"{k}: {v}" for k, v in sorted(options.items()))
        else:
            options_text = "(No options provided)"

        # Build prompt
        if question_type in ("mcq", "mcq_multi"):
            prompt = MCQ_DIRECT_PROMPT.format(
                question=question,
                options_text=options_text,
            )
        else:
            # Fallback for other question types
            prompt = f"Question: {question}\n\nProvide a concise answer:"

        # Call LLM directly
        response = await self._client.chat.completions.create(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=self.temperature,
            max_tokens=50,  # MCQ answers are short
        )

        raw_answer = response.choices[0].message.content.strip()
        latency_ms = (time.perf_counter() - start_time) * 1000

        # Extract answer
        answer = extract_answer(raw_answer, question_type, options)

        return ModelResponse(
            answer=answer,
            response_text=raw_answer,
            latency_ms=latency_ms,
            metadata={
                "mode": "direct_answer",
                "model_name": self.model_name,
                "retrieval_skipped": True,
            },
        )

    def get_config_summary(self) -> dict[str, Any]:
        return {
            "client_type": "direct_answer",
            "model_name": self.model_name,
            "temperature": self.temperature,
        }


# =============================================================================
# Adaptive Client - Routes based on question type
# =============================================================================

# Question types that should skip retrieval (based on experimental evidence)
SKIP_RETRIEVAL_QUESTION_TYPES = {"mcq", "mcq_multi"}

# =============================================================================
# Auto Question Type Detection
# =============================================================================

import re

# MCQ detection patterns
MCQ_PATTERNS = [
    # Explicit choice patterns
    r"which of the following",
    r"all of the following",  # covers "all of the following EXCEPT"
    r"which one of",
    r"select the (best|correct|most)",
    r"choose the (best|correct|most)",
    r"the following .* except",
    # Option markers in question
    r"\b[A-E]\)\s*\w",  # A) something
    r"\b[A-E]\.\s*\w",  # A. something
    r"\([A-E]\)\s*\w",  # (A) something
]

# YesNo detection patterns
YESNO_PATTERNS = [
    r"^(is|are|does|do|can|could|will|would|should|has|have|was|were)\s",
    r"\?$",  # ends with question mark + starts with verb
]


def detect_question_type(
    question: str,
    options: dict[str, str] | None = None,
) -> str:
    """Auto-detect question type from question text and options.

    Returns:
        "mcq" - Multiple choice (skip retrieval)
        "yesno" - Yes/No question (use retrieval)
        "factoid" - Factual question (use retrieval)
        "unknown" - Cannot determine (use retrieval as fallback)
    """
    q_lower = question.lower().strip()

    # 1. If options dict provided with multiple choices → MCQ
    if options and len(options) >= 2:
        return "mcq"

    # 2. Check for MCQ patterns in question text
    for pattern in MCQ_PATTERNS:
        if re.search(pattern, q_lower, re.IGNORECASE):
            return "mcq"

    # 3. Check for embedded options (A. xxx B. xxx format)
    option_count = len(re.findall(r"\b[A-E][.)]\s*\w", question))
    if option_count >= 2:
        return "mcq"

    # 4. Check for YesNo patterns
    for pattern in YESNO_PATTERNS:
        if re.match(pattern, q_lower, re.IGNORECASE):
            # Verify it's a polar question (yes/no answerable)
            if q_lower.endswith("?"):
                # Check if it starts with auxiliary verb
                aux_verbs = ["is", "are", "does", "do", "can", "could", "will",
                            "would", "should", "has", "have", "was", "were"]
                first_word = q_lower.split()[0] if q_lower.split() else ""
                if first_word in aux_verbs:
                    return "yesno"

    # 5. What/Who/Where/When/How questions → factoid
    wh_words = ["what", "who", "where", "when", "how", "why", "which"]
    first_word = q_lower.split()[0] if q_lower.split() else ""
    if first_word in wh_words and "following" not in q_lower:
        return "factoid"

    # 6. Default: unknown (will use retrieval as safe fallback)
    return "unknown"


def should_skip_retrieval_auto(
    question: str,
    question_type: str | None = None,
    options: dict[str, str] | None = None,
) -> tuple[bool, str]:
    """Determine if retrieval should be skipped, with auto-detection.

    Args:
        question: The question text
        question_type: Optional explicit type (if known)
        options: MCQ options dict (if available)

    Returns:
        (skip_retrieval: bool, detected_type: str)
    """
    # If explicit type provided and it's MCQ → skip
    if question_type and question_type.lower() in SKIP_RETRIEVAL_QUESTION_TYPES:
        return True, question_type.lower()

    # Auto-detect from question
    detected = detect_question_type(question, options)

    # MCQ → skip retrieval (LLM knowledge is better)
    if detected == "mcq":
        return True, detected

    # Everything else → use retrieval
    return False, detected or question_type or "unknown"


class AdaptiveModelClient(ModelClient):
    """Adaptive client that routes questions to optimal backend.

    Routing strategy (based on experimental findings):
    - MCQ/MCQ_Multi → DirectAnswerClient (skip retrieval)
    - YesNo, Factoid, List → RLMModelClient (use retrieval)

    Experimental evidence:
    - pubmedqa (yesno): +14.2% with RT-KG → use retrieval
    - geneturing: +4.0% with RT-KG → use retrieval
    - medmcqa (mcq): -3~5% with RT-KG → skip retrieval
    - medqa_us (mcq): -3~5% with RT-KG → skip retrieval

    Usage:
        # Mode 1: Auto-detect question type (recommended)
        client = AdaptiveModelClient(auto_detect=True)
        async with client:
            # RLM auto-detects: MCQ → skip retrieval, YesNo → use retrieval
            response = await client.generate(question)

        # Mode 2: Explicit question type
        client = AdaptiveModelClient(auto_detect=False)
        async with client:
            response = await client.generate(question, "mcq", options=opts)
    """

    def __init__(
        self,
        # RLM settings (for retrieval-based questions)
        max_iterations: int = 30,
        max_depth: int = 1,
        enable_kg_tools: bool = True,
        # Shared settings
        model_name: str = "qwen",
        base_url: str | None = None,
        # Adaptive settings
        auto_detect: bool = True,  # Auto-detect question type
        skip_retrieval_types: set[str] | None = None,
    ):
        self.max_iterations = max_iterations
        self.max_depth = max_depth
        self.enable_kg_tools = enable_kg_tools
        self.model_name = model_name
        self.base_url = base_url or get_config().llm.endpoints[0].url

        # Auto-detection mode
        self.auto_detect = auto_detect

        # Which question types should skip retrieval
        self.skip_retrieval_types = skip_retrieval_types or SKIP_RETRIEVAL_QUESTION_TYPES

        self._rlm_client: RLMModelClient | None = None
        self._direct_client: DirectAnswerClient | None = None

        # Statistics
        self._stats = {
            "direct_count": 0,
            "rlm_count": 0,
            "auto_detected_types": {},  # Track detected types
        }

    async def __aenter__(self):
        # Initialize both clients
        self._rlm_client = RLMModelClient(
            max_iterations=self.max_iterations,
            max_depth=self.max_depth,
            enable_kg_tools=self.enable_kg_tools,
            model_name=self.model_name,
            base_url=self.base_url,
        )
        await self._rlm_client.__aenter__()

        self._direct_client = DirectAnswerClient(
            model_name=self.model_name,
            base_url=self.base_url,
        )
        await self._direct_client.__aenter__()

        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self._rlm_client:
            await self._rlm_client.__aexit__(exc_type, exc_val, exc_tb)
            self._rlm_client = None
        if self._direct_client:
            await self._direct_client.__aexit__(exc_type, exc_val, exc_tb)
            self._direct_client = None

    def _should_skip_retrieval(self, question_type: str) -> bool:
        """Determine if this question type should skip retrieval."""
        return question_type.lower() in self.skip_retrieval_types

    async def generate(
        self,
        question: str,
        question_type: str = "auto",  # "auto" triggers auto-detection
        context: list[str] | None = None,
        options: dict[str, str] | None = None,
    ) -> ModelResponse:
        """Route to appropriate backend based on question type.

        Args:
            question: The question text
            question_type: Question type. Use "auto" for auto-detection.
            context: Optional context passages
            options: MCQ options dict

        Returns:
            ModelResponse with answer and routing metadata
        """
        # Determine routing strategy
        if self.auto_detect or question_type == "auto":
            # Auto-detect question type from question text
            skip_retrieval, detected_type = should_skip_retrieval_auto(
                question=question,
                question_type=question_type if question_type != "auto" else None,
                options=options,
            )
            # Track detection stats
            self._stats["auto_detected_types"][detected_type] = \
                self._stats["auto_detected_types"].get(detected_type, 0) + 1
            effective_type = detected_type
        else:
            # Use explicit type
            skip_retrieval = self._should_skip_retrieval(question_type)
            effective_type = question_type

        if skip_retrieval:
            # MCQ → Direct answer (no retrieval)
            self._stats["direct_count"] += 1
            response = await self._direct_client.generate(
                question=question,
                question_type=effective_type,
                context=context,
                options=options,
            )
            # Add routing info to metadata
            response.metadata["routing"] = {
                "mode": "direct",
                "detected_type": effective_type,
                "auto_detected": self.auto_detect or question_type == "auto",
            }
            return response
        else:
            # YesNo, Factoid, etc. → RLM with retrieval
            self._stats["rlm_count"] += 1
            response = await self._rlm_client.generate(
                question=question,
                question_type=effective_type,
                context=context,
                options=options,
            )
            # Add routing info to metadata
            response.metadata["routing"] = {
                "mode": "rlm",
                "detected_type": effective_type,
                "auto_detected": self.auto_detect or question_type == "auto",
            }
            return response

    def get_cache(self) -> BenchmarkCache:
        """Get RLM cache for trace analysis."""
        return self._rlm_client.get_cache() if self._rlm_client else BenchmarkCache()

    def get_config_summary(self) -> dict[str, Any]:
        return {
            "client_type": "adaptive",
            "skip_retrieval_types": list(self.skip_retrieval_types),
            "max_iterations": self.max_iterations,
            "max_depth": self.max_depth,
            "enable_kg_tools": self.enable_kg_tools,
            "model_name": self.model_name,
            "routing_stats": self._stats.copy(),
        }
