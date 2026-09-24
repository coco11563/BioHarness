"""
Model client protocol and implementations.

Defines the abstract interface for LLM and RAG backends,
plus concrete implementations for OpenAI-compatible APIs.
"""

import asyncio
import re
import time
import unicodedata
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

from openai import AsyncOpenAI


@dataclass
class ModelResponse:
    """Response from a model backend.

    Attributes:
        answer: Extracted answer for evaluation
        response_text: Full response text for logging
        latency_ms: Response time in milliseconds
        metadata: Optional extra info (e.g., RAG retrieval info)
    """
    answer: str
    response_text: str
    latency_ms: float
    metadata: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "answer": self.answer,
            "response_text": self.response_text,
            "latency_ms": self.latency_ms,
            "metadata": self.metadata,
        }


class ModelClient(ABC):
    """Abstract interface for LLM/RAG backends.

    Subclasses must implement the generate() method.
    """

    @abstractmethod
    async def generate(
        self,
        question: str,
        question_type: str,
        context: list[str] | None = None,
        options: dict[str, str] | None = None,
        *,
        item_key: tuple[str, str] | None = None,
    ) -> ModelResponse:
        """Generate answer for a question.

        Args:
            question: The question text
            question_type: Type of question (yesno, mcq, etc.)
            context: Optional context passages
            options: MCQ options dict
            item_key: Optional (dataset, item_id) tuple for per-item lookups
                     (e.g., router-extracted entities). Most clients ignore.

        Returns:
            ModelResponse with answer and metadata
        """
        ...

    async def __aenter__(self):
        """Async context manager entry."""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        pass


class LLMClient(ModelClient):
    """OpenAI-compatible LLM client with load balancing.

    Supports multiple backends with round-robin load balancing.

    Example:
        client = LLMClient(
            backends=["http://127.0.0.1:8005/v1", "http://127.0.0.1:8006/v1"],
            model="qwen",
        )
        async with client:
            response = await client.generate(
                question="Is aspirin a drug?",
                question_type="yesno",
            )
    """

    def __init__(
        self,
        backends: list[str],
        model: str = "qwen",
        api_key: str = "EMPTY",
        max_tokens: int = 512,
        temperature: float = 0.1,
    ):
        """Initialize LLM client.

        Args:
            backends: List of API base URLs
            model: Model name to use
            api_key: API key (default "EMPTY" for local vLLM)
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
        """
        self.backends = backends
        self.model = model
        self.api_key = api_key
        self.max_tokens = max_tokens
        self.temperature = temperature
        self._clients: list[AsyncOpenAI] = []
        self._backend_index = 0
        self._lock = asyncio.Lock()

    @staticmethod
    def _normalize_response_text(text: str) -> str:
        """Normalize model text before structured extraction."""
        normalized = unicodedata.normalize("NFKC", text or "")
        normalized = re.sub(r"</?think>", "\n", normalized, flags=re.IGNORECASE)
        return normalized.strip()

    async def __aenter__(self):
        """Initialize async clients."""
        self._clients = [
            AsyncOpenAI(base_url=url, api_key=self.api_key)
            for url in self.backends
        ]
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Close async clients."""
        for client in self._clients:
            await client.close()
        self._clients = []

    async def _get_next_client(self) -> AsyncOpenAI:
        """Get next client in round-robin fashion."""
        async with self._lock:
            client = self._clients[self._backend_index]
            self._backend_index = (self._backend_index + 1) % len(self._clients)
            return client

    def _build_prompt(
        self,
        question: str,
        question_type: str,
        context: list[str] | None = None,
        options: dict[str, str] | None = None,
    ) -> str:
        """Build prompt based on question type."""
        parts = []

        # Add context if provided
        if context:
            parts.append("Context:")
            for ctx in context:
                parts.append(ctx)
            parts.append("")

        # Add question
        parts.append(f"Question: {question}")

        # Add options for MCQ
        if options:
            parts.append("\nOptions:")
            for letter, text in sorted(options.items()):
                parts.append(f"  {letter}. {text}")

        # Add type-specific instructions
        if question_type == "yesno":
            parts.append("\nAnswer with exactly one lowercase word on the first line: 'yes', 'no', or 'maybe'. Do not include reasoning.")
        elif question_type in ("mcq", "mcq_multi"):
            if question_type == "mcq_multi":
                parts.append("\nSelect all correct options. Answer with letter(s) only (e.g., A, C, D).")
            else:
                parts.append("\nAnswer with the letter of the correct option only (A, B, C, D, or E).")
        elif question_type == "factoid":
            parts.append("\nProvide a short, factual answer.")
        elif question_type == "list":
            # Check if this is a GO annotation question
            if "Gene Ontology" in question or "GO annotation" in question.lower():
                parts.append("\nList Gene Ontology annotations using the exact format with prefixes:")
                parts.append("  - 'enables <function>'")
                parts.append("  - 'located_in <location>'")
                parts.append("  - 'involved_in <process>'")
                parts.append("  - 'is_active_in <location>'")
                parts.append("  - 'acts_upstream_of_or_within <process>'")
                parts.append("Answer as a comma-separated list, e.g.: enables protein binding, located_in cytoplasm, involved_in cell cycle")
            else:
                parts.append("\nList all relevant items, separated by commas.")
        elif question_type == "summary":
            parts.append("\nProvide a comprehensive summary.")
        elif question_type == "expression":
            parts.append("\nList the tissues where this gene is expressed, separated by commas.")

        return "\n".join(parts)

    def _extract_answer(self, response_text: str, question_type: str) -> str:
        """Extract structured answer from response text."""
        text = self._normalize_response_text(response_text)

        if question_type == "yesno":
            text_lower = text.lower()
            lines = [line.strip() for line in text_lower.splitlines() if line.strip()]

            for line in reversed(lines[-5:]):
                if line in ("yes", "no", "maybe"):
                    return line
                labeled = re.search(r"(?:final answer|answer)\s*[:\-]\s*(yes|no|maybe)\b", line)
                if labeled:
                    return labeled.group(1)
                leading = re.match(r"^(yes|no|maybe)\b", line)
                if leading:
                    return leading.group(1)

            first_word = text_lower.split()[0] if text_lower.split() else ""
            if first_word in ("yes", "no", "maybe"):
                return first_word

            matches = re.findall(r"\b(yes|no|maybe)\b", text_lower)
            if matches:
                return matches[-1]

            return text_lower[:20]

        if question_type == "mcq":
            # Extract single letter
            match = re.search(r'\b([A-E])\b', text.upper())
            if match:
                return match.group(1)
            # Check if response starts with letter
            if text and text[0].upper() in "ABCDE":
                return text[0].upper()
            return text[:1].upper() if text else ""

        if question_type == "mcq_multi":
            # Extract multiple letters
            letters = re.findall(r'\b([A-E])\b', text.upper())
            if letters:
                return str(sorted(set(letters)))
            return "[]"

        # For other types, return cleaned text
        return text

    async def generate(
        self,
        question: str,
        question_type: str,
        context: list[str] | None = None,
        options: dict[str, str] | None = None,
    ) -> ModelResponse:
        """Generate answer for a question."""
        prompt = self._build_prompt(question, question_type, context, options)

        client = await self._get_next_client()

        start_time = time.perf_counter()
        try:
            response = await client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                extra_body={"chat_template_kwargs": {"enable_thinking": False}},
            )
            response_text = response.choices[0].message.content or ""
        except Exception as e:
            response_text = f"ERROR: {str(e)}"

        latency_ms = (time.perf_counter() - start_time) * 1000

        answer = self._extract_answer(response_text, question_type)

        return ModelResponse(
            answer=answer,
            response_text=response_text,
            latency_ms=latency_ms,
            metadata={"model": self.model},
        )


class RAGClient(ModelClient):
    """Abstract base for RAG system clients.

    Subclass this and implement generate() for your RAG system.

    Example:
        class MyRAGClient(RAGClient):
            async def generate(self, question, question_type, context=None, options=None):
                # Call your RAG API
                result = await self.rag_api.query(question)
                return ModelResponse(
                    answer=result.answer,
                    response_text=result.full_response,
                    latency_ms=result.latency,
                    metadata={"sources": result.source_count},
                )
    """

    def __init__(self, endpoint: str, **kwargs):
        """Initialize RAG client.

        Args:
            endpoint: RAG system API endpoint
            **kwargs: Additional configuration
        """
        self.endpoint = endpoint
        self.config = kwargs

    async def generate(
        self,
        question: str,
        question_type: str,
        context: list[str] | None = None,
        options: dict[str, str] | None = None,
    ) -> ModelResponse:
        """Generate answer using RAG system.

        Override this method in your subclass.
        """
        raise NotImplementedError("Subclass must implement generate()")


@dataclass
class ClientConfig:
    """Configuration for model clients."""
    client_type: str = "llm"  # "llm" or "rag"
    backends: list[str] = field(default_factory=lambda: ["http://127.0.0.1:8005/v1"])
    model: str = "qwen"
    api_key: str = "EMPTY"
    max_tokens: int = 512
    temperature: float = 0.1
    rag_endpoint: str | None = None

    def create_client(self) -> ModelClient:
        """Create client instance from config."""
        if self.client_type == "llm":
            return LLMClient(
                backends=self.backends,
                model=self.model,
                api_key=self.api_key,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
            )
        elif self.client_type == "rag":
            if not self.rag_endpoint:
                raise ValueError("rag_endpoint required for RAG client")
            return RAGClient(endpoint=self.rag_endpoint)
        else:
            raise ValueError(f"Unknown client type: {self.client_type}")
