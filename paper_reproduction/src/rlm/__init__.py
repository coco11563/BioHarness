"""RLM Integration Layer for Biomedical QA.

RLM Philosophy:
- RLM executes code to manage long-context retrieval and reasoning
- BiomedicalREPL extends LocalREPL with domain-specific tools
- Tools are injected into REPL globals for code execution
- Each tool performs ONE clear operation (single responsibility)

This package provides:
- BiomedicalREPL: Extended LocalREPL with retrieval/KG tools
- BiomedicalRLMPipeline: High-level orchestration with caching/tracing
- RETRIEVAL_TOOLS: Search, filter, ranking, structure tools
- KG_TOOLS: LightRAG, PathRAG, MS-GraphRAG wrappers

Usage:
    from src.rlm import BiomedicalRLMPipeline, BiomedicalREPL

    # High-level API
    pipeline = BiomedicalRLMPipeline(max_iterations=30, max_depth=1)
    result = pipeline.answer(question, question_type="yesno")

    # Low-level REPL access
    repl = BiomedicalREPL(enable_kg_tools=True)
    repl.setup()
    result = repl.execute("papers = search_papers('metformin diabetes')")
"""

from .biomedical_repl import BiomedicalREPL
from .pipeline import BiomedicalRLMPipeline, PipelineConfig, PipelineResult
from .tools import RETRIEVAL_TOOLS
from .kg_tools import KG_TOOLS

__all__ = [
    "BiomedicalREPL",
    "BiomedicalRLMPipeline",
    "PipelineConfig",
    "PipelineResult",
    "RETRIEVAL_TOOLS",
    "KG_TOOLS",
]
