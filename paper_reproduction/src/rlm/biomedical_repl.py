"""BiomedicalREPL - Extended LocalREPL with biomedical retrieval tools.

RLM Philosophy:
- RLM executes code to manage long-context retrieval and reasoning
- BiomedicalREPL extends LocalREPL with domain-specific tools
- Tools are injected into REPL globals for code execution
- Each tool performs ONE clear operation (single responsibility)

This module provides:
- Retrieval tools (search_papers, rerank_papers, search_chunks, etc.)
- Filter tools (search_mesh_terms, keyword_search, get_pmids_by_mesh)
- Structure tools (get_paper_sections, get_paper_abstracts)
- KG tools (kg_lightrag_query, kg_pathrag_query, kg_graphrag_query)
- Helper tools (format_papers, format_chunks, extract_answer)

DESIGN: Fail-Fast
- All tool errors propagate immediately
- No fallback or mock data

CONCURRENCY FIX:
- LocalREPL's _temp_cwd() uses os.chdir() which is process-wide and NOT thread-safe
- When running with concurrency > 1, multiple threads can race on chdir()
- BiomedicalREPL overrides _temp_cwd() to be a no-op since our tools use absolute paths
"""

import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Any

# Add RLM to path (needed for LocalREPL base class)
_rlm_path = Path(__file__).parent.parent.parent / ".ref_project" / "rlm"
if str(_rlm_path) not in sys.path:
    sys.path.insert(0, str(_rlm_path))

from rlm.environments.local_repl import LocalREPL

from .tools import RETRIEVAL_TOOLS
from .kg_tools import KG_TOOLS
from ..utils.answer_extraction import extract_answer


class BiomedicalREPL(LocalREPL):
    """Extended LocalREPL with biomedical retrieval and KG tools.

    Provides a code execution environment for RLM with injected tools
    for searching 27.3M PubMed abstracts, 129M full-text chunks,
    and optional knowledge graph queries.

    Tool Categories:
    1. Search: search_papers, search_chunks, search_by_entities
    2. Filter: search_mesh_terms, get_pmids_by_mesh, keyword_search
    3. Ranking: rerank_papers, rerank_chunks
    4. Structure: get_paper_abstracts, get_paper_sections
    5. Advanced: hybrid_search, iterative_search
    6. KG (optional): kg_lightrag_query, kg_pathrag_query, kg_graphrag_query
    7. Helpers: format_papers, format_chunks, extract_answer

    Usage:
        repl = BiomedicalREPL(enable_kg_tools=True)
        repl.setup()

        result = repl.execute('''
            # Search for papers
            papers = search_papers("Does metformin help diabetes?", limit=20)

            # Rerank for precision
            top_papers = rerank_papers("metformin diabetes efficacy", papers, top_k=5)

            # Format for analysis
            context = format_papers(top_papers)
            print(context)

            # Get detailed evidence from full-text
            pmids = [p['pmid'] for p in top_papers if p['have_fulltext']]
            chunks = search_chunks("treatment efficacy", paper_ids=pmids)
        ''')
    """

    def __init__(
        self,
        enable_kg_tools: bool = True,
        context_payload: dict | list | str | None = None,
        setup_code: str | None = None,
        atlas_context: dict | None = None,
        **kwargs,
    ):
        """Initialize BiomedicalREPL.

        Args:
            enable_kg_tools: Whether to inject KG tools (default True)
            context_payload: Context data to load into environment
            setup_code: Python code to run on initialization
            atlas_context: Optional scRNA atlas dict, injected as `atlas` variable
            **kwargs: Additional args passed to LocalREPL
        """
        self._enable_kg_tools = enable_kg_tools
        self._atlas_context = atlas_context

        super().__init__(
            context_payload=context_payload,
            setup_code=setup_code,
            **kwargs,
        )

    @contextmanager
    def _temp_cwd(self):
        """Override LocalREPL's _temp_cwd to be a no-op for thread safety.

        LocalREPL's _temp_cwd() uses os.chdir() which is process-wide and
        causes race conditions when running with concurrency > 1.

        BiomedicalREPL tools use absolute paths and don't need directory changes,
        so we simply yield without changing directories.
        """
        yield

    def setup(self):
        """Setup the environment with biomedical tools."""
        super().setup()

        # Inject all retrieval tools
        for name, func in RETRIEVAL_TOOLS.items():
            self.globals[name] = func

        # Inject KG tools if enabled
        if self._enable_kg_tools:
            for name, func in KG_TOOLS.items():
                self.globals[name] = func

        # Add commonly used imports
        self.globals["json"] = __import__("json")
        self.globals["re"] = __import__("re")

        # Add centralized answer extraction (benchmark compliant)
        self.globals["extract_answer"] = extract_answer

        # Inject scRNA atlas context (Disco + HPA fallback) — empty dict if not provided
        self.globals["atlas"] = self._atlas_context or {}

    def get_available_tools(self) -> list[str]:
        """Get list of available tool names."""
        tools = list(RETRIEVAL_TOOLS.keys())
        if self._enable_kg_tools:
            tools.extend(KG_TOOLS.keys())
        tools.append("extract_answer")
        return tools

    def get_tool_help(self) -> str:
        """Get help text describing all available tools."""
        help_text = """
## Available Tools in BiomedicalREPL

### Search Tools
- search_papers(query, limit=50) - Search 27.3M PubMed papers
- search_chunks(query, limit=50, paper_ids=None) - Search 129M full-text chunks
- search_section_aware_chunks(query, limit=20, paper_ids=None, preferred_section_types=None) - Section-aware full-text search
- search_by_entities(entities, limit=30) - Search by biomedical entities

### Filter Tools
- search_mesh_terms(query, entities=None) - Find relevant MeSH terms
- get_pmids_by_mesh(mesh_terms, limit=1000) - Get PMIDs by MeSH terms
- keyword_search(keywords, limit=100, match_all=False) - Full-text keyword search

### Ranking Tools
- rerank_papers(query, papers, top_k=10) - Rerank papers with cross-encoder
- rerank_chunks(query, chunks, top_k=10) - Rerank chunks with cross-encoder

### Structure Tools
- get_paper_abstracts(pmids) - Fetch abstracts for specific PMIDs
- get_paper_sections(pmid) - Get hierarchical structure of PMC full-text
- expand_section_evidence(paper_id, seed_section_ids, query, top_k=10) - Expand intro/background hits into results/discussion/conclusion

### Advanced Tools
- hybrid_search(query, limit=50, use_mesh=True, use_keywords=True, use_vector=True)
- iterative_search(query, rounds=2, papers_per_round=20)

### Helper Tools
- format_papers(papers, max_papers=5) - Format papers for LLM context
- format_chunks(chunks, max_chunks=10) - Format chunks for LLM context
- extract_answer(text, question_type) - Extract structured answer
"""
        if self._enable_kg_tools:
            help_text += """
### Knowledge Graph Tools
- kg_lightrag_query(query, mode="hybrid", top_k=20) - LightRAG-style query
- kg_pathrag_query(query, top_k=40, path_depth=2) - PathRAG-style query
- kg_graphrag_query(query, search_type="local", community_level=2) - MS-GraphRAG style
"""
        return help_text
