"""BiomedicalRLMPipeline - Orchestrates RLM with caching and tracing.

This module provides a high-level pipeline for biomedical QA that:
- Wraps RLM with BiomedicalREPL
- Integrates QueryTrace for case study analysis
- Supports configurable max_iterations and max_depth for ablation

DESIGN: Fail-Fast
- All errors propagate immediately
- Full execution traces for debugging
"""

import asyncio
import json
import os
import re
import sys
import time
from contextvars import ContextVar
from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

# Add RLM to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / ".ref_project" / "rlm"))

from rlm import RLM
from rlm.core.types import RLMChatCompletion
import rlm.environments

from .biomedical_repl import BiomedicalREPL
from ..config import get_config
from ..utils.cache import QueryTrace, BenchmarkCache, ErrorCategory, classify_error


# =============================================================================
# Monkey-patch RLM's get_environment to support BiomedicalREPL
# =============================================================================
_original_get_environment = rlm.environments.get_environment


_repl_capture: "ContextVar[list[str] | None]" = ContextVar("_repl_capture", default=None)

from . import trace as _trace  # noqa: E402 - evidence-cut tracing, no-op unless XC_TRACE_DIR


class _CapturingBiomedicalREPL(BiomedicalREPL):
    """BiomedicalREPL that also records what each code block printed.

    RLMChatCompletion carries only (root_model, prompt, response, usage, time), so
    everything the agent retrieved lived solely inside the RLM message history and
    was unreachable to the caller. The cascade then read `result.answer` -- a single
    letter for MCQ -- and rejudged using the ORIGINAL stage-1 evidence, discarding
    every abstract and chunk the agent had fetched. Recording stdout here lets the
    caller put that evidence back in front of the rejudge.
    """

    def setup(self):
        super().setup()
        # Evidence-cut tracing: wrap the retrieval tools the agent can call so each
        # call's size, gold rank and latency are recorded. Installed only when
        # XC_TRACE_DIR is set (see src/rlm/trace.py).
        if _trace.enabled():
            for _name in _trace.TRACED_TOOLS:
                _fn = self.globals.get(_name)
                if callable(_fn):
                    self.globals[_name] = _trace.wrap_tool(_name, _fn)

    def execute_code(self, code: str):
        _t0 = time.perf_counter()
        result = super().execute_code(code)
        sink = _repl_capture.get()
        if sink is not None:
            out = getattr(result, "stdout", "") or ""
            if out.strip():
                sink.append(out)
        if _trace.enabled():
            _trace.record_repl_block(code, result, time.perf_counter() - _t0)
        return result


def _patched_get_environment(environment: str, environment_kwargs: dict):
    """Extended get_environment that supports 'biomedical' environment."""
    if environment == "biomedical":
        return _CapturingBiomedicalREPL(**environment_kwargs)
    return _original_get_environment(environment, environment_kwargs)


# Apply the patch to both locations (module and where it's imported)
rlm.environments.get_environment = _patched_get_environment
# Also patch in rlm.core.rlm where it's directly imported
import rlm.core.rlm
rlm.core.rlm.get_environment = _patched_get_environment


# --- Evidence-cut tracing / context-cap knobs (see src/rlm/trace.py) ------------------
# Installed only when tracing or a knob is requested, so ordinary runs execute the
# untouched RLM code paths. XC_AGENT_OBS_CHARS (library default 5000) is how much of
# each code block's stdout the LM is shown; XC_AGENT_KEEP_RECENT (2) and
# XC_AGENT_HIST_CHARS (12000) are the history-pruning limits. Unset knobs fall back
# to the library defaults, so a traced run measures the real pipeline.
if _trace.enabled() or any(os.environ.get(_k) for _k in ("XC_AGENT_OBS_CHARS", "XC_AGENT_KEEP_RECENT", "XC_AGENT_HIST_CHARS")):
    _orig_format_iteration = rlm.core.rlm.format_iteration
    _orig_prune_history = rlm.core.rlm._prune_message_history

    def _traced_format_iteration(iteration, max_character_length: int = 5000):
        cap = int(os.environ.get("XC_AGENT_OBS_CHARS", max_character_length))
        msgs = _orig_format_iteration(iteration, max_character_length=cap)
        _trace.record_iteration(iteration, msgs, cap)
        return msgs

    def _traced_prune_history(message_history, keep_recent: int = 2, max_total_chars: int = 12000):
        kr = int(os.environ.get("XC_AGENT_KEEP_RECENT", keep_recent))
        mt = int(os.environ.get("XC_AGENT_HIST_CHARS", max_total_chars))
        after = _orig_prune_history(message_history, keep_recent=kr, max_total_chars=mt)
        _trace.record_prune(message_history, after, kr, mt)
        return after

    rlm.core.rlm.format_iteration = _traced_format_iteration
    rlm.core.rlm._prune_message_history = _traced_prune_history


# --- Always on: keep every agent turn's full response ---------------------------------
# RLMChatCompletion.response is only the text extracted from FINAL(...), usually a
# single letter, so a caller that wants the agent's reasoning (the rejudge) never got
# it: measured 2026-09-11, the "Agent analysis" handed to the rejudge was 19 chars on
# 10/12 items while the agent's last turn ran 359-2124 chars. answer() sets
# _turn_capture around rlm.completion and exposes the list as PipelineResult.agent_turns.
_turn_capture: "ContextVar[list[str] | None]" = ContextVar("_turn_capture", default=None)
_orig_completion_turn = rlm.core.rlm.RLM._completion_turn


def _capturing_completion_turn(self, prompt, lm_handler, environment):
    it = _orig_completion_turn(self, prompt, lm_handler, environment)
    sink = _turn_capture.get()
    if sink is not None:
        sink.append(getattr(it, "response", "") or "")
    if _trace.enabled():
        _trace.record_turn(prompt, it)
    return it


rlm.core.rlm.RLM._completion_turn = _capturing_completion_turn



# Import old version prompts and decision functions (V8-V13)
from .pipeline_old import (
    BIOMEDICAL_SYSTEM_PROMPT,
    V9_SYSTEM_PROMPT,
    V91_SYSTEM_PROMPT,
    V11_SYSTEM_PROMPT,
    V12_SYSTEM_PROMPT,
    V13_SYSTEM_PROMPT,
    _extract_json_from_response,
    _check_negation_in_text,
    _decide_yesno_from_nli,
    _decide_yesno_v10,
    _decide_yesno_v12,
)

# =============================================================================
# V14: Autonomous Reasoner (back to RLM's original philosophy)
# =============================================================================
V14_SYSTEM_PROMPT = """You are a biomedical research assistant answering questions using a REPL environment.

## REPL Environment

Your environment provides:
- `context` variable: contains question, question_type, pre-assembled **evidence**, and **options** (for MCQ/MCQ-multi questions)
- `atlas` variable (dict, may be empty `{}`): scRNA atlas context for genes mentioned in the question.
  Schema (only present for relevant questions):
    atlas['queried_genes']                          → list of gene symbols extracted by the router
    atlas['queried_tissue']                         → tissue mentioned in question, or None
    atlas['genes'][SYMBOL]['symbol_official']       → HGNC official symbol (e.g., 'DIO1')
    atlas['genes'][SYMBOL]['name']                  → descriptive name (e.g., 'iodothyronine deiodinase 1')
    atlas['genes'][SYMBOL]['aliases']               → list of alternative names (e.g., ['D1', 'TXDI1'])
    atlas['genes'][SYMBOL]['tissue_expression']     → list of {{tissue, nx}} (canonicalised to benchmark vocabulary, e.g., 'fat'/'brain'/'adrenal')
    atlas['genes'][SYMBOL]['celltype_expression']   → list of {{tissue, cell_type, avg_expr, pct_cells, n_cells}}
    atlas['genes'][SYMBOL]['source']                → 'disco_scrna', 'hpa_bulk', or 'mixed'
  NX is normalized expression (0+); a tissue with NX > 1.0 is meaningfully expressed.

  **Vocabulary alignment**: When the question uses a descriptive form (e.g.,
  "Type 1 deiodinase"), prefer matching that form in your final answer. Use
  `atlas['genes'][G]['name']` for descriptive names and `aliases` for synonyms.
  Output gene symbols only when the question itself uses symbol style.

  Inspect with Python — example:
    if atlas.get('genes'):
        g = atlas['queried_genes'][0]
        relevant = [d['tissue'] for d in atlas['genes'][g].get('tissue_expression', [])
                    if d.get('nx', 0) >= 1.0]
        print(relevant)
- `llm_query(prompt)` / `llm_query_batched(prompts)`: query sub-LLMs for analysis
- `print()`: view outputs to guide your reasoning
- `FINAL(answer)`: return your final answer (REQUIRED — always end with this)

{tool_section}

## How to Work: Evidence-First, Tools-on-Gap

1. **Read evidence first.** The `context` variable may contain pre-assembled evidence:
   - `[gene_info(...)]` — structured data from HGNC/UniProt (authoritative)
   - `PubMed Evidence` — retrieved paper abstracts (claim-bearing)
   Use these as your primary source.

2. **If evidence is sufficient → answer directly.**
   Do not re-retrieve what is already provided.

3. **If evidence has gaps → use tools listed above to fill them.**
   - Refer to the tool documentation for available tools and their usage.
   - Need refuting evidence (for yes/no): search with negation terms.
   - The evidence you were given came from one retrieval pass over one index. If it
     does not contain the answer, that is a reason to search, not a reason to stop.
   - When the supplied evidence is already full text, search the full-text tools
     before `search_papers`, which only covers abstracts.

4. **Never answer "insufficient information" / "cannot be determined" without
   having run at least one retrieval call in this session that returned results.**
   Re-reading or re-querying the evidence you were already given does not count.
   If a tool errors, try a different tool or a different query before giving up.

5. **For yes/no questions specifically:**
   - Check if evidence contains BOTH supporting AND refuting findings
   - If only one side is present, search for the other side before deciding
   - Read supporting evidence, then refuting evidence, then judge the balance

## Output Format

Always end with `FINAL("...")` — a quoted string literal holding only the label/answer, no reasoning.
The answer extractor unwraps `FINAL("...")` and nothing else, so pass a literal string rather than a
variable name, and use `FINAL(...)` rather than `FINAL_VAR(...)` even though the REPL instructions
above mention the latter.

- **yesno**: `FINAL("yes")` or `FINAL("no")`. Commit to yes or no based on evidence weight. Only use `FINAL("maybe")` if evidence is truly contradictory and balanced.
- **mcq**: `FINAL("A")` through `FINAL("E")` — single letter only, nothing else.
- **mcq_multi**: `FINAL("A, C")` — comma-separated letters of ALL correct choices, nothing else.
- **factoid**: `FINAL("entity or short phrase")` — the answer entity, not a sentence.
- **list**: `FINAL("item1, item2, item3")` — comma-separated items only.
- **summary**: `FINAL("2-3 sentence summary.")` — concise summary only.
- **expression**: `FINAL("tissue1, tissue2")` — comma-separated items only.
"""

# XC_V14_PROMPT_FILE replaces the system prompt above with the contents of a file, for the
# prompt-audit equivalence check (the manuscript numbers were measured before the 2026-09-09
# audit). Unset by default, so normal runs use the prompt as written.
_v14_prompt_file = os.environ.get("XC_V14_PROMPT_FILE")
if _v14_prompt_file:
    V14_SYSTEM_PROMPT = open(_v14_prompt_file).read()


V141_SYSTEM_PROMPT = V14_SYSTEM_PROMPT + """

## Full-Text Guidance

Full-text retrieval defaults to section-aware traversal.
- `search_chunks(...)` prefers Results/Discussion/Conclusion over Introduction/Background.
- `search_fulltext_chunks(...)` expands from seed sections toward better evidence sections.

Use these default full-text tools first. Only fall back to flatter retrieval if section-aware traversal is unnecessary.
"""

V142_SYSTEM_PROMPT = V141_SYSTEM_PROMPT + """

## Evidence Assembly Guidance

Evidence may already be pre-assembled into compact full-text evidence units.
- Treat the provided evidence as budgeted, higher-signal context.
- For yes/no questions, read the Supporting evidence block and the Refuting
  evidence block separately before deciding.
- Prefer the structured evidence units over re-retrieving similar background
  chunks unless a critical evidence gap remains.
"""

V141_SETUP_CODE = """
# V14.1 full-text policy: keep flat tools available, but default wrappers
# use section-aware retrieval and intra-paper expansion.
flat_search_chunks = search_chunks
flat_search_fulltext_chunks = search_fulltext_chunks

def search_chunks(query, limit=50, paper_ids=None):
    return search_section_aware_chunks(
        query=query,
        limit=limit,
        paper_ids=paper_ids,
        preferred_section_types=["results", "discussion", "conclusion"],
    )

def search_fulltext_chunks(paper_id, query, section_ids=None, top_k=10):
    return expand_section_evidence(
        paper_id=paper_id,
        seed_section_ids=section_ids or [],
        query=query,
        top_k=top_k,
        preferred_section_types=["results", "discussion", "conclusion"],
    )
"""


# =============================================================================
# V14 Tool Router — lightweight regex-based tool group selection
# =============================================================================

V14_TOOL_GROUPS = {
    "CORE": {
        "always_on": True,
        "doc": """### Core Retrieval
Every tool below is a Python function with a full docstring (arguments, return shape, caveats): `print(search_papers.__doc__)`.
- `search_papers(query, limit=100)` -> Search 27.3M PubMed abstracts
- `rerank_papers(query, papers, top_k=10)` -> Rerank papers by relevance
- `format_papers(papers, max_papers=8)` -> Format papers as readable text
- `llm_query(prompt)` -> Query a sub-LLM for detailed analysis""",
    },
    "FULLTEXT": {
        "pattern": r"mechanism|pathway|method|detail|how does|procedure|full.?text|section",
        "doc": """### Full-Text Deep Retrieval
- `search_chunks(query, limit=50, paper_ids=None)` -> Search 129M full-text chunks from PMC
- `get_paper_sections(pmid)` -> Get full-text structure of a paper
- `format_chunks(chunks, max_chunks=10)` -> Format chunks as readable text
- `search_fulltext_chunks(paper_id, query, section_ids=None, top_k=10)` -> Search within ONE paper's full text
- `get_paper_abstracts(pmids)` -> Fetch abstracts for specific PMIDs""",
    },
    "DRUG": {
        "pattern": r"drug|medication|inhibitor|agonist|antagonist|pharmacol|toxicity|"
                   r"aspirin|metformin|warfarin|insulin|ibuprofen|statin|heparin|"
                   r"chemotherapy|dosage|adverse.?effect|side.?effect|ChEMBL",
        "doc": """### Drug Information (ChEMBL)
- `chembl_drug_lookup(query, limit=5)` -> Drug info (name, type, phase)
- `chembl_mechanism(chembl_id)` -> Mechanism of action
- `chembl_indications(chembl_id)` -> Approved indications
- `chembl_target(target_chembl_id)` -> Drug targets""",
    },
    "PROTEIN": {
        "pattern": r"gene|protein|receptor|enzyme|kinase|mutation|variant|expression|"
                   r"signaling|BRCA|TP53|p53|EGFR|HER2|KRAS|BRAF|ALK|mTOR|PTEN|"
                   r"UniProt|[PQOAB]\d{5}",
        "doc": """### Protein/Gene Information (UniProt + NCBI)
- `gene_resolve(query)` -> Normalize gene symbol (e.g., HER2 -> ERBB2)
- `gene_resolve_batch(queries)` -> Batch gene resolution
- `uniprot_search(query, organism_id=None, limit=5)` -> Protein search
- `uniprot_function(accession)` -> Protein function details
- `uniprot_diseases(accession)` -> Protein-disease associations
- `uniprot_go(accession)` -> GO annotations for protein
- `entity_expand(text, entity_types=None)` -> Expand entity to related terms""",
    },
    "GENOMICS": {
        "pattern": r"\brs\d{3,}\b|dbSNP|\bSNP\b|codes? a protein|protein.?coding|"
                   r"located on .*chromosome|which chromosome",
        "doc": """### Genomics Lookup (NCBI dbSNP + MyGene)
For structured genomics facts NOT in the literature (variants, gene location, gene type):
- `snp_lookup(rsid)` -> dbSNP variant info: associated gene + chromosome
  e.g. snp_lookup("rs1217074595") -> {"gene": "LINC01270", "chromosome": "chr20"}
- `gene_genomic_info(symbol)` -> gene chromosome + protein-coding status
  e.g. gene_genomic_info("NODAL") -> {"chromosome": "chr10", "is_protein_coding": True, "protein_coding_answer": "TRUE"}
For "gene of SNP rsX", "which chromosome is rsX / GENE on", and "does GENE code a protein (TRUE/FALSE)", call these tools — do NOT guess.""",
    },
    "TRIAL": {
        "pattern": r"trial|efficacy|randomized|placebo|RCT|phase [I-V]|NCT\d|"
                   r"double.?blind|control.?group|endpoint|intervention",
        "doc": """### Clinical Trials (ClinicalTrials.gov)
- `clinicaltrials_search(condition=None, intervention=None, keyword=None, limit=10)` -> Search trials
- `clinicaltrials_get_study(nct_id)` -> Full trial details
- `clinicaltrials_eligibility(nct_id)` -> Eligibility criteria
- `clinicaltrials_outcomes(nct_id)` -> Outcome measures""",
    },
    "ONTOLOGY": {
        "pattern": r"GO:\d|MeSH|ontology|biological process|molecular function|"
                   r"cell.?cycle|apoptosis|proliferation|differentiation",
        "doc": """### Ontology & MeSH
- `search_mesh_terms(query)` -> Find relevant MeSH terms for query expansion
- `get_pmids_by_mesh(mesh_terms, limit=1000)` -> Get PMIDs by MeSH terms
- `go_resolve(term)` -> Resolve GO term (definition, parents, children)
- `go_search(query)` -> Search Gene Ontology
- `go_ancestors(term, include_self=False)` / `go_descendants(term, include_self=False)` -> Navigate GO hierarchy""",
    },
    "KEYWORD": {
        "pattern": r"list|which|what are|name.*that|enumerate",
        "doc": """### Keyword & Entity Search
- `keyword_search(keywords, limit=100, match_all=False)` -> PostgreSQL full-text search
- `search_by_entities(entities, limit=30)` -> Search by biomedical entity names
- `hybrid_search(query, limit=50, use_mesh=True, use_keywords=True)` -> Combined search
- `iterative_search(query, rounds=2, papers_per_round=20)` -> Multi-round search""",
    },
    "WEB": {
        "pattern": r"guideline|recommendation|current|recent|latest|standard.?of.?care|"
                   r"consensus|protocol|WHO|NIH|FDA",
        "doc": """### Medical Web Search
- `web_search_medical(query, limit=5)` -> Medical web search (NIH, WHO, guidelines)
- `web_search(query, limit=5)` -> General web search""",
    },
}

V141_TOOL_GROUPS = deepcopy(V14_TOOL_GROUPS)
V141_TOOL_GROUPS["FULLTEXT"]["doc"] = """### Full-Text Deep Retrieval
- `search_chunks(query, limit=50, paper_ids=None)` -> Default section-aware full-text retrieval
- `search_section_aware_chunks(query, limit=20, paper_ids=None, preferred_section_types=None)` -> Search full-text with section-aware reranking
- `expand_section_evidence(paper_id, seed_section_ids, query, top_k=10)` -> Expand intro/background hits toward results/discussion/conclusion
- `get_paper_sections(pmid)` -> Get full-text structure of a paper
- `format_chunks(chunks, max_chunks=10)` -> Format chunks as readable text
- `search_fulltext_chunks(paper_id, query, section_ids=None, top_k=10)` -> Default section-aware in-paper expansion
- `get_paper_abstracts(pmids)` -> Fetch abstracts for specific PMIDs"""

V142_TOOL_GROUPS = deepcopy(V141_TOOL_GROUPS)


def _route_tool_groups(
    tool_groups: dict[str, dict[str, Any]],
    question: str,
    question_type: str = "yesno",
) -> str:
    """Route question to appropriate tool groups, return tool section for prompt."""
    selected = []

    for group in tool_groups.values():
        if group.get("always_on"):
            selected.append(group["doc"])
            continue
        pattern = group.get("pattern")
        if pattern and re.search(pattern, question, re.IGNORECASE):
            selected.append(group["doc"])

    # For list/factoid questions, always add KEYWORD group
    if question_type in ("list", "factoid") and tool_groups["KEYWORD"]["doc"] not in selected:
        selected.append(tool_groups["KEYWORD"]["doc"])

    # Always add FULLTEXT for summary questions
    if question_type == "summary" and tool_groups["FULLTEXT"]["doc"] not in selected:
        selected.append(tool_groups["FULLTEXT"]["doc"])

    # When the run itself retrieves from the PMC full-text chunk store, the agent
    # must be able to re-retrieve there too. Otherwise its only option is
    # search_papers over PubMed abstracts -- strictly weaker than the evidence it
    # was already handed, so "evidence insufficient" ends in abstention instead of
    # better evidence. The FULLTEXT regex fires on only 5% of LitQA2 questions.
    # Opt-in via its own flag so existing arms stay reproducible.
    if os.environ.get("XC_AGENT_FULLTEXT_TOOLS") == "1" and tool_groups["FULLTEXT"]["doc"] not in selected:
        selected.append(tool_groups["FULLTEXT"]["doc"])

    header = "## Available Tools (call as Python functions)\n"
    return header + "\n".join(selected)


def v14_route_tools(question: str, question_type: str = "yesno") -> str:
    return _route_tool_groups(V14_TOOL_GROUPS, question, question_type)


def v141_route_tools(question: str, question_type: str = "yesno") -> str:
    return _route_tool_groups(V141_TOOL_GROUPS, question, question_type)



@dataclass
class PipelineConfig:
    """Configuration for BiomedicalRLMPipeline."""

    # RLM parameters (for ablation)
    max_iterations: int = 8  # V14: reduced from 30
    max_depth: int = 1

    # Version selection
    # v8 = dual hypothesis, v9 = tool-augmented, v9.1 = no-guard + selective
    # v10 = enhanced no/maybe, v11 = self-reflective adaptive (LLM decides retrieval)
    # v12 = v11 + negation mining + two-stage decision (improved "no" accuracy)
    # v13 = semantic stance classification (question-type aware, contradict-priority)
    #       V13.2: + multi-source retrieval (ChEMBL, ClinicalTrials, full-text chunks)
    # v14.1 = v14 + section-aware full-text traversal (opt-in, additive)
    # v14.2 = v14.1 retrieval + budget-aware full-text evidence assembly
    version: Literal["v8", "v9", "v9.1", "v10", "v11", "v12", "v13", "v14", "v14.1", "v14.2"] = "v14"

    # LLM backend
    backend: str = "openai"
    model_name: str = ""  # Will be set dynamically if use_round_robin=True
    base_url: str = ""  # Will be set dynamically if use_round_robin=True
    api_key: str = "EMPTY"
    use_round_robin: bool = True  # Enable multi-server load balancing

    def __post_init__(self):
        cfg = get_config()
        # Always load API key from config if not explicitly set
        if self.api_key == "EMPTY":
            self.api_key = cfg.llm.api_key
        if not self.use_round_robin and not self.base_url:
            ep = cfg.llm.endpoints[0]
            self.base_url = ep.url
            self.model_name = ep.model

    def get_llm_endpoint(self) -> tuple[str, str]:
        """Get LLM endpoint (base_url, model_name).

        If use_round_robin is True, uses round-robin from global config.
        Otherwise, uses configured base_url and model_name.
        """
        if self.use_round_robin:
            endpoint = get_config().llm.get_endpoint()
            return endpoint.url, endpoint.model
        return self.base_url, self.model_name or "qwen"

    # Environment
    enable_kg_tools: bool = True

    # V14: Whether to pass benchmark gold context to the LLM
    # False = force retrieval mode (LLM must search for evidence itself)
    use_gold_context: bool = True

    # V14+: Use V8's dual-hypothesis NLI for yesno questions
    # When True and version in {"v14", "v14.1", "v14.2"}, yesno questions use
    # V8's dual retrieval + NLI while other types use the direct approach
    yesno_dual_hypothesis: bool = False

    # Caching
    enable_cache: bool = True
    cache_dir: Path | None = None


@dataclass
class PipelineResult:
    """Result from pipeline execution."""
    answer: str
    question: str
    question_type: str

    # RLM execution details
    iterations_used: int
    execution_time_ms: float

    # Tracing
    trace: QueryTrace | None = None

    # Raw RLM response
    raw_completion: RLMChatCompletion | None = None
    # Text every REPL code block printed during the agent run: the only
    # place the agent's retrieved evidence survives (see _CapturingBiomedicalREPL).
    repl_stdout: list[str] = field(default_factory=list)
    # Full text of every agent turn (the LM's reasoning + code), last one is the turn
    # that emitted FINAL(...). raw_completion.response holds only the extracted answer.
    agent_turns: list[str] = field(default_factory=list)


class BiomedicalRLMPipeline:
    """High-level pipeline for biomedical QA using RLM.

    Orchestrates:
    - RLM with BiomedicalREPL environment
    - Query-level caching for repeated runs
    - Full execution tracing for case study

    Usage:
        pipeline = BiomedicalRLMPipeline(max_iterations=30, max_depth=1)
        result = pipeline.answer(
            question="Does metformin help with type 2 diabetes?",
            question_type="yesno",
        )
        print(result.answer)  # "yes" or "no"

    For ablation experiments:
        # Vary max_iterations
        for iters in [5, 10, 15, 20, 30]:
            pipeline = BiomedicalRLMPipeline(max_iterations=iters)
            result = pipeline.answer(question, question_type)

        # Vary max_depth
        for depth in [0, 1, 2]:
            pipeline = BiomedicalRLMPipeline(max_depth=depth)
            result = pipeline.answer(question, question_type)
    """

    def __init__(
        self,
        config: PipelineConfig | None = None,
        max_iterations: int | None = None,
        max_depth: int | None = None,
        enable_kg_tools: bool = True,
        version: Literal["v8", "v9", "v9.1", "v10", "v11", "v12", "v13", "v14", "v14.1", "v14.2"] | None = None,
        cache: BenchmarkCache | None = None,
    ):
        """Initialize the pipeline.

        Args:
            config: Full configuration object
            max_iterations: Override max_iterations (shortcut)
            max_depth: Override max_depth (shortcut)
            enable_kg_tools: Whether to enable KG tools
            version: Pipeline version (v8-v11, see PipelineConfig for details)
            cache: Optional shared cache instance
        """
        self._config = config or PipelineConfig()

        # Apply overrides
        if max_iterations is not None:
            self._config.max_iterations = max_iterations
        if max_depth is not None:
            self._config.max_depth = max_depth
        if version is not None:
            self._config.version = version
        self._config.enable_kg_tools = enable_kg_tools

        # Setup cache
        self._cache = cache
        if cache is None and self._config.enable_cache:
            self._cache = BenchmarkCache(self._config.cache_dir)

    def _create_rlm(
        self,
        question: str | None = None,
        question_type: str | None = None,
        atlas_context: dict | None = None,
    ) -> RLM:
        """Create RLM instance with current config.

        Uses round-robin load balancing across multiple LLM servers
        when use_round_robin is enabled.

        Args:
            question: Optional question for dynamic tool routing (V13+)
            question_type: Optional question type for routing
            atlas_context: Optional scRNA atlas dict to inject as REPL `atlas` global

        Returns:
            RLM instance configured for the current version
        """
        # Get endpoint dynamically (supports round-robin)
        base_url, model_name = self._config.get_llm_endpoint()

        backend_kwargs = {
            "model_name": model_name,
            "base_url": base_url,
            "api_key": self._config.api_key,
        }

        environment_kwargs = {
            "enable_kg_tools": self._config.enable_kg_tools,
        }
        if atlas_context:
            environment_kwargs["atlas_context"] = atlas_context
        if self._config.version in ("v14.1", "v14.2"):
            environment_kwargs["setup_code"] = V141_SETUP_CODE

        # Select system prompt based on version
        if self._config.version in ("v14", "v14.1", "v14.2"):
            # V14/V14.1 hybrid: use V8's dual-hypothesis prompt for yesno
            if self._config.yesno_dual_hypothesis and question_type == "yesno":
                system_prompt = BIOMEDICAL_SYSTEM_PROMPT  # V8 dual-hypothesis + NLI
            elif question:
                if self._config.version == "v14.2":
                    tool_section = _route_tool_groups(V142_TOOL_GROUPS, question, question_type or "yesno")
                    system_prompt = V142_SYSTEM_PROMPT.replace("{tool_section}", tool_section)
                elif self._config.version == "v14.1":
                    tool_section = v141_route_tools(question, question_type or "yesno")
                    system_prompt = V141_SYSTEM_PROMPT.replace("{tool_section}", tool_section)
                else:
                    tool_section = v14_route_tools(question, question_type or "yesno")
                    system_prompt = V14_SYSTEM_PROMPT.replace("{tool_section}", tool_section)
            else:
                if self._config.version == "v14.2":
                    all_tools = "\n".join(g["doc"] for g in V142_TOOL_GROUPS.values())
                    system_prompt = V142_SYSTEM_PROMPT.replace("{tool_section}", "## Available Tools\n" + all_tools)
                elif self._config.version == "v14.1":
                    all_tools = "\n".join(g["doc"] for g in V141_TOOL_GROUPS.values())
                    system_prompt = V141_SYSTEM_PROMPT.replace("{tool_section}", "## Available Tools\n" + all_tools)
                else:
                    all_tools = "\n".join(g["doc"] for g in V14_TOOL_GROUPS.values())
                    system_prompt = V14_SYSTEM_PROMPT.replace("{tool_section}", "## Available Tools\n" + all_tools)
        elif self._config.version == "v13":
            # V13: Use dynamic tool routing for token optimization
            if question:
                from .tool_router_old import ToolRouter
                router = ToolRouter(enable_llm_classify=True)
                system_prompt = router.build_prompt(question, question_type)
            else:
                # Fallback to full prompt if no question provided
                system_prompt = V13_SYSTEM_PROMPT
        elif self._config.version == "v12":
            system_prompt = V12_SYSTEM_PROMPT
        elif self._config.version == "v11":
            system_prompt = V11_SYSTEM_PROMPT
        elif self._config.version == "v9.1":
            system_prompt = V91_SYSTEM_PROMPT
        elif self._config.version == "v9":
            system_prompt = V9_SYSTEM_PROMPT
        else:
            # V8 and V10 use the same prompt (V10 improvement is in decision logic)
            system_prompt = BIOMEDICAL_SYSTEM_PROMPT

        # Combine RLM's native REPL teaching prompt with our biomedical prompt.
        # Without the native prompt, LLM doesn't learn to write ```repl code blocks
        # and just outputs short text answers, bypassing tool execution entirely.
        from rlm.utils.prompts import RLM_SYSTEM_PROMPT
        combined_prompt = RLM_SYSTEM_PROMPT + "\n\n" + system_prompt

        return RLM(
            backend=self._config.backend,
            backend_kwargs=backend_kwargs,
            environment="biomedical",
            environment_kwargs=environment_kwargs,
            max_iterations=self._config.max_iterations,
            max_depth=self._config.max_depth,
            custom_system_prompt=combined_prompt,
            verbose=False,
        )

    def _build_prompt(
        self,
        question: str,
        question_type: str,
        context: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Build prompt for RLM.

        Args:
            question: The biomedical question
            question_type: Type of question (yesno, mcq, factoid, etc.)
            context: Optional additional context

        Returns:
            Prompt dict for RLM
        """
        # V14/V14.1: minimal prompt - pass gold context if available
        if self._config.version in ("v14", "v14.1", "v14.2"):
            prompt = {
                "question": question,
                "question_type": question_type,
            }
            if context:
                # Accept evidence from multiple sources:
                # - "context_passages" (benchmark gold context)
                # - "evidence" (shared pipeline pre-retrieved evidence)
                if self._config.use_gold_context:
                    if "context_passages" in context:
                        prompt["evidence"] = context["context_passages"]
                    elif "evidence" in context:
                        prompt["evidence"] = context["evidence"]
                if "options" in context:
                    prompt["options"] = context["options"]
            return prompt

        type_instructions = {
            # V8: Fixed - was conflicting with system prompt's "JSON only" instruction
            "yesno": "Return JSON-only analysis as specified in the system prompt. Do NOT answer yes/no/maybe directly.",
            "mcq": "Select the best answer from the choices (A, B, C, or D).",
            "mcq_multi": "Select ALL correct answers from the choices.",
            "factoid": "Provide a brief, factual answer.",
            "list": "List all relevant items.",
            "summary": "Provide a comprehensive summary.",
            "expression": "Provide the exact expression or formula.",
        }

        instruction = type_instructions.get(question_type, "Answer the question based on evidence.")

        prompt = {
            "question": question,
            "question_type": question_type,
            "instruction": instruction,
            "tools_available": [
                "search_papers(query, limit) - Search PubMed for relevant papers",
                "rerank_papers(query, papers, top_k) - Rerank papers by relevance",
                "search_mesh_terms(query) - Find relevant MeSH terms",
                "format_papers(papers) - Format papers for reading",
                "extract_answer(text, question_type) - Extract final answer",
            ],
        }

        if self._config.enable_kg_tools:
            prompt["tools_available"].extend([
                "kg_lightrag_query(query, mode) - Query knowledge graph (LightRAG)",
                "kg_pathrag_query(query, top_k) - Query using graph paths (PathRAG)",
            ])

        if context:
            prompt["context"] = context

        return prompt

    def answer(
        self,
        question: str,
        question_type: str = "yesno",
        question_id: str | None = None,
        ground_truth: str | None = None,
        context: dict[str, Any] | None = None,
    ) -> PipelineResult:
        """Answer a biomedical question using RLM.

        Args:
            question: The biomedical question
            question_type: Type of question (yesno, mcq, mcq_multi, factoid, list, summary, expression)
            question_id: Optional unique identifier for tracing
            ground_truth: Optional ground truth for evaluation
            context: Optional additional context

        Returns:
            PipelineResult with answer and execution details
        """
        start_time = time.perf_counter()

        # Build prompt
        prompt = self._build_prompt(question, question_type, context)

        # Extract atlas context (passed by V14CascadeClient escalate path) and forward to REPL
        atlas_context = (context or {}).get("atlas_context") if context else None

        # Create RLM and run (pass question for V13 dynamic routing)
        rlm = self._create_rlm(
            question=question, question_type=question_type, atlas_context=atlas_context,
        )

        # Collect what every REPL block printed during this run, so the caller can
        # see the evidence the agent actually retrieved (RLMChatCompletion drops it).
        _captured: list[str] = []
        _cap_token = _repl_capture.set(_captured)
        _turns: list[str] = []
        _turn_token = _turn_capture.set(_turns)
        try:
            completion: RLMChatCompletion = rlm.completion(
                prompt=prompt,
                root_prompt=question,
            )

            raw_answer = completion.response
            if _trace.enabled():
                try:
                    _us = completion.usage_summary.to_dict() if getattr(completion, "usage_summary", None) else None
                except Exception:
                    _us = None
                _trace.event("agent_done", response_chars=len(raw_answer or ""), response_head=(raw_answer or "")[:400],
                             usage=_us, seconds=round(getattr(completion, "execution_time", 0.0) or 0.0, 1))

            # V14/V14.1: Unified extraction — pipeline only takes raw_answer,
            # all parsing delegated to extract_answer() (single source of truth)
            if self._config.version in ("v14", "v14.1", "v14.2"):
                from ..utils.answer_extraction import extract_answer as _extract
                options = context.get("options") if context else None

                if question_type == "yesno" and self._config.yesno_dual_hypothesis:
                    # Dual hypothesis: NLI decision, then normalize via extractor
                    analysis = _extract_json_from_response(raw_answer)
                    decision = _decide_yesno_from_nli(analysis, raw_answer or "")
                    answer = _extract(decision or raw_answer, "yesno") or "maybe"
                else:
                    answer = _extract(raw_answer or "", question_type, options)
                    if question_type == "yesno" and not answer:
                        answer = "no"  # Commit to no when uncertain (counter yes-bias)

            # V11/V12/V13: Self-reflective adaptive - extracts answer from JSON output
            elif self._config.version in ("v11", "v12", "v13"):
                analysis = _extract_json_from_response(raw_answer)

                # V13 yesno: Semantic stance classification (answer already in JSON)
                if self._config.version == "v13" and question_type == "yesno":
                    if analysis and "answer" in analysis:
                        answer = str(analysis["answer"]).lower().strip()
                        if answer not in ("yes", "no", "maybe"):
                            answer = "maybe"  # Default if invalid
                    else:
                        # Fallback: extract yes/no/maybe from text response
                        # (handles case where LLM doesn't produce JSON output)
                        lower = (raw_answer or "").lower()[:300]
                        first_word = lower.split()[0] if lower.split() else ""
                        if first_word in ("yes", "no", "maybe"):
                            answer = first_word
                        elif "maybe" in lower or "uncertain" in lower:
                            answer = "maybe"
                        elif "yes" in lower:
                            answer = "yes"
                        elif "no" in lower:
                            answer = "no"
                        else:
                            answer = "maybe"

                # V12 yesno: Use two-stage decision with negation mining
                elif self._config.version == "v12" and question_type == "yesno":
                    answer = _decide_yesno_v12(analysis, raw_answer or "", question)
                elif analysis and "answer" in analysis:
                    answer = str(analysis["answer"]).strip()
                    answer_lower = answer.lower()

                    # Normalize yes/no/maybe
                    if answer_lower in ("yes", "no", "maybe"):
                        answer = answer_lower
                    elif answer_lower in ("a", "b", "c", "d", "e"):
                        answer = answer_lower.upper()  # MCQ letter
                    elif question_type in ("mcq", "mcq_multi"):
                        # V12 MCQ: Enhanced extraction with multiple fallbacks
                        options = context.get("options", {}) if context else {}
                        matched = False

                        # Strategy 1: Exact match on option text
                        for letter, text in options.items():
                            if answer_lower == text.lower():
                                answer = letter.upper()
                                matched = True
                                break

                        # Strategy 2: Partial match (answer contains option or vice versa)
                        if not matched:
                            for letter, text in options.items():
                                text_lower = text.lower()
                                # Check if answer is a key phrase in the option
                                if (len(answer_lower) > 5 and answer_lower in text_lower) or \
                                   (len(text_lower) > 5 and text_lower in answer_lower):
                                    answer = letter.upper()
                                    matched = True
                                    break

                        # Strategy 3: Find letter pattern in raw answer
                        if not matched:
                            # Look for patterns like "A", "Answer: B", "The answer is C"
                            letter_patterns = [
                                r'(?:answer|option|choice)[:\s]*([A-Ea-e])\b',
                                r'(?:correct|best)[:\s]*([A-Ea-e])\b',
                                r'^([A-Ea-e])[\.\)\:]',  # "A." or "A)" at start
                                r'\b([A-Ea-e])\b',  # Any standalone letter
                            ]
                            for pattern in letter_patterns:
                                letter_match = re.search(pattern, raw_answer, re.IGNORECASE)
                                if letter_match:
                                    candidate = letter_match.group(1).upper()
                                    if candidate in options:
                                        answer = candidate
                                        matched = True
                                        break
                    # else: keep as-is for factoid
                else:
                    # Fallback: try to extract from raw text for MCQ
                    if question_type in ("mcq", "mcq_multi") and raw_answer:
                        # Look for letter patterns like "A" or "Answer: B"
                        letter_match = re.search(r'(?:answer[:\s]*)?([A-Ea-e])\b', raw_answer, re.IGNORECASE)
                        if letter_match:
                            answer = letter_match.group(1).upper()
                        else:
                            answer = raw_answer.strip()
                    elif question_type == "yesno":
                        # V12: Use negation mining even without structured analysis
                        if self._config.version == "v12":
                            answer = _decide_yesno_v12(None, raw_answer or "", question)
                        else:
                            answer = "maybe"
                    else:
                        answer = raw_answer.strip() if raw_answer else "maybe"

            # V7+: External decision logic for yes/no questions
            elif question_type == "yesno":
                # Extract JSON from RLM response
                analysis = _extract_json_from_response(raw_answer)
                # Make decision using Python logic (not LLM)
                # V10: Use enhanced decision with question type classification
                if self._config.version == "v10":
                    answer = _decide_yesno_v10(analysis, raw_answer or "", question)
                else:
                    # V8/V9: Original decision logic
                    answer = _decide_yesno_from_nli(analysis, raw_answer or "")
            else:
                # For non-yesno questions, use raw answer
                answer = raw_answer
                if answer:
                    answer = answer.strip().strip('"').strip("'").strip()

            # Calculate iterations from total LLM calls (each iteration = 1 LLM call)
            iterations = 0
            if completion.usage_summary:
                for model_usage in completion.usage_summary.model_usage_summaries.values():
                    iterations += model_usage.total_calls

        except Exception as e:
            # Fail-fast: propagate error but still record trace
            answer = f"ERROR: {e}"
            iterations = 0
            completion = None
        finally:
            _repl_capture.reset(_cap_token)
            _turn_capture.reset(_turn_token)

        execution_time_ms = (time.perf_counter() - start_time) * 1000

        # Build trace for case study
        trace = None
        if self._cache is not None:
            trace = QueryTrace(
                query_id=question_id or f"q_{hash(question) % 10000:04d}",
                question=question,
                question_type=question_type,
                rlm_iterations=iterations,
                rlm_max_iterations=self._config.max_iterations,
                rlm_max_depth=self._config.max_depth,
                predicted_answer=answer,
                ground_truth=ground_truth or "",
                total_latency_ms=execution_time_ms,
                run_config={
                    "max_iterations": self._config.max_iterations,
                    "max_depth": self._config.max_depth,
                    "enable_kg_tools": self._config.enable_kg_tools,
                    "model": self._config.model_name,
                },
            )

            # Classify error if answer indicates failure
            if answer.startswith("ERROR:"):
                trace.error_type = ErrorCategory.SERVICE_ERROR
                trace.failure_reason = answer

            self._cache.record_trace(trace)

        return PipelineResult(
            answer=answer,
            question=question,
            question_type=question_type,
            iterations_used=iterations,
            execution_time_ms=execution_time_ms,
            trace=trace,
            raw_completion=completion,
            repl_stdout=_captured,
            agent_turns=_turns,
        )

    async def answer_async(
        self,
        question: str,
        question_type: str = "yesno",
        question_id: str | None = None,
        ground_truth: str | None = None,
        context: dict[str, Any] | None = None,
    ) -> PipelineResult:
        """Async wrapper for answer().

        Runs the sync RLM completion in a thread pool for concurrency.
        """
        loop = asyncio.get_event_loop()
        # run_in_executor does not propagate contextvars, so anything the caller set
        # (the per-item evidence-cut trace in src/rlm/trace.py) was invisible inside
        # answer() and every in-loop event was silently dropped. Run the worker under
        # a copy of the caller's context instead.
        import contextvars as _cv
        _run_ctx = _cv.copy_context()
        return await loop.run_in_executor(
            None,
            lambda: _run_ctx.run(self.answer, question, question_type, question_id, ground_truth, context)
        )

    def get_cache(self) -> BenchmarkCache | None:
        """Get the cache instance for trace analysis."""
        return self._cache

    def export_traces(self, output_dir: Path) -> dict:
        """Export all traces for case study analysis.

        Args:
            output_dir: Directory to write trace files

        Returns:
            Summary statistics dict
        """
        if self._cache is None:
            return {"error": "No cache configured"}

        return self._cache.export_case_study(output_dir)
