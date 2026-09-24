"""Dynamic tool routing for RLM V13.

Reduces token usage by 50-75% (800 → 200-400 tokens) through intelligent
tool selection based on question content.

Architecture (V2 - Hybrid):
    Question → Regex Guards → Embedding Classify → LLM Fallback → Group Selection
                   ↓                 ↓                  ↓              ↓
              显式ID强制加入     主力分类器       低置信度兜底      生成工具说明

Key insight: Regex should be a "safety net" (high precision guard), NOT
the primary intent classifier. The primary classifier should understand
semantics (embedding similarity or LLM).

Usage:
    from src.rlm.tool_router import ToolRouter

    # Legacy mode (regex-based)
    router = ToolRouter(enable_embedding_classify=False)

    # Recommended mode (embedding-based with regex guards)
    router = ToolRouter(enable_embedding_classify=True)

    selected = router.route("Does warfarin reduce stroke risk in phase 3 trials?")
    # Returns: {"CORE", "B_DRUG", "D_TRIAL"} - captures multi-intent!
"""

import math
import re
import time
from dataclasses import dataclass, field
from typing import Any

from ..utils.async_helper import run_async


# =============================================================================
# Tool Group Definitions
# =============================================================================

TOOL_GROUPS: dict[str, dict[str, Any]] = {
    "CORE": {
        "tools": ["search_papers", "rerank_papers", "format_papers", "llm_query"],
        "tokens": 120,
        "always_on": True,
        "description": "Core Retrieval (PubMed 27.3M abstracts)",
        "tool_docs": """### Core Retrieval (PubMed 27.3M abstracts)
- search_papers(query, limit=100) → Search PubMed abstracts
- rerank_papers(question, papers, top_k=10) → Rerank by relevance
- format_papers(papers, max_papers=8) → Format for LLM reading
- llm_query(prompt) → Query sub-LLM for analysis""",
        "selection_rules": [],
        "step2_snippet": "",
    },
    "A_FULLTEXT": {
        "tools": ["search_chunks", "get_paper_sections", "format_chunks"],
        "tokens": 80,
        "always_on": False,
        "description": "Full-Text Search (PMC 129M chunks)",
        "tool_docs": """### Full-Text Search (PMC 129M chunks) - USE WHEN ABSTRACTS INSUFFICIENT
- search_chunks(query, limit=50, paper_ids=None) → Search full-text chunks
- get_paper_sections(pmid) → Get hierarchical structure of a paper (sections, chunks)
- format_chunks(chunks, max_chunks=10) → Format chunks for reading""",
        "selection_rules": [
            "**Abstract evidence insufficient** → search_chunks or get_paper_sections for full-text",
        ],
        "step2_snippet": """
# Full-text deep search (if abstracts insufficient)
top_pmids = [p.get('pmid') for p in reranked[:5] if p.get('have_fulltext')]
if top_pmids:
    try:
        chunks = search_chunks(question, limit=20, paper_ids=top_pmids)
        if chunks:
            evidence_parts.append(f"Full-Text Evidence:\\n{{format_chunks(chunks, max_chunks=5)}}")
            print(f"Full-text: Found {{len(chunks)}} chunks")
    except Exception as e:
        print(f"Full-text search failed: {{e}}")""",
    },
    "B_DRUG": {
        "tools": ["chembl_drug_lookup", "chembl_mechanism", "chembl_indications"],
        "tokens": 90,
        "always_on": False,
        "description": "Drug Information (ChEMBL)",
        "tool_docs": """### Drug Information (ChEMBL) - USE FOR DRUG QUESTIONS
- chembl_drug_lookup(query, limit=5) → Drug info (mechanism, indications)
- chembl_mechanism(chembl_id) → Drug mechanism of action
- chembl_indications(chembl_id) → Get approved indications""",
        "selection_rules": [
            "**Drug names present** (aspirin, metformin, warfarin...) → chembl_drug_lookup FIRST",
        ],
        "step2_snippet": """
# Drug lookup (ChEMBL)
drug_patterns = r'\\b(aspirin|metformin|warfarin|insulin|ibuprofen|morphine|heparin|prednisone|sertraline|clopidogrel|pembrolizumab|nivolumab|trastuzumab|imatinib|olaparib|osimertinib)\\b'
drug_match = re.search(drug_patterns, question.lower())
if drug_match:
    drug_name = drug_match.group(1)
    print(f"Drug detected: {{drug_name}} → ChEMBL lookup")
    try:
        drug_info = chembl_drug_lookup(drug_name, limit=3)
        if drug_info:
            drug_text = f"ChEMBL Drug Info for {{drug_name}}:\\n"
            for d in drug_info[:2]:
                drug_text += f"- {{d.get('pref_name', '')}}: {{d.get('molecule_type', '')}}\\n"
                if d.get('chembl_id'):
                    mechs = chembl_mechanism(d['chembl_id'])
                    if mechs:
                        drug_text += f"  Mechanism: {{mechs[0].get('mechanism', 'N/A')}}\\n"
            evidence_parts.append(drug_text)
    except Exception as e:
        print(f"ChEMBL lookup failed: {{e}}")""",
    },
    "C_PROTEIN": {
        "tools": ["uniprot_search", "uniprot_diseases", "gene_resolve"],
        "tokens": 85,
        "always_on": False,
        "description": "Protein/Gene Information (UniProt)",
        "tool_docs": """### Protein/Gene Information (UniProt) - USE FOR PROTEIN/GENE QUESTIONS
- uniprot_search(query, limit=5) → Protein search
- uniprot_diseases(accession) → Protein-disease associations
- gene_resolve(query) → Normalize gene symbol (e.g., "HER2" → "ERBB2")""",
        "selection_rules": [
            "**Gene/protein symbol** (HER2, BRCA1, p53...) → gene_resolve() to normalize FIRST",
            "**Protein/gene function question** → uniprot_search after gene_resolve",
        ],
        "step2_snippet": """
# Gene/protein normalization
gene_patterns = r'\\b(HER2|ERBB2|BRCA[12]|TP53|p53|EGFR|KRAS|BRAF|ALK|PD-?[L]?1|CTLA-?4|JAK2|FLT3|IDH[12]|PTEN|mTOR|VEGF|RB1|APC|ATM|MLH1|MSH[26])\\b'
gene_match = re.search(gene_patterns, question, re.IGNORECASE)
if gene_match:
    gene_symbol = gene_match.group(1).upper()
    print(f"Gene detected: {{gene_symbol}} → normalizing")
    try:
        gene_info = gene_resolve(gene_symbol)
        if gene_info and gene_info.get('symbol'):
            normalized = gene_info['symbol']
            if normalized != gene_symbol:
                expanded_query = expanded_query.replace(gene_symbol, f"{{gene_symbol}} OR {{normalized}}")
            prot_info = uniprot_search(normalized, limit=2)
            if prot_info and prot_info[0].get('accession'):
                diseases = uniprot_diseases(prot_info[0]['accession'])
                if diseases:
                    evidence_parts.append(f"Protein {{normalized}}: associated with {{', '.join([d.get('name','') for d in diseases[:3]])}}")
    except Exception as e:
        print(f"Gene resolution failed: {{e}}")""",
    },
    "D_TRIAL": {
        "tools": ["clinicaltrials_search", "clinicaltrials_get_study"],
        "tokens": 70,
        "always_on": False,
        "description": "Clinical Trials (ClinicalTrials.gov)",
        "tool_docs": """### Clinical Trials (ClinicalTrials.gov) - USE FOR TRIAL/EFFICACY QUESTIONS
- clinicaltrials_search(condition=None, intervention=None, limit=10) → Search trials
- clinicaltrials_get_study(nct_id) → Get full trial details""",
        "selection_rules": [
            "**Clinical trial/efficacy question** → clinicaltrials_search",
        ],
        "step2_snippet": """
# Clinical trials search
trial_keywords = ["trial", "efficacy", "randomized", "placebo", "intervention"]
if any(kw in question.lower() for kw in trial_keywords):
    print("Trial keywords detected → ClinicalTrials.gov")
    try:
        trials = clinicaltrials_search(condition=question[:100], limit=5)
        if trials:
            trial_text = "ClinicalTrials.gov:\\n"
            for t in trials[:3]:
                trial_text += f"- {{t.get('title', '')[:100]}} ({{t.get('status', '')}})\\n"
            evidence_parts.append(trial_text)
    except Exception as e:
        print(f"ClinicalTrials lookup failed: {{e}}")""",
    },
    "E_ONTOLOGY": {
        "tools": ["search_mesh_terms", "go_resolve"],
        "tokens": 60,
        "always_on": False,
        "description": "Entity Resolution & Ontology",
        "tool_docs": """### Entity Resolution & Ontology - USE TO EXPAND/NORMALIZE QUERIES
- search_mesh_terms(query) → Find relevant MeSH terms for query expansion
- go_resolve(term) → Resolve Gene Ontology term (get definition, parents, children)""",
        "selection_rules": [
            "**Complex query** → search_mesh_terms to expand, then search_papers",
            "**GO term mentioned** (apoptosis, cell cycle...) → go_resolve for ontology context",
        ],
        "step2_snippet": """
# MeSH term expansion for query enhancement
print("Expanding query with MeSH terms...")
try:
    mesh_terms = search_mesh_terms(question[:100])
    if mesh_terms:
        top_mesh = [m.get('descriptor_name', '') for m in mesh_terms[:3]]
        if top_mesh:
            expanded_query = f"({{question}}) OR ({{' OR '.join(top_mesh)}})"
            print(f"  MeSH expansion: {{top_mesh}}")
except Exception as e:
    print(f"MeSH expansion failed: {{e}}")""",
    },
    "F_WEB": {
        "tools": ["web_search_medical"],
        "tokens": 40,
        "always_on": False,
        "description": "Medical Web Search",
        "tool_docs": """### Medical Web Search - USE FOR GUIDELINES/RECENT EVIDENCE
- web_search_medical(query, limit=5) → Medical web search (NIH, WHO, guidelines)""",
        "selection_rules": [
            "**Guidelines/recent evidence needed** → web_search_medical",
        ],
        "step2_snippet": """
# Medical web search (guidelines, recent evidence)
print("Searching medical web sources...")
try:
    web_results = web_search_medical(question[:150], limit=5)
    if web_results:
        web_text = "Web Evidence (NIH/WHO/Guidelines):\\n"
        for w in web_results[:3]:
            web_text += f"- {{w.get('title', '')}}: {{w.get('snippet', '')[:100]}}\\n"
        evidence_parts.append(web_text)
except Exception as e:
    print(f"Web search failed: {{e}}")""",
    },
}

# Sorted group order for deterministic output
TOOL_GROUP_ORDER = ["CORE", "A_FULLTEXT", "B_DRUG", "C_PROTEIN", "D_TRIAL", "E_ONTOLOGY", "F_WEB"]


# =============================================================================
# Regex Guards (HIGH PRECISION safety net - explicit IDs only)
# These are ALWAYS checked and force-add their groups when matched.
# They do NOT gate or skip other classification methods.
# =============================================================================

REGEX_GUARDS: dict[str, str] = {
    "B_DRUG": r"\bChEMBL\d+\b",
    "C_PROTEIN": r"\bUniProt\b|\b[OPQAB]\d{5}\b",
    "D_TRIAL": r"\bNCT\d{6,}\b",
    "E_ONTOLOGY": r"\bGO:\d+\b|\bMeSH:[A-Z]\d+\b",
}


# =============================================================================
# Legacy Rule-Based Triggers (kept for backward compatibility)
# In hybrid mode, these are used as "lexical hints" not "gates"
# =============================================================================

# Strong triggers - legacy lexical patterns (no longer gates routing in hybrid mode)
STRONG_TRIGGERS: dict[str, str] = {
    "B_DRUG": r"\b(aspirin|metformin|ibuprofen|warfarin|morphine|fondaparinux|"
              r"methadone|etoricoxib|fenofibrate|amoxapine|heparin|insulin|"
              r"atorvastatin|lisinopril|omeprazole|acetaminophen|penicillin|"
              r"prednisone|gabapentin|sertraline|clopidogrel|oxycodone|"
              r"ChEMBL\d+)\b",
    "C_PROTEIN": r"\b(BRCA[12]|TP53|p53|EGFR|HER2|ERBB2|KRAS|BRAF|ALK|"
                 r"ROS1|MET|RET|NTRK|PIK3CA|AKT|mTOR|PTEN|CDK[46]|"
                 r"PD-?[L]?1|CTLA-?4|JAK2|BCR-ABL|FLT3|IDH[12]|"
                 r"UniProt|[PQOAB]\d{5})\b",
    "D_TRIAL": r"\b(NCT\d+|phase\s+[I-IV]+|randomized\s+controlled|"
               r"double[- ]blind|placebo[- ]controlled|RCT)\b",
    "E_ONTOLOGY": r"\b(GO:\d+|MeSH:[A-Z]\d+)\b",
}

# Weak triggers - lexical hints (do not gate main classifier)
WEAK_TRIGGERS: dict[str, str] = {
    "B_DRUG": r"\b(drug|medication|inhibitor|agonist|antagonist|therapy|"
              r"treatment|pharmaceutical|dose|dosage|efficacy|toxicity|"
              r"adverse\s+effect|side\s+effect|pharmacokinetic|pharmacodynamic)\b",
    "C_PROTEIN": r"\b(gene|protein|receptor|enzyme|kinase|phosphatase|"
                 r"transcription\s+factor|mutation|variant|expression|"
                 r"signaling|pathway)\b",
    "D_TRIAL": r"\b(trial|efficacy|safety|placebo|outcome|endpoint|"
               r"intervention|treatment\s+arm|control\s+group)\b",
    "A_FULLTEXT": r"\b(mechanism|pathway|how\s+does|detail|specific|"
                  r"describe|explain|methodology|method|procedure)\b",
    "E_ONTOLOGY": r"\b(ontology|hierarchy|term|definition|concept|"
                  r"biological\s+process|molecular\s+function|cellular\s+component)\b",
    "F_WEB": r"\b(guideline|recommendation|current|recent|latest|"
             r"standard\s+of\s+care|consensus|protocol)\b",
}


# =============================================================================
# Intent Prototypes for Embedding-Based Classification
# Each group has representative sentences that capture its semantic intent.
# =============================================================================

INTENT_PROTOTYPES: dict[str, list[str]] = {
    "A_FULLTEXT": [
        "What are the methods and protocol details?",
        "Describe the study design in detail",
        "What is the full text methodology?",
        "Explain the procedure step by step",
        "What are the specific experimental methods?",
    ],
    "B_DRUG": [
        "What is the drug mechanism of action?",
        "What are the drug side effects and toxicity?",
        "What is the recommended dosage?",
        "What are the pharmacokinetics?",
        "Is this medication effective for treatment?",
    ],
    "C_PROTEIN": [
        "What is the gene mutation effect?",
        "What is the protein function?",
        "How does this signaling pathway work?",
        "What is the gene expression pattern?",
        "Is this receptor involved in the disease?",
    ],
    "D_TRIAL": [
        "What did the clinical trial show?",
        "What were the phase 3 study results?",
        "Was the randomized controlled trial positive?",
        "What was the primary endpoint outcome?",
        "Is the treatment effective in trials?",
    ],
    "E_ONTOLOGY": [
        "What is the MeSH term definition?",
        "What is the Gene Ontology classification?",
        "What biological process is involved?",
        "What is the ontology hierarchy?",
        "How are these concepts related?",
    ],
    "F_WEB": [
        "What do the clinical guidelines recommend?",
        "What is the current standard of care?",
        "What are the latest treatment recommendations?",
        "What does the consensus statement say?",
        "What are the practice guidelines?",
    ],
}

# Embedding classification thresholds
INTENT_MIN_SCORE = 0.32  # Minimum cosine similarity to consider
INTENT_MARGIN = 0.04     # Margin for low-confidence detection
INTENT_TOP_K = 3         # Max groups from embedding classification


# =============================================================================
# Question Type to Default Groups Mapping
# =============================================================================

QUESTION_TYPE_DEFAULTS: dict[str, list[str]] = {
    "comparative": ["A_FULLTEXT"],
    "effect": ["B_DRUG", "C_PROTEIN"],
    "association": ["C_PROTEIN", "E_ONTOLOGY"],
    "necessity": ["D_TRIAL", "F_WEB"],
    "summary": ["A_FULLTEXT", "E_ONTOLOGY"],
    "list": ["A_FULLTEXT"],
    "factual": [],
}


# =============================================================================
# Intent Classification Prompt (Lightweight LLM fallback)
# =============================================================================

INTENT_CLASSIFIER_PROMPT = """问题: {question}

选择需要的工具组（可多选，不确定就多选）:
A. 全文深入(文章结构/方法细节)
B. 药物信息(ChEMBL)
C. 蛋白质/基因(UniProt)
D. 临床试验
E. 本体扩展(MeSH/GO)
F. 医学网搜(指南/最新)

候选提示: {weak_hints}
格式: A,C 或 空（如果核心搜索足够）"""


# =============================================================================
# Routing Log for Observability
# =============================================================================

@dataclass
class RoutingLog:
    """Record of tool routing decision."""
    question_id: str
    selected_groups: list[str]
    triggered_by: str  # "guard" | "embedding" | "llm" | "weak_rule" | "type_default" | "fallback" | "none"
    token_estimate: int
    strong_triggers_matched: list[str] = field(default_factory=list)
    weak_triggers_matched: list[str] = field(default_factory=list)
    llm_selected: list[str] = field(default_factory=list)
    # V2: Embedding classification fields
    embedding_selected: list[str] = field(default_factory=list)
    embedding_scores: dict[str, float] = field(default_factory=dict)
    guard_matched: list[str] = field(default_factory=list)
    timestamp: float = field(default_factory=time.time)


# =============================================================================
# ToolRouter Class
# =============================================================================

class ToolRouter:
    """Dynamic tool routing for RLM.

    Routes questions to appropriate tool groups to reduce token usage
    while maintaining high recall for relevant tools.

    Architecture:
    - Legacy mode (enable_embedding_classify=False): Regex-based routing
    - Hybrid mode (enable_embedding_classify=True): Embedding primary + regex guards + LLM fallback
    """

    # Maximum number of groups before falling back to full toolset
    MAX_GROUPS = 4

    def __init__(
        self,
        enable_llm_classify: bool = True,
        enable_embedding_classify: bool = False,
        use_legacy_weak_fallback: bool = False,
    ):
        """Initialize ToolRouter.

        Args:
            enable_llm_classify: Whether to use LLM for intent classification fallback
            enable_embedding_classify: Whether to use embedding-based primary classification (recommended)
            use_legacy_weak_fallback: Whether to fall back to weak triggers when no other match
        """
        self._enable_llm_classify = enable_llm_classify
        self._enable_embedding_classify = enable_embedding_classify
        self._use_legacy_weak_fallback = use_legacy_weak_fallback
        self._tool_doc_cache: dict[frozenset[str], str] = {}
        self._routing_logs: list[RoutingLog] = []
        # Embedding caches (lazy-loaded)
        self._embedding_cache: dict[str, list[float]] = {}
        self._intent_proto_cache: dict[str, list[list[float]]] | None = None

    def route(
        self,
        question: str,
        question_type: str | None = None,
        question_id: str | None = None,
    ) -> set[str]:
        """Route question to appropriate tool groups.

        Args:
            question: The biomedical question
            question_type: Optional question type for default mapping
            question_id: Optional ID for logging

        Returns:
            Set of selected tool group names
        """
        if self._enable_embedding_classify:
            return self._route_hybrid(question, question_type, question_id)
        else:
            return self._route_legacy(question, question_type, question_id)

    def _route_legacy(
        self,
        question: str,
        question_type: str | None,
        question_id: str | None,
    ) -> set[str]:
        """Legacy regex-based routing (for backward compatibility)."""
        selected: set[str] = {"CORE"}
        strong_matched: list[str] = []
        weak_matched: list[str] = []
        triggered_by = "none"

        # Step 1: Strong rule triggers (high precision)
        for group, pattern in STRONG_TRIGGERS.items():
            if re.search(pattern, question, re.IGNORECASE):
                selected.add(group)
                strong_matched.append(group)

        if strong_matched:
            triggered_by = "strong_rule"

        # Step 2: Weak rule candidates
        for group, pattern in WEAK_TRIGGERS.items():
            if group not in selected:
                if re.search(pattern, question, re.IGNORECASE):
                    weak_matched.append(group)

        # Step 3: LLM classification (only if weak candidates and no strong triggers)
        llm_selected: list[str] = []
        if weak_matched and not strong_matched and self._enable_llm_classify:
            llm_selected = self._intent_classify(question, weak_matched)
            selected.update(llm_selected)
            if llm_selected:
                triggered_by = "llm"
        elif weak_matched and not strong_matched:
            selected.update(weak_matched)
            triggered_by = "weak_rule"

        # Step 4: Question type defaults (if no other triggers)
        if len(selected) == 1 and question_type:
            defaults = QUESTION_TYPE_DEFAULTS.get(question_type, [])
            selected.update(defaults)
            if defaults:
                triggered_by = "type_default"

        # Step 5: Upper limit check
        if len(selected) > self.MAX_GROUPS:
            selected = set(TOOL_GROUP_ORDER)
            triggered_by = "fallback"

        self._log_routing(
            question_id=question_id or f"q_{hash(question) % 10000:04d}",
            selected=selected,
            triggered_by=triggered_by,
            strong_matched=strong_matched,
            weak_matched=weak_matched,
            llm_selected=llm_selected,
        )

        return selected

    def _route_hybrid(
        self,
        question: str,
        question_type: str | None,
        question_id: str | None,
    ) -> set[str]:
        """Hybrid routing: embedding primary + regex guards + LLM fallback.

        This is the recommended architecture where:
        1. Regex guards force-add groups for explicit IDs (safety net)
        2. Embedding similarity is the primary classifier (semantic understanding)
        3. LLM is the fallback for low-confidence cases
        """
        selected: set[str] = {"CORE"}
        guard_matched: list[str] = []
        lexical_strong: list[str] = []
        lexical_weak: list[str] = []
        triggered_by = "none"

        # Step 1: Regex guards (explicit IDs only - force add, never skip)
        for group, pattern in REGEX_GUARDS.items():
            if re.search(pattern, question, re.IGNORECASE):
                selected.add(group)
                guard_matched.append(group)
        if guard_matched:
            triggered_by = "guard"

        # Step 2: Collect lexical hints (for LLM fallback candidates)
        for group, pattern in STRONG_TRIGGERS.items():
            if re.search(pattern, question, re.IGNORECASE):
                lexical_strong.append(group)
        for group, pattern in WEAK_TRIGGERS.items():
            if re.search(pattern, question, re.IGNORECASE):
                lexical_weak.append(group)

        lexical_hints = sorted(set(lexical_strong + lexical_weak))

        # Step 3: Embedding primary classifier
        embedding_selected: list[str] = []
        embedding_scores: dict[str, float] = {}
        low_confidence = True

        try:
            embedding_selected, embedding_scores, low_confidence = self._embedding_classify(question)
            if embedding_selected:
                selected.update(embedding_selected)
                if triggered_by == "none":
                    triggered_by = "embedding"
        except Exception:
            # Embedding failed, will rely on LLM fallback
            pass

        # Step 4: LLM fallback for low confidence or no embedding results
        llm_selected: list[str] = []
        if self._enable_llm_classify:
            candidates = self._build_candidate_groups(lexical_hints, embedding_scores)
            if candidates and (low_confidence or not embedding_selected):
                try:
                    llm_selected = self._intent_classify(question, candidates)
                    if llm_selected:
                        selected.update(llm_selected)
                        if triggered_by in ("none", "guard"):
                            triggered_by = "llm"
                except Exception:
                    pass

        # Step 5: Legacy weak fallback (opt-in)
        if self._use_legacy_weak_fallback and not embedding_selected and not llm_selected:
            if lexical_hints:
                selected.update(lexical_hints)
                if triggered_by == "none":
                    triggered_by = "weak_rule"

        # Step 6: Question type defaults (if still only CORE)
        if len(selected) == 1 and question_type:
            defaults = QUESTION_TYPE_DEFAULTS.get(question_type, [])
            selected.update(defaults)
            if defaults and triggered_by == "none":
                triggered_by = "type_default"

        # Step 7: Upper limit check
        if len(selected) > self.MAX_GROUPS:
            selected = set(TOOL_GROUP_ORDER)
            triggered_by = "fallback"

        self._log_routing(
            question_id=question_id or f"q_{hash(question) % 10000:04d}",
            selected=selected,
            triggered_by=triggered_by,
            strong_matched=lexical_strong,
            weak_matched=lexical_weak,
            llm_selected=llm_selected,
            embedding_selected=embedding_selected,
            embedding_scores=embedding_scores,
            guard_matched=guard_matched,
        )

        return selected

    def _embedding_classify(
        self,
        question: str,
    ) -> tuple[list[str], dict[str, float], bool]:
        """Primary intent classifier using embedding similarity.

        Returns:
            (selected_groups, scores_by_group, is_low_confidence)
        """
        from ..utils.clients import embed_client

        # Get prototype embeddings (lazy-loaded)
        proto = self._get_intent_proto_embeddings()
        if not proto:
            return [], {}, True

        # Query embedding (with cache)
        query_vec = self._embedding_cache.get(question)
        if query_vec is None:
            query_vec = run_async(embed_client.embed_single(question))
            self._embedding_cache[question] = query_vec

        # Score each group by best prototype similarity
        scores: dict[str, float] = {}
        for group, vectors in proto.items():
            best = max(self._cosine_similarity(query_vec, v) for v in vectors)
            scores[group] = best

        if not scores:
            return [], {}, True

        # Determine selection and confidence
        sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        top_score = sorted_scores[0][1]
        second_score = sorted_scores[1][1] if len(sorted_scores) > 1 else 0.0
        low_confidence = top_score < INTENT_MIN_SCORE or (top_score - second_score) < INTENT_MARGIN

        selected = [
            g for g, s in sorted_scores
            if s >= INTENT_MIN_SCORE
        ][:INTENT_TOP_K]

        return selected, scores, low_confidence

    def _get_intent_proto_embeddings(self) -> dict[str, list[list[float]]]:
        """Lazy-load and cache embeddings for intent prototypes."""
        if self._intent_proto_cache is not None:
            return self._intent_proto_cache

        from ..utils.clients import embed_client

        texts: list[str] = []
        groups: list[str] = []
        for group, protos in INTENT_PROTOTYPES.items():
            for p in protos:
                texts.append(p)
                groups.append(group)

        try:
            embeddings = run_async(embed_client.embed(texts)) if texts else []
        except Exception:
            self._intent_proto_cache = {}
            return self._intent_proto_cache

        cache: dict[str, list[list[float]]] = {}
        for group, emb in zip(groups, embeddings):
            cache.setdefault(group, []).append(emb)

        self._intent_proto_cache = cache
        return cache

    @staticmethod
    def _cosine_similarity(a: list[float], b: list[float]) -> float:
        """Compute cosine similarity."""
        if not a or not b or len(a) != len(b):
            return 0.0
        dot = sum(x * y for x, y in zip(a, b))
        norm_a = math.sqrt(sum(x * x for x in a))
        norm_b = math.sqrt(sum(y * y for y in b))
        if norm_a == 0.0 or norm_b == 0.0:
            return 0.0
        return dot / (norm_a * norm_b)

    @staticmethod
    def _build_candidate_groups(
        lexical_hints: list[str],
        scores: dict[str, float],
    ) -> list[str]:
        """Merge lexical hints with top embedding candidates for LLM fallback."""
        candidates = list(dict.fromkeys(lexical_hints))
        if scores:
            top = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:INTENT_TOP_K]
            for group, _ in top:
                if group not in candidates:
                    candidates.append(group)
        return candidates

    def _intent_classify(
        self,
        question: str,
        weak_hints: list[str],
    ) -> list[str]:
        """Use lightweight LLM call to classify intent."""
        from ..utils.clients import LLMClient
        from ..config import get_config

        cfg = get_config()

        group_to_letter = {
            "A_FULLTEXT": "A",
            "B_DRUG": "B",
            "C_PROTEIN": "C",
            "D_TRIAL": "D",
            "E_ONTOLOGY": "E",
            "F_WEB": "F",
        }
        letter_to_group = {v: k for k, v in group_to_letter.items()}

        hint_letters = [group_to_letter.get(g, g) for g in weak_hints if g in group_to_letter]
        hints_str = ", ".join(hint_letters) if hint_letters else "无"

        prompt = INTENT_CLASSIFIER_PROMPT.format(
            question=question[:200],
            weak_hints=hints_str,
        )

        try:
            llm = LLMClient(base_urls=cfg.llm.servers, model=cfg.llm.model)
            response = run_async(llm.chat(prompt, max_tokens=20))

            selected = []
            if response:
                response = response.strip().upper()
                for letter in response.replace(",", " ").split():
                    letter = letter.strip()
                    if letter in letter_to_group:
                        selected.append(letter_to_group[letter])

            return selected
        except Exception:
            return weak_hints

    def _log_routing(
        self,
        question_id: str,
        selected: set[str],
        triggered_by: str,
        strong_matched: list[str],
        weak_matched: list[str],
        llm_selected: list[str],
        embedding_selected: list[str] | None = None,
        embedding_scores: dict[str, float] | None = None,
        guard_matched: list[str] | None = None,
    ) -> None:
        """Log routing decision for observability."""
        token_estimate = sum(
            TOOL_GROUPS[g]["tokens"]
            for g in selected
            if g in TOOL_GROUPS
        )

        log = RoutingLog(
            question_id=question_id,
            selected_groups=sorted(selected),
            triggered_by=triggered_by,
            token_estimate=token_estimate,
            strong_triggers_matched=strong_matched,
            weak_triggers_matched=weak_matched,
            llm_selected=llm_selected,
            embedding_selected=embedding_selected or [],
            embedding_scores=embedding_scores or {},
            guard_matched=guard_matched or [],
        )
        self._routing_logs.append(log)

    def build_prompt(
        self,
        question: str,
        question_type: str | None = None,
    ) -> str:
        """Build dynamic system prompt with selected tools.

        Generates:
        - Dynamic tool documentation (only selected groups)
        - Dynamic tool selection rules (only relevant rules)
        - Dynamic STEP 2 code (conditional tool usage snippets)
        """
        groups = self.route(question, question_type)
        tool_docs = self._get_tool_docs(groups)
        selection_rules = self._get_selection_rules(groups)
        step2_extra = self._get_step2_code(groups)
        return (
            V13_BASE_PROMPT
            .replace("__TOOL_DOCS__", tool_docs)
            .replace("__SELECTION_RULES__", selection_rules)
            .replace("__STEP2_EXTRA__", step2_extra)
        )

    def _get_tool_docs(self, groups: set[str]) -> str:
        """Get tool documentation for selected groups (cached)."""
        cache_key = frozenset(groups)
        if cache_key in self._tool_doc_cache:
            return self._tool_doc_cache[cache_key]

        docs_parts = []
        for group_name in TOOL_GROUP_ORDER:
            if group_name in groups:
                group = TOOL_GROUPS.get(group_name)
                if group:
                    docs_parts.append(group["tool_docs"])

        docs = "\n\n".join(docs_parts)
        self._tool_doc_cache[cache_key] = docs
        return docs

    def _get_selection_rules(self, groups: set[str]) -> str:
        """Generate dynamic TOOL SELECTION RULES based on selected groups."""
        rules = []
        idx = 1
        for group_name in TOOL_GROUP_ORDER:
            if group_name in groups:
                group = TOOL_GROUPS.get(group_name)
                if group:
                    for rule in group.get("selection_rules", []):
                        rules.append(f"{idx}. {rule}")
                        idx += 1
        if not rules:
            return "1. Use search_papers for all queries"
        return "\n".join(rules)

    def _get_step2_code(self, groups: set[str]) -> str:
        """Generate dynamic STEP 2 code snippets for selected non-core groups.

        These snippets are injected into STEP 2 of the workflow, providing
        explicit code for the LLM to use specialized tools.
        """
        snippets = []
        # Order: E_ONTOLOGY first (query expansion), then C_PROTEIN, B_DRUG, D_TRIAL, F_WEB, A_FULLTEXT last
        step2_order = ["E_ONTOLOGY", "C_PROTEIN", "B_DRUG", "D_TRIAL", "F_WEB", "A_FULLTEXT"]
        for group_name in step2_order:
            if group_name in groups:
                group = TOOL_GROUPS.get(group_name)
                if group and group.get("step2_snippet"):
                    snippets.append(group["step2_snippet"])
        return "\n".join(snippets) if snippets else ""

    def get_routing_logs(self) -> list[RoutingLog]:
        """Get all routing logs for analysis."""
        return list(self._routing_logs)

    def clear_routing_logs(self) -> None:
        """Clear routing logs."""
        self._routing_logs.clear()

    def get_stats(self) -> dict[str, Any]:
        """Get routing statistics."""
        if not self._routing_logs:
            return {"total_routes": 0}

        total = len(self._routing_logs)
        trigger_counts: dict[str, int] = {}
        total_tokens = 0
        group_counts: dict[str, int] = {}

        for log in self._routing_logs:
            trigger_counts[log.triggered_by] = trigger_counts.get(log.triggered_by, 0) + 1
            total_tokens += log.token_estimate
            for g in log.selected_groups:
                group_counts[g] = group_counts.get(g, 0) + 1

        return {
            "total_routes": total,
            "avg_tokens": total_tokens / total,
            "trigger_distribution": trigger_counts,
            "group_usage": group_counts,
        }

    def validate_tools(self, groups: set[str]) -> tuple[bool, list[str]]:
        """Validate that all tools in selected groups exist."""
        from .tools import RETRIEVAL_TOOLS

        missing = []
        for group_name in groups:
            group = TOOL_GROUPS.get(group_name)
            if group:
                for tool in group["tools"]:
                    if tool not in RETRIEVAL_TOOLS and tool != "llm_query":
                        missing.append(f"{group_name}:{tool}")

        return len(missing) == 0, missing


# =============================================================================
# V13 Base Prompt Template
# =============================================================================

V13_BASE_PROMPT = """You are a biomedical QA agent operating inside a REPL that can execute Python.

## AVAILABLE TOOLS

__TOOL_DOCS__

## TOOL SELECTION RULES (check in order)
__SELECTION_RULES__

## YOUR TASK
Answer yes/no/maybe questions using **semantic stance classification** instead of keyword matching.

## CORE INSIGHT (V13.1)
PubMedQA "no" answers are often expressed through **semantic negation**, not explicit negation words:
- "Is X better than Y?" → "no" = "similar results" (COMPARATIVE)
- "Is X necessary?" → "no" = "does not support routine use" (NECESSITY)
- "Does X alter Y?" → "no" = "unchanged/stable/no effect" (EFFECT)
- "Is X associated?" → "no" = "insufficient evidence" (ASSOCIATION)

**Key rules**:
- COMPARATIVE: "similar/equivalent/no difference" = CONTRADICT
- EFFECT: "unchanged/stable/no effect/did not alter" = CONTRADICT
- NECESSITY: "not routine/not standard of care" = CONTRADICT

## WORKFLOW

```repl
import json

# Get question from context
question = context.get("question", "") if isinstance(context, dict) else str(context)
question_type = context.get("question_type", "unknown") if isinstance(context, dict) else "unknown"
options = context.get("options", None) if isinstance(context, dict) else None
print(f"Question: {{question[:200]}}...")
print(f"Type: {{question_type}}")

# === STEP 1: QUESTION TYPE ANALYSIS (V13.1: Added effect type) ===
qtype_prompt = f\'\'\'You are a biomedical QA question analyzer.
Given a yes/no question, output JSON ONLY:
{{
  "question_type": "comparative" | "necessity" | "effect" | "association" | "factual",
  "hypothesis": "...",
  "polarity_indicators": ["..."]
}}

Rules (check in order):
1. EFFECT: asks if X alters/affects/changes/impacts/modifies Y
   - Keywords: alter, affect, change, impact, modify, influence, impair
   - Example: "Does desflurane alter left ventricular function?"
   - Hypothesis: "X alters/affects Y"

2. COMPARATIVE: asks if X is better/superior/more effective than Y
   - Keywords: better, superior, more effective, improved, preferred
   - Example: "Is late-night cortisol a better screening test?"
   - Hypothesis: "X is better than Y"

3. NECESSITY: asks if X is necessary/required/routine/indicated/recommended
   - Keywords: necessary, required, routine, indicated, recommended, standard of care
   - Example: "Is adjuvant radiation necessary for stage III thymoma?"
   - Hypothesis: "X is necessary/required"

4. ASSOCIATION: asks if X is associated/linked/correlated with Y
   - Keywords: associated, linked, correlated, risk factor, predictor
   - Example: "Is obesity associated with diabetes?"
   - Hypothesis: "X is associated with Y"

5. FACTUAL: other yes/no questions that don\'t fit above categories

Question: "{{question}}"
\'\'\'

qtype_result = llm_query(qtype_prompt)
print("=== QUESTION TYPE ANALYSIS ===")
print(qtype_result)

try:
    qtype_data = json.loads(qtype_result) if isinstance(qtype_result, str) else qtype_result
except:
    qtype_data = {{"question_type": "factual", "hypothesis": question, "polarity_indicators": []}}
```

```repl
# === STEP 2: RETRIEVE EVIDENCE (Dynamic Multi-Source) ===
import re

evidence_parts = []
expanded_query = question
__STEP2_EXTRA__
# Core: Search PubMed abstracts (always executed)
papers = search_papers(expanded_query, limit=100)
print(f"Retrieved {{len(papers)}} papers from PubMed (expanded: {{expanded_query != question}})")

if papers:
    reranked = rerank_papers(question, papers, top_k=10)
    evidence_text = format_papers(reranked, max_papers=8)
    evidence_parts.append(f"PubMed Evidence:\\n{{evidence_text}}")
else:
    evidence_text = "No PubMed evidence found."
    print("WARNING: No papers retrieved")

# Combine all evidence
all_evidence = "\\n\\n".join(evidence_parts) if evidence_parts else "No evidence found."
print(f"Total evidence sources: {{len(evidence_parts)}}")
```

```repl
# === STEP 3: SEMANTIC STANCE CLASSIFICATION ===
q_type = qtype_data.get("question_type", "factual")
hypothesis = qtype_data.get("hypothesis", question)

stance_prompt = f\'\'\'You are a biomedical evidence stance classifier.

Task: Classify each evidence statement\'s stance toward the hypothesis.

Hypothesis: "{{hypothesis}}"
Question type: {{q_type}}

Evidence:
{{all_evidence}}

CRITICAL RULES for stance classification (V13.1):

1. For EFFECT questions ("Does X alter/affect/change Y?"):
   - "unchanged", "stable", "no change", "did not alter", "no effect on" → CONTRADICT
   - "remained stable", "no significant change", "preserved", "maintained" → CONTRADICT
   - "altered", "affected", "changed", "impaired", "modified" → SUPPORT

2. For COMPARATIVE questions ("Is X better/superior/more effective?"):
   - "similar", "equivalent", "no difference", "comparable", "not superior" → CONTRADICT
   - "non-inferior but not superior" → CONTRADICT (not better = no)
   - "better", "superior", "more effective", "improved" → SUPPORT

3. For NECESSITY questions ("Is X necessary/required/routine?"):
   - "not routine", "not indicated", "not recommended" → CONTRADICT
   - "does not support routine use", "no clear indication" → CONTRADICT
   - "not standard of care", "optional", "selective use" → CONTRADICT
   - "recommended", "indicated", "necessary", "standard of care" → SUPPORT

4. For ASSOCIATION questions ("Is X associated with Y?"):
   - "no association", "not correlated", "not a risk factor" → CONTRADICT
   - "not an independent predictor", "lost significance" → CONTRADICT
   - "associated", "correlated", "risk factor", "predictor" → SUPPORT

5. General rules (apply to all types):
   - "insufficient evidence", "inconclusive", "unclear" → INSUFFICIENT
   - "limited evidence" without direction → INSUFFICIENT
   - "mixed results", "conflicting" → INSUFFICIENT
   - Only SUPPORT if evidence POSITIVELY and DIRECTLY supports hypothesis
   - Strength = "strong" if explicit conclusion; "weak" if hedged

Output JSON:
{{
  "evidence_stances": [
    {{"sentence": "key sentence", "stance": "support|contradict|insufficient", "strength": "strong|weak", "signal": "signal word"}},
    ...
  ],
  "summary": {{
    "support_count": N,
    "contradict_count": N,
    "insufficient_count": N,
    "has_strong_contradict": true/false,
    "has_strong_support": true/false
  }}
}}
\'\'\'

stance_result = llm_query(stance_prompt)
print("=== STANCE CLASSIFICATION ===")
print(stance_result)

try:
    stance_data = json.loads(stance_result) if isinstance(stance_result, str) else stance_result
except:
    stance_data = {{"summary": {{"support_count": 0, "contradict_count": 0, "insufficient_count": 0}}}}
```

```repl
# === STEP 4: STANCE-BASED DECISION (V13.1 Logic) ===
summary = stance_data.get("summary", {{}})
support = summary.get("support_count", 0)
contradict = summary.get("contradict_count", 0)
insufficient = summary.get("insufficient_count", 0)
has_strong_contradict = summary.get("has_strong_contradict", False)
has_strong_support = summary.get("has_strong_support", False)

# V13.1 Decision Rules (contradict-priority to reduce yes-bias)
# Rule 1: Strong contradict always wins
if has_strong_contradict:
    final_answer = "no"
    reason = "Strong contradicting evidence found"
# Rule 2: Strong support only if NO contradiction
elif has_strong_support and contradict == 0:
    final_answer = "yes"
    reason = "Strong supporting evidence, no contradiction"
# Rule 3: Mixed strong signals = maybe
elif has_strong_support and contradict > 0:
    final_answer = "maybe"
    reason = "Mixed evidence: both support and contradict"
# Rule 4: Any contradict without support = no
elif contradict > 0 and support == 0:
    final_answer = "no"
    reason = "Contradicting evidence only, no support"
# Rule 5: Weak support only if truly no other signals
elif support > 0 and contradict == 0 and insufficient == 0:
    final_answer = "yes"
    reason = "Weak support but no contradiction"
# Rule 6: Predominantly insufficient = maybe
elif insufficient > support and insufficient > contradict:
    final_answer = "maybe"
    reason = "Predominantly insufficient evidence"
# Rule 7: Weak contradict > weak support (conservative)
elif contradict > 0:
    final_answer = "no"
    reason = "Weak contradict outweighs weak support"
else:
    final_answer = "maybe"
    reason = "Unclear or mixed signals"

print(f"=== V13.1 DECISION ===")
print(f"Support: {{support}}, Contradict: {{contradict}}, Insufficient: {{insufficient}}")
print(f"Strong contradict: {{has_strong_contradict}}, Strong support: {{has_strong_support}}")
print(f"Final answer: {{final_answer}}")
print(f"Reason: {{reason}}")

state = {{
    "answer": final_answer,
    "question_type": q_type,
    "hypothesis": hypothesis,
    "stance_summary": summary,
    "reasoning": reason
}}
```

```repl
# === STEP 5: OUTPUT ===
print("=== FINAL OUTPUT ===")
answer = json.dumps(state)
print(answer)
FINAL_VAR("answer")
```

## CRITICAL RULES

1. **ALWAYS call FINAL_VAR("answer")** in STEP 5 to report your answer - this is REQUIRED
2. **ALWAYS use the provided tools** (search_papers, llm_query, etc.) - do NOT fabricate evidence
3. **Follow ALL 5 steps** exactly as shown in WORKFLOW above

## KEY PRINCIPLES (V13.1)

1. **SEMANTIC STANCE**: Classify evidence by meaning, not keywords
2. **5 QUESTION TYPES**: effect, comparative, necessity, association, factual
3. **EFFECT TYPE**: "unchanged/stable" = CONTRADICT for "Does X alter Y?" questions
4. **CONTRADICT-PRIORITY**: Strong contradict → "no" (reduces yes-bias)
5. **CONSERVATIVE RULE**: Weak contradict > weak support → "no"
6. **INSUFFICIENT ≠ SUPPORT**: "limited evidence" should NOT support "yes"
"""


# =============================================================================
# Convenience Functions
# =============================================================================

def route_question(
    question: str,
    question_type: str | None = None,
    use_embedding: bool = False,
) -> set[str]:
    """Convenience function to route a question.

    Args:
        question: The biomedical question
        question_type: Optional question type
        use_embedding: Whether to use embedding-based classification

    Returns:
        Set of selected tool group names
    """
    router = ToolRouter(enable_embedding_classify=use_embedding)
    return router.route(question, question_type)


def build_dynamic_prompt(
    question: str,
    question_type: str | None = None,
    use_embedding: bool = False,
) -> str:
    """Convenience function to build dynamic prompt.

    Args:
        question: The biomedical question
        question_type: Optional question type
        use_embedding: Whether to use embedding-based classification

    Returns:
        Dynamic system prompt with selected tools
    """
    router = ToolRouter(enable_embedding_classify=use_embedding)
    return router.build_prompt(question, question_type)
