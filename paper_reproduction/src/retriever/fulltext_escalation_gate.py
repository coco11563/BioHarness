"""Selective gate for supplementing v14 with full-text evidence.

The gate is intentionally conservative:
- Keep current v14 abstract-first behavior as the default.
- Only escalate when the question type and full-text coverage suggest that
  deeper reading is likely to help.
- Only keep full-text augmentation when the retrieved/assembled evidence has
  enough diversity and section quality to justify the extra context.
"""

from __future__ import annotations

from dataclasses import dataclass
import re


ELIGIBLE_ALWAYS = {"list", "summary"}
LOW_SIGNAL_ROLES = {"introduction", "background", "methods", "other"}
DEPTH_PATTERN = re.compile(
    r"\b("
    r"mechanism|pathway|procedure|protocol|full.?text|section|"
    r"experimental|experiment|trial|outcome|result|results|"
    r"how does|how do|why does|why do|compare|versus|vs\.?"
    r")\b",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class FulltextEscalationDecision:
    escalate: bool
    reason: str
    mode: str = "abstract_only"


class FulltextEscalationGate:
    """Conservative gate for v14.3 full-text supplementation."""

    def __init__(
        self,
        *,
        min_candidate_fulltext: int = 3,
        min_yesno_pmids: int = 2,
        min_synthesis_pmids: int = 2,
    ):
        self._min_candidate_fulltext = min_candidate_fulltext
        self._min_yesno_pmids = min_yesno_pmids
        self._min_synthesis_pmids = min_synthesis_pmids

    @staticmethod
    def _needs_deep_read(question: str) -> bool:
        return bool(DEPTH_PATTERN.search(question or ""))

    def precheck(
        self,
        *,
        question: str,
        question_type: str,
        candidate_have_pmc_in_articles: int,
        candidate_in_papergraph: int,
    ) -> FulltextEscalationDecision:
        if candidate_in_papergraph <= 0:
            return FulltextEscalationDecision(False, "coverage_miss")

        if question_type in ELIGIBLE_ALWAYS:
            if candidate_in_papergraph < self._min_candidate_fulltext:
                return FulltextEscalationDecision(False, "insufficient_fulltext_candidates")
            return FulltextEscalationDecision(True, "eligible_synthesis_question", "section_aware")

        if question_type == "yesno":
            if not self._needs_deep_read(question):
                return FulltextEscalationDecision(False, "yesno_default_to_abstract")
            if candidate_in_papergraph < self._min_candidate_fulltext:
                return FulltextEscalationDecision(False, "insufficient_fulltext_candidates")
            if candidate_have_pmc_in_articles < max(2, self._min_candidate_fulltext - 1):
                return FulltextEscalationDecision(False, "insufficient_pmc_overlap")
            return FulltextEscalationDecision(True, "eligible_yesno_deep_read", "section_aware")

        if question_type in {"factoid", "expression"}:
            if not self._needs_deep_read(question):
                return FulltextEscalationDecision(False, "factoid_default_to_abstract")
            if candidate_in_papergraph < self._min_candidate_fulltext:
                return FulltextEscalationDecision(False, "insufficient_fulltext_candidates")
            return FulltextEscalationDecision(True, "eligible_detail_question", "section_aware")

        return FulltextEscalationDecision(False, "question_type_not_eligible")

    def postcheck(
        self,
        *,
        question_type: str,
        distinct_pmids_in_section_chunks: int,
        section_aware_roles: list[str] | tuple[str, ...],
        assembly_mode: str,
        assembled_units: int,
        assembled_support_units: int,
        assembled_refute_units: int,
        assembled_pmids: int,
    ) -> FulltextEscalationDecision:
        if assembly_mode in {"abstract_fallback", "coverage_miss", "retrieval_miss"}:
            return FulltextEscalationDecision(False, f"assembly_{assembly_mode}")

        roles = {role for role in section_aware_roles if role}
        if roles and roles.issubset(LOW_SIGNAL_ROLES):
            return FulltextEscalationDecision(False, "low_signal_sections_only")

        if question_type == "yesno":
            if assembled_support_units <= 0 or assembled_refute_units <= 0:
                return FulltextEscalationDecision(False, "missing_polarity_balance")
            if max(assembled_pmids, distinct_pmids_in_section_chunks) < self._min_yesno_pmids:
                return FulltextEscalationDecision(False, "insufficient_cross_paper_support")
            return FulltextEscalationDecision(True, "balanced_yesno_fulltext", "section_aware")

        if question_type in ELIGIBLE_ALWAYS:
            if assembled_units < 3:
                return FulltextEscalationDecision(False, "too_few_evidence_units")
            if max(assembled_pmids, distinct_pmids_in_section_chunks) < self._min_synthesis_pmids:
                return FulltextEscalationDecision(False, "insufficient_cross_paper_support")
            return FulltextEscalationDecision(True, "synthesis_fulltext_ready", "section_aware")

        if question_type in {"factoid", "expression"}:
            if assembled_units < 2:
                return FulltextEscalationDecision(False, "too_few_evidence_units")
            return FulltextEscalationDecision(True, "detail_fulltext_ready", "section_aware")

        return FulltextEscalationDecision(False, "question_type_not_eligible")
