"""Budget-aware assembly of full-text evidence units.

V14.2 keeps the v14.1 retrieval policy:
1. Recall papers from abstracts
2. Restrict full-text search to the candidate papers that actually have full text
3. Search chunks with section-aware priors

This module only changes how retrieved full-text chunks are assembled before
they are passed into the agent. It converts raw chunks into short evidence
units, applies section filtering, preserves polarity for yes/no questions, and
packs the final context into a bounded token budget.
"""

from __future__ import annotations

import math
import re
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Iterable, Sequence


BLOCKED_TITLE_PATTERNS = (
    r"\backnowledg",
    r"\bfunding\b",
    r"\bconflicts?\b",
    r"\bcompeting interests?\b",
    r"\bdisclosures?\b",
    r"\bauthor contributions?\b",
    r"\breferences?\b",
    r"\bbibliograph",
    r"\bsupplement",
    r"\bappendix\b",
    r"\bethics?\b",
    r"\bdata availability\b",
    r"\bcoi\b",
)

BLOCKED_ROLES = {
    "acknowledgement",
    "acknowledgment",
    "funding",
    "references",
    "reference",
    "supplement",
    "supplementary",
    "conflict",
    "coi",
}

YESNO_ROLE_WEIGHT = {
    "results": 1.15,
    "conclusion": 1.10,
    "discussion": 1.05,
    "abstract": 0.88,
    "methods": 0.70,
    "background": 0.62,
    "introduction": 0.60,
    "other": 0.72,
}

GENERAL_ROLE_WEIGHT = {
    "results": 1.10,
    "conclusion": 1.08,
    "discussion": 1.04,
    "abstract": 0.95,
    "methods": 0.82,
    "background": 0.70,
    "introduction": 0.68,
    "other": 0.80,
}


def _estimate_tokens(text: str) -> int:
    return max(1, math.ceil(len(text) / 4))


def _normalize_space(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "")).strip()


def _sentence_split(text: str) -> list[str]:
    parts = re.split(r"(?<=[.!?])\s+", text)
    return [part.strip() for part in parts if part.strip()]


def _compress_text(text: str, *, max_sentences: int, max_chars: int) -> str:
    clean = _normalize_space(text)
    if not clean:
        return ""

    sentences = _sentence_split(clean)
    if not sentences:
        return clean[:max_chars].rstrip(" ,;:")

    kept = []
    chars = 0
    for sentence in sentences:
        if len(kept) >= max_sentences:
            break
        extra = len(sentence) + (1 if kept else 0)
        if kept and chars + extra > max_chars:
            break
        if not kept and len(sentence) > max_chars:
            kept.append(sentence[:max_chars].rstrip(" ,;:"))
            chars = len(kept[0])
            break
        kept.append(sentence)
        chars += extra

    if not kept:
        kept = [clean[:max_chars].rstrip(" ,;:")]

    result = " ".join(kept)
    return result[:max_chars].rstrip(" ,;:")


def _blocked_section(role: str, title: str) -> bool:
    role_l = (role or "").lower().strip()
    title_l = (title or "").lower().strip()
    if role_l in BLOCKED_ROLES:
        return True
    raw = f"{role_l} {title_l}".strip()
    return any(re.search(pattern, raw) for pattern in BLOCKED_TITLE_PATTERNS)


@dataclass
class EvidenceUnit:
    pmid: str
    section_id: str
    section_role: str
    section_title: str
    polarity: str
    retrieval_stages: tuple[str, ...]
    score: float
    text: str
    token_estimate: int


class FulltextEvidenceAssembler:
    """Convert raw full-text chunks into short, budget-aware evidence units."""

    def __init__(
        self,
        *,
        evidence_token_budget: int = 6000,
        per_paper_cap: int = 2,
        max_sentences_per_unit: int = 2,
        max_chars_per_unit: int = 420,
    ):
        self._evidence_token_budget = evidence_token_budget
        self._per_paper_cap = per_paper_cap
        self._max_sentences_per_unit = max_sentences_per_unit
        self._max_chars_per_unit = max_chars_per_unit

    @property
    def evidence_token_budget(self) -> int:
        return self._evidence_token_budget

    def _unit_role_weight(self, role: str, question_type: str) -> float:
        table = YESNO_ROLE_WEIGHT if question_type == "yesno" else GENERAL_ROLE_WEIGHT
        return table.get((role or "other").lower().strip(), table["other"])

    def _build_units(
        self,
        chunks: Sequence[dict[str, Any]],
        *,
        polarity: str,
        question_type: str,
    ) -> list[EvidenceUnit]:
        grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)

        for chunk in chunks:
            role = (chunk.get("section_role") or chunk.get("section_type") or "other").lower().strip() or "other"
            title = chunk.get("section_title") or ""
            if _blocked_section(role, title):
                continue
            text = _normalize_space(chunk.get("text") or "")
            if len(text) < 80:
                continue
            key = (
                str(chunk.get("pmid") or ""),
                str(chunk.get("section_id") or ""),
                polarity,
            )
            grouped[key].append({**chunk, "section_role": role, "section_title": title, "text": text})

        units: list[EvidenceUnit] = []
        for (pmid, section_id, unit_polarity), group in grouped.items():
            group.sort(
                key=lambda item: (
                    -(item.get("adjusted_score", item.get("score", 0.0)) or 0.0),
                    item.get("sequence_order", 0),
                )
            )
            primary = group[0]
            selected = [primary]

            # Merge one adjacent hit from the same section when available.
            for candidate in group[1:]:
                if abs((candidate.get("sequence_order", 0) or 0) - (primary.get("sequence_order", 0) or 0)) <= 1:
                    selected.append(candidate)
                    break

            texts = []
            seen_texts = set()
            for item in sorted(selected, key=lambda x: x.get("sequence_order", 0)):
                text = item["text"]
                if text not in seen_texts:
                    seen_texts.add(text)
                    texts.append(text)

            compressed = _compress_text(
                " ".join(texts),
                max_sentences=self._max_sentences_per_unit,
                max_chars=self._max_chars_per_unit,
            )
            if not compressed:
                continue

            score = (primary.get("adjusted_score", primary.get("score", 0.0)) or 0.0)
            score *= self._unit_role_weight(primary["section_role"], question_type)
            stages = tuple(
                sorted(
                    {
                        str(item.get("retrieval_stage") or "seed")
                        for item in selected
                        if item.get("retrieval_stage")
                    }
                    or {"seed"}
                )
            )
            units.append(
                EvidenceUnit(
                    pmid=pmid or "unknown",
                    section_id=section_id,
                    section_role=primary["section_role"],
                    section_title=primary["section_title"],
                    polarity=unit_polarity,
                    retrieval_stages=stages,
                    score=score,
                    text=compressed,
                    token_estimate=_estimate_tokens(compressed),
                )
            )

        units.sort(key=lambda unit: unit.score, reverse=True)
        return units

    def _select_units(
        self,
        units: Sequence[EvidenceUnit],
        *,
        budget_tokens: int,
        max_units: int,
    ) -> tuple[list[EvidenceUnit], int]:
        selected: list[EvidenceUnit] = []
        paper_counts: dict[str, int] = defaultdict(int)
        used_tokens = 0

        for unit in units:
            if len(selected) >= max_units:
                break
            if paper_counts[unit.pmid] >= self._per_paper_cap:
                continue

            unit_cost = unit.token_estimate + 18  # header/formatting overhead
            if selected and used_tokens + unit_cost > budget_tokens:
                continue
            if not selected and unit_cost > budget_tokens:
                # Keep at least one unit by truncation semantics upstream.
                selected.append(unit)
                used_tokens += unit_cost
                paper_counts[unit.pmid] += 1
                break

            selected.append(unit)
            used_tokens += unit_cost
            paper_counts[unit.pmid] += 1

        return selected, used_tokens

    @staticmethod
    def _format_unit(unit: EvidenceUnit, index: int) -> str:
        title = f" | title:{unit.section_title}" if unit.section_title else ""
        stages = f" | stage:{'/'.join(unit.retrieval_stages)}" if unit.retrieval_stages else ""
        return (
            f"{index}. [PMID:{unit.pmid} | role:{unit.section_role}{title}{stages}] "
            f"{unit.text}"
        )

    def assemble(
        self,
        *,
        question_type: str,
        pos_chunks: Sequence[dict[str, Any]],
        neg_chunks: Sequence[dict[str, Any]] | None = None,
        abstract_evidence_text: str = "",
    ) -> tuple[str, dict[str, Any]]:
        pos_units = self._build_units(pos_chunks, polarity="support", question_type=question_type)
        neg_units = self._build_units(neg_chunks or [], polarity="refute", question_type=question_type)

        if question_type == "yesno":
            support_budget = int(self._evidence_token_budget * 0.46)
            refute_budget = int(self._evidence_token_budget * 0.46)
            note_budget = self._evidence_token_budget - support_budget - refute_budget

            support_units, support_tokens = self._select_units(
                pos_units, budget_tokens=support_budget, max_units=4
            )
            refute_units, refute_tokens = self._select_units(
                neg_units, budget_tokens=refute_budget, max_units=4
            )

            if not support_units and not refute_units:
                return abstract_evidence_text, {
                    "assembly_mode": "abstract_fallback",
                    "assembly_budget_tokens": self._evidence_token_budget,
                    "assembly_estimated_tokens": _estimate_tokens(abstract_evidence_text),
                    "assembled_units": 0,
                    "assembled_support_units": 0,
                    "assembled_refute_units": 0,
                    "assembled_pmids": 0,
                }

            blocks = ["## Full-Text Evidence Units"]

            blocks.append("\n### Supporting evidence")
            if support_units:
                blocks.extend(self._format_unit(unit, idx) for idx, unit in enumerate(support_units, 1))
            else:
                blocks.append("1. No clear supporting full-text units were selected from the candidate full-text papers.")

            blocks.append("\n### Refuting evidence")
            if refute_units:
                blocks.extend(self._format_unit(unit, idx) for idx, unit in enumerate(refute_units, 1))
            else:
                blocks.append("1. No clear refuting full-text units were selected from the candidate full-text papers.")

            evidence_text = "\n".join(blocks)
            if _estimate_tokens(evidence_text) > self._evidence_token_budget + note_budget:
                evidence_text = evidence_text[: (self._evidence_token_budget + note_budget) * 4]

            selected = support_units + refute_units
            return evidence_text, {
                "assembly_mode": "yesno_polarity_units",
                "assembly_budget_tokens": self._evidence_token_budget,
                "assembly_estimated_tokens": _estimate_tokens(evidence_text),
                "assembled_units": len(selected),
                "assembled_support_units": len(support_units),
                "assembled_refute_units": len(refute_units),
                "assembled_pmids": len({unit.pmid for unit in selected}),
                "assembled_roles": sorted({unit.section_role for unit in selected}),
            }

        selected_units, used_tokens = self._select_units(
            pos_units,
            budget_tokens=self._evidence_token_budget,
            max_units=6 if question_type in {"factoid", "expression"} else 7,
        )
        if not selected_units:
            return abstract_evidence_text, {
                "assembly_mode": "abstract_fallback",
                "assembly_budget_tokens": self._evidence_token_budget,
                "assembly_estimated_tokens": _estimate_tokens(abstract_evidence_text),
                "assembled_units": 0,
                "assembled_support_units": 0,
                "assembled_refute_units": 0,
                "assembled_pmids": 0,
            }

        blocks = ["## Full-Text Evidence Units"]
        blocks.extend(self._format_unit(unit, idx) for idx, unit in enumerate(selected_units, 1))
        evidence_text = "\n".join(blocks)
        return evidence_text, {
            "assembly_mode": "budgeted_units",
            "assembly_budget_tokens": self._evidence_token_budget,
            "assembly_estimated_tokens": min(_estimate_tokens(evidence_text), used_tokens),
            "assembled_units": len(selected_units),
            "assembled_support_units": len(selected_units),
            "assembled_refute_units": 0,
            "assembled_pmids": len({unit.pmid for unit in selected_units}),
            "assembled_roles": sorted({unit.section_role for unit in selected_units}),
        }
