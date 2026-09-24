"""EvidenceState - Stateful evidence collection for adaptive RLM workflow.

RLM Philosophy:
- Model controls the entire workflow through code
- EvidenceState tracks what evidence has been collected
- Tools are atomic and composable
- Model decides when to expand, drill down, or finalize

Usage in REPL:
    state = EvidenceState(query=question)

    # Initial search
    hits = search_papers(question, limit=100)
    top5 = rerank_papers(question, hits, top_k=5)
    state.add_papers(top5)

    # Judge and iterate
    judgment = judge_evidence(state)
    if judgment['status'] == 'need_more':
        # Expand...

    # Format final evidence
    evidence = state.format_evidence(max_tokens=4000)
"""

from dataclasses import dataclass, field
from typing import Any


@dataclass
class EvidenceSnippet:
    """A piece of evidence from a paper."""
    paper_id: str
    text: str
    source: str  # "abstract" | "chunk" | "section"
    score: float = 0.0
    section_path: str | None = None
    stance: str | None = None  # "support" | "refute" | "neutral"
    meta: dict[str, Any] = field(default_factory=dict)

    def token_cost(self) -> int:
        """Estimate token cost (rough: 1 token per 4 chars)."""
        return max(1, len(self.text) // 4)


@dataclass
class EvidenceState:
    """Tracks evidence collection progress for adaptive search.

    The model uses this state object to:
    1. Track which papers have been retrieved
    2. Accumulate evidence snippets
    3. Monitor token budget
    4. Record gaps that need more evidence
    """
    query: str
    question_type: str = "yesno"

    # Retrieved papers (by ID)
    paper_ids: set[str] = field(default_factory=set)
    paper_cache: dict[str, dict] = field(default_factory=dict)  # id -> paper metadata

    # Collected evidence
    snippets: list[EvidenceSnippet] = field(default_factory=list)

    # Token tracking
    token_used: int = 0
    token_budget: int = 8000  # Default budget

    # Gaps and status
    gaps: list[str] = field(default_factory=list)
    status: str = "searching"  # "searching" | "enough" | "exhausted"

    # Iteration tracking
    iterations: int = 0
    max_iterations: int = 5

    def add_papers(self, papers: list[dict]) -> int:
        """Add papers to retrieved set. Returns count of new papers."""
        new_count = 0
        for p in papers:
            pid = p.get("id") or p.get("pmid") or p.get("paper_id")
            if pid and pid not in self.paper_ids:
                self.paper_ids.add(str(pid))
                self.paper_cache[str(pid)] = p
                new_count += 1
        return new_count

    def add_snippet(self, snippet: EvidenceSnippet) -> bool:
        """Add a snippet if within budget. Returns True if added."""
        cost = snippet.token_cost()
        if self.token_used + cost > self.token_budget:
            return False
        self.snippets.append(snippet)
        self.token_used += cost
        return True

    def add_snippets(self, snippets: list[EvidenceSnippet]) -> int:
        """Add multiple snippets. Returns count added."""
        added = 0
        for s in snippets:
            if self.add_snippet(s):
                added += 1
        return added

    def add_abstracts(self, papers: list[dict]) -> int:
        """Convenience: add paper abstracts as snippets."""
        added = 0
        for p in papers:
            pid = str(p.get("id") or p.get("pmid") or p.get("paper_id"))
            abstract = p.get("abstract", "")
            if not abstract:
                continue
            snippet = EvidenceSnippet(
                paper_id=pid,
                text=abstract,
                source="abstract",
                score=p.get("score", 0.0),
                meta={"title": p.get("title", "")}
            )
            if self.add_snippet(snippet):
                added += 1
        return added

    def add_chunks(self, chunks: list[dict], paper_id: str) -> int:
        """Add chunks from fulltext as snippets."""
        added = 0
        for c in chunks:
            snippet = EvidenceSnippet(
                paper_id=paper_id,
                text=c.get("content", c.get("text", "")),
                source="chunk",
                score=c.get("score", 0.0),
                section_path=c.get("section_path", c.get("section_id", "")),
            )
            if self.add_snippet(snippet):
                added += 1
        return added

    def papers_with_fulltext(self) -> list[dict]:
        """Get papers that have fulltext available."""
        return [
            p for p in self.paper_cache.values()
            if p.get("have_fulltext")
        ]

    def get_evidence_texts(self) -> list[str]:
        """Get all snippet texts for judge_evidence."""
        return [s.text for s in self.snippets]

    def format_evidence(self, max_tokens: int | None = None) -> str:
        """Format collected evidence for LLM context.

        Sorts by score and packs within token limit.
        """
        budget = max_tokens or (self.token_budget - 1000)  # Leave room for prompt

        # Sort by score descending
        sorted_snippets = sorted(self.snippets, key=lambda s: s.score, reverse=True)

        lines = []
        used = 0

        for i, s in enumerate(sorted_snippets, 1):
            cost = s.token_cost()
            if used + cost > budget:
                continue

            # Format with citation
            source_tag = f"[{s.source}]"
            title = s.meta.get("title", "")[:50]
            header = f"[{i}] {s.paper_id} {source_tag}"
            if title:
                header += f" - {title}"

            lines.append(f"{header}")
            lines.append(f"    {s.text[:1000]}")  # Truncate long texts
            lines.append("")
            used += cost

        return "\n".join(lines)

    def summary(self) -> dict[str, Any]:
        """Get state summary for debugging/logging."""
        return {
            "query": self.query[:50] + "...",
            "papers_retrieved": len(self.paper_ids),
            "snippets_collected": len(self.snippets),
            "token_used": self.token_used,
            "token_budget": self.token_budget,
            "status": self.status,
            "gaps": self.gaps,
            "iterations": self.iterations,
            "papers_with_fulltext": len(self.papers_with_fulltext()),
        }

    def __repr__(self) -> str:
        return (
            f"EvidenceState(papers={len(self.paper_ids)}, "
            f"snippets={len(self.snippets)}, "
            f"tokens={self.token_used}/{self.token_budget}, "
            f"status={self.status})"
        )


# =============================================================================
# Factory function for REPL
# =============================================================================

def create_evidence_state(
    query: str,
    question_type: str = "yesno",
    token_budget: int = 8000,
) -> EvidenceState:
    """Create a new EvidenceState for adaptive evidence collection.

    Usage in REPL:
        state = create_evidence_state(question, question_type="yesno")

        # Initial search
        hits = search_papers(question, limit=100)
        top5 = rerank_papers(question, hits, top_k=5)
        state.add_papers(top5)
        state.add_abstracts(top5)

        # Check if enough
        judgment = judge_evidence(state)
        print(f"Status: {judgment['status']}")

        # Continue if needed...
    """
    return EvidenceState(
        query=query,
        question_type=question_type,
        token_budget=token_budget,
    )
