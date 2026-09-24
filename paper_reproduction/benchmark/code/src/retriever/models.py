"""Minimal retriever models for benchmark-safe RT-KG runs."""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class RetrievedAbstract:
    """A retrieved PubMed abstract with metadata."""

    pmid: str
    title: str
    abstract: str
    score: float
    rank: int
    have_fulltext: bool = False
    paper_uuid: Optional[str] = None
    pmc: Optional[str] = None
    matched_mesh: list[str] = field(default_factory=list)
