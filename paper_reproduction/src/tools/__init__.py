"""External tools for biomedical RAG."""

from .gene_resolver import GeneResolver, GeneHit
from .go_resolver import GOResolver, GOTerm

__all__ = ["GeneResolver", "GeneHit", "GOResolver", "GOTerm"]
