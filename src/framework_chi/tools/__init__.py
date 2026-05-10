"""Tool registry stubs (gene resolver, UniProt, GO, Disco).

The full tool implementations are out of scope for this open-source
release; the registry below is the contract that the runtime checks at
construction time. Subclass ``V14CascadeClient`` and inject your own
tool dispatcher to wire concrete implementations.
"""

from __future__ import annotations

from typing import Protocol


class ToolCallable(Protocol):
    name: str
    description: str

    async def __call__(self, query: str) -> dict[str, object]: ...


REGISTRY: dict[str, str] = {
    "gene_resolver":   "Resolve a gene symbol/alias to its official symbol + Entrez id.",
    "uniprot_lookup":  "Fetch UniProt canonical entry for a gene/protein.",
    "go_annotations":  "Fetch GO annotations for a gene id.",
    "pubmed_search":   "Search PubMed for abstracts matching a query.",
    "disco_atlas":     "Look up cell-type / tissue expression in the scRNA atlas (optional).",
}
