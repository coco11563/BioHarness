"""LightRAG implementation.

LightRAG provides entity/relation-centric retrieval with 4 query modes:
- naive: Pure vector search (baseline)
- local: Entity-centric search
- global: Relation-centric search
- hybrid: Combined local + global (recommended)
"""

from .client import LightRAGClient
from .query import QueryMode, QueryExecutor

__all__ = [
    "LightRAGClient",
    "QueryMode",
    "QueryExecutor",
]
