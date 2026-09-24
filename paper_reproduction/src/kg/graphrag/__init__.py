"""MS GraphRAG implementation.

GraphRAG provides hierarchical community-based retrieval:
- Builds hierarchical communities using Louvain/Leiden algorithm
- Generates community summaries (reports) using LLM
- Supports local search (entity/relation focus) and global search (community aggregation)
"""

from .client import GraphRAGClient
from .community import CommunityDetector, CommunityReportGenerator
from .search import LocalSearch, GlobalSearch

__all__ = [
    "GraphRAGClient",
    "CommunityDetector",
    "CommunityReportGenerator",
    "LocalSearch",
    "GlobalSearch",
]
