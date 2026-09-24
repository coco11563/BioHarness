"""PathRAG implementation.

PathRAG provides multi-hop path-based retrieval for KG-RAG.
Instead of single-hop entity/relation search, it finds paths
connecting relevant entities and uses path evidence for answers.
"""

from .client import PathRAGClient
from .pathfinder import PathFinder, PathScore

__all__ = [
    "PathRAGClient",
    "PathFinder",
    "PathScore",
]
