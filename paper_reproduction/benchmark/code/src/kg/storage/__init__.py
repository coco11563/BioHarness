"""Lightweight shadow package for kg.storage."""

from pathlib import Path

_LOCAL_DIR = Path(__file__).resolve().parent
_REAL_DIR = Path(__file__).resolve().parents[5] / "src" / "kg" / "storage"
__path__ = [str(_LOCAL_DIR), str(_REAL_DIR)]

from .graph_store import GraphStore
from .vector_store import KGVectorStore

__all__ = [
    "GraphStore",
    "KGVectorStore",
]
