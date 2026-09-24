"""Lightweight shadow package for retriever.

This avoids importing the full src/retriever/__init__.py, which eagerly pulls
in cached retriever and cache modules not needed by RT-KG experiments.
"""

from pathlib import Path

_LOCAL_DIR = Path(__file__).resolve().parent
_REAL_DIR = Path(__file__).resolve().parents[5] / "src" / "retriever"
__path__ = [str(_LOCAL_DIR), str(_REAL_DIR)]

from .models import RetrievedAbstract
from .vector_search import VectorSearch

__all__ = [
    "RetrievedAbstract",
    "VectorSearch",
]
