"""Lightweight shadow package for kg.extraction."""

from pathlib import Path

_LOCAL_DIR = Path(__file__).resolve().parent
_REAL_DIR = Path(__file__).resolve().parents[5] / "src" / "kg" / "extraction"
__path__ = [str(_LOCAL_DIR), str(_REAL_DIR)]

from .entity_extractor import EntityExtractor
from .relation_extractor import RelationExtractor

__all__ = [
    "EntityExtractor",
    "RelationExtractor",
]
