"""Lightweight shadow package for kg.lightrag."""

from pathlib import Path

_LOCAL_DIR = Path(__file__).resolve().parent
_REAL_DIR = Path(__file__).resolve().parents[5] / "src" / "kg" / "lightrag"
__path__ = [str(_LOCAL_DIR), str(_REAL_DIR)]
