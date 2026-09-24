"""Lightweight shadow package for kg.

This benchmark package shadows src/kg/__init__.py to avoid eager imports of all
KG implementations, especially GraphRAG's optional heavy dependency chain.
Submodules are resolved from the real src/kg package directory via __path__.
"""

from pathlib import Path

_LOCAL_DIR = Path(__file__).resolve().parent
_REAL_KG_DIR = Path(__file__).resolve().parents[4] / "src" / "kg"
__path__ = [str(_LOCAL_DIR), str(_REAL_KG_DIR)]
