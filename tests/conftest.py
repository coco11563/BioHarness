"""Pytest configuration: skip async tests when pytest-asyncio is missing."""

from __future__ import annotations

collect_ignore: list[str] = []

try:
    import pytest_asyncio  # noqa: F401
except ImportError:  # pragma: no cover
    collect_ignore.append("unit/test_cascade_client.py")
