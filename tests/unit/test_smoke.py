"""Smoke tests for package import, manifest, CLI, scanner."""

from __future__ import annotations

import io
import json
import subprocess
import sys
from contextlib import redirect_stdout
from pathlib import Path

import pytest

if sys.version_info >= (3, 11):
    import tomllib
else:  # pragma: no cover
    import tomli as tomllib  # type: ignore[no-redef]

ROOT = Path(__file__).resolve().parents[2]


def test_package_imports() -> None:
    import bioharness
    import bioharness.agent
    import bioharness.cascade
    import bioharness.cli
    import bioharness.clients
    import bioharness.config
    import bioharness.tools
    assert bioharness.__version__


def test_manifest_parses() -> None:
    data = tomllib.loads((ROOT / "MANIFEST.toml").read_text())
    assert data["framework_eval"]["plugin_entry_point"] == "pipeline"
    headline = data["headline"]
    # Revised Table 1: Overall9 67.2 over 21,752 scored items; LitQA2 separate.
    assert headline["total_items"] == 21752
    assert headline["continuous_mean"] == 0.672131
    assert headline["binary_accuracy"] == 0.740943
    litqa2 = headline["supplementary"]["litqa2"]
    assert litqa2["total_items"] == 199
    assert litqa2["continuous_mean"] == 0.618090


def test_cli_version() -> None:
    from bioharness.cli.main import main

    buf = io.StringIO()
    with redirect_stdout(buf):
        with pytest.raises(SystemExit) as exc:
            main(["--version"])
    assert exc.value.code == 0
    assert buf.getvalue().strip()


def test_cli_config_emits_json() -> None:
    from bioharness.cli.main import main

    buf = io.StringIO()
    with redirect_stdout(buf):
        rc = main(["config"])
    assert rc == 0
    payload = json.loads(buf.getvalue())
    assert payload["cascade"]["cascade_threshold"] == 0.7
    assert payload["cascade"]["retrieval_top_k"] == 20
    assert payload["cascade"]["dense_collection"] == "paper-full"
    assert "force_agent" in payload["services"]


def test_forbidden_string_scanner_passes_on_repo() -> None:
    rc = subprocess.run(
        [sys.executable, "scripts/scan_forbidden_strings.py"],
        cwd=ROOT, check=False, capture_output=True, text=True,
    )
    assert rc.returncode == 0, (
        f"scanner found hits:\nstdout:\n{rc.stdout}\nstderr:\n{rc.stderr}"
    )


def test_pyproject_entry_point_resolves() -> None:
    """The pyproject framework_eval.methods entry point must import."""
    data = tomllib.loads((ROOT / "pyproject.toml").read_text())
    eps = data["project"]["entry-points"]["framework_eval.methods"]
    assert eps["pipeline"]
    module_name, _, attr = eps["pipeline"].partition(":")
    import importlib
    mod = importlib.import_module(module_name)
    assert hasattr(mod, attr)
