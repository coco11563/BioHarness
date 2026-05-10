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
    import framework_chi
    import framework_chi.cascade
    import framework_chi.agent
    import framework_chi.clients
    import framework_chi.cli
    import framework_chi.config
    import framework_chi.tools
    assert framework_chi.__version__


def test_manifest_parses() -> None:
    data = tomllib.loads((ROOT / "MANIFEST.toml").read_text())
    assert data["framework_eval"]["plugin_entry_point"] == "v14-cascade-dual-rerank-grounded"
    assert data["headline"]["total_items"] == 19302
    assert data["headline"]["binary_accuracy"] == 0.766035


def test_cli_version() -> None:
    from framework_chi.cli.main import main

    buf = io.StringIO()
    with redirect_stdout(buf):
        with pytest.raises(SystemExit) as exc:
            main(["--version"])
    assert exc.value.code == 0
    assert buf.getvalue().strip()


def test_cli_ablations_emits_json() -> None:
    from framework_chi.cli.main import main

    buf = io.StringIO()
    with redirect_stdout(buf):
        rc = main(["ablations"])
    assert rc == 0
    payload = json.loads(buf.getvalue())
    assert payload["cascade_threshold"] == 0.7
    # `-grounded` in the headline method id corresponds to this default.
    assert payload["enable_grounded_gate"] is True
    assert payload["enable_dual_rerank"] is True


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
    assert eps["v14-cascade-dual-rerank-grounded"]
    module_name, _, attr = eps["v14-cascade-dual-rerank-grounded"].partition(":")
    import importlib
    mod = importlib.import_module(module_name)
    assert hasattr(mod, attr)
