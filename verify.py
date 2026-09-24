"""Package gate for the BioHarness re-implementation (``pipeline`` method).

This script checks that the package is wired correctly. It does NOT verify
the paper's numbers: this package is a re-implementation that does not
reproduce the revised Table 1 columns (see README.md). The paper's numbers
(Overall9 67.2 over 21,752 scored items, LitQA2 61.8 supplementary) are
reproduced offline from the stored per-item scores by ``python verify.py``
in the BioHarness_Eval_Framework repository; the research code that produced
them is in ``paper_reproduction/``.

Two operating modes:

  ``python verify.py``           offline (default): import the
                                  PipelineCascadeClient through the plugin
                                  registry and validate the shipped cached
                                  smoke set against MANIFEST.toml. No network
                                  or GPU required.

  ``python verify.py --live``    live: runs a single end-to-end smoke item
                                  against the user-provided infrastructure
                                  (see docs/infra.md) and reports whether the
                                  pipeline glue is wired correctly. Full
                                  benchmark runs go through
                                  ``framework-eval run --method pipeline ...``;
                                  expect them to differ from Table 1.

Exit codes:
    0  verification passed
    1  hash drift, smoke failure, or live smoke item failure
    2  CLI / configuration error
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import sys
from pathlib import Path

if sys.version_info >= (3, 11):
    import tomllib
else:  # pragma: no cover
    import tomli as tomllib  # type: ignore[no-redef]

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE / "src"))

MANIFEST_PATH = _HERE / "MANIFEST.toml"


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# ----------------------------------------------------------------------
# Plugin smoke
# ----------------------------------------------------------------------


def _check_plugin_resolves(manifest: dict) -> str | None:
    """The pyproject entry point must import without touching network."""
    entry_name = manifest["framework_eval"]["plugin_entry_point"]
    try:
        from framework_eval.plugins import discover_entry_points, load_method
    except ImportError as exc:
        return f"framework-eval not installed: {exc}"

    eps = {e.name: e for e in discover_entry_points()}
    if entry_name not in eps:
        return (
            f"entry point {entry_name!r} not registered. Install this package "
            f"with `pip install -e .` and re-run."
        )
    try:
        cls = load_method(entry_name)
    except Exception as exc:
        return f"entry point {entry_name!r} failed to import: {exc}"
    if cls.__name__ != "PipelineCascadeClient":
        return (
            f"entry point {entry_name!r} resolved to {cls.__name__}, "
            "expected PipelineCascadeClient"
        )
    return None


# ----------------------------------------------------------------------
# Cached smoke
# ----------------------------------------------------------------------


def _check_cached_smoke(manifest: dict) -> tuple[bool, str]:
    smoke = manifest.get("smoke_set")
    if not smoke:
        return True, "no [smoke_set] in manifest; skipping"
    rel = smoke.get("relative_path")
    expected = smoke.get("sha256", "")
    if expected.startswith("TBD-"):
        return True, "smoke set hash is TBD; skipping until first recording"
    path = _HERE / rel
    if not path.exists():
        return False, f"smoke set {rel!r} listed in manifest but missing"
    actual = _sha256(path)
    if actual != expected:
        return False, (
            f"smoke set drift: {rel!r}\n"
            f"  expected sha256 {expected}\n"
            f"  actual   sha256 {actual}"
        )
    n = sum(1 for line in path.read_text().splitlines() if line.strip())
    return True, f"{n} cached items hash-equal"


# ----------------------------------------------------------------------
# Live re-run (deferred entry; full live wrapper lives in scripts)
# ----------------------------------------------------------------------


async def _live_smoke(manifest: dict) -> tuple[bool, str]:
    """Construct the cascade client and run a single sanity item."""
    try:
        from bioharness.cascade.client import PipelineCascadeClient
    except ImportError as exc:
        return False, f"bioharness not importable: {exc}"

    client = PipelineCascadeClient()
    try:
        from framework_eval.eval.types import Item

        item = Item(
            id="live-smoke", dataset="bioasq",
            question="Is metformin a first-line treatment for type 2 diabetes?",
            question_type="yesno", answer="yes",
        )
        pred = await client.generate(item)
    except Exception as exc:
        await client.aclose()
        return False, f"live smoke generate raised: {exc}"
    await client.aclose()

    if not pred.answer:
        return False, "live smoke returned an empty answer"
    return True, f"answer={pred.answer!r} stage={pred.extras.get('stage')}"


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--live", action="store_true",
        help="run a single live item against the user-provided LLM stack",
    )
    parser.add_argument(
        "--print-manifest-id", action="store_true",
        help="print the manifest_id and exit",
    )
    args = parser.parse_args(argv)

    manifest = tomllib.loads(MANIFEST_PATH.read_text())

    if args.print_manifest_id:
        print(manifest.get("manifest_id", "unknown"))
        return 0

    print(f"BioHarness verify.py — manifest {manifest['manifest_id']}")
    print(f"  headline method ........ {manifest['headline']['method_id']}")
    print(f"  framework-eval pin ..... {manifest['framework_eval']['required_version_range']}")

    failures: list[str] = []

    print("  plugin entry point ..... ", end="")
    err = _check_plugin_resolves(manifest)
    if err:
        print("FAIL")
        failures.append(f"  plugin: {err}")
    else:
        print("ok")

    print("  cached smoke set ....... ", end="")
    ok, detail = _check_cached_smoke(manifest)
    print(f"{'ok' if ok else 'FAIL'} ({detail})")
    if not ok:
        failures.append(f"  smoke: {detail}")

    if args.live and not failures:
        print("  live smoke item ........ ", end="")
        ok, detail = asyncio.run(_live_smoke(manifest))
        print(f"{'ok' if ok else 'FAIL'} ({detail})")
        if not ok:
            failures.append(f"  live: {detail}")

    if failures:
        print()
        print("VERIFICATION FAILED:")
        for f in failures:
            print(f)
        return 1

    print("PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
