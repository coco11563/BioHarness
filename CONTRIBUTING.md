# Contributing to XCompass_Chi

## Development setup

```bash
git clone https://github.com/coco11563/XCompass_Chi.git
cd XCompass_Chi
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
pre-commit install
```

## Running checks locally

```bash
ruff check .
ruff format --check .
mypy
pytest -q
python verify.py        # offline smoke (live tests gated by --live)
```

## Submitting a change

The pipeline contract (cascade routing logic, threshold table, prompt
templates) is part of the reproducibility surface. If your change alters
any of:

- prompt templates (system + per-question-type instructions)
- cascade threshold or escalation conditions
- tool registry or pre-call dispatch
- constrained-generation or re-judgment shape

you must also:

1. Bump the framework-chi version in `pyproject.toml`.
2. Re-record the offline cached smoke set with `scripts/build_cached_smoke.py`
   and update `golden/cached_smoke.jsonl` + the `[smoke_set]` hash in
   `MANIFEST.toml`.
3. Document the user-visible impact in `docs/ablations.md`.
4. Add a `CHANGELOG.md` entry.

## Forbidden content

The pre-commit hook + CI both run `scripts/scan_forbidden_strings.py`,
which rejects any code or commit message containing non-release metadata,
local tooling names, or credentials. The full list lives in that script.

## Reporting issues

Open an issue at <https://github.com/coco11563/XCompass_Chi/issues>
with the failing command, full traceback, OS / Python version, and the
manifest id (`python verify.py --print-manifest-id`).
