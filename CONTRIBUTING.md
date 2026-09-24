# Contributing to BioHarness

## Development setup

```bash
git clone https://github.com/coco11563/BioHarness.git
cd BioHarness
python -m venv .venv && source .venv/bin/activate
pip install "git+https://github.com/coco11563/BioHarness_Eval_Framework.git"
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

1. Add a `CHANGELOG.md` entry describing the user-visible impact. The
   version in `pyproject.toml` is bumped at release time, not per change.
2. If an offline cached smoke set has been recorded (`golden/`, written by
   `scripts/build_cached_smoke.py`; none is recorded yet), re-record it and
   add its hash as a `[smoke_set]` section in `MANIFEST.toml`.

## `paper_reproduction/` is frozen

`paper_reproduction/` is the research code (September 2026 state) behind
the paper's numbers. Do not modify or reformat it; it is
excluded from ruff linting and formatting (`paper_reproduction/ruff.toml`).
Corrections to its documentation go in its README; changes to its code
belong in a separate, clearly labelled release note.

## Forbidden content

The pre-commit hook + CI both run `scripts/scan_forbidden_strings.py`,
which rejects any code or commit message containing non-release metadata,
local tooling names, or credentials. The full list lives in that script.

## Reporting issues

Open an issue at <https://github.com/coco11563/BioHarness/issues>
with the failing command, full traceback, OS / Python version, and the
manifest id (`python verify.py --print-manifest-id`).
