# Changelog

All notable changes to `bioHarness` are documented in this file.
Format: [Keep a Changelog](https://keepachangelog.com/en/1.1.0/);
versioning: [SemVer](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Changed
- **Rebranded to `bioHarness`** (formerly `XCompass_Chi`). **Breaking:**
  - Python package `framework_chi` → `bioharness` (import path:
    `from bioharness... import ...`).
  - Environment-variable prefix `FRAMEWORK_*` → `BIOHARNESS_*`. The old
    `FRAMEWORK_*` names are still read for one release with a deprecation
    warning; update your configs to `BIOHARNESS_*`.
  - CLI `framework-chi` → `bioharness`; repo URLs → `coco11563/bioHarness`.
  - Unchanged: the framework-eval method id (`pipeline`), predictions, and
    headline numbers.
- The companion eval framework
  ([bioharness_eval_framework](https://github.com/coco11563/bioharness_eval_framework))
  added an additive **`continuous-v2` scoring protocol** (SQuAD/BioASQ
  token-F1 for factoid items; non-factoid subtasks unchanged). Headline
  numbers under the new protocol: `binary_accuracy=0.766035`,
  `continuous_v2=0.688298` on 19,302 items.
  - `bioHarness` predictions are **unchanged**; only the framework-side
    factoid aggregator differs.
  - Verify with `python verify.py --protocol continuous-v2` from the eval
    framework. See its `docs/continuous-v2-protocol.md` for the full
    specification.

### Added
- **Atlas (D) component** (`bioharness.cascade.atlas`) for the SciHorizon
  `expression` subtask (gene → tissue list):
  - A recall + fixed 27-tissue-vocabulary + parseable-JSON prompt, replacing
    the free-text expression prompt that produced 0/210 parseable answers
    (~6% set-F1, a formatting artifact). This `-D` path ships and runs by
    default.
  - An opt-in `+D` *atlas-as-context* path: `bioharness.clients.atlas.AtlasClient`
    fetches a gene's HPA bulk tissue expression and injects it as a
    supplementary reference block. Enable with `BIOHARNESS_ENABLE_ATLAS=1`
    and a reachable `BIOHARNESS_ATLAS_URL` (off by default; fail-soft to `-D`).
  - Ablation (SciHorizon expression, non-empty GT, n=175, set-F1):
    `-D` 65.3 → `+D` 78.8 (**+13.5 pp**, McNemar p < 1e-12); `+D` also beats a
    direct HPA lookup (73.9). GT is NCBI Gene-derived, atlas is HPA →
    cross-database structured-knowledge retrieval.
- Method-side reproducibility manifest (`MANIFEST.toml`) pinning the
  framework-eval contract version, headline numbers, live tolerance
  band, infra requirements, and offline cached smoke set.
- Project skeleton: `pyproject.toml`, `src/bioharness/`, tests
  layout, CI workflow, pre-commit config (mirrors the eval-framework
  scaffold).

### Notes
- Plugs into framework-eval as the `pipeline` method via the
  `framework_eval.methods` entry point.
- Headline numbers (binary 0.766 / continuous 0.691 on 19,302 items)
  are reproduced byte-equal by the eval-framework's `python verify.py`
  using the shipped run snapshot; this repository is for **re-running**
  the method end-to-end against user-provided infrastructure.
