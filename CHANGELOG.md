# Changelog

All notable changes to `bioHarness` are documented in this file.
Format: [Keep a Changelog](https://keepachangelog.com/en/1.1.0/);
versioning: [SemVer](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Changed
- The companion eval framework
  ([XCompass_Eval_Framework](https://github.com/coco11563/XCompass_Eval_Framework))
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
