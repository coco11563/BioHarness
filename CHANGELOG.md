# Changelog

All notable changes to `XCompass_Chi` are documented in this file.
Format: [Keep a Changelog](https://keepachangelog.com/en/1.1.0/);
versioning: [SemVer](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Method-side reproducibility manifest (`MANIFEST.toml`) pinning the
  framework-eval contract version, headline numbers, live tolerance
  band, infra requirements, and offline cached smoke set.
- Project skeleton: `pyproject.toml`, `src/framework_chi/`, tests
  layout, CI workflow, pre-commit config (mirrors the eval-framework
  scaffold).

### Notes
- Plugs into framework-eval as the `pipeline` method via the
  `framework_eval.methods` entry point.
- Headline numbers (binary 0.766 / continuous 0.691 on 19,302 items)
  are reproduced byte-equal by the eval-framework's `python verify.py`
  using the shipped run snapshot; this repository is for **re-running**
  the method end-to-end against user-provided infrastructure.
