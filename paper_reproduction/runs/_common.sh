# Sourced by every run script: repository-relative paths, optional env.sh, python.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/.." && pwd)"
cd "$ROOT"
[ -f "$HERE/env.sh" ] && source "$HERE/env.sh"
export PYTHONPATH=src
PY="${PYTHON:-python}"
RUNNER=(scripts/run_unified_benchmark.py)
# Per-dataset result files land in output/unified_benchmark/{ablation,baselines}/<method>/.
# The runner resumes from an existing file there. The single-dataset Ours scripts
# (MedXpertQA, GeneTuring genomics, LitQA2) remove their file first; the April and
# baseline scripts resume, so remove their files by hand before a fresh run.
