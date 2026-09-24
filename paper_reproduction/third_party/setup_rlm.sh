#!/usr/bin/env bash
# Recreate the copy of the Recursive Language Models library (rlm 0.1.0) that the
# research runs imported, at paper_reproduction/.ref_project/rlm (the path that
# src/rlm/pipeline.py and src/rlm/biomedical_repl.py put on sys.path), and install it in
# editable mode into the active Python environment.
#
# Upstream: https://github.com/alexzhang13/rlm (MIT licence), commit db41d15 (2026-01-05).
# Local patch (third_party/rlm_db41d15_local.patch):
#   rlm/core/rlm.py       message-history pruning (_prune_message_history, keep_recent=2,
#                         max_total_chars=12000), applied after every iteration   [before April runs]
#   rlm/utils/parsing.py  format_iteration default max_character_length 20000 -> 5000 [before April runs]
#   rlm/clients/openai.py chat_template_kwargs.enable_thinking=False, max_tokens=4096 [before April runs]
#                         timeout=600 s, max_retries=1 on the OpenAI clients          [added 2026-09-08]
# The result matches the research copy in every source file. The one difference is
# pyproject.toml: upstream says requires-python ">=3.11", the research copy ">=3.10"
# (no effect under the CPython 3.12 used for the runs).
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEST="$HERE/../.ref_project/rlm"
COMMIT=db41d150faf10425f18b415dd655543a227c552e
if [ ! -d "$DEST/.git" ]; then
  git clone https://github.com/alexzhang13/rlm.git "$DEST"
fi
git -C "$DEST" checkout --quiet "$COMMIT"
PATCH="$HERE/rlm_db41d15_local.patch"
if git -C "$DEST" apply --reverse --check "$PATCH" 2>/dev/null; then
  echo "local patch already applied"
else
  git -C "$DEST" apply "$PATCH"
fi
python -m pip install -e "$DEST"
