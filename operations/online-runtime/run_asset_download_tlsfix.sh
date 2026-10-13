#!/usr/bin/env bash
# Runs in a child shell; never changes the interactive terminal's strict-mode settings.
set -eo pipefail
TV_ACTIVATE=/opt/dpcvol/datasets/naie/task-vector/runtime/py28-transformers4513-20261011/activate-task-vector.sh
TV_SCRIPT=/home/work/user-job-dir/app/download_task_vector_assets_tlsfix.py
TV_ENVIRONMENT=/opt/dpcvol/datasets/naie/task-vector/experiments/exp-20261011-01/environment
test -f "$TV_ACTIVATE"
test -f "$TV_SCRIPT"
printf '%s  %s\n' 'ada60125a07d3dc8f4b2b38d9b32648ffc77da24f512b74a81860fa65d0d0c43' "$TV_SCRIPT" | sha256sum -c -
export TV_INSTALL_MODE=1
source "$TV_ACTIVATE"
set -euo pipefail
export HF_ENDPOINT=https://hf-mirror.com
export HF_HUB_DISABLE_XET=1
TV_PYTHON="$PYTHON_BIN"
"$TV_PYTHON" -B - <<'PY'
from importlib import metadata
import py7zr
assert metadata.version('py7zr') == '1.1.3', 'Expected the already installed py7zr==1.1.3'
print('REUSING_PY7ZR=' + metadata.version('py7zr'))
PY
TV_STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
mkdir -p "$TV_ENVIRONMENT"
TV_DOWNLOAD_LOG="$TV_ENVIRONMENT/asset-download-tlsfix-$TV_STAMP.log"
printf 'DOWNLOAD_LOG=%s\n' "$TV_DOWNLOAD_LOG"
"$TV_PYTHON" -B -u "$TV_SCRIPT" --workers 2 "$@" 2>&1 | tee "$TV_DOWNLOAD_LOG"
