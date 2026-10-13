#!/usr/bin/env bash
set -eo pipefail
TV_ACTIVATE=/opt/dpcvol/datasets/naie/task-vector/runtime/py28-transformers4513-20261011/activate-task-vector.sh
TV_SCRIPT=/home/work/user-job-dir/app/download_task_vector_assets.py
TV_ENVIRONMENT=/opt/dpcvol/datasets/naie/task-vector/experiments/exp-20261011-01/environment
test -f "$TV_ACTIVATE"
test -f "$TV_SCRIPT"
printf '%s  %s\n' '2627aac9e53dbe75e3eaaacc54776ccb3ecea1fed97e216795be33d5accc495d' "$TV_SCRIPT" | sha256sum -c -
export TV_INSTALL_MODE=1
source "$TV_ACTIVATE"
set -euo pipefail
export HF_ENDPOINT=https://hf-mirror.com
export HF_HUB_DISABLE_XET=1
TV_PYTHON="$PYTHON_BIN"
TV_STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
mkdir -p "$TV_ENVIRONMENT"
TV_CONSTRAINTS="$TV_ENVIRONMENT/asset-install-constraints-$TV_STAMP.txt"
cat > "$TV_CONSTRAINTS" <<'TXT'
torch==2.8.0
torch-npu==2.8.0.post5
transformers==4.51.3
datasets==3.6.0
numpy==1.26.4
TXT
"$TV_PYTHON" -m pip install --disable-pip-version-check -c "$TV_CONSTRAINTS" 'py7zr==1.1.3' 2>&1 | tee "$TV_ENVIRONMENT/asset-install-$TV_STAMP.log"
TV_DOWNLOAD_LOG="$TV_ENVIRONMENT/asset-download-$TV_STAMP.log"
printf 'DOWNLOAD_LOG=%s\n' "$TV_DOWNLOAD_LOG"
"$TV_PYTHON" -B -u "$TV_SCRIPT" --workers 2 2>&1 | tee "$TV_DOWNLOAD_LOG"
