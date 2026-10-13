#!/usr/bin/env bash
# A single NPU experiment process. Eight-card training uses launch_pool.py.
set -euo pipefail
PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${TV_PYTHON:-python}"
if [[ $# -lt 1 ]]; then
  echo 'Usage: bash deploy/run_modelmate.sh CONFIG.yaml [PHASE=core]' >&2
  exit 2
fi
CONFIG="$1"
PHASE="${2:-core}"
if [[ ! -f "$CONFIG" ]]; then
  echo "Configuration does not exist: $CONFIG" >&2
  exit 2
fi
if [[ -n "${TV_CANN_ENV:-}" ]]; then
  set +u
  source "$TV_CANN_ENV"
  set -u
fi
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export PYTHONPATH="$PROJECT_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
cd -- "$PROJECT_ROOT"
exec "$PYTHON_BIN" -m task_vector.cli run --config "$CONFIG" --phase "$PHASE"
