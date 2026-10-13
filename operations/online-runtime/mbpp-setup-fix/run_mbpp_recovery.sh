#!/usr/bin/env bash
set -eo pipefail

TV_CODE_DIR=/home/work/user-job-dir/app/task-vector-code-20261011T071308Z
TV_RUN_ROOT=/opt/dpcvol/datasets/naie/task-vector/experiments/exp-20261011-01
TV_PATCH=/home/work/user-job-dir/app/task-vector-mbpp-setup-fix-20261011.zip
TV_ACTIVATE=/opt/dpcvol/datasets/naie/task-vector/runtime/py28-transformers4513-20261011/activate-task-vector.sh

export TV_INSTALL_MODE=0
source "$TV_ACTIVATE"
set -euo pipefail
unset TV_MODEL_PATH TV_SEMANTIC_MODEL_PATH TV_PERSISTENT_ROOT
export PYTHONPATH="$TV_CODE_DIR/src${PYTHONPATH:+:$PYTHONPATH}"
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1

TV_PYTHON=/opt/dpcvol/datasets/naie/task-vector/runtime/py28-transformers4513-20261011/bin/python
TV_CONFIG="$TV_CODE_DIR/configs/modelmate.local.yaml"
TV_LOG="$TV_RUN_ROOT/environment/mbpp-recovery-$(date -u +%Y%m%dT%H%M%SZ).log"
cd "$TV_CODE_DIR"
printf 'LOG=%s\n' "$TV_LOG"

printf '%s  %s\n' \
  fbd7d980016028ef21b16fdeeffd23a4f43b4ef5d1611d0bb86fba9c458d3c6f \
  "$TV_PATCH" | sha256sum -c -

"$TV_PYTHON" -B "$TV_PATCH" \
  --project-root "$TV_CODE_DIR" \
  --evidence-root "$TV_RUN_ROOT/environment" \
  2>&1 | tee -a "$TV_LOG"

"$TV_PYTHON" -B -u - "$TV_CODE_DIR" <<'PY' 2>&1 | tee -a "$TV_LOG"
import json
import sys
from pathlib import Path
from task_vector.code_execution import execute_code

path = Path(sys.argv[1]) / "tests/fixtures/mbpp_927.json"
row = json.loads(path.read_text(encoding="utf-8"))
result = execute_code(row["code"], row["test_list"], row["test_setup_code"])
print(json.dumps(result, ensure_ascii=False, indent=2))
if not result["pass_all"] or result["passed"] != [True, True, True]:
    raise SystemExit("MBPP_927_REFERENCE=FAIL")
print("MBPP_927_REFERENCE=PASS")
PY

"$TV_PYTHON" -B -u "$TV_CODE_DIR/ops/recover_mbpp_prepare.py" \
  --config "$TV_CONFIG" \
  2>&1 | tee -a "$TV_LOG"

"$TV_PYTHON" -B -u -m task_vector.cli inventory \
  --config "$TV_CONFIG" \
  2>&1 | tee -a "$TV_LOG"

printf '\nLOG=%s\n' "$TV_LOG"
printf 'NEXT=real-model smoke; no training job submitted\n'
