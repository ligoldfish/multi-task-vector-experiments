set +u
set -eo pipefail

TV_CODE=/home/work/user-job-dir/app/task-vector-code-20261011T071308Z
TV_ROOT=/opt/dpcvol/datasets/naie/task-vector/experiments/exp-20261011-01
TV_PATCH=/home/work/user-job-dir/app/task-vector-portable-checkpoint-fix-20261012.zip
TV_PYTHON=/opt/dpcvol/datasets/naie/task-vector/runtime/py28-transformers4513-20261011/bin/python
TV_ACTIVATE=/opt/dpcvol/datasets/naie/task-vector/runtime/py28-transformers4513-20261011/activate-task-vector.sh
TV_OLD_ATTEMPT="$TV_ROOT/runs/engineering_smoke/prefill-fix-20261011T151729.406024Z-213752"
TV_LOG="$TV_ROOT/environment/portable-checkpoint-recovery-$(date -u +%Y%m%dT%H%M%SZ).log"

printf 'LOG=%s\n' "$TV_LOG"
printf '%s  %s\n' \
  '558f10d32044b476e8bcf5f43b42daf69ec5f990de67b9c095bfd2c7175d478a' \
  "$TV_PATCH" | sha256sum -c -

export TV_INSTALL_MODE=0
source "$TV_ACTIVATE"
set -euo pipefail
unset TV_MODEL_PATH TV_SEMANTIC_MODEL_PATH TV_PERSISTENT_ROOT
export PYTHONPATH="$TV_CODE/src${PYTHONPATH:+:$PYTHONPATH}"
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1
cd "$TV_CODE"

"$TV_PYTHON" -I -B "$TV_PATCH" \
  --project-root "$TV_CODE" \
  --evidence-root "$TV_ROOT/environment" 2>&1 | tee -a "$TV_LOG"

printf '\nSTAGE=portable_checkpoint_tests\n' | tee -a "$TV_LOG"
TV_TEST_NPU=1 "$TV_PYTHON" -B tests/test_portable_checkpoint.py \
  2>&1 | tee -a "$TV_LOG"

printf '\nSTAGE=recovery_contract_tests\n' | tee -a "$TV_LOG"
"$TV_PYTHON" -B tests/test_portable_recovery.py \
  2>&1 | tee -a "$TV_LOG"

printf '\nSTAGE=complete_smoke_with_verified_cached_calibration\n' | tee -a "$TV_LOG"
"$TV_PYTHON" -B -u ops/recover_portable_checkpoint.py \
  --config configs/modelmate.local.yaml \
  --previous-attempt "$TV_OLD_ATTEMPT" 2>&1 | tee -a "$TV_LOG"

printf '\nLOG=%s\n' "$TV_LOG"
