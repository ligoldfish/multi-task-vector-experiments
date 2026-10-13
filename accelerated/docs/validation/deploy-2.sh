bash <<'BASH'
set -eo pipefail
TV_OLD=/home/work/user-job-dir/app/task-vector-code-20261011T071308Z
TV_NEW=/home/work/user-job-dir/app/task-vector-accelerated-20261013
TV_ROOT=/opt/dpcvol/datasets/naie/task-vector/experiments/exp-20261013-accelerated-01
TV_PY=/opt/dpcvol/datasets/naie/task-vector/runtime/py28-transformers4513-20261011/bin/python
source /opt/dpcvol/datasets/naie/task-vector/runtime/py28-transformers4513-20261011/activate-task-vector.sh
unset TV_MODEL_PATH TV_PERSISTENT_ROOT TV_SEMANTIC_MODEL_PATH
export PYTHONPATH="$TV_NEW/src:$TV_NEW${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONNOUSERSITE=1 TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8
cd "$TV_NEW"
"$TV_PY" ops/prepare_accelerated.py --previous-project "$TV_OLD" --new-root "$TV_ROOT" --batch-policy validated
mkdir -p "$TV_ROOT/environment"
TV_LOG="$TV_ROOT/environment/notebook-accelerated-$(date -u +%Y%m%dT%H%M%SZ).log"
exec > >(tee -a "$TV_LOG") 2>&1
printf 'LOG=%s\n' "$TV_LOG"
"$TV_PY" -m task_vector.cli preflight --config configs/modelmate.local.yaml
"$TV_PY" -m task_vector.cli assets --config configs/modelmate.local.yaml --hash-weights
"$TV_PY" -m task_vector.cli smoke --config configs/modelmate.local.yaml --device npu:0
"$TV_PY" - <<'PY'
from task_vector.config import load_config
from task_vector.runner import ExperimentRunner
from task_vector.workflow import smoke_evidence
runner = ExperimentRunner(load_config("configs/modelmate.local.yaml"))
try:
    evidence = smoke_evidence(runner)
    print(evidence)
    if not evidence["valid"]:
        raise SystemExit("真实模型检查未通过，停止")
finally:
    runner.close()
print("ACCELERATED_NOTEBOOK_READY=PASS")
print("NEXT=核对输出后，再停止旧作业并提交新八卡作业")
PY
BASH
