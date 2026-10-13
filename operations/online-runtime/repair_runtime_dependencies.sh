#!/usr/bin/env bash
# Supplemental installation only; preserve the verified experiment source archive.
set -eo pipefail
TV_RUNTIME=/opt/dpcvol/datasets/naie/task-vector/runtime/py28-transformers4513-20261011
TV_EXPERIMENT=/opt/dpcvol/datasets/naie/task-vector/experiments/exp-20261011-01
TV_UPLOAD=/home/work/user-job-dir/app
TV_CODE="$TV_UPLOAD/task-vector-code-20261011T071308Z"
export TV_INSTALL_MODE=1
source "$TV_RUNTIME/activate-task-vector.sh"
set -euo pipefail
test "$PYTHON_BIN" = "$TV_RUNTIME/bin/python"
test -f "$TV_UPLOAD/bootstrap_task_vector.py"
TV_FIX_DIR="$(mktemp -d "$TV_EXPERIMENT/environment/dependency-fix-XXXXXXXX")"
printf 'DEPENDENCY_FIX_DIR=%s\n' "$TV_FIX_DIR"

"$PYTHON_BIN" -B - <<'PY' > "$TV_FIX_DIR/constraints.txt"
from importlib.metadata import version
from pathlib import Path
import sys
expected = Path('/opt/dpcvol/datasets/naie/task-vector/runtime/py28-transformers4513-20261011')
assert Path(sys.prefix).resolve() == expected.resolve(), sys.prefix
assert Path(sys.executable).resolve().is_relative_to(expected.resolve())
for name in ('torch', 'torch-npu', 'transformers', 'datasets', 'scipy', 'pandas', 'scikit-learn'):
    print(f'{name}=={version(name)}')
PY

cat > "$TV_FIX_DIR/requirements.txt" <<'REQ'
numpy==1.26.4
einops==0.8.1
cloudpickle>=2,<4
decorator>=5.1,<6
ml-dtypes>=0.5,<1
tornado>=6.4,<7
REQ

"$PYTHON_BIN" -m pip install \
  --constraint "$TV_FIX_DIR/constraints.txt" \
  --upgrade-strategy only-if-needed \
  -r "$TV_FIX_DIR/requirements.txt" 2>&1 | tee "$TV_FIX_DIR/install.log"

"$PYTHON_BIN" -B - "$TV_FIX_DIR/extra-dependencies.json" <<'PY'
import importlib.metadata as md
import json, sys
from pathlib import Path
import numpy, einops, cloudpickle, decorator, ml_dtypes, tornado
assert numpy.__version__ == '1.26.4', numpy.__version__
assert einops.__version__ == '0.8.1', einops.__version__
for module in (numpy, einops):
    assert Path(module.__file__).resolve().is_relative_to(Path(sys.prefix).resolve()), module.__file__
report = {name: {'version': md.version(name), 'module_path': module.__file__}
          for name, module in (('numpy', numpy), ('einops', einops),
                               ('cloudpickle', cloudpickle), ('decorator', decorator),
                               ('ml-dtypes', ml_dtypes), ('tornado', tornado))}
Path(sys.argv[1]).write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
print(json.dumps(report, indent=2), flush=True)
PY

"$PYTHON_BIN" -B - "$TV_FIX_DIR/pip-check.log" <<'PY'
import importlib, subprocess, sys
from pathlib import Path
result = subprocess.run([sys.executable, '-m', 'pip', 'check'],
                        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
Path(sys.argv[1]).write_text(result.stdout, encoding='utf-8')
print(result.stdout, end='', flush=True)
if result.returncode:
    names = ('getopt', 'inspect', 'multiprocessing')
    allowed = {f'op-compile-tool 0.1.0 requires {name}, which is not installed.'
               for name in names}
    issues = {line.strip() for line in result.stdout.splitlines() if line.strip()}
    if result.returncode != 1 or not issues or not issues <= allowed:
        raise SystemExit(result.returncode)
    for name in names:
        assert name in sys.stdlib_module_names, name
        module = importlib.import_module(name)
        print(f'STDLIB_IMPORT={name}: {module.__file__}', flush=True)
    print('Only verified standard-library dependency metadata notices remain.', flush=True)
print('DEPENDENCY_CHECK=PASS', flush=True)
PY

"$PYTHON_BIN" -B "$TV_UPLOAD/bootstrap_task_vector.py" \
  --code-dir "$TV_CODE" --resume-install 2>&1 | tee "$TV_FIX_DIR/bootstrap.log"

printf 'DEPENDENCY_FIX=PASS\nDEPENDENCY_FIX_DIR=%s\n' "$TV_FIX_DIR"
