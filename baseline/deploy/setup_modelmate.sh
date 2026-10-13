#!/usr/bin/env bash
# Run inside the selected ModelMate Ascend training image.
set -euo pipefail
PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${TV_PYTHON:-python}"
if [[ -n "${TV_CANN_ENV:-}" ]]; then
  if [[ ! -f "$TV_CANN_ENV" ]]; then
    echo "TV_CANN_ENV is not an existing activation script: $TV_CANN_ENV" >&2
    exit 2
  fi
  set +u
  source "$TV_CANN_ENV"
  set -u
fi
PERSISTENT_ROOT="${TV_PERSISTENT_ROOT:-/opt/dpcvol/datasets/task-vector}"
mkdir -p -- "$PERSISTENT_ROOT/environment"
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
CONSTRAINTS="$PERSISTENT_ROOT/environment/constraints-$STAMP.txt"
BEFORE="$PERSISTENT_ROOT/environment/before-$STAMP.json"
export TV_SETUP_CONSTRAINTS="$CONSTRAINTS" TV_SETUP_BEFORE="$BEFORE"
"$PYTHON_BIN" - <<'PY'
import importlib.metadata as md
import json, os
from pathlib import Path
import torch
import torch_npu
versions = {name: md.version(name) for name in ('torch', 'torch-npu')}
for name, value in versions.items():
    if not value.startswith('2.8.'):
        raise SystemExit(f'{name}={value}: use the authorized ModelMate 2.8 image')
if not torch.npu.is_available():
    raise SystemExit('Ascend NPU is unavailable; activate the image CANN environment first')
Path(os.environ['TV_SETUP_CONSTRAINTS']).write_text('\n'.join(f'{name}=={value}' for name, value in versions.items()) + '\n')
Path(os.environ['TV_SETUP_BEFORE']).write_text(json.dumps(versions, indent=2) + '\n')
print({'preserved_framework_versions': versions, 'visible_npus': torch.npu.device_count()})
PY
"$PYTHON_BIN" -m pip install --constraint "$CONSTRAINTS" --upgrade-strategy only-if-needed -r "$PROJECT_ROOT/deploy/requirements_modelmate.txt"
"$PYTHON_BIN" -m pip install --no-deps --editable "$PROJECT_ROOT"
"$PYTHON_BIN" - <<'PY'
import importlib.metadata as md
import json, os
from pathlib import Path
before = json.loads(Path(os.environ['TV_SETUP_BEFORE']).read_text())
after = {name: md.version(name) for name in before}
if before != after:
    raise SystemExit(f'Framework version changed unexpectedly: before={before}, after={after}')
import torch, torch_npu, transformers
assert transformers.__version__ == '4.51.3'
print({'framework_unchanged': after, 'transformers': transformers.__version__, 'npu_available': torch.npu.is_available()})
PY
"$PYTHON_BIN" -m pip freeze > "$PERSISTENT_ROOT/environment/pip-freeze-$STAMP.txt"
echo "Environment recorded in $PERSISTENT_ROOT/environment. Configure explicit model/data paths before preflight and smoke."
