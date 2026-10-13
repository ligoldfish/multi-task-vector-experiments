#!/usr/bin/env bash
set -euo pipefail
TV_KIT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
TV_EXPORT_PYTHON="${TV_EXPORT_PYTHON:-/opt/dpcvol/datasets/naie/task-vector/runtime/py28-transformers4513-20261011/bin/python}"
exec "$TV_EXPORT_PYTHON" -B "$TV_KIT_DIR/export_portable.py" "$@"
