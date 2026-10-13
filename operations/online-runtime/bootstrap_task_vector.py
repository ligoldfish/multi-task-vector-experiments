#!/usr/bin/env python3
"""Create a private task-vector environment around the verified DPC NPU stack."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys


BASE_ACTIVATION = Path("/opt/dpcvol/datasets/naie/opd-decision-point-runtime/decision_point_opd_validation_final-v1/activate-opd.sh")
TASK_ROOT = Path("/opt/dpcvol/datasets/naie/task-vector")
VENDOR_ROOT = Path("/opt/dpcvol/models/transition_dm/_runtime_assets/lib/py/transition_dm_py311_cann85_295d1c5f90b2")


ACTIVATION = r'''#!/usr/bin/env bash
# Private package precedence; reuse the verified Ascend stack without installing into it.
_tv_activate_task_vector() {
  local tv_runtime=@@RUNTIME@@
  local tv_site=@@SITE@@
  local tv_base=@@BASE@@
  [[ -x "$tv_runtime/bin/python" && -d "$tv_site" && -r "$tv_base" ]] || {
    printf 'Task-vector runtime prerequisite missing\n' >&2
    return 2
  }
  set +u
  source "$tv_base" || return $?
  unset PYTHONHOME
  export PYTHON_BIN="$tv_runtime/bin/python"
  export VIRTUAL_ENV="$tv_runtime"
  export PATH="$tv_runtime/bin:$PATH"
  export PYTHONPATH="$tv_site${PYTHONPATH:+:$PYTHONPATH}"
  export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 PIP_USER=0
  export PIP_REQUIRE_VIRTUALENV=1
  unset PIP_TARGET PIP_PREFIX PIP_ROOT PIP_PYTHON PIP_BREAK_SYSTEM_PACKAGES
  export PIP_CONFIG_FILE=/dev/null
  export HF_HOME="$tv_runtime/cache/huggingface"
  export HF_HUB_CACHE="$HF_HOME/hub"
  export HF_DATASETS_CACHE="$HF_HOME/datasets"
  export TRANSFORMERS_CACHE="$HF_HOME"
  export XDG_CACHE_HOME="$tv_runtime/cache/xdg"
  export PIP_CACHE_DIR="$tv_runtime/cache/pip"
  export TORCH_HOME="$tv_runtime/cache/torch"
  export TORCH_EXTENSIONS_DIR="$tv_runtime/cache/torch-extensions"
  export TOKENIZERS_PARALLELISM=false
  if [[ "${TV_INSTALL_MODE:-0}" == 1 ]]; then
    export HF_HUB_OFFLINE=0 TRANSFORMERS_OFFLINE=0 HF_DATASETS_OFFLINE=0
  else
    export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
  fi
  hash -r 2>/dev/null || true
}
_tv_activate_task_vector
'''


VERIFY = r'''
import importlib.metadata as md
import json, os, sys
from pathlib import Path
import torch, torch_npu, transformers, datasets, sentence_transformers
import scipy, yaml, sacrebleu, rouge_score
from transformers import Qwen3Config, Qwen3ForCausalLM

runtime, vendor, receipt = map(Path, sys.argv[1:4])
assert Path(sys.prefix).resolve() == runtime.resolve(), (sys.prefix, runtime)
assert sys.prefix != sys.base_prefix, "Not an independent virtual environment"
assert Path(sys.executable).resolve().is_relative_to(runtime.resolve())
assert transformers.__version__ == "4.51.3", transformers.__version__
assert datasets.__version__ == "3.6.0", datasets.__version__
assert Path(transformers.__file__).resolve().is_relative_to(runtime.resolve())
for module in (torch, torch_npu):
    assert Path(module.__file__).resolve().is_relative_to(vendor.resolve()), module.__file__
assert torch.__version__.startswith("2.8."), torch.__version__
assert md.version("torch-npu").startswith("2.8."), md.version("torch-npu")
assert torch.npu.is_available() and torch.npu.device_count() >= 1

x = torch.ones((16, 16), device="npu:0", dtype=torch.bfloat16, requires_grad=True)
y = x @ x
loss = y.float().sum()
loss.backward()
torch.npu.synchronize()
output = y.detach().float().cpu()
gradient = x.grad.detach().float().cpu()
assert torch.equal(output, torch.full_like(output, 16.0))
assert torch.equal(gradient, torch.full_like(gradient, 32.0))

packages = {}
for name, module in (
    ("torch", torch), ("torch-npu", torch_npu),
    ("transformers", transformers), ("datasets", datasets),
    ("sentence-transformers", sentence_transformers), ("scipy", scipy),
    ("PyYAML", yaml), ("sacrebleu", sacrebleu), ("rouge-score", rouge_score),
):
    packages[name] = {"version": md.version(name), "module_path": module.__file__}
report = {
    "status": "PASS", "python": sys.version, "executable": sys.executable,
    "prefix": sys.prefix, "base_prefix": sys.base_prefix,
    "packages": packages, "visible_npus": torch.npu.device_count(),
    "qwen3_model_class": Qwen3ForCausalLM.__name__,
    "bf16_forward_mean": output.mean().item(),
    "bf16_loss": loss.detach().cpu().item(),
    "bf16_gradient_mean": gradient.mean().item(),
}
receipt.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
print("ENVIRONMENT_READY=PASS", flush=True)
'''


def run_bash(script: str, *arguments: str, stdin: str | None = None) -> None:
    subprocess.run(
        ["bash", "-c", script, "task-vector-bootstrap", *map(str, arguments)],
        input=stdin, text=True, check=True,
        env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--code-dir", required=True, type=Path)
    parser.add_argument("--runtime-dir", type=Path, default=TASK_ROOT / "runtime/py28-transformers4513-20261011")
    parser.add_argument("--experiment-root", type=Path, default=TASK_ROOT / "experiments/exp-20261011-01")
    parser.add_argument("--resume-install", action="store_true", help="Retry a failed installation in this script's existing private environment.")
    args = parser.parse_args()
    if sys.platform != "linux":
        parser.error("Run this script in the ModelMate Notebook Terminal")
    code = args.code_dir.resolve(strict=True)
    runtime = args.runtime_dir.resolve()
    experiment = args.experiment_root.resolve()
    for directory in (runtime, experiment):
        if directory == TASK_ROOT or not directory.is_relative_to(TASK_ROOT.resolve()):
            parser.error(f"Use a new directory below {TASK_ROOT}: {directory}")
    for relative in ("PACKAGE_MANIFEST.json", "pyproject.toml", "deploy/setup_modelmate.sh"):
        if not (code / relative).is_file():
            parser.error(f"Missing code file: {code / relative}")
    if not BASE_ACTIVATION.is_file():
        parser.error(f"Verified base activation is missing: {BASE_ACTIVATION}")
    manifest_sha = hashlib.sha256((code / "PACKAGE_MANIFEST.json").read_bytes()).hexdigest()
    marker = runtime / "task-vector-bootstrap.json"
    activation = runtime / "activate-task-vector.sh"
    print(f"CODE_DIR={code}\nRUNTIME_DIR={runtime}\nEXPERIMENT_ROOT={experiment}", flush=True)

    if runtime.exists():
        if not args.resume_install:
            parser.error("Runtime already exists. Preserve it; --resume-install is for retrying this private installation before any job uses it.")
        if not marker.is_file() or not (runtime / "pyvenv.cfg").is_file():
            parser.error("Existing runtime does not contain this bootstrap's venv and marker")
        record = json.loads(marker.read_text(encoding="utf-8"))
        if record["package_manifest_sha256"] != manifest_sha or record["runtime_dir"] != str(runtime):
            parser.error("Existing runtime belongs to a different code package or path")
        if not activation.is_file() or hashlib.sha256(activation.read_bytes()).hexdigest() != record["activation_sha256"]:
            parser.error("Existing activation script differs from the bootstrap receipt")
    else:
        if args.resume_install:
            parser.error("--resume-install requires an existing private runtime")
        runtime.parent.mkdir(parents=True, exist_ok=True)
        run_bash(
            'set -eo pipefail; source "$1"; unset PYTHONHOME; '
            '"$PYTHON_BIN" -B -m venv --system-site-packages --copies "$2"',
            BASE_ACTIVATION, runtime,
        )
        probe = subprocess.run(
            ["bash", "-c",
             'set -eo pipefail; source "$1" >&2; unset PYTHONHOME; exec "$2" -I -B -c "$3"',
             "task-vector-bootstrap", str(BASE_ACTIVATION), str(runtime / "bin/python"),
             'import json,sys,sysconfig; print(json.dumps(dict(prefix=sys.prefix,base_prefix=sys.base_prefix,executable=sys.executable,purelib=sysconfig.get_path("purelib"))))'],
            check=True, text=True, capture_output=True,
        )
        properties = json.loads(probe.stdout)
        if Path(properties["prefix"]).resolve() != runtime or properties["prefix"] == properties["base_prefix"]:
            raise RuntimeError(f"New interpreter is not an independent venv: {properties}")
        if not Path(properties["executable"]).resolve().is_relative_to(runtime):
            raise RuntimeError("New interpreter resolves outside its private runtime")
        site = Path(properties["purelib"]).resolve()
        if not site.is_relative_to(runtime):
            raise RuntimeError("New site-packages resolves outside its private runtime")
        content = ACTIVATION.replace("@@RUNTIME@@", shlex.quote(str(runtime))).replace("@@SITE@@", shlex.quote(str(site))).replace("@@BASE@@", shlex.quote(str(BASE_ACTIVATION)))
        with activation.open("x", encoding="utf-8", newline="\n") as handle:
            handle.write(content)
        record = {
            "runtime_dir": str(runtime), "code_dir_at_install": str(code),
            "package_manifest_sha256": manifest_sha,
            "activation_sha256": hashlib.sha256(activation.read_bytes()).hexdigest(),
            "base_activation": str(BASE_ACTIVATION), "interpreter": properties,
            "pyvenv_cfg": (runtime / "pyvenv.cfg").read_text(encoding="utf-8"),
        }
        with marker.open("x", encoding="utf-8") as handle:
            json.dump(record, handle, ensure_ascii=False, indent=2)
            handle.write("\n")

    evidence = experiment / "environment"
    evidence.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    log = evidence / f"bootstrap-install-{stamp}.log"
    print(f"ACTIVATION={activation}\nINSTALL_LOG={log}", flush=True)
    run_bash(
        'set -eo pipefail; export TV_INSTALL_MODE=1; source "$1"; '
        'set -euo pipefail; export TV_PYTHON="$PYTHON_BIN" TV_PERSISTENT_ROOT="$3" TV_CANN_ENV="$1"; '
        'cd -- "$2"; bash deploy/setup_modelmate.sh 2>&1 | tee "$4"',
        activation, code, experiment, log,
    )
    verification = evidence / f"bootstrap-verification-{stamp}.json"
    run_bash(
        'set -eo pipefail; export TV_INSTALL_MODE=0; source "$1"; '
        'set -euo pipefail; exec "$PYTHON_BIN" -B - "$2" "$3" "$4"',
        activation, runtime, VENDOR_ROOT, verification, stdin=VERIFY,
    )
    print(f"RUNTIME_PYTHON={runtime / 'bin/python'}\nCANN_ENV={activation}\nEXPERIMENT_ROOT={experiment}\nVERIFICATION={verification}", flush=True)


if __name__ == "__main__":
    try:
        main()
    except subprocess.CalledProcessError as error:
        if error.stderr:
            print(error.stderr, file=sys.stderr, flush=True)
        print(f"Installation or verification failed (exit={error.returncode}). Preserve the runtime and logs, and send the terminal output for diagnosis.", file=sys.stderr, flush=True)
        raise SystemExit(error.returncode or 1)
