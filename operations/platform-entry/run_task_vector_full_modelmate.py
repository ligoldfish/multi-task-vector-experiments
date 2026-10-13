"""Outer ModelMate entry; place beside, never inside, the prepared project.

Only platform data/checkpoint metadata is consumed. The original full launcher
still chooses the runtime and verifies all scientific provenance and resources.
--verify-only is a read-only Notebook probe and does not launch training.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import runpy
import sys


PROJECT_DIRECTORY = "task-vector-code-20261011T071308Z"
EXPERIMENT_ROOT = Path("/opt/dpcvol/datasets/naie/task-vector/experiments/exp-20261011-01")
EXPECTED_SETTINGS = {
    "runtime_python": "/opt/dpcvol/datasets/naie/task-vector/runtime/py28-transformers4513-20261011/bin/python",
    "cann_env": "/opt/dpcvol/datasets/naie/task-vector/runtime/py28-transformers4513-20261011/activate-task-vector.sh",
    "expected_devices": 8,
}


def parse_arguments(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    for field in ("data_url", "checkpoint_url"):
        parser.add_argument("--" + field, "--" + field.replace("_", "-"),
                            nargs="?", const="", default="", metavar="PLATFORM_METADATA",
                            help="ModelMate metadata only; original DPC inputs remain in force")
    parser.add_argument("--verify-only", action="store_true",
                        help="Verify local source binding; do not start a worker or write evidence")
    return parser.parse_args(argv)


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(2**20), b""):
            digest.update(block)
    return digest.hexdigest()


def object_sha256(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                     separators=(",", ":"), default=str).encode()).hexdigest()


def source_manifest(root):
    """Match task_vector.config.source_manifest without importing its runtime."""
    root = Path(root)
    entries = {}
    for folder in ("src", "configs", "deploy", "third_party", "ops"):
        for path in sorted((root / folder).rglob("*")):
            if path.is_file() and "__pycache__" not in path.parts and ".git" not in path.parts:
                entries[path.relative_to(root).as_posix()] = file_sha256(path)
    names = ["pyproject.toml", "docs/原始实验方案.md", "docs/补充协议与实现选择.md"]
    names += [p.name for p in sorted(root.glob("run_modelmate*.py"))]
    for name in names:
        path = root / name
        if path.is_file():
            entries[name] = file_sha256(path)
    return entries


def inspect_project(project=None, experiment_root=None):
    project = (Path(__file__).resolve().parent / PROJECT_DIRECTORY
               if project is None else Path(project)).resolve()
    root = EXPERIMENT_ROOT if experiment_root is None else Path(experiment_root)
    entry = project / "run_modelmate_full.py"
    for path in (entry, project / "ops/modelmate_job.py",
                 project / "configs/modelmate.local.yaml", project / "ops/settings.local.json"):
        if not path.is_file():
            raise FileNotFoundError(f"Required prepared project file missing: {path}")
    settings = json.loads((project / "ops/settings.local.json").read_text(encoding="utf-8-sig"))
    if settings != EXPECTED_SETTINGS or type(settings.get("expected_devices")) is not int:
        raise ValueError("Prepared runtime/settings differ from exp-20261011-01; inspect settings before launch")
    smoke_path = root / "runs/smoke.json"
    smoke = json.loads(smoke_path.read_text(encoding="utf-8-sig"))
    required_checks = ("loda_true_gao_microbatch_step", "loda_checkpoint_resume_exact",
                       "loda_resume_does_not_repeat_step")
    if smoke.get("passed") is not True or any(smoke.get("checks", {}).get(key) is not True
                                               for key in required_checks):
        raise ValueError(f"Required completed engineering smoke is absent or failed: {smoke_path}")
    manifest = source_manifest(project)
    source_digest = object_sha256(manifest)
    if smoke.get("source_manifest_sha256") != source_digest:
        raise ValueError("Prepared source differs from canonical smoke; no files or smoke records were modified. "
                         f"current={source_digest}, smoke={smoke.get('source_manifest_sha256')}")
    return {"project": str(project), "entry": str(entry), "stage": "full",
            "experiment_root": str(root), "expected_devices": 8,
            "source_manifest_sha256": source_digest, "source_files": len(manifest),
            "entry_sha256": file_sha256(entry), "canonical_smoke": str(smoke_path),
            "canonical_smoke_sha256": file_sha256(smoke_path),
            "source_binding": "PASS",
            "remaining_validation": "Original worker verifies full config/model/data binding and 8 NPUs"}


def write_receipt(record, args, raw_argv):
    """Save only this new wrapper's evidence, outside the scientific source tree."""
    source = Path(__file__).resolve().read_bytes()
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    destination = Path(record["experiment_root"]) / "environment/platform-entry" / f"{stamp}-{os.getpid()}"
    destination.mkdir(parents=True, exist_ok=False)
    (destination / Path(__file__).name).write_bytes(source)
    metadata = {}
    for field in ("data_url", "checkpoint_url"):
        value = getattr(args, field)
        metadata[field] = {"value_length": len(value),
                           "value_sha256": hashlib.sha256(value.encode()).hexdigest()}
    receipt = {**record, "recorded_utc": datetime.now(timezone.utc).isoformat(),
               "wrapper_sha256": hashlib.sha256(source).hexdigest(),
               "platform_argument_names": [part.split("=", 1)[0] for part in raw_argv if part.startswith("--")],
               "platform_metadata": metadata,
               "platform_values_used_as_experiment_inputs": False}
    path = destination / "entry.json"
    path.write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return path


def run_original_entry(project):
    entry = Path(project) / "run_modelmate_full.py"
    previous_argv, previous_path = sys.argv, sys.path[:]
    try:
        sys.argv = [str(entry)]
        sys.path.insert(0, str(project))
        runpy.run_path(str(entry), run_name="__main__")
    finally:
        sys.argv = previous_argv
        sys.path[:] = previous_path


def main(argv=None):
    raw_argv = list(sys.argv[1:] if argv is None else argv)
    args = parse_arguments(raw_argv)
    try:
        record = inspect_project()
        if args.verify_only:
            print(json.dumps(record, ensure_ascii=False, indent=2), flush=True)
            print("PLATFORM_ENTRY_VERIFY=PASS", flush=True)
            print("TRAINING_STARTED=NO", flush=True)
            return 0
        receipt = write_receipt(record, args, raw_argv)
    except (OSError, ValueError) as exc:
        print(f"PLATFORM_ENTRY_PRECHECK=FAIL: {exc}", file=sys.stderr, flush=True)
        return 2
    print("PLATFORM_ENTRY_METADATA=ACCEPTED", flush=True)
    print(f"PLATFORM_ENTRY_RECEIPT={receipt}", flush=True)
    print(f"SOURCE_BINDING=PASS source_sha256={record['source_manifest_sha256']}", flush=True)
    print("ORIGINAL_FULL_ENTRY=START", flush=True)
    run_original_entry(Path(record["project"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
