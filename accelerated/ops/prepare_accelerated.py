#!/usr/bin/env python3
"""Prepare a separate engineering restart; never modify the previous experiment."""
from __future__ import annotations
import argparse
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import sys
import yaml

PROJECT = Path(__file__).resolve().parents[1]
DPC_ROOT = Path("/opt/dpcvol/datasets")
sys.path.insert(0, str(PROJECT / "src"))
from task_vector.acceleration import DEFAULT_ACCELERATION
from task_vector.config import scientific_config, validate_config
from task_vector.io import atomic_json, file_sha256, object_sha256, read_json

UNCHANGED = (
    "data.py", "prompts.py", "training.py", "ltv.py", "loda.py", "gao.py",
    "subspaces.py", "representations.py", "losses.py", "baselines.py",
    "joint.py", "sequential.py", "modeling.py", "code_execution.py", "io.py",
)


def _data_digest(manifest):
    content = {key: value for key, value in manifest.items() if key != "manifest_content_sha256"}
    return hashlib.sha256(json.dumps(content, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def prepare(previous_project, destination, *, batch_policy="validated", check_only=False):
    previous_project, destination = Path(previous_project).resolve(), Path(destination).resolve()
    old_config_path = previous_project / "configs/modelmate.local.yaml"
    old_config = yaml.safe_load(old_config_path.read_text(encoding="utf-8"))
    old_root = Path(old_config["platform"]["persistent_root"]).resolve()
    if destination == old_root or old_root in destination.parents or destination in old_root.parents:
        raise ValueError("New experiment root must be independent of the previous root")
    if not destination.is_relative_to(DPC_ROOT):
        raise ValueError("New root must be under /opt/dpcvol/datasets")
    if destination.exists():
        raise FileExistsError(f"Use a fresh experiment root; existing directory preserved: {destination}")
    for path in (PROJECT / "configs/modelmate.local.yaml", PROJECT / "ops/settings.local.json"):
        if path.exists():
            raise FileExistsError(f"Already configured; existing file preserved: {path}")
    source_checks = {}
    for name in UNCHANGED:
        relative = Path("src/task_vector") / name
        before, after = previous_project / relative, PROJECT / relative
        if file_sha256(before) != file_sha256(after):
            raise ValueError(f"Scientific implementation preimage differs: {relative}")
        source_checks[str(relative)] = file_sha256(after)
    original_data = Path(old_config.get("runtime", {}).get("data_root", old_root / "data")).resolve()
    manifest = read_json(original_data / "manifest.json")
    if manifest.get("status") != "complete" or manifest.get("manifest_content_sha256") != _data_digest(manifest):
        raise ValueError("Previous prepared data manifest is incomplete or changed")
    required = set(old_config["data"]["tasks"] + old_config["data"]["extension_tasks"] + old_config["data"]["auxiliary_tasks"])
    if set(manifest["tasks"]) != required:
        raise ValueError("Prepared data must contain all seven documented tasks")
    verified = {}
    for task, record in manifest["tasks"].items():
        if not {"train", "tune", "discovery", "test", "test_formal"} <= set(record["files"]):
            raise ValueError(f"Prepared data is missing required splits: {task}")
        for split, entry in record["files"].items():
            path = original_data / task / f"{split}.jsonl"
            if Path(entry["path"]).resolve() != path.resolve() or file_sha256(path) != entry["sha256"]:
                raise ValueError(f"Prepared split path/content changed: {task}/{split}")
            with path.open(encoding="utf-8") as handle:
                count = sum(bool(line.strip()) for line in handle)
            if count != entry["count"]:
                raise ValueError(f"Prepared split count changed: {task}/{split}")
            verified[f"{task}/{split}"] = {"sha256": entry["sha256"], "count": count}
    settings = read_json(previous_project / "ops/settings.local.json")
    if set(settings) != {"runtime_python", "cann_env", "expected_devices"} or settings["expected_devices"] != 8:
        raise ValueError("Previous ModelMate runtime settings differ from the eight-card contract")
    for value in (settings["runtime_python"], settings["cann_env"]):
        if value and not Path(value).is_file():
            raise FileNotFoundError(value)
    config = deepcopy(old_config)
    config["platform"]["persistent_root"] = str(destination)
    config.setdefault("runtime", {}).update(data_root=str(destination / "data"), output_root=str(destination / "runs"))
    config.setdefault("training", {})["resume_from"] = None
    config["acceleration"] = deepcopy(DEFAULT_ACCELERATION)
    config["acceleration"]["generation_batch_policy"] = batch_policy
    validate_config(config)
    # Confirm only placement and explicit engineering settings have changed.
    old_science, new_science = scientific_config(old_config), scientific_config(config)
    for value in (old_science, new_science):
        value["platform"].pop("persistent_root", None)
        value.pop("runtime", None)
    if old_science != new_science:
        raise ValueError("Restart unexpectedly changes scientific settings")
    old_ledger = old_root / "runs/test_access_ledger.json"
    receipt = {"schema_version": 1, "created_utc": datetime.now(timezone.utc).isoformat(),
        "previous_project": str(previous_project), "previous_root": str(old_root), "new_root": str(destination),
        "previous_config_sha256": file_sha256(old_config_path),
        "scientific_settings_equal_except_placement": True, "unchanged_scientific_sources": source_checks,
        "prepared_splits": verified, "previous_test_access_ledger_sha256": file_sha256(old_ledger) if old_ledger.is_file() else None,
        "test_independence": "Engineering restart reuses previously exposed fixed test IDs; not new independent confirmation",
        "training_reuse": False, "prediction_reuse": False, "old_locks_or_signatures_rewritten": False,
        "data_policy": "byte-identical split copies; no resampling, downloads or reference/mutation changes",
        "acceleration": config["acceleration"], "check_only": check_only}
    if check_only:
        return receipt
    destination.mkdir(parents=True, exist_ok=False)
    shutil.copytree(original_data, destination / "data")
    new_manifest = deepcopy(manifest)
    for task, record in new_manifest["tasks"].items():
        for split, entry in record["files"].items():
            path = destination / "data" / task / f"{split}.jsonl"
            if file_sha256(path) != entry["sha256"]:
                raise ValueError(f"Copied data failed SHA256: {path}")
            entry["path"] = str(path)
    new_manifest["manifest_content_sha256"] = _data_digest(new_manifest)
    atomic_json(destination / "data/manifest.json", new_manifest)
    (PROJECT / "configs/modelmate.local.yaml").write_text(yaml.safe_dump(config, allow_unicode=True, sort_keys=False), encoding="utf-8")
    atomic_json(PROJECT / "ops/settings.local.json", settings)
    atomic_json(destination / "environment/acceleration-restart.json", receipt)
    return receipt


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--previous-project", required=True)
    parser.add_argument("--new-root", required=True)
    parser.add_argument("--batch-policy", choices=("strict", "validated", "audited"), default="validated")
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args(argv)
    result = prepare(args.previous_project, args.new_root, batch_policy=args.batch_policy, check_only=args.check_only)
    print(json.dumps({"PREPARE_ACCELERATED": "CHECK_PASS" if args.check_only else "PASS",
        "root": result["new_root"], "batch_policy": args.batch_policy,
        "verified_splits": len(result["prepared_splits"]), "old_experiment_unchanged": True}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
