"""Recover the incomplete seven-task preparation after the MBPP setup-order fix.

The four completed core tasks remain byte-for-byte unchanged. Pending tasks are
prepared in a separate directory before publishing a complete combined manifest.
This entry performs no training, model-weight loading, or job submission.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import sys

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))

from task_vector.code_execution import EXECUTION_PROTOCOL_VERSION
from task_vector.config import load_config
from task_vector.data import Sample, _hash, audit_splits, prepare_dataset
from task_vector.io import atomic_json, file_sha256

CORE = ("math", "logic", "translation", "paraphrase")
PENDING = ("code_generation", "code_repair", "summary")
SPLITS = ("train", "tune", "discovery", "test", "test_formal", "test_confirmation")


def verify_tasks(root, manifest, tasks):
    """Check actual files and their disjoint groups, not only recorded counts."""
    hashes = {}
    for task in tasks:
        entry = manifest["tasks"][task]
        if set(entry["files"]) != set(SPLITS):
            raise ValueError(f"Unexpected prepared splits for {task}")
        samples_by_split = {}
        for split in SPLITS:
            record = entry["files"][split]
            path = root / task / f"{split}.jsonl"
            if Path(record["path"]).resolve() != path.resolve():
                raise ValueError(f"Prepared path does not match {path}")
            digest = file_sha256(path)
            if digest != record["sha256"]:
                raise ValueError(f"Prepared file changed: {path}")
            rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
            samples = [Sample.from_dict(row) for row in rows]
            if len(samples) != record["count"] or _hash([sample.id for sample in samples]) != record["ids_sha256"]:
                raise ValueError(f"Prepared count/IDs changed: {path}")
            samples_by_split[split] = samples
            hashes[str(path.relative_to(root))] = digest
        if audit_splits(samples_by_split) != entry["audit"]:
            raise ValueError(f"Prepared group audit changed: {task}")
    return hashes


def publish_pending(root, original, pending_root, pending):
    if pending.get("status") != "complete" or set(pending["tasks"]) != set(PENDING):
        raise ValueError("Pending preparation has not completed all three tasks")
    if _hash({k: v for k, v in pending.items() if k != "manifest_content_sha256"}) != pending.get("manifest_content_sha256"):
        raise ValueError("Pending manifest hash mismatch")
    verify_tasks(pending_root, pending, PENDING)
    for split in SPLITS:
        assignments = []
        for task in PENDING[:2]:
            rows = [json.loads(line) for line in (pending_root / task / f"{split}.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
            assignments.append([row["metadata"]["program_id"] for row in rows])
        if assignments[0] != assignments[1]:
            raise ValueError(f"MBPP generation/repair program assignments differ: {split}")
    original_hashes = verify_tasks(root, original, CORE)
    combined = deepcopy(original)
    for task in PENDING:
        shutil.copytree(pending_root / task, root / task, dirs_exist_ok=True)
        entry = deepcopy(pending["tasks"][task])
        for split, record in entry["files"].items():
            record["path"] = str((root / task / f"{split}.jsonl").resolve())
        combined["tasks"][task] = entry
    shutil.copytree(pending_root / "verification", root / "verification", dirs_exist_ok=True)
    combined["status"] = "complete"
    combined.pop("manifest_content_sha256", None)
    combined["manifest_content_sha256"] = _hash(combined)
    verify_tasks(root, combined, CORE + PENDING)
    if verify_tasks(root, combined, CORE) != original_hashes:
        raise ValueError("A completed core task changed during recovery")
    atomic_json(root / "manifest.json", combined)
    return combined


def recover(config, tokenizer, config_path):
    root = Path(config["platform"]["persistent_root"]) / "data"
    manifest_path = root / "manifest.json"
    original = json.loads(manifest_path.read_text(encoding="utf-8"))
    if original.get("status") != "preparing" or set(original.get("tasks", {})) != set(CORE):
        raise ValueError("This recovery requires the incomplete manifest with exactly four completed core tasks")
    if original.get("split_seed") != config["data"]["split_seed"]:
        raise ValueError("Split seed differs from the existing prepared data")
    if list(config["data"]["tasks"]) != list(CORE) or list(config["data"]["extension_tasks"]) != list(PENDING[:2]) or list(config["data"]["auxiliary_tasks"]) != ["summary"]:
        raise ValueError("The original seven-task scope is required")
    original_hashes = verify_tasks(root, original, CORE)
    capacity_path = root / "code_generation" / "capacity_preflight.json"
    old_mbpp_source = json.loads(capacity_path.read_text(encoding="utf-8"))["source"]
    for task in CORE:
        for source_file in original["tasks"][task]["source"].get("files", []):
            if file_sha256(source_file["path"]) != source_file["sha256"]:
                raise ValueError(f"Official source changed: {source_file['path']}")
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    evidence = root.parent / "environment" / f"mbpp-recovery-{stamp}-{os.getpid()}"
    evidence.mkdir(parents=True, exist_ok=False)
    print(f"RECOVERY_EVIDENCE={evidence}", flush=True)
    backup = evidence / "before"
    backup.mkdir()
    shutil.copy2(manifest_path, backup / "manifest.json")
    for name in (*PENDING, "verification"):
        if (root / name).exists():
            shutil.copytree(root / name, backup / name)
    pending_root = evidence / "pending-data"
    receipt = {"status": "preparing", "execution_protocol": EXECUTION_PROTOCOL_VERSION,
               "original_manifest_sha256": file_sha256(manifest_path),
               "config_sha256": file_sha256(config_path), "core_file_hashes": original_hashes,
               "pending_tasks": list(PENDING), "pending_root": str(pending_root),
               "original_mbpp_source": old_mbpp_source,
               "source_sha256": {name: file_sha256(PROJECT / name) for name in (
                   "src/task_vector/code_execution.py", "src/task_vector/data.py", "ops/recover_mbpp_prepare.py")}}
    atomic_json(evidence / "recovery.json", receipt)
    pending_config = deepcopy(config)
    pending_config["data"]["tasks"] = list(PENDING)
    try:
        print("STAGE=prepare_pending_three_tasks", flush=True)
        pending = prepare_dataset(pending_config, tokenizer, pending_root)
        for task in PENDING[:2]:
            if pending["tasks"][task]["source"] != old_mbpp_source:
                raise ValueError(f"Official MBPP source changed during recovery: {task}")
        if file_sha256(manifest_path) != receipt["original_manifest_sha256"]:
            raise ValueError("Active manifest changed while recovery was preparing data")
        print("STAGE=publish_complete_manifest", flush=True)
        combined = publish_pending(root, original, pending_root, pending)
    except BaseException as exc:
        receipt.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        atomic_json(evidence / "recovery.json", receipt)
        raise
    receipt.update(status="complete", manifest_content_sha256=combined["manifest_content_sha256"],
                   core_unchanged=True)
    atomic_json(evidence / "recovery.json", receipt)
    print("TASK\t" + "\t".join(SPLITS), flush=True)
    for task in CORE + PENDING:
        print(task + "\t" + "\t".join(str(combined["tasks"][task]["files"][split]["count"]) for split in SPLITS), flush=True)
    print("CORE_DATA_UNCHANGED=PASS", flush=True)
    print("DATA_PREPARATION=PASS", flush=True)
    print(f"MANIFEST={manifest_path}", flush=True)
    return combined


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(PROJECT / "configs" / "modelmate.local.yaml"))
    args = parser.parse_args()
    config = load_config(args.config)
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(config["model"]["path"], local_files_only=True)
    recover(config, tokenizer, args.config)


if __name__ == "__main__":
    main()
