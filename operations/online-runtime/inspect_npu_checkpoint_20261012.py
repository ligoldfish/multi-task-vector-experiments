"""Inspect local engineering checkpoint structure without unpickling it."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import pickletools
import zipfile


def sha256(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(2**20), b""):
            h.update(block)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", required=True)
    parser.add_argument("--experiment-root", required=True)
    parser.add_argument("--attempt", required=True)
    args = parser.parse_args()
    project, root, attempt = (Path(value).resolve() for value in
                            (args.project_root, args.experiment_root, args.attempt))
    if root not in attempt.parents:
        raise ValueError("Engineering attempt must be under the supplied experiment root")
    report = {"kind": "static_checkpoint_inspection", "unpickled": False,
              "scientific_training": False, "formal_evaluation": False,
              "attempt": str(attempt), "files": {}}
    for relative in ("smoke.json", "loda_training_smoke/uninterrupted/training-state.json",
                     "loda_training_smoke/uninterrupted/epochs.json",
                     "loda_training_smoke/uninterrupted/steps.jsonl"):
        path = attempt / relative
        content = path.read_text(encoding="utf-8")
        value = ([json.loads(line) for line in content.splitlines() if line.strip()]
                 if relative.endswith(".jsonl") else json.loads(content))
        report["files"][relative] = {"sha256": sha256(path), "content": value}
        if relative == "smoke.json":
            print("FAILED_SMOKE=" + json.dumps(value, ensure_ascii=False), flush=True)
        elif relative.endswith("training-state.json"):
            print("TRAINING_STATE=" + json.dumps(value, ensure_ascii=False), flush=True)
        else:
            print(relative.rsplit("/", 1)[-1].upper() + "=" + json.dumps(value, ensure_ascii=False), flush=True)
    checkpoint = attempt / "loda_training_smoke/uninterrupted/last.pt"
    with zipfile.ZipFile(checkpoint) as archive:
        candidates = [info for info in archive.infolist() if info.filename.endswith("/data.pkl")]
        if len(candidates) != 1:
            raise ValueError("Expected one tensor-checkpoint data.pkl record")
        info = candidates[0]
        if info.file_size > 16 * 2**20:
            raise ValueError("Checkpoint metadata exceeds inspection limit")
        operations = list(pickletools.genops(archive.read(info)))
    globals_found = sorted({arg for op, arg, pos in operations if op.name == "GLOBAL"})
    contexts = []
    for index, (op, arg, pos) in enumerate(operations):
        if op.name in {"GLOBAL", "STACK_GLOBAL"}:
            contexts.append([{"offset": item_pos, "opcode": item_op.name,
                              "argument": str(item_arg)[:240]}
                             for item_op, item_arg, item_pos in operations[max(0, index - 2):index + 12]])
    record = {"path": str(checkpoint), "bytes": checkpoint.stat().st_size,
              "sha256": sha256(checkpoint), "pickle_globals": globals_found,
              "stack_global_count": sum(op.name == "STACK_GLOBAL" for op, arg, pos in operations),
              "global_opcode_contexts": contexts}
    report["checkpoint"] = record
    print("CHECKPOINT=" + json.dumps({k: v for k, v in record.items() if k != "global_opcode_contexts"}), flush=True)
    print("GLOBAL_CONTEXTS=" + json.dumps(contexts, ensure_ascii=False), flush=True)
    package = project / "src/task_vector"
    # This is the exact training-source manifest used by the original trainer.
    source_hashes = {path.name: sha256(path) for path in sorted(package.glob("*.py"))}
    encoded = json.dumps(source_hashes, ensure_ascii=False, sort_keys=True,
                         separators=(",", ":"), default=str).encode()
    source_digest = hashlib.sha256(encoded).hexdigest()
    report["training_source_files"] = source_hashes
    report["training_sources_sha256"] = source_digest
    training = report["files"]["loda_training_smoke/uninterrupted/training-state.json"]["content"]
    expected = training["provenance"]["training_sources_sha256"]
    if source_digest != expected:
        raise ValueError("Current source differs from the completed engineering update")
    print(f"TRAINING_SOURCES_MATCH=PASS sha256={source_digest}", flush=True)
    print("SAVED_INPUTS=" + json.dumps({name: (attempt / name).is_file() for name in (
        "loda_training_smoke/engineering_train_half_a_statistics.pt",
        "loda_training_smoke/engineering_train_half_b_statistics.pt",
        "loda_training_smoke/uninterrupted/initial.pt",
        "loda_training_smoke/uninterrupted/raw.pt",
        "loda_training_smoke/uninterrupted/post.pt",
        "loda_training_smoke/uninterrupted/best.pt")}), flush=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    evidence = root / "environment" / f"checkpoint-inspection-{stamp}-{os.getpid()}"
    evidence.mkdir(parents=True, exist_ok=False)
    output = evidence / "inspection.json"
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"INSPECTION_JSON={output}", flush=True)
    print("STATIC_CHECKPOINT_INSPECTION=COMPLETE", flush=True)


if __name__ == "__main__":
    main()
