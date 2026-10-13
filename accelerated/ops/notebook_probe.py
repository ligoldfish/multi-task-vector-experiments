#!/usr/bin/env python3
"""Read-only, bounded ModelMate inventory; never assigns asset paths or trains."""
from __future__ import annotations

import argparse
from collections import deque
from datetime import datetime, timezone
import importlib.metadata as metadata
import json
import os
from pathlib import Path
import platform
import re
import subprocess
import sys
import time


DPC_ROOT = Path("/opt/dpcvol/datasets")
MODEL_FIELDS = ("model_type", "architectures", "num_hidden_layers", "hidden_size",
                "num_attention_heads", "num_key_value_heads", "head_dim", "torch_dtype")
RAW_SUFFIXES = {".json", ".jsonl", ".ndjson", ".tsv", ".csv", ".parquet", ".arrow", ".xml", ".en", ".de"}
SPLIT_NAME = re.compile(r"(?:^|[-_.])(train|test|validation|valid|dev|eval|tst)(?:[-_.]|$)", re.I)


def command(argv, timeout=15):
    """Run only caller-supplied read-only diagnostics, without a shell."""
    try:
        result = subprocess.run(argv, capture_output=True, text=True, timeout=timeout,
                                encoding="utf-8", errors="replace", check=False)
        return {"argv": argv, "returncode": result.returncode,
                "stdout": result.stdout[:20000], "stderr": result.stderr[:4000],
                "output_truncated": len(result.stdout) > 20000 or len(result.stderr) > 4000}
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {"argv": argv, "error": str(exc)}


def read_metadata(path, limit=512 * 1024):
    """Read JSON metadata only; sample files and model weights are never opened."""
    if path.stat().st_size > limit:
        return {"metadata_not_read": "size_limit", "bytes": path.stat().st_size}
    with path.open(encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        return {"metadata_type": type(value).__name__}
    if path.name == "config.json":
        return {key: value[key] for key in MODEL_FIELDS if key in value}
    if path.name == "dataset_dict.json":
        return {"splits": value.get("splits")}
    if path.name == "state.json":
        return {"fingerprint": value.get("_fingerprint"),
                "data_filenames": [item.get("filename") for item in value.get("_data_files", []) if isinstance(item, dict)]}
    return {key: value[key] for key in ("dataset_name", "builder_name", "config_name", "version", "splits") if key in value}


def inventory_assets(root, *, max_depth=5, max_directories=1500, max_entries=20000,
                     max_candidates=200, max_seconds=20):
    result = {"root": str(root), "mapping_assigned": False, "sample_contents_read": False,
              "limits": {"depth": max_depth, "directories": max_directories, "entries": max_entries,
                         "candidates": max_candidates, "seconds": max_seconds},
              "candidates": [], "errors": [], "error_count": 0, "symlinks_skipped": 0,
              "directories_scanned": 0, "entries_scanned": 0, "truncated": False}
    if not root.is_dir():
        result["error_count"] = 1
        result["errors"] = ["Asset root is missing or is not a directory"]
        return result
    root = root.resolve()
    if root != DPC_ROOT and DPC_ROOT not in root.parents:
        result["error_count"] = 1
        result["errors"] = ["Resolved asset root is outside /opt/dpcvol/datasets"]
        return result
    queue = deque([(root, 0)])
    started = time.monotonic()

    def error(path, exc):
        result["error_count"] += 1
        if len(result["errors"]) < 30:
            result["errors"].append({"path": str(path), "error": str(exc)})

    while queue:
        if (result["directories_scanned"] >= max_directories or
                result["entries_scanned"] >= max_entries or
                len(result["candidates"]) >= max_candidates or
                time.monotonic() - started >= max_seconds):
            result["truncated"] = True
            break
        folder, depth = queue.popleft()
        result["directories_scanned"] += 1
        files = {}
        try:
            with os.scandir(folder) as entries:
                for entry in entries:
                    result["entries_scanned"] += 1
                    if result["entries_scanned"] > max_entries or time.monotonic() - started >= max_seconds:
                        result["truncated"] = True
                        break
                    try:
                        if entry.is_symlink():
                            result["symlinks_skipped"] += 1
                        elif entry.is_dir(follow_symlinks=False) and depth < max_depth:
                            queue.append((Path(entry.path), depth + 1))
                        elif entry.is_file(follow_symlinks=False):
                            files[entry.name] = Path(entry.path)
                    except OSError as exc:
                        error(entry.path, exc)
        except OSError as exc:
            error(folder, exc)
            continue
        markers = [name for name in ("dataset_dict.json", "state.json", "dataset_info.json", "info.json", "modules.json") if name in files]
        split_files = sorted(name for name in files if Path(name).suffix.lower() in RAW_SUFFIXES and SPLIT_NAME.search(name))
        mbpp_files = sorted(name for name in files if Path(name).suffix.lower() in RAW_SUFFIXES and "mbpp" in name.casefold())
        record = {"directory": str(folder), "depth": depth, "metadata": {},
                  "split_filenames": split_files[:40], "mbpp_filenames": mbpp_files[:20]}
        if len(split_files) > 40 or len(mbpp_files) > 20:
            record["filenames_truncated"] = True
        for name in (["config.json"] if "config.json" in files else []) + [m for m in markers if m != "modules.json"]:
            try:
                info = read_metadata(files[name])
                if name != "config.json" or info:
                    record["metadata"][name] = info
            except (OSError, ValueError, TypeError) as exc:
                error(files[name], exc)
        if "modules.json" in markers:
            record["sentence_transformer_marker"] = "modules.json"
        if record["metadata"] or markers or split_files or mbpp_files:
            result["candidates"].append(record)
        if result["truncated"]:
            break
    result["elapsed_seconds"] = round(time.monotonic() - started, 3)
    return result


def cann_candidates():
    candidates = set()
    for name in ("ASCEND_HOME_PATH", "ASCEND_TOOLKIT_HOME", "TV_CANN_ENV"):
        value = os.environ.get(name)
        if value:
            path = Path(value)
            if path.is_file() and not path.is_symlink():
                candidates.add(str(path))
            if path.is_dir():
                for leaf in ("set_env.sh", "version.info"):
                    candidate = path / leaf
                    if candidate.is_file():
                        candidates.add(str(candidate))
    for raw in ("/usr/local/Ascend/set_env.sh", "/usr/local/Ascend/ascend-toolkit/set_env.sh",
                "/usr/local/Ascend/ascend-toolkit/latest/set_env.sh",
                "/usr/local/Ascend/ascend-toolkit/latest/version.info"):
        if Path(raw).is_file():
            candidates.add(raw)
    return {"existing_candidate_paths": sorted(candidates), "activation_script_selected": None,
            "candidate_is_not_a_verified_activation_choice": True}


def npu_details(enabled, with_memory):
    result = {"torch_import_requested": enabled, "device_count": None,
              "cards": [], "context": "this Notebook/container only"}
    if not enabled:
        result["note"] = "Use --npu or --npu-memory for a runtime probe; npu-smi output is reported separately."
        return result
    # Isolate vendor startup output so stdout of the parent remains one JSON document.
    code = r'''
import json, sys
report = {"available": False, "device_count": None, "cards": []}
try:
    import torch
    import torch_npu
    report["available"] = bool(torch.npu.is_available())
    report["device_count"] = int(torch.npu.device_count())
    for index in range(report["device_count"]):
        card = {"index": index}
        if sys.argv[1] == "memory" and report["available"]:
            try:
                free, total = torch.npu.mem_get_info(index)
                card.update(free_bytes=int(free), total_bytes=int(total))
            except (AttributeError, RuntimeError, TypeError) as exc:
                card["memory_error"] = str(exc)
        report["cards"].append(card)
except Exception as exc:
    report["error"] = f"{type(exc).__name__}: {exc}"
print("\n__TASK_VECTOR_NPU_JSON__" + json.dumps(report))
'''
    try:
        probe = subprocess.run([sys.executable, "-B", "-c", code, "memory" if with_memory else "count"],
                               capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=30,
                               check=False)
        marker = "__TASK_VECTOR_NPU_JSON__"
        prefix, found, payload = probe.stdout.rpartition(marker)
        if found:
            result.update(json.loads(payload.strip()))
        else:
            result["error"] = "Runtime NPU probe produced no structured response"
        result["runtime_probe_returncode"] = probe.returncode
        if prefix.strip():
            result["startup_stdout"] = prefix[-4000:]
        if probe.stderr.strip():
            result["startup_stderr"] = probe.stderr[-4000:]
    except (OSError, subprocess.TimeoutExpired, ValueError) as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--assets-root", default=str(DPC_ROOT))
    parser.add_argument("--max-depth", type=int, choices=range(0, 6), default=5)
    parser.add_argument("--max-directories", type=int, default=1500)
    parser.add_argument("--max-entries", type=int, default=20000)
    parser.add_argument("--max-candidates", type=int, default=200)
    parser.add_argument("--max-seconds", type=float, default=20)
    parser.add_argument("--npu", action="store_true", help="Import torch/torch_npu and inspect this container's NPU count")
    parser.add_argument("--npu-memory", action="store_true", help="Also query per-card memory; does not start training")
    args = parser.parse_args(argv)
    if min(args.max_directories, args.max_entries, args.max_candidates, args.max_seconds) <= 0:
        parser.error("All inventory limits must be positive")
    packages = {}
    for name in ("torch", "torch-npu", "transformers", "datasets", "PyYAML", "sacrebleu", "rouge-score", "sentence-transformers", "vllm-ascend", "verl"):
        try:
            packages[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            packages[name] = None
    report = {"schema_version": 1, "checked_utc": datetime.now(timezone.utc).isoformat(),
              "read_only": True, "cwd": str(Path.cwd()), "executable": sys.executable,
              "python_version": sys.version, "platform": platform.platform(), "packages": packages,
              "npu": npu_details(args.npu or args.npu_memory, args.npu_memory),
              "npu_smi": command(["npu-smi", "info"]),
              "dpc_df": command(["df", "-B1", str(DPC_ROOT)]),
              "dpc_findmnt": command(["findmnt", "-T", str(DPC_ROOT), "-o", "TARGET,SOURCE,FSTYPE,OPTIONS"]),
              "cann": cann_candidates(),
              "asset_inventory": inventory_assets(Path(args.assets_root), max_depth=args.max_depth,
                  max_directories=args.max_directories, max_entries=args.max_entries,
                  max_candidates=args.max_candidates, max_seconds=args.max_seconds),
              "limitations": ["Candidates never establish model/dataset identity or official split provenance.",
                               "Notebook resources do not establish the resources of a separate training job.",
                               "No files are created, removed, edited, downloaded, or moved by this tool."]}
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
