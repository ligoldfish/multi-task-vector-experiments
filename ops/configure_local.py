#!/usr/bin/env python3
"""Create a new local-only experiment configuration without changing its science."""
from __future__ import annotations

import argparse
from copy import deepcopy
import json
import os
from pathlib import Path
import sys

import yaml


PROJECT = Path(__file__).resolve().parents[1]
DPC_ROOT = Path("/opt/dpcvol/datasets")
TASKS = ("math", "logic", "translation", "paraphrase", "code_generation", "code_repair", "summary")
RAW_SUFFIXES = {".json", ".jsonl", ".ndjson", ".tsv", ".csv", ".parquet"}
REQUIRED = {"model_path", "semantic_model_path", "persistent_root", "runtime_python", "cann_env", "expected_devices", "data_sources"}
SOURCE_KEYS = {"path", "files", "train_split", "eval_split", "split", "world", "max_depth", "source_depth", "revision", "allow_remote", "dataset_id", "config"}


def require_path(raw, label, *, directory=None, within_dpc=True):
    if not isinstance(raw, str) or not raw.strip():
        raise ValueError(f"{label}: an explicit absolute existing path is required")
    given = Path(raw)
    if not given.is_absolute():
        raise ValueError(f"{label}: use an absolute path")
    path = given.resolve(strict=True)
    if within_dpc and (path == DPC_ROOT or DPC_ROOT not in path.parents):
        raise ValueError(f"{label}: resolved path must be a child of /opt/dpcvol/datasets: {path}")
    if directory is True and not path.is_dir():
        raise ValueError(f"{label}: a directory is required")
    if directory is False and not path.is_file():
        raise ValueError(f"{label}: a file is required")
    return path


def source_configuration(task, raw, original):
    if not isinstance(raw, dict):
        raise ValueError(f"data_sources.{task}: use a path/files object")
    unknown = set(raw) - SOURCE_KEYS
    if unknown:
        raise ValueError(f"data_sources.{task}: unsupported keys {sorted(unknown)}")
    has_path, has_files = bool(raw.get("path")), bool(raw.get("files"))
    if has_path == has_files:
        raise ValueError(f"data_sources.{task}: provide exactly one of path or files")
    result = deepcopy(original)
    train = result.get("train_split", "train")
    evaluation = result.get("eval_split", "validation" if task == "paraphrase" else "test")
    for key, expected in (("train_split", train), ("eval_split", evaluation)):
        if key in raw and raw[key] != expected:
            raise ValueError(f"{task}.{key}: official source split is fixed at {expected!r}")
    if train == evaluation:
        raise ValueError(f"{task}: official training and evaluation splits must be distinct")
    for key in ("dataset_id", "config"):
        if key in raw and raw[key] != result.get(key):
            raise ValueError(f"{task}.{key}: cannot change the documented dataset identity")
    if raw.get("allow_remote") not in (None, False):
        raise ValueError(f"{task}: this tool only configures local assets")
    result.update(allow_remote=False, train_split=train, eval_split=evaluation)
    if has_path:
        path = require_path(raw["path"], f"{task}.path")
        if path.is_file() and path.suffix.casefold() not in RAW_SUFFIXES:
            raise ValueError(f"{task}.path: unsupported raw file; archives must first be extracted")
        result["path"] = str(path)
        result.pop("files", None)
    else:
        files = raw["files"]
        if not isinstance(files, dict) or not all(isinstance(key, str) for key in files):
            raise ValueError(f"{task}.files: official split -> path or path list is required")
        if not {train, evaluation}.issubset(files):
            raise ValueError(f"{task}.files: both official {train!r} and {evaluation!r} are required")
        normalized, seen = {}, {}
        for split, values in files.items():
            paths = [values] if isinstance(values, str) else values
            if not isinstance(paths, list) or not paths:
                raise ValueError(f"{task}.files.{split}: nonempty path or list required")
            resolved = []
            for value in paths:
                path = require_path(value, f"{task}.files.{split}", directory=False)
                if path.suffix.casefold() not in RAW_SUFFIXES:
                    raise ValueError(f"{task}.files.{split}: unsupported file suffix {path.suffix}")
                if str(path) in seen:
                    raise ValueError(f"{task}: the same file cannot occur twice or in multiple official splits: {path}")
                seen[str(path)] = split
                resolved.append(str(path))
            normalized[split] = resolved[0] if isinstance(values, str) else resolved
        result["files"] = normalized
        result["path"] = None
    if "split" in raw:
        if raw["split"] not in (train, evaluation):
            raise ValueError(f"{task}.split: must identify an actual documented official source split")
        result["split"] = raw["split"]
    if "revision" in raw:
        revision = raw["revision"]
        if revision is not None and not isinstance(revision, str):
            raise ValueError(f"{task}.revision: use a source revision string or null")
        result["revision"] = revision
    if task == "logic":
        if raw.get("world", "OWA") != "OWA" or raw.get("max_depth", 3) != 3:
            raise ValueError("ProofWriter must remain OWA with maximum depth 3")
        result.update(world="OWA", max_depth=3)
        if "source_depth" in raw:
            depth = raw["source_depth"]
            if isinstance(depth, bool) or not isinstance(depth, int) or not 0 <= depth <= 3:
                raise ValueError("ProofWriter source_depth must truthfully identify a corpus depth in 0..3")
            result["source_depth"] = depth
    elif {"world", "max_depth", "source_depth"} & set(raw):
        raise ValueError(f"{task}: ProofWriter-specific keys are not applicable")
    return result


def output_path(raw, label):
    path = Path(raw).absolute()
    if os.path.lexists(path):
        raise FileExistsError(f"{label} already exists; refusing to overwrite: {path}")
    parent = path.parent.resolve(strict=True)
    if not parent.is_dir():
        raise ValueError(f"{label}: output parent must already be a directory")
    if parent != PROJECT and PROJECT not in parent.parents:
        raise ValueError(f"{label}: output must be within this project checkout")
    return parent / path.name


def write_pair_exclusive(config_path, config_bytes, settings_path, settings_bytes):
    """Reserve both new outputs exclusively; roll back our own files on an exception."""
    handles, published = [], []
    try:
        for destination in (config_path, settings_path):
            fd = os.open(destination, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
            handle = os.fdopen(fd, "wb", buffering=0)
            handles.append(handle)
            published.append((destination, os.fstat(handle.fileno())))
        for handle, content in zip(handles, (config_bytes, settings_bytes)):
            pending = memoryview(content)
            while pending:
                written = handle.write(pending)
                if not written:
                    raise OSError("Configuration write did not make progress")
                pending = pending[written:]
            os.fsync(handle.fileno())
        for handle in handles:
            try:
                handle.close()
            except OSError:
                pass
    except BaseException:
        for handle in handles:
            try:
                handle.close()
            except OSError:
                pass
        for destination, identity in reversed(published):
            try:
                current = destination.stat(follow_symlinks=False)
                if (current.st_dev, current.st_ino) == (identity.st_dev, identity.st_ino):
                    destination.unlink()
            except FileNotFoundError:
                pass
        raise
    finally:
        for handle in handles:
            handle.close()


def configure(paths, config_output, settings_output):
    if not isinstance(paths, dict) or set(paths) != REQUIRED:
        keys = set(paths) if isinstance(paths, dict) else set()
        raise ValueError(f"paths JSON keys: missing={sorted(REQUIRED - keys)}, extra={sorted(keys - REQUIRED)}")
    if type(paths["expected_devices"]) is not int or paths["expected_devices"] != 8:
        raise ValueError("expected_devices must be integer 8")
    config_path = output_path(config_output, "config")
    settings_path = output_path(settings_output, "settings")
    if config_path == settings_path:
        raise ValueError("Configuration and settings outputs must be different files")
    model = require_path(paths["model_path"], "model_path", directory=True)
    semantic = require_path(paths["semantic_model_path"], "semantic_model_path", directory=True)
    persistent = require_path(paths["persistent_root"], "persistent_root", directory=True)
    if os.path.lexists(persistent / "data" / "manifest.json"):
        raise FileExistsError("This persistent root already has a data manifest; select a new run root")
    python = require_path(paths["runtime_python"], "runtime_python", directory=False)
    if not os.access(python, os.X_OK):
        raise ValueError("runtime_python is not executable")
    if not (python.parent.parent / "pyvenv.cfg").is_file():
        raise ValueError("runtime_python must belong to an explicit persistent venv (pyvenv.cfg missing)")
    cann = paths["cann_env"]
    if cann is not None:
        cann = str(require_path(cann, "cann_env", directory=False, within_dpc=False))
    sources = paths["data_sources"]
    if not isinstance(sources, dict) or set(sources) != set(TASKS):
        raise ValueError("data_sources must explicitly contain all seven canonical task keys")
    with (PROJECT / "configs" / "experiment.yaml").open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    original = deepcopy(config)
    config["model"]["path"] = str(model)
    config["metrics"]["semantic_model_path"] = str(semantic)
    config["platform"]["persistent_root"] = str(persistent)
    for task in TASKS:
        config["data"]["sources"][task] = source_configuration(task, sources[task], original["data"]["sources"][task])
    # Direct validation intentionally bypasses environment overrides and imports no model/NPU.
    sys.path.insert(0, str(PROJECT / "src"))
    from task_vector.config import validate_config
    validate_config(config)
    settings = {"runtime_python": str(python), "cann_env": cann, "expected_devices": 8}
    config_bytes = yaml.safe_dump(config, allow_unicode=True, sort_keys=False).encode("utf-8")
    settings_bytes = (json.dumps(settings, ensure_ascii=False, indent=2) + "\n").encode("utf-8")
    # All path and scientific validation has finished before creating either output.
    write_pair_exclusive(config_path, config_bytes, settings_path, settings_bytes)
    return {"config": str(config_path), "settings": str(settings_path),
            "persistent_root": str(persistent), "all_sources_local": True,
            "scientific_hyperparameters_unchanged": True,
            "note": "Official split contents, model identity, capacity, and NPU correctness still require assets/prepare/preflight/smoke."}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--paths", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--settings", required=True)
    args = parser.parse_args(argv)
    try:
        with Path(args.paths).open(encoding="utf-8") as handle:
            paths = json.load(handle)
        result = configure(paths, args.config, args.settings)
    except Exception as exc:
        print(f"Configuration rejected: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
