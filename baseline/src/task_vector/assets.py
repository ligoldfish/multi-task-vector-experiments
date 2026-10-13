"""Read-only checks of explicitly configured ModelMate asset paths."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

from .io import atomic_json, file_sha256, object_sha256


def _file_record(path, hash_file=False):
    record = {"name": path.name, "bytes": path.stat().st_size}
    if hash_file:
        record["sha256"] = file_sha256(path)
    return record


def _model_check(path_value, *, expected=None, hash_weights=False):
    if not path_value:
        return {"status": "missing", "reason": "An explicit local path is required; no directory guessing is performed"}
    path = Path(path_value).expanduser()
    result = {"path": str(path), "status": "missing"}
    if not path.is_dir():
        result["reason"] = "Configured model directory does not exist"
        return result
    cfg_path = path / "config.json"
    if not cfg_path.is_file():
        result["reason"] = "config.json is absent"
        return result
    with cfg_path.open(encoding="utf-8") as handle:
        cfg = json.load(handle)
    result["config"] = {key: cfg.get(key) for key in ("model_type", "architectures", "num_hidden_layers", "hidden_size", "num_attention_heads", "num_key_value_heads", "head_dim", "torch_dtype")}
    mismatch = {key: {"expected": value, "actual": cfg.get(key)} for key, value in (expected or {}).items() if cfg.get(key) != value}
    if mismatch:
        result.update(status="invalid", reason="Model configuration differs", mismatches=mismatch)
        return result
    index_files = [p for p in (path / "model.safetensors.index.json", path / "pytorch_model.bin.index.json") if p.is_file()]
    weights = []
    if index_files:
        for index in index_files:
            with index.open(encoding="utf-8") as handle:
                shards = set(json.load(handle)["weight_map"].values())
            for name in sorted(shards):
                weight = (path / name).resolve()
                if path.resolve() not in weight.parents:
                    raise ValueError("A model index references a weight outside its directory")
                weights.append(weight)
    else:
        weights = [p for p in (path / "model.safetensors", path / "pytorch_model.bin") if p.is_file()]
    missing = [str(p) for p in weights if not p.is_file()]
    if not weights or missing:
        result.update(reason="Model weight files are incomplete", missing_shards=missing)
        return result
    tokenizer_files = [p for p in (path / "tokenizer.json", path / "tokenizer.model", path / "vocab.json") if p.is_file()]
    if not tokenizer_files:
        result["reason"] = "No local tokenizer vocabulary file was found"
        return result
    result.update(status="ready", weight_files=[_file_record(p, hash_weights) for p in sorted(set(weights))],
                  config_sha256=file_sha256(cfg_path), tokenizer_files=[_file_record(p, True) for p in tokenizer_files],
                  tokenizer_config_present=(path / "tokenizer_config.json").is_file())
    return result


def inspect_assets(config, output_path=None, *, hash_weights=False):
    """Inspect presence/integrity; dataset provenance is validated by prepare.

    A ready asset inventory does not mean NPU capacity or scientific validity
    has passed. Weight hashes can be requested explicitly for a full snapshot.
    """
    expected = {"model_type": "qwen3", "num_hidden_layers": 28, "hidden_size": 2048,
                "num_attention_heads": 16, "num_key_value_heads": 8, "head_dim": 128,
                **config["model"].get("expected_config", {})}
    model = _model_check(config["model"].get("path"), expected=expected, hash_weights=hash_weights)
    tasks = list(config["data"]["tasks"]) + list(config["data"].get("extension_tasks", [])) + list(config["data"].get("auxiliary_tasks", []))
    sources = config["data"].get("sources", {})
    datasets = {}
    for task in tasks:
        source = sources.get(task, {})
        if isinstance(source, str):
            source = {"path": source}
        path_value = source.get("path")
        if not path_value:
            status = "remote_explicitly_allowed" if source.get("allow_remote", False) else "missing"
            datasets[task] = {"status": status, "dataset_id": source.get("dataset_id"),
                              "reason": "No explicit local source path configured"}
            continue
        path = Path(path_value).expanduser()
        record = {"path": str(path), "status": "ready" if path.exists() else "missing"}
        if path.is_file():
            record.update(_file_record(path, hash_weights))
        elif path.is_dir():
            # A compact top-level inventory, not an expensive recursive scan of
            # unrelated datasets in a persistent volume.
            record["top_level_entries"] = sorted(p.name for p in path.iterdir())
            record["format"] = "hf_saved_dataset" if (path / "dataset_dict.json").exists() or (path / "state.json").exists() else "raw_dataset_directory"
        datasets[task] = record
    semantic_path = config.get("metrics", {}).get("semantic_model_path")
    semantic = {"path": semantic_path, "status": "ready" if semantic_path and Path(semantic_path).is_dir() else "missing"}
    result = {"schema_version": 1, "checked_utc": datetime.now(timezone.utc).isoformat(),
              "model": model, "datasets": datasets, "semantic_model": semantic,
              "persistent_root": config["platform"]["persistent_root"],
              "scientific_validation": "not_evaluated", "device_capacity": "not_evaluated",
              "mvp_ready": model["status"] == "ready" and semantic["status"] == "ready" and
                all(datasets[t]["status"] in {"ready", "remote_explicitly_allowed"} for t in config["data"]["tasks"]),
              "all_scope_ready": model["status"] == "ready" and semantic["status"] == "ready" and
                all(item["status"] in {"ready", "remote_explicitly_allowed"} for item in datasets.values()),
              "hash_weights": hash_weights}
    result["inventory_sha256"] = object_sha256({k: v for k, v in result.items() if k != "checked_utc"})
    if output_path:
        atomic_json(output_path, result)
    return result


def main():
    from .config import load_config
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--hash-weights", action="store_true")
    args = parser.parse_args()
    result = inspect_assets(load_config(args.config), args.output, hash_weights=args.hash_weights)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
