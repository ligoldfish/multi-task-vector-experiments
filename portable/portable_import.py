#!/usr/bin/env python3
"""Import a verified portable experiment into a NEW runtime directory.

Only completed training is reused. Original representations, data manifest and
archived evidence remain byte-for-byte unchanged. Destination request records
explicitly name the original request and the import receipt.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path, PurePosixPath
import shutil
import sys


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def object_hash(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                     separators=(",", ":"), default=str).encode()).hexdigest()


def data_hash(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True).encode()).hexdigest()


def inside(root, relative):
    relative = PurePosixPath(str(relative))
    if relative.is_absolute() or ".." in relative.parts or "\\" in str(relative):
        raise ValueError(f"Unsafe relative path: {relative}")
    root = Path(root).resolve()
    target = root.joinpath(*relative.parts)
    if not target.resolve().is_relative_to(root):
        raise ValueError(f"Path leaves bundle: {relative}")
    return target


def verify_bundle(bundle):
    bundle = Path(bundle).resolve()
    manifest = read_json(bundle / "EXPORT_MANIFEST.json")
    if manifest.get("schema_version") != 1 or not isinstance(manifest.get("files"), dict):
        raise ValueError("Unsupported export manifest")
    for relative, expected in manifest["files"].items():
        path = inside(bundle, relative)
        if path.is_symlink() or not path.is_file() or sha256(path) != expected:
            raise ValueError(f"Bundle integrity failed: {relative}")
    index = read_json(bundle / "PORTABLE_INDEX.json")
    if index.get("schema_version") != 2 or index.get("reuse_policy") != "completed_training_only_reevaluate":
        raise ValueError("Unsupported portable bundle or reuse policy")
    for required in ("PORTABLE_INDEX.json", "code/pyproject.toml", "experiment/data/manifest.json"):
        if required not in manifest["files"]:
            raise ValueError(f"Required entry not covered by export manifest: {required}")
    return index, manifest


def activate_code(bundle):
    code = Path(bundle).resolve() / "code"
    sys.path.insert(0, str(code / "src"))
    return code


def verify_model(model, archived):
    """Match EXACT recognized base files; model name/architecture is insufficient."""
    model = Path(model).resolve()
    expected = archived.get("sha256", {})
    actual_names = {p.name for p in model.glob("*") if p.is_file() and
                    p.suffix in {".json", ".safetensors", ".bin", ".model", ".txt"}}
    if not expected or actual_names != set(expected):
        raise ValueError(f"Base file inventory differs; missing={sorted(set(expected)-actual_names)}, "
                         f"extra={sorted(actual_names-set(expected))}. Use an exact snapshot of the original base.")
    actual = {name: sha256(model / name) for name in sorted(expected)}
    if actual != expected or object_hash(actual) != archived.get("fingerprint"):
        raise ValueError("Target base/tokenizer/config contents differ from the trained base")
    return actual


def verify_data(bundle, manifest):
    if manifest.get("status") != "complete":
        raise ValueError("Prepared data manifest is incomplete")
    required_tasks = {"math", "logic", "translation", "paraphrase", "code_generation", "code_repair", "summary"}
    if set(manifest.get("tasks", {})) != required_tasks:
        raise ValueError("Portable continuation requires all seven prepared task manifests")
    unhashed = {k: v for k, v in manifest.items() if k != "manifest_content_sha256"}
    if data_hash(unhashed) != manifest.get("manifest_content_sha256"):
        raise ValueError("Prepared manifest content hash differs")
    ids = {}
    for task, entry in manifest["tasks"].items():
        for split, expected in entry["files"].items():
            path = inside(bundle, f"experiment/data/{task}/{split}.jsonl")
            if sha256(path) != expected["sha256"]:
                raise ValueError(f"Prepared data SHA mismatch: {task}/{split}")
            rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
            sample_ids = [row["id"] for row in rows]
            if (len(rows) != expected["count"] or len(set(sample_ids)) != len(rows) or
                    data_hash(sample_ids) != expected["ids_sha256"]):
                raise ValueError(f"Prepared IDs/count differ: {task}/{split}")
            ids[(task, split)] = sample_ids
    return ids


def training_request(config, sources, manifest, base, item):
    from task_vector.config import scientific_config
    config = deepcopy(config)
    if item.get("layers") is not None:
        config["ltv"]["layers"] = item["layers"]
    return object_hash({"config": scientific_config(config), "route": item["route"],
                        "seed": item["seed"], "scope": item["scope"] if item["route"] == "loda" else "independent",
                        "task": item["task"], "data": manifest["manifest_content_sha256"],
                        "base": base, "sources": sources})


def statistics_signature(config, sources, manifest, base, train_ids):
    return object_hash({"model": base, "loda": config["loda"],
                        "prompt": config["training"]["max_prompt_tokens"],
                        "samples": manifest["manifest_content_sha256"],
                        "sources": sources, "train_ids": train_ids})


def tensor_mapping_digest(mapping):
    """Hash exact CPU tensor bytes, including shape and dtype, without casting."""
    import torch
    digest = hashlib.sha256()
    for name in sorted(mapping):
        tensor = mapping[name].detach().cpu().contiguous()
        digest.update(object_hash({"key": name, "shape": list(tensor.shape), "dtype": str(tensor.dtype)}).encode())
        digest.update(tensor.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def semantic_files(path):
    root = Path(path).resolve()
    return {p.relative_to(root).as_posix(): sha256(p) for p in sorted(root.rglob("*"))
            if p.is_file() and not any(v in {".cache", ".git", "__pycache__"} for v in p.relative_to(root).parts)}


def import_bundle(bundle, destination, model, device, semantic_model=None):
    bundle, destination, model = Path(bundle).resolve(), Path(destination).resolve(), Path(model).resolve()
    if device != "cpu" and not (device.startswith("cuda:") or device.startswith("npu:")):
        raise ValueError("Device must be cpu, cuda:N or npu:N")
    if destination.exists():
        raise ValueError(f"Destination already exists; use a NEW directory: {destination}")
    if destination.is_relative_to(bundle) or bundle.is_relative_to(destination):
        raise ValueError("Bundle and destination must be separate directories")
    index, export_manifest = verify_bundle(bundle)
    code = activate_code(bundle)
    import torch
    import yaml
    from task_vector.config import source_manifest, scientific_config, validate_config
    from task_vector.runner import ExperimentRunner
    from task_vector.representations import Representation
    from task_vector.subspaces import load_statistics, save_statistics

    sources = source_manifest(code)
    if sources != index["source_manifest"] or object_hash(sources) != index["source_manifest_sha256"]:
        raise ValueError("Frozen experiment source does not match its original training snapshot")
    original_config = deepcopy(index["config_used"])
    validate_config(original_config)
    verify_model(model, index["base_fingerprint"])
    base = index["base_fingerprint"]["fingerprint"]
    manifest = read_json(bundle / "experiment/data/manifest.json")
    ids = verify_data(bundle, manifest)
    original_semantic = inside(bundle, index.get("semantic_model", "assets/semantic_model"))
    semantic_model = Path(semantic_model).resolve() if semantic_model else original_semantic
    original_semantic_files = semantic_files(original_semantic)
    if not original_semantic_files or semantic_files(semantic_model) != original_semantic_files:
        raise ValueError("Target semantic scorer differs from the bundled scorer")

    config = deepcopy(original_config)
    config["model"]["path"] = str(model)
    config["platform"]["device"] = device
    config["platform"]["persistent_root"] = str(destination)
    if "assets_root" in config["platform"]:
        config["platform"]["assets_root"] = str(bundle / "assets")
    config.setdefault("runtime", {})["data_root"] = str(destination / "data")
    config["runtime"]["output_root"] = str(destination / "runs")
    config["metrics"]["semantic_model_path"] = str(semantic_model)
    if original_config.get("training", {}).get("resume_from"):
        raise ValueError("A portable COMPLETE-training import cannot inherit an active resume_from setting")
    validate_config(config)

    # Audit every original completion BEFORE creating the new destination.
    audited = []
    for item in index["training"]:
        folder = inside(bundle, item["directory"])
        record_path = inside(bundle, item["record"])
        record = read_json(record_path)
        expected_request = training_request(original_config, sources, manifest, base, item)
        if (record.get("completion_schema") != 2 or record.get("status") != "complete" or
                record.get("request_signature") != expected_request):
            raise ValueError(f"Original completed request/source/config does not match: {record_path}")
        original_provenance = record.get("provenance", {})
        if (original_provenance.get("base_fingerprint") != base or
                original_provenance.get("data_manifest_sha256") != object_hash(manifest) or
                original_provenance.get("source_manifest_sha256") != object_hash(sources)):
            raise ValueError(f"Original training provenance differs: {record_path}")
        artifacts = {}
        for field, filename in (("raw_path", "raw.pt"), ("post_path", "post.pt"), ("initial_path", "initial.pt")):
            path = folder / filename
            archived = record.get("artifacts", {}).get(field, {})
            if sha256(path) != archived.get("sha256"):
                raise ValueError(f"Original artifact changed: {path}")
            rep = Representation.load(path)
            digest = ExperimentRunner.representation_digest(rep)
            if (rep.metadata.get("base_fingerprint") != base or
                    rep.metadata.get("training_request_signature") != expected_request or
                    rep.seed != item["seed"] or rep.task != item["task"] or
                    digest != archived.get("representation_digest")):
                raise ValueError(f"Original representation contract differs: {path}")
            artifacts[field] = {"filename": filename, "sha256": archived["sha256"], "representation_digest": digest}
        audited.append((item, record, artifacts, sha256(record_path)))

    # No old locks, predictions or smoke records enter the active runs directory.
    destination.mkdir(parents=True)
    shutil.copytree(bundle / "experiment/data", destination / "data")
    (destination / "runs").mkdir()
    config_path = destination / "portable-config.yaml"
    config_path.write_text(yaml.safe_dump(config, allow_unicode=True, sort_keys=False), encoding="utf-8")
    receipt = {"schema_version": 1, "kind": "portable_completed_training_import",
               "created_utc": datetime.now(timezone.utc).isoformat(), "bundle": str(bundle),
               "source_root": str(code), "destination": str(destination),
               "base_fingerprint": base, "original_device": original_config["platform"]["device"],
               "target_device": device, "runtime_validation": "pending_real_model_smoke_on_target",
               "export_manifest_sha256": sha256(bundle / "EXPORT_MANIFEST.json"),
               "original_manifest_sha256": sha256(bundle / "experiment/data/manifest.json"),
               "manifest_bytes_unchanged": True, "representation_bytes_unchanged": True,
               "source_manifest_sha256": object_hash(sources),
               "original_config_sha256": object_hash(scientific_config(original_config)),
               "target_config_sha256": object_hash(scientific_config(config)),
               "target_full_config_sha256": object_hash(config),
               "reuse_policy": "completed_training_only_reevaluate", "training": [], "statistics": []}
    runner = ExperimentRunner(config, source_root=code)
    if runner.base_fingerprint() != base:
        raise ValueError("Destination model fingerprint unexpectedly changed")
    for item, original, artifacts, original_record_sha in audited:
        folder = inside(bundle, item["directory"])
        out = runner.directory(item["scope"], item["route"], item["seed"], item["task"], item.get("layers"))
        out.mkdir(parents=True, exist_ok=False)
        # Completed adapters and descriptive logs only. Optimizer and uncommitted
        # state stay in the immutable archive and cannot trigger silent resume.
        for filename in ("raw.pt", "post.pt", "initial.pt", "subspaces.pt", "best-adapter.pt", "epochs.json", "steps.jsonl"):
            if (folder / filename).is_file():
                shutil.copy2(folder / filename, out / filename)
        new_request = training_request(config, sources, manifest, base, item)
        result = deepcopy(original)
        new_artifacts = {}
        for field, info in artifacts.items():
            result[field] = str(out / info["filename"])
            new_artifacts[field] = {"path": result[field], "sha256": info["sha256"],
                                    "representation_digest": info["representation_digest"]}
        for field in ("best_checkpoint", "last_checkpoint", "resume_from"):
            result[field] = None
        result["best_adapter_path"] = str(out / "best-adapter.pt") if (out / "best-adapter.pt").exists() else None
        result["best_checkpoint_resume_supported"] = False
        result["artifacts"] = new_artifacts
        result["request_signature"] = new_request
        result["training_origin"] = "imported_completed"
        origin = {"original_record": item["record"], "original_record_sha256": original_record_sha,
                  "original_request_signature": original["request_signature"],
                  "export_manifest_sha256": receipt["export_manifest_sha256"],
                  "original_provenance": original.get("provenance"),
                  "original_directory": item["directory"], "weights_retrained": False,
                  "optimizer_resume_supported": False, "original_runtime_seconds": original.get("runtime_seconds")}
        result["imported_from"] = origin
        training_path = out / "training.json"
        training_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        receipt["training"].append({"record": training_path.relative_to(destination).as_posix(),
                                    "record_sha256": sha256(training_path), "request_signature": new_request,
                                    "imported_from": origin, "artifacts": new_artifacts})

    stats_root = bundle / "experiment/runs/statistics/frozen_base"
    for path in sorted(stats_root.glob("*.pt")) if stats_root.exists() else []:
        task = path.stem
        stats = load_statistics(path)
        old_signature = statistics_signature(original_config, sources, manifest, base, ids[(task, "train")])
        if stats.metadata.get("runner_signature") != old_signature:
            raise ValueError(f"Archived calibration signature differs: {path}")
        digest = tensor_mapping_digest(stats)
        stats.metadata["runner_signature"] = statistics_signature(config, sources, manifest, base, ids[(task, "train")])
        stats.metadata["portable_import"] = {"original_sha256": sha256(path), "tensor_digest": digest,
                                             "original_runner_signature": old_signature,
                                             "calibration_backend": original_config["platform"]["device"],
                                             "recalibrated_on_target": False}
        target = destination / "runs/statistics/frozen_base" / path.name
        target.parent.mkdir(parents=True, exist_ok=True)
        save_statistics(stats, target)
        if tensor_mapping_digest(load_statistics(target)) != digest:
            raise ValueError(f"Calibration tensors changed during import: {target}")
        receipt["statistics"].append({"path": target.relative_to(destination).as_posix(),
                                      "sha256": sha256(target), "tensor_digest": digest})
    receipt["receipt_content_sha256"] = object_hash(receipt)
    receipt_path = destination / "IMPORT_RECEIPT.json"
    receipt_path.write_text(json.dumps(receipt, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"IMPORTED_COMPLETED_TRAINING": len(audited), "IMPORT_STATUS": "PASS",
                      "CONFIG": str(config_path), "RECEIPT": str(receipt_path),
                      "NEXT": "portable_run.py --destination ... preflight, then smoke, then all",
                      "MODEL_RUNTIME_VALIDATED": False}, ensure_ascii=False, indent=2))
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", required=True)
    parser.add_argument("--destination", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--device", required=True, help="cpu, cuda:0 or npu:0")
    parser.add_argument("--semantic-model")
    args = parser.parse_args()
    import_bundle(args.bundle, args.destination, args.model, args.device, args.semantic_model)


if __name__ == "__main__":
    main()
