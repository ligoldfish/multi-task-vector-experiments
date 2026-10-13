#!/usr/bin/env python3
"""Attach completed standalone vectors to a new, audited experiment placement.

This is completed-adapter reuse, not optimizer-checkpoint resume. The original
seven-task prepared data are required to continue the same experiment. Nothing
is overwritten, and original tensor/manifest/source evidence is retained.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import io
import json
from pathlib import Path, PurePosixPath
import shutil
import stat
import sys
import zipfile


TASKS = {"math", "logic", "translation", "paraphrase", "code_generation", "code_repair", "summary"}
SPLITS = {"train", "tune", "discovery", "test", "test_formal", "test_confirmation"}


def object_hash(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                    separators=(",", ":"), default=str).encode()).hexdigest()


def data_hash(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True).encode()).hexdigest()


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def safe_relative(name):
    p = PurePosixPath(name)
    if (p.is_absolute() or not p.parts or any(v in {"..", "."} for v in p.parts)
            or "\\" in name or ":" in name):
        raise ValueError(f"Unsafe bundle path: {name}")
    return p


class VectorBundle:
    """Read an exported directory or ZIP without executing or extracting it."""
    def __init__(self, path):
        self.path = Path(path).resolve()
        self.archive = None
        self.digests = None
        if self.path.is_file():
            self.archive = zipfile.ZipFile(self.path)
            info = self.archive.infolist()
            names = [i.filename for i in info if not i.is_dir()]
            if len(names) != len(set(names)):
                raise ValueError("Duplicate ZIP entries")
            for i in info:
                safe_relative(i.filename.rstrip("/"))
                if stat.S_ISLNK(i.external_attr >> 16):
                    raise ValueError(f"Symlink ZIP entry is not allowed: {i.filename}")
            self.names = set(names)
        elif self.path.is_dir():
            self.names = set()
            for entry in self.path.rglob("*"):
                if entry.is_symlink():
                    raise ValueError(f"Symlink bundle entry is not allowed: {entry}")
                if entry.is_file():
                    self.names.add(entry.relative_to(self.path).as_posix())
        else:
            raise ValueError(f"Vector bundle is absent: {self.path}")
        for name in self.names:
            safe_relative(name)
        self.manifest_bytes = self.read("EXPORT_MANIFEST.json")
        manifest = json.loads(self.manifest_bytes)
        if manifest.get("schema_version") != 1 or not isinstance(manifest.get("files"), dict):
            raise ValueError("Unsupported export manifest")
        if self.names != set(manifest["files"]) | {"EXPORT_MANIFEST.json"}:
            raise ValueError("Bundle file inventory differs from EXPORT_MANIFEST.json")
        self.digests = manifest["files"]
        for name, expected in manifest["files"].items():
            if hashlib.sha256(self.read(name)).hexdigest() != expected:
                raise ValueError(f"Bundle SHA256 mismatch: {name}")
        self.index = self.json("VECTOR_INDEX.json")
        self.contract = self.json("BASE_MODEL_CONTRACT.json")
        if self.index.get("schema_version") != 1 or not isinstance(self.index.get("representations"), list):
            raise ValueError("Unsupported standalone VECTOR_INDEX.json")

    def read(self, name):
        safe_relative(name)
        if name not in self.names:
            raise ValueError(f"Missing bundle member: {name}")
        raw = (self.archive.read(name) if self.archive else
               self.path.joinpath(*PurePosixPath(name).parts).read_bytes())
        if self.digests and name in self.digests and hashlib.sha256(raw).hexdigest() != self.digests[name]:
            raise ValueError(f"Bundle SHA256 mismatch or concurrent change: {name}")
        return raw

    def json(self, name):
        return json.loads(self.read(name))

    def close(self):
        if self.archive:
            self.archive.close()


def normalize_protocol(config):
    """Exclude placement only; prepared bytes bind raw-source choices separately."""
    result = deepcopy(config)
    result.pop("platform", None)
    result.pop("runtime", None)
    result.pop("acceleration", None)
    result.get("model", {}).pop("path", None)
    result.get("metrics", {}).pop("semantic_model_path", None)
    result.get("data", {}).pop("sources", None)
    result.get("training", {}).pop("resume_from", None)
    return result


def verify_prepared_data(root):
    root = Path(root).resolve()
    path = root / "manifest.json"
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("status") != "complete" or set(manifest.get("tasks", {})) != TASKS:
        raise ValueError("Continuation requires the original complete seven-task prepared data")
    if data_hash({k: v for k, v in manifest.items() if k != "manifest_content_sha256"}) != manifest.get("manifest_content_sha256"):
        raise ValueError("Prepared data manifest content hash differs")
    required = {"manifest.json": sha256(path)}
    for task, entry in manifest["tasks"].items():
        if not SPLITS <= set(entry.get("files", {})):
            raise ValueError(f"Missing original prepared splits: {task}")
        for split, expected in entry["files"].items():
            if split not in SPLITS:
                raise ValueError(f"Unknown prepared split: {split}")
            name = f"{task}/{split}.jsonl"
            path = root / name
            if sha256(path) != expected.get("sha256"):
                raise ValueError(f"Prepared data file SHA256 differs: {name}")
            rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
            ids = [row["id"] for row in rows]
            if (len(rows) != expected.get("count") or len(ids) != len(set(ids))
                    or data_hash(ids) != expected.get("ids_sha256")):
                raise ValueError(f"Prepared data IDs/count differ: {name}")
            required[name] = expected["sha256"]
        for name in ("rejected.jsonl", "capacity_preflight.json", "group_audit.json"):
            if (root / task / name).is_file():
                required[f"{task}/{name}"] = sha256(root / task / name)
    return manifest, required


def verify_training_compatibility(source_root, original_digest, target_digest):
    """Accept exact code or one documented completed-representation transition."""
    if original_digest == target_digest:
        return {"exact_source_match": True, "source_training_package_sha256": original_digest,
                "target_training_package_sha256": target_digest, "policy": "exact_training_package"}
    path = Path(source_root) / "provenance/representation-compatibility.json"
    if not path.is_file():
        raise ValueError("Scientific Python source differs; no documented representation compatibility")
    contract = json.loads(path.read_text(encoding="utf-8"))
    if (original_digest != "7e8f712d4874a19da43716aba98c755dd7294a6ab13080d7a92d0c994e59398d"
            or target_digest != "b9788996fcf2009bc07c62e632810077469c444cae4436244fc42b46d53ffded"
            or contract.get("source_training_package_sha256") != original_digest
            or contract.get("target_training_package_sha256") != target_digest):
        raise ValueError("Scientific Python source is outside the documented compatibility contract")
    required = {"data.py", "prompts.py", "training.py", "ltv.py", "loda.py", "gao.py", "subspaces.py",
                "representations.py", "losses.py", "baselines.py", "joint.py", "sequential.py", "modeling.py", "code_execution.py", "io.py"}
    checks = contract.get("unchanged_method_sources", {})
    if set(checks) != {"src/task_vector/" + name for name in required}:
        raise ValueError("Scientific Python source compatibility omits a required method file")
    for name, expected in checks.items():
        if sha256(Path(source_root) / name) != expected:
            raise ValueError("Scientific Python source method changed: " + name)
    return {"exact_source_match": False, "source_training_package_sha256": original_digest,
            "target_training_package_sha256": target_digest, "policy": "documented_completed_representation_transition",
            "unchanged_method_files": checks, "compatibility_contract_sha256": sha256(path),
            "new_evaluation_required": True, "optimizer_resume_supported": False}


def attach_vectors(bundle_path, config_path, source_root, destination, prepared_data=None):
    """Audit first, then create an immutable evidence snapshot and fresh run root."""
    import yaml
    source_root, destination = Path(source_root).resolve(), Path(destination).resolve()
    config_path = Path(config_path).resolve()
    if destination.exists():
        raise ValueError(f"Destination already exists; choose a NEW directory: {destination}")
    if destination.is_relative_to(source_root) or source_root.is_relative_to(destination):
        raise ValueError("Destination and source code must be separate directories")
    if not (source_root / "src/task_vector/training.py").is_file():
        raise ValueError("--source-root must name the project scientific source directory")
    sys.path.insert(0, str(source_root / "src"))
    from task_vector.config import source_manifest, scientific_config, validate_config
    from task_vector.data import load_samples
    from task_vector.io import object_sha256
    from task_vector.representations import Representation
    from task_vector.runner import ExperimentRunner
    from task_vector.training import _content_digest, _sample_content

    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    validate_config(config)
    template = yaml.safe_load((source_root / "configs/experiment.yaml").read_text(encoding="utf-8"))
    if normalize_protocol(config) != normalize_protocol(template):
        raise ValueError("Target scientific configuration differs from the published experiment protocol")
    device = config["platform"]["device"]
    if device != "cpu" and not (device.startswith("cuda:") or device.startswith("npu:")):
        raise ValueError("Target device must be cpu, cuda:N or npu:N")
    if config["training"].get("resume_from"):
        raise ValueError("This completed-vector import does not resume an old optimizer checkpoint")
    prepared = Path(prepared_data or config.get("runtime", {}).get("data_root")
                    or Path(config["platform"]["persistent_root"]) / "data").resolve()
    if prepared.is_relative_to(destination) or destination.is_relative_to(prepared):
        raise ValueError("Original prepared data and NEW destination must be separate directories")
    manifest, data_files = verify_prepared_data(prepared)
    model = Path(config["model"]["path"]).resolve()
    semantic = Path(config["metrics"]["semantic_model_path"]).resolve()
    if not semantic.is_dir():
        raise ValueError("Set metrics.semantic_model_path to the local all-MiniLM-L6-v2 snapshot")
    package = source_root / "src/task_vector"
    package_hash = object_sha256({p.name: sha256(p) for p in sorted(package.glob("*.py"))})
    bundle = VectorBundle(bundle_path)
    try:
        contract = bundle.contract
        expected_files = contract.get("files", {})
        model_files = {p.name for p in model.glob("*") if p.is_file() and
                       p.suffix in {".json", ".safetensors", ".bin", ".model", ".txt"}}
        actual = {name: sha256(model / name) for name in sorted(model_files)}
        if (not expected_files or actual != expected_files
                or object_hash(actual) != contract.get("fingerprint")
                or contract.get("model_id") != config["model"]["id"]
                or contract.get("expected_config") != config["model"]["expected_config"]
                or contract.get("enable_thinking") is not False
                or contract.get("intervention_scale") != 1.0):
            raise ValueError("Target base/tokenizer/config contents differ from the original base contract")
        base = contract["fingerprint"]
        audited = []
        seen = set()
        compatibility_checks = {}
        for item in bundle.index["representations"]:
            route, scope, task, seed, layers = (item.get(k) for k in ("route", "scope", "task", "seed", "layers"))
            if (route not in {"ltv", "loda"} or task not in TASKS or type(seed) is not int
                    or seed not in config["analysis"]["seeds"]
                    or scope not in {"mvp", "expanded", "loo-math", "loo-logic", "loo-translation", "loo-paraphrase"}
                    or (layers is not None and (route != "ltv" or not isinstance(layers, list)
                                                or any(type(v) is not int or not 0 <= v < 28 for v in layers)))):
                raise ValueError("Invalid standalone vector identity")
            allowed_tasks = list(config["data"]["tasks"])
            if scope == "expanded":
                allowed_tasks += list(config["data"]["extension_tasks"])
            elif scope.startswith("loo-"):
                allowed_tasks.remove(scope[4:])
            if task not in allowed_tasks:
                raise ValueError(f"Imported vector task is outside the recorded scope: {scope}/{task}")
            identity = (route, scope if route == "loda" else "independent", seed, task, tuple(layers or [14]))
            if identity in seen:
                raise ValueError(f"Duplicate standalone vector identity: {identity}")
            seen.add(identity)
            original_name = item["training_origin"]
            record = bundle.json(original_name)
            provenance = record.get("provenance", {})
            request = record.get("request_signature")
            if (record.get("completion_schema") != 2 or record.get("status") != "complete" or not request
                    or provenance.get("request_signature") != request
                    or provenance.get("base_fingerprint") != base
                    or record.get("provenance_sha256") != object_hash(provenance)
                    or provenance.get("data_manifest_sha256") != object_hash(manifest)):
                raise ValueError(f"Original completion/model/prepared-manifest provenance differs: {original_name}")
            compatibility_checks[request] = verify_training_compatibility(
                source_root, provenance.get("training_sources_sha256"), package_hash)
            samples = {split: load_samples(prepared, task, split) for split in ("train", "tune")}
            sample_digest = _content_digest({split: _sample_content(rows) for split, rows in samples.items()})
            if sample_digest != provenance.get("sample_content_sha256"):
                raise ValueError(f"Original train/tune content differs: {task}")
            artifacts = {}
            for field, filename in (("raw_path", "raw.pt"), ("post_path", "post.pt"), ("initial_path", "initial.pt")):
                info = item["exports"][field]
                original_info = record.get("artifacts", {}).get(field, {})
                raw = bundle.read(info["path"])
                digest = hashlib.sha256(raw).hexdigest()
                rep = Representation.load(io.BytesIO(raw))
                content_digest = ExperimentRunner.representation_digest(rep)
                expected_layers = layers or config["ltv"]["layers"]
                subspace_meta = rep.metadata.get("subspaces", {})
                wrong_loda_scope = (route == "loda" and (not subspace_meta or any(
                    info.get("task") != task or info.get("seen_tasks") != allowed_tasks
                    for info in subspace_meta.values())))
                if (digest != info.get("sha256") or digest != original_info.get("sha256")
                        or content_digest != info.get("representation_digest")
                        or content_digest != original_info.get("representation_digest")
                        or rep.metadata.get("base_fingerprint") != base
                        or rep.metadata.get("training_request_signature") != request
                        or rep.metadata.get("source_scope") != (scope if route == "loda" else "independent")
                        or rep.metadata.get("source_ids") != [sample.id for sample in samples["train"]]
                        or (route == "loda" and rep.metadata.get("heldout_excluded") !=
                            (scope[4:] if scope.startswith("loo-") else None))
                        or wrong_loda_scope
                        or (route == "ltv" and (rep.metadata.get("layers") != expected_layers or
                                               set(rep.values) != {f"model.layers.{layer}" for layer in expected_layers}))
                        or rep.task != task or rep.seed != seed
                        or rep.space != ("activation" if route == "ltv" else "parameter")):
                    raise ValueError(f"Original representation tensor/identity/provenance differs: {info['path']}")
                artifacts[field] = {"filename": filename, "member": info["path"], "sha256": digest,
                                    "representation_digest": content_digest}
            audited.append((item, record, artifacts))
        if not audited:
            raise ValueError("At least one completed task representation is required")

        # All input contracts have passed; create only a fresh destination.
        destination.mkdir(parents=True, exist_ok=False)
        archive = destination / "vector-origin-archive"
        archive.mkdir()
        code = archive / "code"
        shutil.copytree(source_root, code, ignore=shutil.ignore_patterns(".git", "__pycache__", ".pytest_cache", "*.pyc", ".venv*"))
        archived_sources = source_manifest(code)
        if archived_sources != source_manifest(source_root):
            raise ValueError("Source changed during import; incomplete destination is retained")
        archive_files = {}
        for name in sorted(bundle.names):
            target = archive.joinpath(*safe_relative("vectors/" + name).parts)
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(bundle.read(name))
            archive_files[target.relative_to(archive).as_posix()] = sha256(target)
        for path in code.rglob("*"):
            if path.is_file():
                archive_files[path.relative_to(archive).as_posix()] = sha256(path)
        for name, expected in data_files.items():
            target = destination / "data" / name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(prepared / name, target)
            if sha256(target) != expected:
                raise ValueError("Prepared data changed during import; incomplete destination is retained")
        manifest_copy = archive / "original-prepared-manifest.json"
        shutil.copyfile(prepared / "manifest.json", manifest_copy)
        archive_files[manifest_copy.name] = sha256(manifest_copy)
        export_path = archive / "EXPORT_MANIFEST.json"
        export_path.write_text(json.dumps({"schema_version": 1, "files": archive_files}, indent=2, ensure_ascii=False), encoding="utf-8")
        target_config = deepcopy(config)
        target_config["platform"]["persistent_root"] = str(destination)
        target_config.setdefault("runtime", {})["data_root"] = str(destination / "data")
        target_config["runtime"]["output_root"] = str(destination / "runs")
        (destination / "portable-config.yaml").write_text(yaml.safe_dump(target_config, allow_unicode=True, sort_keys=False), encoding="utf-8")
        receipt = {"schema_version": 1, "kind": "portable_completed_training_import",
                   "created_utc": datetime.now(timezone.utc).isoformat(), "bundle": str(archive),
                   "source_root": str(code), "destination": str(destination), "base_fingerprint": base,
                   "original_device": "recorded_in_each_original_training_origin", "target_device": device,
                   "runtime_validation": "pending_real_model_smoke_on_target",
                   "export_manifest_sha256": sha256(export_path),
                   "original_manifest_sha256": sha256(destination / "data/manifest.json"),
                   "manifest_bytes_unchanged": True, "representation_bytes_unchanged": True,
                   "source_manifest_sha256": object_hash(archived_sources),
                   "target_config_sha256": object_hash(scientific_config(target_config)),
                   "target_full_config_sha256": object_hash(target_config),
                   "reuse_policy": "completed_training_only_reevaluate", "training": [], "statistics": [],
                   "standalone_vector_import": True, "optimizer_resume_supported": False,
                   "scientific_python_source_verified_against_training": all(v["exact_source_match"] for v in compatibility_checks.values()),
                   "completed_representation_compatibility_verified": True,
                   "source_compatibility": compatibility_checks,
                   "original_full_source_manifest_verified": False,
                   "source_evidence_note": "Original data and representation contents are verified. Exact source equivalence or explicitly bounded method compatibility is recorded separately. Target placement and future computations form new evidence.",
                   "semantic_scorer_original_weights_verified": False,
                   "semantic_scorer_note": "Vector-only has no original scorer snapshot hash; target scorer files are recorded for the new evaluation.",
                   "semantic_scorer_files": {p.relative_to(semantic).as_posix(): sha256(p) for p in sorted(semantic.rglob("*")) if p.is_file()},
                   "original_test_exposure": "Relocation and reevaluation do not make previously evaluated samples unseen."}
        # Generate target request signatures without changing archived evidence.
        from task_vector.runner import ExperimentRunner
        runner = ExperimentRunner(target_config, source_root=code)
        if runner.base_fingerprint() != base:
            raise ValueError("Target base changed during import; incomplete destination is retained")
        for item, original, artifacts in audited:
            out = runner.directory(item["scope"], item["route"], item["seed"], item["task"], item.get("layers"))
            out.mkdir(parents=True, exist_ok=False)
            new_artifacts = {}
            for field, info in artifacts.items():
                path = out / info["filename"]
                path.write_bytes(bundle.read(info["member"]))
                if sha256(path) != info["sha256"]:
                    raise ValueError("Vector changed during import; incomplete destination is retained")
                new_artifacts[field] = {"path": str(path), "sha256": info["sha256"],
                                        "representation_digest": info["representation_digest"]}
            request_config = deepcopy(target_config)
            if item.get("layers") is not None:
                request_config["ltv"]["layers"] = item["layers"]
            request = object_hash({"config": scientific_config(request_config), "route": item["route"],
                                   "seed": item["seed"], "scope": item["scope"] if item["route"] == "loda" else "independent",
                                   "task": item["task"], "data": manifest["manifest_content_sha256"],
                                   "base": base, "sources": archived_sources})
            original_record = "vectors/" + item["training_origin"]
            origin = {"original_record": original_record, "original_record_sha256": sha256(archive / original_record),
                      "original_request_signature": original["request_signature"],
                      "export_manifest_sha256": receipt["export_manifest_sha256"],
                      "original_provenance": original["provenance"], "weights_retrained": False,
                      "optimizer_resume_supported": False, "original_runtime_seconds": original.get("runtime_seconds"),
                      "scientific_python_sources_verified": compatibility_checks[original["request_signature"]]["exact_source_match"],
                      "representation_compatibility": compatibility_checks[original["request_signature"]],
                      "full_original_source_manifest_verified": False}
            result = deepcopy(original)
            result.update({field: info["path"] for field, info in new_artifacts.items()})
            result.update(artifacts=new_artifacts, request_signature=request, training_origin="imported_completed", imported_from=origin,
                          best_checkpoint=None, last_checkpoint=None, resume_from=None,
                          best_adapter_path=None, best_checkpoint_resume_supported=False)
            record_path = out / "training.json"
            record_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
            receipt["training"].append({"record": record_path.relative_to(destination).as_posix(),
                                         "record_sha256": sha256(record_path), "request_signature": request,
                                         "imported_from": origin, "artifacts": new_artifacts})
        receipt["receipt_content_sha256"] = object_hash(receipt)
        receipt_path = destination / "IMPORT_RECEIPT.json"
        receipt_path.write_text(json.dumps(receipt, indent=2, ensure_ascii=False), encoding="utf-8")
        return receipt
    finally:
        bundle.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", "--bundle", dest="bundle", required=True, help="Exported vector ZIP or extracted directory")
    parser.add_argument("--config", required=True, help="Experiment protocol YAML with target base/scorer paths and device")
    parser.add_argument("--source-root", default=str(Path(__file__).resolve().parents[1]))
    parser.add_argument("--destination", required=True, help="A NEW experiment directory")
    parser.add_argument("--prepared-data", help="Original complete data directory, including unchanged manifest.json and all JSONLs")
    args = parser.parse_args()
    receipt = attach_vectors(args.bundle, args.config, args.source_root, args.destination, args.prepared_data)
    print(json.dumps({"ATTACH_VECTORS": "PASS", "IMPORTED_COMPLETED_TRAINING": len(receipt["training"]),
                      "WEIGHTS_RETRAINED": False, "OLD_OPTIMIZER_RESUMED": False,
                      "TARGET_RUNTIME_VALIDATED": False, "DESTINATION": receipt["destination"],
                      "NEXT": "python portable/portable_run.py --destination NEW_ROOT inventory; then preflight, smoke and all"}, indent=2))


if __name__ == "__main__":
    main()
