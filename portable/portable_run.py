#!/usr/bin/env python3
"""Run the unchanged scientific workflow with audited completed-training imports.

The ordinary task_vector CLI intentionally does not accept imported requests;
this adapter distinguishes original training evidence from target placement.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import json
from pathlib import Path
import sys

from portable_import import read_json, sha256, object_hash, inside, activate_code


def load_portable_context(destination):
    destination = Path(destination).resolve()
    receipt = read_json(destination / "IMPORT_RECEIPT.json")
    unhashed = {k: v for k, v in receipt.items() if k != "receipt_content_sha256"}
    if (receipt.get("kind") != "portable_completed_training_import" or
            object_hash(unhashed) != receipt.get("receipt_content_sha256") or
            receipt.get("destination") != str(destination)):
        raise ValueError("Import receipt changed or destination was moved; import anew from the original bundle")
    bundle = Path(receipt["bundle"])
    if sha256(bundle / "EXPORT_MANIFEST.json") != receipt["export_manifest_sha256"]:
        raise ValueError("Archived export manifest changed")
    code = activate_code(bundle)
    import yaml
    from task_vector.config import source_manifest, scientific_config, validate_config
    config = yaml.safe_load((destination / "portable-config.yaml").read_text(encoding="utf-8"))
    validate_config(config)
    if (object_hash(scientific_config(config)) != receipt["target_config_sha256"] or
            object_hash(config) != receipt["target_full_config_sha256"] or
            object_hash(source_manifest(code)) != receipt["source_manifest_sha256"]):
        raise ValueError("Imported scientific configuration or frozen source changed")
    if sha256(destination / "data/manifest.json") != receipt["original_manifest_sha256"]:
        raise ValueError("Imported prepared data manifest changed")
    return config, receipt, code


def portable_runner(config, receipt, code):
    from task_vector.runner import ExperimentRunner
    from task_vector.representations import Representation

    class PortableRunner(ExperimentRunner):
        def __init__(self):
            super().__init__(config, source_root=code)
            self.import_receipt = deepcopy(receipt)
            self.imported_records = {entry["request_signature"]: entry for entry in receipt["training"]}

        def _verify_training_exports(self, result, request_signature, base):
            if result.get("training_origin") != "imported_completed":
                return super()._verify_training_exports(result, request_signature, base)
            expected = self.imported_records.get(request_signature)
            if (expected is None or result.get("completion_schema") != 2 or result.get("status") != "complete" or
                    result.get("request_signature") != request_signature or
                    result.get("imported_from") != expected["imported_from"] or
                    result.get("artifacts") != expected["artifacts"] or base != receipt["base_fingerprint"]):
                raise ValueError("Completed import does not match its audited target request/receipt")
            record = inside(self.root, expected["record"])
            if self._artifact_hash(record) != expected["record_sha256"]:
                raise ValueError(f"Imported completion record changed: {record}")
            origin = result["imported_from"]
            original_record = inside(Path(receipt["bundle"]), origin["original_record"])
            if self._artifact_hash(original_record) != origin["original_record_sha256"]:
                raise ValueError("Archived original completion record changed")
            for field in ("raw_path", "post_path", "initial_path"):
                path = Path(result[field])
                info = expected["artifacts"][field]
                if (not path.is_file() or not path.resolve().is_relative_to(self.output_root.resolve()) or
                        self._artifact_hash(path) != info["sha256"]):
                    raise ValueError(f"Imported representation bytes changed: {path}")
                rep = Representation.load(path)
                if (rep.metadata.get("base_fingerprint") != base or
                        rep.metadata.get("training_request_signature") != origin["original_request_signature"] or
                        self.representation_digest(rep) != info["representation_digest"]):
                    raise ValueError(f"Imported representation provenance/tensors changed: {path}")

        def statistics(self, scope):
            for entry in self.import_receipt.get("statistics", []):
                path = inside(self.root, entry["path"])
                if self._artifact_hash(path) != entry["sha256"]:
                    raise ValueError(f"Imported frozen-base calibration changed: {path}")
            return super().statistics(scope)

    return PortableRunner()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--destination", required=True, help="Created by portable_import.py")
    parser.add_argument("command", choices=("inventory", "preflight", "smoke", "core", "all", "discover", "lock",
                                             "final", "formal", "heldout", "ablations", "extended", "evaluate"))
    parser.add_argument("--scope", default="mvp")
    parser.add_argument("--split", default="test", choices=("train", "tune", "discovery", "test", "test_formal", "test_confirmation"))
    parser.add_argument("--route", choices=("ltv", "loda"), default="ltv")
    parser.add_argument("--task", default="math")
    parser.add_argument("--target", help="Evaluation target task; defaults to --task")
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--version", choices=("raw", "post", "initial"), default="post")
    args = parser.parse_args()
    config, receipt, code = load_portable_context(args.destination)
    runner = portable_runner(config, receipt, code)
    try:
        if runner.base_fingerprint() != receipt["base_fingerprint"]:
            raise ValueError("Target base differs from the imported contract")
        if args.command == "inventory":
            for entry in receipt["training"]:
                result = read_json(inside(runner.root, entry["record"]))
                runner._verify_training_exports(result, entry["request_signature"], receipt["base_fingerprint"])
            result = {"completed_imports": len(receipt["training"]), "target_device": config["platform"]["device"],
                      "runtime_validation": "Run smoke on this target", "predictions_reused": 0}
        elif args.command == "preflight":
            from task_vector.runtime import preflight
            result = preflight(config, runner.root / "environment" / "portable-target")
        elif args.command == "smoke":
            from task_vector.smoke import run_smoke
            from task_vector.workflow import smoke_evidence
            if (runner.output_root / "smoke.json").exists():
                result = smoke_evidence(runner)
                if not result["valid"]:
                    raise ValueError("Existing target smoke is invalid; use core to archive and repeat it")
            else:
                result = run_smoke(config, task=args.task, output_dir=runner.output_root)
        elif args.command == "evaluate":
            if args.split.startswith("test"):
                raise ValueError("Use core/all/final for locked test evaluation. Single-vector demonstration uses --split tune.")
            folder = runner.directory(args.scope, args.route, args.seed, args.task)
            if not (folder / "training.json").is_file():
                raise ValueError("Requested completed vector is absent; evaluate never starts training")
            runner.train(args.scope, args.route, args.seed, args.task)
            rep = runner.load_representation(folder / f"{args.version}.pt")
            result = runner.evaluate(rep, args.target or args.task, args.split,
                                     label=f"portable-{args.route}-{args.task}-{args.version}",
                                     scope=args.scope, seed=args.seed)
        else:
            from task_vector.workflow import run_phase
            result = run_phase(runner, args.command, args.scope, args.split)
        from task_vector.cli import _compact
        display = result if args.command == "inventory" else _compact(result)
        print(json.dumps({"command": args.command, "result": display}, ensure_ascii=False, indent=2, default=str))
        return 42 if isinstance(result, dict) and any(result.get(key) is False for key in ("completed", "passed", "passes")) else 0
    finally:
        runner.release_model()


if __name__ == "__main__":
    sys.exit(main())
