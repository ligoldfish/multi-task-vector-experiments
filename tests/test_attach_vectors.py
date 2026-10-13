"""Tiny synthetic CPU integration checks; these are never scientific results."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import shutil
import sys
import tempfile
import unittest
import zipfile

import torch
import yaml

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "tools"))
sys.path.insert(0, str(REPO / "portable"))
sys.path.insert(0, str(REPO / "baseline/src"))
import attach_vectors as attach
import portable_run
from task_vector.config import source_manifest
from task_vector.data import load_samples
from task_vector.representations import Representation, LowRankFactor
from task_vector.runner import ExperimentRunner
from task_vector.training import _content_digest, _sample_content


def dump(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def seal(bundle):
    dump(bundle / "EXPORT_MANIFEST.json", {"schema_version": 1, "files": {
        p.relative_to(bundle).as_posix(): attach.sha256(p) for p in sorted(bundle.rglob("*"))
        if p.is_file() and p.name != "EXPORT_MANIFEST.json"}})


class AttachIntegration(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.work = Path(tempfile.mkdtemp(prefix="task-vector-attach-tests-"))
        cls.source, cls.data, cls.model, cls.scorer, cls.bundle = (cls.work / n for n in ("baseline", "data", "model", "scorer", "vectors"))
        for name in ("src", "configs", "docs"):
            shutil.copytree(REPO / "baseline" / name, cls.source / name,
                            ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
        shutil.copyfile(REPO / "baseline/pyproject.toml", cls.source / "pyproject.toml")
        cls.model.mkdir()
        cls.scorer.mkdir()
        dump(cls.model / "config.json", {"fixture": "SYNTHETIC_NOT_MODEL_WEIGHTS"})
        (cls.model / "model.safetensors").write_bytes(b"synthetic model fixture")
        dump(cls.scorer / "config.json", {"fixture": "SYNTHETIC_NOT_SCORER"})
        cls.config = yaml.safe_load((cls.source / "configs/experiment.yaml").read_text(encoding="utf-8"))
        cls.config["platform"].update(device="cpu", persistent_root=str(cls.work / "old-runtime"))
        cls.config["model"]["path"] = str(cls.model)
        cls.config["metrics"]["semantic_model_path"] = str(cls.scorer)
        cls.config_path = cls.work / "target-config.yaml"
        cls.config_path.write_text(yaml.safe_dump(cls.config, allow_unicode=True), encoding="utf-8")
        manifest = {"status": "complete", "tasks": {}, "synthetic_test_only": True}
        for task in sorted(attach.TASKS):
            entries = {}
            for split in sorted(attach.SPLITS):
                row = {"id": f"{task}:{split}:1", "task": task, "prompt": "Synthetic prompt", "target": "1",
                       "group_id": f"{task}:{split}:1", "official_split": "train" if split in {"train", "tune", "discovery"} else "test"}
                path = cls.data / task / f"{split}.jsonl"
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(json.dumps(row) + "\n", encoding="utf-8")
                entries[split] = {"path": str(path), "sha256": attach.sha256(path), "count": 1,
                                  "ids_sha256": attach.data_hash([row["id"]])}
            manifest["tasks"][task] = {"files": entries}
        manifest["manifest_content_sha256"] = attach.data_hash(manifest)
        dump(cls.data / "manifest.json", manifest)
        cls.manifest = manifest
        cls.bundle.mkdir()
        model_files = {p.name: attach.sha256(p) for p in cls.model.iterdir()}
        cls.base = attach.object_hash(model_files)
        contract = {"files": model_files, "fingerprint": cls.base, "model_id": "Qwen/Qwen3-1.7B",
                    "expected_config": cls.config["model"]["expected_config"], "enable_thinking": False, "intervention_scale": 1.0}
        dump(cls.bundle / "BASE_MODEL_CONTRACT.json", contract)
        package = cls.source / "src/task_vector"
        package_hash = attach.object_hash({p.name: attach.sha256(p) for p in sorted(package.glob("*.py"))})
        index = {"schema_version": 1, "representations": []}
        for route, family in (("ltv", "independent"), ("loda", "mvp")):
            for task in ("math", "logic", "translation", "paraphrase"):
                request = attach.object_hash({"SYNTHETIC_ORIGINAL_REQUEST": route + task})
                prefix = f"task_vectors/{family}/{route}/17/{task}/"
                metadata = {"base_fingerprint": cls.base, "training_request_signature": request,
                            "model_id": "Qwen/Qwen3-1.7B", "layers": [14], "source_scope": family,
                            "source_ids": [f"{task}:train:1"], "heldout_excluded": None,
                            "subspaces": {"model.layers.0": {"task": task, "seen_tasks": ["math", "logic", "translation", "paraphrase"]}}}
                exports, original_artifacts, paths = {}, {}, {}
                for field in ("raw_path", "post_path", "initial_path"):
                    name = prefix + field.removesuffix("_path") + ".pt"
                    path = cls.bundle / name
                    path.parent.mkdir(parents=True, exist_ok=True)
                    rep = (Representation("activation", task, 17, values={"model.layers.14": torch.tensor([1., 2.])}, metadata=metadata)
                           if route == "ltv" else Representation("parameter", task, 17,
                               factors={"model.layers.0.self_attn.q_proj": [LowRankFactor(torch.ones(2, 1), torch.ones(1, 2))]}, metadata=metadata))
                    rep.save(path)
                    digest = ExperimentRunner.representation_digest(rep)
                    exports[field] = {"path": name, "sha256": attach.sha256(path), "representation_digest": digest}
                    original_artifacts[field] = {**exports[field], "path": "/old/location/" + path.name}
                    paths[field] = original_artifacts[field]["path"]
                samples = {split: load_samples(cls.data, task, split) for split in ("train", "tune")}
                provenance = {"base_fingerprint": cls.base, "data_manifest_sha256": attach.object_hash(manifest),
                              "training_sources_sha256": package_hash, "source_manifest_sha256": "original broader ops/config digest unavailable",
                              "sample_content_sha256": _content_digest({split: _sample_content(rows) for split, rows in samples.items()}),
                              "request_signature": request}
                record = {"status": "complete", "completion_schema": 2, "request_signature": request,
                          "provenance": provenance, "provenance_sha256": attach.object_hash(provenance),
                          "artifacts": original_artifacts, "runtime_seconds": 1.0, "completed_epochs": 5, **paths}
                origin = prefix + "training-origin.json"
                dump(cls.bundle / origin, record)
                index["representations"].append({"route": route, "scope": "mvp", "task": task, "seed": 17,
                                                 "layers": None, "training_origin": origin, "exports": exports})
        dump(cls.bundle / "VECTOR_INDEX.json", index)
        seal(cls.bundle)
        cls.index = index

    def destination(self, label):
        return self.work / (label + "-" + next(tempfile._get_candidate_names()))

    def test_directory_import_reuses_all_eight_without_model_loading_or_training(self):
        dest = self.destination("reuse")
        receipt = attach.attach_vectors(self.bundle, self.config_path, self.source, dest, self.data)
        self.assertEqual(len(receipt["training"]), 8)
        self.assertTrue(receipt["scientific_python_source_verified_against_training"])
        self.assertFalse(receipt["original_full_source_manifest_verified"])
        config, loaded_receipt, code = portable_run.load_portable_context(dest)
        runner = portable_run.portable_runner(config, loaded_receipt, code)
        runner.load_model = lambda: self.fail("Imported completed adapters must not be retrained")
        for route in ("ltv", "loda"):
            for task in ("math", "logic", "translation", "paraphrase"):
                record = runner.train("mvp", route, 17, task)
                self.assertEqual(record["training_origin"], "imported_completed")
                self.assertFalse(record["best_checkpoint_resume_supported"])
                self.assertIsNone(record["last_checkpoint"])
        self.assertEqual(attach.sha256(dest / "data/manifest.json"), attach.sha256(self.data / "manifest.json"))
        self.assertFalse((dest / "runs/smoke.json").exists())
        self.assertFalse((dest / "runs/evaluation").exists())
        self.assertFalse((dest / "runs/statistics").exists())

    def test_zip_import_and_data_relocation_with_manifest_bytes_preserved(self):
        archive = self.work / "synthetic-vectors.zip"
        with zipfile.ZipFile(archive, "w") as handle:
            for p in self.bundle.rglob("*"):
                if p.is_file():
                    handle.write(p, p.relative_to(self.bundle).as_posix())
        relocated = self.work / "data-relocated"
        shutil.copytree(self.data, relocated)
        receipt = attach.attach_vectors(archive, self.config_path, self.source, self.destination("zip"), relocated)
        self.assertEqual(len(receipt["training"]), 8)

    def test_regenerated_manifest_cannot_claim_original_split_identity(self):
        changed = self.work / "regenerated-data"
        shutil.copytree(self.data, changed)
        manifest = deepcopy(self.manifest)
        manifest["tasks"]["math"]["files"]["train"]["path"] = "/different/materialization/path"
        manifest.pop("manifest_content_sha256")
        manifest["manifest_content_sha256"] = attach.data_hash(manifest)
        dump(changed / "manifest.json", manifest)
        with self.assertRaisesRegex(ValueError, "manifest provenance"):
            attach.attach_vectors(self.bundle, self.config_path, self.source, self.destination("bad-data"), changed)

    def test_missing_original_data_rejected_before_destination_creation(self):
        dest = self.destination("no-data")
        with self.assertRaises(FileNotFoundError):
            attach.attach_vectors(self.bundle, self.config_path, self.source, dest, self.work / "missing-data")
        self.assertFalse(dest.exists())

    def test_changed_scientific_source_rejected(self):
        changed = self.work / "changed-baseline"
        shutil.copytree(self.source, changed)
        with (changed / "src/task_vector/training.py").open("a", encoding="utf-8") as handle:
            handle.write("\n# altered scientific source\n")
        with self.assertRaisesRegex(ValueError, "Scientific Python source"):
            attach.attach_vectors(self.bundle, self.config_path, changed, self.destination("bad-source"), self.data)

    def test_changed_tensor_rejected_even_if_export_manifest_is_resealed(self):
        changed = self.work / "changed-vector"
        shutil.copytree(self.bundle, changed)
        item = self.index["representations"][0]
        path = changed / item["exports"]["post_path"]["path"]
        rep = Representation.load(path)
        rep.values[rep.keys[0]][0] += 1
        rep.save(path)
        seal(changed)
        with self.assertRaisesRegex(ValueError, "tensor/identity/provenance"):
            attach.attach_vectors(changed, self.config_path, self.source, self.destination("bad-vector"), self.data)

    def test_loda_scope_cannot_be_relabeled_as_a_heldout_experiment(self):
        changed = self.work / "changed-scope"
        shutil.copytree(self.bundle, changed)
        index = deepcopy(self.index)
        for item in index["representations"]:
            if item["route"] == "loda" and item["task"] == "logic":
                item["scope"] = "loo-math"
                break
        dump(changed / "VECTOR_INDEX.json", index)
        seal(changed)
        with self.assertRaisesRegex(ValueError, "tensor/identity/provenance"):
            attach.attach_vectors(changed, self.config_path, self.source, self.destination("bad-scope"), self.data)

    def test_ltv_layer14_cannot_be_relabeled_as_layer6(self):
        changed = self.work / "changed-layers"
        shutil.copytree(self.bundle, changed)
        index = deepcopy(self.index)
        index["representations"][0]["layers"] = [6]
        dump(changed / "VECTOR_INDEX.json", index)
        seal(changed)
        with self.assertRaisesRegex(ValueError, "tensor/identity/provenance"):
            attach.attach_vectors(changed, self.config_path, self.source, self.destination("bad-layer"), self.data)

    def test_actual_exporter_zip_can_be_attached_and_reused_without_retraining(self):
        import export_portable as export
        from task_vector.subspaces import Statistics, save_statistics
        root = self.work / "actual-exporter-old-root"
        code = self.work / "actual-exporter-old-code"
        shutil.copytree(self.source, code)
        shutil.copytree(self.data, root / "data")
        config = deepcopy(self.config)
        config["platform"]["persistent_root"] = str(root)
        config["runtime"] = {"data_root": str(root / "data"), "output_root": str(root / "runs")}
        config_path = code / "configs/modelmate.local.yaml"
        config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
        runner = ExperimentRunner(config, source_root=code)
        fingerprint = runner.base_fingerprint()
        sources = export.source_manifest(code)
        for item in self.index["representations"]:
            folder = runner.directory(item["scope"], item["route"], item["seed"], item["task"])
            folder.mkdir(parents=True, exist_ok=True)
            request = export.request_hash(config, sources, self.manifest, fingerprint, item)
            original = json.loads((self.bundle / item["training_origin"]).read_text(encoding="utf-8"))
            record = deepcopy(original)
            record["request_signature"] = request
            record["provenance"].update(request_signature=request, source_manifest_sha256=export.object_hash(sources))
            record["provenance_sha256"] = export.object_hash(record["provenance"])
            record["artifacts"] = {}
            for field in ("raw_path", "post_path", "initial_path"):
                rep = Representation.load(self.bundle / item["exports"][field]["path"])
                rep.metadata["training_request_signature"] = request
                target = folder / (field.removesuffix("_path") + ".pt")
                rep.save(target)
                record[field] = str(target)
                record["artifacts"][field] = {"path": str(target), "sha256": export.file_hash(target),
                                              "representation_digest": runner.representation_digest(rep)}
            dump(folder / "training.json", record)
        for task in ("math", "logic", "translation", "paraphrase"):
            save_statistics(Statistics({"model.layers.0": torch.eye(2)}, metadata={"SYNTHETIC_TEST_ONLY": True}),
                            root / "runs/statistics/frozen_base" / f"{task}.pt")
        downloads = self.work / "actual-exporter-downloads"
        downloads.mkdir()
        export.main(["--code", str(code), "--experiment", str(root), "--config", str(config_path), "--output", str(downloads)])
        vector_zip = next(downloads.glob("task-vectors-only-*.zip"))
        destination = self.destination("actual-export-attached")
        receipt = attach.attach_vectors(vector_zip, self.config_path, self.source, destination, root / "data")
        target_config, receipt, target_code = portable_run.load_portable_context(destination)
        target_runner = portable_run.portable_runner(target_config, receipt, target_code)
        target_runner.load_model = lambda: self.fail("Exporter-produced completed training should be reused")
        self.assertEqual(len(receipt["training"]), 8)
        for route in ("ltv", "loda"):
            for task in ("math", "logic", "translation", "paraphrase"):
                result = target_runner.train("mvp", route, 17, task)
                self.assertEqual(result["training_origin"], "imported_completed")

    def test_changed_config_and_wrong_base_rejected(self):
        changed = deepcopy(self.config)
        changed["training"]["learning_rate"] = .02
        changed["supplement_label"] = "attempted-protocol-change"
        path = self.work / "changed-config.yaml"
        path.write_text(yaml.safe_dump(changed), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "scientific configuration"):
            attach.attach_vectors(self.bundle, path, self.source, self.destination("bad-config"), self.data)
        wrong = self.work / "wrong-model"
        shutil.copytree(self.model, wrong)
        (wrong / "model.safetensors").write_bytes(b"changed model weights")
        changed = deepcopy(self.config)
        changed["model"]["path"] = str(wrong)
        path = self.work / "wrong-base.yaml"
        path.write_text(yaml.safe_dump(changed), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "base/tokenizer/config"):
            attach.attach_vectors(self.bundle, path, self.source, self.destination("bad-base"), self.data)

    def test_existing_destination_not_overwritten(self):
        dest = self.destination("existing")
        dest.mkdir()
        marker = dest / "keep.txt"
        marker.write_text("retain", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "already exists"):
            attach.attach_vectors(self.bundle, self.config_path, self.source, dest, self.data)
        self.assertEqual(marker.read_text(), "retain")


if __name__ == "__main__":
    unittest.main(verbosity=2)
