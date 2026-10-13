"""CPU integration tests with deliberately TINY synthetic files, never real results."""
from copy import deepcopy
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
import zipfile

import torch
import yaml

import export_portable as ex
import portable_import as imp
import portable_run as run

ORIGINAL = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ORIGINAL / "src"))
from task_vector.representations import Representation, LowRankFactor
from task_vector.runner import ExperimentRunner
from task_vector.subspaces import Statistics, save_statistics, load_statistics


def dump(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


class PortableIntegration(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Keep test artifacts; no deletion of user files/environments.
        cls.work = Path(tempfile.mkdtemp(prefix="portable-handoff-test-"))
        cls.code, cls.root, cls.base, cls.semantic = (cls.work / name for name in ("source", "old-experiment", "base", "semantic"))
        for path in (cls.code, cls.root, cls.base, cls.semantic):
            path.mkdir()
        for name in ("src", "configs", "deploy", "third_party", "ops", "docs"):
            shutil.copytree(ORIGINAL / name, cls.code / name,
                            ignore=shutil.ignore_patterns("__pycache__", ".git", "*.pyc"))
        shutil.copy2(ORIGINAL / "pyproject.toml", cls.code / "pyproject.toml")
        for path in ORIGINAL.glob("run_modelmate*.py"):
            shutil.copy2(path, cls.code / path.name)
        dump(cls.base / "config.json", {"model_type": "qwen3", "synthetic_test_only": True})
        (cls.base / "model.safetensors").write_bytes(b"synthetic-not-a-real-model")
        dump(cls.semantic / "config.json", {"synthetic_test_only": True})
        (cls.semantic / "model.safetensors").write_bytes(b"synthetic-not-a-real-semantic-model")
        config = yaml.safe_load((cls.code / "configs/experiment.yaml").read_text(encoding="utf-8"))
        config["platform"].update(device="cpu", persistent_root=str(cls.root))
        config["acceleration"]["enabled"] = False
        config["model"]["path"] = str(cls.base)
        config["metrics"]["semantic_model_path"] = str(cls.semantic)
        config["runtime"] = {"data_root": str(cls.root / "data"), "output_root": str(cls.root / "runs")}
        (cls.code / "configs/modelmate.local.yaml").write_text(yaml.safe_dump(config, allow_unicode=True), encoding="utf-8")
        cls.config = config
        manifest = {"status": "complete", "tasks": {}, "synthetic_test_only": True}
        for task in ex.TASKS:
            entries = {}
            for split in ("train", "tune", "discovery", "test", "test_formal", "test_confirmation"):
                rows = [{"id": f"{task}:{split}:{i}", "task": task, "prompt": "Synthetic prompt", "target": "1",
                         "group_id": f"{task}:{split}:{i}", "official_split": "train" if split in {"train", "tune", "discovery"} else "test"}
                        for i in range(2)]
                path = cls.root / "data" / task / f"{split}.jsonl"
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
                entries[split] = {"path": str(path), "count": len(rows), "sha256": ex.file_hash(path),
                                  "ids_sha256": ex.data_hash([r["id"] for r in rows])}
            manifest["tasks"][task] = {"files": entries}
        manifest["manifest_content_sha256"] = ex.data_hash(manifest)
        dump(cls.root / "data/manifest.json", manifest)
        cls.manifest = manifest
        runner = ExperimentRunner(config, source_root=cls.code)
        fingerprint = runner.base_fingerprint()
        sources = ex.source_manifest(cls.code)
        for route, family in (("ltv", "independent"), ("loda", "mvp")):
            for task in ex.TASKS[:4]:
                folder = cls.root / "runs/training" / family / route / "17" / task
                folder.mkdir(parents=True)
                item = {"route": route, "scope": "mvp", "seed": 17, "task": task, "layers": None}
                request = ex.request_hash(config, sources, manifest, fingerprint, item)
                metadata = {"base_fingerprint": fingerprint, "training_request_signature": request,
                            "model_id": "Qwen/Qwen3-1.7B", "layers": [14], "hook": "decoder_block_input_before_rmsnorm"}
                artifacts, paths = {}, {}
                for field in ("raw_path", "post_path", "initial_path"):
                    path = folder / (field.removesuffix("_path") + ".pt")
                    rep = (Representation("activation", task, 17, values={"model.layers.14": torch.tensor([1., 2.])}, metadata=metadata)
                           if route == "ltv" else Representation("parameter", task, 17,
                               factors={"model.layers.0.self_attn.q_proj": [LowRankFactor(torch.ones(2, 1), torch.ones(1, 2))]}, metadata=metadata))
                    rep.save(path)
                    artifacts[field] = {"path": str(path), "sha256": ex.file_hash(path), "representation_digest": runner.representation_digest(rep)}
                    paths[field] = str(path)
                record = {"completion_schema": 2, "status": "complete", "request_signature": request,
                          "artifacts": artifacts, "provenance": {"base_fingerprint": fingerprint,
                              "source_manifest_sha256": ex.object_hash(sources), "data_manifest_sha256": ex.object_hash(manifest), "request_signature": request},
                          "best_nll": .5, "completed_epochs": 3, "stop_reason": "patience", "runtime_seconds": 1, **paths}
                dump(folder / "training.json", record)
                (folder / "last.pt").write_bytes(b"optimizer-must-not-be-exported")
                if route == "loda":
                    torch.save({"synthetic_subspace": torch.eye(2)}, folder / "subspaces.pt")
        for task in ex.TASKS[:4]:
            ids = [f"{task}:train:{i}" for i in range(2)]
            signature = imp.statistics_signature(config, sources, manifest, fingerprint, ids)
            path = cls.root / f"runs/statistics/frozen_base/{task}.pt"
            save_statistics(Statistics({"model.layers.0": torch.eye(2)}, metadata={"runner_signature": signature, "sample_ids": ids}), path)
        dump(cls.root / "runs/smoke.json", {"synthetic_test_only": True})
        dump(cls.root / "runs/protocol/mvp.core-seed17.lock.json", {"synthetic_test_only": True})
        cls.full, cls.vectors, cls.index = ex.collect(cls.code, cls.root, cls.code / "configs/modelmate.local.yaml", Path(__file__).parent)
        cls.bundle = cls.work / "bundle"
        archive = cls.work / "synthetic-full.zip"
        ex.write_zip(archive, cls.full, {"PORTABLE_INDEX.json": cls.index})
        with zipfile.ZipFile(archive) as handle:
            handle.extractall(cls.bundle)
        cls.destination = cls.work / "imported"
        imp.import_bundle(cls.bundle, cls.destination, cls.base, "cpu")
        cls.target_config, cls.receipt, cls.target_code = run.load_portable_context(cls.destination)

    def test_independent_exports_are_byte_identical_and_optimizer_excluded(self):
        self.assertEqual(len(self.index["training"]), 8)
        self.assertEqual(sum(n.endswith("post.pt") for n in self.vectors), 8)
        self.assertFalse(any(n.endswith("last.pt") or n.endswith("best.pt") for n in self.full))
        for field in ("raw_path", "post_path", "initial_path"):
            for item in self.index["training"]:
                self.assertEqual(ex.file_hash(self.full[item["directory"] + "/" + field.removesuffix("_path") + ".pt"][0]), item["standalone_exports"][field]["sha256"])

    def test_reuse_completed_training_without_loading_model(self):
        runner = run.portable_runner(self.target_config, self.receipt, self.target_code)
        runner.load_model = lambda: self.fail("Completed training must not load/train a model")
        for task in ex.TASKS[:4]:
            for route in ("ltv", "loda"):
                result = runner.train("mvp", route, 17, task)
                self.assertEqual(result["training_origin"], "imported_completed")
                self.assertFalse(result["best_checkpoint_resume_supported"])
                original = self.root / "runs/training" / ("independent" if route == "ltv" else "mvp") / route / "17" / task / "post.pt"
                self.assertEqual(ex.file_hash(Path(result["post_path"])), ex.file_hash(original))

    def test_original_data_and_no_valid_old_locks_or_smoke(self):
        self.assertEqual(ex.file_hash(self.destination / "data/manifest.json"), ex.file_hash(self.root / "data/manifest.json"))
        self.assertFalse((self.destination / "runs/smoke.json").exists())
        self.assertFalse((self.destination / "runs/protocol").exists())
        self.assertFalse((self.destination / "runs/evaluation").exists())

    def test_statistics_preserve_exact_tensors_and_record_import(self):
        for entry in self.receipt["statistics"]:
            actual = load_statistics(self.destination / entry["path"])
            original = load_statistics(self.root / "runs/statistics/frozen_base" / Path(entry["path"]).name)
            self.assertEqual(imp.tensor_mapping_digest(actual), imp.tensor_mapping_digest(original))
            # Statistics depend on content, not placement. The original byte-
            # identical inputs and source produce the same legitimate signature.
            self.assertEqual(actual.metadata["runner_signature"], original.metadata["runner_signature"])
            self.assertFalse(actual.metadata["portable_import"]["recalibrated_on_target"])
            self.assertEqual(actual.metadata["portable_import"]["original_sha256"], ex.file_hash(self.root / "runs/statistics/frozen_base" / Path(entry["path"]).name))

    def test_changed_scientific_configuration_is_rejected(self):
        config_path = self.destination / "portable-config.yaml"
        original = config_path.read_bytes()
        try:
            changed = deepcopy(self.target_config)
            changed["training"]["learning_rate"] = .02
            config_path.write_text(yaml.safe_dump(changed), encoding="utf-8")
            with self.assertRaises(ValueError):
                run.load_portable_context(self.destination)
        finally:
            config_path.write_bytes(original)

    def test_changed_vector_is_rejected(self):
        runner = run.portable_runner(self.target_config, self.receipt, self.target_code)
        folder = runner.directory("mvp", "ltv", 17, "math")
        path = folder / "post.pt"
        original = path.read_bytes()
        try:
            path.write_bytes(original + b"changed")
            with self.assertRaises(ValueError):
                runner.train("mvp", "ltv", 17, "math")
        finally:
            path.write_bytes(original)

    def test_wrong_base_and_existing_destination_are_rejected(self):
        wrong = self.work / "wrong-base"
        shutil.copytree(self.base, wrong)
        (wrong / "model.safetensors").write_bytes(b"different-weights")
        with self.assertRaises(ValueError):
            imp.verify_model(wrong, self.index["base_fingerprint"])
        with self.assertRaises(ValueError):
            imp.import_bundle(self.bundle, self.destination, self.base, "cpu")

    def test_changed_bundle_is_rejected(self):
        path = self.bundle / "code/pyproject.toml"
        original = path.read_bytes()
        try:
            path.write_bytes(original + b"\n#changed\n")
            with self.assertRaises(ValueError):
                imp.verify_bundle(self.bundle)
        finally:
            path.write_bytes(original)

    def test_both_download_archives_and_separate_process_inventory(self):
        output = self.work / "downloads"
        output.mkdir()
        ex.main(["--code", str(self.code), "--experiment", str(self.root), "--output", str(output)])
        archives = sorted(output.glob("*.zip"))
        self.assertEqual(len(archives), 2)
        for path in archives:
            self.assertTrue(path.with_name(path.name + ".sha256.txt").read_text().startswith(ex.file_hash(path)))
            with zipfile.ZipFile(path) as archive:
                manifest = json.loads(archive.read("EXPORT_MANIFEST.json"))
                for member, expected in manifest["files"].items():
                    self.assertEqual(ex.hashlib.sha256(archive.read(member)).hexdigest(), expected)
                if path.name.startswith("task-vectors-only"):
                    self.assertEqual(sum(n.endswith("post.pt") for n in archive.namelist()), 8)
                    self.assertNotIn("code/pyproject.toml", archive.namelist())
                    self.assertIn("use_task_vector.py", archive.namelist())
                    self.assertFalse(any(n.startswith("experiment/data/") for n in archive.namelist()))
        result = subprocess.run([sys.executable, "-B", str(self.bundle / "portable_run.py"),
                                 "--destination", str(self.destination), "inventory"],
                                text=True, capture_output=True, timeout=90)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn('"completed_imports": 8', result.stdout)


if __name__ == "__main__":
    unittest.main(verbosity=2)
