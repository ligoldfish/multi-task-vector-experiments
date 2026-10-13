"""Synthetic CPU fixtures: no model files, accelerator, or downloaded data."""
from __future__ import annotations

import hashlib
import importlib.util
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
import zipfile

import torch


ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("repo_analyze_vectors", ROOT / "tools/analyze_vectors.py")
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)


def json_bytes(value):
    return json.dumps(value, ensure_ascii=False, indent=2).encode()


def tensor_bytes(value):
    stream = io.BytesIO()
    torch.save(value, stream)
    return stream.getvalue()


class BundleFixture:
    def __init__(self):
        self.files = {}
        self.entries = []
        contract_files = {"config.json": "0" * 64, "model.safetensors": "1" * 64}
        self.base = audit.object_hash(contract_files)
        self.files["BASE_MODEL_CONTRACT.json"] = json_bytes({
            "model_id": "synthetic-test-base", "fingerprint": self.base, "files": contract_files,
            "expected_config": {"hidden_size": 3, "num_hidden_layers": 2,
                                "num_attention_heads": 1, "num_key_value_heads": 1, "head_dim": 2}})

    def add(self, task, route, *, scope="mvp", scale=1.0, layer=0, zero=False):
        folder = f"task_vectors/{scope}/{route}/17/{task}"
        signature = audit.object_hash({"route": route, "task": task, "scope": scope})
        provenance = {"base_fingerprint": self.base, "request_signature": signature,
                      "sample_content_sha256": "2" * 64, "encoded_samples_sha256": "3" * 64,
                      "source_manifest_sha256": "4" * 64}
        package = ROOT / "baseline/src/task_vector"
        if package.is_dir():
            provenance["training_sources_sha256"] = audit.object_hash({
                p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(package.glob("*.py"))})
        origin = {"status": "complete", "completion_schema": 2, "request_signature": signature,
                  "provenance": provenance, "provenance_sha256": audit.object_hash(provenance),
                  "completed_epochs": 3, "optimizer_steps": 90, "steps": 45, "best_nll": 0.5,
                  "runtime_seconds": 100.0, "stop_reason": "patience", "artifacts": {},
                  "raw_path": "/historical/path/raw.pt", "post_path": "/historical/path/post.pt",
                  "initial_path": "/historical/path/initial.pt"}
        entry = {"route": route, "scope": scope, "task": task, "seed": 17,
                 "layers": [layer] if route == "ltv" else None, "exports": {},
                 "primary": folder + "/post.pt", "training_origin": folder + "/training-origin.json"}
        for version in ("post", "raw", "initial"):
            metadata = {"version": version if version != "initial" else "raw", "model_id": "synthetic-test-base",
                        "base_fingerprint": self.base, "training_request_signature": signature,
                        "source_scope": "independent" if route == "ltv" else scope}
            payload = {"schema_version": 1, "space": "activation" if route == "ltv" else "parameter",
                       "task": task, "seed": 17, "metadata": metadata, "values": {}, "factors": {}}
            if route == "ltv":
                metadata["layers"] = [layer]
                payload["values"] = {f"model.layers.{layer}": torch.tensor([scale, 2., -1.])}
            else:
                metadata["branches"] = ["general", "isolated"]
                a = torch.tensor([[1., -1., 2.]])
                b = torch.tensor([[2.], [-3.]])
                # Factor rescaling changes factor coordinates but preserves BA.
                payload["factors"] = {f"model.layers.{layer}.self_attn.q_proj": [
                    {"A": a / scale, "B": b * scale},
                    {"A": a.clone(), "B": -b.clone() if zero else b * 0.5}]}
            data = tensor_bytes(payload)
            path = folder + "/" + version + ".pt"
            self.files[path] = data
            artifact = {"path": path, "sha256": hashlib.sha256(data).hexdigest(),
                        "representation_digest": audit.payload_digest(payload)}
            entry["exports"][version + "_path"] = artifact
            origin["artifacts"][version + "_path"] = {**artifact, "path": "/historical/path/" + version + ".pt"}
        self.files[entry["training_origin"]] = json_bytes(origin)
        self.entries.append(entry)
        return entry

    def materialize(self, destination, *, zipped=True):
        self.files["VECTOR_INDEX.json"] = json_bytes({"schema_version": 1, "base_contract": "BASE_MODEL_CONTRACT.json",
                                                     "representations": self.entries, "primary_version": "post"})
        manifest = {"schema_version": 1, "files": {name: hashlib.sha256(data).hexdigest() for name, data in self.files.items()}}
        all_files = {**self.files, "EXPORT_MANIFEST.json": json_bytes(manifest)}
        if zipped:
            with zipfile.ZipFile(destination, "x") as archive:
                for name, data in all_files.items():
                    archive.writestr(name, data)
        else:
            destination.mkdir()
            for name, data in all_files.items():
                path = destination / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(data)
        return destination


class VectorAuditTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)

    def test_zip_and_directory_inspection_preserve_origin(self):
        fixture = BundleFixture()
        fixture.add("math", "ltv")
        fixture.add("logic", "loda")
        for n, zipped in enumerate((True, False)):
            bundle = fixture.materialize(self.root / (f"bundle{n}.zip" if zipped else "extracted"), zipped=zipped)
            result = audit.analyze(bundle, self.root / f"out{n}", geometry=False)
            self.assertEqual(result["representation_count"], 2)
            self.assertEqual(result["bundle_integrity"]["status"], "PASS")
            origin = result["representations"][0]["training_origin"]
            self.assertEqual(origin["completed_epochs"], 3)
            self.assertEqual(origin["optimizer_steps"], 90)
            self.assertEqual(origin["raw_path"], "/historical/path/raw.pt")
            self.assertEqual(result["representations"][0]["provenance_consistency"], "PASS")
            self.assertFalse(result["geometry_enabled"])

    def test_parameter_norm_includes_branch_cross_terms(self):
        fixture = BundleFixture()
        entry = fixture.add("math", "loda", scale=4.)
        payload = torch.load(io.BytesIO(fixture.files[entry["primary"]]), weights_only=True)
        expected = sum(term["B"] @ term["A"] for term in payload["factors"].values().__iter__().__next__())
        report = audit.inspect_payload(payload)
        self.assertAlmostEqual(report["represented_norm"], float(expected.double().norm()), places=10)
        self.assertTrue(next(iter(report["modules"].values()))["cross_terms_included"])
        self.assertFalse(report["factor_flattening"])

    def test_initial_is_explicit_untrained_control(self):
        fixture = BundleFixture()
        fixture.add("math", "ltv")
        bundle = fixture.materialize(self.root / "bundle.zip")
        result = audit.analyze(bundle, self.root / "out", version="initial", geometry=False)
        self.assertTrue(result["representations"][0]["is_untrained_control"])
        self.assertEqual(result["representations"][0]["selected_version"], "initial")

    def test_inspection_digest_does_not_require_numpy(self):
        fixture = BundleFixture()
        entry = fixture.add("math", "ltv")
        payload = torch.load(io.BytesIO(fixture.files[entry["primary"]]), weights_only=True)
        expected = audit.payload_digest(payload)
        with patch.object(torch.Tensor, "numpy", side_effect=RuntimeError("Numpy is not available")):
            self.assertEqual(audit.payload_digest(payload), expected)

    def test_tampering_fails_before_creating_output(self):
        fixture = BundleFixture()
        entry = fixture.add("math", "ltv")
        directory = fixture.materialize(self.root / "bundle", zipped=False)
        (directory / entry["primary"]).write_bytes(b"tamper")
        with self.assertRaisesRegex(ValueError, "SHA256 mismatch"):
            audit.analyze(directory, self.root / "out", geometry=False)
        self.assertFalse((self.root / "out").exists())

    def test_changed_metadata_even_rehashed_bundle_fails_origin(self):
        fixture = BundleFixture()
        entry = fixture.add("math", "ltv")
        payload = torch.load(io.BytesIO(fixture.files[entry["primary"]]), weights_only=True)
        payload["metadata"]["training_request_signature"] = "wrong"
        fixture.files[entry["primary"]] = tensor_bytes(payload)
        entry["exports"]["post_path"]["sha256"] = hashlib.sha256(fixture.files[entry["primary"]]).hexdigest()
        bundle = fixture.materialize(self.root / "bundle.zip")
        with self.assertRaisesRegex(ValueError, "request signature differs"):
            audit.analyze(bundle, self.root / "out", geometry=False)

    def test_origin_provenance_hash_is_checked(self):
        fixture = BundleFixture()
        entry = fixture.add("math", "ltv")
        origin = json.loads(fixture.files[entry["training_origin"]])
        origin["provenance"]["encoded_samples_sha256"] = "bad"
        fixture.files[entry["training_origin"]] = json_bytes(origin)
        bundle = fixture.materialize(self.root / "bundle.zip")
        with self.assertRaisesRegex(ValueError, "provenance content hash"):
            audit.analyze(bundle, self.root / "out", geometry=False)

    def test_committed_representation_digest_is_checked(self):
        fixture = BundleFixture()
        entry = fixture.add("math", "ltv")
        entry["exports"]["post_path"]["representation_digest"] = "0" * 64
        origin = json.loads(fixture.files[entry["training_origin"]])
        origin["artifacts"]["post_path"]["representation_digest"] = "0" * 64
        fixture.files[entry["training_origin"]] = json_bytes(origin)
        bundle = fixture.materialize(self.root / "bundle.zip")
        with self.assertRaisesRegex(ValueError, "Actual representation content digest"):
            audit.analyze(bundle, self.root / "out", geometry=False)

    def test_nonfinite_and_wrong_dtype_are_rejected(self):
        fixture = BundleFixture()
        entry = fixture.add("math", "ltv")
        payload = torch.load(io.BytesIO(fixture.files[entry["primary"]]), weights_only=True)
        payload["values"]["model.layers.0"][0] = float("nan")
        with self.assertRaisesRegex(ValueError, "Nonfinite"):
            audit.inspect_payload(payload)
        payload["values"]["model.layers.0"] = torch.ones(3, dtype=torch.float16)
        with self.assertRaisesRegex(ValueError, "FP32 CPU"):
            audit.inspect_payload(payload)

    def test_base_architecture_shapes_are_checked(self):
        fixture = BundleFixture()
        entry = fixture.add("math", "ltv")
        contract = json.loads(fixture.files["BASE_MODEL_CONTRACT.json"])
        contract["expected_config"]["hidden_size"] = 9
        fixture.files["BASE_MODEL_CONTRACT.json"] = json_bytes(contract)
        bundle = fixture.materialize(self.root / "bundle.zip")
        with self.assertRaisesRegex(ValueError, "input width differs"):
            audit.analyze(bundle, self.root / "out", geometry=False)

    def test_output_is_never_overwritten(self):
        fixture = BundleFixture()
        fixture.add("math", "ltv")
        bundle = fixture.materialize(self.root / "bundle.zip")
        output = self.root / "out"
        output.mkdir()
        sentinel = output / "keep.txt"
        sentinel.write_text("keep", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "new directory"):
            audit.analyze(bundle, output, geometry=False)
        self.assertEqual(sentinel.read_text(encoding="utf-8"), "keep")

    def test_unsafe_and_duplicate_zip_members_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "archive-relative"):
            audit.member_name("folder\\escape")
        for n, name in enumerate(("../escape", "/absolute", "C:/absolute")):
            bundle = self.root / f"unsafe{n}.zip"
            with zipfile.ZipFile(bundle, "x") as archive:
                archive.writestr(name, b"x")
            with self.assertRaisesRegex(ValueError, "archive-relative"):
                audit.Bundle(bundle)
        bundle = self.root / "duplicate.zip"
        with zipfile.ZipFile(bundle, "x") as archive:
            archive.writestr("duplicate", b"a")
            with self.assertWarns(UserWarning):
                archive.writestr("duplicate", b"b")
        with self.assertRaisesRegex(ValueError, "duplicate ZIP"):
            audit.Bundle(bundle)

    def test_extra_directory_files_are_not_silently_ignored(self):
        fixture = BundleFixture()
        fixture.add("math", "ltv")
        bundle = fixture.materialize(self.root / "bundle", zipped=False)
        (bundle / "extra.txt").write_text("unexpected", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "exactly the bundle"):
            audit.analyze(bundle, self.root / "out", geometry=False)

    @unittest.skipUnless((ROOT / "baseline/src/task_vector/analysis.py").is_file(), "Frozen baseline not yet staged")
    def test_geometry_stays_within_route_scope_and_layer_and_uses_actual_BA(self):
        fixture = BundleFixture()
        fixture.add("math", "ltv")
        fixture.add("logic", "ltv", scale=2.)
        fixture.add("math", "ltv", scope="expanded", layer=1)
        fixture.add("math", "loda", scale=1.)
        fixture.add("logic", "loda", scale=4.)
        fixture.add("math", "loda", scope="loo-translation", zero=True)
        bundle = fixture.materialize(self.root / "bundle.zip")
        representation, _ = audit.geometry_functions(ROOT / "baseline")
        with patch.object(representation.LowRankFactor, "dense", side_effect=AssertionError("Dense expansion is forbidden")):
            result = audit.analyze(bundle, self.root / "out", save_decomposition=True)
        self.assertEqual(len(result["geometry"]), 4)
        loda = next(g for g in result["geometry"] if g["group"]["route"] == "loda" and g["group"]["scope"] == "mvp")
        self.assertAlmostEqual(loda["represented_cosines"][0][1], 1., places=12)
        self.assertEqual(loda["represented_gram"][0][0], loda["represented_gram"][1][1])
        self.assertEqual(set(loda["decompositions"]), {"mean", "svd-q0", "svd-q1", "svd-q2"})
        self.assertEqual(loda["decompositions"]["svd-q0"]["shared_norms"], [0., 0.])
        self.assertEqual(loda["decompositions"]["svd-q1"]["metadata"]["used_rank"], 1)
        for group in result["geometry"]:
            for decomposition in group["decompositions"].values():
                if decomposition["status"] == "COMPLETE":
                    self.assertLess(max(decomposition["metadata"]["reconstruction_relative_errors"]), 1e-6)
                    for saved in decomposition["saved"]:
                        payload = torch.load(self.root / "out" / saved["path"], weights_only=True)
                        self.assertTrue(payload["metadata"]["analysis_only"])
        zero = next(g for g in result["geometry"] if g["group"]["scope"] == "loo-translation")
        self.assertIsNone(json.loads((self.root / "out/analysis.json").read_text(encoding="utf-8"))["geometry"][-1]["represented_cosines"][0][0])
        self.assertTrue(math_isnan(zero["represented_cosines"][0][0]))


def math_isnan(value):
    return value != value


if __name__ == "__main__":
    unittest.main()
