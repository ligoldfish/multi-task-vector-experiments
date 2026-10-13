"""Recovery contract tests; tensor reconstruction runs only when torch is present.

The stdlib fixtures exercise orchestration and rejection paths without claiming
to emulate NPU arithmetic. The optional tensor cases use the real digest code.
"""
from contextlib import contextmanager, redirect_stdout
from copy import deepcopy
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import sys
import tempfile
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch


PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "ops"))
try:
    SPEC = importlib.util.spec_from_file_location(
        "portable_recovery_tests_subject", PROJECT / "ops" / "recover_portable_checkpoint.py")
    RECOVERY = importlib.util.module_from_spec(SPEC)
    SPEC.loader.exec_module(RECOVERY)
finally:
    sys.path.pop(0)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def sample_content(samples):
    return [{key: getattr(sample, key, None) for key in
             ("id", "task", "prompt", "target", "group_id", "metadata")} for sample in samples]


def module(name, **attributes):
    result = ModuleType(name)
    result.__dict__.update(attributes)
    return result


class Statistics(dict):
    def __init__(self, matrices=(), *, metadata=None):
        super().__init__(matrices)
        self.metadata = dict(metadata or {})


class Values(list):
    def tolist(self):
        return list(self)


class Tokens:
    """Only the metadata operations used by the cache's input checker."""
    def __init__(self, values):
        self.values = values
        self.ndim = 2
        self.shape = (len(values), len(values[0]))

    def cpu(self):
        return self

    def tolist(self):
        return self.values

    def bool(self):
        return self

    def all(self):
        return all(all(row) for row in self.values)


class Tokenizer:
    tag = "original"

    def __call__(self, prompt, **kwargs):
        number = int(prompt[1:])
        return {"input_ids": Tokens([[number, number + 1]]),
                "attention_mask": Tokens([[1, 1]])}


def encode_sample(tokenizer, sample, **kwargs):
    return {"id": sample.id, "prompt": sample.prompt, "target": sample.target,
            "tokenizer": tokenizer.tag}


class Generator:
    def manual_seed(self, seed):
        return self


def fixtures():
    samples = [SimpleNamespace(id=f"math:train:{number}", task="math", official_split="train",
                               prompt=f"p{number}", target=str(number), group_id=str(number),
                               metadata={}) for number in range(16)]
    config = {"training": {"epochs": 3, "resume_from": "never-reuse-old.pt",
                           "max_prompt_tokens": 512, "max_target_tokens": 512},
              "loda": {"statistics_samples": 128, "rank_general": 8, "rank_isolated": 8,
                       "statistics_seed": 17, "max_statistics_tokens": 64},
              "locked_data_counts": {"train": 300, "tune": 37, "discovery": 37}}
    tokenizer = Tokenizer()
    native_config = {"hidden_size": 2, "use_cache": True}
    parameter = SimpleNamespace(device="npu:0", dtype="torch.bfloat16")
    model = SimpleNamespace(config=SimpleNamespace(to_dict=lambda: native_config),
                            parameters=lambda: iter([parameter]))
    stats = {}
    for index, label in enumerate(RECOVERY.HALVES):
        half = samples[index * 8:(index + 1) * 8]
        metadata = {"statistic": "token_second_moment_mean", "centered": False,
                    "statistics_seed": 17, "sample_count": 8,
                    "sample_ids": [sample.id for sample in half],
                    "token_positions": [[0, 1] for _ in half],
                    "prompt_token_hashes": [digest({"input_ids": [[int(sample.target), int(sample.target) + 1]]})
                                            for sample in half],
                    "token_counts": {"model.layers.0": 16}, "source_split": "train",
                    "engineering_only": True, "actual_source_task": "math",
                    "calibration_scope": "two disjoint 8-sample halves of the same source task training pool"}
        stats[label] = Statistics({"model.layers.0": [[1 + index, 0], [0, 2]]}, metadata=metadata)
    spaces = {"model.layers.0": {"A_G": [[1, 0]], "A_I": [[0, 1]], "lambda": [0.75]}}
    encoded = [encode_sample(tokenizer, sample) for sample in samples]
    provenance = {"sample_content_sha256": digest({"train": sample_content(samples),
                                                   "tune": sample_content(samples[:2])}),
                  "native_model_config_sha256": digest(native_config),
                  "encoded_samples_sha256": digest({"train": encoded, "tune": encoded[:2]}),
                  "subspaces_content_sha256": digest(spaces)}
    return SimpleNamespace(samples=samples, config=config, tokenizer=tokenizer, model=model,
                           parameter=parameter, native_config=native_config, stats=stats,
                           spaces=spaces, provenance=provenance)


@contextmanager
def metadata_modules():
    subspaces = module("task_vector.subspaces", Statistics=Statistics,
                       uniform_token_positions=lambda length, maximum: Values(range(min(length, maximum))),
                       collect_statistics=object(), build_subspaces=object())
    package = module("task_vector", subspaces=subspaces)
    package.__path__ = []
    values = {"task_vector": package, "task_vector.subspaces": subspaces,
              "task_vector.io": module("task_vector.io", object_sha256=digest),
              "task_vector.training": module("task_vector.training", _content_digest=digest,
                                             _sample_content=sample_content),
              "task_vector.prompts": module("task_vector.prompts", encode_sample=encode_sample,
                                           render_prompt=lambda tokenizer, sample, **kwargs: sample.prompt),
              "torch": module("torch", Generator=Generator,
                              randperm=lambda count, **kwargs: Values(range(count)))}
    with patch.dict(sys.modules, values):
        yield subspaces


class InputContractTests(unittest.TestCase):
    def setUp(self):
        self.ids = [f"math:train:{number}" for number in range(16)]
        self.expected = {"config_sha256": "config", "data_manifest_sha256": "data",
                         "base_fingerprint": "base", "source_manifest_sha256": "new-source"}
        self.previous = {"passed": False, "status": "failed", "task": "math", "sample_split": "train",
                         "scientific_training": False, "formal_evaluation": False,
                         "sample_ids": self.ids[:2], "device": "npu:0", "error_type": "UnpicklingError",
                         "protocol_version": "same_shape_causal_prefill_v2",
                         **self.expected, "source_manifest_sha256": "old-source"}
        self.state = {"status": "exported_uncommitted", "steps": 1, "optimizer_steps": 2,
                      "completed_epochs": 1, "device": "npu:0", "provenance": {
                          "kind": "engineering_LoDA_GAO_smoke", "base_fingerprint": "base",
                          "source_task": "math", "source_split": "train", "scientific_training": False,
                          "sample_ids": self.ids, "diagnostic_loss_ids": self.ids[:2],
                          "initial_override_sha256": None}}

    def validate(self, state=True):
        RECOVERY.validate_attempt_inputs(self.previous, self.state if state else None,
                                         self.expected, self.ids, "npu:0")

    def test_engineering_config_is_independent_and_preserves_locked_scope(self):
        original = fixtures().config
        before = deepcopy(original)
        changed = RECOVERY.engineering_config(original)
        self.assertEqual(original, before)
        self.assertEqual(changed["locked_data_counts"], original["locked_data_counts"])
        self.assertEqual((changed["training"]["epochs"], changed["training"]["effective_batch_size"],
                          changed["training"]["micro_batch_size"]), (1, 16, 2))
        self.assertNotIn("resume_from", changed["training"])
        changed["loda"]["rank_general"] = 99
        self.assertEqual(original["loda"]["rank_general"], 8)

    def test_declared_source_patch_is_allowed_but_training_inputs_must_match(self):
        self.validate()
        for field in ("config_sha256", "data_manifest_sha256", "base_fingerprint"):
            with self.subTest(field=field):
                saved = self.previous[field]
                self.previous[field] = "changed"
                with self.assertRaisesRegex(ValueError, "inputs changed"):
                    self.validate()
                self.previous[field] = saved

    def test_completed_scientific_or_different_protocol_attempt_is_rejected(self):
        changes = {"passed": True, "status": "passed", "scientific_training": True,
                   "formal_evaluation": True, "sample_split": "test", "task": "logic",
                   "protocol_version": "older", "error_type": "RuntimeError", "device": "npu:1"}
        for field, value in changes.items():
            with self.subTest(field=field):
                saved = self.previous[field]
                self.previous[field] = value
                with self.assertRaises(ValueError):
                    self.validate()
                self.previous[field] = saved
        # The canonical record can still be the preceding failed prefill attempt.
        self.previous["protocol_version"] = "older"
        self.previous["error_type"] = "AssertionError"
        self.validate(state=False)

    def test_incomplete_update_and_changed_sample_provenance_are_rejected(self):
        for field, value in (("status", "running"), ("steps", 2), ("optimizer_steps", 1),
                             ("completed_epochs", 0), ("device", "cpu")):
            with self.subTest(field=field):
                saved = self.state[field]
                self.state[field] = value
                with self.assertRaises(ValueError):
                    self.validate()
                self.state[field] = saved
        self.state["provenance"]["sample_ids"] = list(reversed(self.ids))
        with self.assertRaisesRegex(ValueError, "provenance"):
            self.validate()


class CachedInputContractTests(unittest.TestCase):
    def setUp(self):
        self.fixture = fixtures()
        self.environment = metadata_modules()
        self.subspaces = self.environment.__enter__()
        self.addCleanup(self.environment.__exit__, None, None, None)
        self.cache = self.make_cache()
        self.config = RECOVERY.engineering_config(self.fixture.config)

    def make_cache(self):
        value = self.fixture
        return RECOVERY.CachedEngineeringInputs(value.stats, value.spaces, value.config,
                                                value.samples, value.provenance, "npu:0")

    def collect(self, half):
        value = self.fixture
        return self.cache.collect_statistics(value.model, value.tokenizer,
                                              value.samples[half * 8:(half + 1) * 8], self.config, "npu:0")

    def collect_both(self):
        return {label: self.collect(index) for index, label in enumerate(RECOVERY.HALVES)}

    def build(self, stats):
        return self.cache.build_subspaces(stats, RECOVERY.HALVES[0], self.config,
                                          seen_tasks=list(RECOVERY.HALVES))

    def test_exact_two_halves_and_one_build_are_accepted_without_mutating_saved_metadata(self):
        original = deepcopy(self.fixture.stats[RECOVERY.HALVES[0]].metadata)
        stats = self.collect_both()
        self.assertIs(self.build(stats), self.fixture.spaces)
        self.cache.assert_consumed()
        stats[RECOVERY.HALVES[0]].metadata["new_output_only"] = True
        self.assertEqual(self.fixture.stats[RECOVERY.HALVES[0]].metadata, original)
        with self.assertRaisesRegex(ValueError, "additional"):
            self.collect(0)
        with self.assertRaisesRegex(ValueError, "order"):
            self.build(stats)

    def test_wrong_half_or_incomplete_consumption_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "ordering"):
            self.collect(1)
        with self.assertRaisesRegex(ValueError, "exactly two"):
            self.cache.assert_consumed()
        with self.assertRaisesRegex(ValueError, "order"):
            self.build(self.fixture.stats)

    def test_scope_rejects_nontrain_duplicate_and_changed_sample_contents(self):
        for field, value in (("task", "logic"), ("official_split", "test"),
                             ("id", self.fixture.samples[1].id), ("target", "changed")):
            with self.subTest(field=field):
                saved = getattr(self.fixture.samples[0], field)
                setattr(self.fixture.samples[0], field, value)
                with self.assertRaises(ValueError):
                    self.make_cache()
                setattr(self.fixture.samples[0], field, saved)

    def test_config_device_precision_native_config_and_encoding_are_bound(self):
        changes = ((self.config["loda"], "statistics_seed", 99),
                   (self.fixture.native_config, "hidden_size", 3),
                   (self.fixture.parameter.__dict__, "device", "cpu"),
                   (self.fixture.parameter.__dict__, "dtype", "torch.float32"),
                   (self.fixture.tokenizer.__dict__, "tag", "changed"))
        for values, field, changed in changes:
            with self.subTest(field=field):
                existed, original = field in values, values.get(field)
                values[field] = changed
                with self.assertRaises(ValueError):
                    self.collect(0)
                if existed:
                    values[field] = original
                else:
                    del values[field]

    def test_metadata_statistics_and_subspace_changes_are_rejected(self):
        stats = self.fixture.stats[RECOVERY.HALVES[0]]
        stats.metadata["prompt_token_hashes"][0] = "wrong-token-content"
        with self.assertRaisesRegex(ValueError, "metadata"):
            self.collect(0)
        self.fixture = fixtures()
        self.cache = self.make_cache()
        collected = self.collect_both()
        collected[RECOVERY.HALVES[1]]["model.layers.0"][0][0] = 999
        with self.assertRaisesRegex(ValueError, "changed statistics"):
            self.build(collected)
        self.fixture = fixtures()
        self.cache = self.make_cache()
        collected = self.collect_both()
        self.fixture.spaces["model.layers.0"]["lambda"] = [0.5]
        with self.assertRaisesRegex(ValueError, "contents changed"):
            self.build(collected)

    def test_changed_model_identity_and_seen_task_order_are_rejected(self):
        self.collect(0)
        self.fixture.model = deepcopy(self.fixture.model)
        with self.assertRaisesRegex(ValueError, "model/tokenizer"):
            self.collect(1)
        self.fixture = fixtures()
        self.cache = self.make_cache()
        collected = self.collect_both()
        with self.assertRaisesRegex(ValueError, "task order"):
            self.cache.build_subspaces(collected, RECOVERY.HALVES[0], self.config,
                                      seen_tasks=list(reversed(RECOVERY.HALVES)))

    def test_patch_scope_restores_original_functions_even_when_smoke_fails(self):
        original = self.subspaces.collect_statistics, self.subspaces.build_subspaces
        with self.assertRaisesRegex(RuntimeError, "smoke failed"):
            with self.cache.installed():
                self.assertEqual(self.subspaces.collect_statistics, self.cache.collect_statistics)
                raise RuntimeError("smoke failed")
        self.assertEqual((self.subspaces.collect_statistics, self.subspaces.build_subspaces), original)


class RecoveryPublicationTests(unittest.TestCase):
    @contextmanager
    def recovery_run(self, failure=None):
        with tempfile.TemporaryDirectory(prefix="portable-recovery-") as temporary:
            root = Path(temporary)
            runs, original = root / "runs", root / "old-attempt"
            runs.mkdir()
            loda = original / "loda_training_smoke"
            uninterrupted = loda / "uninterrupted"
            uninterrupted.mkdir(parents=True)
            config = {"platform": {"persistent_root": str(root), "device": "npu:0"},
                      "runtime": {"data_root": str(root / "data"), "output_root": str(runs)}}
            ids = [f"math:train:{index}" for index in range(16)]
            source = {"files": {"io.py": "new-portable-io"}}
            expected = {"config_sha256": digest({"locked": "config"}),
                        "data_manifest_sha256": digest({"locked": "data"}),
                        "base_fingerprint": "same-base", "source_manifest_sha256": digest(source)}
            previous = {"passed": False, "status": "failed", "task": "math", "sample_split": "train",
                        "sample_ids": ids[:2], "device": "npu:0", "scientific_training": False,
                        "formal_evaluation": False, "error_type": "UnpicklingError",
                        "protocol_version": "same_shape_causal_prefill_v2", **expected,
                        "source_manifest_sha256": "old-source"}
            canonical = runs / "smoke.json"
            old_bytes = json.dumps(previous).encode()
            canonical.write_bytes(old_bytes)
            (original / "smoke.json").write_bytes(old_bytes)
            initial = uninterrupted / "initial.pt"
            initial.write_bytes(b"unchanged CPU initial representation")
            provenance = {"kind": "engineering_LoDA_GAO_smoke", "base_fingerprint": "same-base",
                          "source_task": "math", "source_split": "train", "scientific_training": False,
                          "sample_ids": ids, "diagnostic_loss_ids": ids[:2], "initial_override_sha256": None,
                          "subspaces_content_sha256": RECOVERY.EXPECTED_SUBSPACES_SHA256,
                          "training_sources_sha256": "old-training-sources"}
            state = {"status": "exported_uncommitted", "steps": 1, "optimizer_steps": 2,
                     "completed_epochs": 1, "device": "npu:0", "provenance": provenance,
                     "provenance_sha256": digest(provenance), "initial_path": str(initial)}
            (uninterrupted / "training-state.json").write_text(json.dumps(state), encoding="utf-8")
            for label in RECOVERY.HALVES:
                (loda / f"{label}_statistics.pt").write_bytes(b"unchanged CPU statistics")
            saved_artifacts = {path: path.read_bytes() for path in original.rglob("*") if path.is_file()}
            samples = [SimpleNamespace(id=value) for value in ids]
            runner = SimpleNamespace(output_root=runs, config=config, manifest={"locked": "data"},
                                     source_root=PROJECT, taskset=None,
                                     base_fingerprint=lambda: "same-base", samples=lambda *args: samples)

            class Cache:
                collect_calls = 2
                build_calls = 1

                def __init__(self, *args):
                    pass

                @contextmanager
                def installed(self):
                    yield self

                def assert_consumed(self):
                    if failure == "unconsumed":
                        raise ValueError("cache was not consumed")

            def atomic_json(path, value):
                Path(path).write_text(json.dumps(value), encoding="utf-8")

            def read_json(path):
                return json.loads(Path(path).read_bytes())

            def run_smoke(config, *, task, output_dir):
                (output_dir / "fresh-update-evidence.txt").write_text("new current-source GAO execution")
                if failure == "smoke":
                    atomic_json(output_dir / "smoke.json", {"passed": False, "status": "failed"})
                    raise RuntimeError("fresh smoke failed")
                atomic_json(output_dir / "smoke.json", {"passed": True, "status": "passed", **expected,
                    "checks": {"loda_true_gao_microbatch_step": True, "loda_checkpoint_resume_exact": True,
                               "loda_resume_does_not_repeat_step": True}})
                if failure == "concurrent":
                    canonical.write_bytes(b'{"newer":"record"}')

            modules = {"task_vector.config": module("task_vector.config", load_config=lambda path: deepcopy(config),
                           scientific_config=lambda cfg: {"locked": "config"}, source_manifest=lambda path: source),
                       "task_vector.io": module("task_vector.io", atomic_json=atomic_json, object_sha256=digest,
                           file_sha256=lambda path: hashlib.sha256(Path(path).read_bytes()).hexdigest(), read_json=read_json),
                       "task_vector.representations": module("task_vector.representations",
                           Representation=SimpleNamespace(load=lambda path: object())),
                       "task_vector.runner": module("task_vector.runner", ExperimentRunner=lambda *args, **kwargs: runner),
                       "task_vector.smoke": module("task_vector.smoke", run_smoke=run_smoke),
                       "task_vector.subspaces": module("task_vector.subspaces", load_statistics=lambda path: {}),
                       "task_vector.training": module("task_vector.training", _content_digest=digest),
                       "task_vector.workflow": module("task_vector.workflow",
                           smoke_evidence=lambda *args: {"valid": failure != "workflow"})}
            with patch.dict(sys.modules, modules), patch.object(RECOVERY, "CachedEngineeringInputs", Cache), \
                    patch.object(RECOVERY, "reconstruct_subspaces", return_value={"verified": "spaces"}), \
                    patch.object(sys, "path", list(sys.path)), \
                    patch.object(sys, "argv", ["recover", "--config", "fixture.yaml", "--previous-attempt", str(original)]), \
                    redirect_stdout(io.StringIO()):
                yield SimpleNamespace(canonical=canonical, old=old_bytes, runs=runs, saved=saved_artifacts,
                                      expected=expected, execute=RECOVERY.main)

    def test_only_complete_current_source_smoke_is_published_and_old_inputs_stay_byte_exact(self):
        with self.recovery_run() as case:
            case.execute()
            record = json.loads(case.canonical.read_bytes())
            self.assertTrue(record["passed"])
            self.assertEqual(record["source_manifest_sha256"], case.expected["source_manifest_sha256"])
            attempt = Path(record["engineering_attempt_directory"])
            self.assertEqual((attempt / "previous-smoke.json").read_bytes(), case.old)
            receipt = json.loads((attempt / "recovery.json").read_bytes())
            self.assertEqual(receipt["status"], "passed")
            self.assertFalse(receipt["legacy_optimizer_checkpoint_loaded"])
            self.assertFalse(receipt["old_updates_reused"])
            self.assertEqual(receipt["original_training_sources_sha256"], "old-training-sources")
            for path, original in case.saved.items():
                self.assertEqual(path.read_bytes(), original)

    def test_failure_retains_attempt_evidence_and_never_replaces_canonical(self):
        for failure in ("smoke", "unconsumed", "workflow"):
            with self.subTest(failure=failure), self.recovery_run(failure) as case:
                with self.assertRaises((RuntimeError, ValueError)):
                    case.execute()
                self.assertEqual(case.canonical.read_bytes(), case.old)
                attempts = list((case.runs / "engineering_smoke").iterdir())
                self.assertEqual(len(attempts), 1)
                receipt = json.loads((attempts[0] / "recovery.json").read_bytes())
                self.assertEqual(receipt["status"], "failed")
                self.assertTrue(receipt["canonical_unchanged"])
                self.assertTrue((attempts[0] / "fresh-update-evidence.txt").exists())
                for path, original in case.saved.items():
                    self.assertEqual(path.read_bytes(), original)

    def test_concurrent_canonical_change_is_not_overwritten(self):
        with self.recovery_run("concurrent") as case:
            with self.assertRaisesRegex(RuntimeError, "changed during retry"):
                case.execute()
            self.assertEqual(case.canonical.read_bytes(), b'{"newer":"record"}')


@unittest.skipUnless(importlib.util.find_spec("torch") is not None, "real tensor fixture requires torch")
class TensorReconstructionTests(unittest.TestCase):
    def setUp(self):
        import torch
        sys.path.insert(0, str(PROJECT / "src"))
        self.addCleanup(sys.path.remove, str(PROJECT / "src"))
        from task_vector.subspaces import Statistics as TensorStatistics
        from task_vector.training import _content_digest
        self.digest = _content_digest
        layer = "model.layers.0"
        self.stats = {RECOVERY.HALVES[0]: TensorStatistics({layer: torch.tensor([[2., 3.], [-1., 4.]])},
                                                         metadata={"half": "a", "sample_ids": ["a"]}),
                      RECOVERY.HALVES[1]: TensorStatistics({layer: torch.tensor([[5., 4.], [0., 7.]])},
                                                         metadata={"half": "b", "sample_ids": ["b"]})}
        a_g, a_i = torch.tensor([[0.25, -0.5]]), torch.tensor([[-0.75, 0.125]])
        meta = {"route": "engineering_LoDA_GAO", "epsilon": 1e-12,
                "calibration": {label: deepcopy(stats.metadata) for label, stats in self.stats.items()}}
        factors = {f"{layer}.self_attn.{kind}": [SimpleNamespace(A=a_g.clone(), B=torch.zeros(rows, 1)),
                                               SimpleNamespace(A=a_i.clone(), B=torch.zeros(rows, 1))]
                   for kind, rows in (("q_proj", 2), ("k_proj", 1), ("v_proj", 1))}
        self.initial = SimpleNamespace(space="parameter", task="math", seed=47017, factors=factors,
            metadata={"version": "raw", "A_general_trainable": False, "isolated_enabled": True,
                      "branches": ["general", "isolated"], "lambda": {layer: [0.8125]},
                      "subspaces": {layer: meta}})
        self.spaces = {layer: {"A_G": a_g, "A_I": a_i, "S_self": torch.tensor([[2., 1.], [1., 4.]]),
                              "S_other": torch.tensor([[5., 2.], [2., 7.]]), "lambda": torch.tensor([0.8125]),
                              "metadata": deepcopy(meta)}}

    def test_reconstruction_preserves_all_keys_fp32_bytes_lambda_and_metadata_without_eigh(self):
        import torch
        expected = self.digest(self.spaces)
        with patch.object(torch.linalg, "eigh", side_effect=AssertionError("must not recompute bases")):
            rebuilt = RECOVERY.reconstruct_subspaces(self.initial, self.stats, expected)
        self.assertEqual(self.digest(rebuilt), expected)
        self.assertEqual(set(rebuilt["model.layers.0"]), set(self.spaces["model.layers.0"]))
        self.assertEqual(rebuilt["model.layers.0"]["A_G"].dtype, torch.float32)
        rebuilt["model.layers.0"]["metadata"]["route"] = "changed-new-copy"
        self.assertEqual(self.initial.metadata["subspaces"]["model.layers.0"]["route"], "engineering_LoDA_GAO")

    def test_any_changed_matrix_lambda_basis_or_calibration_metadata_is_rejected(self):
        expected = self.digest(self.spaces)
        changes = (lambda: self.stats[RECOVERY.HALVES[0]]["model.layers.0"].add_(1),
                   lambda: self.initial.metadata["lambda"]["model.layers.0"].__setitem__(0, 0.5),
                   lambda: self.initial.factors["model.layers.0.self_attn.k_proj"][0].A.add_(1),
                   lambda: self.stats[RECOVERY.HALVES[1]].metadata.update(extra_unbound_field=True))
        for change in changes:
            with self.subTest(change=change):
                saved_initial, saved_stats = deepcopy(self.initial), deepcopy(self.stats)
                change()
                with self.assertRaises(ValueError):
                    RECOVERY.reconstruct_subspaces(self.initial, self.stats, expected)
                self.initial, self.stats = saved_initial, saved_stats


if __name__ == "__main__":
    unittest.main(verbosity=2)
