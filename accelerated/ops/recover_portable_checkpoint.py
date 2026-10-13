"""Rerun the complete engineering smoke using verified saved calibration inputs.

Only the two frozen-base statistics collections and spectral decompositions are
reused. The current trainer executes a new GAO batch and its ordinary safe
checkpoint/resume checks. No legacy optimizer checkpoint is deserialized.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace

from recover_prefill_smoke import publish_passed_attempt


HALVES = ("engineering_train_half_a", "engineering_train_half_b")
EXPECTED_SUBSPACES_SHA256 = "84503033e5aca5dd84b27a9b039c6bd8b070cfc016ddd0832fd32a1531f232a0"


def engineering_config(config):
    """Match the configuration constructed by the unchanged engineering smoke."""
    result = deepcopy(config)
    result["supplement_label"] = "engineering_smoke_only_one_LoDA_logical_batch"
    result.setdefault("training", {}).update(epochs=1, micro_batch_size=2,
        effective_batch_size=16, gradient_accumulation_steps=8, warmup_steps=0,
        save_every_steps=1, weight_decay_loda=0.)
    result["training"].pop("resume_from", None)
    result.setdefault("loda", {}).update(statistics_samples=8, train_general_a=False,
                                        isolated_enabled=True)
    return result


def validate_attempt_inputs(previous, training_state, expected, sample_ids, device):
    """Reject changed inputs before loading any calibration tensors or model."""
    if (previous.get("passed") is not False or previous.get("status") != "failed"
            or previous.get("task") != "math" or previous.get("sample_split") != "train"
            or previous.get("scientific_training") is not False
            or previous.get("formal_evaluation") is not False):
        raise ValueError("Recovery requires a failed engineering-only math/train smoke")
    for field in ("config_sha256", "data_manifest_sha256", "base_fingerprint"):
        if previous.get(field) != expected[field]:
            raise ValueError(f"Original smoke inputs changed: {field}")
    if len(sample_ids) != 16 or len(set(sample_ids)) != 16 or previous.get("sample_ids") != sample_ids[:2]:
        raise ValueError("Original engineering training sample identities changed")
    if previous.get("device") != str(device):
        raise ValueError("Recovery device differs from the failed engineering smoke")
    if training_state is None:
        return
    if (previous.get("protocol_version") != "same_shape_causal_prefill_v2"
            or previous.get("error_type") != "UnpicklingError"
            or training_state.get("status") != "exported_uncommitted"
            or training_state.get("steps") != 1 or training_state.get("optimizer_steps") != 2
            or training_state.get("completed_epochs") != 1
            or training_state.get("device") != str(device)):
        raise ValueError("Saved engineering update is not the expected completed one-batch attempt")
    provenance = training_state.get("provenance", {})
    required = {"kind": "engineering_LoDA_GAO_smoke", "base_fingerprint": expected["base_fingerprint"],
                "source_task": "math", "source_split": "train", "scientific_training": False,
                "sample_ids": sample_ids, "diagnostic_loss_ids": sample_ids[:2],
                "initial_override_sha256": None}
    if any(provenance.get(key) != value for key, value in required.items()):
        raise ValueError("Saved engineering calibration provenance differs")


def reconstruct_subspaces(initial, stats, expected_digest):
    """Recover the exact FP32 bases and statistics without eigendecomposition."""
    import torch
    from task_vector.subspaces import _symmetric_cpu
    from task_vector.training import _content_digest

    if (initial.space != "parameter" or initial.task != "math" or initial.seed != 47017
            or initial.metadata.get("version") != "raw"
            or initial.metadata.get("A_general_trainable") is not False
            or initial.metadata.get("isolated_enabled") is not True
            or initial.metadata.get("branches") != ["general", "isolated"]):
        raise ValueError("Saved initial representation is not the original engineering LoDA initialization")
    if tuple(stats) != HALVES:
        raise ValueError("Expected exactly the two ordered engineering calibration halves")
    metadata = initial.metadata.get("subspaces", {})
    layers = tuple(sorted(metadata))
    if not layers or any(set(stats[label]) != set(layers) for label in HALVES):
        raise ValueError("Initial representation and statistics layers differ")
    if set(initial.factors) != {f"{layer}.self_attn.{kind}" for layer in layers
                                for kind in ("q_proj", "k_proj", "v_proj")}:
        raise ValueError("Initial representation does not contain all native Q/K/V factors")
    result = {}
    for layer in layers:
        calibration = metadata[layer].get("calibration", {})
        if any(calibration.get(label) != stats[label].metadata for label in HALVES):
            raise ValueError("Saved statistics metadata differs from the original bound calibration metadata")
        terms = initial.factors[f"{layer}.self_attn.q_proj"]
        if len(terms) != 2:
            raise ValueError("Initial representation requires General and Isolated factors")
        a_general, a_isolated = terms[0].A, terms[1].A
        for kind in ("q_proj", "k_proj", "v_proj"):
            current = initial.factors[f"{layer}.self_attn.{kind}"]
            if (len(current) != 2 or not torch.equal(current[0].A, a_general)
                    or not torch.equal(current[1].A, a_isolated)
                    or any(bool(torch.count_nonzero(term.B)) for term in current)):
                raise ValueError("Saved initialization has inconsistent bases or nonzero B")
        current = _symmetric_cpu(stats[HALVES[0]][layer])
        # Preserve the original sum order and float64 symmetry before FP32.
        other = sum((_symmetric_cpu(stats[label][layer]) for label in HALVES[1:]),
                    torch.zeros_like(current))
        result[layer] = {"A_G": a_general.detach().cpu().clone(),
                         "A_I": a_isolated.detach().cpu().clone(),
                         "S_self": current.float(), "S_other": other.float(),
                         "lambda": torch.tensor(initial.metadata["lambda"][layer], dtype=torch.float32),
                         "metadata": deepcopy(metadata[layer])}
    if _content_digest(result) != expected_digest:
        raise ValueError("Reconstructed subspaces differ from the original bound content digest")
    return result


class CachedEngineeringInputs:
    """Strictly scoped equivalents for exactly two collections and one build."""

    def __init__(self, stats, spaces, config, samples, provenance, device):
        from task_vector.io import object_sha256
        from task_vector.training import _content_digest, _sample_content

        self.stats, self.spaces = stats, spaces
        self.config = engineering_config(config)
        self.config_digest = object_sha256(self.config)
        self.samples = list(samples)
        if (len(self.samples) != 16 or len({sample.id for sample in self.samples}) != 16
                or any(sample.task != "math" or sample.official_split != "train" for sample in self.samples)):
            raise ValueError("Cached calibration requires exactly 16 unique math official-train samples")
        self.provenance, self.device = provenance, str(device)
        self.sample_digest = _content_digest({"train": _sample_content(self.samples),
                                             "tune": _sample_content(self.samples[:2])})
        if self.sample_digest != provenance.get("sample_content_sha256"):
            raise ValueError("Prepared engineering sample contents changed")
        self.stats_digests = {label: self._stats_digest(stats[label]) for label in HALVES}
        self.collect_calls = self.build_calls = 0
        self.model = self.tokenizer = None

    @staticmethod
    def _stats_digest(stats):
        from task_vector.training import _content_digest
        return _content_digest({"matrices": dict(stats), "metadata": stats.metadata})

    def _validate_context(self, model, tokenizer, config, device):
        from task_vector.io import object_sha256
        from task_vector.prompts import encode_sample
        from task_vector.training import _content_digest

        if object_sha256(config) != self.config_digest or str(device) != self.device:
            raise ValueError("Cached calibration configuration or device differs")
        if self.model is not None:
            if model is not self.model or tokenizer is not self.tokenizer:
                raise ValueError("Cached calibration model/tokenizer changed between halves")
            return
        native_config = model.config.to_dict() if hasattr(model.config, "to_dict") else vars(model.config)
        if _content_digest(native_config) != self.provenance.get("native_model_config_sha256"):
            raise ValueError("Cached calibration native model configuration differs")
        parameter = next(model.parameters())
        if str(parameter.device) != self.device or str(parameter.dtype) != "torch.bfloat16":
            raise ValueError("Cached calibration model device or BF16 precision differs")
        cfg = config["training"]
        encoded = [encode_sample(tokenizer, sample, max_prompt=cfg["max_prompt_tokens"],
                                 max_target=cfg["max_target_tokens"]) for sample in self.samples]
        if _content_digest({"train": encoded, "tune": encoded[:2]}) != self.provenance.get("encoded_samples_sha256"):
            raise ValueError("Cached calibration tokenizer or encoded samples changed")
        self.model, self.tokenizer = model, tokenizer

    def collect_statistics(self, model, tokenizer, samples, config, device):
        import torch
        from task_vector.io import object_sha256
        from task_vector.prompts import render_prompt
        from task_vector.subspaces import Statistics, uniform_token_positions
        from task_vector.training import _content_digest, _sample_content

        if self.collect_calls >= 2 or self.build_calls:
            raise ValueError("Unexpected additional cached statistics request")
        self._validate_context(model, tokenizer, config, device)
        label = HALVES[self.collect_calls]
        half = self.samples[self.collect_calls * 8:(self.collect_calls + 1) * 8]
        if _content_digest(_sample_content(list(samples))) != _content_digest(_sample_content(half)):
            raise ValueError("Cached statistics half sample contents or ordering differ")
        stats = self.stats[label]
        cfg = config["loda"]
        chosen = torch.randperm(8, generator=torch.Generator().manual_seed(int(cfg.get("statistics_seed", 17)))).tolist()
        positions, hashes = [], []
        for index in chosen:
            prompt = render_prompt(tokenizer, half[index], variant="canonical")
            tokens = tokenizer(prompt, add_special_tokens=False, return_tensors="pt")
            ids = tokens["input_ids"]
            if ids.ndim != 2 or ids.shape[0] != 1 or ids.shape[1] > config["training"]["max_prompt_tokens"]:
                raise ValueError("Cached calibration prompt shape or length differs")
            if "attention_mask" in tokens and not bool(tokens["attention_mask"].bool().all()):
                raise ValueError("Cached calibration prompts require an unpadded mask")
            positions.append(uniform_token_positions(ids.shape[1], int(cfg.get("max_statistics_tokens", 64))).tolist())
            hashes.append(object_sha256({"input_ids": ids.cpu().tolist()}))
        expected = {"statistic": "token_second_moment_mean", "centered": False,
                    "statistics_seed": int(cfg.get("statistics_seed", 17)),
                    "sample_count": 8, "sample_ids": [half[index].id for index in chosen],
                    "token_positions": positions, "prompt_token_hashes": hashes,
                    "token_counts": {layer: sum(map(len, positions)) for layer in stats},
                    "source_split": "train", "engineering_only": True, "actual_source_task": "math",
                    "calibration_scope": "two disjoint 8-sample halves of the same source task training pool"}
        if any(stats.metadata.get(key) != value for key, value in expected.items()):
            raise ValueError(f"Cached statistics prompt sampling or metadata differs: {label}")
        self.collect_calls += 1
        print(f"REUSED_STATISTICS={label} prompt_inputs_verified=PASS", flush=True)
        return Statistics(stats, metadata=deepcopy(stats.metadata))

    def build_subspaces(self, stats_by_task, task, config, *, seen_tasks=None):
        from task_vector.io import object_sha256
        from task_vector.training import _content_digest

        if self.collect_calls != 2 or self.build_calls or tuple(stats_by_task) != HALVES:
            raise ValueError("Unexpected cached subspace build order or statistics set")
        if (task != HALVES[0] or tuple(seen_tasks or ()) != HALVES
                or object_sha256(config) != self.config_digest):
            raise ValueError("Cached subspace task, task order or configuration differs")
        if any(self._stats_digest(stats_by_task[label]) != self.stats_digests[label] for label in HALVES):
            raise ValueError("Cached subspace build received changed statistics")
        if _content_digest(self.spaces) != self.provenance.get("subspaces_content_sha256"):
            raise ValueError("Cached subspace contents changed before training")
        self.build_calls += 1
        print("REUSED_SUBSPACES=PASS original_content_digest_verified no_eigendecomposition", flush=True)
        return self.spaces

    def assert_consumed(self):
        if (self.collect_calls, self.build_calls) != (2, 1):
            raise ValueError("Complete smoke must consume exactly two cached collections and one subspace build")

    @contextmanager
    def installed(self):
        from task_vector import subspaces

        original_collect, original_build = subspaces.collect_statistics, subspaces.build_subspaces
        subspaces.collect_statistics, subspaces.build_subspaces = self.collect_statistics, self.build_subspaces
        try:
            yield self
        finally:
            subspaces.collect_statistics, subspaces.build_subspaces = original_collect, original_build


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--previous-attempt", required=True)
    args = parser.parse_args()
    project = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(project / "src"))
    from task_vector.config import load_config, scientific_config, source_manifest
    from task_vector.io import atomic_json, file_sha256, object_sha256, read_json
    from task_vector.representations import Representation
    from task_vector.runner import ExperimentRunner
    from task_vector.smoke import run_smoke
    from task_vector.subspaces import load_statistics
    from task_vector.training import _content_digest
    from task_vector.workflow import smoke_evidence

    config = load_config(args.config)
    root = Path(config["platform"]["persistent_root"])
    config.setdefault("runtime", {}).setdefault("data_root", str(root / "data"))
    config["runtime"].setdefault("output_root", str(root / "runs"))
    runner = ExperimentRunner(config, source_root=project)
    canonical = runner.output_root / "smoke.json"
    previous_bytes = canonical.read_bytes()
    previous = json.loads(previous_bytes)
    original = Path(args.previous_attempt).resolve()
    original_smoke = read_json(original / "smoke.json")
    loda_source = original / "loda_training_smoke"
    training_path = loda_source / "uninterrupted" / "training-state.json"
    training_state = read_json(training_path)
    samples = runner.samples("math", "train")[:16]
    sample_ids = [sample.id for sample in samples]
    expected = {"config_sha256": object_sha256(scientific_config(config)),
                "data_manifest_sha256": object_sha256(runner.manifest),
                "base_fingerprint": runner.base_fingerprint(),
                "source_manifest_sha256": object_sha256(source_manifest(project))}
    device = config["platform"]["device"]
    validate_attempt_inputs(previous, None, expected, sample_ids, device)
    validate_attempt_inputs(original_smoke, training_state, expected, sample_ids, device)
    provenance = training_state["provenance"]
    if training_state.get("provenance_sha256") != object_sha256(provenance):
        raise ValueError("Saved engineering provenance digest is corrupt")
    if provenance.get("subspaces_content_sha256") != EXPECTED_SUBSPACES_SHA256:
        raise ValueError("Saved subspace digest differs from the inspected engineering attempt")
    initial_path = loda_source / "uninterrupted" / "initial.pt"
    if Path(training_state["initial_path"]).resolve() != initial_path.resolve():
        raise ValueError("Saved initial representation path differs from the selected attempt")
    input_paths = [training_path, original / "smoke.json", initial_path]
    print("STAGE=validate_saved_engineering_inputs", flush=True)
    stats = {}
    for label in HALVES:
        path = loda_source / f"{label}_statistics.pt"
        input_paths.append(path)
        stats[label] = load_statistics(path)
    initial = Representation.load(initial_path)
    print("STAGE=reconstruct_saved_subspaces", flush=True)
    spaces = reconstruct_subspaces(initial, stats, provenance["subspaces_content_sha256"])
    cache = CachedEngineeringInputs(stats, spaces, config, samples, provenance, device)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    attempt = runner.output_root / "engineering_smoke" / f"portable-checkpoint-fix-{stamp}-{os.getpid()}"
    attempt.mkdir(parents=True, exist_ok=False)
    (attempt / "previous-smoke.json").write_bytes(previous_bytes)
    receipt = {"kind": "engineering_smoke_cached_calibration_recovery", "status": "running",
               "scientific_training": False, "formal_evaluation": False,
               "attempt": str(attempt), "canonical_smoke": str(canonical),
               "original_attempt": str(original), "previous_failed_smoke_sha256": hashlib.sha256(previous_bytes).hexdigest(),
               "input_files_sha256": {str(path): file_sha256(path) for path in input_paths},
               "helper_sha256": file_sha256(__file__), "expected_provenance": expected,
               "cached_subspaces_content_sha256": _content_digest(spaces),
               "original_training_sources_sha256": provenance["training_sources_sha256"],
               "original_smoke_source_manifest_sha256": original_smoke["source_manifest_sha256"],
               "legacy_optimizer_checkpoint_loaded": False, "old_updates_reused": False,
               "reused": "exact saved frozen-base calibration matrices and bases only",
               "rerun": "complete current-source smoke, including new GAO update and standard weights-only resume"}
    atomic_json(attempt / "recovery.json", receipt)
    print(f"SMOKE_ATTEMPT={attempt}", flush=True)
    print("CACHED_CALIBRATION_INPUTS=PASS", flush=True)
    try:
        with cache.installed():
            run_smoke(config, task="math", output_dir=attempt)
        cache.assert_consumed()
        receipt.update(cached_collect_calls=cache.collect_calls, cached_build_calls=cache.build_calls)
        atomic_json(attempt / "recovery.json", receipt)
        attempt_view = SimpleNamespace(output_root=attempt, config=runner.config,
            manifest=runner.manifest, source_root=runner.source_root,
            base_fingerprint=runner.base_fingerprint, taskset=runner.taskset)
        evidence = smoke_evidence(attempt_view, "mvp")
        if not evidence["valid"]:
            raise ValueError(f"New smoke was not accepted by the workflow: {evidence}")
        candidate = read_json(attempt / "smoke.json")
        candidate["calibration_reuse_receipt"] = str((attempt / "recovery.json").resolve())
        candidate["calibration_reuse_subspaces_content_sha256"] = receipt["cached_subspaces_content_sha256"]
        atomic_json(attempt / "smoke.json", candidate)
        publish_passed_attempt(canonical, attempt, previous_bytes, expected, atomic_json)
        evidence = smoke_evidence(runner, "mvp")
        if not evidence["valid"]:
            raise ValueError(f"Published smoke was not accepted by the workflow: {evidence}")
        receipt.update(status="passed", workflow_smoke_evidence=evidence,
                       previous_artifacts_preserved=True)
        atomic_json(attempt / "recovery.json", receipt)
        print("SMOKE_CHECKS=" + json.dumps(read_json(canonical)["checks"], ensure_ascii=False), flush=True)
        print("REAL_MODEL_SMOKE=PASS", flush=True)
        print("WORKFLOW_SMOKE_EVIDENCE=PASS", flush=True)
        print(f"CANONICAL_SMOKE={canonical}", flush=True)
        print("NEXT=training-job configuration; no training job submitted", flush=True)
    except BaseException as exc:
        receipt.update(status="failed", error=f"{type(exc).__name__}: {exc}",
                       cached_collect_calls=cache.collect_calls, cached_build_calls=cache.build_calls,
                       canonical_unchanged=canonical.read_bytes() == previous_bytes)
        atomic_json(attempt / "recovery.json", receipt)
        raise


if __name__ == "__main__":
    main()
