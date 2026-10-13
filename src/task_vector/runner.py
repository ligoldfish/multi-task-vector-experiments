"""Artifact-cached experiment operations with strict split provenance.

Every final evaluation is reached through a frozen discovery decision file.
Training jobs can run independently on NPU cards; an individual job retains
the documented effective batch and is never implicitly converted to DDP.
"""
from __future__ import annotations
from contextlib import contextmanager
from copy import deepcopy
from itertools import combinations
from pathlib import Path
import gc
import time
import numpy as np
import torch
from .io import atomic_json, atomic_torch_save, object_sha256, file_sha256, read_json, read_jsonl, write_jsonl
from .representations import Representation, LowRankFactor, linear_combination, scaled, random_like, match_norm, norm
from .analysis import (decompose, random_subspace_control, gram_matrix, cosine_matrix, layer_subspace_angles,
                       relation_consistency, paired_bootstrap, transfer_matrix, discover_groups,
                       choose_discovery_candidate, group_residual_decomposition)
from .config import create_lock, verify_lock, scientific_config, source_manifest
from .runtime import configure_device, seed_everything, memory_evidence, synchronize


class ExperimentRunner:
    def __init__(self, config, *, source_root=None):
        self.config = deepcopy(config)
        self.source_root = Path(source_root or Path(__file__).resolve().parents[2])
        self.root = Path(config["platform"]["persistent_root"])
        self.data_root = Path(config.get("runtime", {}).get("data_root", self.root / "data"))
        self.output_root = Path(config.get("runtime", {}).get("output_root", self.root / "runs"))
        self.output_root.mkdir(parents=True, exist_ok=True)
        self.device = None
        self.model = self.tokenizer = None
        self._loaded_base_fingerprint = None
        self._manifest = None
        self._artifact_hash_cache = {}
        self._representation_cache = {}
        self._protocol_cache = {}
        self._active_protocol_phase = "final"
        self._generation_executor = None

    @property
    def manifest(self):
        if self._manifest is None:
            self._manifest = read_json(self.data_root / "manifest.json")
            if self._manifest.get("status") != "complete":
                raise ValueError("Prepared data manifest is incomplete; resolve its recorded preflight failure")
        return self._manifest

    def samples(self, task, split):
        from .data import load_samples
        return load_samples(self.data_root, task, split)

    def load_model(self):
        # Training/calibration on the coordinator must not compete with eight
        # persistent inference copies. Cached training returns before this call.
        self.close_generation_workers()
        fingerprint = self.base_fingerprint()
        if self.model is not None and self._loaded_base_fingerprint != fingerprint:
            self.release_model()
        if self.model is None:
            from .modeling import load_model
            self.device = configure_device(self.config["platform"]["device"])
            self.model, self.tokenizer = load_model(self.config, self.device)
            self._loaded_base_fingerprint = fingerprint
        return self.model, self.tokenizer

    def release_model(self):
        self.model = self.tokenizer = None
        self._loaded_base_fingerprint = None
        gc.collect()
        if self.device is not None:
            api = getattr(torch, str(self.device).split(":")[0], None)
            if hasattr(api, "empty_cache"):
                api.empty_cache()

    def close_generation_workers(self):
        executor = self._generation_executor
        self._generation_executor = None
        if executor is not None:
            executor.close()

    def close(self):
        self.close_generation_workers()
        self.release_model()

    def _parallel_generation(self):
        if self._generation_executor is None:
            from .parallel_evaluation import PersistentGenerationExecutor
            self.release_model()
            self._generation_executor = PersistentGenerationExecutor(self.config, self.output_root)
        return self._generation_executor

    def _calibration_samples(self, task, scope):
        # E6 heldout tasks and SAMSum never become timing-selection data.
        allowed = task in self.taskset(scope)
        if task == "joint":
            # A joint panel can contain different pairs in different folds.
            # Keep its prescribed batch 1 rather than selecting on heldout
            # subtasks or silently changing the timing population by scope.
            return None, False
        return (self.samples(task, "tune"), True) if allowed else (None, False)

    def generate_merged(self, samples, config, task, merged_weights_path,
                        merged_weights_sha256, variant="canonical", scope="mvp"):
        """Generate from the recorded native-dtype sequential merge, unchanged."""
        calibration, allowed = self._calibration_samples(task, scope)
        return self._parallel_generation().generate_merged(samples, config,
            task=task, variant=variant, merged_weights_path=merged_weights_path,
            merged_weights_sha256=merged_weights_sha256,
            calibration_samples=calibration, allow_calibration=allowed,
            panel_label=f"sequential/{task}")

    def taskset(self, scope):
        tasks = list(self.config["data"]["tasks"])
        if scope == "expanded":
            return tasks + list(self.config["data"]["extension_tasks"])
        if scope.startswith("loo-"):
            target = scope[4:]
            if target not in tasks:
                raise ValueError(f"Unknown heldout task {target}")
            return [t for t in tasks if t != target]
        if scope != "mvp":
            raise ValueError(scope)
        return tasks

    def directory(self, scope, route, seed, task, layers=None):
        suffix = "" if route != "ltv" or not layers or layers == [14] else "-layers-" + "-".join(map(str, layers))
        # LTV is independent of all other source tasks; reuse the same four vectors in every fold.
        family = "independent" if route == "ltv" else scope
        return self.output_root / "training" / family / f"{route}{suffix}" / str(seed) / task

    def statistics(self, scope):
        from .subspaces import collect_statistics, save_statistics, load_statistics
        self.load_model()
        stats = {}
        for task in self.taskset(scope):
            path = self.output_root / "statistics" / "frozen_base" / f"{task}.pt"
            from filelock import FileLock
            path.parent.mkdir(parents=True, exist_ok=True)
            signature = object_sha256({"model": self.base_fingerprint(), "loda": self.config["loda"],
                                       "prompt": self.config["training"]["max_prompt_tokens"],
                                       "samples": self.manifest.get("manifest_content_sha256", object_sha256(self.manifest)),
                                       "sources": source_manifest(self.source_root),
                                       "train_ids": [s.id for s in self.samples(task, "train")]})
            with FileLock(str(path) + ".lock", timeout=3600):
                if path.exists():
                    stats[task] = load_statistics(path)
                    if stats[task].metadata.get("runner_signature") != signature:
                        raise ValueError(f"Stale calibration statistics: {path}; use a new run root")
                else:
                    stats[task] = collect_statistics(self.model, self.tokenizer, self.samples(task, "train"), self.config, self.device)
                    stats[task].metadata["runner_signature"] = signature
                    save_statistics(stats[task], path)
        return stats

    def train(self, scope, route, seed, task, layers=None):
        from filelock import FileLock
        output = self.directory(scope, route, seed, task, layers)
        output.mkdir(parents=True, exist_ok=True)
        with FileLock(str(output / "training.lock"), timeout=0):
            return self._train(scope, route, seed, task, layers)

    def _train(self, scope, route, seed, task, layers=None):
        from .training import train_task
        if task not in self.taskset(scope):
            raise ValueError("Heldout or unknown task cannot be trained in this scope")
        output = self.directory(scope, route, seed, task, layers)
        config = deepcopy(self.config)
        if layers is not None:
            config["ltv"]["layers"] = layers
        provenance = {"base_fingerprint": self.base_fingerprint(),
                      "data_manifest_sha256": object_sha256(self.manifest),
                      "source_manifest_sha256": object_sha256(source_manifest(self.source_root))}
        request_signature = object_sha256({"config": scientific_config(config), "route": route, "seed": seed,
                                           "scope": scope if route == "loda" else "independent", "task": task,
                                           "data": self.manifest["manifest_content_sha256"],
                                           "base": provenance["base_fingerprint"],
                                           "sources": source_manifest(self.source_root)})
        provenance["request_signature"] = request_signature
        if (output / "training.json").exists():
            result = read_json(output / "training.json")
            if result.get("request_signature") != request_signature:
                raise ValueError(f"Cached training protocol/source differs: {output}; use a new run root")
            self._verify_training_exports(result, request_signature, provenance["base_fingerprint"])
            return result
        pending = output / "training-state.json"
        if pending.exists():
            result = read_json(pending)
            if result.get("status") == "exported_uncommitted":
                if result.get("provenance", {}).get("request_signature") != request_signature:
                    raise ValueError("Uncommitted export belongs to different sources/configuration")
                return self._commit_training(result, request_signature, provenance, config, scope, route, task, output)
        self.load_model()
        subspaces = None
        if route == "loda":
            from .subspaces import build_subspaces
            stats = self.statistics(scope)
            config["loda"]["seen_tasks"] = self.taskset(scope)
            subspaces = build_subspaces(stats, task, config)
            atomic_torch_save(output / "subspaces.pt", subspaces)
        # Resume only this run's explicit or last complete logical-step checkpoint.
        if not config["training"].get("resume_from") and (output / "last.pt").exists():
            config["training"]["resume_from"] = str(output / "last.pt")
        result = train_task(self.model, self.tokenizer, self.samples(task, "train"), self.samples(task, "tune"),
                            task=task, route=route, seed=seed, output_dir=output, config=config,
                            device=self.device, subspaces=subspaces, provenance=provenance)
        return self._commit_training(result, request_signature, provenance, config, scope, route, task, output)

    def _commit_training(self, result, request_signature, provenance, config, scope, route, task, output):
        """Publish completion last; interrupted export repair never retrains."""
        result = dict(result)
        artifacts = {}
        for field in ("raw_path", "post_path", "initial_path"):
            path = Path(result[field])
            rep = Representation.load(path)
            rep.metadata.update(model_id=config["model"]["id"], model_revision=config["model"].get("revision"),
                                base_fingerprint=provenance["base_fingerprint"], training_request_signature=request_signature,
                                source_ids=[s.id for s in self.samples(task, "train")],
                                source_scope=scope if route == "loda" else "independent",
                                heldout_excluded=scope[4:] if scope.startswith("loo-") else None)
            rep.save(path)
            artifacts[field] = {"path": str(path), "sha256": file_sha256(path),
                                "representation_digest": self.representation_digest(rep)}
        result.update(request_signature=request_signature, completion_schema=2, status="complete", artifacts=artifacts)
        self._verify_training_exports(result, request_signature, provenance["base_fingerprint"])
        atomic_json(output / "training.json", result)
        return result

    def _verify_training_exports(self, result, request_signature, base):
        if result.get("completion_schema") != 2:
            raise ValueError("Training completion lacks transaction/provenance evidence; do not silently reuse legacy artifacts")
        for field in ("raw_path", "post_path", "initial_path"):
            path = Path(result[field])
            expected = result.get("artifacts", {}).get(field, {})
            if not path.is_file() or self._artifact_hash(path) != expected.get("sha256"):
                raise ValueError(f"Incomplete or changed training export: {path}")
            rep = Representation.load(path)
            if rep.metadata.get("base_fingerprint") != base or rep.metadata.get("training_request_signature") != request_signature:
                raise ValueError(f"Training export provenance differs: {path}")

    def _artifact_hash(self, path):
        path = Path(path)
        stat = path.stat()
        key = (str(path.resolve()), stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns)
        if key not in self._artifact_hash_cache:
            self._artifact_hash_cache[key] = file_sha256(path)
        return self._artifact_hash_cache[key]

    def base_fingerprint(self):
        path = self.config["model"].get("path")
        if not path:
            raise ValueError("An explicit local model path is required")
        root = Path(path)
        manifest_path = self.output_root / "base_fingerprint.json"
        source = {str(p.relative_to(root)): {"size": p.stat().st_size, "mtime_ns": p.stat().st_mtime_ns}
                  for p in sorted(root.glob("*")) if p.is_file() and (p.suffix in {".json", ".safetensors", ".bin", ".model", ".txt"})}
        if manifest_path.exists():
            saved = read_json(manifest_path)
            if saved["files"] == source and saved["path"] == str(root.resolve()):
                return saved["fingerprint"]
        hashes = {name: file_sha256(root / name) for name in source}
        fingerprint = object_sha256(hashes)
        atomic_json(manifest_path, {"path": str(root.resolve()), "files": source, "sha256": hashes, "fingerprint": fingerprint})
        return fingerprint

    def representations(self, scope, route, seed, layers=None, version="post"):
        return [self.load_representation(self.directory(scope, route, seed, task, layers) / f"{version}.pt") for task in self.taskset(scope)]

    def load_representation(self, path):
        path = Path(path)
        stat = path.stat()
        absolute = str(path.resolve())
        # Recheck the particular file when it is used; whole registries are
        # checked once at phase entry rather than once per generated panel.
        for _, lock in self._protocol_cache.values():
            expected = lock["discovery_decisions"].get("artifacts", {}).get(absolute)
            if expected and self._artifact_hash(path) != expected["sha256"]:
                raise ValueError(f"Frozen representation artifact changed: {path}")
        key = (absolute, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns)
        if key not in self._representation_cache:
            self._representation_cache[key] = Representation.load(path)
        return self._representation_cache[key]

    def asset_fingerprints(self):
        """Freeze asset contents, rather than only their mutable directory paths."""
        from .metrics import semantic_asset_fingerprint
        from .prompts import canonical_task
        data = self.config["data"]
        tasks = data.get("tasks", []) + data.get("extension_tasks", []) + data.get("auxiliary_tasks", [])
        semantic = semantic_asset_fingerprint(self.config) if any(canonical_task(task) == "paraphrase" for task in tasks) else None
        return {"base": self.base_fingerprint(), "semantic": semantic}

    def lock_path(self, scope):
        return self.output_root / "protocol" / f"{scope}.lock.json"

    def core_lock_path(self, scope, seed):
        return self.output_root / "protocol" / f"{scope}.core-seed{seed}.lock.json"

    def freeze_artifacts(self, scope, seeds=None, decisions=None, layers=None):
        """Freeze actual exports and selected candidate files, not mutable paths."""
        seeds = self.config["analysis"]["seeds"] if seeds is None else seeds
        paths = set()
        for seed in seeds:
            for route in ("ltv", "loda") if layers is None else ("ltv",):
                for task in self.taskset(scope):
                    folder = self.directory(scope, route, seed, task, layers)
                    for name in ("raw.pt", "post.pt", "initial.pt"):
                        paths.add(folder / name)
        def visit(value):
            if isinstance(value, dict):
                if value.get("representation_path"):
                    path = Path(value["representation_path"])
                    if self.representation_digest(Representation.load(path)) != value.get("representation_digest"):
                        raise ValueError(f"Selected candidate differs from discovery: {path}")
                    paths.add(path)
                for item in value.values():
                    visit(item)
            elif isinstance(value, list):
                for item in value:
                    visit(item)
        visit(decisions or {})
        return {str(p.resolve()): {"sha256": self._artifact_hash(p)} for p in sorted(paths)}

    def verify_artifacts(self, artifacts):
        for name, record in artifacts.items():
            path = Path(name)
            if not path.is_file() or self._artifact_hash(path) != record["sha256"]:
                raise ValueError(f"Frozen representation artifact changed before test access: {path}")

    def lock_core(self, scope, seed):
        decisions = {"phase": "core", "seed": seed, "selection_splits": ["train", "tune"],
                     "structure_selection_performed": False, "asset_fingerprints": self.asset_fingerprints(),
                     "artifacts": self.freeze_artifacts(scope, [seed])}
        return create_lock(self.config, self.manifest, decisions, self.core_lock_path(scope, seed), self.source_root)

    def complete_core_stage(self, scope, seed):
        tasks = self.taskset(scope)
        labels = ["base"] + [f"{route}-{task}-primary" for route in ("ltv", "loda") for task in tasks]
        panels = []
        for label in labels:
            for target in tasks:
                folder = self.output_root / "evaluation" / scope / str(seed) / "test" / "canonical" / label / target
                score = read_json(folder / "scores.json")
                expected = len(self.samples(target, "test"))
                if score.get("count") != expected or len(read_jsonl(folder / "predictions.jsonl")) != expected:
                    raise ValueError("First-round core generation panel is incomplete")
                panels.append({"label": label, "task": target, "count": expected,
                               "scores_sha256": self._artifact_hash(folder / "scores.json"),
                               "predictions_sha256": self._artifact_hash(folder / "predictions.jsonl"),
                               "newly_generated": score.get("newly_generated", expected)})
        lock = self.verify_evaluation_protocol(scope, seed)
        result = {"completed": True, "scope": scope, "seed": seed, "split": "test", "phase": "core",
                  "protocol_lock": lock["lock_sha256"], "panels": panels,
                  "core_states": len(labels), "core_generations": sum(p["count"] for p in panels),
                  "extra_E1_controls_and_tune_discovery_excluded": True,
                  "structure_selection_performed": False}
        if scope == "mvp" and seed == 17 and len(tasks) == 4 and not self.config.get("supplement_label"):
            if result["core_generations"] != 9216:
                raise ValueError("The documented first round requires exactly 9216 core predictions")
        destination = self.output_root / "analysis" / scope / str(seed) / "core-test-completed.json"
        atomic_json(destination, result)
        if scope == "mvp" and seed == 17:
            atomic_json(self.output_root / "analysis" / scope / "first-round-completed.json", result)
        return result

    @contextmanager
    def protocol_phase(self, phase):
        previous = self._active_protocol_phase
        self._active_protocol_phase = phase
        try:
            yield
        finally:
            self._active_protocol_phase = previous

    def verify_evaluation_protocol(self, scope, seed):
        phase = self._active_protocol_phase
        path = self.core_lock_path(scope, seed) if phase == "core" else self.lock_path(scope)
        return self.verify_frozen_protocol(path)

    def verify_frozen_protocol(self, path):
        """Verify main, core or control locks through the same cached path."""
        path = Path(path)
        stat = path.stat()
        assets = self.asset_fingerprints()
        key = (stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns,
               object_sha256(scientific_config(self.config)), object_sha256(self.manifest),
               object_sha256(self.config.get("acceleration", {})),
               object_sha256(source_manifest(self.source_root)), object_sha256(assets))
        cached = self._protocol_cache.get(str(path))
        if cached and cached[0] == key:
            return cached[1]
        lock = verify_lock(path, self.config, self.manifest, self.source_root)
        decisions = lock["discovery_decisions"]
        if decisions.get("asset_fingerprints") != assets:
            raise ValueError("Model or semantic metric assets differ from the frozen protocol; use a new run")
        self.verify_artifacts(decisions.get("artifacts", {}))
        self._protocol_cache[str(path)] = (key, lock)
        return lock

    def evaluate(self, representation, task, split, *, label, scope="mvp", seed=17, variant="canonical", samples=None, config=None):
        from .evaluation import generate_predictions
        from .metrics import score_predictions
        cfg = config or self.config
        protocol_lock = None
        current_base = self.base_fingerprint()
        if representation is not None and representation.metadata.get("base_fingerprint") != current_base:
            raise ValueError("Representation base fingerprint differs from the current model or is missing")
        if split.startswith("test"):
            protocol_lock = self.verify_evaluation_protocol(scope, seed)
        panel = self.samples(task, split) if samples is None else samples
        if protocol_lock is not None:
            self.record_final_access(task, split, panel, scope=scope, seed=seed, label=label,
                                     variant=variant, lock_hash=protocol_lock["lock_sha256"])
        from .metrics import semantic_asset_fingerprint
        semantic_used = task == "paraphrase" or (task == "joint" and any("paraphrase" in sample.metadata.get("pair", []) for sample in panel))
        semantic_asset = semantic_asset_fingerprint(cfg) if semantic_used else None
        generation_provenance = {"config": object_sha256(scientific_config(cfg)), "base": current_base,
                                 "representation": None if representation is None else self.representation_digest(representation),
                                 "source": object_sha256(source_manifest(self.source_root)),
                                 "data": object_sha256(self.manifest), "semantic_asset": semantic_asset}
        acceleration = cfg.get("acceleration", {})
        parallel = bool(acceleration.get("enabled", False))
        if parallel:
            generation_provenance["execution"] = object_sha256(acceleration)
        sample_digests = {s.id: object_sha256(s.to_dict()) for s in panel}
        signature = object_sha256({"config": scientific_config(cfg), "task": task, "split": split, "variant": variant,
                                   "generation_provenance": generation_provenance,
                                   "sample_ids": [s.id for s in panel], "base": current_base,
                                   "sample_contents": [s.to_dict() for s in panel],
                                   "semantic_asset": semantic_asset,
                                   "representation": generation_provenance["representation"]})
        folder = self.output_root / "evaluation" / scope / str(seed) / split / variant / label / task
        if (folder / "scores.json").exists():
            saved = read_json(folder / "scores.json")
            if saved.get("signature") == signature:
                saved["rows"] = read_jsonl(folder / "predictions.jsonl")
                return saved
            raise ValueError(f"Stale evaluation artifact: {folder}; use a new output root")
        # A formal superset reuses identical prior predictions, generating only
        # unseen IDs. No scoring or selection choices are changed by this cache.
        reused = {}
        if split == "test_formal":
            for old_split in ("test",):
                old = self.output_root / "evaluation" / scope / str(seed) / old_split / variant / label / task
                if (old / "scores.json").is_file():
                    old_score = read_json(old / "scores.json")
                    provenance = old_score.get("generation_provenance", {})
                    if provenance == generation_provenance:
                        old_digests = old_score.get("generation_sample_digests", {})
                        reused.update({r["id"]: r for r in read_jsonl(old / "predictions.jsonl")
                                       if r["id"] in sample_digests and old_digests.get(r["id"]) == sample_digests[r["id"]]})
        remaining = [s for s in panel if s.id not in reused]
        if remaining and not parallel:
            self.load_model()
        started = time.perf_counter()
        execution_receipt = None
        if remaining and parallel:
            calibration, allowed = self._calibration_samples(task, scope)
            generated, execution_receipt = self._parallel_generation().generate(
                remaining, representation, cfg, task=task, variant=variant,
                representation_digest=generation_provenance["representation"],
                panel_label=f"{scope}/{seed}/{split}/{variant}/{label}/{task}",
                calibration_samples=calibration, allow_calibration=allowed)
        elif remaining:
            generated = generate_predictions(self.model, self.tokenizer, remaining, representation, cfg, self.device, variant=variant)
        else:
            generated = []
        mapping = {**reused, **{r["id"]: r for r in generated}}
        if len(mapping) != len(panel) or set(mapping) != {s.id for s in panel}:
            raise ValueError("Evaluation predictions do not cover each sample exactly once")
        rows = [mapping[s.id] for s in panel]
        scored = score_predictions(task, rows, cfg)
        synchronize(self.device)
        scored.update(signature=signature, label=label, task=task, split=split, variant=variant,
                      scope=scope, seed=seed, runtime_seconds=time.perf_counter() - started,
                      memory=memory_evidence(self.device), newly_generated=len(generated), reused_predictions=len(panel)-len(generated),
                      generation_provenance=generation_provenance, generation_sample_digests=sample_digests)
        if parallel:
            scored["execution_receipt"] = execution_receipt
        write_jsonl(folder / "predictions.jsonl", scored["rows"])
        atomic_json(folder / "scores.json", {k: v for k, v in scored.items() if k != "rows"})
        return scored

    def record_final_access(self, task, split, samples, *, scope, seed, label, variant, lock_hash):
        """Audit every exposure, including cached reads, without claiming new data."""
        from datetime import datetime, timezone
        from filelock import FileLock
        path = self.output_root / "test_access_ledger.json"
        event = {"utc_time": datetime.now(timezone.utc).isoformat(), "scope": scope, "seed": seed,
                 "label": label, "variant": variant, "task": task, "split": split,
                 "sample_count": len(samples), "sample_ids_sha256": object_sha256([sample.id for sample in samples]),
                 "group_ids_sha256": object_sha256([sample.group_id for sample in samples]),
                 "protocol_lock_sha256": lock_hash,
                 "independence": "separately reserved group pool" if split == "test_confirmation" else "not assumed; formal supersets can include previously exposed test samples"}
        with FileLock(str(path) + ".lock", timeout=60):
            ledger = read_json(path) if path.exists() else {"schema_version": 1, "events": []}
            ledger["events"].append(event)
            atomic_json(path, ledger)

    @staticmethod
    def representation_digest(rep):
        # Tensor bytes, rather than nondeterministic pickle serialization.
        import hashlib
        digest = hashlib.sha256()
        digest.update(object_sha256({"space": rep.space, "keys": rep.keys}).encode())
        for key in rep.keys:
            tensors = [rep.values[key]] if rep.space == "activation" else [v for f in rep.factors[key] for v in (f.B, f.A)]
            for tensor in tensors:
                digest.update(str(tuple(tensor.shape)).encode())
                digest.update(tensor.detach().float().cpu().contiguous().numpy().tobytes())
        return digest.hexdigest()

    def delta(self, condition, base, task, seed, *, config=None):
        from .metrics import aggregate_predictions
        acceleration = self.config.get("acceleration", {})
        if acceleration.get("enabled", False):
            from .bootstrap import accelerated_paired_bootstrap
            engineering = {"task": task, "seed": seed, "condition": condition.get("label"),
                           "base": base.get("label"), "status": "running"}
            progress_path = self.output_root / "engineering/bootstrap-progress.json"
            atomic_json(progress_path, engineering)
            try:
                result = accelerated_paired_bootstrap(task, condition["rows"], base["rows"],
                    config or self.config,
                    repetitions=self.config["evaluation"]["bootstrap_samples"],
                    confidence=self.config["evaluation"]["confidence_level"], seed=seed,
                    workers=int(acceleration.get("bootstrap_workers", 1)),
                    engineering_record=engineering)
            except BaseException:
                engineering["status"] = "failed"
                atomic_json(progress_path, engineering)
                raise
            engineering["status"] = "completed"
            atomic_json(progress_path, engineering)
            return result
        return paired_bootstrap(condition["rows"], base["rows"],
                                lambda rows: aggregate_predictions(task, rows, config or self.config)["metrics"],
                                repetitions=self.config["evaluation"]["bootstrap_samples"],
                                confidence=self.config["evaluation"]["confidence_level"], seed=seed)

    def core_panel(self, scope, route, seed, split="discovery", layers=None, variant="canonical"):
        tasks = self.taskset(scope)
        reps = self.representations(scope, route, seed, layers)
        tag = route if not layers or layers == [14] else route + "-layers-" + "-".join(map(str, layers))
        base = {t: self.evaluate(None, t, split, label="base", scope=scope, seed=seed, variant=variant) for t in tasks}
        scores, effects, e1 = {}, {}, {}
        for rep in reps:
            source = rep.task
            scores[source], effects[source] = {}, {}
            for target in tasks:
                condition = self.evaluate(rep, target, split, label=f"{tag}-{source}-primary", scope=scope, seed=seed, variant=variant)
                scores[source][target] = condition["metrics"][condition["primary_metric"]]
                effects[source][target] = self.delta(condition, base[target], target, seed)
            own = effects[source][source][base[source]["primary_metric"]]
            controls = {}
            initial = self.load_representation(self.directory(scope, route, seed, source, layers) / "initial.pt")
            controls["initial"] = self.evaluate(initial, source, split, label=f"{tag}-{source}-initial", scope=scope, seed=seed, variant=variant)
            for repeat in range(self.config["evaluation"]["random_repeats"]):
                rnd = random_like(rep, seed + 70001 + repeat)
                controls[f"random-{repeat}"] = self.evaluate(rnd, source, split, label=f"{tag}-{source}-random-{repeat}", scope=scope, seed=seed, variant=variant)
            e1[source] = {"own_task": own, "primary_metric": base[source]["primary_metric"], "controls": {k: {"metrics": v["metrics"], "delta": self.delta(v, base[source], source, seed)} for k, v in controls.items()},
                          "initial_norm": norm(initial), "trained_norm": norm(rep)}
        matrix = transfer_matrix(tasks, scores, {t: base[t]["metrics"][base[t]["primary_metric"]] for t in tasks}, {t: base[t]["sample_scores"] for t in tasks})
        result = {"route": route, "scope": scope, "seed": seed, "split": split, "variant": variant,
                  "matrix": matrix, "effects": effects, "E1": e1, "primary_strength": 1.0}
        atomic_json(self.output_root / "analysis" / scope / str(seed) / f"{tag}-{split}-{variant}-core.json", result)
        return result

    def tune_strengths(self, scope, route, seed):
        chosen = {}
        for rep in self.representations(scope, route, seed):
            task, candidates = rep.task, []
            for i, strength in enumerate(self.config["analysis"]["strength_grid"]):
                score = self.evaluate(scaled(rep, strength), task, "tune", label=f"{route}-{task}-strength-{i}", scope=scope, seed=seed)
                candidates.append({"strength": strength, "score": score["metrics"][score["primary_metric"]]})
            chosen[task] = {"selected": max(candidates, key=lambda x: x["score"]), "candidates": candidates, "selection_split": "tune", "budget": len(candidates)}
        return chosen

    def geometry(self, scope, seed):
        results = {}
        for route in ("ltv", "loda"):
            reps = self.representations(scope, route, seed)
            initial = [self.load_representation(self.directory(scope, route, seed, task) / "initial.pt") for task in self.taskset(scope)]
            cos = cosine_matrix(reps)
            results[route] = {"tasks": self.taskset(scope), "cosine": _finite_list(cos), "initial_cosine": _finite_list(cosine_matrix(initial)),
                              "trained_vs_initial_relation": relation_consistency(cos, cosine_matrix(initial))}
            if route == "loda":
                results[route]["layer_angles"] = {f"{a.task}--{b.task}": layer_subspace_angles(a, b) for a, b in combinations(reps, 2)}
        results["cross_space"] = relation_consistency(cosine_matrix(self.representations(scope, "ltv", seed)), cosine_matrix(self.representations(scope, "loda", seed)))
        atomic_json(self.output_root / "analysis" / scope / str(seed) / "geometry.json", results)
        return results

    def component_panel(self, scope, route, seed, split="discovery", selected=None, variant="canonical"):
        if selected is None and split != "discovery":
            raise ValueError("Component selection is restricted to the structure discovery split")
        reps, tasks = self.representations(scope, route, seed), self.taskset(scope)
        candidates = [("mean", 0)] + [("svd", q) for q in self.config["analysis"]["ranks"]] if selected is None else [(selected["method"], selected["q"])]
        base = {t: self.evaluate(None, t, split, label="base", scope=scope, seed=seed, variant=variant) for t in tasks}
        records, details = [], {}
        for method, q in candidates:
            decomposition = decompose(reps, method, q)
            nulls = random_subspace_control(reps, decomposition, 1 if method == "mean" else q, seed + 80009)
            gains, results = {}, {}
            # Global shared mean is evaluated on ALL targets, while each s_t/r_t
            # is also evaluated directionally to retain task-conditioned evidence.
            global_shared = linear_combination(decomposition.shared, [1 / len(reps)] * len(reps), "global_shared")
            for target in tasks:
                condition = self.evaluate(global_shared, target, split, label=f"{route}-{method}-q{q}-global-shared", scope=scope, seed=seed, variant=variant)
                gains[target] = condition["metrics"][condition["primary_metric"]] - base[target]["metrics"][base[target]["primary_metric"]]
            for i, source in enumerate(tasks):
                components = {"shared": decomposition.shared[i], "residual": decomposition.residual[i],
                              "reconstruction": linear_combination([decomposition.shared[i], decomposition.residual[i]], [1, 1]), "random_shared": nulls[i]}
                results[source] = {}
                scored_components = {}
                for name, representation in components.items():
                    results[source][name] = {}
                    scored_components[name] = {}
                    for target in tasks:
                        evaluated = self.evaluate(representation, target, split, label=f"{route}-{method}-q{q}-{source}-{name}", scope=scope, seed=seed, variant=variant)
                        scored_components[name][target] = evaluated
                        results[source][name][target] = {"metrics": evaluated["metrics"], "effect": self.delta(evaluated, base[target], target, seed)}
                results[source]["shared_minus_random_shared"] = {
                    target: self.delta(scored_components["shared"][target], scored_components["random_shared"][target], target, seed)
                    for target in tasks}
                results[source]["delete_shared"] = {"alias_of": "residual", "reason": "v_t - s_t = r_t; no independent evidence"}
            key = f"{method}-q{q}"
            details[key] = {"metadata": decomposition.metadata, "projection_weights": decomposition.weights.tolist(), "results": results}
            records.append({"method": method, "q": q, "gains": gains})
        stds = {t: float(np.std(base[t]["sample_scores"], ddof=1)) for t in tasks}
        if selected is None:
            if split != "discovery":
                raise ValueError("Component selection is restricted to the structure discovery split")
            choice, ranked = choose_discovery_candidate(records, tasks, stds)
        else:
            choice, ranked = dict(selected), records
        output = {"selected": choice, "candidates": ranked, "details": details,
                  "selection_split": "discovery", "evaluation_split": split,
                  "locked_choice_used": selected is not None,
                  "selection_objective": "mean source primary gain divided by source baseline sample std; raw units separately retained"}
        atomic_json(self.output_root / "analysis" / scope / str(seed) / f"{route}-{split}-{variant}-components.json", output)
        return output

    def functional_clustering(self, scope, route, seed, core):
        from .metrics import aggregate_predictions
        tasks, reps = self.taskset(scope), self.representations(scope, route, seed)
        base = {t: self.evaluate(None, t, "discovery", label="base", scope=scope, seed=seed) for t in tasks}
        predictions = {rep.task: {t: self.evaluate(rep, t, "discovery", label=f"{route}-{rep.task}-primary", scope=scope, seed=seed) for t in tasks} for rep in reps}
        rng = np.random.default_rng(seed + 91007)
        matrices = []
        std = np.array(core["matrix"]["column_std"])
        for _ in range(self.config["analysis"]["cluster_bootstrap_samples"]):
            matrix = np.zeros((len(tasks), len(tasks)))
            for j, target in enumerate(tasks):
                rows = base[target]["rows"]
                groups = {}
                for index, row in enumerate(rows):
                    groups.setdefault(row.get("group_id", row["id"]), []).append(index)
                grouped = list(groups.values())
                indices = [i for g in rng.integers(0, len(grouped), len(grouped)) for i in grouped[g]]
                base_metric = aggregate_predictions(target, [rows[i] for i in indices], self.config)
                primary = base_metric["primary_metric"]
                for i, source in enumerate(tasks):
                    mapping = {r["id"]: r for r in predictions[source][target]["rows"]}
                    sampled = [mapping[rows[k]["id"]] for k in indices]
                    value = aggregate_predictions(target, sampled, self.config)["metrics"][primary] - base_metric["metrics"][primary]
                    matrix[i, j] = value / std[j] if std[j] > 1e-12 else 0.
            matrices.append(matrix)
        result = discover_groups(core["matrix"], bootstrap_matrices=matrices,
                                 min_stability=self.config["analysis"]["cluster_min_stability"],
                                 max_groups=self.config["analysis"]["cluster_max_groups"])
        atomic_json(self.output_root / "analysis" / scope / str(seed) / f"{route}-clusters.json", result)
        return result

    def composition_candidates(self, scope, route, seed, decomposition_choice):
        reps = self.representations(scope, route, seed)
        tasks = self.taskset(scope)
        decomposition = decompose(reps, decomposition_choice["method"], decomposition_choice["q"])
        choices, panel = {}, {}
        base = {t: self.evaluate(None, t, "tune", label="base", scope=scope, seed=seed) for t in tasks}
        stds = {t: float(np.std(base[t]["sample_scores"], ddof=1)) for t in tasks}
        for left, right in combinations(range(len(tasks)), 2):
            pair = [tasks[left], tasks[right]]
            pair_key = "--".join(pair)
            vectors = [reps[left], reps[right]]
            mean = linear_combination(vectors, [.5, .5], "mean")
            primitives = {"original-a": vectors[0], "original-b": vectors[1],
                          "sum": linear_combination(vectors, [1, 1], "sum"), "mean": mean,
                          "decomposed": decomposition.compose([left, right])}
            if route == "loda":
                from .baselines import merge_baseline
                for baseline in self.config["analysis"]["baselines"]:
                    kwargs = {"keep_fraction": self.config["analysis"]["ties_density"]} if baseline in {"ties", "knots"} else {}
                    primitives[baseline] = merge_baseline(vectors, baseline, **kwargs)
            strategies = {}
            for strategy, rep in primitives.items():
                strategies[strategy] = [(scaled(rep, strength), {"strength": strength}) for strength in self.config["analysis"]["strength_grid"]]
            strategies["weighted"] = [(linear_combination(vectors, weights, "weighted"), {"weights": weights}) for weights in self.config["analysis"]["pair_weight_grid"]]
            budgets = {len(v) for v in strategies.values()}
            if len(budgets) != 1:
                raise ValueError("Every E5 strategy must receive exactly the same candidate budget")
            choices[pair_key], panel[pair_key] = {}, {}
            for norm_condition in ("raw", "matched"):
                choices[pair_key][norm_condition], panel[pair_key][norm_condition] = {}, {}
                for strategy, candidates in strategies.items():
                    records, evaluated_representations = [], []
                    for index, (rep, parameters) in enumerate(candidates):
                        candidate = rep
                        # Zero scale is a genuine BASE control, not an impossible zero-to-nonzero norm match.
                        if norm_condition == "matched" and norm(candidate) > 1e-12:
                            candidate = match_norm(candidate, mean)
                        gains, metrics = {}, {}
                        for target in pair:
                            score = self.evaluate(candidate, target, "tune", label=f"{route}-compose-{pair_key}-{norm_condition}-{strategy}-{index}", scope=scope, seed=seed)
                            metrics[target] = score["metrics"]
                            gains[target] = score["metrics"][score["primary_metric"]] - base[target]["metrics"][base[target]["primary_metric"]]
                        candidate_path = self.output_root / "compositions" / scope / str(seed) / route / pair_key / norm_condition / strategy / f"candidate{index}.pt"
                        candidate.save(candidate_path)
                        evaluated_representations.append(candidate)
                        records.append({"index": index, "parameters": parameters, "gains": gains, "metrics": metrics,
                                        "representation_path": str(candidate_path), "norm": norm(candidate),
                                        "representation_digest": self.representation_digest(candidate)})
                    selected, ranked = choose_discovery_candidate(records, pair, stds)
                    selected = {**selected, "nominal_candidate_budget": len(records),
                                "distinct_coordinate_candidates": _distinct_represented_count(evaluated_representations),
                                "coordinate_equality_relative_tolerance": 1e-5,
                                "norm_condition": norm_condition,
                                "budget_note": "positive scalar choices may coincide after fixed-norm matching; report distinct candidates explicitly"}
                    choices[pair_key][norm_condition][strategy] = selected
                    panel[pair_key][norm_condition][strategy] = ranked
            # Same-norm sum/mean collapse is explicitly retained as a control result.
            choices[pair_key]["degeneracy"] = {"matched_sum_equals_mean_at_positive_scale": True,
                                               "mean_decomposition_full_group_equals_mean": decomposition.metadata.get("full_group_composition_degenerates_to_mean", False),
                                               "candidate_budget_per_strategy": next(iter(budgets))}
        return {"choices": choices, "tune_panel": panel, "selection_split": "tune", "norm_reference": "pair mean", "joint_not_used_for_selection": True}

    def composition_final(self, scope, route, seed, choices, thresholds, split="test"):
        from .joint import build_joint_samples
        cfg = deepcopy(self.config)
        cfg["joint"] = {"thresholds": thresholds}
        results = {}
        for pair_key, conditions in choices.items():
            pair = pair_key.split("--")
            panel = {t: self.samples(t, split) for t in pair}
            count = min(self.config["analysis"]["joint_count"], *(len(v) for v in panel.values()))
            joint_samples = build_joint_samples(panel, seed, count)
            base = {t: self.evaluate(None, t, split, label="base", scope=scope, seed=seed) for t in pair}
            joint_base = self.evaluate(None, "joint", split, label=f"compose-{pair_key}-base", scope=scope, seed=seed, samples=joint_samples, config=cfg)
            results[pair_key] = {"joint_base": _score_summary(joint_base), "conditions": {}, "degeneracy": conditions["degeneracy"], "joint_count": count}
            for norm_condition in ("raw", "matched"):
                results[pair_key]["conditions"][norm_condition] = {}
                for strategy, choice in conditions[norm_condition].items():
                    rep = self.load_representation(choice["representation_path"])
                    if self.representation_digest(rep) != choice.get("representation_digest"):
                        raise ValueError("Selected composition representation differs from its frozen discovery digest")
                    original_panel = {}
                    for target in pair:
                        score = self.evaluate(rep, target, split, label=f"{route}-compose-{pair_key}-{norm_condition}-{strategy}", scope=scope, seed=seed)
                        original_panel[target] = {"metrics": score["metrics"], "effect": self.delta(score, base[target], target, seed)}
                    joint = self.evaluate(rep, "joint", split, label=f"{route}-compose-{pair_key}-{norm_condition}-{strategy}", scope=scope, seed=seed, samples=joint_samples, config=cfg)
                    results[pair_key]["conditions"][norm_condition][strategy] = {"locked_choice": choice, "original_task_panel": original_panel,
                                                                                  "joint": _score_summary(joint), "joint_effect": self.delta(joint, joint_base, "joint", seed, config=cfg)}
        atomic_json(self.output_root / "analysis" / scope / str(seed) / f"{route}-{split}-composition.json", results)
        return results

    def branch_panel(self, scope, seed, split="test"):
        results = {}
        tasks = self.taskset(scope)
        base = {t: self.evaluate(None, t, split, label="base", scope=scope, seed=seed) for t in tasks}
        for version in ("raw", "post"):
            for rep in self.representations(scope, "loda", seed, version=version):
                results[f"{rep.task}-{version}"] = {}
                for branch, index in (("general", 0), ("isolated", 1)):
                    component = Representation("parameter", f"{rep.task}:{branch}", seed,
                                               factors={k: [terms[index]] for k, terms in rep.factors.items()}, metadata={**rep.metadata, "branches": [branch]})
                    results[f"{rep.task}-{version}"][branch] = {}
                    for target in tasks:
                        score = self.evaluate(component, target, split, label=f"loda-{rep.task}-{version}-{branch}", scope=scope, seed=seed)
                        results[f"{rep.task}-{version}"][branch][target] = {"metrics": score["metrics"], "effect": self.delta(score, base[target], target, seed)}
                results[f"{rep.task}-{version}"]["full"] = {}
                for target in tasks:
                    score = self.evaluate(rep, target, split, label=f"loda-{rep.task}-{version}-full", scope=scope, seed=seed)
                    results[f"{rep.task}-{version}"]["full"][target] = {"metrics": score["metrics"], "effect": self.delta(score, base[target], target, seed)}
        atomic_json(self.output_root / "analysis" / scope / str(seed) / f"loda-{split}-branches.json", results)
        return results

    def heldout_final(self, heldout, seed, decisions, split="test"):
        scope = f"loo-{heldout}"
        seen = self.taskset(scope)
        output = {"heldout": heldout, "seen_tasks": seen, "source_only_decisions": decisions,
                  "audit": {"heldout_train_tune_discovery_used": False, "LoDA_rebuilt_and_retrained": True}, "routes": {}}
        base = self.evaluate(None, heldout, split, label="base", scope=scope, seed=seed)
        for route in ("ltv", "loda"):
            reps = self.representations(scope, route, seed)
            choice = decisions[str(seed)][route]["decomposition"]
            decomposition = decompose(reps, choice["method"], choice["q"])
            candidates = {"source_mean": linear_combination(reps, [1 / len(reps)] * len(reps)),
                          "source_sum": linear_combination(reps, [1] * len(reps)),
                          "global_shared": linear_combination(decomposition.shared, [1 / len(reps)] * len(reps)),
                          "composed_sources": decomposition.compose()}
            output["routes"][route] = {}
            for name, rep in candidates.items():
                score = self.evaluate(rep, heldout, split, label=f"{route}-heldout-{name}", scope=scope, seed=seed)
                output["routes"][route][name] = {"metrics": score["metrics"], "effect": self.delta(score, base, heldout, seed)}
        atomic_json(self.output_root / "analysis" / scope / str(seed) / f"{split}-heldout.json", output)
        return output

    def summary_final(self, scope, seed, decisions, split="test"):
        base = self.evaluate(None, "summary", split, label="base", scope=scope, seed=seed)
        results = {}
        for route in ("ltv", "loda"):
            reps = self.representations(scope, route, seed)
            choice = decisions[str(seed)][route]["decomposition"]
            decomposition = decompose(reps, choice["method"], choice["q"])
            conditions = {"mean": linear_combination(reps, [1 / len(reps)] * len(reps)),
                          "shared": linear_combination(decomposition.shared, [1 / len(reps)] * len(reps)),
                          "composed": decomposition.compose()}
            results[route] = {}
            for name, rep in conditions.items():
                score = self.evaluate(rep, "summary", split, label=f"{route}-summary-{name}", scope=scope, seed=seed)
                results[route][name] = {"metrics": score["metrics"], "effect": self.delta(score, base, "summary", seed)}
        atomic_json(self.output_root / "analysis" / scope / str(seed) / f"{split}-unseen-summary.json", results)
        return results

    def group_final(self, scope, route, seed, decisions, split="test"):
        reps, tasks = self.representations(scope, route, seed), self.taskset(scope)
        selected = decisions["decomposition"]
        global_decomp = decompose(reps, selected["method"], selected["q"])
        base = {t: self.evaluate(None, t, split, label="base", scope=scope, seed=seed) for t in tasks}
        results = {}
        for name, settings in decisions["group_residual_svd"].items():
            indices = [tasks.index(t) for t in settings["tasks"]]
            sources = [global_decomp.residual[i] for i in indices]
            q = min(settings["selected"]["q"], len(indices))
            group_decomp = decompose(sources, "svd", q)
            nulls = random_subspace_control(sources, group_decomp, q, seed + 99013 + int(name))
            results[name] = {"source_tasks": settings["tasks"], "q": q, "components": {}}
            for local, source_index in enumerate(indices):
                task = tasks[source_index]
                components = {"global": global_decomp.shared[source_index], "group": group_decomp.shared[local],
                              "specific": group_decomp.residual[local], "random_group": nulls[local],
                              "reconstruction": linear_combination([global_decomp.shared[source_index], group_decomp.shared[local], group_decomp.residual[local]], [1, 1, 1])}
                results[name]["components"][task] = {}
                scored_components = {}
                for component, rep in components.items():
                    results[name]["components"][task][component] = {}
                    scored_components[component] = {}
                    for target in tasks:
                        score = self.evaluate(rep, target, split, label=f"{route}-group{name}-{task}-{component}", scope=scope, seed=seed)
                        scored_components[component][target] = score
                        results[name]["components"][task][component][target] = {"metrics": score["metrics"], "effect": self.delta(score, base[target], target, seed)}
                results[name]["components"][task]["group_minus_random_group"] = {
                    target: self.delta(scored_components["group"][target], scored_components["random_group"][target], target, seed)
                    for target in tasks}
        atomic_json(self.output_root / "analysis" / scope / str(seed) / f"{route}-{split}-group-components.json", results)
        return results


def _score_summary(score):
    return {key: value for key, value in score.items() if key not in {"rows", "sample_scores"}}


def _distinct_represented_count(representations, tolerance=1e-5):
    """Count actual update directions, independent of low-rank factor encoding."""
    unique = []
    for candidate in representations:
        if not any(norm(linear_combination([candidate, existing], [1., -1.])) <=
                   tolerance * max(norm(candidate), norm(existing), 1e-12) for existing in unique):
            unique.append(candidate)
    return len(unique)


def _finite_list(matrix):
    return [[float(x) if np.isfinite(x) else None for x in row] for row in np.asarray(matrix)]
