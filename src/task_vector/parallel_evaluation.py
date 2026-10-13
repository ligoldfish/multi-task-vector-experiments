"""Persistent independent generation workers with coordinator-only publication.

Workers never score, choose scientific conditions, read test locks, or publish a
canonical evaluation.  They execute one already-authorized panel, and the parent
waits for every shard before returning rows in the original sample order.
"""
from __future__ import annotations

import atexit
from contextlib import contextmanager, nullcontext
from copy import deepcopy
from datetime import datetime, timezone
import multiprocessing
import os
from pathlib import Path
import queue
import time
import traceback
import uuid
import weakref

from .io import atomic_json, file_sha256, object_sha256, read_json


_MERGED_MODEL_CACHE = weakref.WeakKeyDictionary()


@contextmanager
def merged_projection_context(model, path, expected_sha256):
    """Use saved native projection weights, restoring exact base tensors."""
    import torch
    path = Path(path).resolve()
    stat = path.stat()
    key = (str(path), expected_sha256, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns)
    cached = _MERGED_MODEL_CACHE.get(model, {}) if model is not None else {}
    if not expected_sha256 or (cached.get("key") != key and file_sha256(path) != expected_sha256):
        raise ValueError("Sequential merged projection artifact hash differs")
    if cached.get("key") == key:
        weights, original = cached["weights"], cached["original"]
    else:
        weights = torch.load(path, map_location="cpu", weights_only=True)
        after = path.stat()
        if (after.st_size, after.st_mtime_ns, after.st_ctime_ns) != key[2:]:
            raise ValueError("Sequential merged projection artifact changed during verification")
        names = {name for name, module in model.named_modules()
                 if name.endswith((".q_proj", ".k_proj", ".v_proj")) and hasattr(module, "weight")}
        if not names or set(weights) != names:
            raise ValueError("Sequential merged weights must cover exactly all base QKV projections")
        original = cached.get("original", {})
        for name in names:
            parameter = model.get_submodule(name).weight
            value = weights[name]
            if value.shape != parameter.shape or value.dtype != parameter.dtype:
                raise ValueError(f"Sequential native projection shape/dtype differs: {name}")
            if name not in original:
                original[name] = parameter.detach().cpu().clone()
        # One stage's weights per live model, with a single frozen-base copy.
        _MERGED_MODEL_CACHE[model] = {"key": key, "weights": weights, "original": original}
    try:
        with torch.no_grad():
            for name, value in weights.items():
                model.get_submodule(name).weight.copy_(value.to(model.get_submodule(name).weight.device))
        yield
    finally:
        with torch.no_grad():
            for name, value in original.items():
                model.get_submodule(name).weight.copy_(value.to(model.get_submodule(name).weight.device))


def partition_samples(samples, workers):
    """Contiguous, balanced, nonempty shards; never drop a remainder."""
    if workers < 1:
        raise ValueError("Need at least one generation worker")
    ids = [str(sample.id) for sample in samples]
    if len(set(ids)) != len(ids):
        raise ValueError("A generation panel must contain unique sample IDs")
    quotient, remainder = divmod(len(samples), min(workers, len(samples)) or 1)
    shards, offset = [], 0
    for rank in range(min(workers, len(samples))):
        size = quotient + int(rank < remainder)
        shards.append(samples[offset:offset + size])
        offset += size
    return shards


def merge_shards(samples, shard_rows):
    """Fail closed on duplicate, missing, extra, or misassigned predictions."""
    expected = [str(sample.id) for sample in samples]
    if len(set(expected)) != len(expected):
        raise ValueError("A generation panel must contain unique sample IDs")
    rows = {}
    for shard in shard_rows:
        for row in shard:
            identity = str(row["id"])
            if identity in rows:
                raise ValueError(f"Duplicate generated prediction: {identity}")
            rows[identity] = row
    if set(rows) != set(expected):
        raise ValueError("Generation shards do not exactly cover the authorized panel")
    return [rows[identity] for identity in expected]


def _utc():
    return datetime.now(timezone.utc).isoformat()


def _worker_main(rank, name, config, request_queue, result_queue, directory):
    """Spawn entry: no accelerator tensors ever cross a multiprocessing queue."""
    path = Path(directory)
    progress_path = path / f"worker-{rank}.json"
    base = {"rank": rank, "device": name, "pid": os.getpid()}
    try:
        threads = int(config.get("acceleration", {}).get("worker_cpu_threads", 2))
        for variable in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
            os.environ[variable] = str(threads)
        import torch
        torch.set_num_threads(threads)
        from .runtime import configure_device, memory_evidence, synchronize
        from .modeling import load_model
        from .data import Sample
        from .representations import Representation
        from .batching import calibrate_generation_batches, generate_with_batch_policy

        atomic_json(progress_path, {**base, "status": "loading", "updated_utc": _utc()})
        local_config = deepcopy(config)
        local_config["platform"]["device"] = name
        device = configure_device(name)
        model, tokenizer = load_model(local_config, device)
        atomic_json(progress_path, {**base, "status": "idle", "updated_utc": _utc()})
        result_queue.put({"kind": "ready", "rank": rank})
        while True:
            request = request_queue.get()
            if request is None:
                break
            started = time.perf_counter()
            last_update = 0.0
            operation = request["operation"]
            cfg = request["config"]
            cfg["platform"]["device"] = name
            stage = request["kind"]
            progress_state = {"completed_samples": 0,
                              "total_samples": len(request.get("samples", []))}

            def progress(event):
                nonlocal last_update
                now = time.perf_counter()
                progress_state.update(event)
                # Shared DPC metadata is not written once per token.
                finished = event.get("phase") == "generation_completed"
                if now - last_update < 5 and not finished:
                    return
                last_update = now
                atomic_json(progress_path, {**base, **progress_state, "operation": operation,
                    "panel": request.get("panel"), "status": stage,
                    "elapsed_seconds": now - started, "updated_utc": _utc()})

            progress({"stage": "start", "completed_samples": 0,
                      "total_samples": len(request.get("samples", []))})
            if stage == "calibrate":
                samples = [Sample.from_dict(row) for row in request["samples"]]
                accel = cfg.get("acceleration", {})
                result = calibrate_generation_batches(model, tokenizer, samples, cfg, device,
                    task=request["task"], variant=request["variant"], source_split="tune",
                    candidates=tuple(accel.get("generation_batch_candidates", [1, 2, 4, 8])),
                    mode=accel.get("generation_batch_policy", "strict"),
                    max_samples=int(accel.get("calibration_max_samples", 8)),
                    progress_callback=progress)
                result_queue.put({"kind": "result", "operation": operation, "rank": rank,
                                  "calibration": result})
            elif stage == "generate":
                samples = [Sample.from_dict(row) for row in request["samples"]]
                representation = (Representation.load(request["representation_path"])
                                  if request.get("representation_path") else None)
                context = (merged_projection_context(model, request["merged_weights_path"],
                    request["merged_weights_sha256"]) if request.get("merged_weights_path") else nullcontext())
                with context:
                    rows, receipt = generate_with_batch_policy(model, tokenizer, samples,
                        representation, cfg, device, variant=request["variant"],
                        policy=request["policy"], progress_callback=progress)
                synchronize(device)
                result_queue.put({"kind": "result", "operation": operation, "rank": rank,
                    "rows": rows, "receipt": receipt,
                    "runtime_seconds": time.perf_counter() - started,
                    "memory": memory_evidence(device)})
            else:
                raise ValueError(f"Unknown generation request: {stage}")
            atomic_json(progress_path, {**base, **progress_state, "operation": operation, "status": "idle",
                                       "elapsed_seconds": time.perf_counter() - started,
                                       "updated_utc": _utc()})
    except BaseException:
        error = traceback.format_exc()
        atomic_json(progress_path, {**base, "status": "failed", "error": error, "updated_utc": _utc()})
        result_queue.put({"kind": "error", "rank": rank, "error": error})
        raise


class PersistentGenerationExecutor:
    """One model per visible card, reused across panels until training resumes."""

    def __init__(self, config, output_root, *, context=None, worker_target=None):
        self.config = deepcopy(config)
        accel = config.get("acceleration", {})
        self.devices = list(accel.get("evaluation_devices", [config["platform"]["device"]]))
        if not self.devices or len(set(self.devices)) != len(self.devices):
            raise ValueError("Evaluation devices must be nonempty and distinct")
        self.output_root = Path(output_root)
        self.session = uuid.uuid4().hex
        self.directory = self.output_root / "engineering" / "generation" / self.session
        self.directory.mkdir(parents=True, exist_ok=True)
        self.context = context or multiprocessing.get_context("spawn")
        self.worker_target = worker_target or _worker_main
        self.requests, self.processes = [], []
        self.results = self.context.Queue()
        self.policies = {}
        self.closed, self.busy = False, False
        self.started = False
        self.last_receipt = None
        atexit.register(self.close)

    def _start(self):
        if self.closed:
            raise RuntimeError("Generation executor is closed")
        if self.started:
            return
        try:
            # Spawn imports the entry module before entering _worker_main.
            # Set BLAS limits before spawn, then restore the coordinator's env.
            threads = str(self.config.get("acceleration", {}).get("worker_cpu_threads", 2))
            variables = ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")
            original_environment = {name: os.environ.get(name) for name in variables}
            try:
                for name in variables:
                    os.environ[name] = threads
                for rank, name in enumerate(self.devices):
                    requests = self.context.Queue()
                    process = self.context.Process(target=self.worker_target,
                        args=(rank, name, self.config, requests, self.results, str(self.directory)),
                        name=f"task-vector-generation-{rank}", daemon=False)
                    process.start()
                    self.requests.append(requests)
                    self.processes.append(process)
            finally:
                for name, value in original_environment.items():
                    if value is None:
                        os.environ.pop(name, None)
                    else:
                        os.environ[name] = value
            self._wait(None, set(range(len(self.devices))), ready=True)
            self.started = True
        except BaseException:
            self.close()
            raise

    def _snapshot(self, operation, completed, expected):
        workers = []
        for rank in range(len(self.devices)):
            path = self.directory / f"worker-{rank}.json"
            if path.is_file():
                workers.append(read_json(path))
        record = {"session": self.session, "status": "running", "operation": operation,
                  "completed_workers": len(completed), "expected_workers": len(expected),
                  "workers": workers, "updated_utc": _utc()}
        current = [worker for worker in workers if worker.get("operation") == operation
                   and worker.get("rank") in expected]
        record["panel"] = next((worker.get("panel") for worker in current if worker.get("panel")), None)
        record["status"] = ("completed" if set(completed) == expected else "running")
        record["completed_samples"] = sum(int(worker.get("completed_samples", 0)) for worker in current)
        record["total_samples"] = sum(int(worker.get("total_samples", 0)) for worker in current)
        estimates = []
        unfinished_without_rate = False
        for worker in current:
            done = int(worker.get("completed_samples", 0))
            total = int(worker.get("total_samples", 0))
            if done > 0:
                estimates.append(max(0, total - done) * float(worker.get("elapsed_seconds", 0)) / done)
            elif total:
                unfinished_without_rate = True
        is_generation = bool(operation and operation.startswith("panel-"))
        record["panel_eta_seconds"] = (max(estimates) if is_generation and estimates
                                        and not unfinished_without_rate else None)
        record["eta_scope"] = "current_generation_panel_only_sample_throughput_excludes_scoring_bootstrap"
        atomic_json(self.output_root / "engineering/generation/progress.json", record)

    def _wait(self, operation, expected, *, ready=False):
        completed = {}
        last_heartbeat = time.monotonic()
        started = last_heartbeat
        heartbeat = float(self.config.get("acceleration", {}).get("heartbeat_seconds", 30))
        startup_limit = float(self.config.get("acceleration", {}).get("worker_startup_timeout_seconds", 900))
        while set(completed) != expected:
            try:
                result = self.results.get(timeout=1)
            except queue.Empty:
                for rank, process in enumerate(self.processes):
                    if rank not in completed and not process.is_alive():
                        raise RuntimeError(f"Generation worker {rank} exited: {process.exitcode}")
                if ready and time.monotonic() - started > startup_limit:
                    raise TimeoutError("Generation workers did not load within startup timeout")
            else:
                if result["kind"] == "error":
                    raise RuntimeError(f"Generation worker {result['rank']} failed:\n{result['error']}")
                if result["kind"] != ("ready" if ready else "result"):
                    raise RuntimeError("Unexpected generation worker event")
                if not ready and result.get("operation") != operation:
                    raise RuntimeError("Generation result belongs to another panel")
                rank = result["rank"]
                if rank not in expected or rank in completed:
                    raise RuntimeError("Unexpected or duplicate generation worker result")
                completed[rank] = result
            if time.monotonic() - last_heartbeat >= heartbeat:
                self._snapshot(operation, completed, expected)
                print(f"GENERATION_HEARTBEAT operation={operation} shards={len(completed)}/{len(expected)}", flush=True)
                last_heartbeat = time.monotonic()
        self._snapshot(operation, completed, expected)
        return completed

    def _policy(self, task, variant, config, calibration_samples, allow_calibration):
        key = (task, variant)
        if not allow_calibration:
            return {"mode": "strict", "effective_batch_size": 1,
                    "reason": "heldout_or_unseen_target_no_calibration_access"}
        if not calibration_samples:
            raise ValueError("Batch calibration requires nonempty source tune samples")
        accel = config.get("acceleration", {})
        evaluation = config.get("evaluation", {})
        contract = object_sha256({"task": task, "variant": variant,
            "model": config.get("model", {}),
            "generation_budgets": {key: evaluation.get(key)
                                   for key in ("max_new_tokens", "max_prompt_tokens")},
            "mode": accel.get("generation_batch_policy", "strict"),
            "candidates": accel.get("generation_batch_candidates", [1, 2, 4, 8]),
            "max_samples": accel.get("calibration_max_samples", 8),
            "tune_samples": [sample.to_dict() for sample in calibration_samples]})
        policy_path = self.output_root / "engineering/batch-calibration" / f"{task}-{variant}.json"
        if key in self.policies:
            if self.policies[key].get("calibration_contract_sha256") != contract:
                raise ValueError("Frozen engineering batch calibration contract differs")
            return self.policies[key]
        if policy_path.is_file():
            policy = read_json(policy_path)
            content = {key: value for key, value in policy.items() if key != "policy_sha256"}
            if (policy.get("calibration_contract_sha256") != contract
                    or policy.get("policy_sha256") != object_sha256(content)):
                raise ValueError("Frozen engineering batch calibration contract differs")
            self.policies[key] = policy
            return policy
        if accel.get("generation_batch_policy", "strict") == "strict":
            policy = {"mode": "strict", "effective_batch_size": 1, "task": task,
                "variant": variant, "calibration_contract_sha256": contract,
                "reason": "strict_batch_one_policy_no_shape_selection",
                "selection_scope": "fixed_target_variant_across_conditions_and_seeds",
                "all_panel_token_equality_guaranteed": True}
            policy["policy_sha256"] = object_sha256(policy)
            self.policies[key] = policy
            atomic_json(policy_path, policy)
            return policy
        operation = "calibrate-" + uuid.uuid4().hex
        self.requests[0].put({"kind": "calibrate", "operation": operation,
            "task": task, "variant": variant, "config": deepcopy(config),
            "samples": [sample.to_dict() for sample in calibration_samples]})
        policy = self._wait(operation, {0})[0]["calibration"]
        policy["task"], policy["variant"] = task, variant
        policy["selection_scope"] = "fixed_target_variant_across_conditions_and_seeds"
        policy["calibration_contract_sha256"] = contract
        policy["policy_sha256"] = object_sha256(policy)
        self.policies[key] = policy
        atomic_json(policy_path, policy)
        return policy

    def generate(self, samples, representation, config, *, task, variant="canonical",
                 panel_label=None, calibration_samples=None, allow_calibration=True,
                 merged_weights_path=None, merged_weights_sha256=None, representation_digest=None):
        if self.busy:
            raise RuntimeError("Only the coordinator may submit one authorized panel at a time")
        if not samples:
            return [], {"devices": self.devices, "shards": [], "generated_samples": 0}
        shards = partition_samples(samples, len(self.devices))
        self.busy = True
        try:
            self._start()
            policy = self._policy(task, variant, config, calibration_samples, allow_calibration)
            operation = "panel-" + uuid.uuid4().hex
            representation_path = None
            if representation is not None:
                identity = (object_sha256({"tensor_digest": representation_digest,
                    "metadata": representation.metadata}) if representation_digest else operation)
                representation_path = self.directory / "representations" / f"{identity}.pt"
                if not representation_path.is_file():
                    representation.save(representation_path)
            started = time.perf_counter()
            for rank, shard in enumerate(shards):
                self.requests[rank].put({"kind": "generate", "operation": operation,
                    "panel": panel_label, "config": deepcopy(config), "variant": variant,
                    "representation_path": str(representation_path) if representation_path else None,
                    "merged_weights_path": str(merged_weights_path) if merged_weights_path else None,
                    "merged_weights_sha256": merged_weights_sha256,
                    "policy": policy, "samples": [sample.to_dict() for sample in shard]})
            results = self._wait(operation, set(range(len(shards))))
            for rank, shard in enumerate(shards):
                if {str(row["id"]) for row in results[rank]["rows"]} != {str(sample.id) for sample in shard}:
                    raise ValueError("A worker returned predictions outside its assigned shard")
            rows = merge_shards(samples, [results[rank]["rows"] for rank in range(len(shards))])
            receipt = {"session": self.session, "operation": operation, "devices": self.devices,
                       "policy": policy, "generated_samples": len(rows), "panel": panel_label,
                       "sample_ids_sha256": object_sha256([sample.id for sample in samples]),
                       "runtime_seconds": time.perf_counter() - started,
                       "coordinator_only_canonical_publication": True,
                       "phase_barrier": "all_assigned_shards_completed_before_scoring",
                       "shards": [{key: value for key, value in results[rank].items()
                                   if key not in {"rows", "kind"}} for rank in range(len(shards))]}
            self.last_receipt = receipt
            atomic_json(self.directory / f"{operation}.json", receipt)
            return rows, receipt
        except BaseException:
            self.close()
            raise
        finally:
            self.busy = False

    def generate_merged(self, samples, config, *, merged_weights_path,
                        merged_weights_sha256, **kwargs):
        if not merged_weights_path or not merged_weights_sha256:
            raise ValueError("Sequential generation requires the exact committed weight artifact")
        return self.generate(samples, None, config,
            merged_weights_path=merged_weights_path,
            merged_weights_sha256=merged_weights_sha256, **kwargs)

    def close(self):
        if self.closed:
            return
        self.closed = True
        for requests in self.requests:
            requests.put(None)
        deadline = time.monotonic() + 10
        for process in self.processes:
            process.join(timeout=max(0, deadline - time.monotonic()))
        for process in self.processes:
            if process.is_alive():
                process.terminate()
                process.join(timeout=5)
            if process.is_alive():
                # A stuck accelerator process must not outlive coordinator
                # shutdown or retain a card after the writer lock is released.
                process.kill()
                process.join()
            else:
                process.join(timeout=0)
        for requests in self.requests:
            requests.close()
            requests.cancel_join_thread()
        self.results.close()
        self.results.cancel_join_thread()
        atomic_json(self.directory / "closed.json", {"session": self.session,
                    "status": "closed", "all_workers_reaped": True,
                    "worker_pids": [process.pid for process in self.processes],
                    "updated_utc": _utc()})
        progress_path = self.output_root / "engineering/generation/progress.json"
        if progress_path.is_file() and read_json(progress_path).get("session") == self.session:
            atomic_json(progress_path, {"session": self.session, "status": "closed", "updated_utc": _utc()})
        atexit.unregister(self.close)

