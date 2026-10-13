"""Phase-local independent training pool; never changes a task's optimizer batch."""
from __future__ import annotations
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
import os

from .config import scientific_config, source_manifest
from .io import atomic_json, object_sha256, read_json
from .acceleration import DEFAULT_ACCELERATION


def train_phase(runner, scope, seed, *, routes=("ltv", "loda"), layers=None):
    settings = runner.config.get("acceleration", {})
    if not settings.get("enabled") or not settings.get("parallel_training", True):
        for route in routes:
            for task in runner.taskset(scope):
                runner.train(scope, route, seed, task, layers) if layers is not None else runner.train(scope, route, seed, task)
        return
    if hasattr(runner, "close_generation_workers"):
        runner.close_generation_workers()
    runner.release_model()
    base, sources = runner.base_fingerprint(), source_manifest(runner.source_root)
    cfg = deepcopy(runner.config)
    if layers is not None:
        cfg["ltv"]["layers"] = list(layers)
    missing = {}
    verified = []
    for route in routes:
        missing[route] = []
        for task in runner.taskset(scope):
            directory = runner.directory(scope, route, seed, task, layers)
            signature = object_sha256({"config": scientific_config(cfg), "route": route, "seed": seed,
                "scope": scope if route == "loda" else "independent", "task": task,
                "data": runner.manifest["manifest_content_sha256"], "base": base, "sources": sources})
            path = directory / "training.json"
            if not path.is_file():
                missing[route].append(task)
                continue
            record = read_json(path)
            if record.get("request_signature") != signature:
                raise ValueError(f"Training pool cannot reuse changed sources/config: {path}")
            runner._verify_training_exports(record, signature, base)
            verified.append(str(path))
    if not any(missing.values()):
        return
    # One pool per route permits reuse of different completed task subsets.
    # Two routes with identical missing sets share all eight slots in one pool.
    groups = {}
    for route, tasks in missing.items():
        if tasks:
            groups.setdefault(tuple(tasks), []).append(route)
    from deploy.launch_pool import _run_jobs, _acquire_pool_lock
    devices = settings.get("evaluation_devices", DEFAULT_ACCELERATION["evaluation_devices"])
    if any(not d.startswith("npu:") for d in devices):
        raise ValueError("Independent training pool requires NPU devices")
    indices = [int(d.split(":")[1]) for d in devices]
    for tasks, group_routes in groups.items():
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
        output = runner.root / "scheduler" / f"accelerated-{scope}-seed{seed}-{stamp}"
        output.mkdir(parents=True, exist_ok=False)
        record = {"pid": os.getpid(), "scope": scope, "seed": seed, "started_utc": stamp,
                  "output": str(output), "devices": indices, "layers": layers,
                  "verified_completed": verified, "phase_local_only": True}
        args = SimpleNamespace(scope=scope, seed=seed, routes=",".join(group_routes), tasks=list(tasks), layers=layers)
        lock, owner = _acquire_pool_lock(runner.root, scope, seed, record)
        state = {"status": "starting"}
        try:
            state, code = _run_jobs(runner.config, args, indices, output, runner.source_root, record)
            if code:
                raise RuntimeError(f"Phase training pool exited {code}: {output / 'pool.json'}")
        finally:
            atomic_json(owner, {**record, "status": state.get("status", "failed"),
                               "ended_utc": datetime.now(timezone.utc).isoformat()})
            lock.release()
