#!/usr/bin/env python
"""Run independent task jobs on NPU cards without changing per-job batch size.

This launches processes inside an already allocated ModelMate training job;
it does not submit, stop or modify ModelMate platform jobs.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import traceback
import yaml


def _require_smoke(checker, scope):
    from task_vector.workflow import smoke_evidence
    evidence = smoke_evidence(checker, scope)
    if not evidence.get("valid") or not str(evidence.get("observed_device", "")).startswith("npu"):
        raise SystemExit("Run a valid matching real-NPU smoke with all intervention/recovery checks before launching the pool")
    return evidence


def _acquire_pool_lock(root, scope, seed, record):
    """OS lock survives as a file on Linux; owner JSON is diagnostic evidence."""
    from filelock import FileLock, Timeout
    from task_vector.io import atomic_json
    folder = Path(root) / "locks"
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"pool-{scope}-seed{seed}.lock"
    lock = FileLock(str(path), timeout=0)
    try:
        lock.acquire()
    except Timeout as exc:
        raise RuntimeError(f"Another pool holds the OS lock; inspect its owner and platform jobs: {path}") from exc
    try:
        atomic_json(path.with_suffix(".owner.json"), {**record, "status": "running"})
    except BaseException:
        lock.release()
        raise
    return lock, path.with_suffix(".owner.json")


def _run_jobs(config, args, devices, output, project, record):
    """Run independent children and reap every started child before returning.

    The caller must hold the pool OS lock through this function. Popen failures
    also terminate/reap other children and close their logs before lock release.
    """
    from task_vector.io import atomic_json, object_sha256
    output, project = Path(output), Path(project)
    env = dict(os.environ, PYTHONUNBUFFERED="1", TOKENIZERS_PARALLELISM="false")
    if config.get("acceleration", {}).get("enabled"):
        for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
            env[name] = str(config["acceleration"].get("training_cpu_threads", 8))
    env["PYTHONPATH"] = str(project / "src") + os.pathsep + env.get("PYTHONPATH", "")
    jobs = [{"route": route, "task": task, "status": "pending"} for route in args.routes.split(",") for task in args.tasks]
    state = {**record, "kind": "independent_single_npu_tasks", "effective_batch_per_task": 16,
             "config_sha256": object_sha256(config), "jobs": jobs}
    running = {}
    failure = interrupted = False
    previous = {}
    def stop(signum, frame):
        nonlocal interrupted
        interrupted = True
    try:
        for signum in (signal.SIGTERM, signal.SIGINT):
            previous[signum] = signal.signal(signum, stop)
        while True:
            for device, (process, job, stream) in list(running.items()):
                code = process.poll()
                if code is not None:
                    job.update(status="completed" if code == 0 else "failed", returncode=code,
                               ended_utc=datetime.now(timezone.utc).isoformat())
                    stream.close()
                    del running[device]
                    failure |= code != 0
            if interrupted:
                break
            if not failure:
                for device in devices:
                    if device in running:
                        continue
                    pending = next((job for job in jobs if job["status"] == "pending"), None)
                    if pending is None:
                        break
                    per_device = dict(config)
                    per_device["platform"] = {**config["platform"], "device": f"npu:{device}"}
                    path = output / f"config-npu{device}.yaml"
                    path.write_text(yaml.safe_dump(per_device, allow_unicode=True, sort_keys=False), encoding="utf-8")
                    log = output / f"{pending['route']}-{pending['task']}.log"
                    stream = log.open("w", encoding="utf-8", buffering=1)
                    command = [sys.executable, "-m", "task_vector.cli", "train", "--config", str(path),
                               "--scope", args.scope, "--route", pending["route"],
                               "--seed", str(args.seed), "--task", pending["task"]]
                    if getattr(args, "layers", None):
                        command += ["--layers", *map(str, args.layers)]
                    pending.update(status="running", device=f"npu:{device}", command=command, log=str(log),
                                   started_utc=datetime.now(timezone.utc).isoformat())
                    try:
                        process = subprocess.Popen(command, cwd=project, env=env, stdout=stream, stderr=subprocess.STDOUT)
                    except BaseException:
                        stream.close()
                        pending.update(status="launch_failed", ended_utc=datetime.now(timezone.utc).isoformat())
                        raise
                    pending["pid"] = process.pid
                    running[device] = process, pending, stream
            state["status"] = "failed" if failure else "running" if running else "completed"
            atomic_json(output / "pool.json", state)
            if not running:
                break
            time.sleep(2)
    except Exception as exc:
        failure = True
        state.update(error_type=type(exc).__name__, error=str(exc), traceback=traceback.format_exc())
        print(f"Independent pool failed: {exc}", file=sys.stderr, flush=True)
    finally:
        # Signal every live child first, then wait/reap all of them. Holding the
        # outer OS lock through this cleanup prevents a replacement pool writer.
        try:
            for process, job, stream in running.values():
                if process.poll() is None:
                    try:
                        process.terminate()
                    except ProcessLookupError:
                        pass  # Child exited between poll and terminate; still wait/reap it.
                    job["status"] = "termination_requested"
            for process, job, stream in running.values():
                try:
                    try:
                        code = process.wait(timeout=30)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        code = process.wait()
                    job.update(status="interrupted" if interrupted else "cancelled_after_failure" if failure else "completed" if code == 0 else "failed",
                               returncode=code, ended_utc=datetime.now(timezone.utc).isoformat())
                    failure |= code != 0 and not interrupted
                finally:
                    stream.close()
        finally:
            state["status"] = "interrupted" if interrupted else "failed" if failure else "completed"
            state["ended_utc"] = datetime.now(timezone.utc).isoformat()
            try:
                atomic_json(output / "pool.json", state)
            finally:
                for signum, handler in previous.items():
                    signal.signal(signum, handler)
    return state, 130 if interrupted else 1 if failure else 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--scope", choices=["mvp", "expanded", "loo-math", "loo-logic", "loo-translation", "loo-paraphrase"], default="mvp")
    parser.add_argument("--seed", type=int, choices=[17, 29, 43], default=17)
    parser.add_argument("--devices", default="0,1,2,3,4,5,6,7")
    parser.add_argument("--routes", default="ltv,loda")
    args = parser.parse_args()
    devices = [int(x) for x in args.devices.split(",")]
    if not devices or len(set(devices)) != len(devices) or min(devices) < 0:
        parser.error("NPU device IDs must be distinct nonnegative integers")
    routes = args.routes.split(",")
    if not routes or set(routes) - {"ltv", "loda"}:
        parser.error("routes must contain ltv and/or loda")
    from task_vector.config import load_config
    from task_vector.io import atomic_json
    config = load_config(args.config)
    tasks = list(config["data"]["tasks"])
    if args.scope == "expanded":
        tasks += list(config["data"]["extension_tasks"])
    if args.scope.startswith("loo-"):
        tasks.remove(args.scope[4:])
    args.tasks = tasks
    root = Path(config["platform"]["persistent_root"])
    root.mkdir(parents=True, exist_ok=True)
    config.setdefault("runtime", {})
    config["runtime"].setdefault("data_root", str(root / "data"))
    config["runtime"].setdefault("output_root", str(root / "runs"))
    from task_vector.runner import ExperimentRunner
    checker = ExperimentRunner(config)
    _require_smoke(checker, args.scope)
    import torch
    import torch_npu  # noqa: F401
    if not torch.npu.is_available() or max(devices) >= torch.npu.device_count():
        raise SystemExit("Requested pool devices are not visible in this training container")
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    output = root / "scheduler" / f"{args.scope}-seed{args.seed}-{stamp}"
    output.mkdir(parents=True, exist_ok=False)
    record = {"pid": os.getpid(), "scope": args.scope, "seed": args.seed,
              "output": str(output), "started_utc": stamp, "devices": devices}
    lock, owner_path = _acquire_pool_lock(root, args.scope, args.seed, record)
    project = Path(__file__).resolve().parents[1]
    state = {"status": "starting"}
    try:
        state, code = _run_jobs(config, args, devices, output, project, record)
    finally:
        try:
            atomic_json(owner_path, {**record, "status": state["status"],
                                     "ended_utc": datetime.now(timezone.utc).isoformat()})
        finally:
            lock.release()  # No unlink: Linux OS lock ownership, not file presence, controls reuse.
    print(json.dumps({"status": state["status"], "record": str(output / "pool.json")}, ensure_ascii=False))
    return code


if __name__ == "__main__":
    raise SystemExit(main())
