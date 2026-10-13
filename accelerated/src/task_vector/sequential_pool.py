"""Run independent cyclic orders on separate cards; each order stays sequential."""
from __future__ import annotations
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
import os
import subprocess
import sys
import time
import yaml
from .io import atomic_json
from .acceleration import DEFAULT_ACCELERATION


def parallel_train_orders(runner, provenance):
    settings = runner.config.get("acceleration", {})
    if not settings.get("enabled") or not settings.get("parallel_sequential_orders", True):
        return False
    runner.close_generation_workers()
    runner.release_model()
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    directory = runner.root / "scheduler" / f"sequential-orders-{stamp}"
    directory.mkdir(parents=True, exist_ok=False)
    jobs = [{"seed": seed, "order": order, "status": "pending"}
            for seed in runner.config["analysis"]["seeds"]
            for order in runner.config["analysis"]["sequential_orders"]]
    running, streams = {}, []
    state = {"status": "running", "jobs": jobs, "training_only": True,
             "stage_order_within_each_order_preserved": True}
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(runner.source_root / "src") + os.pathsep + environment.get("PYTHONPATH", "")
    environment["PYTHONUNBUFFERED"] = "1"
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        environment[name] = str(settings.get("training_cpu_threads", 8))
    failed = False
    try:
        while True:
            for device, (process, job) in list(running.items()):
                code = process.poll()
                if code is not None:
                    job.update(status="completed" if code == 0 else "failed", returncode=code,
                               ended_utc=datetime.now(timezone.utc).isoformat())
                    failed |= code != 0
                    del running[device]
            if failed:
                raise RuntimeError(f"Sequential order worker failed: {directory / 'pool.json'}")
            for device in settings.get("evaluation_devices", DEFAULT_ACCELERATION["evaluation_devices"]):
                if device in running:
                    continue
                job = next((item for item in jobs if item["status"] == "pending"), None)
                if job is None:
                    break
                config = deepcopy(runner.config)
                config["platform"]["device"] = device
                config["sequential"] = {"tasks": runner.taskset("mvp"), "seed": job["seed"],
                    "order": job["order"], "data_root": str(runner.data_root),
                    "output_root": str(runner.output_root), "provenance": provenance}
                prefix = f"seed{job['seed']}-" + "__".join(job["order"])
                snapshot = directory / f"{prefix}.yaml"
                snapshot.write_text(yaml.safe_dump(config, allow_unicode=True, sort_keys=False), encoding="utf-8")
                log = directory / f"{prefix}.log"
                stream = log.open("w", encoding="utf-8")
                streams.append(stream)
                process = subprocess.Popen([sys.executable, "-u", "-m", "task_vector.sequential_pool", str(snapshot)],
                    cwd=runner.source_root, env=environment, stdout=stream, stderr=subprocess.STDOUT)
                running[device] = process, job
                job.update(status="running", device=device, pid=process.pid, log=str(log),
                           started_utc=datetime.now(timezone.utc).isoformat())
            atomic_json(directory / "pool.json", state)
            if not running:
                break
            time.sleep(2)
        state["status"] = "completed"
        return True
    except BaseException:
        state["status"] = "failed_or_interrupted"
        raise
    finally:
        # Every child is reaped before the platform coordinator releases its lock.
        for process, _ in running.values():
            if process.poll() is None:
                process.terminate()
        for process, job in running.values():
            try:
                code = process.wait(timeout=15)
            except subprocess.TimeoutExpired:
                process.kill()
                code = process.wait()
            job.update(status="cancelled", returncode=code)
        for stream in streams:
            stream.close()
        state["ended_utc"] = datetime.now(timezone.utc).isoformat()
        atomic_json(directory / "pool.json", state)


def main():
    from .config import load_config
    from .runtime import configure_device
    from .sequential import run_sequential
    config = load_config(sys.argv[1])
    run_sequential(config, device=configure_device(config["platform"]["device"]), split=None)


if __name__ == "__main__":
    main()
