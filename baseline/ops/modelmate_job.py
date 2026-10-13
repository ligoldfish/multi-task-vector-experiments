"""No-argument ModelMate job launchers using the prepared persistent interpreter.

The platform's Python imports only this standard-library bootstrap. A child bash
activates the configured CANN script and execs the persistent runtime Python.
Scientific code, data preparation, installation and job submission stay outside
this module. Exit 42 means an experiment gate stopped the requested full suite.
"""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import tempfile
import traceback


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SETTINGS_PATH = PROJECT_ROOT / "ops" / "settings.local.json"
CONFIG_PATH = PROJECT_ROOT / "configs" / "modelmate.local.yaml"
STAGES = {"check", "full", "full-formal", "formal"}
TASKS = ("math", "logic", "translation", "paraphrase", "code_generation", "code_repair", "summary")
FORMAL_SCOPES = ("mvp", "expanded", "loo-math", "loo-logic", "loo-translation", "loo-paraphrase")
JOB_MARKER = "TASK_VECTOR_PLATFORM_JOB_DIR="
ENV_OVERRIDES = ("TV_MODEL_PATH", "TV_PERSISTENT_ROOT", "TV_SEMANTIC_MODEL_PATH")
_active_child = None
_received_signal = None


class JobInterrupted(BaseException):
    def __init__(self, signum):
        self.signum = signum
        super().__init__(f"Platform job received signal {signum}")


def _now():
    return datetime.now(timezone.utc).isoformat()


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=path.name + ".", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
            json.dump(value, handle, ensure_ascii=False, indent=2, default=str)
            handle.write("\n")
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def _settings():
    settings = _read_json(SETTINGS_PATH)
    if set(settings) != {"runtime_python", "cann_env", "expected_devices"}:
        raise ValueError("settings.local.json must contain runtime_python, cann_env and expected_devices")
    runtime = Path(settings["runtime_python"])
    if not runtime.is_absolute() or not runtime.is_file():
        raise ValueError("runtime_python must be an existing absolute persistent Python path")
    persistent = Path("/opt/dpcvol/datasets")
    if persistent not in runtime.parents or persistent not in runtime.resolve().parents:
        raise ValueError("runtime_python must remain inside /opt/dpcvol/datasets, including its resolved target")
    if not os.access(runtime, os.X_OK):
        raise ValueError("Configured persistent runtime_python is not executable")
    if type(settings["expected_devices"]) is not int or settings["expected_devices"] != 8:
        raise ValueError("The authorized job resource requires expected_devices=8")
    if settings["cann_env"] is not None:
        activation = Path(settings["cann_env"])
        if not activation.is_absolute() or not activation.is_file():
            raise ValueError("cann_env must be null or an existing absolute activation script")
    if not CONFIG_PATH.is_file():
        raise FileNotFoundError(f"Configure the Notebook first: {CONFIG_PATH}")
    return settings


def _launcher_manifest():
    files = list((PROJECT_ROOT / "ops").rglob("*.py"))
    files += [SETTINGS_PATH, CONFIG_PATH]
    files += [PROJECT_ROOT / f"run_modelmate_{stage.replace('-', '_')}.py" for stage in sorted(STAGES)]
    return {str(path.relative_to(PROJECT_ROOT)): _sha256(path) for path in sorted(set(files)) if path.is_file()}


def _bootstrap_diagnostic(stage, job_id, error, **extra):
    """Fallback is explicitly temporary: config/runtime may not yet be readable."""
    record = {"stage": stage, "job_id": job_id, "status": "bootstrap_or_worker_failed",
              "recorded_utc": _now(), "error": str(error), "temporary_location": True,
              "note": "Bootstrap diagnostics only; authoritative worker evidence uses configured persistent_root/runs/platform_jobs.",
              **extra}
    try:
        path = PROJECT_ROOT / "runs" / "bootstrap_diagnostics" / f"{job_id}.json"
        _json(path, record)
        print(f"Temporary bootstrap diagnostic: {path}", file=sys.stderr, flush=True)
    except Exception as exc:
        print(f"Could not save bootstrap diagnostic: {exc}", file=sys.stderr, flush=True)


def launch(stage):
    """Called by the root scripts; accepts no platform command arguments."""
    if stage not in STAGES:
        raise ValueError(stage)
    if len(sys.argv) != 1:
        print("This ModelMate entry takes no arguments; edit the fixed local settings/config in Notebook.", file=sys.stderr)
        return 2
    job_id = f"{stage}-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S.%fZ')}-{os.getpid()}"
    observed_job_dir = None
    process = None
    previous = {}
    try:
        if os.name != "posix":
            raise RuntimeError("Launch this entry in the Linux ModelMate Ascend container")
        settings = _settings()
        bash = shutil.which("bash")
        if not bash:
            raise RuntimeError("The ModelMate job container must provide bash")
        environment = os.environ.copy()
        for key in (*ENV_OVERRIDES, "TV_CANN_ENV"):
            environment.pop(key, None)
        environment["PYTHONUNBUFFERED"] = "1"
        environment["PYTHONIOENCODING"] = "utf-8"
        environment["TOKENIZERS_PARALLELISM"] = "false"
        environment["PYTHONNOUSERSITE"] = "1"
        # Values are positional shell arguments, never interpolated shell code.
        shell = 'set -eo pipefail; if [ -n "$1" ]; then set +u; source "$1"; fi; export PYTHONPATH="$2/src${PYTHONPATH:+:$PYTHONPATH}"; cd -- "$2"; exec "$3" -u "$2/ops/modelmate_job.py" --worker "$4" "$5"'
        command = [bash, "-c", shell, "modelmate-bootstrap", settings["cann_env"] or "",
                   str(PROJECT_ROOT), settings["runtime_python"], stage, job_id]
        process = subprocess.Popen(command, cwd=PROJECT_ROOT, env=environment,
                                   stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                   text=True, encoding="utf-8", errors="replace", bufsize=1,
                                   start_new_session=True)

        def forward(signum, frame):
            if process.poll() is None:
                print(f"Forwarding signal {signum} to the worker/process group; checkpoints are retained.", flush=True)
                try:
                    os.killpg(process.pid, signum)
                except ProcessLookupError:
                    pass

        for signum in (signal.SIGTERM, signal.SIGINT):
            previous[signum] = signal.signal(signum, forward)
        for line in process.stdout:
            print(line, end="", flush=True)
            if line.startswith(JOB_MARKER):
                observed_job_dir = line[len(JOB_MARKER):].strip()
        code = process.wait()
        if code and observed_job_dir is None:
            _bootstrap_diagnostic(stage, job_id, f"Worker exited {code} before producing persistent job evidence", returncode=code)
        return 128 - code if code < 0 else code
    except BaseException as exc:
        if process is not None and process.poll() is None:
            os.killpg(process.pid, signal.SIGTERM)
            process.wait()
        _bootstrap_diagnostic(stage, job_id, exc, traceback=traceback.format_exc())
        print(f"Job bootstrap failed: {exc}", file=sys.stderr, flush=True)
        return 1
    finally:
        for signum, handler in previous.items():
            signal.signal(signum, handler)


def _existing(value, *, directory=False):
    if not isinstance(value, str) or not value:
        raise ValueError("A nonempty explicit local asset path is required")
    path = Path(value)
    if not path.is_absolute():
        raise ValueError(f"Asset paths must be absolute: {path}")
    persistent = Path("/opt/dpcvol/datasets")
    if persistent not in path.parents or persistent not in path.resolve().parents:
        raise ValueError(f"Local assets must remain inside /opt/dpcvol/datasets, including resolved targets: {path}")
    if not (path.is_dir() if directory else path.exists()):
        raise FileNotFoundError(path)
    return path


def _assets_and_manifest(runner, job_dir):
    """Do not gate on inspect_assets ready flags: explicit files mappings are valid."""
    from task_vector.assets import inspect_assets, _model_check
    from task_vector.io import object_sha256
    config = runner.config
    declared = list(config["data"]["tasks"]) + list(config["data"]["extension_tasks"]) + list(config["data"]["auxiliary_tasks"])
    if len(declared) != len(TASKS) or set(declared) != set(TASKS):
        raise ValueError("The full runbook requires the original seven configured tasks")
    _existing(config["model"].get("path"), directory=True)
    semantic = _existing(config["metrics"].get("semantic_model_path"), directory=True)
    semantic_check = _model_check(str(semantic))
    if semantic_check["status"] != "ready" or not (semantic / "modules.json").is_file():
        raise ValueError(f"The complete local sentence-transformers model is required: {semantic_check}")
    inventory = inspect_assets(config, output_path=job_dir / "assets.json")
    if inventory["model"]["status"] != "ready":
        raise ValueError(f"The local Qwen3-1.7B snapshot is incomplete: {inventory['model']}")
    local_sources = {}
    for task in TASKS:
        source = config["data"]["sources"].get(task)
        source = {"path": source} if isinstance(source, str) else source
        if not isinstance(source, dict):
            raise ValueError(f"No local official source configured for {task}")
        found = []
        if source.get("path"):
            found.append(str(_existing(source["path"])))
        if source.get("files"):
            if not isinstance(source["files"], dict):
                raise ValueError(f"{task}: source.files must map official split names to files")
            for split, paths in source["files"].items():
                paths = [paths] if isinstance(paths, str) else paths
                if not isinstance(paths, list) or not paths:
                    raise ValueError(f"{task}/{split}: an explicit nonempty file list is required")
                for value in paths:
                    path = _existing(value)
                    if not path.is_file():
                        raise ValueError(f"{task}/{split}: a split source must be a file: {path}")
                    found.append(str(path))
        if not found:
            raise ValueError(f"{task}: this runbook requires local source.path or source.files; prepare in Notebook")
        local_sources[task] = found
    manifest = runner.manifest
    if set(manifest["tasks"]) != set(TASKS):
        raise ValueError("Prepared manifest must contain all seven tasks; use Notebook prepare --scope all in a new root")
    unhashed = {key: value for key, value in manifest.items() if key != "manifest_content_sha256"}
    expected_digest = hashlib.sha256(json.dumps(unhashed, ensure_ascii=False, sort_keys=True).encode("utf-8")).hexdigest()
    if manifest.get("manifest_content_sha256") != expected_digest:
        raise ValueError("Prepared manifest content hash is invalid")
    files = {}
    for task in TASKS:
        entries = manifest["tasks"][task]["files"]
        if not {"train", "tune", "discovery", "test", "test_formal"} <= set(entries):
            raise ValueError(f"Prepared manifest is missing required splits for {task}")
        for split, entry in entries.items():
            path = runner.data_root / task / f"{split}.jsonl"
            if not path.is_file() or Path(entry["path"]).resolve() != path.resolve():
                raise ValueError(f"Prepared split path is missing or differs from its manifest: {path}")
            digest = _sha256(path)
            with path.open(encoding="utf-8") as handle:
                count = sum(bool(line.strip()) for line in handle)
            if digest != entry["sha256"] or count != entry["count"]:
                raise ValueError(f"Prepared split hash/count differs: {path}")
            files[f"{task}/{split}"] = {"path": str(path), "sha256": digest, "count": count}
    record = {"passed": True, "local_sources": local_sources, "prepared_files": files,
              "data_manifest_sha256": object_sha256(manifest), "semantic_check": semantic_check,
              "asset_inventory_flags_are_advisory_for_files_mapping": True}
    _json(job_dir / "prepared_assets.json", record)
    return inventory, record


def _ensure_smoke(runner, job_dir, *, allow_create):
    from task_vector.workflow import smoke_evidence
    path = runner.output_root / "smoke.json"
    if path.exists():
        evidence = smoke_evidence(runner)
        _json(job_dir / "smoke_validation.json", evidence)
        if not evidence["valid"]:
            raise ValueError("Existing smoke is failed/stale/another backend; it is preserved. Inspect evidence and use a new configured experiment root.")
        return evidence
    if not allow_create:
        raise FileNotFoundError("Formal evaluation requires an existing valid NPU smoke; no new smoke or training is run")
    from task_vector.smoke import run_smoke
    runner.release_model()
    run_smoke(runner.config, task="math", output_dir=runner.output_root)
    evidence = smoke_evidence(runner)
    _json(job_dir / "smoke_validation.json", evidence)
    if not evidence["valid"]:
        raise RuntimeError("Real-model NPU smoke did not produce valid matching evidence")
    return evidence


def _formal_precheck(runner, job_dir):
    """Read/verify every scope before exposing any additional final-test outputs."""
    from task_vector.config import verify_lock, source_manifest
    expected_seeds = {str(seed) for seed in runner.config["analysis"]["seeds"]}
    assets = runner.asset_fingerprints()
    base = runner.base_fingerprint()
    sources = source_manifest(runner.source_root)
    checked = {}
    for scope in FORMAL_SCOPES:
        lock = verify_lock(runner.lock_path(scope), runner.config, runner.manifest, runner.source_root)
        if lock["discovery_decisions"].get("asset_fingerprints") != assets:
            raise ValueError(f"{scope}: frozen model/semantic assets differ")
        marker_path = runner.output_root / "analysis" / scope / "test-completed.json"
        marker = _read_json(marker_path)
        if marker.get("scope") != scope or marker.get("protocol_lock") != lock["lock_sha256"] or {str(seed) for seed in marker.get("seeds", [])} != expected_seeds:
            raise ValueError(f"{scope}: primary test completion record differs from its lock/seeds")
        gate = _read_json(runner.output_root / "analysis" / scope / "gate.json")
        if not gate.get("passes") or not gate.get("fully_validated") or {str(seed) for seed in gate.get("seeds", [])} != expected_seeds:
            raise ValueError(f"{scope}: the complete primary validity gate has not passed")
        for seed in runner.config["analysis"]["seeds"]:
            if str(seed) not in lock["discovery_decisions"]:
                raise ValueError(f"{scope}: missing frozen decisions for seed {seed}")
            for route in ("ltv", "loda"):
                for task in runner.taskset(scope):
                    folder = runner.directory(scope, route, seed, task)
                    training = _read_json(folder / "training.json")
                    signature = _training_signature(runner, scope, route, seed, task, base=base, sources=sources)
                    if training.get("request_signature") != signature:
                        raise ValueError(f"{scope}: stale/missing completed training signature: {folder}")
                    runner._verify_training_exports(training, signature, base)
        checked[scope] = {"lock_sha256": lock["lock_sha256"], "primary_completion": str(marker_path),
                          "formal_split": "test_formal", "training_or_selection_permitted": False}
    _json(job_dir / "formal_precheck.json", {"passed": True, "scopes": checked,
        "note": "Formal expansion can include previously exposed test rows; it is not an independent confirmation set."})
    return checked


def _training_signature(runner, scope, route, seed, task, *, base, sources):
    from task_vector.config import scientific_config
    from task_vector.io import object_sha256
    return object_sha256({"config": scientific_config(runner.config), "route": route, "seed": seed,
        "scope": scope if route == "loda" else "independent", "task": task,
        "data": runner.manifest["manifest_content_sha256"], "sources": sources, "base": base})


def _initial_pool_status(runner):
    """Reuse only committed, matching exports; missing jobs still use the pool."""
    from task_vector.config import source_manifest
    from task_vector.io import object_sha256
    base, sources = runner.base_fingerprint(), source_manifest(runner.source_root)
    verified, missing = [], []
    for route in ("ltv", "loda"):
        for task in runner.taskset("mvp"):
            path = runner.directory("mvp", route, 17, task) / "training.json"
            if not path.is_file():
                missing.append({"route": route, "task": task, "path": str(path)})
                continue
            record = _read_json(path)
            signature = _training_signature(runner, "mvp", route, 17, task, base=base, sources=sources)
            if record.get("request_signature") != signature:
                raise ValueError(f"Initial pool completed training has different inputs: {path}; use a new experiment root")
            runner._verify_training_exports(record, signature, base)
            verified.append({"route": route, "task": task, "path": str(path),
                "training_json_sha256": _sha256(path), "request_signature": signature,
                "artifacts": record.get("artifacts", {})})
    return {"scope": "mvp", "seed": 17, "pool_skipped": not missing,
        "verified_completed": verified, "missing_training_records": missing,
        "base_fingerprint": base, "source_manifest_sha256": object_sha256(sources),
        "policy": "verify_request_signature_and_all_export_hashes_before_reuse; no_training_from_this_check"}


def _acquire_writer_lock(persistent_root, stage, job_dir):
    """Serialize platform writers; retain the OS-lock file and owner evidence."""
    from filelock import FileLock
    lock_path = Path(persistent_root) / "locks" / "platform-writer.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    lock = FileLock(str(lock_path), timeout=0)
    lock.acquire()
    try:
        _json(lock_path.with_suffix(".owner.json"), {"pid": os.getpid(), "stage": stage,
              "job_dir": str(job_dir), "started_utc": _now(), "status": "running"})
    except BaseException:
        lock.release()
        raise
    return lock


def _run_pool(runner, job_dir, settings):
    global _active_child
    reuse = _initial_pool_status(runner)
    _json(job_dir / "pool_reuse.json", reuse)
    if reuse["pool_skipped"]:
        return
    command = [sys.executable, str(PROJECT_ROOT / "deploy" / "launch_pool.py"), "--config", str(CONFIG_PATH),
               "--scope", "mvp", "--seed", "17", "--devices", ",".join(str(index) for index in range(settings["expected_devices"]))]
    _json(job_dir / "pool_command.json", {"command": command, "kind": "independent_single_NPU_jobs",
        "effective_batch_per_task": 16, "first_seed_only": True})
    try:
        _active_child = subprocess.Popen(command, cwd=PROJECT_ROOT, env=os.environ.copy())
        code = _active_child.wait()
        _json(job_dir / "pool_result.json", {"returncode": code, "ended_utc": _now()})
        if _received_signal is not None:
            raise JobInterrupted(_received_signal)
        if code:
            raise RuntimeError(f"Independent training pool exited {code}; inspect persistent scheduler logs and per-task last.pt")
    finally:
        _active_child = None


def _worker_signal(signum, frame):
    global _received_signal
    _received_signal = signum
    if _active_child is not None and _active_child.poll() is None:
        # The pool handles termination, waits for its children, and retains state.
        _active_child.send_signal(signum)
        return
    raise JobInterrupted(signum)


def _execute_stage(runner, stage, job_dir, settings, inventory, context, smoke):
    """Dispatch the fixed scientific workflow after environment/asset validation."""
    if stage == "check":
        from task_vector.suite import experiment_inventory
        _json(runner.output_root / "preflight.json", context)
        _json(runner.output_root / "assets.json", inventory)
        planned = experiment_inventory(runner.config)
        _json(runner.output_root / "experiment_inventory.json", planned)
        _json(job_dir / "experiment_inventory.json", planned)
        return {"completed": True, "smoke": smoke, "prepared_tasks": list(TASKS)}
    if stage in {"full", "full-formal"}:
        _run_pool(runner, job_dir, settings)
        from task_vector.workflow import run_phase
        # The workflow preserves the 256-sample primary stages before expanding
        # every final and control panel. The legacy name is a compatibility alias.
        return run_phase(runner, "all", "mvp", "test_formal")
    if stage == "formal":
        checked = _formal_precheck(runner, job_dir)
        from task_vector.workflow import run_phase
        result = run_phase(runner, "formal", "mvp", "test_formal")
        _json(job_dir / "formal_progress.json", {"prechecked_scopes": checked, "result": result})
        return {**result, "training_rerun": False, "selection_rerun": False}
    raise ValueError(f"Unknown platform stage: {stage}")


def _worker(stage, job_id):
    global _received_signal
    settings = _settings()
    _received_signal = None
    for key in ENV_OVERRIDES:
        os.environ.pop(key, None)
    job_dir = None
    runner = lock = None
    acquired = False
    metadata = {}
    requested_split = None if stage == "check" else "test_formal"
    outcome = {"stage": stage, "job_id": job_id, "split": requested_split,
               "completed": False, "status": "starting", "exit_code": 1}
    code = 1
    try:
        import yaml
        raw_config = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
        persistent_root = Path(raw_config["platform"]["persistent_root"])
        volume_root = Path("/opt/dpcvol/datasets")
        if not persistent_root.is_absolute() or volume_root not in persistent_root.parents or volume_root not in persistent_root.resolve().parents:
            raise ValueError("platform.persistent_root must be an independent directory under /opt/dpcvol/datasets")
        job_dir = persistent_root / "runs" / "platform_jobs" / job_id
        if persistent_root.resolve() not in job_dir.resolve().parents:
            raise ValueError("The platform job evidence directory resolves outside this experiment root")
        job_dir.mkdir(parents=True, exist_ok=False)
        print(JOB_MARKER + str(job_dir), flush=True)
        metadata = {"stage": stage, "job_id": job_id, "split": requested_split, "status": "starting", "started_utc": _now(),
            "pid": os.getpid(), "python": sys.executable, "project_root": str(PROJECT_ROOT),
            "persistent_root": str(persistent_root), "config_path": str(CONFIG_PATH),
            "settings": settings, "launcher_sources": _launcher_manifest(),
            "cann_env_sha256": _sha256(settings["cann_env"]) if settings["cann_env"] else None,
            "inherited_asset_overrides_cleared": list(ENV_OVERRIDES),
            "environment": {name: os.environ.get(name) for name in ("PATH", "PYTHONPATH", "LD_LIBRARY_PATH", "ASCEND_RT_VISIBLE_DEVICES", "ASCEND_DEVICE_ID")}}
        _json(job_dir / "metadata.json", metadata)
        (job_dir / "config.snapshot.yaml").write_text(CONFIG_PATH.read_text(encoding="utf-8"), encoding="utf-8")
        _json(job_dir / "settings.snapshot.json", settings)
        if Path(sys.executable).resolve() != Path(settings["runtime_python"]).resolve():
            raise RuntimeError("Worker did not start with the configured persistent runtime_python")
        lock = _acquire_writer_lock(persistent_root, stage, job_dir)
        acquired = True
        signal.signal(signal.SIGTERM, _worker_signal)
        signal.signal(signal.SIGINT, _worker_signal)
        from task_vector.config import load_config, scientific_config, source_manifest
        from task_vector.io import object_sha256
        config = load_config(CONFIG_PATH)
        config.setdefault("runtime", {})
        config["runtime"].setdefault("data_root", str(persistent_root / "data"))
        config["runtime"].setdefault("output_root", str(persistent_root / "runs"))
        if Path(config["runtime"]["data_root"]).resolve() != (persistent_root / "data").resolve() or Path(config["runtime"]["output_root"]).resolve() != (persistent_root / "runs").resolve():
            raise ValueError("Runbook runtime roots must be persistent_root/data and persistent_root/runs")
        if not str(config["platform"]["device"]).startswith("npu:"):
            raise ValueError("A real Ascend NPU device is required")
        import importlib.metadata as md
        import torch
        import torch_npu  # noqa: F401
        versions = {name: md.version(name) for name in ("torch", "torch-npu")}
        runtime_probe = {"python": sys.executable, "framework_versions": versions,
            "npu_available": bool(torch.npu.is_available()), "visible_devices": int(torch.npu.device_count()),
            "expected_devices": settings["expected_devices"], "recorded_utc": _now()}
        _json(job_dir / "context.json", runtime_probe)
        if any(not version.startswith("2.8.") for version in versions.values()):
            raise RuntimeError(f"Use the authorized torch/torch_npu2.8 runtime, without installing replacements: {versions}")
        if not runtime_probe["npu_available"] or runtime_probe["visible_devices"] < settings["expected_devices"]:
            raise RuntimeError(f"This job requires at least {settings['expected_devices']} visible NPUs; detected {runtime_probe['visible_devices']}, available={runtime_probe['npu_available']}. Check the training job allocation")
        from task_vector.runtime import preflight
        context = preflight(config, job_dir)
        _json(job_dir / "context.json", context)
        from task_vector.runner import ExperimentRunner
        runner = ExperimentRunner(config, source_root=PROJECT_ROOT)
        inventory, prepared = _assets_and_manifest(runner, job_dir)
        metadata.update(status="running", config_sha256=object_sha256(scientific_config(config)),
                        source_manifest_sha256=object_sha256(source_manifest(PROJECT_ROOT)),
                        base_fingerprint=runner.base_fingerprint(),
                        data_manifest_sha256=prepared["data_manifest_sha256"])
        _json(job_dir / "metadata.json", metadata)
        smoke = _ensure_smoke(runner, job_dir, allow_create=stage != "formal")
        result = _execute_stage(runner, stage, job_dir, settings, inventory, context, smoke)
        complete = bool(result.get("completed"))
        code = 0 if complete else 42
        outcome.update(status="completed" if complete else "stopped_by_gate", completed=complete, exit_code=code, result=result)
    except BaseException as exc:
        code = 128 + exc.signum if isinstance(exc, JobInterrupted) else 1
        outcome.update(status="interrupted" if isinstance(exc, JobInterrupted) else "failed", completed=False,
                       exit_code=code, error_type=type(exc).__name__, error=str(exc), traceback=traceback.format_exc())
        if job_dir is not None:
            _json(job_dir / "failure.json", outcome)
        else:
            _bootstrap_diagnostic(stage, job_id, exc, traceback=traceback.format_exc())
        print(f"{stage} job failed: {exc}", file=sys.stderr, flush=True)
    finally:
        if runner is not None:
            try:
                runner.release_model()
            except Exception as exc:
                outcome["model_release_error"] = str(exc)
        if job_dir is not None:
            if acquired and runner is not None:
                try:
                    from task_vector.reporting import build_report
                    report = build_report(runner.output_root)
                    _json(job_dir / "report.json", report)
                    (job_dir / "report.md").write_text(Path(report["path"]).read_text(encoding="utf-8"), encoding="utf-8")
                except Exception as exc:
                    outcome["report_error"] = str(exc)
                    if code == 0:
                        code = 1
                        outcome.update(status="report_failed", completed=False, exit_code=1)
            outcome["ended_utc"] = _now()
            _json(job_dir / "result.json", outcome)
            metadata.update(status=outcome["status"], ended_utc=outcome["ended_utc"], exit_code=code)
            _json(job_dir / "metadata.json", metadata)
            if acquired:
                _json(persistent_root / "locks" / "platform-writer.owner.json", {"pid": os.getpid(), "stage": stage,
                    "job_dir": str(job_dir), "ended_utc": _now(), "status": outcome["status"]})
            print(json.dumps({"stage": stage, "status": outcome["status"], "completed": outcome["completed"],
                              "exit_code": code, "job_dir": str(job_dir)}, ensure_ascii=False), flush=True)
        if acquired:
            lock.release()  # Release the OS lock; retain the lock file and owner/evidence artifacts.
    return code


if __name__ == "__main__":
    if len(sys.argv) != 4 or sys.argv[1] != "--worker" or sys.argv[2] not in STAGES:
        raise SystemExit("Use a root run_modelmate_*.py entry without arguments")
    raise SystemExit(_worker(sys.argv[2], sys.argv[3]))
