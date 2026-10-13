"""CPU contracts for fixed ModelMate dispatch, provenance and writer exclusion."""
from types import SimpleNamespace
import json
import os

import pytest
from filelock import Timeout

from ops import modelmate_job as job
from task_vector import config as config_module
from task_vector import workflow
from task_vector.io import object_sha256


@pytest.mark.parametrize("stage", ["full", "full-formal"])
def test_full_and_compatibility_alias_run_primary_then_all_formal(tmp_path, monkeypatch, stage):
    calls = []
    runner = SimpleNamespace(output_root=tmp_path, config={})
    monkeypatch.setattr(job, "_run_pool", lambda *args: calls.append("pool"))
    monkeypatch.setattr(workflow, "run_phase", lambda *args: calls.append(args) or {"completed": True})
    monkeypatch.setattr(job, "_formal_precheck", lambda *args: pytest.fail("full dispatch must use all workflow"))
    result = job._execute_stage(runner, stage, tmp_path / "job", {"expected_devices": 8}, {}, {}, {})
    assert calls == ["pool", (runner, "all", "mvp", "test_formal")]
    assert result["completed"] is True


def test_formal_dispatch_uses_eval_only_whole_workflow_without_pool(tmp_path, monkeypatch):
    calls = []
    runner = SimpleNamespace(output_root=tmp_path, config={})
    monkeypatch.setattr(job, "_run_pool", lambda *args: pytest.fail("formal must not launch training pool"))
    monkeypatch.setattr(job, "_formal_precheck", lambda *args: calls.append("precheck") or {"mvp": "verified"})
    monkeypatch.setattr(workflow, "run_phase", lambda *args: calls.append(args) or {"completed": True, "evaluation_only": True})
    result = job._execute_stage(runner, "formal", tmp_path / "job", {}, {}, {}, {})
    assert calls == ["precheck", (runner, "formal", "mvp", "test_formal")]
    assert result["training_rerun"] is False and result["selection_rerun"] is False
    assert json.loads((tmp_path / "job" / "formal_progress.json").read_text())["result"]["evaluation_only"] is True


def test_incomplete_workflow_is_not_reported_completed(tmp_path, monkeypatch):
    runner = SimpleNamespace(output_root=tmp_path, config={})
    monkeypatch.setattr(job, "_run_pool", lambda *args: None)
    monkeypatch.setattr(workflow, "run_phase", lambda *args: {"completed": False, "stop_reason": "gate"})
    result = job._execute_stage(runner, "full", tmp_path / "job", {}, {}, {}, {})
    assert result == {"completed": False, "stop_reason": "gate"}


def test_platform_writer_lock_blocks_concurrent_same_root_and_retains_evidence(tmp_path):
    root = tmp_path / "experiment"
    lock = job._acquire_writer_lock(root, "full", root / "runs" / "platform_jobs" / "one")
    lock_path = root / "locks" / "platform-writer.lock"
    owner_path = root / "locks" / "platform-writer.owner.json"
    try:
        with pytest.raises(Timeout):
            job._acquire_writer_lock(root, "formal", root / "runs" / "platform_jobs" / "two")
        assert json.loads(owner_path.read_text())["stage"] == "full"
    finally:
        lock.release()
    assert owner_path.exists()
    if os.name == "posix":  # UnixFileLock retains its file; WindowsFileLock removes its own file.
        assert lock_path.exists()
    following = job._acquire_writer_lock(root, "formal", root / "runs" / "platform_jobs" / "two")
    following.release()
    assert json.loads(owner_path.read_text())["stage"] == "formal"
    if os.name == "posix":
        assert lock_path.exists()


def _formal_runner(tmp_path, monkeypatch, *, include_base=True):
    runner = SimpleNamespace(config={"analysis": {"seeds": [17]}},
        source_root=tmp_path / "source", output_root=tmp_path / "runs",
        manifest={"manifest_content_sha256": "data-hash"}, verified=[])
    runner.source_root.mkdir()
    runner.asset_fingerprints = lambda: {"base": "actual-base", "semantic": "actual-semantic"}
    runner.base_fingerprint = lambda: "actual-base"
    runner.lock_path = lambda scope: runner.output_root / "protocol" / f"{scope}.lock.json"
    runner.taskset = lambda scope: ["math"]
    runner.directory = lambda scope, route, seed, task: runner.output_root / "training" / scope / route / str(seed) / task
    runner._verify_training_exports = lambda record, signature, base: runner.verified.append((record, signature, base))
    lock = {"lock_sha256": "frozen-protocol", "discovery_decisions": {
        "asset_fingerprints": runner.asset_fingerprints(), "17": {}}}
    monkeypatch.setattr(config_module, "verify_lock", lambda *args: lock)
    sources = config_module.source_manifest(runner.source_root)
    for scope in job.FORMAL_SCOPES:
        job._json(runner.output_root / "analysis" / scope / "test-completed.json",
            {"scope": scope, "protocol_lock": "frozen-protocol", "seeds": [17]})
        job._json(runner.output_root / "analysis" / scope / "gate.json",
            {"passes": True, "fully_validated": True, "seeds": [17]})
        for route in ("ltv", "loda"):
            request = {"config": config_module.scientific_config(runner.config), "route": route, "seed": 17,
                "scope": scope if route == "loda" else "independent", "task": "math",
                "data": "data-hash", "sources": sources}
            if include_base:
                request["base"] = "actual-base"
            job._json(runner.directory(scope, route, 17, "math") / "training.json",
                {"request_signature": object_sha256(request)})
    return runner


def test_formal_precheck_hashes_base_and_delegates_actual_export_validation(tmp_path, monkeypatch):
    runner = _formal_runner(tmp_path, monkeypatch)
    checked = job._formal_precheck(runner, tmp_path / "job")
    assert set(checked) == set(job.FORMAL_SCOPES)
    assert len(runner.verified) == len(job.FORMAL_SCOPES) * 2
    assert all(base == "actual-base" for record, signature, base in runner.verified)
    assert all(record["request_signature"] == signature for record, signature, base in runner.verified)


def test_formal_precheck_rejects_legacy_request_missing_base(tmp_path, monkeypatch):
    runner = _formal_runner(tmp_path, monkeypatch, include_base=False)
    with pytest.raises(ValueError, match="stale/missing completed training signature"):
        job._formal_precheck(runner, tmp_path / "job")
    assert not runner.verified


def test_formal_requires_existing_smoke_and_does_not_create_one(tmp_path):
    runner = SimpleNamespace(output_root=tmp_path)
    with pytest.raises(FileNotFoundError, match="no new smoke or training"):
        job._ensure_smoke(runner, tmp_path / "job", allow_create=False)
    assert not (tmp_path / "smoke.json").exists()


def test_formal_preserves_invalid_existing_smoke(tmp_path, monkeypatch):
    runner = SimpleNamespace(output_root=tmp_path)
    original = b'{"passed":false,"old_evidence":true}\n'
    (tmp_path / "smoke.json").write_bytes(original)
    monkeypatch.setattr(workflow, "smoke_evidence", lambda *args: {"valid": False})
    with pytest.raises(ValueError, match="preserved"):
        job._ensure_smoke(runner, tmp_path / "job", allow_create=False)
    assert (tmp_path / "smoke.json").read_bytes() == original


def _pool_runner(tmp_path, *, records=8, corrupt_request=False):
    runner = SimpleNamespace(config={"analysis": {"seeds": [17, 29, 43]}},
        source_root=tmp_path / "source", output_root=tmp_path / "runs",
        manifest={"manifest_content_sha256": "data-hash"}, verified=[])
    runner.source_root.mkdir()
    runner.base_fingerprint = lambda: "actual-base"
    runner.taskset = lambda scope: ["math", "logic", "translation", "paraphrase"]
    runner.directory = lambda scope, route, seed, task: runner.output_root / "training" / route / str(seed) / task
    runner._verify_training_exports = lambda record, signature, base: runner.verified.append(signature)
    sources = config_module.source_manifest(runner.source_root)
    count = 0
    for route in ("ltv", "loda"):
        for task in runner.taskset("mvp"):
            if count < records:
                signature = job._training_signature(runner, "mvp", route, 17, task, base="actual-base", sources=sources)
                job._json(runner.directory("mvp", route, 17, task) / "training.json",
                    {"request_signature": "old-source" if corrupt_request else signature, "artifacts": {}})
            count += 1
    return runner


def test_all_eight_verified_initial_exports_skip_pool_and_record_reuse(tmp_path, monkeypatch):
    runner = _pool_runner(tmp_path)
    monkeypatch.setattr(job.subprocess, "Popen", lambda *args, **kwargs: pytest.fail("completed pool must not launch children"))
    job._run_pool(runner, tmp_path / "job", {"expected_devices": 8})
    record = json.loads((tmp_path / "job" / "pool_reuse.json").read_text())
    assert record["pool_skipped"] is True and len(record["verified_completed"]) == 8
    assert not record["missing_training_records"] and len(runner.verified) == 8


def test_incomplete_initial_pool_is_not_treated_as_completed(tmp_path):
    runner = _pool_runner(tmp_path, records=7)
    record = job._initial_pool_status(runner)
    assert record["pool_skipped"] is False and len(record["missing_training_records"]) == 1
    assert len(record["verified_completed"]) == 7


def test_initial_pool_reuse_rejects_source_change_and_changed_exports(tmp_path):
    runner = _pool_runner(tmp_path, corrupt_request=True)
    with pytest.raises(ValueError, match="different inputs"):
        job._initial_pool_status(runner)
    assert not runner.verified


def test_initial_pool_reuse_does_not_skip_export_validation(tmp_path):
    runner = _pool_runner(tmp_path)
    def changed_export(*args):
        raise ValueError("Incomplete or changed training export")
    runner._verify_training_exports = changed_export
    with pytest.raises(ValueError, match="changed training export"):
        job._initial_pool_status(runner)
