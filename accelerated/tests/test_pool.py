"""Pool lifecycle tests use fake children and never allocate or import an NPU."""
from types import SimpleNamespace
import json
import os
import signal

import pytest
import yaml

from deploy import launch_pool as pool
from task_vector import workflow


def test_pool_lock_uses_os_ownership_and_preserves_owner_evidence(tmp_path):
    record = {"pid": 123, "scope": "mvp", "seed": 17}
    lock, owner = pool._acquire_pool_lock(tmp_path, "mvp", 17, record)
    try:
        with pytest.raises(RuntimeError, match="holds the OS lock"):
            pool._acquire_pool_lock(tmp_path, "mvp", 17, {"pid": 456})
        assert json.loads(owner.read_text())["pid"] == 123
    finally:
        lock.release()
    assert owner.is_file()
    if os.name == "posix":
        assert (owner.parent / "pool-mvp-seed17.lock").exists()
    next_lock, _ = pool._acquire_pool_lock(tmp_path, "mvp", 17, {"pid": 456})
    next_lock.release()
    assert json.loads(owner.read_text())["pid"] == 456


def _inputs(tmp_path):
    output = tmp_path / "scheduler"
    output.mkdir()
    args = SimpleNamespace(scope="mvp", seed=17, routes="ltv", tasks=["math", "logic"])
    config = {"platform": {"device": "npu:0"}, "training": {"micro_batch_size": 2, "gradient_accumulation_steps": 8}}
    return output, args, config


def test_popen_failure_reaps_children_closes_streams_before_pool_lock_release(tmp_path, monkeypatch):
    output, args, config = _inputs(tmp_path)
    args.tasks.append("translation")
    lock, owner = pool._acquire_pool_lock(tmp_path, "mvp", 17, {"pid": 123})
    events, streams = [], []
    class Child:
        def __init__(self, pid):
            self.pid, self.code = pid, None
        def poll(self):
            return self.code
        def terminate(self):
            events.append(("terminate", self.pid))
        def wait(self, timeout=None):
            with pytest.raises(RuntimeError, match="holds the OS lock"):
                pool._acquire_pool_lock(tmp_path, "mvp", 17, {"pid": 456})
            events.append(("wait", self.pid))
            self.code = -15
            return self.code
    children = [Child(999), Child(1000)]
    def popen(*args, **kwargs):
        streams.append(kwargs["stdout"])
        if len(streams) <= 2:
            return children[len(streams) - 1]
        raise OSError("injected Popen failure")
    monkeypatch.setattr(pool.subprocess, "Popen", popen)
    before = {signum: signal.getsignal(signum) for signum in (signal.SIGTERM, signal.SIGINT)}
    try:
        state, code = pool._run_jobs(config, args, [0, 1, 2], output, tmp_path, {"pid": 123})
        assert code == 1 and state["status"] == "failed"
        assert state["error_type"] == "OSError" and state["jobs"][2]["status"] == "launch_failed"
        assert events == [("terminate", 999), ("terminate", 1000), ("wait", 999), ("wait", 1000)]
        assert all(child.poll() == -15 for child in children)
        assert len(streams) == 3 and all(stream.closed for stream in streams)
        assert all(signal.getsignal(signum) == handler for signum, handler in before.items())
    finally:
        lock.release()
    following, _ = pool._acquire_pool_lock(tmp_path, "mvp", 17, {"pid": 456})
    following.release()


def test_pool_success_preserves_per_job_batch_and_device_assignment(tmp_path, monkeypatch):
    output, args, config = _inputs(tmp_path)
    streams = []
    class Child:
        pid = 999
        def poll(self):
            return 0
    def popen(*args, **kwargs):
        streams.append(kwargs["stdout"])
        return Child()
    monkeypatch.setattr(pool.subprocess, "Popen", popen)
    monkeypatch.setattr(pool.time, "sleep", lambda *args: None)
    state, code = pool._run_jobs(config, args, [0, 1], output, tmp_path, {"pid": 123})
    assert code == 0 and state["status"] == "completed"
    assert state["effective_batch_per_task"] == 16 and all(stream.closed for stream in streams)
    for device in (0, 1):
        child_config = yaml.safe_load((output / f"config-npu{device}.yaml").read_text())
        assert child_config["platform"]["device"] == f"npu:{device}"
        assert child_config["training"] == config["training"]
    assert config["platform"]["device"] == "npu:0"


@pytest.mark.parametrize("valid,device,accepted", [(True, "npu:0", True), (False, "npu:0", False), (True, "cpu", False)])
def test_pool_requires_unified_smoke_validation_and_actual_npu(monkeypatch, valid, device, accepted):
    calls = []
    checker = object()
    def evidence(runner, scope):
        calls.append((runner, scope))
        return {"valid": valid, "observed_device": device}
    monkeypatch.setattr(workflow, "smoke_evidence", evidence)
    if accepted:
        assert pool._require_smoke(checker, "mvp")["valid"]
    else:
        with pytest.raises(SystemExit, match="intervention/recovery"):
            pool._require_smoke(checker, "mvp")
    assert calls == [(checker, "mvp")]
