"""CPU-only scheduling/lifecycle checks for independent sequential orders."""
from copy import deepcopy
import json
from pathlib import Path
import subprocess
import sys
from types import ModuleType, SimpleNamespace

import pytest
import yaml

from task_vector import sequential_pool


class FakeProcess:
    def __init__(self, owner, command, kwargs):
        self.owner = owner
        self.command = command
        self.kwargs = kwargs
        self.index = len(owner.processes)
        self.pid = 1000 + self.index
        self.config = yaml.safe_load(Path(command[-1]).read_text(encoding="utf-8"))
        self.returncode = None
        self.terminated = self.killed = self.waited = False
        self.finishes = owner.clock + (2 if owner.failure and self.index == 0 else 5 + self.index % 3)
        if owner.failure and self.index:
            self.finishes = 1000

    def poll(self):
        if self.returncode is None and self.owner.clock >= self.finishes:
            self.returncode = 7 if self.owner.failure and self.index == 0 else 0
        return self.returncode

    def terminate(self):
        self.terminated = True
        if not self.owner.ignore_terminate:
            self.returncode = -15

    def wait(self, timeout=None):
        self.waited = True
        if self.returncode is None and timeout is not None:
            raise subprocess.TimeoutExpired(self.command, timeout)
        if self.returncode is None:
            self.returncode = -9 if self.killed else 0
        return self.returncode

    def kill(self):
        self.killed = True
        self.returncode = -9


class FakeProcesses:
    def __init__(self, *, failure=False, ignore_terminate=False, launch_error=None, interrupt=False):
        self.clock = 0
        self.failure = failure
        self.ignore_terminate = ignore_terminate
        self.launch_error = launch_error
        self.interrupt = interrupt
        self.processes = []
        self.peak_live = 0
        self.launch_devices = []

    def popen(self, command, **kwargs):
        if self.launch_error is not None and len(self.processes) == self.launch_error:
            raise OSError("synthetic child launch failure")
        process = FakeProcess(self, command, kwargs)
        live = [item for item in self.processes if item.poll() is None]
        device = process.config["platform"]["device"]
        assert device not in {item.config["platform"]["device"] for item in live}
        self.processes.append(process)
        self.launch_devices.append(device)
        self.peak_live = max(self.peak_live, len(live) + 1)
        return process

    def sleep(self, seconds):
        if self.interrupt:
            raise KeyboardInterrupt("synthetic coordinator interruption")
        self.clock += seconds


def make_runner(tmp_path, devices=None):
    orders = [
        ["math", "logic", "translation", "paraphrase"],
        ["logic", "translation", "paraphrase", "math"],
        ["translation", "paraphrase", "math", "logic"],
        ["paraphrase", "math", "logic", "translation"],
    ]
    config = {
        "platform": {"device": "npu:0"},
        "analysis": {"seeds": [17, 29, 43], "sequential_orders": orders},
        "acceleration": {"enabled": True, "training_cpu_threads": 1},
        "training": {"micro_batch_size": 2, "effective_batch_size": 16},
    }
    if devices is not None:
        config["acceleration"]["evaluation_devices"] = devices
    actions = []
    return SimpleNamespace(
        config=config, root=tmp_path / "root", source_root=tmp_path,
        data_root=tmp_path / "data", output_root=tmp_path / "runs",
        close_generation_workers=lambda: actions.append("close_generation_workers"),
        release_model=lambda: actions.append("release_model"),
        taskset=lambda scope: ["math", "logic", "translation", "paraphrase"],
        actions=actions,
    )


def install(monkeypatch, processes):
    monkeypatch.setattr(sequential_pool.subprocess, "Popen", processes.popen)
    monkeypatch.setattr(sequential_pool.time, "sleep", processes.sleep)


def pool_state(runner):
    records = list((runner.root / "scheduler").glob("sequential-orders-*/pool.json"))
    assert len(records) == 1
    return json.loads(records[0].read_text(encoding="utf-8"))


@pytest.mark.parametrize("devices,expected_limit", [(None, 8), (["npu:2", "npu:5"], 2)])
def test_seed_order_coverage_concurrency_bound_and_child_snapshots(monkeypatch, tmp_path, devices, expected_limit):
    runner = make_runner(tmp_path, devices)
    original = deepcopy(runner.config)
    processes = FakeProcesses()
    install(monkeypatch, processes)
    provenance = {"seed": "source-digest", "locked": True}
    assert sequential_pool.parallel_train_orders(runner, provenance) is True
    assert runner.config == original
    assert runner.actions == ["close_generation_workers", "release_model"]
    assert processes.peak_live == expected_limit
    actual = {(item.config["sequential"]["seed"], tuple(item.config["sequential"]["order"]))
              for item in processes.processes}
    expected = {(seed, tuple(order)) for seed in original["analysis"]["seeds"]
                for order in original["analysis"]["sequential_orders"]}
    assert actual == expected
    assert len(processes.processes) == len(expected) == 12
    assert set(processes.launch_devices) == set(devices or [f"npu:{index}" for index in range(8)])
    for process in processes.processes:
        assert process.command[1:4] == ["-u", "-m", "task_vector.sequential_pool"]
        assert process.kwargs["cwd"] == runner.source_root
        assert process.kwargs["stderr"] == subprocess.STDOUT
        assert process.kwargs["stdout"].closed
        for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
            assert process.kwargs["env"][name] == "1"
        child = process.config["sequential"]
        assert child["provenance"] == provenance
        assert child["data_root"] == str(runner.data_root)
        assert child["output_root"] == str(runner.output_root)
        assert child["tasks"] == runner.taskset("mvp")
        assert process.config["training"] == original["training"]
    state = pool_state(runner)
    assert state["status"] == "completed"
    assert state["training_only"] is True
    assert state["stage_order_within_each_order_preserved"] is True
    assert all(job["status"] == "completed" and job["returncode"] == 0 for job in state["jobs"])


@pytest.mark.parametrize("ignore_terminate", [False, True])
def test_failure_stops_scheduling_and_reaps_other_workers(monkeypatch, tmp_path, ignore_terminate):
    runner = make_runner(tmp_path, ["npu:0", "npu:1", "npu:2"])
    processes = FakeProcesses(failure=True, ignore_terminate=ignore_terminate)
    install(monkeypatch, processes)
    with pytest.raises(RuntimeError, match="worker failed"):
        sequential_pool.parallel_train_orders(runner, {})
    assert len(processes.processes) == 3
    for process in processes.processes[1:]:
        assert process.terminated and process.waited
        assert process.killed is ignore_terminate
        assert process.poll() is not None
    assert all(item.kwargs["stdout"].closed for item in processes.processes)
    state = pool_state(runner)
    assert state["status"] == "failed_or_interrupted"
    assert [job["status"] for job in state["jobs"][:3]] == ["failed", "cancelled", "cancelled"]
    assert all(job["status"] == "pending" for job in state["jobs"][3:])


def test_child_launch_failure_reaps_previously_started_workers(monkeypatch, tmp_path):
    runner = make_runner(tmp_path, ["npu:0", "npu:1", "npu:2"])
    processes = FakeProcesses(launch_error=2)
    install(monkeypatch, processes)
    with pytest.raises(OSError, match="launch failure"):
        sequential_pool.parallel_train_orders(runner, {})
    assert len(processes.processes) == 2
    assert all(process.terminated and process.waited for process in processes.processes)
    assert all(process.kwargs["stdout"].closed for process in processes.processes)
    assert pool_state(runner)["status"] == "failed_or_interrupted"


def test_coordinator_interrupt_reaps_all_workers(monkeypatch, tmp_path):
    runner = make_runner(tmp_path, ["npu:0", "npu:1"])
    processes = FakeProcesses(interrupt=True)
    install(monkeypatch, processes)
    with pytest.raises(KeyboardInterrupt, match="coordinator interruption"):
        sequential_pool.parallel_train_orders(runner, {})
    assert all(process.terminated and process.waited for process in processes.processes)
    assert pool_state(runner)["status"] == "failed_or_interrupted"


@pytest.mark.parametrize("enabled,parallel", [(False, True), (True, False)])
def test_disabled_pool_has_no_device_or_process_side_effects(monkeypatch, tmp_path, enabled, parallel):
    runner = make_runner(tmp_path)
    runner.config["acceleration"].update(enabled=enabled, parallel_sequential_orders=parallel)
    processes = FakeProcesses()
    install(monkeypatch, processes)
    assert sequential_pool.parallel_train_orders(runner, {}) is False
    assert not processes.processes and not runner.actions
    assert not runner.root.exists()


def test_child_main_is_training_only_and_never_requests_a_test_split(monkeypatch):
    config = {"platform": {"device": "npu:7"}, "sequential": {"seed": 17}}
    calls = []
    modules = {}
    for name in ("config", "runtime", "sequential"):
        modules[name] = ModuleType(f"task_vector.{name}")
        monkeypatch.setitem(sys.modules, f"task_vector.{name}", modules[name])
    modules["config"].load_config = lambda path: config if path == "snapshot.yaml" else pytest.fail("wrong snapshot")
    modules["runtime"].configure_device = lambda device: "configured:" + device
    modules["sequential"].run_sequential = lambda *args, **kwargs: calls.append((args, kwargs))
    monkeypatch.setattr(sys, "argv", ["sequential_pool.py", "snapshot.yaml"])
    sequential_pool.main()
    assert calls == [((config,), {"device": "configured:npu:7", "split": None})]
