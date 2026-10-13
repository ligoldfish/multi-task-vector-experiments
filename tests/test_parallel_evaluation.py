"""CPU-only orchestration tests: spawn, completeness, barriers and lifecycle."""
from copy import deepcopy
import multiprocessing
import os
import time

import pytest

from task_vector.parallel_evaluation import (
    PersistentGenerationExecutor, merge_shards, partition_samples,
)


class Example:
    def __init__(self, identity):
        self.id = identity

    def to_dict(self):
        return {"id": self.id, "task": "math", "prompt": self.id,
                "target": "42", "group_id": self.id, "official_split": "train",
                "metadata": {}}


def fake_worker(rank, device, config, requests, results, directory):
    """Exercise the actual spawn/queue protocol without loading a model."""
    results.put({"kind": "ready", "rank": rank})
    calibrations = 0
    while True:
        request = requests.get()
        if request is None:
            return
        if request["kind"] == "calibrate":
            calibrations += 1
            results.put({"kind": "result", "operation": request["operation"], "rank": rank,
                         "calibration": {"mode": "validated", "effective_batch_size": 4}})
        else:
            # Deliberately complete higher ranks first.
            time.sleep(.01 * (3 - rank))
            rows = [{"id": sample["id"], "prediction": sample["prompt"]}
                    for sample in request["samples"]]
            if config.get("acceleration", {}).get("test_bad_worker") and rank == 0:
                rows[0]["id"] = "unexpected"
            results.put({"kind": "result", "operation": request["operation"], "rank": rank,
                         "rows": rows, "receipt": {"pid": os.getpid(),
                         "calibration_calls": calibrations,
                         "batch_size": request["policy"]["effective_batch_size"]}})


@pytest.fixture
def config():
    return {"platform": {"device": "cpu"}, "model": {"id": "dummy"}, "evaluation": {},
            "acceleration": {"enabled": True, "evaluation_devices": ["cpu:0", "cpu:1", "cpu:2"],
                             "generation_batch_policy": "validated", "heartbeat_seconds": 30}}


@pytest.mark.parametrize("count,workers", [(0, 8), (1, 8), (17, 8), (256, 8), (5, 2)])
def test_partition_is_complete_ordered_and_balanced(count, workers):
    samples = [Example(str(index)) for index in range(count)]
    shards = partition_samples(samples, workers)
    assert [sample.id for shard in shards for sample in shard] == [sample.id for sample in samples]
    if shards:
        sizes = [len(shard) for shard in shards]
        assert max(sizes) - min(sizes) <= 1
        assert min(sizes) > 0


def test_partition_rejects_duplicate_input():
    with pytest.raises(ValueError, match="unique"):
        partition_samples([Example("same"), Example("same")], 2)


@pytest.mark.parametrize("rows", [
    [[{"id": "0"}], [{"id": "0"}, {"id": "1"}]],
    [[{"id": "0"}]],
    [[{"id": "0"}, {"id": "1"}, {"id": "extra"}]],
])
def test_merge_rejects_incomplete_duplicate_or_extra(rows):
    with pytest.raises(ValueError):
        merge_shards([Example("0"), Example("1")], rows)


def test_merge_returns_canonical_sample_order():
    assert [row["id"] for row in merge_shards([Example("0"), Example("1")],
        [[{"id": "1"}], [{"id": "0"}]])] == ["0", "1"]


def test_spawn_workers_are_persistent_and_policy_is_not_reselected(config, tmp_path):
    samples = [Example(str(index)) for index in range(11)]
    pool = PersistentGenerationExecutor(config, tmp_path, worker_target=fake_worker)
    try:
        rows, first = pool.generate(samples, None, config, task="math", calibration_samples=samples[:2])
        assert [row["id"] for row in rows] == [sample.id for sample in samples]
        assert first["phase_barrier"] == "all_assigned_shards_completed_before_scoring"
        _, second = pool.generate(samples, None, config, task="math", calibration_samples=samples[:2])
        assert [item["receipt"]["pid"] for item in first["shards"]] == [
            item["receipt"]["pid"] for item in second["shards"]]
        assert second["shards"][0]["receipt"]["calibration_calls"] == 1
        assert all(item["receipt"]["batch_size"] == 4 for item in second["shards"])
    finally:
        pool.close()
    assert all(not process.is_alive() for process in pool.processes)


def test_unseen_target_never_uses_calibration_data(config, tmp_path):
    samples = [Example(str(index)) for index in range(4)]
    pool = PersistentGenerationExecutor(config, tmp_path, worker_target=fake_worker)
    try:
        _, receipt = pool.generate(samples, None, config, task="summary", allow_calibration=False)
        assert receipt["policy"]["effective_batch_size"] == 1
        assert all(item["receipt"]["calibration_calls"] == 0 for item in receipt["shards"])
    finally:
        pool.close()


def test_bad_worker_result_fails_closed(config, tmp_path):
    config["acceleration"]["test_bad_worker"] = True
    samples = [Example(str(index)) for index in range(4)]
    pool = PersistentGenerationExecutor(config, tmp_path, worker_target=fake_worker)
    with pytest.raises(ValueError, match="outside its assigned shard"):
        pool.generate(samples, None, config, task="math", allow_calibration=False)
    assert pool.closed and all(not process.is_alive() for process in pool.processes)


def test_policy_survives_training_pool_restart_and_rejects_changes(config, tmp_path):
    samples = [Example(str(index)) for index in range(4)]
    first = PersistentGenerationExecutor(config, tmp_path, worker_target=fake_worker)
    try:
        first.generate(samples, None, config, task="math", calibration_samples=samples[:2])
    finally:
        first.close()
    second = PersistentGenerationExecutor(config, tmp_path, worker_target=fake_worker)
    try:
        _, receipt = second.generate(samples, None, config, task="math", calibration_samples=samples[:2])
        assert receipt["shards"][0]["receipt"]["calibration_calls"] == 0
        changed = deepcopy(config)
        changed["acceleration"]["generation_batch_candidates"] = [1, 2]
        with pytest.raises(ValueError, match="contract differs"):
            second.generate(samples, None, changed, task="math", calibration_samples=samples[:2])
    finally:
        second.close()


def test_duplicate_authorized_samples_fail_before_workers_start(config, tmp_path):
    pool = PersistentGenerationExecutor(config, tmp_path, worker_target=fake_worker)
    try:
        with pytest.raises(ValueError, match="unique"):
            pool.generate([Example("0"), Example("0")], None, config, task="math")
        assert not pool.processes
    finally:
        pool.close()


def test_native_merged_weights_restore_exactly_after_exception(tmp_path):
    import torch
    from task_vector.io import file_sha256
    from task_vector.parallel_evaluation import merged_projection_context

    class Toy(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.attention = torch.nn.ModuleDict({
                name: torch.nn.Linear(2, 2, bias=False, dtype=torch.bfloat16)
                for name in ("q_proj", "k_proj", "v_proj")})

    model = Toy()
    original = {name: parameter.detach().clone() for name, parameter in model.named_parameters()}
    weights = {name.removesuffix(".weight"): value + 1 for name, value in original.items()}
    path = tmp_path / "merged.pt"
    torch.save(weights, path)
    with pytest.raises(RuntimeError, match="forced"):
        with merged_projection_context(model, path, file_sha256(path)):
            for name, parameter in model.named_parameters():
                assert torch.equal(parameter, weights[name.removesuffix(".weight")])
            raise RuntimeError("forced")
    for name, parameter in model.named_parameters():
        assert torch.equal(parameter, original[name])


def test_native_merged_weights_reject_bad_hash_before_modification(tmp_path):
    from task_vector.parallel_evaluation import merged_projection_context
    path = tmp_path / "merged.pt"
    path.write_bytes(b"untrusted")
    with pytest.raises(ValueError, match="hash differs"):
        with merged_projection_context(None, path, "wrong"):
            pytest.fail("bad artifact was entered")


def test_native_merge_cache_reuses_verification_but_invalidates_changed_file(tmp_path, monkeypatch):
    import torch
    import task_vector.parallel_evaluation as module

    model = torch.nn.Module()
    model.attention = torch.nn.ModuleDict({
        name: torch.nn.Linear(2, 2, bias=False, dtype=torch.bfloat16)
        for name in ("q_proj", "k_proj", "v_proj")})
    original = {name: parameter.detach().clone() for name, parameter in model.named_parameters()}
    path = tmp_path / "merged.pt"
    torch.save({name.removesuffix(".weight"): value + 1 for name, value in original.items()}, path)
    actual_hash = module.file_sha256
    calls = []

    def counted(filename):
        calls.append(str(filename))
        return actual_hash(filename)

    monkeypatch.setattr(module, "file_sha256", counted)
    expected = actual_hash(path)
    for _ in range(2):
        with module.merged_projection_context(model, path, expected):
            pass
    assert len(calls) == 1
    torch.save({name.removesuffix(".weight"): value + 2 for name, value in original.items()}, path)
    with pytest.raises(ValueError, match="hash differs"):
        with module.merged_projection_context(model, path, expected):
            pytest.fail("changed file was reused")
    assert len(calls) == 2
    for name, parameter in model.named_parameters():
        assert torch.equal(parameter, original[name])


def test_runner_parallel_generation_keeps_coordinator_protocol_and_publication(config, tmp_path, monkeypatch):
    from task_vector.data import Sample
    from task_vector.runner import ExperimentRunner

    config["platform"]["persistent_root"] = str(tmp_path)
    config["data"] = {"tasks": ["math"]}
    config["training"] = {}
    source = tmp_path / "source"
    source.mkdir()
    runner = ExperimentRunner(config, source_root=source)
    runner._manifest = {"status": "complete", "manifest_content_sha256": "data"}
    monkeypatch.setattr(runner, "base_fingerprint", lambda: "base")
    sample = Sample("math:1", "math", "problem", "42", "group:1", "test")
    tune = Sample("math:tune", "math", "source problem", "42", "group:2", "train")
    events = []

    def verify(*args):
        events.append("protocol")
        return {"lock_sha256": "frozen"}

    class Executor:
        def generate(self, samples, representation, cfg, **kwargs):
            events.append("generation")
            assert kwargs["calibration_samples"] == [tune]
            return [{**item.to_dict(), "prediction": "42"} for item in samples], {"devices": ["cpu"]}

    def score(task, rows, cfg):
        events.append("score")
        return {"rows": rows, "count": len(rows), "metrics": {"accuracy": 1},
                "primary_metric": "accuracy", "sample_scores": [1]}

    monkeypatch.setattr(runner, "verify_evaluation_protocol", verify)
    monkeypatch.setattr(runner, "_parallel_generation", lambda: Executor())
    monkeypatch.setattr(runner, "samples", lambda task, split: [tune])
    monkeypatch.setattr(runner, "load_model", lambda: pytest.fail("coordinator loaded inference model"))
    monkeypatch.setattr("task_vector.metrics.score_predictions", score)
    monkeypatch.setattr("task_vector.runner.synchronize", lambda *args: None)
    monkeypatch.setattr("task_vector.runner.memory_evidence", lambda *args: {})
    result = runner.evaluate(None, "math", "test", label="base", samples=[sample])
    assert events == ["protocol", "generation", "score"]
    assert result["execution_receipt"]["devices"] == ["cpu"]
    assert result["generation_provenance"]["execution"]
    assert (runner.output_root / "test_access_ledger.json").is_file()
    assert (runner.output_root / "evaluation/mvp/17/test/canonical/base/math/scores.json").is_file()


def test_coordinator_model_load_closes_generation_workers(config, tmp_path, monkeypatch):
    from task_vector.runner import ExperimentRunner
    config["platform"]["persistent_root"] = str(tmp_path)
    runner = ExperimentRunner(config)
    closed = []

    class Executor:
        def close(self):
            closed.append(True)

    runner._generation_executor = Executor()
    runner.model, runner.tokenizer = object(), object()
    runner._loaded_base_fingerprint = "base"
    monkeypatch.setattr(runner, "base_fingerprint", lambda: "base")
    runner.load_model()
    assert closed == [True] and runner._generation_executor is None


def test_close_kills_and_reaps_worker_that_ignores_terminate(config, tmp_path):
    pool = PersistentGenerationExecutor(config, tmp_path, worker_target=fake_worker)
    events = []

    class StubbornProcess:
        pid = 123

        def __init__(self):
            self.alive = True

        def is_alive(self):
            return self.alive

        def join(self, timeout=None):
            events.append(("join", timeout))

        def terminate(self):
            events.append(("terminate", None))

        def kill(self):
            events.append(("kill", None))
            self.alive = False

    process = StubbornProcess()
    pool.processes.append(process)
    pool.close()
    assert not process.is_alive()
    assert [name for name, _ in events] == ["join", "terminate", "join", "kill", "join"]
    assert events[-1] == ("join", None)
    from task_vector.io import read_json
    assert read_json(pool.directory / "closed.json")["all_workers_reaped"] is True


def test_protocol_cache_revalidates_changed_acceleration_settings(config, tmp_path, monkeypatch):
    from task_vector.runner import ExperimentRunner
    config["platform"]["persistent_root"] = str(tmp_path)
    runner = ExperimentRunner(config)
    runner._manifest = {"status": "complete", "manifest_content_sha256": "data"}
    path = tmp_path / "frozen.json"
    path.write_text("{}", encoding="utf-8")
    assets = {"base": "base"}
    monkeypatch.setattr(runner, "asset_fingerprints", lambda: assets)
    monkeypatch.setattr(runner, "verify_artifacts", lambda artifacts: None)
    monkeypatch.setattr("task_vector.runner.source_manifest", lambda source: {})
    calls = []

    def verify(*args):
        calls.append(True)
        if len(calls) > 1:
            raise ValueError("Frozen execution settings differ")
        return {"discovery_decisions": {"asset_fingerprints": assets, "artifacts": {}}}

    monkeypatch.setattr("task_vector.runner.verify_lock", verify)
    runner.verify_frozen_protocol(path)
    runner.verify_frozen_protocol(path)
    assert len(calls) == 1
    runner.config["acceleration"]["worker_cpu_threads"] = 3
    with pytest.raises(ValueError, match="execution settings differ"):
        runner.verify_frozen_protocol(path)
    assert len(calls) == 2
