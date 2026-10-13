from copy import deepcopy
from types import SimpleNamespace
import pytest
from ops import modelmate_job, monitor_accelerated
from task_vector.data import Sample
from task_vector.io import atomic_json, read_json


def test_monitor_reads_pool_and_panel_without_tensor_loading(tmp_path, capsys):
    atomic_json(tmp_path / "runs/platform_jobs/job/metadata.json", {"status": "running"})
    atomic_json(tmp_path / "scheduler/pool/pool.json", {"status": "running", "seed": 17,
        "jobs": [{"status": "completed"}, {"status": "running", "device": "npu:1", "task": "math"}]})
    atomic_json(tmp_path / "runs/training/independent/ltv/17/math/training.json", {})
    atomic_json(tmp_path / "runs/engineering/generation/progress.json", {
        "status": "running", "completed_workers": 2, "expected_workers": 8,
        "completed_samples": 64, "total_samples": 256, "panel_eta_seconds": 10})
    # A tensor with invalid contents must be irrelevant to the JSON monitor.
    (tmp_path / "runs/last.pt").write_bytes(b"not a tensor")
    monitor_accelerated.snapshot(tmp_path)
    text = capsys.readouterr().out
    assert "UNKNOWN" in text and "npu:1" in text and "256" in text
    assert "0:00:10" in text and "训练记录= 1" in text


def _probe_runner(tmp_path, mismatch=False):
    config = {"platform": {"device": "cpu"}, "training": {},
        "evaluation": {"max_new_tokens": {"math": 512}},
        "acceleration": {"enabled": True, "evaluation_devices": [f"npu:{i}" for i in range(8)]}}
    source = Sample("math:train:1", "math", "question", "answer", "group", "train")
    calls = []
    def generate(samples, representation, current, **kwargs):
        assert representation is None and len({s.id for s in samples}) == 8
        assert all(s.official_split == "train" and s.prompt == source.prompt for s in samples)
        assert current["evaluation"]["max_new_tokens"]["math"] == 3
        assert kwargs["allow_calibration"] is False
        calls.append(kwargs)
        return [{"prediction_token_ids": [1, 2, 9 if mismatch and i else 3]} for i in range(8)], {"probe": True}
    def samples(task, split):
        assert (task, split) == ("math", "train")
        return [source]
    runner = SimpleNamespace(config=config, output_root=tmp_path / "runs", source_root=tmp_path,
        samples=samples, _parallel_generation=lambda: SimpleNamespace(generate=generate))
    return runner, calls


def test_cross_card_probe_uses_only_train_and_does_not_change_budgets(tmp_path):
    runner, calls = _probe_runner(tmp_path)
    original = deepcopy(runner.config)
    modelmate_job._acceleration_probe(runner, tmp_path / "job")
    assert runner.config == original and len(calls) == 1
    record = read_json(tmp_path / "job/acceleration_probe.json")
    assert record["passed"] and record["new_test_access"] is False
    assert not (runner.output_root / "test_access_ledger.json").exists()
    assert not (runner.output_root / "engineering/batch-calibration").exists()


def test_cross_card_mismatch_is_rejected_before_any_training(tmp_path):
    runner, _ = _probe_runner(tmp_path, mismatch=True)
    with pytest.raises(AssertionError, match="across cards"):
        modelmate_job._acceleration_probe(runner, tmp_path / "job")
    assert not (tmp_path / "job/acceleration_probe.json").exists()
