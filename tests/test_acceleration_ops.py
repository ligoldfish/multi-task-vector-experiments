from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
import json
import pytest
import yaml

from task_vector.acceleration import DEFAULT_ACCELERATION, validate_acceleration
from task_vector.config import create_lock, scientific_config, verify_lock
from task_vector.io import atomic_json
from task_vector import training_pool
from ops import prepare_accelerated as prepare_ops


def test_engineering_settings_separately_frozen(tmp_path):
    config = {"platform": {"device": "cpu"}, "training": {"resume_from": None}}
    accelerated = deepcopy(config)
    accelerated["acceleration"] = deepcopy(DEFAULT_ACCELERATION)
    assert scientific_config(accelerated) == scientific_config(config)
    path = tmp_path / "lock.json"
    record = create_lock(accelerated, {}, {"phase": "core"}, path, tmp_path / "source")
    assert record["execution"] == DEFAULT_ACCELERATION
    verify_lock(path, accelerated, {}, tmp_path / "source")
    accelerated["acceleration"]["generation_batch_policy"] = "strict"
    with pytest.raises(ValueError, match="Engineering"):
        verify_lock(path, accelerated, {})


@pytest.mark.parametrize("settings", [
    {"unknown": 1}, {"enabled": "yes"}, {"evaluation_devices": ["npu:0", "npu:0"]},
    {"generation_batch_candidates": [1, 8, 2]}, {"generation_batch_candidates": [True, 2]},
    {"generation_batch_policy": "automatic"}, {"bootstrap_workers": 0},
    {"evaluation_devices": ["cpu", "npu:0"]},
    {"evaluation_devices": ["npu:0:1"]}, {"evaluation_devices": [["npu:0"]]},
    {"training_cpu_threads": 33},
])
def test_invalid_engineering_settings_rejected(settings):
    with pytest.raises(ValueError):
        validate_acceleration(settings)


def test_serial_training_preserves_original_task_sequence(tmp_path):
    calls = []
    runner = SimpleNamespace(config={}, taskset=lambda _: ["math", "logic"],
        train=lambda *args: calls.append(args))
    training_pool.train_phase(runner, "mvp", 17)
    assert calls == [("mvp", "ltv", 17, "math"), ("mvp", "ltv", 17, "logic"),
                     ("mvp", "loda", 17, "math"), ("mvp", "loda", 17, "logic")]


def test_parallel_training_groups_missing_routes_and_keeps_layers(tmp_path, monkeypatch):
    from deploy import launch_pool
    calls, releases = [], []
    config = {"acceleration": deepcopy(DEFAULT_ACCELERATION), "training": {}, "ltv": {"layers": [14]}}
    runner = SimpleNamespace(config=config, root=tmp_path, source_root=tmp_path,
        manifest={"manifest_content_sha256": "data"}, taskset=lambda _: ["math", "logic"],
        base_fingerprint=lambda: "base", release_model=lambda: releases.append("model"),
        close_generation_workers=lambda: releases.append("workers"),
        directory=lambda *args: tmp_path / "missing" / args[1] / args[3])
    class Lock:
        def release(self):
            releases.append("lock")
    monkeypatch.setattr(launch_pool, "_acquire_pool_lock", lambda *args: (Lock(), tmp_path / "owner.json"))
    def run(config, args, devices, output, project, record):
        calls.append((args.routes, args.tasks, args.layers, devices))
        return {"status": "completed"}, 0
    monkeypatch.setattr(launch_pool, "_run_jobs", run)
    training_pool.train_phase(runner, "mvp", 29)
    assert calls == [("ltv,loda", ["math", "logic"], None, list(range(8)))]
    assert releases == ["workers", "model", "lock"]


def _prepare_fixture(tmp_path, monkeypatch):
    previous, project, volume = (tmp_path / name for name in ("previous", "new-project", "volume"))
    previous.mkdir(); project.mkdir(); volume.mkdir()
    monkeypatch.setattr(prepare_ops, "PROJECT", project)
    monkeypatch.setattr(prepare_ops, "DPC_ROOT", volume)
    for root in (previous, project):
        (root / "configs").mkdir(); (root / "ops").mkdir()
        folder = root / "src/task_vector"
        folder.mkdir(parents=True)
        for name in prepare_ops.UNCHANGED:
            (folder / name).write_text("same scientific implementation\n")
    config = yaml.safe_load((Path(__file__).resolve().parents[1] / "configs/experiment.yaml").read_text(encoding="utf-8"))
    old_root = volume / "original"
    old_root.mkdir()
    config["platform"]["persistent_root"] = str(old_root)
    (previous / "configs/modelmate.local.yaml").write_text(yaml.safe_dump(config), encoding="utf-8")
    python = volume / "python"; python.write_text("runtime")
    atomic_json(previous / "ops/settings.local.json", {"runtime_python": str(python), "cann_env": None, "expected_devices": 8})
    data = old_root / "data"
    manifest = {"status": "complete", "tasks": {}}
    for task in config["data"]["tasks"] + config["data"]["extension_tasks"] + config["data"]["auxiliary_tasks"]:
        folder = data / task
        folder.mkdir(parents=True)
        manifest["tasks"][task] = {"files": {}}
        for split in ("train", "tune", "discovery", "test", "test_formal"):
            path = folder / f"{split}.jsonl"
            path.write_text('{"id":"unchanged"}\n', encoding="utf-8")
            manifest["tasks"][task]["files"][split] = {"path": str(path), "sha256": prepare_ops.file_sha256(path), "count": 1}
    manifest["manifest_content_sha256"] = prepare_ops._data_digest(manifest)
    atomic_json(data / "manifest.json", manifest)
    return previous, project, volume, old_root


def test_prepare_restart_preserves_split_bytes_and_previous_artifacts(tmp_path, monkeypatch):
    previous, project, volume, old_root = _prepare_fixture(tmp_path, monkeypatch)
    before = (old_root / "data/manifest.json").read_bytes()
    destination = volume / "accelerated"
    result = prepare_ops.prepare(previous, destination, batch_policy="validated")
    assert result["training_reuse"] is False
    assert result["scientific_settings_equal_except_placement"]
    assert (old_root / "data/manifest.json").read_bytes() == before
    assert (destination / "data/math/train.jsonl").read_bytes() == (old_root / "data/math/train.jsonl").read_bytes()
    manifest = json.loads((destination / "data/manifest.json").read_text())
    assert manifest["manifest_content_sha256"] == prepare_ops._data_digest(manifest)
    assert manifest["tasks"]["math"]["files"]["train"]["path"] == str(destination / "data/math/train.jsonl")
    assert not (destination / "runs/training").exists()
    with pytest.raises(FileExistsError):
        prepare_ops.prepare(previous, destination)


def test_prepare_rejects_modified_scientific_preimage_before_any_write(tmp_path, monkeypatch):
    previous, project, volume, old_root = _prepare_fixture(tmp_path, monkeypatch)
    (project / "src/task_vector/training.py").write_text("modified")
    destination = volume / "accelerated"
    with pytest.raises(ValueError, match="preimage"):
        prepare_ops.prepare(previous, destination)
    assert not destination.exists()


def test_prepare_check_only_has_no_side_effect(tmp_path, monkeypatch):
    previous, project, volume, old_root = _prepare_fixture(tmp_path, monkeypatch)
    destination = volume / "accelerated"
    assert prepare_ops.prepare(previous, destination, check_only=True)["check_only"]
    assert not destination.exists()
    assert not (project / "configs/modelmate.local.yaml").exists()
