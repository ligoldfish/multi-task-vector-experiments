import json
import pytest
from task_vector.cli import parser, main
from task_vector.config import load_config, create_lock, verify_lock
from task_vector.packaging import create_package
from task_vector.io import object_sha256


def test_cli_inventory_without_torch_model_or_npu(tmp_path):
    assert main(["inventory", "--config", "configs/experiment.yaml", "--set", f"platform.persistent_root={tmp_path.as_posix()}"]) == 0
    manifest = json.loads((tmp_path / "runs" / "experiment_inventory.json").read_text())
    assert manifest["budget"]["core_test_primary_generations"] == 9216
    assert any(row["requirement"] == "E6" for row in manifest["inventory"])
    assert manifest["scientific_success_assumed"] is False


def test_protocol_lock_detects_config_data_source_change(tmp_path):
    config = load_config("configs/experiment.yaml")
    source = tmp_path / "source"
    (source / "src").mkdir(parents=True)
    (source / "src" / "x.py").write_text("x=1")
    path = tmp_path / "lock.json"
    create_lock(config, {"ids": [1, 2]}, {"q": 0}, path, source)
    verify_lock(path, config, {"ids": [1, 2]}, source)
    with pytest.raises(ValueError, match="cannot be overwritten"):
        create_lock(config, {"ids": [1, 2]}, {"q": 1}, path, source)
    with pytest.raises(ValueError, match="Dataset differs"):
        verify_lock(path, config, {"ids": [2, 3]}, source)
    (source / "src" / "x.py").write_text("x=2")
    with pytest.raises(ValueError, match="Source files differ"):
        verify_lock(path, config, {"ids": [1, 2]}, source)


def test_scientific_override_cannot_silently_change_budget():
    with pytest.raises(ValueError, match="supplement_label"):
        load_config("configs/experiment.yaml", {"training.learning_rate": .005})


def test_package_excludes_runtime_models_and_venv(tmp_path):
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "x.py").write_text("x=1")
    (tmp_path / ".venv").mkdir()
    (tmp_path / ".venv" / "secret").write_text("no")
    result = create_package(tmp_path, tmp_path / "bundle.zip")
    import zipfile
    with zipfile.ZipFile(result["path"]) as archive:
        assert "src/x.py" in archive.namelist()
        assert not any(".venv" in x for x in archive.namelist())
        assert "PACKAGE_MANIFEST.json" in archive.namelist()


@pytest.mark.parametrize("phase,split,result,code", [
    ("core", "test", {"completed": False, "stop_reason": "validity_or_consistency_gate"}, 42),
    ("heldout", "test", {"completed": False, "folds": {"math": {"completed": False}}}, 42),
    ("all", "test_formal", {"completed": False, "primary": {"completed": False}}, 42),
    ("formal", "test_formal", {"completed": True, "evaluation_only": True}, 0),
])
def test_cli_workflow_exit_status_and_runner_release(tmp_path, monkeypatch, capsys, phase, split, result, code):
    events = []
    class FakeRunner:
        def __init__(self, config, *, source_root):
            events.append(("construct", config["platform"]["device"]))
        def release_model(self):
            events.append(("release",))
    def run(runner, selected_phase, scope, selected_split):
        events.append(("run", selected_phase, scope, selected_split))
        return result
    monkeypatch.setattr("task_vector.runner.ExperimentRunner", FakeRunner)
    monkeypatch.setattr("task_vector.workflow.run_phase", run)
    actual = main(["run", "--config", "configs/experiment.yaml", "--phase", phase, "--split", split,
                   "--device", "cpu", "--set", f"platform.persistent_root={tmp_path.as_posix()}"])
    assert actual == code
    assert events == [("construct", "cpu"), ("run", phase, "mvp", split), ("release",)]
    assert json.loads(capsys.readouterr().out)["result"]["completed"] is result["completed"]


def test_cli_releases_model_when_workflow_raises(tmp_path, monkeypatch):
    events = []
    class FakeRunner:
        def __init__(self, config, *, source_root):
            pass
        def release_model(self):
            events.append("release")
    def reject(*args):
        raise ValueError("Frozen artifact was replaced")
    monkeypatch.setattr("task_vector.runner.ExperimentRunner", FakeRunner)
    monkeypatch.setattr("task_vector.workflow.run_phase", reject)
    with pytest.raises(ValueError, match="Frozen artifact"):
        main(["run", "--config", "configs/experiment.yaml", "--phase", "formal", "--device", "cpu",
              "--set", f"platform.persistent_root={tmp_path.as_posix()}"])
    assert events == ["release"]


@pytest.mark.parametrize("field", ["passed", "passes"])
def test_cli_non_workflow_failure_returns_42(tmp_path, monkeypatch, field):
    monkeypatch.setattr("task_vector.runtime.preflight", lambda *args: {field: False})
    assert main(["preflight", "--config", "configs/experiment.yaml", "--device", "cpu",
                 "--set", f"platform.persistent_root={tmp_path.as_posix()}"]) == 42
