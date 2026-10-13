"""Small integration checks for immutable exports and formal prediction reuse."""
from copy import deepcopy
from pathlib import Path
import os

import pytest
import torch

from task_vector.config import scientific_config, source_manifest
from task_vector.data import Sample
from task_vector.io import atomic_json, object_sha256, read_json
from task_vector.representations import Representation
from task_vector.runner import ExperimentRunner


@pytest.fixture
def runner(tmp_path, monkeypatch):
    if os.name == "nt":
        tmp_path = Path("\\\\?\\" + str(tmp_path))
    source = tmp_path / "source"
    source.mkdir()
    model = tmp_path / "model"
    model.mkdir()
    (model / "config.json").write_text("{}", encoding="utf-8")
    config = {"platform": {"persistent_root": str(tmp_path), "device": "cpu"},
        "runtime": {"data_root": str(tmp_path / "data"), "output_root": str(tmp_path / "runs")},
        "model": {"id": "tiny", "path": str(model)}, "data": {"tasks": ["math"]},
        "training": {"max_prompt_tokens": 512}, "loda": {}, "ltv": {"layers": [14]},
        "evaluation": {}, "metrics": {}, "analysis": {"seeds": [17]}}
    result = ExperimentRunner(config, source_root=source)
    result._manifest = {"status": "complete", "manifest_content_sha256": "prepared-data"}
    monkeypatch.setattr(result, "base_fingerprint", lambda: "base-fingerprint")
    return result


def rep(value=1.):
    return Representation("activation", "math", 17, values={"layer14": torch.tensor([value, 2.])},
                          metadata={"base_fingerprint": "base-fingerprint"})


def sample(identity="math:0", prompt="problem", target="42"):
    return Sample(identity, "math", prompt, target, identity, "official-evaluation", {"source": "official"})


@pytest.mark.parametrize("replace_before_freeze", [False, True])
def test_selected_candidate_content_replacement_rejected(runner, replace_before_freeze):
    for route in ("ltv", "loda"):
        directory = runner.directory("mvp", route, 17, "math")
        directory.mkdir(parents=True)
        for name in ("raw.pt", "post.pt", "initial.pt"):
            rep().save(directory / name)
    candidate = runner.output_root / "chosen.pt"
    chosen = rep()
    chosen.save(candidate)
    decisions = {"nested": [{"representation_path": str(candidate),
                              "representation_digest": runner.representation_digest(chosen)}]}
    if replace_before_freeze:
        rep(7.).save(candidate)
        with pytest.raises(ValueError, match="differs from discovery"):
            runner.freeze_artifacts("mvp", decisions=decisions)
    else:
        artifacts = runner.freeze_artifacts("mvp", decisions=decisions)
        runner._protocol_cache["fake-lock"] = (None, {"discovery_decisions": {"artifacts": artifacts}})
        assert torch.equal(runner.load_representation(candidate).values["layer14"], chosen.values["layer14"])
        rep(7.).save(candidate)
        with pytest.raises(ValueError, match="Frozen representation artifact changed"):
            runner.load_representation(candidate)


def test_interrupted_export_commit_repairs_without_model_load_or_training(runner, monkeypatch):
    directory = runner.directory("mvp", "ltv", 17, "math")
    directory.mkdir(parents=True)
    signature = object_sha256({"config": scientific_config(runner.config), "route": "ltv", "seed": 17,
        "scope": "independent", "task": "math", "data": runner.manifest["manifest_content_sha256"],
        "base": runner.base_fingerprint(), "sources": source_manifest(runner.source_root)})
    result = {"status": "exported_uncommitted", "steps": 1,
              "provenance": {"request_signature": signature}}
    for field, name in (("raw_path", "raw.pt"), ("post_path", "post.pt"), ("initial_path", "initial.pt")):
        rep().save(directory / name)
        result[field] = str(directory / name)
    atomic_json(directory / "training-state.json", result)
    monkeypatch.setattr(runner, "samples", lambda task, split: [sample()])
    monkeypatch.setattr(runner, "load_model", lambda: pytest.fail("export repair loaded model"))
    monkeypatch.setattr("task_vector.training.train_task", lambda *a, **k: pytest.fail("export repair retrained"))
    original_save = Representation.save
    interrupted = False

    def save(self, path):
        nonlocal interrupted
        if Path(path).name == "post.pt" and not interrupted:
            interrupted = True
            raise RuntimeError("interrupted metadata export")
        return original_save(self, path)

    monkeypatch.setattr(Representation, "save", save)
    with pytest.raises(RuntimeError, match="interrupted metadata export"):
        runner.train("mvp", "ltv", 17, "math")
    assert not (directory / "training.json").exists()
    repaired = runner.train("mvp", "ltv", 17, "math")
    assert repaired["status"] == "complete" and repaired["steps"] == 1
    assert repaired["completion_schema"] == 2
    for field in ("raw_path", "post_path", "initial_path"):
        exported = Representation.load(repaired[field])
        assert exported.metadata["base_fingerprint"] == "base-fingerprint"
        assert exported.metadata["training_request_signature"] == signature
    assert runner.train("mvp", "ltv", 17, "math")["request_signature"] == signature


@pytest.fixture
def evaluation(runner, monkeypatch):
    calls = []
    monkeypatch.setattr(runner, "verify_evaluation_protocol", lambda *a: {"lock_sha256": "frozen"})
    monkeypatch.setattr(runner, "record_final_access", lambda *a, **k: None)

    def load():
        runner.model, runner.tokenizer, runner.device = object(), object(), "cpu"

    def generate(model, tokenizer, samples, representation, config, device, **kwargs):
        calls.append([(s.id, s.prompt) for s in samples])
        return [{**s.to_dict(), "prediction": f"prediction-for-{s.prompt}", "variant": "canonical"}
                for s in samples]

    def score(task, rows, config):
        return {"metrics": {"accuracy": 1.}, "primary_metric": "accuracy", "count": len(rows),
                "rows": deepcopy(rows), "sample_scores": [1.] * len(rows)}

    monkeypatch.setattr(runner, "load_model", load)
    monkeypatch.setattr("task_vector.evaluation.generate_predictions", generate)
    monkeypatch.setattr("task_vector.metrics.score_predictions", score)
    monkeypatch.setattr("task_vector.runner.synchronize", lambda *a: None)
    monkeypatch.setattr("task_vector.runner.memory_evidence", lambda *a: {})
    return runner, calls


def test_formal_reuses_only_identical_sample_content(evaluation):
    runner, calls = evaluation
    runner.evaluate(None, "math", "test", label="base", samples=[sample()])
    panel = [sample(), sample("math:1", "new problem")]
    formal = runner.evaluate(None, "math", "test_formal", label="base", samples=panel)
    assert calls == [[("math:0", "problem")], [("math:1", "new problem")]]
    assert formal["reused_predictions"] == 1 and formal["newly_generated"] == 1
    saved = read_json(runner.output_root / "evaluation" / "mvp" / "17" / "test" / "canonical" / "base" / "math" / "scores.json")
    assert saved["generation_sample_digests"]["math:0"] == object_sha256(sample().to_dict())


@pytest.mark.parametrize("changed", ["prompt", "target", "group_id", "metadata"])
def test_same_id_with_changed_content_is_generated_again(evaluation, changed):
    runner, calls = evaluation
    runner.evaluate(None, "math", "test", label="base", samples=[sample()])
    current = sample()
    setattr(current, changed, {"source": "different"} if changed == "metadata" else "changed")
    formal = runner.evaluate(None, "math", "test_formal", label="base", samples=[current])
    assert len(calls) == 2 and calls[-1][0][0] == current.id
    assert formal["reused_predictions"] == 0 and formal["newly_generated"] == 1
    assert formal["rows"][0][changed] == getattr(current, changed)


@pytest.mark.parametrize("source", ["data", "source", "base", "representation", "semantic_asset"])
def test_formal_rejects_mismatched_generation_sources(evaluation, source):
    runner, calls = evaluation
    runner.evaluate(None, "math", "test", label="base", samples=[sample()])
    old = runner.output_root / "evaluation" / "mvp" / "17" / "test" / "canonical" / "base" / "math" / "scores.json"
    saved = read_json(old)
    saved["generation_provenance"][source] = "changed-source"
    atomic_json(old, saved)
    formal = runner.evaluate(None, "math", "test_formal", label="base", samples=[sample()])
    assert len(calls) == 2
    assert formal["reused_predictions"] == 0 and formal["newly_generated"] == 1
