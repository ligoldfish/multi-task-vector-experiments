"""Control-workflow wiring, with actual callbacks and no model training."""
from copy import deepcopy
from pathlib import Path
import os

import pytest

from task_vector import workflow
from task_vector.data import Sample
from task_vector.io import atomic_json, file_sha256, object_sha256, read_json, write_jsonl


TASKS = ["math", "logic", "translation", "paraphrase"]


class ControlRunner:
    def __init__(self, root):
        self.output_root, self.data_root, self.source_root = root / "runs", root / "data", root / "source"
        self.source_root.mkdir(parents=True)
        self.config = {"analysis": {"seeds": [17], "ltv_layer_ablations": [[7], [21], [7, 14, 21]]},
            "platform": {"device": "cpu"}, "training": {"max_prompt_tokens": 512},
            "evaluation": {"do_sample": False}, "metrics": {"fixture": True}}
        self.manifest = {"status": "complete", "manifest_content_sha256": "data-sha"}
        self.device, self.events, self.forbid_training = "cpu", [], False

    def taskset(self, scope):
        assert scope == "mvp"
        return list(TASKS)

    def verify_evaluation_protocol(self, scope, seed):
        self.events.append(("main_lock", scope, seed))
        return {"lock_sha256": "main-lock-sha"}

    def verify_frozen_protocol(self, path):
        self.events.append(("layer_lock", Path(path).name))
        return read_json(path)

    def train(self, *args):
        assert not self.forbid_training, "evaluation-only retrained a layer control"
        self.events.append(("train", *args))

    def freeze_artifacts(self, scope, seeds, *, layers):
        self.events.append(("freeze", scope, seeds, layers))
        return {f"layer-{'-'.join(map(str, layers))}": {"sha256": "export-sha"}}

    def core_panel(self, *args):
        self.events.append(("layer_panel", *args))
        return {"completed": True}

    def evaluate(self, rep, task, split, **kwargs):
        assert rep is None and kwargs["label"] == "base"
        self.events.append(("baseline", task, split))
        return {"task": task, "split": split, "metrics": {"accuracy": .25}, "rows": []}

    def release_model(self):
        self.events.append(("release",))

    def samples(self, *args):
        raise AssertionError("sequential callback reread runner.samples instead of using supplied samples")

    def asset_fingerprints(self):
        return {"base": "base-sha", "semantic": "semantic-sha"}

    def base_fingerprint(self):
        return "base-sha"

    def record_final_access(self, task, split, samples, **kwargs):
        self.events.append(("access", task, split, [s.id for s in samples], kwargs))

    def delta(self, score, baseline, task, seed):
        self.events.append(("delta", task, seed, score["count"], baseline["split"]))
        return {"fixture_effect": score["metrics"]["accuracy"] - baseline["metrics"]["accuracy"]}


@pytest.fixture
def controls(tmp_path, monkeypatch):
    if os.name == "nt":
        tmp_path = Path("\\\\?\\" + str(tmp_path))
    runner = ControlRunner(tmp_path)
    state = {"calls": [], "generated": [], "scored": []}
    monkeypatch.setattr(workflow, "_require_gate", lambda r, scope: r.events.append(("gate", scope)))

    def sequential(config, **kwargs):
        state["calls"].append({"config": deepcopy(config), **kwargs})
        return [{"order": TASKS[i:] + TASKS[:i]} for i in range(4)]

    def generate(model, tokenizer, samples, representation, config, device):
        assert representation is None
        assert model == "actual-merged-model" and tokenizer == "tokenizer"
        state["generated"].append([s.id for s in samples])
        return [{**s.to_dict(), "prediction": "correct" if s.id.endswith(":0") else "incorrect"}
                for s in samples]

    def score(task, rows, config):
        state["scored"].append([r["id"] for r in rows])
        return {"metrics": {"accuracy": sum(r["prediction"] == "correct" for r in rows) / len(rows)},
                "primary_metric": "accuracy", "count": len(rows), "rows": deepcopy(rows)}

    monkeypatch.setattr("task_vector.sequential.run_sequential", sequential)
    monkeypatch.setattr("task_vector.evaluation.generate_predictions", generate)
    monkeypatch.setattr("task_vector.metrics.score_predictions", score)
    monkeypatch.setattr("task_vector.runtime.synchronize", lambda *a: None)
    monkeypatch.setattr("task_vector.runtime.memory_evidence", lambda device: {"device": str(device), "peak_bytes": 123})
    return runner, state


def source_sample(index=0, prompt=None):
    return Sample(f"logic:{index}", "logic", prompt or f"premise-{index}", "True",
                  f"group-{index}", "official-evaluation", {"label": "True"})


def callback_config(runner, captured, split):
    config = deepcopy(captured["config"])
    panel = runner.output_root / "sequential" / "seed17" / "order" / "stage1_logic" / "evaluation" / split / "logic"
    config["sequential"].update(order=TASKS, stage=1, stage_signature="stage-sha",
                              merged_weights_sha256="merged-sha", split=split, evaluation_dir=str(panel))
    return config, panel


def write_primary_panel(directory, result):
    atomic_json(directory / "evaluation-request.json", {"stage_signature": "stage-sha", "split": "test"})
    atomic_json(directory / "metrics.json", result)
    write_jsonl(directory / "predictions.jsonl", result["rows"])
    names = ("evaluation-request.json", "metrics.json", "predictions.jsonl")
    atomic_json(directory / "evaluation-completed.json", {"schema_version": 2, "status": "complete",
        "artifacts": {name: file_sha256(directory / name) for name in names}})


def primary_and_formal(controls):
    runner, state = controls
    primary = workflow.run_ablations(runner, "test")
    first = state["calls"][-1]
    config, panel = callback_config(runner, first, "test")
    scored = first["evaluate_fn"]("actual-merged-model", "tokenizer", [source_sample()], config, "cpu")
    write_primary_panel(panel, scored)
    runner.forbid_training = True
    formal = workflow.run_ablations(runner, "test_formal", evaluation_only=True)
    return runner, state, primary, formal, panel


def test_control_wiring_passes_split_provenance_hooks_and_evalonly_flag(controls):
    runner, state = controls
    result = workflow.run_ablations(runner, "test")
    assert result["completed"] and result["split"] == "test"
    assert result["seeds"]["17"]["sequential_orders"] == 4
    first = state["calls"][0]
    assert first["split"] == "test" and first["evaluation_only"] is False and first["device"] == "cpu"
    provenance = first["config"]["sequential"]["provenance"]
    assert provenance["base_fingerprint"] == "base-sha"
    assert provenance["protocol_lock_sha256"] == "main-lock-sha"
    assert provenance["data_manifest_sha256"] == object_sha256(runner.manifest)
    assert provenance["asset_fingerprints"] == runner.asset_fingerprints()
    assert len([event for event in runner.events if event[0] == "train"]) == 12
    assert len([event for event in runner.events if event[0] == "layer_lock"]) == 3
    samples = [source_sample()]
    first["access_fn"]("logic", "test", samples, {"order": TASKS, "stage": 1})
    access = runner.events[-1]
    assert access[:4] == ("access", "logic", "test", ["logic:0"])
    assert access[-1]["lock_hash"] == "main-lock-sha"
    assert access[-1]["scope"] == "mvp" and access[-1]["seed"] == 17
    assert "stage1" in access[-1]["label"]
    runner.forbid_training = True
    workflow.run_ablations(runner, "test_formal", evaluation_only=True)
    assert state["calls"][-1]["split"] == "test_formal"
    assert state["calls"][-1]["evaluation_only"] is True
    assert state["calls"][-1]["config"]["sequential"]["provenance"] == provenance
    assert len([event for event in runner.events if event[0] == "train"]) == 12
    assert len([event for event in runner.events if event[0] == "layer_lock"]) == 6


def test_formal_callback_generates_only_new_ids_and_rescores_whole_panel(controls):
    runner, state, primary, formal, panel = primary_and_formal(controls)
    old_hashes = {str(p): file_sha256(p) for p in panel.iterdir() if p.is_file()}
    captured = state["calls"][-1]
    config, _ = callback_config(runner, captured, "test_formal")
    result = captured["evaluate_fn"]("actual-merged-model", "tokenizer",
                                      [source_sample(), source_sample(1)], config, "cpu")
    assert state["generated"] == [["logic:0"], ["logic:1"]]
    assert state["scored"] == [["logic:0"], ["logic:0", "logic:1"]]
    assert result["metrics"]["accuracy"] == .5 and result["count"] == 2
    assert result["newly_generated"] == result["reused_predictions"] == 1
    assert result["runtime_seconds"] >= 0 and result["memory"]["peak_bytes"] == 123
    assert result["effect"] == {"fixture_effect": .25}
    assert result["evaluated_on_actual_merged_model"] is True and result["split"] == "test_formal"
    assert all(file_sha256(name) == digest for name, digest in old_hashes.items())
    assert primary["evaluation_only"] is False and formal["evaluation_only"] is True
    assert (runner.output_root / "analysis" / "controls" / "test-completed.json").is_file()
    assert (runner.output_root / "analysis" / "controls" / "test_formal-completed.json").is_file()


@pytest.mark.parametrize("changed", ["stage_signature", "merged_weights_sha256", "sample"])
def test_formal_callback_never_reuses_changed_stage_or_same_id_content(controls, changed):
    runner, state, _, _, _ = primary_and_formal(controls)
    captured = state["calls"][-1]
    config, _ = callback_config(runner, captured, "test_formal")
    current = source_sample()
    if changed == "sample":
        current.prompt = "a different premise"
    else:
        config["sequential"][changed] = "changed"
    result = captured["evaluate_fn"]("actual-merged-model", "tokenizer", [current], config, "cpu")
    assert state["generated"] == [["logic:0"], ["logic:0"]]
    assert result["reused_predictions"] == 0 and result["newly_generated"] == 1
    assert result["rows"][0]["prompt"] == current.prompt


def test_formal_callback_rejects_changed_primary_prediction_artifact(controls):
    runner, state, _, _, panel = primary_and_formal(controls)
    write_jsonl(panel / "predictions.jsonl", [{"id": "logic:0", "prediction": "tampered"}])
    captured = state["calls"][-1]
    config, _ = callback_config(runner, captured, "test_formal")
    with pytest.raises(ValueError, match="Sequential primary evaluation changed"):
        captured["evaluate_fn"]("actual-merged-model", "tokenizer", [source_sample()], config, "cpu")
    assert state["generated"] == [["logic:0"]]


@pytest.mark.parametrize("wrong_lock", [False, True])
def test_evalonly_rejects_missing_or_wrong_layer_lock_before_any_evaluation(controls, wrong_lock):
    runner, state = controls
    runner.forbid_training = True
    if wrong_lock:
        atomic_json(runner.output_root / "protocol" / "mvp.layers-7-seed17.lock.json",
                    {"discovery_decisions": {"main_protocol_lock": "another-main-lock"}})
    with pytest.raises(ValueError, match="completed layer control|different main protocol"):
        workflow.run_ablations(runner, "test_formal", evaluation_only=True)
    assert not state["calls"]
    assert not any(event[0] in {"train", "baseline", "layer_panel"} for event in runner.events)
