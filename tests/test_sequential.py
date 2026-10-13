from copy import deepcopy
from pathlib import Path
import os

import pytest
import torch
from torch import nn

from task_vector.data import Sample
from task_vector.io import atomic_json, atomic_torch_save, file_sha256, object_sha256, read_json
from task_vector.representations import LowRankFactor, Representation
from task_vector.sequential import run_sequential
from task_vector.subspaces import Statistics, load_statistics


TASKS = ("math", "logic", "translation", "paraphrase")
LAYER = "model.layers.0"


class TinyMergedModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Module()
        self.model.layers = nn.ModuleList([nn.Module()])
        self.model.layers[0].self_attn = nn.Module()
        for kind, width in (("q_proj", 2), ("k_proj", 1), ("v_proj", 1)):
            projection = nn.Linear(2, width, bias=False, dtype=torch.bfloat16)
            with torch.no_grad():
                projection.weight.fill_(1.)
            setattr(self.model.layers[0].self_attn, kind, projection)


@pytest.fixture
def setup_driver(tmp_path, monkeypatch):
    from task_vector import data, sequential
    # CI evidence lives under a long Codex task path on Windows. The extended
    # prefix permits the intentionally nested stage/split/artifact layout.
    if os.name == "nt":
        tmp_path = Path("\\\\?\\" + str(tmp_path))
    config = {"model": {"dtype": "bfloat16"}, "training": {"epochs": 1},
              "loda": {"rank_general": 1, "rank_isolated": 1,
                       "statistics_samples": 1, "recalibration_lambda": 3},
              "evaluation": {"do_sample": False}, "metrics": {},
              "sequential": {"tasks": list(TASKS), "order": list(TASKS), "seed": 17,
                  "data_root": str(tmp_path / "data"), "output_root": str(tmp_path / "output"),
                  "provenance": {"base_fingerprint": "base-sha", "data_manifest_sha256": "data-sha",
                                 "source_manifest_sha256": "source-sha"}}}
    state = {"loads": [], "trains": [], "stats": [], "evaluations": [], "access": [],
             "deny_train_reads": False, "fail_statistics_call": None, "train_fail_once": None,
             "changed_prompt": False}

    def samples(root, task, split):
        state["loads"].append((task, split))
        if state["deny_train_reads"] and split in {"train", "tune"}:
            raise AssertionError("Evaluation-only read training data")
        count = {"train": 2, "tune": 1, "test": 2, "test_formal": 5}.get(split, 1)
        return [Sample(f"{task}-{split}-{i}", task,
            "changed" if state["changed_prompt"] else f"prompt-{i}", "answer", f"{split}-{i}", split)
            for i in range(count)]

    def statistics(model, tokenizer, samples, cfg, device):
        state["stats"].append((samples[0].task, cfg["sequential"]["stage"]))
        if state["fail_statistics_call"] == len(state["stats"]):
            raise RuntimeError("interrupted statistics")
        projection = model.get_submodule(f"{LAYER}.self_attn.q_proj")
        with torch.no_grad():
            energy = float(projection(torch.ones(1, 2, dtype=torch.bfloat16)).float().sum())
        return Statistics({LAYER: torch.diag(torch.tensor([2. + energy, 1.]))},
                          metadata={"task": samples[0].task})

    def train(model, tokenizer, training, tuning, **kwargs):
        index = kwargs["config"]["sequential"]["stage"]
        state["trains"].append({"stage": index, "task": kwargs["task"],
            "config": deepcopy(kwargs["config"]), "provenance": deepcopy(kwargs["provenance"]),
            "before": model.get_submodule(f"{LAYER}.self_attn.q_proj").weight.detach().clone()})
        directory = Path(kwargs["output_dir"])
        if state["train_fail_once"] == index:
            state["train_fail_once"] = None
            atomic_torch_save(directory / "last.pt", {"logical_step": 1})
            raise RuntimeError("interrupted optimizer")
        basis = kwargs["subspaces"][LAYER]
        factors = {}
        for kind, width in (("q_proj", 2), ("k_proj", 1), ("v_proj", 1)):
            # Export includes the true 0.5 General scaling, as the real adapter does.
            factors[f"{LAYER}.self_attn.{kind}"] = [
                LowRankFactor(torch.full((width, 1), .011 * (index + 1)), basis["A_G"]),
                LowRankFactor(torch.full((width, 1), .0029 if index else 0.), basis["A_I"])]
        raw = Representation("parameter", kwargs["task"], kwargs["seed"], factors=factors)
        for name in ("raw.pt", "post.pt", "initial.pt"):
            raw.save(directory / name)
        for name in ("best.pt", "last.pt", "best-adapter.pt"):
            atomic_torch_save(directory / name, {"stage": index})
        bound = {**kwargs["provenance"], "sample_content_sha256": "bound-by-trainer"}
        result = {"raw_path": str(directory / "raw.pt"), "post_path": str(directory / "post.pt"),
            "initial_path": str(directory / "initial.pt"), "best_checkpoint": str(directory / "best.pt"),
            "last_checkpoint": str(directory / "last.pt"),
            "best_adapter_path": str(directory / "best-adapter.pt"),
            "provenance": bound, "provenance_sha256": object_sha256(bound), "steps": 1}
        atomic_json(directory / "training-state.json", {**result, "status": "exported_uncommitted"})
        return result

    def evaluate(model, tokenizer, samples, cfg, device):
        seq = cfg["sequential"]
        snapshot = {name: module.weight.detach().cpu().clone() for name, module in model.named_modules()
                    if name.endswith((".self_attn.q_proj", ".self_attn.k_proj", ".self_attn.v_proj"))}
        expected = torch.load(Path(seq["evaluation_dir"]).parents[2] / "merged_projection_weights.pt",
                              weights_only=True)
        assert all(torch.equal(snapshot[name], expected[name]) for name in snapshot)
        state["evaluations"].append((seq["stage"], samples[0].task, seq["split"], len(samples)))
        return {"count": len(samples), "split": seq["split"], "rows": [
            {"id": s.id, "prediction": "answer", "target": s.target} for s in samples]}

    def access(task, split, samples, context):
        state["access"].append((context["stage"], task, split, len(samples)))
        assert context["base_fingerprint"] == "base-sha"
        assert context["merged_weights_sha256"]

    monkeypatch.setattr(data, "load_samples", samples)
    monkeypatch.setattr(sequential, "collect_statistics", statistics)
    return config, state, {"device": "cpu", "model_loader": lambda config, device: (TinyMergedModel(), None),
                           "train_fn": train, "evaluate_fn": evaluate, "access_fn": access}


def order_dir(config):
    return Path(config["sequential"]["output_root"]) / "sequential" / "seed17" / "__".join(TASKS)


def test_split_expansion_restores_exact_bf16_without_retraining_or_train_reads(setup_driver, monkeypatch):
    config, state, kwargs = setup_driver
    first = run_sequential(config, split="test", **kwargs)
    assert len(state["trains"]) == 4
    assert len(state["stats"]) == 12
    assert len(state["evaluations"]) == len(state["access"]) == 10
    root = order_dir(config)
    stable = {str(p): file_sha256(p) for p in root.rglob("*")
              if p.is_file() and "evaluation" not in p.parts and p.suffix != ".lock"}
    state["deny_train_reads"] = True
    state["loads"].clear()
    with monkeypatch.context() as guarded:
        guarded.setattr("task_vector.sequential.load_statistics",
                        lambda *a: pytest.fail("evaluation-only deserialized training statistics"))
        guarded.setattr(Representation, "load",
                        lambda *a: pytest.fail("evaluation-only deserialized adapter factors"))
        formal = run_sequential(config, split="test_formal", evaluation_only=True, **kwargs)
    assert len(state["trains"]) == 4 and len(state["stats"]) == 12
    assert all(split == "test_formal" for _, split in state["loads"])
    assert all(item[-1] == 5 for item in state["evaluations"][-10:])
    assert formal[0]["evaluation_only"] and formal[0]["split"] == "test_formal"
    assert first[0]["evaluation"][0]["metrics"]["math"]["count"] == 2
    for name, digest in stable.items():
        assert file_sha256(name) == digest
    for index, task in enumerate(TASKS):
        stage = root / f"stage{index}_{task}"
        assert (stage / "evaluation" / "test" / "math" / "evaluation-completed.json").is_file()
        assert (stage / "evaluation" / "test_formal" / "math" / "evaluation-completed.json").is_file()
        tensors = torch.load(stage / "merged_projection_weights.pt", weights_only=True)
        assert all(t.dtype == torch.bfloat16 for t in tensors.values())
        assert Representation.load(stage / "raw.pt").metadata["base_fingerprint"] == "base-sha"
    # Same split returns its cache but still invokes the lock/ledger hook.
    run_sequential(config, split="test_formal", evaluation_only=True, **kwargs)
    assert len(state["evaluations"]) == 20
    assert len(state["access"]) == 30


def test_stage_protocol_double_accumulation_and_no_future_training_reads(setup_driver):
    config, state, kwargs = setup_driver
    report = run_sequential(config, split=None, **kwargs)[0]
    assert not state["evaluations"]
    assert state["loads"] == [(task, split) for task in TASKS for split in ("train", "tune")]
    root = order_dir(config)
    previous = torch.zeros(2, 2)
    previous_weights = None
    rounding_observed = False
    for index, task in enumerate(TASKS):
        stage = root / f"stage{index}_{task}"
        current = load_statistics(stage / "statistics_after_training.pt")[LAYER]
        merged = load_statistics(stage / "statistics_after_merge.pt")[LAYER]
        assert torch.equal(load_statistics(stage / "past_statistics.pt")[LAYER], previous)
        previous = previous + current + merged
        assert torch.equal(load_statistics(stage / "statistics_past_after_stage.pt")[LAYER], previous)
        flags = state["trains"][index]["config"]["loda"]
        assert flags["train_general_a"] == (index == 0)
        assert flags["isolated_enabled"] == (index != 0)
        provenance = state["trains"][index]["provenance"]
        assert provenance["order"] == list(TASKS) and provenance["stage"] == index
        assert provenance["parent_merge_sha256"] == (None if index == 0 else
            file_sha256(root / f"stage{index-1}_{TASKS[index-1]}" / "merged_projection_weights.pt"))
        post = Representation.load(stage / "post.pt")
        assert post.metadata["past_tasks"] == list(TASKS[:index])
        if index == 0:
            assert post.metadata["lambda"][LAYER] == [1.]
        else:
            assert post.metadata["lambda"][LAYER][0] < 1.
        raw = Representation.load(stage / "raw.pt")
        diagonal = torch.tensor(post.metadata["lambda"][LAYER])
        weights = torch.load(stage / "merged_projection_weights.pt", weights_only=True)
        actual = torch.load(stage / "actual_cumulative_update.pt", weights_only=True)["matrices"]
        cumulative = Representation.load(stage / "cumulative_update.pt")
        for name in post.keys:
            assert torch.equal(post.factors[name][0].B, raw.factors[name][0].B * diagonal)
            before = torch.ones_like(weights[name]) if previous_weights is None else previous_weights[name]
            delta = sum(term.dense() for term in post.factors[name])
            assert torch.equal(weights[name], (before.float() + delta).to(torch.bfloat16))
            assert torch.equal(actual[name], weights[name].float() - 1.)
            algebraic = sum(term.dense() for term in cumulative.factors[name])
            rounding_observed |= not torch.equal(actual[name], algebraic)
        previous_weights = weights
    assert rounding_observed, "toy control must exercise native BF16 merge rounding"
    assert report["status"] == "complete" and (root / "order-completed.json").is_file()


@pytest.mark.parametrize("failed_call", [5, 6])
def test_resume_incomplete_postprocessing_reuses_export_and_parent_commits(setup_driver, failed_call):
    config, state, kwargs = setup_driver
    # Fail both before and after merging stage1, exercising no double merge on retry.
    state["fail_statistics_call"] = failed_call
    with pytest.raises(RuntimeError, match="interrupted statistics"):
        run_sequential(config, split=None, **kwargs)
    root = order_dir(config)
    stage0 = root / "stage0_math"
    stage1 = root / "stage1_logic"
    assert (stage0 / "stage-completed.json").is_file()
    assert (stage1 / "training-result.json").is_file()
    assert not (stage1 / "stage-completed.json").exists()
    stage0_digest = file_sha256(stage0 / "stage-completed.json")
    state["fail_statistics_call"] = None
    run_sequential(config, split=None, **kwargs)
    assert [item["stage"] for item in state["trains"]] == [0, 1, 2, 3]
    assert file_sha256(stage0 / "stage-completed.json") == stage0_digest
    assert torch.equal(state["trains"][1]["before"], torch.load(
        stage0 / "merged_projection_weights.pt", weights_only=True)[f"{LAYER}.self_attn.q_proj"])


def test_incomplete_optimizer_uses_stage_local_resume_only(setup_driver):
    config, state, kwargs = setup_driver
    config["training"]["resume_from"] = "/unrelated/order/last.pt"
    state["train_fail_once"] = 1
    with pytest.raises(RuntimeError, match="interrupted optimizer"):
        run_sequential(config, split=None, **kwargs)
    run_sequential(config, split=None, **kwargs)
    assert [item["stage"] for item in state["trains"]] == [0, 1, 1, 2, 3]
    assert "resume_from" not in state["trains"][0]["config"]["training"]
    assert Path(state["trains"][2]["config"]["training"]["resume_from"]) == order_dir(config) / "stage1_logic" / "last.pt"


def test_export_commit_interruption_recovers_training_state_without_optimizer_rerun(setup_driver, monkeypatch):
    from task_vector import sequential
    config, state, kwargs = setup_driver
    original = sequential.atomic_json
    interrupted = False

    def write(path, payload):
        nonlocal interrupted
        if Path(path).name == "training-result.json" and Path(path).parent.name == "stage1_logic" and not interrupted:
            interrupted = True
            raise RuntimeError("interrupted export commit")
        return original(path, payload)

    monkeypatch.setattr(sequential, "atomic_json", write)
    with pytest.raises(RuntimeError, match="interrupted export commit"):
        run_sequential(config, split=None, **kwargs)
    stage1 = order_dir(config) / "stage1_logic"
    assert (stage1 / "training-state.json").is_file()
    assert not (stage1 / "training-result.json").exists()
    run_sequential(config, split=None, **kwargs)
    assert [item["stage"] for item in state["trains"]] == [0, 1, 2, 3]
    assert (stage1 / "stage-completed.json").is_file()


def test_evaluation_only_fails_before_loader_if_training_incomplete(setup_driver):
    config, state, kwargs = setup_driver
    def forbidden(*args):
        raise AssertionError("must reject before model load")
    kwargs["model_loader"] = forbidden
    with pytest.raises(ValueError, match="not been committed"):
        run_sequential(config, evaluation_only=True, **kwargs)
    assert not state["loads"] and not state["trains"]


@pytest.mark.parametrize("kind", ["base", "config", "train_content", "artifact", "evaluation_content"])
def test_provenance_and_tampering_fail_closed(setup_driver, kind):
    config, state, kwargs = setup_driver
    run_sequential(config, split="test", **kwargs)
    root = order_dir(config)
    evaluation_only = True
    if kind == "base":
        config["sequential"]["provenance"]["base_fingerprint"] = "replacement-base"
    elif kind == "config":
        config["loda"]["recalibration_lambda"] = 2
    elif kind == "train_content":
        state["changed_prompt"] = True
        evaluation_only = False
    elif kind == "artifact":
        atomic_torch_save(root / "stage0_math" / "merged_projection_weights.pt", {"corrupt": torch.ones(1)})
    else:
        state["changed_prompt"] = True
    with pytest.raises(ValueError, match="provenance differs|artifact changed"):
        run_sequential(config, split="test", evaluation_only=evaluation_only, **kwargs)
    assert len(state["trains"]) == 4


def test_all_four_rotations_have_separate_training_commits(setup_driver):
    config, state, kwargs = setup_driver
    config["sequential"].pop("order")
    reports = run_sequential(config, split=None, **kwargs)
    assert len(reports) == 4 and len(state["trains"]) == 16
    assert [r["order"] for r in reports] == [list(TASKS[i:] + TASKS[:i]) for i in range(4)]
    assert all(torch.equal(state["trains"][i]["before"], torch.ones(2, 2, dtype=torch.bfloat16))
               for i in range(0, 16, 4))


def test_requires_explicit_base_identity_and_safe_split(setup_driver):
    config, _, kwargs = setup_driver
    with pytest.raises(ValueError, match="Invalid sequential evaluation split"):
        run_sequential(config, split="../test", **kwargs)
    config["sequential"]["provenance"].pop("base_fingerprint")
    with pytest.raises(ValueError, match="base_fingerprint"):
        run_sequential(config, **kwargs)
