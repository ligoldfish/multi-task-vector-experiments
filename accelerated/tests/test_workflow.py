from copy import deepcopy
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from task_vector import workflow
from task_vector.config import scientific_config, source_manifest
from task_vector.io import atomic_json, object_sha256, read_json, write_jsonl
from task_vector.representations import LowRankFactor, Representation, linear_combination
from task_vector.runner import ExperimentRunner, _distinct_represented_count


TASKS = ["math", "logic", "translation"]


def matrix(values=None, tasks=TASKS):
    raw = np.array(values if values is not None else [[9, .1, .2], [.4, 8, .3], [.5, .6, 7]], dtype=float)
    return {"tasks": list(tasks), "raw": raw.tolist(), "standardized": raw.tolist(),
            "column_std": [1.] * len(tasks)}


class GateRunner:
    def __init__(self, root):
        self.output_root = root / "runs"
        self.output_root.mkdir(parents=True)
        self.source_root = root / "source"
        self.source_root.mkdir()
        self.config = {"platform": {"device": "cpu"}, "data": {"tasks": TASKS},
                       "analysis": {"seeds": [17, 29], "gate_min_positive_tasks": "all",
                                    "transfer_min_sign_agreement": .8, "transfer_min_spearman": .5}}
        self.manifest = {"status": "complete", "manifest_content_sha256": "prepared"}
        self.events = []
        self.active_phase = "final"

    def base_fingerprint(self):
        return "base-sha"

    def taskset(self, scope):
        return [task for task in self.config["data"]["tasks"] if not scope.startswith("loo-") or task != scope[4:]]

    def release_model(self):
        self.events.append("release")

    def train(self, scope, route, seed, task):
        self.events.append(("train", scope, route, seed, task))

    def lock_path(self, scope):
        return self.output_root / "protocol" / f"{scope}.lock.json"

    def lock_core(self, scope, seed):
        self.events.append(("lock_core", scope, seed))

    @contextmanager
    def protocol_phase(self, phase):
        previous = self.active_phase
        self.events.append(("phase_enter", phase))
        self.active_phase = phase
        try:
            yield
        finally:
            self.events.append(("phase_exit", phase))
            self.active_phase = previous

    def complete_core_stage(self, scope, seed):
        assert self.active_phase == "core"
        self.events.append(("complete_core", scope, seed))
        atomic_json(self.output_root / "analysis" / scope / "first-round-completed.json", {"completed": True})

    def core_panel(self, scope, route, seed, split="discovery", layers=None, variant="canonical"):
        self.events.append(("core", scope, route, seed, split, self.active_phase))
        tasks = self.taskset(scope)
        if tasks == TASKS:
            values = matrix()
        else:
            n = len(tasks)
            values = matrix(np.arange(n * n).reshape(n, n) / 10 + np.eye(n), tasks)
        write_core(self, scope, route, seed, values, split)


def write_core(runner, scope, route, seed, values, split="discovery"):
    own = {"condition": .6, "delta": .1, "ci": [-.01, .21]}
    evidence = {task: {"primary_metric": "accuracy", "own_task": own,
                       "controls": {"initial": {"delta": {"accuracy": {"condition": .5}}},
                                    "random": {"delta": {"accuracy": {"condition": .55}}}}}
                for task in runner.taskset(scope)}
    atomic_json(runner.output_root / "analysis" / scope / str(seed) / f"{route}-{split}-canonical-core.json",
                {"E1": evidence, "matrix": values})


def smoke_record(runner, *, device="cpu", task="math"):
    return {"passed": True, "task": task, "device": device,
            "checks": {"loda_true_gao_microbatch_step": True,
                       "loda_checkpoint_resume_exact": True,
                       "loda_resume_does_not_repeat_step": True},
            "config_sha256": object_sha256(scientific_config(runner.config)),
            "data_manifest_sha256": object_sha256(runner.manifest),
            "base_fingerprint": runner.base_fingerprint(),
            "source_manifest_sha256": object_sha256(source_manifest(runner.source_root))}


def prepare_gate(runner, second=None, loda=None):
    for seed in runner.config["analysis"]["seeds"]:
        for route in ("ltv", "loda"):
            value = loda if route == "loda" and loda is not None else matrix()
            if second is not None and seed == 29:
                value = second
            write_core(runner, "mvp", route, seed, value)
    atomic_json(runner.output_root / "smoke.json", smoke_record(runner))


def test_directed_transfer_uses_lower_triangle_and_excludes_diagonal():
    first = matrix()
    second = deepcopy(first)
    second["raw"][1][0] *= -1
    second["raw"][2][0] *= -1
    second["raw"][2][1] *= -1
    second["standardized"] = deepcopy(second["raw"])
    for i in range(3):
        second["raw"][i][i] = -100000
    evidence = workflow.directed_transfer_consistency(first, second)
    assert evidence["ordered_pairs"] == 6
    assert evidence["sign_agreement"] == .5
    assert all(row["source"] != row["target"] for row in evidence["pairs"])
    assert {("logic", "math"), ("translation", "math"), ("translation", "logic")} <= {
        (row["source"], row["target"]) for row in evidence["pairs"]}


def test_gate_rejects_nonrepeatable_transfer_despite_positive_own_task(tmp_path):
    runner = GateRunner(tmp_path)
    changed = matrix(np.array(matrix()["raw"]) * -1)
    prepare_gate(runner, second=changed)
    result = workflow.assess_gate(runner, "mvp", [17, 29])
    assert all(result["E1"][route]["passes"] for route in ("ltv", "loda"))
    assert result["passes"] is False
    assert result["functional_reproducibility"]["routes"]["ltv"]["comparisons"]["17-29"]["sign_agreement"] == 0


def test_gate_cross_space_difference_is_descriptive_and_all_seeds_required(tmp_path):
    runner = GateRunner(tmp_path)
    prepare_gate(runner, loda=matrix(np.array(matrix()["raw"]) * -1))
    provisional = workflow.assess_gate(runner, "mvp", [17])
    assert provisional["passes"] is True
    assert provisional["fully_validated"] is False
    assert provisional["functional_reproducibility"]["status"] == "pending_additional_seed"
    result = workflow.assess_gate(runner, "mvp", [17, 29])
    assert result["fully_validated"] is True
    assert result["functional_reproducibility"]["cross_space"]["17"]["sign_agreement"] == 0


def test_zero_variance_raw_signs_retained_but_no_fabricated_correlation():
    first, second = matrix(), matrix()
    first["column_std"] = second["column_std"] = [0, 0, 0]
    evidence = workflow.directed_transfer_consistency(first, second)
    assert evidence["ordered_pairs"] == 6
    assert evidence["correlation_pairs"] == 0
    assert evidence["spearman"] is None
    assert evidence["sign_agreement"] == 1


def test_smoke_checks_device_data_and_source_membership(tmp_path):
    runner = GateRunner(tmp_path)
    record = smoke_record(runner)
    atomic_json(runner.output_root / "smoke.json", record)
    assert workflow.smoke_evidence(runner, "mvp")["valid"]
    assert not workflow.smoke_evidence(runner, "loo-math")["valid"]
    runner.config["platform"]["device"] = "npu:0"
    assert not workflow.smoke_evidence(runner, "mvp")["checks"]["device_backend_matches"]
    runner.config["platform"]["device"] = "cpu"
    runner.manifest["manifest_content_sha256"] = "changed"
    assert not workflow.smoke_evidence(runner, "mvp")["checks"]["dataset_matches"]


@pytest.mark.parametrize("check", ["loda_true_gao_microbatch_step", "loda_checkpoint_resume_exact",
                                   "loda_resume_does_not_repeat_step"])
def test_legacy_passed_smoke_without_true_gao_resume_proof_cannot_open_training(tmp_path, check):
    runner = GateRunner(tmp_path)
    record = smoke_record(runner)
    del record["checks"][check]
    atomic_json(runner.output_root / "smoke.json", record)
    assert workflow.smoke_evidence(runner, "mvp")["valid"] is False


def test_real_smoke_precedes_any_training_and_preserves_stale_record(tmp_path, monkeypatch):
    runner = GateRunner(tmp_path)
    old = smoke_record(runner, device="npu:0")
    atomic_json(runner.output_root / "smoke.json", old)

    def smoke(config, *, task, output_dir):
        runner.events.append("smoke")
        record = smoke_record(runner, task=task)
        atomic_json(Path(output_dir) / "smoke.json", record)
        return record

    monkeypatch.setattr("task_vector.smoke.run_smoke", smoke)
    result = workflow.run_core(runner)
    assert result["completed"] is True
    assert runner.events[:2] == ["release", "smoke"]
    assert runner.events[2][0] == "train"
    histories = list((runner.output_root / "engineering_smoke").glob("previous-*.json"))
    assert len(histories) == 1 and read_json(histories[0]) == old


def test_all_heldout_folds_attempt_source_gate_and_failed_fold_never_final(tmp_path, monkeypatch):
    runner = GateRunner(tmp_path)
    runner.config["analysis"]["heldout_tasks"] = list(TASKS)
    events = []

    def core(runner, scope):
        events.append(("core", scope, runner.taskset(scope)))
        return {"completed": scope != "loo-logic", "stop_reason": "declared_gate"}

    def discover(runner, scope, *, source_fold):
        assert source_fold and scope.startswith("loo-")
        events.append(("discovery", scope))

    monkeypatch.setattr(workflow, "run_core", core)
    monkeypatch.setattr(workflow, "run_discovery", discover)
    monkeypatch.setattr(workflow, "lock_protocol", lambda r, s: events.append(("lock", s)))
    monkeypatch.setattr(workflow, "run_final", lambda r, s, split: events.append(("final", s)) or {"completed": True})
    result = workflow.run_phase(runner, "heldout")
    assert len([event for event in events if event[0] == "core"]) == len(TASKS)
    for _, scope, sources in [event for event in events if event[0] == "core"]:
        assert scope[4:] not in sources
    assert not any(event[1] == "loo-logic" and event[0] in {"discovery", "lock", "final"} for event in events)
    assert result["completed"] is False
    assert result["folds"]["logic"]["heldout_selection_or_test_accessed"] is False


def test_source_fold_discovery_cannot_bypass_its_gate(tmp_path, monkeypatch):
    runner = GateRunner(tmp_path)
    calls = []

    def reject(runner, scope):
        calls.append(scope)
        raise RuntimeError("source-only gate")

    monkeypatch.setattr(workflow, "_require_gate", reject)
    with pytest.raises(RuntimeError, match="source-only"):
        workflow.run_discovery(runner, "loo-math", source_fold=True)
    assert calls == ["loo-math"]


def concrete_runner(tmp_path, tasks=None):
    model = tmp_path / "model"
    model.mkdir()
    (model / "config.json").write_text('{"hidden_size": 2}', encoding="utf-8")
    root = tmp_path / "source"
    root.mkdir()
    config = {"platform": {"persistent_root": str(tmp_path), "device": "cpu"},
              "runtime": {"data_root": str(tmp_path / "data"), "output_root": str(tmp_path / "runs")},
              "model": {"path": str(model)}, "data": {"tasks": tasks or TASKS},
              "analysis": {"seeds": [17, 29]}, "training": {"max_prompt_tokens": 512},
              "loda": {"statistics_samples": 2}}
    runner = ExperimentRunner(config, source_root=root)
    runner._manifest = {"status": "complete", "manifest_content_sha256": "prepared"}
    return runner


def test_fold_statistics_never_reads_heldout_training(tmp_path, monkeypatch):
    from task_vector.subspaces import Statistics
    runner = concrete_runner(tmp_path)
    runner.model, runner.tokenizer, runner.device = object(), object(), "cpu"
    monkeypatch.setattr(runner, "load_model", lambda: (runner.model, runner.tokenizer))
    seen = []

    def samples(task, split):
        assert task != "math" and split == "train"
        seen.append(task)
        return [SimpleNamespace(id=f"{task}:0")]

    monkeypatch.setattr(runner, "samples", samples)
    monkeypatch.setattr("task_vector.subspaces.collect_statistics",
                        lambda *args: Statistics({"model.layers.0": torch.eye(2)}, metadata={"official_split": "train"}))
    stats = runner.statistics("loo-math")
    assert set(stats) == {"logic", "translation"}
    assert "math" not in seen
    assert not (runner.output_root / "statistics" / "frozen_base" / "math.pt").exists()


def test_final_access_ledger_records_ids_and_superset_not_new_confirmation(tmp_path):
    runner = concrete_runner(tmp_path)
    rows = [SimpleNamespace(id="official:1", group_id="group:1")]
    for split in ("test", "test_formal", "test_confirmation"):
        runner.record_final_access("math", split, rows, scope="mvp", seed=17,
                                   label="base", variant="canonical", lock_hash="frozen-sha")
    ledger = read_json(runner.output_root / "test_access_ledger.json")
    assert len(ledger["events"]) == 3
    assert all(event["sample_ids_sha256"] == object_sha256(["official:1"]) for event in ledger["events"])
    assert all(event["protocol_lock_sha256"] == "frozen-sha" for event in ledger["events"])
    assert "not assumed" in ledger["events"][1]["independence"]
    assert ledger["events"][2]["independence"] == "separately reserved group pool"


def test_lock_freezes_assets_and_rejects_changed_assets_before_test_read(tmp_path, monkeypatch):
    runner = concrete_runner(tmp_path, tasks=["math", "paraphrase"])
    semantic = ["semantic-original"]
    monkeypatch.setattr("task_vector.metrics.semantic_asset_fingerprint", lambda config: semantic[0])
    monkeypatch.setattr(runner, "freeze_artifacts", lambda *args, **kwargs: {})
    atomic_json(runner.output_root / "protocol" / "mvp.decisions.json", {"17": {"source_only": True}})
    lock = workflow.lock_protocol(runner, "mvp")
    assert lock["discovery_decisions"]["asset_fingerprints"] == runner.asset_fingerprints()
    monkeypatch.setattr(runner, "samples", lambda *args: pytest.fail("test data accessed before asset check"))
    semantic[0] = "semantic-replaced"
    with pytest.raises(ValueError, match="assets differ"):
        runner.evaluate(None, "math", "test", label="base")
    semantic[0] = "semantic-original"
    (Path(runner.config["model"]["path"]) / "config.json").write_text('{"hidden_size": 3}', encoding="utf-8")
    with pytest.raises(ValueError, match="assets differ"):
        runner.evaluate(None, "math", "test", label="base")


def test_representation_base_guard_precedes_any_data_access(tmp_path, monkeypatch):
    runner = concrete_runner(tmp_path)
    monkeypatch.setattr(runner, "samples", lambda *args: pytest.fail("panel accessed before representation guard"))
    rep = Representation("activation", "math", 17, values={"layer": torch.ones(2)}, metadata={"base_fingerprint": "other-model"})
    with pytest.raises(ValueError, match="base fingerprint"):
        runner.evaluate(rep, "math", "discovery", label="stale")


def test_normalized_candidates_count_actual_matrices_and_preserve_combination_rank():
    identity = torch.eye(3)
    rep = Representation("parameter", "a", 17, factors={"q": [LowRankFactor(identity, identity)]})
    reencoded = Representation("parameter", "b", 17, factors={"q": [LowRankFactor(identity * 2, identity / 2)]})
    assert _distinct_represented_count([rep, reencoded]) == 1
    combo = linear_combination([rep, reencoded, rep, reencoded], [.5, 2, -1, 3])
    assert len(combo.factors["q"]) == 4
    assert sum(factor.A.shape[0] for factor in combo.factors["q"]) == 12
    torch.testing.assert_close(sum(factor.dense() for factor in combo.factors["q"]), identity * 4.5)


def test_component_selection_guard_fails_before_any_final_panel_read(tmp_path, monkeypatch):
    runner = concrete_runner(tmp_path)
    monkeypatch.setattr(runner, "representations", lambda *args: pytest.fail("vectors loaded before selection guard"))
    with pytest.raises(ValueError, match="structure discovery split"):
        runner.component_panel("mvp", "ltv", 17, split="test", selected=None)


def test_group_svd_projects_global_residual_and_reconstructs_all_six_tasks():
    from task_vector.analysis import decompose, group_residual_decomposition
    torch.manual_seed(6)
    vectors = torch.randn(7, 6)
    reps = [Representation("activation", str(i), 17, values={"layer": vectors[:, i]}) for i in range(6)]
    global_parts = decompose(reps, "svd", 1)
    groups = [[0, 2, 4], [1, 3, 5]]
    local_parts = group_residual_decomposition(reps, global_parts, groups, 1)
    for group_index, indices in enumerate(groups):
        source = torch.stack([global_parts.residual[i].values["layer"] for i in indices], dim=1).double()
        basis, _, _ = torch.linalg.svd(source / source.norm(dim=0), full_matrices=False)
        for local, original in enumerate(indices):
            group = local_parts[group_index].shared[local].values["layer"].double()
            specific = local_parts[group_index].residual[local].values["layer"].double()
            torch.testing.assert_close(group, basis[:, :1] @ basis[:, :1].T @ source[:, local], atol=2e-6, rtol=2e-6)
            global_value = global_parts.shared[original].values["layer"].double()
            torch.testing.assert_close(global_value + group + specific, vectors[:, original].double(), atol=2e-6, rtol=2e-6)
            assert abs(float(global_value @ group)) < 2e-6


def test_seed17_core_test_precedes_discovery_and_every_later_seed(tmp_path):
    runner = GateRunner(tmp_path)
    runner.config["data"]["tasks"] = ["math", "logic", "translation", "paraphrase"]
    runner.config["analysis"]["seeds"] = [17, 29, 43]
    atomic_json(runner.output_root / "smoke.json", smoke_record(runner))
    result = workflow.run_core(runner)
    assert result["completed"] is True
    events = runner.events
    first_lock = events.index(("lock_core", "mvp", 17))
    assert events[:first_lock] == [("train", "mvp", route, 17, task)
                                  for route in ("ltv", "loda") for task in runner.taskset("mvp")]
    assert events[first_lock:first_lock + 8] == [
        ("lock_core", "mvp", 17), ("phase_enter", "core"),
        ("core", "mvp", "ltv", 17, "test", "core"),
        ("core", "mvp", "loda", 17, "test", "core"),
        ("complete_core", "mvp", 17), ("phase_exit", "core"),
        ("core", "mvp", "ltv", 17, "discovery", "final"),
        ("core", "mvp", "loda", 17, "discovery", "final")]
    first_29 = next(i for i, event in enumerate(events) if event[:3] == ("train", "mvp", "ltv") and event[3] == 29)
    assert first_29 > first_lock + 7
    assert result["gate"]["evaluation_split"] == "test"
    assert runner.active_phase == "final"


def test_failed_seed17_core_gate_never_starts_later_seed(tmp_path, monkeypatch):
    runner = GateRunner(tmp_path)
    atomic_json(runner.output_root / "smoke.json", smoke_record(runner))
    monkeypatch.setattr(workflow, "assess_gate", lambda *args: {"passes": False})
    result = workflow.run_core(runner)
    assert result["completed"] is False
    assert result["stop_reason"] == "validity_or_consistency_gate"
    assert not any(event[0] == "train" and event[3] == 29 for event in runner.events)


def _populate_exports(runner, *, seeds=None):
    seeds = seeds or runner.config["analysis"]["seeds"]
    for seed in seeds:
        for route in ("ltv", "loda"):
            for task in runner.taskset("mvp"):
                folder = runner.directory("mvp", route, seed, task)
                rep = Representation("activation", task, seed, values={"model.layers.0": torch.tensor([.1, .2])},
                                     metadata={"base_fingerprint": runner.base_fingerprint()})
                for name in ("raw.pt", "post.pt", "initial.pt"):
                    rep.save(folder / name)


def test_actual_first_round_marker_counts_exact_9216_without_selection(tmp_path, monkeypatch):
    tasks = ["math", "logic", "translation", "paraphrase"]
    runner = concrete_runner(tmp_path, tasks=tasks)
    runner.config["analysis"]["seeds"] = [17]
    monkeypatch.setattr("task_vector.metrics.semantic_asset_fingerprint", lambda config: "semantic-fixture")
    monkeypatch.setattr(runner, "samples", lambda task, split: [SimpleNamespace(id=f"{task}:{i}") for i in range(256)])
    _populate_exports(runner)
    labels = ["base"] + [f"{route}-{task}-primary" for route in ("ltv", "loda") for task in tasks]
    for label in labels:
        for task in tasks:
            folder = runner.output_root / "evaluation" / "mvp" / "17" / "test" / "canonical" / label / task
            atomic_json(folder / "scores.json", {"count": 256, "newly_generated": 256})
            write_jsonl(folder / "predictions.jsonl", [{"id": f"{task}:{i}"} for i in range(256)])
    runner.lock_core("mvp", 17)
    with runner.protocol_phase("core"):
        result = runner.complete_core_stage("mvp", 17)
    assert result["core_states"] == 9 and result["core_generations"] == 9216
    assert len(result["panels"]) == 36
    assert result["structure_selection_performed"] is False
    assert result["extra_E1_controls_and_tune_discovery_excluded"] is True
    assert (runner.output_root / "analysis" / "mvp" / "first-round-completed.json").is_file()
    assert not runner.lock_path("mvp").exists()


def test_incomplete_core_predictions_cannot_publish_completion(tmp_path, monkeypatch):
    runner = concrete_runner(tmp_path)
    monkeypatch.setattr(runner, "samples", lambda *args: [None] * 256)
    folder = runner.output_root / "evaluation" / "mvp" / "17" / "test" / "canonical" / "base" / TASKS[0]
    atomic_json(folder / "scores.json", {"count": 256})
    write_jsonl(folder / "predictions.jsonl", [{"id": str(i)} for i in range(255)])
    with pytest.raises(ValueError, match="core generation panel is incomplete"):
        runner.complete_core_stage("mvp", 17)
    assert not (runner.output_root / "analysis" / "mvp" / "first-round-completed.json").exists()


def test_frozen_representation_changed_after_protocol_cache_is_rejected_on_load(tmp_path):
    runner = concrete_runner(tmp_path)
    _populate_exports(runner)
    atomic_json(runner.output_root / "protocol" / "mvp.decisions.json", {"17": {"source_only": True}})
    workflow.lock_protocol(runner, "mvp")
    runner.verify_evaluation_protocol("mvp", 17)
    path = runner.directory("mvp", "ltv", 17, TASKS[0]) / "post.pt"
    original = runner.load_representation(path)
    changed = Representation("activation", original.task, original.seed,
                             values={key: value + 1 for key, value in original.values.items()}, metadata=original.metadata)
    changed.save(path)
    with pytest.raises(ValueError, match="Frozen representation artifact changed"):
        runner.load_representation(path)


def test_run_final_rechecks_artifacts_even_with_cached_protocol(tmp_path, monkeypatch):
    runner = concrete_runner(tmp_path)
    _populate_exports(runner)
    atomic_json(runner.output_root / "protocol" / "mvp.decisions.json", {"17": {"source_only": True}})
    workflow.lock_protocol(runner, "mvp")
    runner.verify_evaluation_protocol("mvp", 17)
    path = runner.directory("mvp", "ltv", 17, TASKS[0]) / "post.pt"
    path.write_bytes(b"replaced after protocol verification")
    monkeypatch.setattr(runner, "core_panel", lambda *args, **kwargs: pytest.fail("final started before artifact validation"))
    with pytest.raises(ValueError, match="Frozen representation artifact changed"):
        workflow.run_final(runner)


def test_selected_candidate_digest_must_match_before_lock(tmp_path):
    runner = concrete_runner(tmp_path)
    _populate_exports(runner)
    candidate = Representation("activation", "combined", 17, values={"layer": torch.ones(2)})
    path = runner.output_root / "candidate.pt"
    candidate.save(path)
    choice = {"representation_path": str(path), "representation_digest": runner.representation_digest(candidate)}
    candidate.values["layer"] += 1
    candidate.save(path)
    with pytest.raises(ValueError, match="Selected candidate differs from discovery"):
        runner.freeze_artifacts("mvp", decisions={"choice": choice})


def _stub_full_pipeline(monkeypatch, runner, *, failed=None):
    events = []
    def core(r, scope="mvp"):
        events.append(("core", scope))
        return {"completed": failed != f"{scope}-core"}
    def final(r, scope="mvp", split="test"):
        events.append(("final", scope, split))
        return {"completed": failed != f"{scope}-final", "scope": scope}
    def controls(r, split="test", *, evaluation_only=False):
        events.append(("controls", split, evaluation_only))
        return {"completed": failed != "controls"}
    def formal(r):
        events.append(("formal",))
        return {"completed": True, "evaluation_only": True}
    monkeypatch.setattr(workflow, "run_core", core)
    monkeypatch.setattr(workflow, "run_discovery", lambda r, scope="mvp", **kw: events.append(("discover", scope)))
    monkeypatch.setattr(workflow, "lock_protocol", lambda r, scope: events.append(("lock", scope)))
    monkeypatch.setattr(workflow, "run_final", final)
    monkeypatch.setattr(workflow, "run_ablations", controls)
    monkeypatch.setattr(workflow, "run_formal", formal)
    monkeypatch.setattr(workflow, "run_heldout_discovery", lambda r: {"math": {"completed": True, "scope": "loo-math"}})
    return events


def test_unified_full_formal_finishes_primary_pipeline_before_eval_only_expansion(tmp_path, monkeypatch):
    runner = GateRunner(tmp_path)
    events = _stub_full_pipeline(monkeypatch, runner)
    result = workflow.run_phase(runner, "all", "mvp", "test_formal")
    assert result["completed"] is True
    assert events == [("core", "mvp"), ("discover", "mvp"), ("lock", "mvp"),
                      ("final", "mvp", "test"), ("final", "loo-math", "test"),
                      ("controls", "test", False), ("core", "expanded"),
                      ("discover", "expanded"), ("lock", "expanded"),
                      ("final", "expanded", "test"), ("formal",)]


@pytest.mark.parametrize("failed", ["mvp-core", "loo-math-final", "controls", "expanded-core"])
def test_unified_full_formal_never_opens_formal_after_incomplete_primary(tmp_path, monkeypatch, failed):
    runner = GateRunner(tmp_path)
    events = _stub_full_pipeline(monkeypatch, runner, failed=failed)
    result = workflow.run_phase(runner, "all", "mvp", "test_formal")
    assert result["completed"] is False
    assert ("formal",) not in events
    assert not any(event[0] == "final" and event[2] == "test_formal" for event in events)


def test_resume_locked_scope_verifies_choices_without_training_or_rediscovery(tmp_path, monkeypatch):
    runner = GateRunner(tmp_path)
    atomic_json(runner.lock_path("expanded"), {"fixture": True})
    choices = {"frozen": "selected-on-discovery"}
    def verify(scope, seed):
        runner.events.append(("verify", scope, seed))
        return {"discovery_decisions": choices}
    runner.verify_evaluation_protocol = verify
    assert workflow.run_discovery(runner, "expanded") == choices
    assert runner.events == [("verify", "expanded", 17)]
    for name in ("run_core", "run_discovery", "lock_protocol"):
        monkeypatch.setattr(workflow, name, lambda *args, **kwargs: pytest.fail("Locked scope retrained or reselected"))
    monkeypatch.setattr(workflow, "run_final", lambda r, scope, split: runner.events.append(("final", scope, split)) or {"completed": True})
    result = workflow.run_phase(runner, "all", "expanded", "test")
    assert result["completed"] is True
    assert runner.events[-2:] == [("verify", "expanded", 17), ("final", "expanded", "test")]


def test_resume_heldout_locks_never_reopens_source_selection(tmp_path, monkeypatch):
    runner = GateRunner(tmp_path)
    runner.config["analysis"]["heldout_tasks"] = list(TASKS)
    for task in TASKS:
        atomic_json(runner.lock_path(f"loo-{task}"), {"fixture": True})
    runner.verify_evaluation_protocol = lambda scope, seed: runner.events.append(("verify", scope, seed))
    for name in ("run_core", "run_discovery", "lock_protocol"):
        monkeypatch.setattr(workflow, name, lambda *args, **kwargs: pytest.fail("Frozen heldout choices were reselected"))
    result = workflow.run_heldout_discovery(runner)
    assert set(result) == set(TASKS)
    assert all(fold["completed"] and fold["reused_locked_discovery"] for fold in result.values())
    assert runner.events == [("verify", f"loo-{task}", 17) for task in TASKS]


def _formal_ready_runner(tmp_path):
    """Small committed markers; no model or training artifacts are fabricated."""
    from task_vector.sequential import cyclic_orders
    runner = GateRunner(tmp_path)
    runner.config["data"]["tasks"] = ["math", "logic", "translation", "paraphrase"]
    runner.config["analysis"].update(seeds=[17, 29], heldout_tasks=list(runner.taskset("mvp")),
                                      ltv_layer_ablations=[[6], [22], [6, 14, 22]])
    runner.asset_fingerprints = lambda: {"base": "base-sha"}
    provenance = workflow._sequential_provenance(runner, {"lock_sha256": "lock-mvp"})
    scopes = ["mvp", "expanded"] + [f"loo-{task}" for task in runner.taskset("mvp")]
    seeds = list(map(str, runner.config["analysis"]["seeds"]))
    for scope in scopes:
        atomic_json(runner.output_root / "analysis" / scope / "test-completed.json",
                    {"protocol_lock": f"lock-{scope}", "seeds": seeds, "scope": scope})
    atomic_json(runner.output_root / "analysis" / "controls" / "test-completed.json",
                {"completed": True, "protocol_lock": "lock-mvp", "seeds": {seed: {} for seed in seeds}})
    for seed in runner.config["analysis"]["seeds"]:
        for layers in runner.config["analysis"]["ltv_layer_ablations"]:
            tag = "-".join(map(str, layers))
            atomic_json(runner.output_root / "protocol" / f"mvp.layers-{tag}-seed{seed}.lock.json",
                        {"discovery_decisions": {"main_protocol_lock": "lock-mvp"}})
        for order in cyclic_orders(runner.taskset("mvp")):
            folder = runner.output_root / "sequential" / f"seed{seed}" / "__".join(order)
            identity = {"schema_version": 2, "route": "sequential_LoDA", "seed": seed,
                        "order": list(order), "provenance": provenance}
            atomic_json(folder / "order-request.json", identity)
            atomic_json(folder / "order-completed.json", {"status": "complete", "order_signature": object_sha256(identity)})
            for index, task in enumerate(order):
                request = {"order": list(order), "stage": index, "task": task, "seed": seed, "provenance": provenance}
                atomic_json(folder / f"stage{index}_{task}" / "stage-completed.json",
                            {"schema_version": 2, "status": "complete", "request": request,
                             "stage_signature": object_sha256(request)})
    def verify(scope, seed):
        runner.events.append(("verify", scope, seed))
        return {"lock_sha256": f"lock-{scope}"}
    def verify_control(path):
        runner.events.append(("verify_control", Path(path).name))
        return read_json(path)
    runner.verify_evaluation_protocol = verify
    runner.verify_frozen_protocol = verify_control
    return runner, scopes


def _formal_callbacks(monkeypatch, runner):
    def final(r, scope="mvp", split="test"):
        runner.events.append(("final", scope, split))
        return {"completed": True}
    def controls(r, split="test", *, evaluation_only=False):
        runner.events.append(("controls", split, evaluation_only))
        return {"completed": True}
    monkeypatch.setattr(workflow, "run_final", final)
    monkeypatch.setattr(workflow, "run_ablations", controls)
    for name in ("run_core", "run_discovery", "lock_protocol"):
        monkeypatch.setattr(workflow, name, lambda *a, **kw: pytest.fail("Formal expansion attempted training or selection"))


def test_formal_checks_all_six_scopes_and_controls_before_any_expanded_test(tmp_path, monkeypatch):
    runner, scopes = _formal_ready_runner(tmp_path)
    _formal_callbacks(monkeypatch, runner)
    result = workflow.run_phase(runner, "formal")
    assert result["completed"] is True and result["evaluation_only"] is True
    first_final = next(i for i, event in enumerate(runner.events) if event[0] == "final")
    assert [event[1] for event in runner.events[:first_final] if event[0] == "verify"] == scopes
    assert len([event for event in runner.events[:first_final] if event[0] == "verify_control"]) == 6
    assert runner.events[first_final:] == [("final", scope, "test_formal") for scope in scopes] + [
        ("controls", "test_formal", True)]


def test_actual_control_driver_expands_with_no_training_and_passes_eval_only_to_sequential(tmp_path, monkeypatch):
    runner, _ = _formal_ready_runner(tmp_path)
    runner.device = "cpu"
    runner.data_root = tmp_path / "data"
    runner.train = lambda *args, **kwargs: pytest.fail("Evaluation-only controls attempted training")
    runner.evaluate = lambda rep, task, split, **kwargs: {"task": task, "split": split}
    monkeypatch.setattr(workflow, "_require_gate", lambda *args: None)
    calls = []
    def sequential(config, **kwargs):
        assert kwargs["evaluation_only"] is True and kwargs["split"] == "test_formal"
        assert callable(kwargs["evaluate_fn"]) and callable(kwargs["access_fn"])
        assert config["sequential"]["provenance"]["protocol_lock_sha256"] == "lock-mvp"
        calls.append(config["sequential"]["seed"])
        return [{}] * 4
    monkeypatch.setattr("task_vector.sequential.run_sequential", sequential)
    result = workflow.run_ablations(runner, "test_formal", evaluation_only=True)
    assert result["completed"] is True and result["evaluation_only"] is True
    assert calls == [17, 29]
    assert all(event[4] == "test_formal" for event in runner.events if event[0] == "core")
    assert read_json(runner.output_root / "analysis" / "controls" / "test_formal-completed.json")["evaluation_only"] is True


@pytest.mark.parametrize("broken", ["scope_marker", "scope_lock", "scope_seeds", "controls_marker",
                                     "controls_lock", "controls_seeds", "layer_lock", "layer_parent",
                                     "order_marker", "stage_marker", "order_signature", "stage_signature", "stage_provenance"])
def test_formal_refuses_incomplete_primary_evidence_before_any_expanded_test(tmp_path, monkeypatch, broken):
    runner, scopes = _formal_ready_runner(tmp_path)
    _formal_callbacks(monkeypatch, runner)
    scope_marker = runner.output_root / "analysis" / scopes[-1] / "test-completed.json"
    control_marker = runner.output_root / "analysis" / "controls" / "test-completed.json"
    if broken == "scope_marker":
        scope_marker.unlink()
    elif broken in {"scope_lock", "scope_seeds"}:
        record = read_json(scope_marker)
        record["protocol_lock" if broken == "scope_lock" else "seeds"] = "wrong-lock" if broken == "scope_lock" else ["17"]
        atomic_json(scope_marker, record)
    elif broken == "controls_marker":
        control_marker.unlink()
    elif broken in {"controls_lock", "controls_seeds"}:
        record = read_json(control_marker)
        record["protocol_lock" if broken == "controls_lock" else "seeds"] = "wrong-lock" if broken == "controls_lock" else {"17": {}}
        atomic_json(control_marker, record)
    elif broken == "layer_lock":
        (runner.output_root / "protocol" / "mvp.layers-22-seed29.lock.json").unlink()
    elif broken == "layer_parent":
        atomic_json(runner.output_root / "protocol" / "mvp.layers-22-seed29.lock.json",
                    {"discovery_decisions": {"main_protocol_lock": "other-main"}})
    else:
        order = "__".join(runner.taskset("mvp"))
        folder = runner.output_root / "sequential" / "seed29" / order
        if broken in {"order_marker", "stage_marker"}:
            path = folder / "order-completed.json" if broken == "order_marker" else folder / "stage3_paraphrase" / "stage-completed.json"
            path.unlink()
        elif broken == "order_signature":
            atomic_json(folder / "order-completed.json", {"status": "complete", "order_signature": "changed"})
        else:
            path = folder / "stage3_paraphrase" / "stage-completed.json"
            record = read_json(path)
            if broken == "stage_signature":
                record["stage_signature"] = "changed"
            else:
                record["request"]["provenance"]["base_fingerprint"] = "another-base"
                record["stage_signature"] = object_sha256(record["request"])
            atomic_json(path, record)
    with pytest.raises((ValueError, FileNotFoundError)):
        workflow.run_formal(runner)
    assert not any(event[0] in {"final", "controls"} for event in runner.events)
