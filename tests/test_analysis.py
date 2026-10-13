import numpy as np
import pytest
import torch
from task_vector.representations import Representation, LowRankFactor, inner, norm, linear_combination, random_like
from task_vector.analysis import decompose, gram_matrix, layer_subspace_angles, paired_bootstrap, transfer_matrix, discover_groups, confirm_transfer_relations, random_subspace_control


def activation(v, task="x"):
    return Representation("activation", task, 17, values={"model.layers.14": torch.tensor(v, dtype=torch.float32)})


def test_normalized_svd_projects_raw_vectors():
    reps = [activation([3, 0, 0], "a"), activation([1, 2, 0], "b"), activation([0, 0, .2], "c")]
    X = torch.stack([r.values[r.keys[0]] for r in reps], dim=1).double()
    U, _, _ = torch.linalg.svd(X / X.norm(dim=0), full_matrices=False)
    result = decompose(reps, "svd", 1)
    for i, shared in enumerate(result.shared):
        assert torch.allclose(shared.values[shared.keys[0]].double(), U[:, :1] @ U[:, :1].T @ X[:, i], atol=1e-6)
    assert max(result.metadata["reconstruction_relative_errors"]) < 1e-6
    zero = decompose(reps, "svd", 0)
    assert norm(linear_combination([zero.compose(), linear_combination(reps, [1, 1, 1])], [1, -1])) < 1e-6


def test_actual_ba_geometry_invariant_to_gauge():
    torch.manual_seed(4)
    B, A = torch.randn(7, 2), torch.randn(2, 5)
    Q = torch.tensor([[2., .5], [0., .2]])
    r1 = Representation("parameter", "a", 17, factors={"q": [LowRankFactor(B, A)]})
    r2 = Representation("parameter", "b", 17, factors={"q": [LowRankFactor(B @ Q, torch.linalg.solve(Q, A))]})
    assert abs(inner(r1, r2) - float(((B @ A) ** 2).sum())) < 1e-4
    assert norm(linear_combination([r1, r2], [1, -1])) < 1e-5
    assert max(layer_subspace_angles(r1, r2)["q"]["input_angles_degrees"]) < .001
    rnd = random_like(r1, 52)
    assert abs(norm(rnd) - norm(r1)) < 1e-5


def test_bootstrap_pairs_ids_and_theory_groups():
    base = [{"id": str(i), "group_id": str(i // 2), "score": 0.} for i in range(6)]
    condition = [{**x, "score": 1.} for x in reversed(base)]
    result = paired_bootstrap(condition, base, lambda rows: {"accuracy": np.mean([x["score"] for x in rows])}, repetitions=20)
    assert result["accuracy"]["ci"] == [1., 1.]
    assert result["accuracy"]["groups"] == 3


def test_zero_variance_retained_raw_no_forced_groups():
    tasks = ["a", "b", "c", "d"]
    scores = {t: {s: 1. for s in tasks} for t in tasks}
    matrix = transfer_matrix(tasks, scores, {t: 0. for t in tasks}, {t: [0., 0.] for t in tasks})
    assert matrix["raw"][0][0] == 1.
    assert discover_groups(matrix)["stable"] is False
    assert discover_groups(matrix)["groups"] == [tasks]


def test_joint_bootstrap_joins_shared_groups_transitively_and_keeps_point_estimate():
    groups = [("a", "x"), ("a", "y"), ("b", "y"), ("c", "z")]
    base = [{"id": str(i), "task": "joint", "group_id": f"{a}|{b}", "score": 0.,
             "metadata": {"subtask_group_ids": {"logic": a, "translation": b}}} for i, (a, b) in enumerate(groups)]
    condition = [{**row, "score": float(i < 3)} for i, row in enumerate(base)]
    observed_cluster_samples = []
    def score(rows):
        ids = [row["id"] for row in rows]
        observed_cluster_samples.append(ids)
        return {"joint_success": float(np.mean([row["score"] for row in rows]))}
    result = paired_bootstrap(list(reversed(condition)), base, score, repetitions=30)["joint_success"]
    assert result["delta"] == .75 and result["groups"] == 2
    assert result["underlying_groups"] == 6 and result["component_sizes"] == [3, 1]
    assert result["bootstrap_unit"] == "joint_subtask_group_connected_component"
    for ids in observed_cluster_samples[2:]:
        assert ids.count("0") == ids.count("1") == ids.count("2")


def test_joint_group_names_are_task_qualified_and_missing_groups_are_rejected():
    rows = [{"id": "1", "task": "joint", "group_id": "one", "score": 0., "metadata": {"subtask_group_ids": {"math": "a", "logic": "b"}}},
            {"id": "2", "task": "joint", "group_id": "two", "score": 0., "metadata": {"subtask_group_ids": {"math": "b", "logic": "a"}}}]
    score = lambda values: {"accuracy": np.mean([row["score"] for row in values])}
    assert paired_bootstrap(rows, rows, score, repetitions=2)["accuracy"]["groups"] == 2
    with pytest.raises(ValueError, match="subtask_group_ids"):
        paired_bootstrap([{**rows[0], "metadata": {}}], [{**rows[0], "metadata": {}}], score, repetitions=2)


def test_confirm_transfer_relations_aligns_names_and_does_not_select_on_test():
    discovery = {"tasks": ["a", "b", "c"], "raw": [[1, 2, 0], [-1, 3, 0], [2, -2, 0]],
                 "standardized": [[1, 1, 0], [-1, 1.5, 0], [2, -1, 0]], "column_std": [1, 2, 0]}
    order = [2, 0, 1]
    raw = np.asarray(discovery["raw"])
    final = {"tasks": ["c", "a", "b"], "raw": raw[np.ix_(order, order)].tolist(),
             "standardized": (raw / np.array([1., 4., 1.]))[np.ix_(order, order)].tolist(), "column_std": [0, 1, 4]}
    result = confirm_transfer_relations(discovery, final)
    assert result["tasks"] == ["a", "b", "c"] and result["off_diagonal"]["relations"] == 6
    assert result["off_diagonal"]["sign_agreement"] == 1
    assert result["off_diagonal"]["standardized_spearman"] == pytest.approx(1)
    assert result["excluded_zero_std_targets"] == ["c"] and result["selection_performed"] is False
    assert all(edge["raw_change"] == 0 for edge in result["edges"])


def test_confirm_transfer_relations_preserves_direction_changes_and_constant_reason():
    discovery = {"tasks": ["a", "b"], "raw": [[0, 1], [-1, 0]], "standardized": [[0, 1], [-1, 0]], "column_std": [1, 1]}
    final = {**discovery, "raw": [[0, -1], [-1, 0]], "standardized": [[0, -1], [-1, 0]]}
    result = confirm_transfer_relations(discovery, final)
    assert result["off_diagonal"]["sign_agreement"] == .5
    assert result["off_diagonal"]["standardized_spearman"] is None
    assert result["off_diagonal"]["correlation_undefined_reason"]


def test_confirm_transfer_relations_missing_stds_labels_per_panel_scale():
    discovery = {"tasks": ["a", "b"], "raw": [[0, 2], [-3, 0]], "standardized": [[0, 1], [-1, 0]]}
    final = {**discovery, "standardized": [[0, .5], [-.25, 0]]}
    result = confirm_transfer_relations(discovery, final)
    assert "baseline_std_unavailable" in result["standardization"]
    assert all(edge["confirmation_at_discovery_scale"] is None for edge in result["edges"])
    assert result["edges"][1]["confirmation_at_comparison_scale"] == .5


def test_joint_single_component_and_legacy_subtask_groups_are_explicit():
    rows = [{"id": str(i), "task": "joint", "score": float(i),
             "metadata": {"subtasks": [{"task": "logic", "group_id": "one_theory"}, {"task": "math", "group_id": str(i)}]}} for i in range(2)]
    result = paired_bootstrap(rows, rows, lambda values: {"joint_success": np.mean([row["score"] for row in values])}, repetitions=3)["joint_success"]
    assert result["connected_components"] == 1 and result["confidence_interval_identifiable"] is False
    assert result["limitation"] and result["ci"] == [0., 0.]


def test_random_subspace_controls_q012_match_norm_and_common_span():
    reps = [activation([3, 1, 0, 2, 0, 0, 0], "a"), activation([0, 2, 4, 0, 1, 0, 0], "b"), activation([0, 0, 1, 0, 0, 3, 2], "c")]
    for q in (0, 1, 2):
        decomposition = decompose(reps, "svd", q)
        controls = random_subspace_control(reps, decomposition, q, 17)
        values = torch.stack([rep.values[rep.keys[0]] for rep in controls], dim=1).double()
        assert int(torch.linalg.matrix_rank(values, atol=1e-5)) <= q
        for control, shared in zip(controls, decomposition.shared):
            assert abs(norm(control) - norm(shared)) < 1e-5
            assert control.metadata["used_rank"] == q


def test_nontrivial_functional_clusters_require_bootstrap_stability():
    tasks = ["a", "b", "c", "d"]
    values = np.array([[0., 0., 0., 0.], [.1, .1, .1, .1], [10., 10., 10., 10.], [10.1, 10.1, 10.1, 10.1]])
    matrix = {"tasks": tasks, "standardized": values.tolist(), "column_std": [1.] * 4}
    stable = discover_groups(matrix, bootstrap_matrices=[values, values + .01], max_groups=2)
    assert stable["stable"] and {frozenset(group) for group in stable["groups"]} == {frozenset({"a", "b"}), frozenset({"c", "d"})}
    crossed = values[[0, 2, 1, 3]]
    unstable = discover_groups(matrix, bootstrap_matrices=[crossed, values], max_groups=2)
    assert unstable["stable"] is False and unstable["groups"] == [tasks]
