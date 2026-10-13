import ast
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from task_vector.baselines import (knots_merge, merge_baseline, ties_merge,
                                   ties_tensor, tsv_merge)
from task_vector.representations import LowRankFactor, Representation, inner


ROOT = Path(__file__).resolve().parents[1]


def _author_functions(path, names):
    # Execute only the named pure tensor functions from pinned author sources;
    # upstream import-time model/data dependencies are intentionally excluded.
    tree = ast.parse(path.read_text(encoding="utf-8"))
    selected = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names]
    assert {node.name for node in selected} == set(names)
    scope = {"torch": torch}
    exec(compile(ast.Module(body=selected, type_ignores=[]), str(path), "exec"), scope)
    return scope


def _representations(count=2, seed=19, shapes=((6, 6), (4, 6))):
    generator = torch.Generator().manual_seed(seed)
    return [Representation("parameter", f"task{i}", 17,
                           factors={f"layer{j}": [LowRankFactor(torch.randn(m, n, generator=generator), torch.eye(n))]
                                    for j, (m, n) in enumerate(shapes)}) for i in range(count)]


def _dense(rep, key):
    return sum(term.dense() for term in rep.factors[key])


def test_ties_matches_author_topk_mass_sign_and_disjoint_mean():
    source = _author_functions(ROOT / "third_party/TIES/src/utils/merge_utils.py",
                               ["topk_values_mask", "resolve_zero_signs", "resolve_sign", "disjoint_merge"])
    generator = torch.Generator().manual_seed(17)
    vectors = torch.randn(4, 31, generator=generator)
    vectors[0, :3] = 0
    pruned = source["topk_values_mask"](vectors, K=.2)[0]
    sign = source["resolve_sign"](pruned, "mass")
    expected = source["disjoint_merge"](pruned, "dis-mean", sign)
    actual, metadata = ties_tensor(vectors, keep_fraction=.2, merge="mean")
    assert torch.equal(actual, expected)
    assert metadata["realized_keep_fraction"][0] >= .2


def test_ties_inclusive_boundary_and_global_cross_layer_pruning():
    # Inclusive source threshold keeps 3 of 10 for K=.2, rather than torch.topk's 2.
    vector = torch.arange(1., 11.).unsqueeze(0)
    actual, metadata = ties_tensor(vector, .2)
    assert int((actual != 0).sum()) == 3
    reps = [Representation("activation", "a", 17,
                           values={"high": torch.tensor([100., 90., 80., 70.]),
                                   "low": torch.tensor([1., 2., 3., 4.])})]
    merged = ties_merge(reps, keep_fraction=.2)
    assert torch.equal(merged.values["low"], torch.zeros(4))
    assert int((merged.values["high"] != 0).sum()) == 2


def test_tsv_matches_author_orthogonalization_for_rectangular_modules():
    source = _author_functions(ROOT / "third_party/TSV/src/utils/TSVM_utils.py",
                               ["compute_and_sum_svd_mem_reduction"])
    reps = _representations()
    vectors = [SimpleNamespace(vector={key: _dense(rep, key).double() for key in rep.keys}) for rep in reps]
    # The upstream function allocates zero tensors using torch's default dtype;
    # use float64 for both implementations to test the mathematical algorithm.
    original_dtype = torch.get_default_dtype()
    try:
        torch.set_default_dtype(torch.float64)
        expected = source["compute_and_sum_svd_mem_reduction"](
            vectors, SimpleNamespace(DATASETS=["a", "b"], device="cpu", num_tasks=2))
    finally:
        torch.set_default_dtype(original_dtype)
    actual = tsv_merge(reps)
    for key in actual.keys:
        assert torch.allclose(_dense(actual, key).double(), expected[key], atol=1e-6)
        assert actual.metadata["settings"]["layers"][key]["retained_per_task"] == min(expected[key].shape) // 2


def test_knots_matches_author_joint_SVD_masked_sV_and_reconstruction():
    source = _author_functions(ROOT / "third_party/KnOTS/masking_ops.py",
                               ["topk_values_mask", "resolve_zero_signs", "resolve_sign", "ties_masking",
                                "chunked_disjoint_mean", "chunked_sum", "masked_merge"])
    reps = _representations(count=3, shapes=((5, 7), (3, 7)))
    bases, coordinates, shapes = {}, [dict() for _ in reps], {}
    for key in reps[0].keys:
        joined = torch.cat([_dense(rep, key).double() for rep in reps], dim=1)
        u, s, v = torch.linalg.svd(joined, full_matrices=False)
        kept = s > 1e-5
        u, s, v = u[:, kept].float(), s[kept].float(), v[kept].float()
        bases[key] = u
        for index, part in enumerate(v.split(7, dim=1)):
            coordinates[index][key] = torch.diag(s) @ part
        shapes[key] = coordinates[0][key].shape
    flat = [torch.cat([row[key].reshape(-1) for key in reps[0].keys]) for row in coordinates]
    masks = source["ties_masking"](flat, topK=20, sign_resolve_mode="sum_of_values")
    merged = source["masked_merge"](torch.stack(flat) * masks, merge_func="mean")
    actual = knots_merge(reps, keep_fraction=.2)
    offset = 0
    for key in actual.keys:
        length = shapes[key].numel()
        expected = bases[key] @ merged[offset:offset + length].reshape(shapes[key])
        assert torch.allclose(_dense(actual, key), expected, atol=1e-6)
        offset += length


def test_actual_update_factor_reparameterization_invariance():
    reps = _representations()
    changed = [Representation("parameter", rep.task, rep.seed,
                              factors={key: [LowRankFactor(term.B * 4, term.A / 4) for term in rep.factors[key]]
                                       for key in rep.keys}) for rep in reps]
    for method in ("ties", "tsv", "knots"):
        original, equivalent = merge_baseline(reps, method), merge_baseline(changed, method)
        for key in original.keys:
            assert torch.allclose(_dense(original, key), _dense(equivalent, key), atol=1e-6)


def test_matrix_baselines_reject_activation_and_zero_knots_is_valid():
    activation = Representation("activation", "x", 17, values={"layer": torch.ones(6)})
    with pytest.raises(ValueError, match="matrix parameter-space"):
        tsv_merge([activation])
    with pytest.raises(ValueError, match="matrix parameter-space"):
        knots_merge([activation])
    zero = Representation("parameter", "zero", 17,
                          factors={"layer": [LowRankFactor(torch.zeros(4, 2), torch.ones(2, 6))]})
    result = knots_merge([zero, zero])
    assert inner(result, result) == 0
    assert result.metadata["settings"]["all_updates_below_singular_threshold"]
