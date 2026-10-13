import math
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from task_vector.subspaces import (Statistics, build_subspaces, collect_statistics,
                                   isolated_energy_basis, load_statistics,
                                   recalibration_diagonal, save_statistics,
                                   top_energy_basis, uniform_token_positions)


def test_uniform_positions_are_unique_complete_and_uniform():
    assert uniform_token_positions(4, 64).tolist() == [0, 1, 2, 3]
    positions = uniform_token_positions(512, 64)
    assert len(positions.unique()) == 64
    assert positions[0] == 0 and positions[-1] == 511
    gaps = positions[1:] - positions[:-1]
    assert int(gaps.max() - gaps.min()) <= 1


def test_general_is_top_energy_subspace():
    matrix = torch.diag(torch.tensor([2.0, 7.0, 1.0, 4.0]))
    basis, values = top_energy_basis(matrix, 2)
    assert torch.allclose(values, torch.tensor([7.0, 4.0], dtype=torch.float64))
    assert torch.allclose(basis @ basis.T, torch.diag(torch.tensor([0., 1., 0., 1.], dtype=torch.float64)))


def test_isolated_whitens_maps_back_and_qr_without_removing_general():
    # Largest raw energy is coordinate 0; largest relative energy is 2.
    current = torch.diag(torch.tensor([10., 3., 2., 1.]))
    other = torch.diag(torch.tensor([100., 3., .01, 1.]))
    basis, metadata = isolated_energy_basis(current, other, 1)
    assert torch.allclose(basis.abs(), torch.tensor([[0.], [0.], [1.], [0.]], dtype=torch.float64))
    assert metadata["regularization_value"] == pytest.approx(1e-4 * float(other.trace()) / 4)
    assert metadata["numerical_jitter"] == 0
    assert metadata["orthogonality_error"] < 1e-12


def test_isolated_matches_generalized_eigen_subspace_for_dense_matrices():
    torch.manual_seed(11)
    x, y = torch.randn(6, 6, dtype=torch.float64), torch.randn(6, 6, dtype=torch.float64)
    current, other = x @ x.T, y @ y.T + torch.eye(6)
    basis, metadata = isolated_energy_basis(current, other, 2)
    regularized = other + metadata["regularization_value"] * torch.eye(6)
    # The returned Euclidean-orthonormal span is an invariant subspace of
    # S_other^-1 S_self, although QR can mix its eigenvector columns.
    projected = torch.linalg.solve(regularized, current) @ basis
    residual = projected - basis @ (basis.T @ projected)
    assert residual.norm() / projected.norm() < 1e-10
    assert torch.allclose(basis.T @ basis, torch.eye(2, dtype=torch.float64), atol=1e-12)


def test_lambda_preserves_one_minus_eta_and_separate_epsilon():
    a = torch.eye(2)
    current, other = torch.diag(torch.tensor([2., 0.])), torch.diag(torch.tensor([1., 0.]))
    result = recalibration_diagonal(a, current, other, gamma=3, epsilon=.5)
    # epsilon makes 1-eta different from 3*g_cur / denominator.
    assert torch.allclose(result, torch.tensor([1 - 1 / 7.5, 1.]))


def test_build_freezes_seen_tasks_and_prescribed_a_scale():
    matrices = {
        "a": {"model.layers.0": torch.diag(torch.tensor([3., 2., 1., .5]))},
        "b": {"model.layers.0": torch.diag(torch.tensor([1., 2., 4., .5]))},
        "heldout": {"model.layers.0": torch.eye(4) * 1e8},
    }
    config = {"loda": {"rank_general": 2, "rank_isolated": 2, "seen_tasks": ["a", "b"]}}
    spec = build_subspaces(matrices, "a", config)["model.layers.0"]
    assert spec["metadata"]["seen_tasks"] == ["a", "b"]
    assert torch.allclose(spec["S_other"], matrices["b"]["model.layers.0"])
    for key in ("A_G", "A_I"):
        assert torch.allclose(spec[key] @ spec[key].T, torch.eye(2) / 3, atol=1e-6)
    with pytest.raises(ValueError, match="seen-task"):
        build_subspaces(matrices, "heldout", config)


def test_zero_other_statistics_cannot_silently_change_regularizer():
    with pytest.raises(ValueError, match="zero energy"):
        isolated_energy_basis(torch.eye(3), torch.zeros(3, 3), 1)


def test_statistics_roundtrip_is_weights_only_safe(tmp_path):
    value = Statistics({"layer": torch.eye(2)}, metadata={"sample_ids": ["1"], "token_counts": {"layer": 4}})
    save_statistics(value, tmp_path / "stats.pt")
    loaded = load_statistics(tmp_path / "stats.pt")
    assert loaded.metadata == value.metadata
    assert torch.equal(loaded["layer"], value["layer"])


class _TinyStatisticsModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = nn.Embedding(10, 3)
        self.model = nn.Module()
        self.model.layers = nn.ModuleList([nn.Module()])
        attn = nn.Module()
        attn.q_proj, attn.k_proj, attn.v_proj = nn.Linear(3, 3), nn.Linear(3, 2), nn.Linear(3, 2)
        self.model.layers[0].self_attn = attn
        with torch.no_grad():
            self.embed.weight.copy_(torch.arange(30).reshape(10, 3).float() / 10)

    def forward(self, input_ids, attention_mask, use_cache=False):
        return self.model.layers[0].self_attn.q_proj(self.embed(input_ids))


class _Tokenizer:
    def __call__(self, text, **kwargs):
        ids = torch.tensor([[int(x) for x in text.split()]])
        return {"input_ids": ids, "attention_mask": torch.ones_like(ids)}


def test_collector_uses_token_mean_and_prompt_positions(monkeypatch):
    import task_vector.prompts
    monkeypatch.setattr(task_vector.prompts, "render_prompt", lambda tokenizer, sample, variant: sample.prompt)
    model = _TinyStatisticsModel().train()
    samples = [SimpleNamespace(id="a", prompt="1 2", official_split="train"),
               SimpleNamespace(id="b", prompt="3 4 5 6", official_split="train")]
    stats = collect_statistics(model, _Tokenizer(), samples,
                               {"loda": {"statistics_samples": 2, "max_statistics_tokens": 3}}, "cpu")
    chosen_features = []
    for identifier, positions in zip(stats.metadata["sample_ids"], stats.metadata["token_positions"]):
        sample = next(s for s in samples if s.id == identifier)
        ids = torch.tensor([int(x) for x in sample.prompt.split()])
        chosen_features.append(model.embed(ids[positions]).detach())
    features = torch.cat(chosen_features)
    assert stats.metadata["token_counts"]["model.layers.0"] == 5
    assert torch.allclose(stats["model.layers.0"], features.T @ features / 5)
    sample_mean = sum(x.T @ x / len(x) for x in chosen_features) / 2
    assert not torch.allclose(stats["model.layers.0"], sample_mean)
    assert model.training
    assert not model.model.layers[0].self_attn.q_proj._forward_pre_hooks
