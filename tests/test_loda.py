import pytest
import torch
from torch import nn
from torch.nn import functional as F

from task_vector.loda import LoDAAdapter, merge_representation_
from task_vector.representations import Representation
from task_vector.sequential import cyclic_orders
from task_vector.subspaces import build_subspaces


class TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Module()
        self.model.layers = nn.ModuleList([nn.Module(), nn.Module()])
        for layer in self.model.layers:
            layer.self_attn = nn.Module()
            layer.self_attn.q_proj = nn.Linear(6, 6, bias=True)
            layer.self_attn.k_proj = nn.Linear(6, 3, bias=False)
            layer.self_attn.v_proj = nn.Linear(6, 3, bias=True)

    def forward(self, x):
        for layer in self.model.layers:
            attn = layer.self_attn
            x = attn.q_proj(x) + F.pad(attn.k_proj(x) + attn.v_proj(x), (0, 3))
        return x


def subspaces():
    stats = {task: {f"model.layers.{layer}": torch.diag(torch.arange(1, 7).float() + index)
                    for layer in range(2)} for index, task in enumerate(("a", "b"))}
    return build_subspaces(stats, "a", {"loda": {"rank_general": 2, "rank_isolated": 2}})


def test_zero_B_native_model_equivalence_and_shape_and_freeze():
    model = TinyModel()
    x = torch.randn(2, 5, 6)
    expected = model(x)
    adapter = LoDAAdapter(model, subspaces())
    assert not adapter.is_active
    assert all(not p.requires_grad for p in model.parameters())
    assert len(list(adapter.parameters())) == 12
    assert all(p.dtype == torch.float32 for p in adapter.parameters())
    assert not any("model.layers" in name for name in adapter.state_dict())
    for layer in adapter.layers.values():
        assert layer.B_G["q_proj"].shape == (6, 2)
        assert layer.B_G["k_proj"].shape == layer.B_G["v_proj"].shape == (3, 2)
        assert layer.A_G.shape == layer.A_I.shape == (2, 6)
    with adapter:
        assert torch.equal(model(x), expected)
        adapter.set_recalibrated(True)
        assert torch.equal(model(x), expected)
    assert not adapter.is_active
    assert torch.equal(model(x), expected)


def test_general_A_shared_across_qkv_and_real_update_scaling():
    model = TinyModel()
    adapter = LoDAAdapter(model, subspaces())
    with torch.no_grad():
        for layer in adapter.layers.values():
            for b in list(layer.B_G.values()) + list(layer.B_I.values()):
                b.normal_()
    x = torch.randn(2, 5, 6)
    for version in ("raw", "post"):
        adapter.set_recalibrated(version == "post")
        rep = adapter.representation("a", 17, version)
        for key, layer in adapter.layers.items():
            for kind in ("q_proj", "k_proj", "v_proj"):
                name = f"{adapter._layer_names[key]}.self_attn.{kind}"
                dense = sum(term.dense() for term in rep.factors[name])
                assert torch.allclose(layer.delta(kind, x, version == "post"), F.linear(x, dense), atol=1e-6)
        with adapter:
            actual = model(x)
        weights = {name: model.get_submodule(name).weight.detach().clone() for name in rep.keys}
        merge_representation_(model, rep)
        assert torch.allclose(model(x), actual, atol=5e-6)
        with torch.no_grad():
            for name, weight in weights.items():
                model.get_submodule(name).weight.copy_(weight)


def test_checkpoint_roundtrip_and_exception_uninstalls_hooks(tmp_path):
    model = TinyModel()
    adapter = LoDAAdapter(model, subspaces())
    with torch.no_grad():
        for layer in adapter.layers.values():
            for b in layer.B_G.values():
                b.normal_()
    rep = adapter.representation("a", 17, "post")
    rep.save(tmp_path / "post.pt")
    restored = Representation.load(tmp_path / "post.pt")
    for name in rep.keys:
        assert torch.equal(restored.factors[name][0].dense(), rep.factors[name][0].dense())
    with pytest.raises(RuntimeError, match="expected failure"):
        with adapter:
            raise RuntimeError("expected failure")
    assert not adapter.is_active
    assert all(not module._forward_hooks for modules in adapter._projections.values() for module in modules.values())


def test_first_sequential_stage_trains_only_general_A_B():
    adapter = LoDAAdapter(TinyModel(), subspaces(), train_general_a=True, isolated_enabled=False)
    for layer in adapter.layers.values():
        assert isinstance(layer.A_G, nn.Parameter) and layer.A_G.requires_grad
        assert not layer.A_I.requires_grad
        assert all(not p.requires_grad for p in layer.B_I.values())
        assert all(p.requires_grad for p in layer.B_G.values())
    assert cyclic_orders(["a", "b", "c", "d"])[-1] == ("d", "a", "b", "c")
