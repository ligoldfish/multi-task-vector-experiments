"""LoDA-MT: shared input subspaces and independent native Q/K/V updates.

Forward hooks add only projection outputs. Qwen's GQA, Q/K normalization,
attention implementation, and RoPE remain the original model code.
"""
from __future__ import annotations

from contextlib import nullcontext
from typing import Mapping, Sequence
import weakref

import torch
from torch import nn
from torch.nn import functional as F

from .representations import LowRankFactor, Representation
from .subspaces import recalibration_diagonal


class _LoDALayer(nn.Module):
    def __init__(self, projections: Mapping[str, nn.Module], subspace: dict,
                 train_general_a: bool, isolated_enabled: bool):
        super().__init__()
        reference = projections["q_proj"].weight
        a_g = subspace["A_G"].detach().to(reference.device, torch.float32).clone()
        a_i = subspace["A_I"].detach().to(reference.device, torch.float32).clone()
        if a_g.ndim != 2 or a_i.ndim != 2 or a_g.shape[1] != reference.shape[1] or a_i.shape[1] != reference.shape[1]:
            raise ValueError("A must have shape [rank, native projection input width]")
        if train_general_a:
            self.A_G = nn.Parameter(a_g)
        else:
            self.register_buffer("A_G", a_g)
        self.register_buffer("A_I", a_i)
        diagonal = subspace.get("lambda", torch.ones(a_g.shape[0]))
        self.register_buffer("Lambda", diagonal.detach().to(reference.device, torch.float32).clone())
        if self.Lambda.shape != (a_g.shape[0],):
            raise ValueError("General recalibration diagonal shape differs from General rank")
        self.B_G, self.B_I = nn.ParameterDict(), nn.ParameterDict()
        for kind, projection in projections.items():
            if projection.weight.shape[1] != reference.shape[1]:
                raise ValueError("Q/K/V must share the input feature dimension")
            self.B_G[kind] = nn.Parameter(torch.zeros((projection.weight.shape[0], a_g.shape[0]),
                                                     device=reference.device, dtype=torch.float32))
            self.B_I[kind] = nn.Parameter(torch.zeros((projection.weight.shape[0], a_i.shape[0]),
                                                     device=reference.device, dtype=torch.float32),
                                         requires_grad=isolated_enabled)
        self.isolated_enabled = bool(isolated_enabled)

    def delta(self, kind: str, x: torch.Tensor, recalibrated: bool):
        # Explicitly disable outer autocast: master A/B and their accumulation
        # stay FP32 even when the frozen native backbone has BF16 weights.
        with torch.autocast(device_type=x.device.type, enabled=False):
            general = F.linear(x.float(), self.A_G)
            if recalibrated:
                general = general * self.Lambda
            result = 0.5 * F.linear(general, self.B_G[kind])
            if self.isolated_enabled:
                result = result + F.linear(F.linear(x.float(), self.A_I), self.B_I[kind])
        return result


class LoDAAdapter(nn.Module):
    """Trainable adapter, installed only while its context is active.

    MT freezes the base and both As. The first *sequential* stage may opt into
    train_general_a=True and isolated_enabled=False, matching official LoDA.
    The model is weakly referenced and is never a registered child module.
    """

    def __init__(self, model: nn.Module, subspaces: Mapping[str, dict],
                 train_general_a: bool = False, isolated_enabled: bool = True):
        super().__init__()
        if not subspaces:
            raise ValueError("LoDA requires subspaces for all selected attention layers")
        self._model_ref = weakref.ref(model)
        self.layers = nn.ModuleDict()
        self._layer_names, self._projections, self._statistics = {}, {}, {}
        self._metadata = {}
        self._handles = []
        self.recalibrated = False
        self.train_general_a = bool(train_general_a)
        self.isolated_enabled = bool(isolated_enabled)
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        for i, layer_name in enumerate(sorted(subspaces)):
            projections = {kind: model.get_submodule(f"{layer_name}.self_attn.{kind}")
                           for kind in ("q_proj", "k_proj", "v_proj")}
            for kind, projection in projections.items():
                if not isinstance(projection, nn.Linear):
                    raise TypeError(f"Expected native nn.Linear projection at {layer_name}.{kind}")
            key = f"layer_{i}"
            self.layers[key] = _LoDALayer(projections, subspaces[layer_name], train_general_a, isolated_enabled)
            self._layer_names[key] = layer_name
            self._projections[key] = projections
            self._statistics[key] = {name: subspaces[layer_name][name].detach().cpu()
                                     for name in ("S_self", "S_other") if name in subspaces[layer_name]}
            self._metadata[key] = dict(subspaces[layer_name].get("metadata", {}))

    @property
    def model(self):
        model = self._model_ref()
        if model is None:
            raise RuntimeError("LoDA base model has been released")
        return model

    @property
    def is_active(self):
        return bool(self._handles)

    def activate(self):
        if self.is_active:
            raise RuntimeError("LoDA adapter is already active; do not register duplicate hooks")
        for key, layer in self.layers.items():
            for kind, projection in self._projections[key].items():
                def add_delta(module, args, output, layer=layer, kind=kind):
                    if not isinstance(output, torch.Tensor):
                        raise TypeError("Native QKV linear projection must return a tensor")
                    return output + layer.delta(kind, args[0], self.recalibrated).to(output.dtype)
                self._handles.append(projection.register_forward_hook(add_delta))
        return self

    def deactivate(self):
        for handle in self._handles:
            handle.remove()
        self._handles.clear()

    def __enter__(self):
        return self.activate()

    def __exit__(self, exc_type, exc, tb):
        self.deactivate()
        return False

    def set_recalibrated(self, enabled: bool):
        if enabled:
            for key, layer in self.layers.items():
                stats = self._statistics[key]
                if stats:
                    meta = self._metadata[key]
                    diagonal = recalibration_diagonal(
                        layer.A_G, stats["S_self"], stats["S_other"],
                        float(meta.get("recalibration_lambda", 3.0)),
                        float(meta.get("epsilon", 1e-12)))
                    layer.Lambda.copy_(diagonal.to(layer.Lambda.device))
        self.recalibrated = bool(enabled)
        return self

    def representation(self, task: str, seed: int, version: str = "raw",
                       branches: Sequence[str] = ("general", "isolated")) -> Representation:
        if version not in {"raw", "post"} or not branches or set(branches) - {"general", "isolated"}:
            raise ValueError("Unknown LoDA version or branch selection")
        # Recompute Lambda for any trained A at the first sequential stage, but
        # export must not switch the live adapter's forward behavior.
        previous = self.recalibrated
        if version == "post":
            self.set_recalibrated(True)
            self.recalibrated = previous
        factors = {}
        lambda_values = {}
        for key, layer in self.layers.items():
            layer_name = self._layer_names[key]
            lambda_values[layer_name] = layer.Lambda.detach().cpu().tolist()
            a_general, a_isolated = layer.A_G.detach().float().cpu(), layer.A_I.detach().float().cpu()
            for kind in ("q_proj", "k_proj", "v_proj"):
                terms = []
                if "general" in branches:
                    b = layer.B_G[kind].detach().float().cpu() * 0.5
                    if version == "post":
                        b = b * layer.Lambda.detach().float().cpu().unsqueeze(0)
                    terms.append(LowRankFactor(b.clone(), a_general.clone()))
                if "isolated" in branches:
                    b = layer.B_I[kind].detach().float().cpu()
                    if not self.isolated_enabled:
                        b = torch.zeros_like(b)
                    terms.append(LowRankFactor(b.clone(), a_isolated.clone()))
                factors[f"{layer_name}.self_attn.{kind}"] = terms
        metadata = {"route": "sequential_LoDA" if self.train_general_a else "LoDA-MT",
                    "version": version, "general_scale": 0.5, "isolated_scale": 1.0,
                    "branches": list(branches), "A_general_trainable": self.train_general_a,
                    "isolated_enabled": self.isolated_enabled,
                    "master_dtype": "float32", "lambda": lambda_values,
                    "subspaces": {self._layer_names[k]: self._metadata[k] for k in self.layers}}
        return Representation("parameter", task, seed, factors=factors, metadata=metadata)


@torch.no_grad()
def merge_representation_(model: nn.Module, representation: Representation):
    """Merge a sequential update into its current base, retaining native modules."""
    if representation.space != "parameter":
        raise ValueError("Only actual parameter updates can be merged")
    for name, terms in representation.factors.items():
        module = model.get_submodule(name)
        if not isinstance(module, nn.Linear):
            raise TypeError(f"Merge destination is not a native Linear: {name}")
        # Each module is materialized independently, avoiding one enormous
        # dense representation; all represented rank terms remain included.
        update = torch.zeros_like(module.weight, dtype=torch.float32)
        for factor in terms:
            b, a = factor.B.to(update.device, torch.float32), factor.A.to(update.device, torch.float32)
            if (b.shape[0], a.shape[1]) != tuple(module.weight.shape):
                raise ValueError(f"Merge shape mismatch at {name}")
            update.addmm_(b, a)
        module.weight.copy_((module.weight.float() + update).to(module.weight.dtype))
