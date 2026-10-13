"""Fixed prompt-position, block-input task vectors.

Hooks live through backward, including non-reentrant checkpoint recomputation.
No hook is ever attached after a block or to a layer's RMSNorm output.
"""
from __future__ import annotations

from contextlib import AbstractContextManager, nullcontext
import math
import torch
from torch import nn

from .representations import Representation


class LTVIntervention(AbstractContextManager):
    def __init__(self, model, representation: Representation):
        if representation.space != "activation":
            raise ValueError("LTV needs an activation representation")
        self.model = model
        self.representation = representation
        self.positions = None
        self._forced_prefill = None
        self._is_prefill = True
        self._handles = []

    def set_positions(self, positions):
        self.positions = positions

    def set_prefill(self, prefill):
        """None restores automatic cache detection; bool explicitly overrides it."""
        self._forced_prefill = prefill

    def _model_pre(self, module, args, kwargs):
        cache = kwargs.get("past_key_values")
        cache_pos = kwargs.get("cache_position")
        occupied = False
        if cache_pos is not None and cache_pos.numel():
            occupied = int(cache_pos.reshape(-1)[0]) > 0
        elif cache is not None:
            if hasattr(cache, "get_seq_length"):
                occupied = int(cache.get_seq_length()) > 0
            elif len(cache):
                occupied = cache[0][0].shape[-2] > 0
        self._is_prefill = not occupied

    def _make_hook(self, key):
        def hook(module, args, kwargs):
            active = self._is_prefill if self._forced_prefill is None else self._forced_prefill
            if not active:
                return None
            hidden = args[0] if args else kwargs["hidden_states"]
            if self.positions is None:
                raise RuntimeError("Set original prompt positions before LTV forward")
            pos = self.positions.to(device=hidden.device, dtype=torch.long)
            if pos.numel() != hidden.shape[0] or bool((pos < 0).any()) or bool((pos >= hidden.shape[1]).any()):
                raise ValueError("Prompt position lies outside the current prefill")
            vector = self.representation.values[key].to(device=hidden.device, dtype=hidden.dtype)
            if vector.shape != (hidden.shape[-1],):
                raise ValueError(f"Incorrect hidden dimension at {key}")
            # Out-of-place update is safe for frozen activations and checkpointing.
            addition = torch.zeros_like(hidden)
            addition[torch.arange(hidden.shape[0], device=hidden.device), pos] = vector
            changed = hidden + addition
            if args:
                return (changed, *args[1:]), kwargs
            return args, {**kwargs, "hidden_states": changed}
        return hook

    def __enter__(self):
        if self._handles:
            raise RuntimeError("An LTV context cannot be entered twice")
        self._handles.append(self.model.register_forward_pre_hook(self._model_pre, with_kwargs=True))
        try:
            for key in self.representation.keys:
                block = self.model.get_submodule(key)
                self._handles.append(block.register_forward_pre_hook(self._make_hook(key), with_kwargs=True))
        except BaseException:
            self.__exit__(None, None, None)
            raise
        return self

    def __exit__(self, exc_type, exc, tb):
        for handle in reversed(self._handles):
            handle.remove()
        self._handles.clear()
        return False


class LTVAdapter(nn.Module):
    def __init__(self, model, layers=(14,), seed=17, initial_representation=None, *,
                 init_low=-0.1, init_high=0.1):
        super().__init__()
        # Avoid registering the entire frozen model as part of the adapter.
        object.__setattr__(self, "model", model)
        self.layers = tuple(map(int, layers))
        self.initialization_seed = int(seed)
        self.init_low, self.init_high = float(init_low), float(init_high)
        if not math.isfinite(self.init_low) or not math.isfinite(self.init_high) or self.init_low >= self.init_high:
            raise ValueError("LTV initialization bounds must be finite and increasing")
        self.initialization_source = "provided_representation" if initial_representation is not None else "uniform"
        if not self.layers or len(set(self.layers)) != len(self.layers):
            raise ValueError("LTV layers must be nonempty and distinct")
        width = int(model.config.hidden_size)
        device = next(model.parameters()).device
        rng = torch.Generator(device="cpu").manual_seed(int(seed))
        self.vectors = nn.ParameterDict()
        for layer in self.layers:
            key = f"model.layers.{layer}"
            model.get_submodule(key)
            value = torch.empty(width, dtype=torch.float32).uniform_(self.init_low, self.init_high, generator=rng)
            if initial_representation is not None:
                if initial_representation.space != "activation" or key not in initial_representation.values:
                    raise ValueError("Initial LTV coordinates differ")
                value = initial_representation.values[key].float().clone()
            self.vectors[str(layer)] = nn.Parameter(value.to(device))
        self.context = None

    def live_representation(self):
        return Representation("activation", "live", 0,
                              values={f"model.layers.{layer}": self.vectors[str(layer)] for layer in self.layers})

    def representation(self, task, seed, version="raw"):
        return Representation("activation", task, int(seed),
            values={k: v.detach().float().cpu().clone() for k, v in self.live_representation().values.items()},
            metadata={"version": version, "layers": list(self.layers), "position": "original_prompt_last",
                      "hook": "decoder_block_input_before_rmsnorm", "inference": "prefill_only",
                      "initialization_seed": self.initialization_seed,
                      "initialization_source": self.initialization_source,
                      "init_low": self.init_low, "init_high": self.init_high,
                      "model_id": getattr(self.model.config, "_name_or_path", "unknown")})

    def set_positions(self, positions):
        if self.context is None:
            raise RuntimeError("Enter LTVAdapter before setting positions")
        self.context.set_positions(positions)

    def set_prefill(self, prefill):
        if self.context is None:
            raise RuntimeError("Enter LTVAdapter before setting prefill")
        self.context.set_prefill(prefill)

    def __enter__(self):
        self.context = LTVIntervention(self.model, self.live_representation())
        self.context.__enter__()
        return self

    def __exit__(self, *args):
        if self.context is not None:
            self.context.__exit__(*args)
            self.context = None


class _ParameterProjection(nn.Module):
    def __init__(self, original, factors):
        super().__init__()
        self.original = original
        # Exact represented BA with scaling already included in B. Preserve
        # the original attention module, GQA, Q/K norms and RoPE unchanged.
        for i, factor in enumerate(factors):
            self.register_buffer(f"A_{i}", factor.A.detach().float().to(original.weight.device))
            self.register_buffer(f"B_{i}", factor.B.detach().float().to(original.weight.device))
        self.count = len(factors)

    @property
    def weight(self):
        return self.original.weight

    def forward(self, x):
        output = self.original(x)
        delta = torch.zeros_like(output, dtype=torch.float32)
        # Explicitly disable ambient autocast: geometry refers to these FP32 BA
        # products, not a BF16-rounded factorization.
        with torch.autocast(device_type=x.device.type, enabled=False):
            for i in range(self.count):
                delta = delta + (x.float() @ getattr(self, f"A_{i}").T) @ getattr(self, f"B_{i}").T
        return output + delta.to(output.dtype)


class ParameterIntervention(AbstractContextManager):
    def __init__(self, model, representation):
        self.model, self.rep, self.originals = model, representation, []

    def __enter__(self):
        try:
            for key in self.rep.keys:
                parent_key, child = key.rsplit(".", 1)
                parent = self.model.get_submodule(parent_key)
                original = getattr(parent, child)
                if not isinstance(original, nn.Linear):
                    raise TypeError(f"Expected an untouched Linear at {key}")
                self.originals.append((parent, child, original))
                setattr(parent, child, _ParameterProjection(original, self.rep.factors[key]))
        except BaseException:
            self.__exit__(None, None, None)
            raise
        return self

    def __exit__(self, exc_type, exc, tb):
        for parent, child, original in reversed(self.originals):
            setattr(parent, child, original)
        self.originals.clear()
        return False


def intervention_context(model, representation, positions=None):
    if representation is None:
        return nullcontext()
    if representation.space == "parameter":
        return ParameterIntervention(model, representation)
    context = LTVIntervention(model, representation)
    if positions is not None:
        context.set_positions(positions)
    return context
