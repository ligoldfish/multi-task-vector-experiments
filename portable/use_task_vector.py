"""Use exported task vectors with PyTorch alone; the experiment package is optional.

The payload is an ordinary dictionary, so inspecting tensors needs only::

    import torch
    payload = torch.load("post.pt", map_location="cpu", weights_only=True)
    # LTV: payload["values"]["model.layers.14"]
    # LoDA: payload["factors"][projection_name][term_index]["A" or "B"]

With an already loaded Hugging Face Qwen3 model and tokenizer::

    from use_task_vector import verify_base_directory, task_vector_context
    verify_base_directory(model_path, "BASE_MODEL_CONTRACT.json")
    tokenizer.padding_side = "left"
    # Render the original task instruction with enable_thinking=False first.
    inputs = tokenizer(rendered_prompts, padding=True, return_tensors="pt").to(device)
    positions = torch.full((inputs.input_ids.shape[0],), inputs.input_ids.shape[1] - 1,
                           device=device, dtype=torch.long)
    with torch.no_grad(), task_vector_context(model, "post.pt", positions) as intervention:
        generated = model.generate(**inputs, do_sample=False, use_cache=True,
                                   num_beams=1, max_new_tokens=512)

LTV changes the decoder block INPUT before its RMSNorm, at the original last
prompt token, and only during prefill. Cached generation detects occupied KV
cache automatically. A custom decoding loop should additionally call
``intervention.set_prefill(step == 0)`` before each forward. For full-prefix
decoding without a cache, keep the ORIGINAL prompt positions fixed and use
``set_prefill(True)``; never move the injection to the latest generated token.
Beam expansion is not supported: positions must match the forward batch size.

LoDA uses the represented FP32 low-rank branch updates in parallel with the
native Q/K/V linears. It does not merge into BF16 base weights: that would add
rounding and need not be numerically equivalent. B already includes the branch
scale and, for post.pt, recalibration; do not apply either a second time.

This helper does not choose prompts, decode limits, scoring, or task-vector
strength. Matching the original experiment requires preserving those settings.
"""
from __future__ import annotations

import argparse
from collections.abc import Mapping
from contextlib import AbstractContextManager, nullcontext
import hashlib
import json
from pathlib import Path

import torch
from torch import nn


def _validate_payload(payload):
    if not isinstance(payload, Mapping) or payload.get("schema_version") != 1:
        raise ValueError("Expected a schema_version=1 plain task-vector dictionary")
    space = payload.get("space")
    values, factors = payload.get("values", {}), payload.get("factors", {})
    if not isinstance(values, Mapping) or not isinstance(factors, Mapping):
        raise ValueError("values and factors must be dictionaries")
    if space == "activation":
        if not values or factors:
            raise ValueError("An LTV payload needs nonempty values and no factors")
        for key, vector in values.items():
            if not isinstance(key, str) or not torch.is_tensor(vector) or vector.ndim != 1:
                raise ValueError("LTV values must map module paths to one-dimensional tensors")
    elif space == "parameter":
        if not factors or values:
            raise ValueError("A LoDA payload needs nonempty factors and no values")
        for key, terms in factors.items():
            if not isinstance(key, str) or not isinstance(terms, (list, tuple)) or not terms:
                raise ValueError("LoDA factors must map projection paths to nonempty term lists")
            for term in terms:
                if not isinstance(term, Mapping) or not all(torch.is_tensor(term.get(k)) for k in ("A", "B")):
                    raise ValueError("Each LoDA term must contain tensor A and tensor B")
                a, b = term["A"], term["B"]
                if a.ndim != 2 or b.ndim != 2 or b.shape[1] != a.shape[0]:
                    raise ValueError("Expected A[rank,in] and B[out,rank]")
    else:
        raise ValueError(f"Unknown representation space: {space!r}")
    return payload


def load_plain(path):
    """Load a plain CPU tensor dictionary without importing task_vector or NPU."""
    return _validate_payload(torch.load(path, map_location="cpu", weights_only=True))


class _LTVContext(AbstractContextManager):
    def __init__(self, model, payload, positions=None):
        self.model, self.payload, self.positions = model, payload, positions
        self._forced_prefill, self._is_prefill, self._handles = None, True, []

    def set_positions(self, positions):
        self.positions = positions

    def set_prefill(self, prefill):
        """None detects cache occupancy; True/False explicitly override it."""
        if prefill is not None and type(prefill) is not bool:
            raise TypeError("prefill must be True, False, or None")
        self._forced_prefill = prefill

    def _model_pre(self, module, args, kwargs):
        cache, cache_pos = kwargs.get("past_key_values"), kwargs.get("cache_position")
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
            pos = torch.as_tensor(self.positions, device=hidden.device, dtype=torch.long)
            if pos.ndim != 1 or pos.numel() != hidden.shape[0] or bool((pos < 0).any()) or bool((pos >= hidden.shape[1]).any()):
                raise ValueError("Prompt positions must match the batch and lie inside the current prefill")
            vector = self.payload["values"][key].to(device=hidden.device, dtype=hidden.dtype)
            if vector.shape != (hidden.shape[-1],):
                raise ValueError(f"Incorrect hidden dimension at {key}")
            addition = torch.zeros_like(hidden)
            addition[torch.arange(hidden.shape[0], device=hidden.device), pos] = vector
            changed = hidden + addition
            return ((changed, *args[1:]), kwargs) if args else (args, {**kwargs, "hidden_states": changed})
        return hook

    def __enter__(self):
        if self._handles:
            raise RuntimeError("An LTV context cannot be entered twice")
        self._handles.append(self.model.register_forward_pre_hook(self._model_pre, with_kwargs=True))
        try:
            for key in sorted(self.payload["values"]):
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


class _Projection(nn.Module):
    def __init__(self, original, terms):
        super().__init__()
        self.original = original
        for i, term in enumerate(terms):
            a, b = term["A"], term["B"]
            if (b.shape[0], a.shape[1]) != tuple(original.weight.shape):
                raise ValueError("Task-vector factors do not match the native projection dimensions")
            self.register_buffer(f"A_{i}", a.detach().float().to(original.weight.device))
            self.register_buffer(f"B_{i}", b.detach().float().to(original.weight.device))
        self.count = len(terms)

    @property
    def weight(self):
        return self.original.weight

    def forward(self, x):
        output = self.original(x)
        delta = torch.zeros_like(output, dtype=torch.float32)
        with torch.autocast(device_type=x.device.type, enabled=False):
            for i in range(self.count):
                delta = delta + (x.float() @ getattr(self, f"A_{i}").T) @ getattr(self, f"B_{i}").T
        return output + delta.to(output.dtype)


class _ParameterContext(AbstractContextManager):
    def __init__(self, model, payload):
        self.model, self.payload, self.originals = model, payload, []

    def __enter__(self):
        if self.originals:
            raise RuntimeError("A LoDA context cannot be entered twice")
        try:
            for key in sorted(self.payload["factors"]):
                parent_key, child = key.rsplit(".", 1)
                parent = self.model.get_submodule(parent_key)
                original = getattr(parent, child)
                if not isinstance(original, nn.Linear):
                    raise TypeError(f"Expected an untouched native Linear at {key}")
                wrapper = _Projection(original, self.payload["factors"][key])
                self.originals.append((parent, child, original))
                setattr(parent, child, wrapper)
        except BaseException:
            self.__exit__(None, None, None)
            raise
        return self

    def __exit__(self, exc_type, exc, tb):
        for parent, child, original in reversed(self.originals):
            setattr(parent, child, original)
        self.originals.clear()
        return False


def task_vector_context(model, path_or_payload, positions=None):
    """Temporarily apply an LTV or LoDA plain payload, restoring the model on exit."""
    if path_or_payload is None:
        return nullcontext()
    payload = _validate_payload(path_or_payload) if isinstance(path_or_payload, Mapping) else load_plain(path_or_payload)
    if payload["space"] == "activation":
        return _LTVContext(model, payload, positions)
    return _ParameterContext(model, payload)


def _file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_base_directory(model_path, contract):
    """Verify exact base-file names, content hashes, and the original fingerprint.

    This verifies a directory before loading it. The caller must load the model
    from that same directory; it cannot prove an already mutated model's state.
    """
    if not isinstance(contract, Mapping):
        with Path(contract).open(encoding="utf-8") as handle:
            contract = json.load(handle)
    expected = contract.get("files")
    if not isinstance(expected, Mapping) or not expected:
        raise ValueError("Base contract must contain a nonempty files->SHA256 mapping")
    for name in expected:
        if not isinstance(name, str) or not name or "/" in name or "\\" in name or name in {".", ".."}:
            raise ValueError(f"Invalid base-contract file name: {name!r}")
    root = Path(model_path).resolve(strict=True)
    if not root.is_dir():
        raise ValueError("model_path must be a directory")
    recognized = {p.name for p in root.iterdir() if p.is_file() and p.suffix in {".json", ".safetensors", ".bin", ".model", ".txt"}}
    if recognized != set(expected):
        raise ValueError(f"Base file set differs: missing={sorted(set(expected) - recognized)}, extra={sorted(recognized - set(expected))}")
    actual = {name: _file_sha256(root / name) for name in sorted(expected)}
    for name, digest in actual.items():
        if digest != expected[name]:
            raise ValueError(f"Base file content differs: {name}")
    fingerprint = hashlib.sha256(json.dumps(actual, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    if fingerprint != contract.get("fingerprint"):
        raise ValueError("Base contract fingerprint does not match its file contents")
    return {"verified": True, "model_path": str(root), "fingerprint": fingerprint, "file_count": len(actual)}


def _inspect(payload):
    result = {key: payload.get(key) for key in ("schema_version", "space", "task", "seed")}
    if payload["space"] == "activation":
        result["values"] = {key: {"shape": list(value.shape), "dtype": str(value.dtype)} for key, value in payload["values"].items()}
    else:
        result["factors"] = {key: [{name: {"shape": list(term[name].shape), "dtype": str(term[name].dtype)} for name in ("A", "B")} for term in terms]
                             for key, terms in payload["factors"].items()}
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inspect a plain task-vector export without the experiment package")
    parser.add_argument("--inspect", required=True, metavar="POST_PT")
    args = parser.parse_args()
    print(json.dumps(_inspect(load_plain(args.inspect)), ensure_ascii=False, indent=2))
