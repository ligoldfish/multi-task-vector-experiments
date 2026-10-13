"""LoDA-MT input statistics and official generalized-energy subspaces.

Only spectral factorization runs on CPU (float64); model forwards and statistic
matrix products run on the selected accelerator. S is a token second moment,
not a centered covariance or an average of unequal per-example covariances.
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch

from .io import atomic_torch_save, object_sha256


class Statistics(dict):
    """Layer matrices with auditable calibration metadata outside the layer keys."""

    def __init__(self, matrices=(), *, metadata=None):
        super().__init__(matrices)
        self.metadata = dict(metadata or {})


def save_statistics(stats: Mapping[str, torch.Tensor], path: str | Path):
    atomic_torch_save(path, {"schema_version": 1,
                             "matrices": {k: v.detach().cpu() for k, v in stats.items()},
                             "metadata": getattr(stats, "metadata", {})})


def load_statistics(path: str | Path) -> Statistics:
    value = torch.load(path, map_location="cpu", weights_only=True)
    if value.get("schema_version") != 1:
        raise ValueError("Unknown statistics schema")
    return Statistics(value["matrices"], metadata=value["metadata"])


def uniform_token_positions(length: int, maximum: int = 64) -> torch.Tensor:
    """Deterministic, unique, uniformly spaced positions including both ends."""
    if length <= 0 or maximum <= 0:
        raise ValueError("Prompt length and maximum token count must be positive")
    if length <= maximum:
        return torch.arange(length, dtype=torch.long)
    if maximum == 1:
        return torch.tensor([0], dtype=torch.long)
    return torch.linspace(0, length - 1, maximum, dtype=torch.float64).round().long()


def _qkv_layers(model):
    result = {}
    for name, module in model.named_modules():
        suffix = ".self_attn.q_proj"
        if name.endswith(suffix):
            layer = name[:-len(suffix)]
            for kind in ("q_proj", "k_proj", "v_proj"):
                projection = model.get_submodule(f"{layer}.self_attn.{kind}")
                if not hasattr(projection, "weight") or projection.weight.ndim != 2:
                    raise TypeError(f"Expected native linear QKV projection: {layer}.{kind}")
                if projection.weight.shape[1] != module.weight.shape[1]:
                    raise ValueError(f"QKV input dimensions differ at {layer}")
            result[layer] = module
    if not result:
        raise ValueError("No model.layers.N.self_attn.q_proj modules found")
    return result


def collect_statistics(model, tokenizer, samples: Sequence, config: dict, device) -> Statistics:
    """Sample training prompts; hook the native Q input after input RMSNorm.

    The Q/K/V projections receive the same input in Qwen3. Hooking Q once avoids
    tripling the token count. The full prompt is forwarded, while at most 64
    non-padding token features per prompt contribute to S.
    """
    from .prompts import render_prompt

    cfg = config.get("loda", {})
    count = int(cfg.get("statistics_samples", 128))
    maximum = int(cfg.get("max_statistics_tokens", 64))
    seed = int(cfg.get("statistics_seed", 17))
    if count <= 0 or maximum <= 0 or len(samples) < count:
        raise ValueError(f"Statistics require {count} training samples; got {len(samples)}")
    ids = [str(s.id if hasattr(s, "id") else s["id"]) for s in samples]
    if len(set(ids)) != len(ids):
        raise ValueError("Calibration candidates contain duplicate sample IDs")
    for sample in samples:
        split = getattr(sample, "official_split", None)
        split = sample.get("official_split") if isinstance(sample, dict) else split
        if split and split != "train":
            raise ValueError("LoDA statistics must use official training data only")
    chosen = torch.randperm(len(samples), generator=torch.Generator().manual_seed(seed))[:count].tolist()
    layers = _qkv_layers(model)
    matrices = {name: torch.zeros((module.weight.shape[1], module.weight.shape[1]),
                                 device=device, dtype=torch.float32)
                for name, module in layers.items()}
    token_counts = dict.fromkeys(layers, 0)
    hooks, selected_positions, selected_ids, prompt_hashes = [], [], [], []
    active_positions = None
    invocations = {}

    def hook_for(layer):
        def accumulate(module, args):
            nonlocal active_positions
            if active_positions is None:
                raise RuntimeError("Statistics hook invoked outside prompt collection")
            x = args[0]
            if x.ndim != 3 or x.shape[0] != 1:
                raise ValueError("Statistics collector forwards one complete prompt at a time")
            positions = active_positions.to(x.device)
            selected = x[0].index_select(0, positions).detach().float()
            # Accumulation is on the model accelerator, never BF16 or float64.
            matrices[layer].addmm_(selected.T, selected)
            token_counts[layer] += selected.shape[0]
            invocations[layer] = invocations.get(layer, 0) + 1
        return accumulate

    training = model.training
    try:
        model.eval()
        for layer, module in layers.items():
            hooks.append(module.register_forward_pre_hook(hook_for(layer)))
        with torch.no_grad():
            for index in chosen:
                sample = samples[index]
                prompt = render_prompt(tokenizer, sample, variant="canonical")
                encoded = tokenizer(prompt, add_special_tokens=False, return_tensors="pt")
                input_ids = encoded["input_ids"].to(device)
                mask = encoded.get("attention_mask", torch.ones_like(encoded["input_ids"])).to(device)
                if input_ids.shape[0] != 1 or not bool(mask.bool().all()):
                    raise ValueError("A single unpadded prompt is required for statistics")
                max_prompt = int(config.get("training", {}).get("max_prompt_tokens", 512))
                if input_ids.shape[1] > max_prompt:
                    raise ValueError("Calibration prompt exceeds the locked prompt token limit")
                active_positions = uniform_token_positions(input_ids.shape[1], maximum)
                selected_positions.append(active_positions.tolist())
                selected_ids.append(ids[index])
                prompt_hashes.append(object_sha256({"input_ids": input_ids.cpu().tolist()}))
                model(input_ids=input_ids, attention_mask=mask, use_cache=False)
    finally:
        active_positions = None
        for handle in hooks:
            handle.remove()
        model.train(training)
    if any(invocations.get(layer, 0) != count for layer in layers):
        raise RuntimeError("Each Q projection must be invoked exactly once per calibration prompt")
    result = {layer: (matrix / token_counts[layer]).cpu() for layer, matrix in matrices.items()}
    metadata = {"statistic": "token_second_moment_mean", "centered": False,
                "statistics_seed": seed, "sample_ids": selected_ids,
                "sample_count": count, "token_positions": selected_positions,
                "prompt_token_hashes": prompt_hashes, "token_counts": token_counts,
                "feature_location": "native Q projection input, after block input RMSNorm",
                "qkv_shared_input": True, "accumulation_dtype": "float32",
                "source_split": "train"}
    return Statistics(result, metadata=metadata)


def _symmetric_cpu(matrix: torch.Tensor) -> torch.Tensor:
    matrix = matrix.detach().to(device="cpu", dtype=torch.float64)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("Second moments must be square matrices")
    if not bool(torch.isfinite(matrix).all()):
        raise ValueError("Non-finite second moment")
    return (matrix + matrix.T) * 0.5


def top_energy_basis(matrix: torch.Tensor, rank: int) -> tuple[torch.Tensor, torch.Tensor]:
    matrix = _symmetric_cpu(matrix)
    if not 0 < rank <= matrix.shape[0]:
        raise ValueError("Subspace rank must be within the input dimension")
    eigenvalues, vectors = torch.linalg.eigh(matrix)
    return vectors[:, -rank:].flip(1).contiguous(), eigenvalues[-rank:].flip(0).contiguous()


def isolated_energy_basis(self_matrix: torch.Tensor, other_matrix: torch.Tensor,
                          rank: int, regularization: float = 1e-4):
    """Official whitening -> top eigenvectors -> map back -> Euclidean QR."""
    current, other = _symmetric_cpu(self_matrix), _symmetric_cpu(other_matrix)
    if current.shape != other.shape or not 0 < rank <= current.shape[0]:
        raise ValueError("Invalid isolated subspace shapes or rank")
    d = current.shape[0]
    alpha = float(regularization) * float(torch.trace(other)) / d
    if alpha <= 0:
        raise ValueError("Other-task statistics have zero energy; isolated subspace is undefined")
    regularized = other + alpha * torch.eye(d, dtype=torch.float64)
    # With the stipulated trace regularizer PSD statistics are SPD. An explicit,
    # logged numerical repair handles finite precision rather than changing r.
    jitter, factor = 0.0, None
    for attempt in range(8):
        factor, info = torch.linalg.cholesky_ex(regularized + jitter * torch.eye(d, dtype=torch.float64))
        if int(info) == 0:
            break
        jitter = max(alpha, float(torch.trace(other)) / d, 1.0) * 1e-12 * 10 ** attempt
    else:
        raise RuntimeError("Other-task second moment remained non-SPD after numerical repair")
    left = torch.linalg.solve_triangular(factor, current, upper=False)
    whitened = torch.linalg.solve_triangular(factor, left.T, upper=False).T
    whitened = (whitened + whitened.T) * 0.5
    eigenvalues, eigenvectors = torch.linalg.eigh(whitened)
    top = eigenvectors[:, -rank:].flip(1)
    mapped = torch.linalg.solve_triangular(factor.T, top, upper=True)
    basis = torch.linalg.qr(mapped, mode="reduced")[0]
    meta = {"regularization_formula": "1e-4*trace(S_other)/d",
            "regularization_coefficient": float(regularization), "regularization_value": alpha,
            "numerical_jitter": jitter, "orthogonality_error": float((basis.T @ basis - torch.eye(rank)).norm()),
            "whitened_eigenvalues": eigenvalues[-rank:].flip(0).tolist()}
    return basis.contiguous(), meta


def recalibration_diagonal(A: torch.Tensor, self_matrix: torch.Tensor,
                           other_matrix: torch.Tensor, gamma: float = 3.0,
                           epsilon: float = 1e-12) -> torch.Tensor:
    """Lambda_i = 1 - e_other/(e_other + gamma*e_self + eps)."""
    a = A.detach().to(device="cpu", dtype=torch.float64)
    current, other = _symmetric_cpu(self_matrix), _symmetric_cpu(other_matrix)
    if a.ndim != 2 or current.shape != other.shape or a.shape[1] != current.shape[0]:
        raise ValueError("Recalibration dimensions differ")
    if gamma < 0 or epsilon <= 0:
        raise ValueError("Invalid recalibration gamma/epsilon")
    energy_self = ((a @ current) * a).sum(dim=1).clamp_min(0)
    energy_other = ((a @ other) * a).sum(dim=1).clamp_min(0)
    eta = energy_other / (energy_other + float(gamma) * energy_self + float(epsilon))
    return (1.0 - eta).float()


def build_subspaces(stats_by_task: Mapping[str, Mapping[str, torch.Tensor]], task: str,
                    config: dict, *, seen_tasks: Sequence[str] | None = None):
    cfg = config.get("loda", {})
    active = tuple(seen_tasks or cfg.get("seen_tasks") or stats_by_task.keys())
    if task not in active or not active or len(set(active)) != len(active):
        raise ValueError("Task must belong to the nonempty, unique seen-task statistics set")
    if any(name not in stats_by_task for name in active):
        raise ValueError("Missing statistics for a seen task")
    layers = tuple(sorted(stats_by_task[task]))
    if any(set(stats_by_task[name]) != set(layers) for name in active):
        raise ValueError("Task statistics must share exactly the same layers")
    rank_g, rank_i = int(cfg.get("rank_general", 8)), int(cfg.get("rank_isolated", 8))
    scale = float(cfg.get("a_scale", 1.0 / math.sqrt(3.0)))
    if scale <= 0:
        raise ValueError("A scaling must be positive")
    result = {}
    for layer in layers:
        current = _symmetric_cpu(stats_by_task[task][layer])
        others = sum((_symmetric_cpu(stats_by_task[t][layer]) for t in active if t != task),
                     torch.zeros_like(current))
        general, eigenvalues = top_energy_basis(current + others, rank_g)
        if len(active) == 1 and not cfg.get("isolated_enabled", True):
            isolated = torch.zeros((current.shape[0], rank_i), dtype=torch.float64)
            isolated_meta = {"disabled": True, "reason": "first sequential stage"}
        else:
            isolated, isolated_meta = isolated_energy_basis(current, others, rank_i,
                                                            float(cfg.get("regularization", 1e-4)))
        a_general, a_isolated = (general.T * scale).float(), (isolated.T * scale).float()
        diagonal = recalibration_diagonal(a_general, current, others,
                                           float(cfg.get("recalibration_lambda", 3)),
                                           float(cfg.get("epsilon", 1e-12)))
        result[layer] = {"A_G": a_general, "A_I": a_isolated,
                         "S_self": current.float(), "S_other": others.float(),
                         "lambda": diagonal,
                         "metadata": {"route": "LoDA-MT", "task": task,
                                      "seen_tasks": list(active), "statistics_source": "fixed_base",
                                      "recalibration_lambda": float(cfg.get("recalibration_lambda", 3)),
                                      "epsilon": float(cfg.get("epsilon", 1e-12)),
                                      "a_scale": scale, "general_eigenvalues": eigenvalues.tolist(),
                                      "isolated": isolated_meta,
                                      "calibration": {t: getattr(stats_by_task[t], "metadata", {}) for t in active}}}
    return result
