"""First-order bidirectional GAO with sequential AdamW optimizer updates.

The official negative rho is sampled once per logical batch. Each source
gradient is normalized per parameter tensor, independently of outer clipping.
"""
from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

import torch


def _backward_group(loss_fn: Callable, group) -> float:
    value = loss_fn(group)
    if isinstance(value, torch.Tensor):
        if value.ndim != 0 or not value.requires_grad:
            raise ValueError("loss_fn must return a differentiable scalar, or float after backward")
        value.backward()
        return float(value.detach())
    return float(value)


def _group_ids(group):
    result = []
    for item in group:
        identifier = item.get("id") if isinstance(item, dict) else getattr(item, "id", None)
        if identifier is not None:
            result.append(str(identifier))
    return result


def gao_step(parameters, optimizer, group1, group2, loss_fn,
             rho_max: float = 0.05, clip_norm: float = 1.0, scheduler=None,
             generator=None) -> dict[str, Any]:
    """Perform source1->target2, then source2->target1 at the updated weights.

    loss_fn(group) may return a scalar tensor (this function calls backward),
    or a numeric loss after performing micro-batch backward accumulation. The
    latter frees each micro-batch graph immediately and gives the exact group
    mean gradient. No source graph or second derivatives are retained.

    Following official code, optimizer.step runs with perturbations applied,
    and perturbations are subtracted afterwards in a finally block. This also
    preserves the source implementation's AdamW behavior when wd is nonzero.
    """
    params = [p for p in parameters if p.requires_grad]
    if not params or rho_max < 0 or clip_norm <= 0 or len(group1) == 0 or len(group2) == 0:
        raise ValueError("GAO requires trainable parameters, nonempty groups, and valid rho/clip")
    if len({id(p) for p in params}) != len(params):
        raise ValueError("Duplicate parameters would perturb and update tensors twice")
    ids1, ids2 = _group_ids(group1), _group_ids(group2)
    if len(ids1) != len(set(ids1)) or len(ids2) != len(set(ids2)) or set(ids1) & set(ids2):
        raise ValueError("GAO groups must contain disjoint unique sample IDs")
    # A CPU generator has a portable saved RNG state across CPU and Ascend.
    draw_device = getattr(generator, "device", torch.device("cpu")) if generator is not None else "cpu"
    rho = -float(rho_max) * float(torch.rand((), generator=generator, device=draw_device))
    directions = []
    for source, target in ((group1, group2), (group2, group1)):
        optimizer.zero_grad(set_to_none=True)
        source_loss = _backward_group(loss_fn, source)
        if any(p.grad is None for p in params):
            raise RuntimeError("GAO source loss did not produce every trainable tensor gradient")
        if any(not bool(torch.isfinite(p.grad).all()) for p in params):
            optimizer.zero_grad(set_to_none=True)
            raise FloatingPointError("Non-finite GAO source gradient")
        perturbations = [rho * p.grad.detach() / (p.grad.detach().norm() + 1e-12) for p in params]
        source_norms = [float(p.grad.detach().norm()) for p in params]
        applied = []
        try:
            with torch.no_grad():
                for p, perturbation in zip(params, perturbations):
                    p.add_(perturbation)
                    applied.append((p, perturbation))
            optimizer.zero_grad(set_to_none=True)
            target_loss = _backward_group(loss_fn, target)
            if any(p.grad is None for p in params):
                raise RuntimeError("GAO target loss did not produce every trainable tensor gradient")
            target_norm = torch.nn.utils.clip_grad_norm_(params, float(clip_norm),
                                                        error_if_nonfinite=True, foreach=False)
            optimizer.step()
        finally:
            with torch.no_grad():
                for p, perturbation in reversed(applied):
                    p.sub_(perturbation)
        if scheduler is not None:
            scheduler.step() if hasattr(scheduler, "step") else scheduler()
        directions.append({"source_loss": source_loss, "target_loss": target_loss,
                           "source_tensor_norms": source_norms,
                           "target_norm_before_clip": float(target_norm)})
    optimizer.zero_grad(set_to_none=True)
    return {"rho": rho, "optimizer_steps": 2, "scheduler_steps": 2 if scheduler is not None else 0,
            "loss": (directions[0]["target_loss"] + directions[1]["target_loss"]) / 2,
            "directions": directions, "group1_ids": ids1, "group2_ids": ids2}
