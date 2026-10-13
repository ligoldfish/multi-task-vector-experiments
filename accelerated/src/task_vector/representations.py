"""Exact representations in activation coordinates or actual parameter updates.

Low-rank factors are a storage format only. All geometry uses the represented
BA matrix, including cross terms; factor concatenation is never a feature vector.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence
import math
import torch


@dataclass
class LowRankFactor:
    B: torch.Tensor
    A: torch.Tensor

    def __post_init__(self):
        if self.B.ndim != 2 or self.A.ndim != 2 or self.B.shape[1] != self.A.shape[0]:
            raise ValueError("Expected B[out,rank] and A[rank,in]")

    def dense(self):
        return self.B.float() @ self.A.float()


@dataclass
class Representation:
    space: str
    task: str
    seed: int
    values: dict[str, torch.Tensor] = field(default_factory=dict)
    factors: dict[str, list[LowRankFactor]] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if self.space not in {"activation", "parameter"}:
            raise ValueError(self.space)
        if self.space == "activation" and (not self.values or self.factors):
            raise ValueError("Activation representation requires values only")
        if self.space == "parameter" and (not self.factors or self.values):
            raise ValueError("Parameter representation requires factors only")

    @property
    def keys(self):
        return tuple(sorted(self.values if self.space == "activation" else self.factors))

    def payload(self):
        return {"schema_version": 1, "space": self.space, "task": self.task,
                "seed": self.seed, "metadata": self.metadata,
                "values": {k: v.detach().float().cpu() for k, v in self.values.items()},
                "factors": {k: [{"B": f.B.detach().float().cpu(), "A": f.A.detach().float().cpu()}
                                for f in fs] for k, fs in self.factors.items()}}

    def save(self, path: str | Path):
        from .io import atomic_torch_save
        atomic_torch_save(path, self.payload())

    @classmethod
    def load(cls, path: str | Path):
        p = torch.load(path, map_location="cpu", weights_only=True)
        if p.get("schema_version") != 1:
            raise ValueError("Unknown representation schema")
        return cls(p["space"], p["task"], p["seed"], p["values"],
                   {k: [LowRankFactor(f["B"], f["A"]) for f in fs] for k, fs in p["factors"].items()},
                   p["metadata"])


def compatible(a: Representation, b: Representation):
    if a.space != b.space or a.keys != b.keys:
        raise ValueError("Different representation spaces or coordinate definitions")
    for k in a.keys:
        shape_a = a.values[k].shape if a.space == "activation" else (a.factors[k][0].B.shape[0], a.factors[k][0].A.shape[1])
        shape_b = b.values[k].shape if b.space == "activation" else (b.factors[k][0].B.shape[0], b.factors[k][0].A.shape[1])
        if shape_a != shape_b:
            raise ValueError(f"Coordinate shape differs at {k}")
    for name in ("model_id", "model_revision", "base_fingerprint"):
        if name in a.metadata and name in b.metadata and a.metadata[name] != b.metadata[name]:
            raise ValueError(f"Incompatible {name}")


def inner(a: Representation, b: Representation) -> float:
    compatible(a, b)
    total = 0.0
    for k in a.keys:
        if a.space == "activation":
            total += float(torch.sum(a.values[k].double().cpu() * b.values[k].double().cpu()))
        else:
            for x in a.factors[k]:
                for y in b.factors[k]:
                    bx, ax, by, ay = (t.detach().double().cpu() for t in (x.B, x.A, y.B, y.A))
                    total += float(torch.sum((bx.T @ by) * (ax @ ay.T)))
    return total


def norm(a: Representation) -> float:
    return math.sqrt(max(inner(a, a), 0.0))


def linear_combination(reps: Sequence[Representation], weights: Sequence[float], task="combined") -> Representation:
    if not reps or len(reps) != len(weights):
        raise ValueError("Representations and weights must be nonempty and aligned")
    for rep in reps[1:]:
        compatible(reps[0], rep)
    first = reps[0]
    meta = dict(first.metadata)
    meta.update({"source_tasks": [r.task for r in reps], "weights": list(map(float, weights))})
    if first.space == "activation":
        vals = {k: sum((r.values[k].detach().float().cpu() * float(w) for r, w in zip(reps, weights)), torch.zeros_like(first.values[k], device="cpu", dtype=torch.float32)) for k in first.keys}
        return Representation(first.space, task, first.seed, values=vals, metadata=meta)
    # Concatenate terms rather than truncating their represented matrix ranks.
    facts = {k: [LowRankFactor(f.B.detach().float().cpu() * float(w), f.A.detach().float().cpu())
                 for r, w in zip(reps, weights) for f in r.factors[k]] for k in first.keys}
    return Representation(first.space, task, first.seed, factors=facts, metadata=meta)


def scaled(rep: Representation, scale: float, task=None):
    return linear_combination([rep], [scale], task or rep.task)


def match_norm(rep: Representation, reference: Representation):
    n, wanted = norm(rep), norm(reference)
    if n <= 1e-12:
        if wanted > 1e-12:
            raise ValueError("Cannot norm-match a zero representation to a nonzero one")
        return scaled(rep, 0)
    return scaled(rep, wanted / n)


def random_like(rep: Representation, seed: int, per_module=True) -> Representation:
    """Isotropic activation null / random input-output subspace parameter null.

    Parameter null preserves each actual matrix's numerical rank and Frobenius
    norm, not the arbitrary norms of its factors. No trained A is reused.
    """
    g = torch.Generator(device="cpu").manual_seed(seed)
    if rep.space == "activation":
        result = Representation(rep.space, f"random:{rep.task}", seed,
                                values={k: torch.randn(v.shape, generator=g) for k, v in rep.values.items()},
                                metadata={**rep.metadata, "null": "isotropic", "null_seed": seed})
        return match_norm(result, rep)
    facts = {}
    for k, fs in rep.factors.items():
        # Exact small-core SVD avoids expanding out*in matrices.
        b = torch.cat([f.B.float().cpu() for f in fs], dim=1)
        a = torch.cat([f.A.float().cpu() for f in fs], dim=0)
        _, rb = torch.linalg.qr(b.double(), mode="reduced")
        _, ra = torch.linalg.qr(a.T.double(), mode="reduced")
        singular = torch.linalg.svdvals(rb @ ra.T)
        tol = max(b.shape[0], a.shape[1]) * torch.finfo(torch.float32).eps * (float(singular.max()) if singular.numel() else 0)
        rank = max(1, int((singular > tol).sum()))
        u = torch.linalg.qr(torch.randn((b.shape[0], rank), generator=g), mode="reduced")[0]
        v = torch.linalg.qr(torch.randn((a.shape[1], rank), generator=g), mode="reduced")[0]
        magnitudes = singular[:rank].float()
        facts[k] = [LowRankFactor(u * magnitudes.unsqueeze(0), v.T)]
    result = Representation(rep.space, f"random:{rep.task}", seed, factors=facts,
                            metadata={**rep.metadata, "null": "random_singular_subspaces", "null_seed": seed,
                                      "per_module_norm": per_module})
    return result if per_module else match_norm(result, rep)


def dense_modules(rep: Representation):
    if rep.space != "parameter":
        raise ValueError("Only parameter representations have matrices")
    for k in rep.keys:
        yield k, sum((f.dense() for f in rep.factors[k]))

