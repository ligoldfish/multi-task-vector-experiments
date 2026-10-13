"""Author-code merge baselines on actual coordinate updates.

TIES uses one global threshold and sign election over the full representation.
TSV-M is the author's layerwise orthogonalization variant. KnOTS-TIES aligns
actual updates by joint SVD before globally merging their sV coordinates.
Sources and pinned commits are in third_party/baseline_sources.json.
"""
from __future__ import annotations

from typing import Sequence

import torch

from .representations import LowRankFactor, Representation, compatible


SOURCE_COMMITS = {
    "ties": {"repository": "prateeky2806/ties-merging",
             "commit": "44e7891fc84f3de7e4caa52664cd864ca3715e91",
             "file": "src/utils/merge_utils.py", "functions": "topk_values_mask/resolve_sign/disjoint_merge"},
    "tsv": {"repository": "AntoAndGar/task_singular_vectors",
            "commit": "a6c97188f7aa0bb20753de4f2939be89dd64f1c3",
            "file": "src/utils/TSVM_utils.py", "functions": "compute_and_sum_svd_mem_reduction"},
    "knots": {"repository": "gstoica27/KnOTS",
              "commit": "baa35e85768b8841820d35edc3e41b25304248c5",
              "file": "task_merger.py;masking_ops.py", "functions": "SVDMerger.apply_svd/ties_masking/masked_merge"},
}


def _validate(representations: Sequence[Representation], *, parameter_only=False):
    if not representations:
        raise ValueError("At least one source representation is required")
    for rep in representations[1:]:
        compatible(representations[0], rep)
    if parameter_only and representations[0].space != "parameter":
        raise ValueError("TSV and KnOTS are matrix parameter-space baselines")


def _matrix(rep, key):
    return sum((factor.B.detach().float().cpu() @ factor.A.detach().float().cpu()
                for factor in rep.factors[key]))


def _dense_factor(matrix):
    """Exact full-rank storage, no unrequested compression to adapter rank."""
    matrix = matrix.detach().float().cpu()
    rows, columns = matrix.shape
    if columns <= rows:
        return LowRankFactor(matrix, torch.eye(columns, dtype=torch.float32))
    return LowRankFactor(torch.eye(rows, dtype=torch.float32), matrix)


def _metadata(representations, method, settings):
    return {**representations[0].metadata, "baseline": method,
            "source_tasks": [rep.task for rep in representations],
            "author_source": SOURCE_COMMITS[method.split("-")[0]],
            "settings": settings, "post_merge_rank_truncation": False,
            "selection_split": "must_be_locked_by_caller"}


def _trim_and_elect(vectors, keep_fraction):
    if vectors.ndim != 2 or not vectors.shape[0] or not vectors.shape[1]:
        raise ValueError("TIES requires a nonempty tasks-by-coordinates matrix")
    fraction = float(keep_fraction)
    if fraction > 1:
        fraction /= 100
    if not 0 < fraction <= 1:
        raise ValueError("keep_fraction must be in (0,1] or an equivalent percentage")
    if not bool(torch.isfinite(vectors).all()):
        raise ValueError("Non-finite merge coordinates")
    if fraction == 1:
        retained = torch.ones_like(vectors, dtype=torch.bool)
    else:
        # Author code uses inclusive >= kthvalue(d-floor(d*K)), not exact
        # torch.topk. Boundary entries and ties can exceed requested density.
        threshold_index = vectors.shape[1] - int(vectors.shape[1] * fraction)
        threshold = vectors.abs().kthvalue(threshold_index, dim=1, keepdim=True).values
        retained = vectors.abs() >= threshold
    pruned = vectors * retained
    signs = pruned.sum(dim=0).sign()
    global_majority = signs.sum().sign()
    signs = torch.where(signs == 0, global_majority, signs)
    mask = retained & torch.where(signs.unsqueeze(0) > 0, pruned > 0, pruned < 0)
    return mask, {"requested_keep_fraction": fraction,
                  "realized_keep_fraction": retained.float().mean(dim=1).tolist(),
                  "retained_sign_fraction": mask.float().mean(dim=1).tolist(),
                  "threshold_rule": "inclusive kthvalue(d-floor(d*K))",
                  "zero_sign_resolution": "global majority of elected coordinate signs"}


def ties_tensor(vectors: torch.Tensor, keep_fraction: float = .2, merge: str = "mean"):
    """Exact global mass-sign TIES with disjoint mean/sum aggregation."""
    if merge not in {"mean", "sum", "max"}:
        raise ValueError("Unsupported TIES disjoint aggregation")
    mask, metadata = _trim_and_elect(vectors, keep_fraction)
    selected = vectors * mask
    if merge == "mean":
        merged = selected.sum(dim=0) / (selected != 0).sum(dim=0).clamp_min(1)
    elif merge == "sum":
        merged = selected.sum(dim=0)
    else:
        index = selected.abs().argmax(dim=0)
        merged = selected[index, torch.arange(selected.shape[1], device=selected.device)]
    return merged, {**metadata, "aggregation": f"disjoint-{merge}"}


def ties_merge(representations: Sequence[Representation], keep_fraction: float = .2,
               merge: str = "mean") -> Representation:
    _validate(representations)
    first = representations[0]
    if first.space == "activation":
        shapes = {key: first.values[key].shape for key in first.keys}
        rows = [torch.cat([rep.values[key].detach().float().cpu().reshape(-1) for key in first.keys])
                for rep in representations]
    else:
        shapes = {key: (first.factors[key][0].B.shape[0], first.factors[key][0].A.shape[1]) for key in first.keys}
        rows = [torch.cat([_matrix(rep, key).reshape(-1) for key in first.keys]) for rep in representations]
    merged, settings = ties_tensor(torch.stack(rows), keep_fraction, merge)
    offset, coordinates = 0, {}
    for key in first.keys:
        size = shapes[key].numel() if isinstance(shapes[key], torch.Size) else shapes[key][0] * shapes[key][1]
        coordinates[key] = merged[offset:offset + size].reshape(shapes[key]).clone()
        offset += size
    metadata = _metadata(representations, "ties", settings)
    if first.space == "activation":
        metadata["adaptation"] = "coordinate TIES on learned activation vectors"
        return Representation("activation", "TIES", first.seed, values=coordinates, metadata=metadata)
    return Representation("parameter", "TIES", first.seed,
                          factors={key: [_dense_factor(value)] for key, value in coordinates.items()}, metadata=metadata)


def tsv_merge(representations: Sequence[Representation]) -> Representation:
    """TSV-M author orthogonalization, retain floor(min(shape)/tasks) per task.

    This rank rule uses matrix dimensions, even for already low-rank updates.
    Null singular directions are not silently removed: doing so would alter
    the author's polar orthogonalization and create a different baseline.
    """
    _validate(representations, parameter_only=True)
    first, count = representations[0], len(representations)
    factors, sizes = {}, {}
    for key in first.keys:
        matrices = [_matrix(rep, key).double() for rep in representations]
        m, n = matrices[0].shape
        dimension = min(m, n)
        retained = dimension // count
        if retained == 0:
            raise ValueError("TSV-M's stipulated dimension/task rank is zero for this module")
        sum_u, sum_s, sum_v = torch.zeros(m, dimension, dtype=torch.float64), torch.zeros(dimension, dtype=torch.float64), torch.zeros(dimension, n, dtype=torch.float64)
        for index, matrix in enumerate(matrices):
            u, s, v = torch.linalg.svd(matrix, full_matrices=False)
            start, end = index * retained, (index + 1) * retained
            sum_u[:, start:end] = u[:, :retained]
            sum_s[start:end] = s[:retained]
            sum_v[start:end] = v[:retained]
        left_u, _, left_v = torch.linalg.svd(sum_u, full_matrices=False)
        right_u, _, right_v = torch.linalg.svd(sum_v, full_matrices=False)
        # U polar * diag(concatenated singular values) * V polar.
        left, right = left_u @ left_v, right_u @ right_v
        factors[key] = [LowRankFactor((left * sum_s.unsqueeze(0)).float(), right.float())]
        sizes[key] = {"matrix_dimension": dimension, "retained_per_task": retained,
                      "total_nonpadding_slots": retained * count}
    metadata = _metadata(representations, "tsv", {"variant": "TSV-M orthogonalization",
                                                  "spectral_device": "cpu", "spectral_dtype": "float64",
                                                  "rank_rule": "floor(min(out,in)/number_of_tasks)", "layers": sizes})
    return Representation("parameter", "TSV-M", first.seed, factors=factors, metadata=metadata)


def knots_merge(representations: Sequence[Representation], keep_fraction: float = .2,
                merge: str = "mean", concat_across_output: bool = True,
                singular_value_threshold: float = 1e-5) -> Representation:
    """KnOTS-TIES: common U, actual per-task sV, global masks, U reconstruction."""
    _validate(representations, parameter_only=True)
    if singular_value_threshold < 0:
        raise ValueError("KnOTS singular-value threshold must be nonnegative")
    first, count = representations[0], len(representations)
    bases, per_task, spectra, shapes = {}, [dict() for _ in representations], {}, {}
    for key in first.keys:
        matrices = [_matrix(rep, key).double() for rep in representations]
        if not concat_across_output:
            matrices = [matrix.T for matrix in matrices]
        joined = torch.cat(matrices, dim=1)
        u, s, v = torch.linalg.svd(joined, full_matrices=False)
        kept = s > float(singular_value_threshold)
        # Author code casts U, s, V to FP32 before constructing aligned sV.
        u, singular, v = u[:, kept].float(), s[kept].float(), v[kept].float()
        bases[key] = u
        parts = v.split(matrices[0].shape[1], dim=1)
        for index, part in enumerate(parts):
            per_task[index][key] = singular.unsqueeze(1) * part
        shapes[key] = (len(singular), matrices[0].shape[1])
        spectra[key] = {"retained_rank": int(kept.sum()), "singular_values": singular.tolist(),
                        "discarded_frobenius_norm": float(s[~kept].norm())}
    total_coordinates = sum(shape[0] * shape[1] for shape in shapes.values())
    if total_coordinates:
        aligned = torch.stack([torch.cat([row[key].reshape(-1) for key in first.keys]) for row in per_task])
        merged, settings = ties_tensor(aligned, keep_fraction, merge)
    else:
        merged, settings = torch.empty(0), {"all_updates_below_singular_threshold": True}
    offset, factors = 0, {}
    for key in first.keys:
        rows, columns = shapes[key]
        size = rows * columns
        coordinates = merged[offset:offset + size].reshape(rows, columns)
        offset += size
        if rows == 0:
            reference = first.factors[key][0]
            factors[key] = [LowRankFactor(torch.zeros(reference.B.shape[0], 1), torch.zeros(1, reference.A.shape[1]))]
        elif concat_across_output:
            factors[key] = [LowRankFactor(bases[key], coordinates)]
        else:
            factors[key] = [LowRankFactor(coordinates.T, bases[key].T)]
    settings.update({"variant": "KnOTS-TIES", "concat_across_output": bool(concat_across_output),
                     "joint_SVD": "horizontal concatenation of actual updates",
                     "singular_value_threshold": float(singular_value_threshold),
                     "global_sV_sign_election": True, "spectral_device": "cpu",
                     "spectral_dtype": "float64", "aligned_dtype": "float32", "layers": spectra})
    return Representation("parameter", "KnOTS-TIES", first.seed, factors=factors,
                          metadata=_metadata(representations, "knots", settings))


def merge_baseline(representations, method, **kwargs):
    methods = {"ties": ties_merge, "tsv": tsv_merge, "tsv-m": tsv_merge,
               "knots": knots_merge, "knots-ties": knots_merge}
    if method.lower() not in methods:
        raise ValueError(f"Unknown author baseline: {method}")
    return methods[method.lower()](representations, **kwargs)
