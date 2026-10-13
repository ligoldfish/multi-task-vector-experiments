"""Geometry and functional inference in the represented, rather than factor, space."""
from __future__ import annotations
from dataclasses import dataclass
from itertools import combinations
from typing import Callable
import math
import numpy as np
import torch
from scipy.cluster.hierarchy import linkage, cut_tree
from scipy.spatial.distance import pdist
from scipy.stats import spearmanr
from .representations import Representation, compatible, inner, norm, linear_combination, random_like, match_norm


def gram_matrix(representations):
    n = len(representations)
    result = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        for j in range(i + 1):
            result[i, j] = result[j, i] = inner(representations[i], representations[j])
    return result


def cosine_matrix(representations):
    gram = gram_matrix(representations)
    lengths = np.sqrt(np.maximum(np.diag(gram), 0))
    denominator = np.outer(lengths, lengths)
    result = np.full_like(gram, np.nan)
    np.divide(gram, denominator, out=result, where=denominator > 1e-24)
    return result


@dataclass
class Decomposition:
    shared: list[Representation]
    residual: list[Representation]
    weights: np.ndarray
    metadata: dict

    def compose(self, indices=None):
        indices = list(range(len(self.shared))) if indices is None else list(indices)
        n = len(indices)
        if not n:
            raise ValueError("Empty combination")
        return linear_combination([self.shared[i] for i in indices] + [self.residual[i] for i in indices],
                                  [1 / n] * n + [1.] * n, "decomposed_composition")


def _project_weights(basis, targets, q, normalize_basis=True):
    if q < 0 or q > len(basis):
        raise ValueError("Invalid task-space rank")
    G = gram_matrix(basis)
    scales = np.sqrt(np.maximum(np.diag(G), 0)) if normalize_basis else np.ones(len(basis))
    inv = np.zeros_like(scales)
    np.divide(1., scales, out=inv, where=scales > 1e-12)
    Gn = G * np.outer(inv, inv)
    eig, R = np.linalg.eigh(Gn)
    order = np.argsort(eig)[::-1]
    eig, R = eig[order], R[:, order]
    tol = max(1., float(eig.max(initial=0))) * 1e-10
    effective = int(np.sum(eig > tol))
    used = min(q, effective)
    K = np.array([[inner(x, y) for y in targets] for x in basis])
    if used == 0:
        weights = np.zeros_like(K)
    else:
        U = R[:, :used]
        weights = inv[:, None] * ((U / eig[:used]) @ U.T) @ (inv[:, None] * K)
    return weights, {"requested_rank": q, "effective_rank": effective, "used_rank": used,
                     "singular_values": np.sqrt(np.maximum(eig, 0)).tolist(),
                     "zero_source_tasks": [basis[i].task for i in range(len(scales)) if scales[i] <= 1e-12],
                     "coordinate_space": basis[0].space, "factor_flattening": False}


def decompose(representations, method="svd", q=1):
    if not representations:
        raise ValueError("No source representations")
    for rep in representations[1:]:
        compatible(representations[0], rep)
    n = len(representations)
    if method == "mean":
        weights = np.full((n, n), 1 / n)
        meta = {"method": "mean", "same_shared_for_all": True,
                "full_group_composition_degenerates_to_mean": True}
    elif method == "svd":
        weights, meta = _project_weights(representations, representations, q)
        meta.update(method="svd", normalized_columns=True,
                    full_group_composition_degenerates_to_mean=(meta["used_rank"] == meta["effective_rank"]),
                    q_zero_composition_degenerates_to_sum=(q == 0))
    else:
        raise ValueError(method)
    shared = [linear_combination(representations, weights[:, i], f"shared:{rep.task}") for i, rep in enumerate(representations)]
    residual = [linear_combination([rep, shared[i]], [1, -1], f"residual:{rep.task}") for i, rep in enumerate(representations)]
    meta["delete_shared_equals_residual"] = True
    meta["reconstruction_relative_errors"] = [norm(linear_combination([s, r, v], [1, 1, -1])) / max(norm(v), 1e-12)
                                              for s, r, v in zip(shared, residual, representations)]
    return Decomposition(shared, residual, weights, meta)


def random_subspace_control(representations, decomposition, q, seed):
    """Random q-dimensional task-space subspace with norm-matched projections.

    Parameter nulls retain each actual update's matrix rank/spectrum, while
    randomizing both input and output singular subspaces independently.
    """
    basis = [random_like(rep, seed + i * 1009) for i, rep in enumerate(representations)]
    weights, meta = _project_weights(basis, representations, q)
    result = []
    for i, source in enumerate(representations):
        raw = linear_combination(basis, weights[:, i], f"random_shared:{source.task}")
        wanted = norm(decomposition.shared[i])
        if norm(raw) <= 1e-12 and wanted > 1e-12:
            # A numerically vanishing random projection is an identified failure.
            raise ValueError("Random subspace projection has zero norm; cannot match the control")
        matched = match_norm(raw, decomposition.shared[i])
        matched.metadata.update(meta, random_seed=seed, null="random_task_subspace", norm_matched=True)
        result.append(matched)
    return result


def group_residual_decomposition(representations, global_decomposition, groups, q):
    if len(representations) < 6:
        raise ValueError("Group-specific SVD is reserved for the six-task stable-group stage")
    if sorted(i for group in groups for i in group) != list(range(len(representations))):
        raise ValueError("Stable groups must partition all source tasks")
    result = {}
    for index, indices in enumerate(groups):
        result[index] = decompose([global_decomposition.residual[i] for i in indices], "svd", min(q, len(indices)))
    return result


def _matrix_svd(factors):
    B = torch.cat([x.B.double().cpu() for x in factors], dim=1)
    At = torch.cat([x.A.double().cpu() for x in factors], dim=0).T
    Qb, Rb = torch.linalg.qr(B, mode="reduced")
    Qa, Ra = torch.linalg.qr(At, mode="reduced")
    U, S, Vh = torch.linalg.svd(Rb @ Ra.T, full_matrices=False)
    tol = (float(S.max()) if S.numel() else 0) * max(B.shape[0], At.shape[0]) * torch.finfo(torch.float32).eps
    rank = int((S > tol).sum())
    return Qb @ U[:, :rank], S[:rank], Qa @ Vh[:rank].T


def layer_subspace_angles(a, b):
    compatible(a, b)
    if a.space != "parameter":
        raise ValueError("Layer matrix subspace angles require actual parameter updates")
    result = {}
    for key in a.keys:
        Ua, Sa, Va = _matrix_svd(a.factors[key])
        Ub, Sb, Vb = _matrix_svd(b.factors[key])
        record = {"rank_a": len(Sa), "rank_b": len(Sb)}
        for name, X, Y in (("output", Ua, Ub), ("input", Va, Vb)):
            vals = torch.linalg.svdvals(X.T @ Y) if X.shape[1] and Y.shape[1] else torch.tensor([])
            record[f"{name}_angles_degrees"] = torch.rad2deg(torch.acos(vals.clamp(0, 1))).tolist()
        result[key] = record
    return result


def relation_consistency(matrix_a, matrix_b):
    if np.shape(matrix_a) != np.shape(matrix_b):
        raise ValueError("Different task sets")
    indices = np.triu_indices(len(matrix_a), k=1)
    a, b = np.asarray(matrix_a)[indices], np.asarray(matrix_b)[indices]
    valid = np.isfinite(a) & np.isfinite(b)
    if valid.sum() < 3 or np.std(a[valid]) < 1e-12 or np.std(b[valid]) < 1e-12:
        return {"spearman": None, "pairs": int(valid.sum()), "reason": "insufficient nonconstant pairs"}
    rho, pvalue = spearmanr(a[valid], b[valid])
    return {"spearman": float(rho), "descriptive_pvalue": float(pvalue), "pairs": int(valid.sum()),
            "interpretation": "exploratory relation consistency; task pairs are not independent samples"}


# Keep the public reference API; CPU bootstrap workers need no torch import.
from .bootstrap import _joint_group_ids, _paired_clusters, paired_bootstrap


def transfer_matrix(tasks, scores, base_scores, base_sample_scores):
    raw = np.array([[scores[source][target] - base_scores[target] for target in tasks] for source in tasks], dtype=float)
    std = np.array([np.std(base_sample_scores[t], ddof=1) if len(base_sample_scores[t]) > 1 else 0. for t in tasks])
    normalized = np.zeros_like(raw)
    np.divide(raw, std[None, :], out=normalized, where=std[None, :] > 1e-12)
    return {"tasks": tasks, "raw": raw.tolist(), "column_std": std.tolist(),
            "standardized": normalized.tolist(),
            "zero_std_columns": [tasks[i] for i in np.flatnonzero(std <= 1e-12)],
            "direction": "rows=source intervention; columns=target task",
            "zero_std_policy": "exclude from clustering distance, retain raw effect"}


def confirm_transfer_relations(discovery, final, *, tolerance=1e-12):
    """Describe frozen directed relations on confirmation data, without selection.

    Inputs are transfer_matrix records (or core records containing ``matrix``).
    Task names are aligned before comparison. With baseline standard deviations,
    discovery column scales are reused for both panels; raw effects stay in each
    target task's own unit. No heterogeneous raw-score average or test re-clustering
    is performed, and no significance threshold is inferred from correlations.
    """
    if not math.isfinite(tolerance) or tolerance < 0:
        raise ValueError("Confirmation tolerance must be finite and nonnegative")
    discovery, final = discovery.get("matrix", discovery), final.get("matrix", final)
    tasks, final_tasks = list(discovery["tasks"]), list(final["tasks"])
    if not tasks or len(set(tasks)) != len(tasks) or len(set(final_tasks)) != len(final_tasks) or set(tasks) != set(final_tasks):
        raise ValueError("Confirmation requires identical unique source/target task sets")
    n = len(tasks)
    order = [final_tasks.index(task) for task in tasks]
    def matrix(record, key, reorder=False):
        value = np.asarray(record[key], dtype=float)
        if value.shape != (n, n) or not np.isfinite(value).all():
            raise ValueError(f"Invalid finite {key} transfer matrix")
        return value[np.ix_(order, order)] if reorder else value
    raw_a, raw_b = matrix(discovery, "raw"), matrix(final, "raw", True)
    local_a, local_b = matrix(discovery, "standardized"), matrix(final, "standardized", True)
    common_scale = "column_std" in discovery and "column_std" in final
    if common_scale:
        std_a = np.asarray(discovery["column_std"], dtype=float)
        std_b = np.asarray(final["column_std"], dtype=float)
        if std_a.shape != (n,) or std_b.shape != (n,) or not np.isfinite(std_a).all() or not np.isfinite(std_b).all() or np.any(std_a < 0) or np.any(std_b < 0):
            raise ValueError("Invalid baseline sample standard deviations")
        std_b = std_b[order]
        columns = (std_a > tolerance) & (std_b > tolerance)
        scaled_a, scaled_b = np.zeros_like(raw_a), np.zeros_like(raw_b)
        np.divide(raw_a, std_a[None, :], out=scaled_a, where=std_a[None, :] > tolerance)
        np.divide(raw_b, std_a[None, :], out=scaled_b, where=std_a[None, :] > tolerance)
        scale = "discovery_BASE_sentence_std_reused_for_both_panels"
    else:
        columns = np.ones(n, dtype=bool)
        scaled_a, scaled_b = local_a, local_b
        scale = "provided_per_panel_standardized_matrices; baseline_std_unavailable"
    sign_a = np.where(np.abs(raw_a) <= tolerance, 0, np.sign(raw_a)).astype(int)
    sign_b = np.where(np.abs(raw_b) <= tolerance, 0, np.sign(raw_b)).astype(int)
    def describe(mask):
        count = int(mask.sum())
        valid = mask & np.broadcast_to(columns, (n, n))
        a, b = scaled_a[valid], scaled_b[valid]
        rho = pvalue = None
        reason = None
        if len(a) < 3 or float(np.std(a)) <= tolerance or float(np.std(b)) <= tolerance:
            reason = "insufficient_nonconstant_standardized_relations"
        else:
            rho, pvalue = map(float, spearmanr(a, b))
        return {"relations": count, "sign_agreement": float(np.mean(sign_a[mask] == sign_b[mask])) if count else None,
                "sign_changes": int(np.sum(sign_a[mask] != sign_b[mask])),
                "standardized_spearman": rho, "descriptive_pvalue": pvalue,
                "standardized_relations": len(a), "correlation_undefined_reason": reason}
    edges = [{"source": source, "target": target, "diagonal": i == j,
              "discovery_raw": float(raw_a[i, j]), "confirmation_raw": float(raw_b[i, j]),
              "raw_change": float(raw_b[i, j] - raw_a[i, j]),
              "discovery_sign": int(sign_a[i, j]), "confirmation_sign": int(sign_b[i, j]),
              "direction_preserved": bool(sign_a[i, j] == sign_b[i, j]),
              "discovery_standardized": float(local_a[i, j]), "confirmation_standardized": float(local_b[i, j]),
              "discovery_at_comparison_scale": float(scaled_a[i, j]) if columns[j] else None,
              "confirmation_at_comparison_scale": float(scaled_b[i, j]) if columns[j] else None,
              "confirmation_at_discovery_scale": float(scaled_b[i, j]) if common_scale and columns[j] else None}
             for i, source in enumerate(tasks) for j, target in enumerate(tasks)]
    return {"tasks": tasks, "direction": "rows=source intervention; columns=target task",
            "all_relations": describe(np.ones((n, n), dtype=bool)),
            "off_diagonal": describe(~np.eye(n, dtype=bool)), "edges": edges,
            "standardization": scale, "excluded_zero_std_targets": [task for task, valid in zip(tasks, columns) if not valid],
            "raw_units": "retained_separately_per_target; never_averaged", "sign_tolerance": tolerance,
            "selection_performed": False, "interpretation": "descriptive_confirmation_of_frozen_directed_discovery_relations; task edges are dependent, p-values are descriptive"}


def discover_groups(matrix, *, bootstrap_matrices=None, min_stability=.8, max_groups=3):
    """Simple average-linkage clustering of outgoing functional profiles.

    Bootstrap coassignment stability chooses a partition; if no nontrivial
    partition is stable, return one group without interpreting it as evidence.
    """
    X = np.asarray(matrix["standardized"], dtype=float)
    valid = np.asarray(matrix["column_std"]) > 1e-12
    tasks = matrix["tasks"]
    if valid.sum() == 0 or len(tasks) < 3 or np.allclose(X[:, valid], X[0, valid]):
        return {"groups": [tasks], "stable": False, "reason": "no identifiable nonconstant functional profiles", "candidates": []}
    Z = linkage(pdist(X[:, valid], metric="euclidean"), method="average")
    candidates = []
    for k in range(2, min(max_groups, len(tasks) - 1) + 1):
        labels = cut_tree(Z, n_clusters=[k]).ravel()
        pairmask = labels[:, None] == labels[None, :]
        np.fill_diagonal(pairmask, False)
        evidence = []
        for bootstrap in bootstrap_matrices or []:
            Y = np.asarray(bootstrap, dtype=float)[:, valid]
            blabel = cut_tree(linkage(pdist(Y), method="average"), n_clusters=[k]).ravel()
            evidence.append(blabel[:, None] == blabel[None, :])
        if evidence and pairmask.any():
            consensus = np.mean(evidence, axis=0)
            within = float(np.min(consensus[pairmask]))
            between = float(np.min(1 - consensus[~pairmask & ~np.eye(len(tasks), dtype=bool)]))
            stability = min(within, between)
        else:
            stability = 0.
            consensus = None
        candidates.append({"k": k, "groups": [[tasks[i] for i in range(len(tasks)) if labels[i] == label] for label in sorted(set(labels))],
                           "stability": stability, "coassignment": None if consensus is None else consensus.tolist()})
    stable = [x for x in candidates if x["stability"] >= min_stability]
    chosen = max(stable, key=lambda x: (x["stability"], -x["k"])) if stable else None
    return {"groups": chosen["groups"] if chosen else [tasks], "stable": chosen is not None,
            "reason": "bootstrap-stable partition" if chosen else "no stable nontrivial partition", "candidates": candidates,
            "linkage": Z.tolist(), "feature": "column-standardized outgoing functional transfer"}


def choose_discovery_candidate(records, tasks, baseline_stds):
    """Select on source discovery only using standardized primary gains."""
    selected = []
    for record in records:
        terms = [record["gains"][t] / baseline_stds[t] for t in tasks if baseline_stds[t] > 1e-12]
        selected.append({**record, "selection_objective": float(np.mean(terms)) if terms else 0.,
                         "selection_tasks": list(tasks), "zero_std_tasks": [t for t in tasks if baseline_stds[t] <= 1e-12]})
    if not selected:
        raise ValueError("No discovery candidates")
    # Keep q=0/first grid candidate on ties, avoiding inflated complexity.
    return max(selected, key=lambda item: item["selection_objective"]), selected
