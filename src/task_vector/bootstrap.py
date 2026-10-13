"""CPU-only paired bootstrap with exact cached corpus sufficient statistics.

The reference bootstrap below is moved unchanged from analysis.py. The accelerated
path consumes the same original-group RNG draws and uses SacreBLEU 2.5.1's own
statistic extraction and corpus-score computation, never sentence-score means.
No NPU framework is imported in this module or its spawned CPU workers.
"""
from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
import multiprocessing
import os
import time
from typing import Callable

import numpy as np


def _joint_group_ids(row):
    """Task-qualified original groups, including readable legacy joint records."""
    metadata = row.get("metadata", {})
    explicit = metadata.get("subtask_group_ids")
    subtasks = metadata.get("subtasks")
    is_joint = row.get("task") == "joint" or explicit is not None or subtasks is not None
    if not is_joint:
        return None
    if explicit is None and subtasks is not None:
        if not isinstance(subtasks, list) or len(subtasks) != 2:
            raise ValueError("Joint bootstrap requires exactly two original subtask groups")
        explicit = {subtask["task"]: subtask["group_id"] for subtask in subtasks}
    if not isinstance(explicit, dict) or len(explicit) != 2 or not all(isinstance(task, str) and isinstance(group, str) and group for task, group in explicit.items()):
        raise ValueError("Joint bootstrap requires two task-qualified subtask_group_ids")
    return tuple(sorted(explicit.items()))


def _paired_clusters(base_rows, condition_rows):
    original = [_joint_group_ids(row) for row in base_rows]
    compared = [_joint_group_ids(row) for row in condition_rows]
    joint = any(item is not None for item in original + compared)
    if not joint:
        groups = {}
        for i, (base, condition) in enumerate(zip(base_rows, condition_rows)):
            group = str(base.get("group_id", base["id"]))
            if group != str(condition.get("group_id", condition["id"])):
                raise ValueError("Paired prediction group IDs differ")
            groups.setdefault(group, []).append(i)
        return list(groups.values()), {"bootstrap_unit": "group_id", "cluster_policy": "single_task_original_group_id", "underlying_groups": len(groups)}
    if any(item is None for item in original + compared) or original != compared:
        raise ValueError("Paired joint predictions must retain identical original subtask groups")
    parent = list(range(len(base_rows)))
    def find(index):
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index
    first_row = {}
    for index, groups in enumerate(original):
        for group in groups:
            if group in first_row:
                parent[find(index)] = find(first_row[group])
            else:
                first_row[group] = index
    components = {}
    for index in range(len(base_rows)):
        components.setdefault(find(index), []).append(index)
    return list(components.values()), {
        "bootstrap_unit": "joint_subtask_group_connected_component",
        "cluster_policy": "conservative_connected_components_sharing_either_task_qualified_subtask_group",
        "underlying_groups": len(first_row), "connected_components": len(components),
        "component_sizes": [len(indices) for indices in components.values()],
        "confidence_interval_identifiable": len(components) > 1,
        "limitation": "Connected components preserve both subtask group dependencies; a single component gives no independently resampled population uncertainty" if len(components) == 1 else None,
    }


def paired_bootstrap(condition_rows, base_rows, score_fn: Callable, *, repetitions=2000, confidence=.95, seed=17):
    """Paired original-group bootstrap; joint rows preserve either-side dependence.

    Joint rows sharing either task-qualified source group are joined transitively
    into one conservative resampling cluster. Point/corpus estimates are unchanged.
    """
    if len(condition_rows) != len(base_rows) or not base_rows:
        raise ValueError("Need nonempty paired predictions")
    if not isinstance(repetitions, int) or isinstance(repetitions, bool) or repetitions < 1 or not 0 < confidence < 1:
        raise ValueError("Need positive bootstrap repetitions and confidence in (0,1)")
    lookup = {str(row["id"]): row for row in condition_rows}
    if len(lookup) != len(condition_rows) or {str(x["id"]) for x in base_rows} != set(lookup):
        raise ValueError("Prediction IDs are not paired one-to-one")
    condition_rows = [lookup[str(x["id"])] for x in base_rows]
    indices, cluster_metadata = _paired_clusters(base_rows, condition_rows)
    rng = np.random.default_rng(seed)
    base_score, condition_score = score_fn(base_rows), score_fn(condition_rows)
    names = list(base_score)
    distribution = {name: [] for name in names}
    for _ in range(repetitions):
        chosen = rng.integers(0, len(indices), len(indices))
        selected = [i for j in chosen for i in indices[j]]
        b = score_fn([base_rows[i] for i in selected])
        c = score_fn([condition_rows[i] for i in selected])
        for name in names:
            distribution[name].append(float(c[name]) - float(b[name]))
    tail = (1 - confidence) / 2
    return {name: {"base": float(base_score[name]), "condition": float(condition_score[name]),
                   "delta": float(condition_score[name]) - float(base_score[name]),
                   "ci": np.quantile(distribution[name], [tail, 1 - tail]).tolist(),
                   "repetitions": repetitions, "groups": len(indices), **cluster_metadata} for name in names}




def _replicate_batch(payload):
    """Top-level spawn-safe callable: no lambda/model/device state is pickled."""
    base, condition, names, selections = payload
    values = []
    for selected in selections:
        b, c = base.score(selected), condition.score(selected)
        values.append([float(c[name]) - float(b[name]) for name in names])
    return values


def accelerated_paired_bootstrap(task, condition_rows, base_rows, config=None, *,
                                repetitions=2000, confidence=.95, seed=17,
                                workers=1, engineering_record=None):
    """Equivalent paired bootstrap, caching per-row corpus matching statistics.

    ``workers`` only affects scheduling. All RNG draws are made in this parent
    process, in precisely the reference loop's order, before ordered CPU work.
    ``engineering_record`` is an optional caller-owned dict; timings and worker
    counts are kept there rather than added to scientific result records.
    """
    from .metrics import PreparedBootstrapMetrics

    started = time.perf_counter()
    if len(condition_rows) != len(base_rows) or not base_rows:
        raise ValueError("Need nonempty paired predictions")
    if not isinstance(repetitions, int) or isinstance(repetitions, bool) or repetitions < 1 or not 0 < confidence < 1:
        raise ValueError("Need positive bootstrap repetitions and confidence in (0,1)")
    if not isinstance(workers, int) or isinstance(workers, bool) or workers < 1:
        raise ValueError("Bootstrap workers must be a positive integer")
    lookup = {str(row["id"]): row for row in condition_rows}
    if len(lookup) != len(condition_rows) or {str(x["id"]) for x in base_rows} != set(lookup):
        raise ValueError("Prediction IDs are not paired one-to-one")
    condition_rows = [lookup[str(x["id"])] for x in base_rows]
    indices, cluster_metadata = _paired_clusters(base_rows, condition_rows)
    base = PreparedBootstrapMetrics(task, base_rows, config or {})
    condition = PreparedBootstrapMetrics(task, condition_rows, config or {})
    all_indices = list(range(len(base_rows)))
    base_score, condition_score = base.score(all_indices), condition.score(all_indices)
    names = list(base_score)
    rng = np.random.default_rng(seed)
    selections = []
    for _ in range(repetitions):
        chosen = rng.integers(0, len(indices), len(indices))
        selections.append([i for j in chosen for i in indices[j]])

    # Cached metrics are cheap. Avoid process startup for simple scalar metrics
    # and tiny jobs; enforce the caller's bounded worker request for text corpora.
    actual_workers = min(workers, os.cpu_count() or 1, repetitions)
    if task not in {"translation", "paraphrase"} or repetitions < 128:
        actual_workers = 1
    if actual_workers == 1:
        values = _replicate_batch((base, condition, names, selections))
    else:
        chunk_size = max(1, (repetitions + actual_workers * 4 - 1) // (actual_workers * 4))
        chunks = [selections[start:start + chunk_size] for start in range(0, repetitions, chunk_size)]
        payloads = [(base, condition, names, chunk) for chunk in chunks]
        with ProcessPoolExecutor(max_workers=actual_workers,
                                 mp_context=multiprocessing.get_context("spawn")) as pool:
            # map preserves replicate order regardless of worker completion order.
            values = [value for batch in pool.map(_replicate_batch, payloads) for value in batch]
    distribution = {name: [row[index] for row in values] for index, name in enumerate(names)}
    tail = (1 - confidence) / 2
    result = {name: {"base": float(base_score[name]), "condition": float(condition_score[name]),
                     "delta": float(condition_score[name]) - float(base_score[name]),
                     "ci": np.quantile(distribution[name], [tail, 1 - tail]).tolist(),
                     "repetitions": repetitions, "groups": len(indices), **cluster_metadata} for name in names}
    if engineering_record is not None:
        engineering_record.update({
            "backend": "cached_official_corpus_statistics_v1",
            "requested_workers": workers, "actual_workers": actual_workers,
            "repetitions": repetitions, "seed": seed,
            "runtime_seconds": time.perf_counter() - started,
            "point_scores_verified_against_uncached": True,
            "resampling": "reference_rng_order_and_original_group_clusters",
        })
    return result
