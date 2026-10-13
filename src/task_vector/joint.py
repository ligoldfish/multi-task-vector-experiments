"""Paired independent subtasks with explicit answers and separate-unit scoring."""
from __future__ import annotations

from collections import defaultdict
import hashlib
from itertools import combinations
import json
import random

from .data import Sample
from .prompts import canonical_task, task_instruction, parse_json_object


def build_joint_samples(samples_by_task: dict[str, list[Sample]], seed: int, count: int) -> list[Sample]:
    """Build ``count`` examples PER unordered task pair, without within-pair reuse.

    Each subtask is a real selected sample with its own ground truth. The two
    subproblems are independent; this panel does not invent compound task labels.
    """
    if count <= 0:
        raise ValueError("Joint count per pair must be positive")
    sources = {canonical_task(task): samples for task, samples in samples_by_task.items()}
    if "joint" in sources or len(sources) < 2:
        raise ValueError("At least two non-joint tasks required")
    output = []
    for task_a, task_b in combinations(sorted(sources), 2):
        pair_seed = int.from_bytes(hashlib.sha256(f"{seed}:{task_a}:{task_b}".encode()).digest()[:8], "big")
        rng = random.Random(pair_seed)
        a, b = list(sources[task_a]), list(sources[task_b])
        if min(len(a), len(b)) < count:
            raise ValueError(f"Joint pair {task_a}/{task_b} needs {count} unique samples per task")
        rng.shuffle(a)
        rng.shuffle(b)
        for index in range(count):
            first, second = a[index], b[index]
            prompt = (f"Subtask A ({task_a}):\n{task_instruction(task_a)}\n\n{first.prompt}"
                      f"\n\nSubtask B ({task_b}):\n{task_instruction(task_b)}\n\n{second.prompt}")
            target = json.dumps({"task_a": first.target, "task_b": second.target}, ensure_ascii=False)
            identity = hashlib.sha256(f"{first.id}|{second.id}".encode()).hexdigest()
            output.append(Sample(f"joint:{task_a}+{task_b}:{identity[:24]}", "joint", prompt, target,
                                 f"joint:{first.group_id}|{second.group_id}", f"{first.official_split}+{second.official_split}",
                                 {"subtasks": [{"key": "task_a", **first.to_dict()}, {"key": "task_b", **second.to_dict()}],
                                   "subtask_group_ids": {task_a: first.group_id, task_b: second.group_id},
                                  "pair": [task_a, task_b], "pair_seed": pair_seed, "pair_index": index,
                                  "construction": "independent_two_subtasks_separate_ground_truth", "count_scope": "per_unordered_pair"}))
    return output


def _parse_answers(prediction: str) -> tuple[dict[str, str], bool]:
    try:
        parsed = parse_json_object(prediction.strip())
    except (ValueError, TypeError):
        return {}, False
    valid = isinstance(parsed, dict) and set(parsed) == {"task_a", "task_b"} and all(isinstance(value, str) for value in parsed.values())
    return (parsed, True) if valid else ({}, False)


def _threshold(task: str, config: dict) -> tuple[str, float]:
    from .metrics import PRIMARY_METRICS
    thresholds = config.get("joint", {}).get("thresholds", {})
    value = thresholds.get(task)
    if value is None:
        raise ValueError(f"Joint nonbinary success requires pre-locked joint.thresholds.{task}; report separate original-unit scores and lock a criterion before joint-success evaluation")
    if isinstance(value, dict):
        return value.get("metric", PRIMARY_METRICS[task]), float(value["threshold"])
    return PRIMARY_METRICS[task], float(value)


def score_joint_predictions(rows: list[dict], config: dict) -> dict:
    from .metrics import score_predictions
    prepared = []
    for raw in rows:
        row = {**raw, "metadata": dict(raw.get("metadata", {})), "scores": dict(raw.get("scores", {}))}
        if "subtask_rows" in row and "joint_success" in row["scores"]:
            prepared.append(row)
            continue
        answers, format_valid = _parse_answers(str(row["prediction"]))
        subtasks = row["metadata"].get("subtasks")
        if not subtasks or len(subtasks) != 2:
            raise ValueError("Joint row must retain two original subtask ground truths")
        subtask_rows, successes = [], []
        for subtask in subtasks:
            task = canonical_task(subtask["task"])
            prediction = answers.get(subtask["key"], "")
            scored = score_predictions(task, [{**subtask, "prediction": prediction, "variant": "canonical", "truncated": bool(row.get("truncated", False))}], config)
            scored_row = scored["rows"][0]
            primary = scored["primary_metric"]
            if primary in {"accuracy", "pass_all"}:
                threshold_metric, threshold = primary, 1.0
                success = scored_row["scores"][primary] == 1
                definition = "binary_correctness"
            else:
                threshold_metric, threshold = _threshold(task, config)
                success = scored_row["scores"][threshold_metric] >= threshold
                definition = "predefined_metric_threshold"
            scored_row["joint_criterion"] = {"metric": threshold_metric, "threshold": threshold, "definition": definition, "success": bool(success and format_valid)}
            subtask_rows.append(scored_row)
            successes.append(bool(success and format_valid))
        row.update({"subtask_rows": subtask_rows, "joint_format_valid": format_valid})
        row["scores"].update({"joint_success": float(all(successes)), "format_valid": float(format_valid)})
        prepared.append(row)
    return aggregate_joint_predictions(prepared, config)


def aggregate_joint_predictions(rows: list[dict], config: dict) -> dict:
    from .metrics import aggregate_predictions
    if not rows:
        raise ValueError("Empty joint panel")
    by_task, by_pair = defaultdict(list), defaultdict(list)
    for row in rows:
        if "subtask_rows" not in row or "joint_success" not in row.get("scores", {}):
            raise ValueError("Joint bootstrap requires cached subtask rows")
        for sub in row["subtask_rows"]:
            by_task[sub["task"]].append(sub)
        pair = "+".join(sorted(row["metadata"]["pair"]))
        by_pair[pair].append(row)
    return {
        "metrics": {"joint_success": sum(row["scores"]["joint_success"] for row in rows) / len(rows), "format_valid": sum(row["scores"]["format_valid"] for row in rows) / len(rows)},
        "primary_metric": "joint_success", "sample_scores": [row["scores"]["joint_success"] for row in rows],
        "rows": rows, "units": {"joint_success": "fraction_meeting_both_locked_criteria", "format_valid": "fraction"},
        "per_task": {task: {key: value for key, value in aggregate_predictions(task, samples, config).items() if key != "rows"} for task, samples in sorted(by_task.items())},
        "per_pair": {pair: {"count": len(samples), "joint_success": sum(row["scores"]["joint_success"] for row in samples) / len(samples)} for pair, samples in sorted(by_pair.items())},
        "count": len(rows), "thresholds": config.get("joint", {}).get("thresholds", {}),
        "interpretation": "paired_independent_subtasks; no averaging_heterogeneous_task_metrics",
        "truncation_rate": sum(bool(row.get("truncated", False)) for row in rows) / len(rows),
    }


def calibrate_joint_thresholds(tune_baseline_rows: dict[str, list[dict]], config: dict) -> dict:
    """Explicit optional protocol: median baseline sentence scores on TUNE only.

    The caller must save and lock the returned thresholds before discovery/test.
    This function never reads a final-test panel or silently calibrates at scoring.
    """
    from statistics import median
    from .metrics import PRIMARY_METRICS, score_predictions
    thresholds = {}
    for name, rows in tune_baseline_rows.items():
        task = canonical_task(name)
        primary = PRIMARY_METRICS[task]
        if primary in {"accuracy", "pass_all"}:
            continue
        if any(row.get("experiment_split") not in {None, "tune"} for row in rows):
            raise ValueError("Joint threshold calibration is restricted to tuning data")
        scored = score_predictions(task, rows, config)
        thresholds[task] = {"metric": primary, "threshold": median(scored["sample_scores"]), "source": "tune_baseline_sentence_score_median", "sample_ids": [row["id"] for row in rows], "not_corpus_score": True}
    return thresholds
