"""Engineering acceleration must preserve every reference bootstrap replicate."""
import copy

import numpy as np
import pytest

from task_vector.bootstrap import (
    _paired_clusters, accelerated_paired_bootstrap, paired_bootstrap,
)
from task_vector.metrics import PreparedBootstrapMetrics, aggregate_predictions


def text_rows(task, condition=False):
    targets = [
        "A small red fox runs over the quiet hill.",
        "The scientist wrote a detailed experimental report.",
        "", "Rain falls.",
        "One two three four five six seven eight nine.",
        "The apple is green; the pear is yellow!",
    ]
    predictions = [
        "A red fox runs over a quiet hill.",
        "A scientist wrote the detailed experimental report.",
        "", "It rains heavily over the hill today.",
        "One two three four five.", "The apple is green.",
    ]
    if condition:
        predictions[0] = targets[0]
        predictions[1] = "The scientist made a report."
        predictions[4] = targets[4]
    rows = []
    for index, (prediction, target) in enumerate(zip(predictions, targets)):
        scores = {"BLEU": 0., "chrF++": 0.} if task == "translation" else {
            "iBLEU": 0., "reference_BLEU": 0., "source_BLEU": 0.,
            "copy_rate": float(index == 0), "copied_token_fraction": index / 7,
            "semantic_source": .3 + index / 10,
            "semantic_reference": .2 + index / 11,
        }
        rows.append({
            "id": f"{task}:{index}", "group_id": f"group:{index // 2}",
            "task": task, "prediction": prediction, "target": target,
            "prompt": "Rewrite the following sentence.", "scores": scores,
            "metadata": {"source_question": "A fox runs over the hill."},
        })
    return rows


@pytest.mark.parametrize("task", ["translation", "paraphrase"])
def test_all_2000_seeded_replicates_and_confidence_intervals_are_exact(task):
    base, condition = text_rows(task), text_rows(task, True)
    prepared = {id(base): PreparedBootstrapMetrics(task, base, {}),
                id(condition): PreparedBootstrapMetrics(task, condition, {})}
    for rows in (base, condition):
        corpus = prepared[id(rows)]
        clusters, _ = _paired_clusters(rows, rows)
        rng = np.random.default_rng(17)
        for _ in range(2000):
            chosen = rng.integers(0, len(clusters), len(clusters))
            indices = [index for group in chosen for index in clusters[group]]
            expected = aggregate_predictions(task, [rows[index] for index in indices], {})["metrics"]
            assert corpus.score(indices) == expected
    score = lambda rows: aggregate_predictions(task, rows, {})["metrics"]
    expected = paired_bootstrap(condition, base, score, repetitions=2000)
    record = {}
    actual = accelerated_paired_bootstrap(
        task, list(reversed(condition)), base, {}, repetitions=2000,
        engineering_record=record,
    )
    assert actual == expected
    assert record["point_scores_verified_against_uncached"] is True
    assert record["actual_workers"] == 1
    assert record["runtime_seconds"] >= 0
    assert all("runtime_seconds" not in value for value in actual.values())


@pytest.mark.parametrize("task", ["translation", "paraphrase"])
def test_spawned_cpu_workers_preserve_ordered_seeded_results(task):
    base, condition = text_rows(task), text_rows(task, True)
    expected = accelerated_paired_bootstrap(task, condition, base, {}, repetitions=128, workers=1)
    record = {}
    actual = accelerated_paired_bootstrap(
        task, condition, base, {}, repetitions=128, workers=2,
        engineering_record=record,
    )
    assert actual == expected
    assert record["actual_workers"] == min(2, __import__("os").cpu_count() or 1)


def test_original_float_summation_order_and_variable_cluster_sizes():
    base = [
        {"id": str(i), "group_id": str(i // 3), "scores": {"accuracy": value}}
        for i, value in enumerate([.1, .2, .3, .4, .5, .6, .7])
    ]
    condition = [{**row, "scores": {"accuracy": 1. - row["scores"]["accuracy"]}} for row in base]
    score = lambda rows: aggregate_predictions("math", rows, {})["metrics"]
    expected = paired_bootstrap(condition, base, score, repetitions=2000, seed=29)
    record = {}
    actual = accelerated_paired_bootstrap("math", condition, base, {}, repetitions=2000, seed=29, workers=8, engineering_record=record)
    assert actual == expected
    assert record["actual_workers"] == 1


def joint_rows(condition=False):
    rows = []
    # First three rows form one transitive component through either-side groups.
    for index, (a, b) in enumerate([("a", "x"), ("a", "y"), ("b", "y"), ("c", "z")]):
        success = float((index % 2 == 0) != condition)
        subtasks = [
            {"id": f"math:{a}", "task": "math", "scores": {"accuracy": success}},
            {"id": f"logic:{b}", "task": "logic", "scores": {"accuracy": 1.}},
        ]
        rows.append({
            "id": f"joint:{index}", "task": "joint",
            "metadata": {"pair": ["math", "logic"], "subtask_group_ids": {"math": a, "logic": b}},
            "scores": {"joint_success": success, "format_valid": 1.},
            "subtask_rows": subtasks,
        })
    return rows


def test_joint_keeps_transitive_either_task_dependency_and_nested_cached_scores():
    base, condition = joint_rows(), joint_rows(True)
    score = lambda rows: aggregate_predictions("joint", rows, {})["metrics"]
    expected = paired_bootstrap(condition, base, score, repetitions=2000)
    actual = accelerated_paired_bootstrap("joint", list(reversed(condition)), base, {}, repetitions=2000, workers=8)
    assert actual == expected
    assert actual["joint_success"]["component_sizes"] == [3, 1]
    assert actual["joint_success"]["groups"] == 2


@pytest.mark.parametrize("workers", [0, -1, True, 1.5])
def test_invalid_worker_requests_are_rejected(workers):
    rows = [{"id": "a", "scores": {"accuracy": 1.}}]
    with pytest.raises(ValueError, match="workers"):
        accelerated_paired_bootstrap("math", rows, rows, workers=workers)


def test_pairing_validation_is_not_bypassed():
    base = [{"id": "a", "group_id": "g", "scores": {"accuracy": 1.}}]
    with pytest.raises(ValueError, match="group IDs"):
        accelerated_paired_bootstrap("math", [{**base[0], "group_id": "other"}], base)
    with pytest.raises(ValueError, match="one-to-one"):
        accelerated_paired_bootstrap("math", [{**base[0], "id": "other"}], base)
    missing_nested = copy.deepcopy(joint_rows())
    del missing_nested[0]["subtask_rows"]
    with pytest.raises(ValueError, match="cached subtask"):
        accelerated_paired_bootstrap("joint", missing_nested, missing_nested)


def test_private_statistic_api_is_version_pinned(monkeypatch):
    from task_vector import metrics
    original = metrics.importlib.metadata.version
    monkeypatch.setattr(metrics.importlib.metadata, "version", lambda name: "9.0" if name == "sacrebleu" else original(name))
    with pytest.raises(RuntimeError, match="sacrebleu==2.5.1"):
        PreparedBootstrapMetrics("translation", text_rows("translation"), {})


def test_corpus_bootstrap_is_not_mean_sentence_score():
    rows = text_rows("translation")
    prepared = PreparedBootstrapMetrics("translation", rows, {})
    result = prepared.score(list(range(len(rows))))
    assert result["BLEU"] != sum(row["scores"]["BLEU"] for row in rows) / len(rows)
    assert result["chrF++"] != sum(row["scores"]["chrF++"] for row in rows) / len(rows)


@pytest.mark.parametrize("task", ["translation", "paraphrase"])
def test_only_primary_cached_field_and_nondefault_text_settings(task):
    rows = text_rows(task)
    primary = "chrF++" if task == "translation" else "iBLEU"
    for row in rows:
        row["scores"] = {primary: row["scores"][primary]}
    config = {"metrics": {task: {"bleu_tokenize": "intl", "lowercase": True,
                                 "bleu_smoothing": "floor", "ibleu_alpha": .75}}}
    prepared = PreparedBootstrapMetrics(task, rows, config)
    for selected in ([0, 1, 0], [2, 2, 3], [5, 4, 3, 2, 1, 0]):
        assert prepared.score(list(selected)) == aggregate_predictions(task, [rows[i] for i in selected], config)["metrics"]


def test_identical_condition_has_exact_zero_effect_and_input_rows_are_unchanged():
    rows = text_rows("paraphrase")
    original = copy.deepcopy(rows)
    results = accelerated_paired_bootstrap("paraphrase", rows, rows, repetitions=128)
    for result in results.values():
        assert result["delta"] == 0.
        assert result["ci"] == [0., 0.]
    assert rows == original
