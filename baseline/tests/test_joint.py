import json

import pytest

from task_vector.data import Sample
from task_vector.joint import build_joint_samples, calibrate_joint_thresholds
from task_vector.metrics import aggregate_predictions, score_predictions


def sources():
    return {"math": [Sample("m1", "math", "1+1?", "#### 2", "m", "test")],
            "logic": [Sample("l1", "logic", "Bob is red. Query: Bob is red.", "True", "l", "test")],
            "translation": [Sample("t1", "translation", "This is a cat.", "Das ist eine Katze.", "t", "test")]}


def test_all_unordered_pairs_have_real_subtask_answers_and_repeatable_pairing():
    samples = build_joint_samples(sources(), 17, 1)
    assert len(samples) == 3
    assert [sample.to_dict() for sample in samples] == [sample.to_dict() for sample in build_joint_samples(sources(), 17, 1)]
    for sample in samples:
        target = json.loads(sample.target)
        assert set(target) == {"task_a", "task_b"}
        assert target["task_a"] == sample.metadata["subtasks"][0]["target"]
        assert sample.metadata["subtask_group_ids"] == {subtask["task"]: subtask["group_id"] for subtask in sample.metadata["subtasks"]}


def test_binary_joint_success_requires_both_answers_and_format():
    sample = build_joint_samples({key: value for key, value in sources().items() if key != "translation"}, 17, 1)[0]
    correct = {**sample.to_dict(), "prediction": sample.target}
    malformed = {**sample.to_dict(), "prediction": "True\n#### 2"}
    result = score_predictions("joint", [correct, malformed])
    assert result["sample_scores"] == [1, 0]
    assert result["metrics"]["joint_success"] == 0.5
    assert result["per_task"]["math"]["units"]["accuracy"] == "fraction"
    assert aggregate_predictions("joint", [result["rows"][0]])["metrics"]["joint_success"] == 1
    duplicate = {**sample.to_dict(), "prediction": '{"task_a":"True","task_a":"True","task_b":"#### 2"}'}
    assert score_predictions("joint", [duplicate])["metrics"]["format_valid"] == 0


def test_nonbinary_joint_needs_explicit_threshold_and_keeps_units():
    sample = build_joint_samples({key: value for key, value in sources().items() if key != "logic"}, 17, 1)[0]
    rows = [{**sample.to_dict(), "prediction": sample.target}]
    with pytest.raises(ValueError, match="pre-locked"):
        score_predictions("joint", rows)
    result = score_predictions("joint", rows, {"joint": {"thresholds": {"translation": {"metric": "chrF++", "threshold": 50}}}})
    assert result["metrics"]["joint_success"] == 1
    assert result["per_task"]["translation"]["units"]["chrF++"] == "0-100"
    assert "chrF++" not in result["metrics"]  # No heterogeneous overall average.


def test_joint_thresholds_only_from_baseline_tuning_panel():
    rows = [{"id": "x", "group_id": "x", "prompt": "This is a cat.", "target": "Das ist eine Katze.", "prediction": "Das ist eine Katze.", "experiment_split": "test"}]
    with pytest.raises(ValueError, match="tuning"):
        calibrate_joint_thresholds({"translation": rows}, {})
    rows[0]["experiment_split"] = "tune"
    threshold = calibrate_joint_thresholds({"translation": rows}, {})
    assert threshold["translation"]["threshold"] == pytest.approx(100)
    assert threshold["translation"]["source"] == "tune_baseline_sentence_score_median"
