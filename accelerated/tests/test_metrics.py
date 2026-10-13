from fractions import Fraction

import pytest

from task_vector.code_execution import execute_code, make_verified_bug
from task_vector.metrics import aggregate_predictions, extract_math_answer, score_predictions, semantic_asset_fingerprint


def row(prediction, target, **kwargs):
    return {"id": "one", "group_id": "g", "prompt": "source", "target": target, "prediction": prediction, "metadata": {}, **kwargs}


def test_math_full_answer_scoring_and_invalid_output():
    assert extract_math_answer("Reasoning 3 + 4. #### 1,200") == Fraction(1200)
    assert extract_math_answer("The final answer is \\boxed{3/4}") == Fraction(3, 4)
    assert extract_math_answer("The final answer is \\boxed{\\frac{3}{4}}") == Fraction(3, 4)
    assert extract_math_answer("#### .5") == Fraction(1, 2)
    assert extract_math_answer("#### 50%") == Fraction(50)
    result = score_predictions("math", [row("#### 1200", "Reasoning #### 1,200"), row("No answer", "#### 3")])
    assert result["metrics"]["accuracy"] == 0.5
    assert result["sample_scores"] == [1.0, 0.0]


def test_label_recode_and_format_controls_scored_against_canonical_gold():
    result = score_predictions("logic", [row("C", "Unknown", variant="label_recode"), row('{"answer":"True"}', "True", variant="output_format"), row("False because it is false", "False")])
    assert result["sample_scores"] == [1, 1, 0]
    extra = score_predictions("logic", [row('{"answer":"True","extra":"reason"}', "True", variant="output_format")])
    assert extra["sample_scores"] == [0]


def test_translation_corpus_score_recomputed_not_averaged():
    rows = [row("Das ist ein sehr langer deutscher Satz mit vielen Wörtern.", "Das ist ein sehr langer deutscher Satz mit vielen Wörtern."), row("Katze", "Hund")]
    scored = score_predictions("translation", rows)
    assert scored["metrics"]["chrF++"] != pytest.approx(sum(scored["sample_scores"]) / 2)
    resampled = aggregate_predictions("translation", [scored["rows"][0]] * 3)
    assert resampled["metrics"]["chrF++"] == pytest.approx(100)
    assert "nw:2" in scored["signatures"]["chrF++"]


def test_ibleu_copy_penalty_and_cached_semantic_bootstrap(monkeypatch):
    import task_vector.metrics as metrics
    calls = []
    def semantic(rows, config):
        calls.append(len(rows))
        return [(0.8, 0.7)] * len(rows)
    monkeypatch.setattr(metrics, "_semantic_scores", semantic)
    source = "How can I learn to play the piano well?"
    target = "What is a good way to become skilled at piano?"
    scored = score_predictions("paraphrase", [{**row(source, target), "prompt": source}, {**row(target, target), "prompt": source}])
    assert calls == [2]
    assert scored["rows"][0]["scores"]["copy_rate"] == 1
    assert scored["rows"][1]["scores"]["copy_rate"] == 0
    assert scored["rows"][1]["scores"]["iBLEU"] > scored["rows"][0]["scores"]["iBLEU"]
    aggregate_predictions("paraphrase", [scored["rows"][1]] * 4)
    assert calls == [2]


def test_semantic_model_is_required_not_silently_skipped(monkeypatch):
    monkeypatch.delenv("TV_SEMANTIC_MODEL", raising=False)
    with pytest.raises(ValueError, match="cannot be omitted"):
        score_predictions("paraphrase", [row("different", "reference")])


def test_semantic_fingerprint_tracks_content_and_not_absolute_placement(tmp_path):
    first, second = tmp_path / "first", tmp_path / "second"
    first.mkdir()
    second.mkdir()
    for path in (first, second):
        (path / "config.json").write_text('{"dimension":384}', encoding="utf-8")
        (path / "model.safetensors").write_bytes(b"only a hash-test fixture; never a model")
    config_a = {"metrics": {"semantic_model_path": str(first)}}
    config_b = {"metrics": {"semantic_model_path": str(second)}}
    assert semantic_asset_fingerprint(config_a) == semantic_asset_fingerprint(config_b)
    (second / "model.safetensors").write_bytes(b"different bytes invalidate the content fingerprint")
    assert semantic_asset_fingerprint(config_a) != semantic_asset_fingerprint(config_b)


def test_real_program_execution_verified_bug_and_cached_aggregation(monkeypatch):
    code = "def add(a, b):\n    return a + b"
    tests = ["assert add(1, 2) == 3", "assert add(-1, 4) == 3"]
    verified = make_verified_bug(code, tests, config={"timeout_seconds": 3})
    assert verified["original_execution"]["pass_all"]
    assert not verified["bug_execution"]["pass_all"]
    result = score_predictions("code_repair", [row(code, code, metadata={"test_list": tests})])
    assert result["metrics"]["pass_all"] == 1
    monkeypatch.setattr("task_vector.metrics.execute_code", lambda *args: pytest.fail("Bootstrap re-executed code"))
    assert aggregate_predictions("code_repair", result["rows"]) ["metrics"]["pass_all"] == 1


def test_infinite_loop_times_out_and_cannot_score_as_success():
    result = execute_code("while True:\n    pass", ["assert True"], config={"timeout_seconds": 0.2})
    assert not result["pass_all"]
    assert result["timeout"] or result["returncode"] != 0


def test_summary_retains_original_rouge_units():
    result = score_predictions("samsum", [row("Alice will meet Bob tomorrow.", "Alice will meet Bob tomorrow.")])
    assert result["metrics"]["rougeL"] == pytest.approx(100)
    assert result["units"]["rougeL"] == "0-100"
    signature = result["signatures"]["ROUGE"]
    assert signature["version"] and signature["metrics"] == ["rouge1", "rouge2", "rougeL"]
    assert signature["use_stemmer"] is True and signature["statistic"] == "F1" and signature["scale"] == 100
    unstemmed = score_predictions("summary", [row("Alice will meet Bob tomorrow.", "Alice will meet Bob tomorrow.")], {"metrics": {"summary": {"use_stemmer": False}}})
    assert unstemmed["signatures"]["ROUGE"]["use_stemmer"] is False
