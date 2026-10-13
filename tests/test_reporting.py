from task_vector.io import atomic_json, read_json
from task_vector.reporting import build_report


def test_report_includes_first_round_sequential_and_relation_evidence(tmp_path):
    first = {"completed": True, "scope": "mvp", "seed": 17, "core_states": 9,
             "core_generations": 9216, "protocol_lock": "core-lock"}
    atomic_json(tmp_path / "analysis/mvp/first-round-completed.json", first)
    atomic_json(tmp_path / "analysis/controls/test_formal-completed.json",
                {"scope": "mvp", "protocol_lock": "main-lock", "seeds": {"17": {}}})
    atomic_json(tmp_path / "analysis/mvp/17/ltv-test_formal-relation-confirmation.json",
                {"off_diagonal": {"relations": 12, "sign_agreement": .75, "standardized_spearman": .6},
                 "selection_performed": False})
    atomic_json(tmp_path / "analysis/mvp/17/ltv-test_formal-components.json", {"direct_delta": True})
    atomic_json(tmp_path / "training/independent/ltv/17/math/training.json", {"status": "complete"})
    atomic_json(tmp_path / "sequential/seed17/order/stage0_math/training.json", {"status": "complete"})
    atomic_json(tmp_path / "evaluation/mvp/17/test_formal/canonical/base/math/scores.json",
                {"count": 1000, "truncation_rate": .1, "runtime_seconds": 5})
    atomic_json(tmp_path / "sequential/seed17/order/stage0_math/evaluation/test_formal/math/metrics.json",
                {"count": 1000, "truncation_rate": .2, "runtime_seconds": 7})
    result = build_report(tmp_path)
    report = read_json(tmp_path / "report.json")
    assert result["training_records"] == 1
    assert report["first_round"]["core_generations"] == 9216
    assert len(report["sequential_training"]) == 1
    assert report["sequential_evaluation_files"] == 1
    assert len(report["relation_confirmations"]) == 1
    assert len(report["scientific_panel_artifacts"]) == 1
    assert report["generation_totals"]["samples_in_saved_panels"] == 2000
    assert report["generation_totals"]["truncated"] == 300
    assert report["generation_totals"]["evaluation_seconds"] == 12
    text = (tmp_path / "report.md").read_text(encoding="utf-8")
    assert "9216" in text and "不重新选择" in text and "不可识别" in text


def test_empty_report_does_not_claim_scientific_evidence(tmp_path):
    build_report(tmp_path)
    assert "尚无正式模型训练证据" in (tmp_path / "report.md").read_text(encoding="utf-8")
