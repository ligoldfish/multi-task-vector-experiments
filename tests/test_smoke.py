import copy
import os
from pathlib import Path

import pytest
import torch

from task_vector.data import Sample
from task_vector.io import read_json
from task_vector.smoke import _run_loda_training_smoke
from test_ltv import tiny_model
from test_training import TinyTokenizer, training_config


class StatisticsTokenizer(TinyTokenizer):
    def __call__(self, text, **kwargs):
        ids = torch.tensor([self.encode(text)], dtype=torch.long)
        return {"input_ids": ids, "attention_mask": torch.ones_like(ids)}


def train_samples(count=16):
    return [Sample(f"logic:{i}", "logic", f"premise {i}", "True", f"group:{i}", "train")
            for i in range(count)]


def test_real_loda_smoke_trains_one_gao_batch_and_resumes_without_step(tmp_path):
    if os.name == "nt":
        tmp_path = Path("\\\\?\\" + str(tmp_path))
    config = training_config(epochs=5)
    config["loda"] = {"rank_general": 2, "rank_isolated": 2, "statistics_samples": 128,
                      "max_statistics_tokens": 8, "rho_max": .05, "recalibration_lambda": 3}
    original = copy.deepcopy(config)
    model = tiny_model()
    base = {name: value.detach().clone() for name, value in model.state_dict().items()}
    result = _run_loda_training_smoke(model, StatisticsTokenizer(), train_samples(), config,
                                     "cpu", tmp_path / "engineering", "base-fingerprint")
    assert config == original
    assert result["engineering_only"] and not result["scientific_training"]
    assert result["sample_split"] == "train" and result["calibration_samples_total"] == 16
    assert result["calibration_samples_per_half"] == 8
    assert result["rank_general"] == result["rank_isolated"] == 2
    assert result["logical_steps"] == 1 and result["optimizer_steps"] == result["scheduler_steps"] == 2
    assert result["resume_additional_optimizer_steps"] == 0
    assert result["checkpoint_state_exact"] and result["rng_state_exact"]
    assert len(result["group1_ids"]) == len(result["group2_ids"]) == 8
    assert set(result["group1_ids"]).isdisjoint(result["group2_ids"])
    assert all(version["exact_logits"] for version in result["raw_post_resume"])
    assert not result["independent_validation"]
    assert set(result["diagnostic_loss_ids"]) <= set(result["sample_ids"])
    assert all(torch.equal(base[name], value) for name, value in model.state_dict().items())
    assert all(not parameter.requires_grad and parameter.grad is None for parameter in model.parameters())
    for stage in ("uninterrupted", "resumed"):
        directory = tmp_path / "engineering" / stage
        assert not (directory / "training.json").exists()
        assert read_json(directory / "training-state.json")["status"] == "exported_uncommitted"
    assert read_json(tmp_path / "engineering" / "loda-engineering.json")["resume_additional_optimizer_steps"] == 0


@pytest.mark.parametrize("mode", ["short", "duplicate", "other_task"])
def test_engineering_smoke_rejects_invalid_source_pool_before_forward(tmp_path, mode):
    selected = train_samples(15 if mode == "short" else 16)
    if mode == "duplicate":
        selected[-1].id = selected[0].id
    if mode == "other_task":
        selected[-1].task = "math"
    with pytest.raises(ValueError, match="16 unique|must not mix"):
        _run_loda_training_smoke(None, None, selected, training_config(), "cpu", tmp_path, "base")
