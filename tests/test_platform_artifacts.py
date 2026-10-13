import copy
import json
import os
from pathlib import Path

import torch
import yaml

from task_vector.assets import inspect_assets
from test_ltv import tiny_model
from test_training import TinyTokenizer, samples, training_config


def base_config(tmp_path):
    with open(Path(__file__).resolve().parents[1] / "configs" / "experiment.yaml", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    config["platform"]["persistent_root"] = str(tmp_path / "persistent")
    return config


def test_asset_inventory_never_guesses_missing_paths(tmp_path):
    config = base_config(tmp_path)
    report = inspect_assets(config, tmp_path / "assets.json")
    assert report["model"]["status"] == "missing"
    assert report["mvp_ready"] is False
    assert all(row["status"] == "missing" for row in report["datasets"].values())
    assert json.loads((tmp_path / "assets.json").read_text())["scientific_validation"] == "not_evaluated"


def test_asset_index_checks_every_pinned_weight_shard(tmp_path):
    config = base_config(tmp_path)
    model = tmp_path / "qwen"
    model.mkdir()
    cfg = {"model_type": "qwen3", **config["model"]["expected_config"]}
    (model / "config.json").write_text(json.dumps(cfg))
    (model / "tokenizer.json").write_text("{}")
    (model / "model.safetensors.index.json").write_text(json.dumps({"weight_map": {"a": "model-1.safetensors", "b": "model-2.safetensors"}}))
    (model / "model-1.safetensors").write_bytes(b"fixture")
    config["model"]["path"] = str(model)
    report = inspect_assets(config)
    assert report["model"]["status"] == "missing"
    assert report["model"]["missing_shards"] == [str((model / "model-2.safetensors").resolve())]
    (model / "model-2.safetensors").write_bytes(b"fixture2")
    report = inspect_assets(config, hash_weights=True)
    assert report["model"]["status"] == "ready"
    assert all(len(row["sha256"]) == 64 for row in report["model"]["weight_files"])


def test_real_smoke_logic_with_tiny_qwen_and_training_only_fixture(tmp_path, monkeypatch):
    if os.name == "nt":
        tmp_path = Path("\\\\?\\" + str(tmp_path))
    config = training_config(epochs=1)
    config["platform"] = {"device": "cpu", "persistent_root": str(tmp_path)}
    config["loda"] = {"rank_general": 2, "rank_isolated": 2}
    model = tiny_model()
    class StatisticsTokenizer(TinyTokenizer):
        def __call__(self, text, **kwargs):
            ids = torch.tensor([self.encode(text)], dtype=torch.long)
            return {"input_ids": ids, "attention_mask": torch.ones_like(ids)}
    class FakeRunner:
        def __init__(self, config):
            self.config = config
            self.root = tmp_path
            self.source_root = tmp_path
            self.device = "cpu"
            self.manifest = {"fixture": True}
        def load_model(self):
            return model, StatisticsTokenizer()
        def samples(self, task, split):
            assert split == "train"
            return samples()[:16]
        def base_fingerprint(self):
            return "tiny-engineering-fixture"
        def release_model(self):
            pass
    monkeypatch.setattr("task_vector.runner.ExperimentRunner", FakeRunner)
    from task_vector.smoke import run_smoke
    record = run_smoke(config, task="logic", output_dir=tmp_path / "smoke")
    assert record["passed"] is True
    assert record["formal_evaluation"] is False
    assert record["checks"]["zero_loda_equals_base_exactly"]
    assert record["checks"]["base_frozen_no_gradient"]
    assert record["checks"]["loda_true_gao_microbatch_step"]
    assert record["checks"]["loda_checkpoint_resume_exact"]
    assert record["checks"]["loda_resume_does_not_repeat_step"]
    assert record["loda_engineering"]["logical_steps"] == 1
    assert record["loda_engineering"]["optimizer_steps"] == 2
    assert record["loda_engineering"]["scheduler_steps"] == 2
    assert len(set(record["loda_engineering"]["sample_ids"])) == 16
    assert record["loda_engineering"]["resume_additional_optimizer_steps"] == 0
    assert record["loda_engineering"]["scientific_training"] is False
    assert len(record["checks"]["nonzero_loda_saved_raw_post_equivalence"]) == 2
    assert len(record["checks"]["cached_full_prefix_generation"]) == 2
    assert record["base_fingerprint"] == "tiny-engineering-fixture"
    assert all(not param.requires_grad for param in model.parameters())


def test_cli_assets_dispatch_writes_persistent_artifact(tmp_path):
    config = base_config(tmp_path)
    path = tmp_path / "run.yaml"
    path.write_text(yaml.safe_dump(config), encoding="utf-8")
    from task_vector.cli import main
    assert main(["assets", "--config", str(path)]) == 0
    assert (tmp_path / "persistent" / "runs" / "assets.json").is_file()
