import copy
import json
from pathlib import Path
import shutil
from types import SimpleNamespace

import pytest
import torch

from task_vector.losses import answer_nll, collate_encoded
from task_vector.ltv import LTVAdapter
from task_vector.training import accumulated_backward, capture_rng_state, restore_rng_state, train_task
from test_ltv import tiny_model


def examples():
    return [{"id": str(i), "input_ids": [3, 4] + answer, "labels": [-100, -100] + answer,
             "prompt_length": 2} for i, answer in enumerate(([6, 2], [7, 8, 2], [9, 2], [10, 11, 12, 2]))]


def test_loss_is_sample_mean_and_includes_eos():
    torch.manual_seed(3)
    batch = collate_encoded(examples(), 0)
    logits = torch.randn(*batch["labels"].shape, 37)
    losses = answer_nll(logits, batch["labels"])
    manual = []
    for i, example in enumerate(examples()):
        token_losses = [-torch.log_softmax(logits[i, pos - 1], 0)[label]
                        for pos, label in enumerate(example["labels"]) if label != -100]
        manual.append(torch.stack(token_losses).mean())
    torch.testing.assert_close(losses, torch.stack(manual))
    token_weighted = sum(losses[i] * (len(examples()[i]["input_ids"]) - 2) for i in range(4)) / 11
    assert not torch.isclose(losses.mean(), token_weighted)


def test_chunked_large_vocabulary_loss_and_gradient_match_reference():
    torch.manual_seed(22)
    logits = torch.randn(2, 7, 4100, requires_grad=True)
    labels = torch.tensor([[-100, -100, 3, 4, 2, -100, -100], [-100, -100, -100, 5, 6, 7, 2]])
    actual = answer_nll(logits, labels, chunk_size=2)
    shifted = labels[:, 1:]
    tokens = torch.nn.functional.cross_entropy(logits[:, :-1].float().transpose(1, 2), shifted,
                                               reduction="none", ignore_index=-100)
    expected = tokens.sum(1) / shifted.ne(-100).sum(1)
    actual.mean().backward()
    grad = logits.grad.clone()
    logits.grad = None
    expected.mean().backward()
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(grad, logits.grad)


def test_microbatch_accumulation_equals_logical_batch_gradient():
    model = tiny_model()
    adapter = LTVAdapter(model, [1], seed=17)
    with adapter:
        batch = collate_encoded(examples(), 0)
        adapter.set_positions(batch["prompt_positions"])
        loss = answer_nll(model(batch["input_ids"], attention_mask=batch["attention_mask"], use_cache=False).logits,
                          batch["labels"]).mean()
        loss.backward()
        expected = adapter.vectors["1"].grad.clone()
        adapter.zero_grad(set_to_none=True)
        actual_loss = accumulated_backward(model, adapter, examples(), micro_batch_size=2, pad_token_id=0, device="cpu")
        torch.testing.assert_close(adapter.vectors["1"].grad, expected, atol=1e-7, rtol=1e-5)
        assert actual_loss == pytest.approx(float(loss.detach()), abs=1e-6)


class TinyTokenizer:
    pad_token_id = 0
    eos_token_id = 2

    def apply_chat_template(self, messages, **kwargs):
        assert kwargs["enable_thinking"] is False
        return "prompt " + messages[-1]["content"]

    def encode(self, text, **kwargs):
        return [3 + ord(char) % 30 for char in text]


def training_config(epochs=2):
    return {"model": {"id": "tiny"}, "ltv": {"layers": [1]},
            "training": {"learning_rate": 0.001, "epochs": epochs, "micro_batch_size": 2,
                         "effective_batch_size": 16, "gradient_accumulation_steps": 8,
                         "weight_decay_ltv": 0.01, "gradient_checkpointing": True,
                         "max_prompt_tokens": 512, "max_target_tokens": 512,
                         "patience": 2, "max_grad_norm": 1.0, "save_every_steps": 1}}


def samples():
    return [SimpleNamespace(id=str(i), task="logic", prompt=str(i), target="True", metadata={}, group_id=str(i)) for i in range(18)]


def test_training_exports_best_and_resumes_identically(tmp_path, monkeypatch):
    tokenizer = TinyTokenizer()
    cfg = training_config()
    model = tiny_model()
    uninterrupted = train_task(model, tokenizer, samples(), samples()[:2], task="logic", route="ltv",
        seed=17, output_dir=tmp_path / "full", config=cfg, device="cpu")
    original_nll = __import__("task_vector.training", fromlist=["evaluate_nll"]).evaluate_nll
    def stop_after_first_epoch(*args, **kwargs):
        raise RuntimeError("simulate interruption after last batch, before validation")
    monkeypatch.setattr("task_vector.training.evaluate_nll", stop_after_first_epoch)
    with pytest.raises(RuntimeError, match="simulate interruption"):
        train_task(tiny_model(), tokenizer, samples(), samples()[:2], task="logic", route="ltv",
            seed=17, output_dir=tmp_path / "resume", config=cfg, device="cpu")
    monkeypatch.setattr("task_vector.training.evaluate_nll", original_nll)
    cfg["training"]["resume_from"] = str(tmp_path / "resume" / "last.pt")
    resumed = train_task(tiny_model(), tokenizer, samples(), samples()[:2], task="logic", route="ltv",
        seed=17, output_dir=tmp_path / "resume", config=cfg, device="cpu")
    from task_vector.representations import Representation
    expected = Representation.load(uninterrupted["raw_path"])
    actual = Representation.load(resumed["raw_path"])
    for key in expected.keys:
        torch.testing.assert_close(expected.values[key], actual.values[key], atol=0, rtol=0)
    assert resumed["optimizer_steps"] == uninterrupted["optimizer_steps"] == 4
    assert resumed["best_nll"] == uninterrupted["best_nll"]
    assert all(not p.requires_grad for p in model.parameters())
    assert not (tmp_path / "full" / "training.json").exists()
    assert json.loads((tmp_path / "full" / "training-state.json").read_text())["status"] == "exported_uncommitted"


def test_rng_roundtrip():
    state = capture_rng_state()
    expected = torch.rand(4)
    restore_rng_state(state)
    torch.testing.assert_close(torch.rand(4), expected, atol=0, rtol=0)


def test_loda_gao_training_integration_two_updates_per_logical_batch(tmp_path):
    model = tiny_model()
    spaces = {}
    for layer in range(3):
        spaces[f"model.layers.{layer}"] = {"A_G": torch.eye(16)[:2] / 3**0.5,
            "A_I": torch.eye(16)[2:4] / 3**0.5, "lambda": torch.ones(2),
            "S_self": torch.eye(16), "S_other": torch.eye(16),
            "metadata": {"recalibration_lambda": 3.0, "epsilon": 1e-12}}
    cfg = training_config(epochs=1)
    cfg["loda"] = {"rho_max": 0.05}
    cfg["training"]["weight_decay_loda"] = 0.0
    result = train_task(model, TinyTokenizer(), samples(), samples()[:2],
        task="logic", route="loda", seed=17, output_dir=tmp_path / "loda",
        config=cfg, device="cpu", subspaces=spaces)
    assert result["steps"] == 2
    assert result["optimizer_steps"] == 4
    from task_vector.representations import Representation, norm
    rep = Representation.load(result["post_path"])
    assert norm(rep) > 0
    assert all(not p.requires_grad for p in model.parameters())
    assert all(not layer.self_attn.q_proj._forward_hooks for layer in model.model.layers)


def _interrupt_before_validation(tmp_path, monkeypatch, *, provenance=None, subspaces=None):
    original = __import__("task_vector.training", fromlist=["evaluate_nll"]).evaluate_nll
    def interrupt(*args, **kwargs):
        raise RuntimeError("checkpoint boundary interruption")
    monkeypatch.setattr("task_vector.training.evaluate_nll", interrupt)
    config = training_config(epochs=1)
    route = "ltv" if subspaces is None else "loda"
    if subspaces is not None:
        config["loda"] = {"rho_max": .05}
        config["training"]["weight_decay_loda"] = 0.
    with pytest.raises(RuntimeError, match="checkpoint boundary interruption"):
        train_task(tiny_model(), TinyTokenizer(), samples()[:2], samples()[:2],
                   task="logic", route=route, seed=17, output_dir=tmp_path / "interrupted",
                   config=config, device="cpu", provenance=provenance, subspaces=subspaces)
    monkeypatch.setattr("task_vector.training.evaluate_nll", original)
    config["training"]["resume_from"] = str(tmp_path / "interrupted" / "last.pt")
    return config


@pytest.mark.parametrize("changed", ["base", "same_id_target", "token_encoding", "caller_source"])
def test_resume_rejects_changed_actual_training_sources(tmp_path, monkeypatch, changed):
    provenance = {"request_signature": "request-a", "base_fingerprint": "base-a",
                  "data_manifest_sha256": "data-a", "source_manifest_sha256": "sources-a"} if changed == "caller_source" else None
    config = _interrupt_before_validation(tmp_path, monkeypatch, provenance=provenance)
    model = tiny_model()
    training, tokenizer = samples()[:2], TinyTokenizer()
    if changed == "base":
        with torch.no_grad():
            next(model.parameters()).view(-1)[0].add_(.125)
    elif changed == "same_id_target":
        training[0].target = "False"
    elif changed == "token_encoding":
        class ChangedTokenizer(TinyTokenizer):
            def encode(self, text, **kwargs):
                return super().encode(text, **kwargs) + [3]
        tokenizer = ChangedTokenizer()
    else:
        provenance = {**provenance, "source_manifest_sha256": "sources-b"}
    with pytest.raises(ValueError, match="Resume checkpoint"):
        train_task(model, tokenizer, training, samples()[:2], task="logic", route="ltv", seed=17,
                   output_dir=tmp_path / "interrupted", config=config, device="cpu", provenance=provenance)
    assert not (tmp_path / "interrupted" / "training.json").exists()
    assert not (tmp_path / "interrupted" / "training-state.json").exists()


def test_resume_rejects_changed_loda_subspaces(tmp_path, monkeypatch):
    spaces = {f"model.layers.{layer}": {"A_G": torch.eye(16)[:2] / 3**.5,
              "A_I": torch.eye(16)[2:4] / 3**.5, "lambda": torch.ones(2)} for layer in range(3)}
    config = _interrupt_before_validation(tmp_path, monkeypatch, subspaces=spaces)
    spaces["model.layers.0"]["A_G"][0, 0] += .1
    with pytest.raises(ValueError, match="Resume checkpoint"):
        train_task(tiny_model(), TinyTokenizer(), samples()[:2], samples()[:2],
                   task="logic", route="loda", seed=17, output_dir=tmp_path / "interrupted",
                   config=config, device="cpu", subspaces=spaces)


def test_single_last_checkpoint_restores_initial_and_best_to_new_directory(tmp_path):
    from task_vector.representations import Representation
    config = training_config(epochs=1)
    source = train_task(tiny_model(), TinyTokenizer(), samples()[:2], samples()[:2],
                        task="logic", route="ltv", seed=17, output_dir=tmp_path / "source",
                        config=config, device="cpu")
    portable = tmp_path / "portable"
    portable.mkdir()
    shutil.copyfile(source["last_checkpoint"], portable / "last.pt")
    config["training"]["resume_from"] = str(portable / "last.pt")
    result = train_task(tiny_model(), TinyTokenizer(), samples()[:2], samples()[:2],
                        task="logic", route="ltv", seed=17, output_dir=tmp_path / "restored",
                        config=config, device="cpu")
    assert result["steps"] == source["steps"] == 1
    assert result["best_nll"] == source["best_nll"]
    for field in ("initial_path", "raw_path", "post_path"):
        original, restored = Representation.load(source[field]), Representation.load(result[field])
        for key in original.keys:
            torch.testing.assert_close(original.values[key], restored.values[key], atol=0, rtol=0)
    assert Path(result["last_checkpoint"]).is_file()
    assert result["best_checkpoint"] is None
    assert not result["best_checkpoint_resume_supported"]
    best = torch.load(result["best_adapter_path"], weights_only=True)
    assert best["kind"] == "adapter_only" and best["resume_supported"] is False
    assert not (tmp_path / "restored" / "training.json").exists()


def test_export_failure_retries_without_retraining_or_publishing_completion(tmp_path, monkeypatch):
    import task_vector.training as training_module
    original = training_module.atomic_json
    def fail_export_marker(path, value):
        if Path(path).name == "training-state.json":
            raise OSError("simulate interrupted export finalization")
        original(path, value)
    monkeypatch.setattr(training_module, "atomic_json", fail_export_marker)
    config = training_config(epochs=1)
    with pytest.raises(OSError, match="interrupted export finalization"):
        train_task(tiny_model(), TinyTokenizer(), samples()[:2], samples()[:2],
                   task="logic", route="ltv", seed=17, output_dir=tmp_path,
                   config=config, device="cpu")
    assert not (tmp_path / "training.json").exists()
    assert (tmp_path / "last.pt").is_file()
    monkeypatch.setattr(training_module, "atomic_json", original)
    def forbid_retraining(*args, **kwargs):
        pytest.fail("completed logical steps were retrained during export recovery")
    monkeypatch.setattr(training_module, "accumulated_backward", forbid_retraining)
    config["training"]["resume_from"] = str(tmp_path / "last.pt")
    result = train_task(tiny_model(), TinyTokenizer(), samples()[:2], samples()[:2],
                        task="logic", route="ltv", seed=17, output_dir=tmp_path,
                        config=config, device="cpu")
    assert result["optimizer_steps"] == 1
    assert Path(result["best_checkpoint"]).is_file()
    assert json.loads((tmp_path / "training-state.json").read_text())["status"] == "exported_uncommitted"
    assert not (tmp_path / "training.json").exists()


def test_resume_rejects_corrupt_initial_and_legacy_unbound_checkpoint(tmp_path, monkeypatch):
    config = _interrupt_before_validation(tmp_path, monkeypatch)
    checkpoint = torch.load(config["training"]["resume_from"], weights_only=True)
    for corruption in ("initial", "legacy"):
        changed = copy.deepcopy(checkpoint)
        if corruption == "initial":
            next(iter(changed["initial_representation"]["values"].values()))[0] += 1
        else:
            changed["schema_version"] = 1
        path = tmp_path / f"{corruption}.pt"
        torch.save(changed, path)
        config["training"]["resume_from"] = str(path)
        with pytest.raises(ValueError, match="Resume checkpoint"):
            train_task(tiny_model(), TinyTokenizer(), samples()[:2], samples()[:2],
                       task="logic", route="ltv", seed=17, output_dir=tmp_path / corruption,
                       config=config, device="cpu")


def test_ltv_initialization_bounds_are_effective_and_recorded(tmp_path):
    from task_vector.representations import Representation
    config = training_config(epochs=1)
    config["ltv"].update(init_low=.2, init_high=.3)
    result = train_task(tiny_model(), TinyTokenizer(), samples()[:2], samples()[:2],
                        task="logic", route="ltv", seed=17, output_dir=tmp_path,
                        config=config, device="cpu")
    initial = Representation.load(result["initial_path"])
    assert initial.metadata["init_low"] == .2 and initial.metadata["init_high"] == .3
    for vector in initial.values.values():
        assert bool(((vector >= .2) & (vector < .3)).all())
    with pytest.raises(ValueError, match="initialization bounds"):
        LTVAdapter(tiny_model(), [1], init_low=.1, init_high=.1)


def test_checkpoint_tensor_digest_handles_bfloat16_and_binds_dtype_and_contents():
    from task_vector.training import _content_digest
    values = torch.tensor([[.5, 1.], [2., 3.]], dtype=torch.bfloat16)
    digest = _content_digest(values)
    assert _content_digest(values.clone()) == digest
    assert _content_digest(values.float()) != digest
    assert _content_digest(values.T.contiguous().T) == digest
    values[0, 0] += .5
    assert _content_digest(values) != digest
