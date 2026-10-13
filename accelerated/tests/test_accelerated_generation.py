"""Numerical continuity and source-only batch policy contracts."""
from types import SimpleNamespace

import pytest
import torch

from task_vector import batching
from task_vector.evaluation import NonFiniteGenerationError, generate_predictions, greedy_generate


class ScriptedModel(torch.nn.Module):
    def __init__(self, plans, *, broken_cache=False):
        super().__init__()
        self.plans = plans
        self.broken_cache = broken_cache
        self.calls = []
        self.generation_config = SimpleNamespace(eos_token_id=[30, 31])

    def forward(self, input_ids, attention_mask, position_ids, past_key_values,
                cache_position, use_cache, logits_to_keep):
        if past_key_values is None:
            keys = input_ids[:, -1].tolist()
            step = 0
        else:
            keys, step = past_key_values
        self.calls.append({"input_ids": input_ids.clone(), "attention_mask": attention_mask.clone(),
                           "position_ids": position_ids.clone(), "cache_position": cache_position.clone()})
        logits = torch.zeros((len(keys), 1, 32))
        for row, key in enumerate(keys):
            plan = self.plans[key]
            token = plan[min(step, len(plan) - 1)]
            logits[row, 0, token] = 1
        return SimpleNamespace(logits=logits,
            past_key_values=None if self.broken_cache else (keys, step + 1))


class Tokenizer:
    pad_token_id = 0
    eos_token_id = 30

    def encode(self, text, **kwargs):
        return [int(char) if char.isdigit() else 7 for char in text]

    def __call__(self, text, **kwargs):
        return {"input_ids": self.encode(text)}

    def decode(self, tokens, **kwargs):
        return ",".join(map(str, (token for token in tokens if token not in {30, 31})))

    def apply_chat_template(self, messages, **kwargs):
        return messages[-1]["content"]


def sample(index, length=1, split="train"):
    return SimpleNamespace(id=f"logic:{index}", task="logic", prompt=str(index) * length,
                           group_id=f"g:{index}", target="4", metadata={}, official_split=split)


def config():
    return {"evaluation": {"generation_batch_size": 1, "max_new_tokens": {"logic": 4}}}


def old_decode(model, ids, mask, maximum):
    """Independent reference for the original row-wise stopping bookkeeping."""
    active = [True] * len(ids)
    rows = [[] for _ in active]
    stopped = [False] * len(active)
    current, cache, width = ids, None, ids.shape[1]
    for step in range(maximum):
        positions = mask.long().cumsum(-1) - 1
        positions.masked_fill_(mask == 0, 0)
        positions = positions[:, -current.shape[1]:]
        cache_pos = torch.arange(0 if step == 0 else width + step - 1,
                                 width if step == 0 else width + step)
        out = model(current, mask, positions, cache, cache_pos, True, 1)
        cache = out.past_key_values
        tokens = out.logits[:, -1].argmax(-1)
        for row in range(len(active)):
            if active[row]:
                token = int(tokens[row])
                rows[row].append(token)
                stopped[row] = token in {30, 31}
                active[row] = not stopped[row]
        if not any(active):
            break
        active_tensor = torch.tensor(active)
        current = torch.where(active_tensor, tokens, torch.zeros_like(tokens))[:, None]
        mask = torch.cat((mask, active_tensor.long()[:, None]), -1)
    return rows, stopped


@pytest.mark.parametrize("maximum", [1, 2, 4, 6])
def test_buffered_decode_preserves_original_tokens_stops_and_model_inputs(maximum):
    plans = {1: [30, 9], 2: [8, 31, 7], 3: [5, 6, 7, 8, 9, 4]}
    ids = torch.tensor([[0, 0, 1], [0, 2, 2], [3, 3, 3]])
    mask = ids.ne(0).long()
    old, new = ScriptedModel(plans), ScriptedModel(plans)
    expected = old_decode(old, ids, mask, maximum)
    actual = greedy_generate(new, ids, mask, max_new_tokens=maximum,
                             eos_token_ids=[30, 31], pad_token_id=0)
    assert actual == expected
    assert len(new.calls) == len(old.calls)
    for left, right in zip(new.calls, old.calls):
        for key in left:
            assert torch.equal(left[key], right[key])


def test_prefill_reset_even_on_model_failure():
    marks = []
    intervention = SimpleNamespace(set_positions=lambda x: None, set_prefill=marks.append)
    with pytest.raises(RuntimeError, match="KV cache"):
        greedy_generate(ScriptedModel({1: [30]}, broken_cache=True), torch.tensor([[1]]),
                        torch.ones((1, 1), dtype=torch.long), max_new_tokens=2,
                        intervention=intervention, eos_token_ids=[30], pad_token_id=0)
    assert marks == [True, None]


def test_batch_override_preserves_row_order_and_restores_training_state():
    model = ScriptedModel({1: [30], 2: [4, 31], 3: [5, 6, 7, 8]})
    model.train()
    events = []
    samples = [sample(1), sample(2, 2), sample(3, 3)]
    rows = generate_predictions(model, Tokenizer(), samples, None, config(), "cpu",
                                batch_size_override=2, progress_callback=events.append)
    assert [row["id"] for row in rows] == [item.id for item in samples]
    assert [row["prediction_token_ids"] for row in rows] == [[30], [4, 31], [5, 6, 7, 8]]
    assert [row["stop_reason"] for row in rows] == ["eos", "eos", "length"]
    assert model.training
    assert events[-1]["completed_samples"] == 3
    assert config()["evaluation"]["generation_batch_size"] == 1


def test_decode_progress_does_not_need_extra_tensor_reads():
    events = []
    greedy_generate(ScriptedModel({1: [4]}), torch.tensor([[1]]), torch.ones((1, 1), dtype=torch.long),
                    max_new_tokens=33, eos_token_ids=[30], pad_token_id=0, progress_callback=events.append)
    assert [event["generated_steps"] for event in events] == [16, 32]


def row(item, token=4):
    return {"id": item.id, "prediction_token_ids": [token, 30], "stop_reason": "eos",
            "truncated": False, "generated_tokens": 2, "max_new_tokens": 4}


def test_source_calibration_never_releases_larger_batch_under_strict_policy(monkeypatch):
    calls = []
    clock = [0.0]

    def fake(*args, batch_size_override, **kwargs):
        assert args[3] is None  # BASE, never an intervention-specific policy.
        calls.append(batch_size_override)
        clock[0] += 8 / batch_size_override
        return [row(item) for item in args[2]]

    monkeypatch.setattr(batching, "generate_predictions", fake)
    monkeypatch.setattr(batching.time, "perf_counter", lambda: clock[0])
    source = [sample(index, index) for index in range(1, 9)]
    receipt = batching.calibrate_generation_batches(None, Tokenizer(), source, config(), "cpu", task="logic")
    assert calls == [1, 2, 4, 8]
    assert receipt["effective_batch_size"] == 1
    assert receipt["fastest_matching_batch_size"] == 8
    assert receipt["all_panel_token_equality_guaranteed"]
    assert not receipt["selection_uses_task_scores"]


def test_validated_calibration_accepts_finite_token_differences_but_rejects_oom(monkeypatch):
    clock = [0.0]

    def fake(*args, batch_size_override, **kwargs):
        clock[0] += 8 / batch_size_override
        if batch_size_override == 8:
            raise RuntimeError("NPU out of memory")
        return [row(item, 9 if batch_size_override == 4 else 4) for item in args[2]]

    monkeypatch.setattr(batching, "generate_predictions", fake)
    monkeypatch.setattr(batching.time, "perf_counter", lambda: clock[0])
    receipt = batching.calibrate_generation_batches(None, Tokenizer(), [sample(i) for i in range(1, 9)],
        config(), "cpu", task="logic", mode="validated")
    assert receipt["effective_batch_size"] == 4
    assert receipt["fastest_matching_batch_size"] == 2
    assert receipt["fastest_valid_batch_size"] == 4
    assert not receipt["token_equality_required_for_selection"]
    assert receipt["normal_finite_bf16_token_differences_permitted"]
    assert receipt["candidates"][2]["token_difference_samples"] == 8
    assert not receipt["all_panel_token_equality_guaranteed"]
    assert receipt["candidates"][-1]["status"] == "out_of_memory"


def test_validated_accepts_legal_eos_length_difference_without_changing_budgets(monkeypatch):
    clock = [0.0]

    def fake(*args, batch_size_override, **kwargs):
        assert kwargs["check_finite_logits"]
        clock[0] += 8 / batch_size_override
        rows = [row(item) for item in args[2]]
        if batch_size_override > 1:
            for result in rows:
                result.update(prediction_token_ids=[4, 5, 6, 7], generated_tokens=4,
                              stop_reason="length", truncated=True)
        return rows

    monkeypatch.setattr(batching, "generate_predictions", fake)
    monkeypatch.setattr(batching.time, "perf_counter", lambda: clock[0])
    receipt = batching.calibrate_generation_batches(None, Tokenizer(), [sample(i) for i in range(1, 9)],
        config(), "cpu", task="logic", mode="validated")
    assert receipt["effective_batch_size"] == 8
    assert receipt["candidates"][-1]["stop_difference_samples"] == 8
    assert receipt["candidates"][-1]["generation_contract_valid"]


@pytest.mark.parametrize("failure", ["coverage", "budget", "stop", "nonfinite"])
def test_validated_rejects_invalid_engineering_candidates(monkeypatch, failure):
    clock = [0.0]

    def fake(*args, batch_size_override, **kwargs):
        clock[0] += 8 / batch_size_override
        if batch_size_override > 1 and failure == "nonfinite":
            raise NonFiniteGenerationError("Active generation logits contain NaN or Inf")
        rows = [row(item) for item in args[2]]
        if batch_size_override > 1:
            if failure == "coverage":
                rows.pop()
            elif failure == "budget":
                rows[0]["max_new_tokens"] = 3
            elif failure == "stop":
                rows[0]["stop_reason"] = "length"
        return rows

    monkeypatch.setattr(batching, "generate_predictions", fake)
    monkeypatch.setattr(batching.time, "perf_counter", lambda: clock[0])
    receipt = batching.calibrate_generation_batches(None, Tokenizer(), [sample(i) for i in range(1, 9)],
        config(), "cpu", task="logic", mode="validated")
    assert receipt["effective_batch_size"] == 1
    expected = "nonfinite_logits" if failure == "nonfinite" else "invalid_generation_contract"
    assert all(record["status"] == expected for record in receipt["candidates"][1:])


def test_audited_calibration_still_requires_reference_token_equality(monkeypatch):
    clock = [0.0]

    def fake(*args, batch_size_override, **kwargs):
        clock[0] += 8 / batch_size_override
        return [row(item, 9 if batch_size_override > 1 else 4) for item in args[2]]

    monkeypatch.setattr(batching, "generate_predictions", fake)
    monkeypatch.setattr(batching.time, "perf_counter", lambda: clock[0])
    receipt = batching.calibrate_generation_batches(None, Tokenizer(), [sample(i) for i in range(1, 9)],
        config(), "cpu", task="logic", mode="audited")
    assert receipt["effective_batch_size"] == 1
    assert receipt["token_equality_required_for_selection"]


@pytest.mark.parametrize("bad_value", [float("nan"), float("inf")])
def test_active_nonfinite_logits_are_rejected(bad_value):
    class BadModel(ScriptedModel):
        def forward(self, *args, **kwargs):
            result = super().forward(*args, **kwargs)
            result.logits[0, -1, 0] = bad_value
            return result

    with pytest.raises(NonFiniteGenerationError, match="NaN or Inf"):
        greedy_generate(BadModel({1: [4]}), torch.tensor([[1]]), torch.ones((1, 1), dtype=torch.long),
                        max_new_tokens=2, eos_token_ids=[30], pad_token_id=0, check_finite_logits=True)


def test_inactive_rows_nonfinite_logits_do_not_invalidate_returned_tokens():
    class InactiveBadModel(ScriptedModel):
        def forward(self, *args, **kwargs):
            result = super().forward(*args, **kwargs)
            if len(self.calls) > 1:
                result.logits[0, -1, 0] = float("nan")
            return result

    tokens, stopped = greedy_generate(InactiveBadModel({1: [30], 2: [4, 30]}),
        torch.tensor([[1], [2]]), torch.ones((2, 1), dtype=torch.long), max_new_tokens=4,
        eos_token_ids=[30], pad_token_id=0, check_finite_logits=True)
    assert tokens == [[30], [4, 30]] and stopped == [True, True]


@pytest.mark.parametrize("source_split", ["test", "discovery", "test_formal"])
def test_calibration_rejects_heldout_split_before_generation(source_split):
    with pytest.raises(ValueError, match="train or tune"):
        batching.calibrate_generation_batches(None, None, [], config(), "cpu", task="logic",
                                              source_split=source_split)


def test_calibration_rejects_disguised_official_test_records():
    with pytest.raises(ValueError, match="Held-out"):
        batching.calibrate_generation_batches(None, Tokenizer(), [sample(1, split="test")],
                                              config(), "cpu", task="logic")


@pytest.mark.parametrize("failure", ["mismatch", "oom", "nonfinite", "match"])
def test_audited_panel_returns_reference_equivalent_rows(monkeypatch, failure):
    calls = []

    def fake(*args, batch_size_override, **kwargs):
        calls.append(batch_size_override)
        if failure == "oom" and batch_size_override > 1:
            raise RuntimeError("out of memory")
        if failure == "nonfinite" and batch_size_override > 1:
            raise NonFiniteGenerationError("Active generation logits contain NaN or Inf")
        token = 9 if failure == "mismatch" and batch_size_override > 1 else 4
        return [row(item, token) for item in args[2]]

    monkeypatch.setattr(batching, "generate_predictions", fake)
    output, receipt = batching.generate_with_batch_policy(None, Tokenizer(), [sample(1)], None, config(), "cpu",
        policy={"mode": "audited", "task": "logic", "effective_batch_size": 4})
    assert output == [row(sample(1))]
    assert calls == [4, 1]
    assert receipt["all_panel_token_equality_guaranteed"]
    assert receipt["fallback"] == {"mismatch": "token_or_stop_mismatch", "oom": "out_of_memory", "nonfinite": "nonfinite_logits",
                                   "match": None}[failure]


def test_unrelated_runtime_errors_are_not_hidden(monkeypatch):
    def broken(*args, **kwargs):
        raise RuntimeError("incorrect adapter shape")

    monkeypatch.setattr(batching, "generate_predictions", broken)
    with pytest.raises(RuntimeError, match="adapter shape"):
        batching.generate_with_batch_policy(None, None, [sample(1)], None, config(), "cpu",
                                            policy={"mode": "validated", "effective_batch_size": 4})


def test_frozen_policy_cannot_change_target_or_variant():
    with pytest.raises(ValueError, match="target task"):
        batching.generate_with_batch_policy(None, None, [sample(1)], None, config(), "cpu",
                                            policy={"task": "math"})
    with pytest.raises(ValueError, match="prompt variant"):
        batching.generate_with_batch_policy(None, None, [], None, config(), "cpu",
                                            policy={"variant": "output_format"})
