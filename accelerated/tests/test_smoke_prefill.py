"""CPU engineering contracts; run directly with Python and the existing torch."""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import torch

from task_vector.losses import collate_encoded
from task_vector.ltv import intervention_context
from task_vector.representations import Representation
from task_vector.smoke import (
    SMOKE_PROTOCOL_VERSION,
    _same_shape_causal_checks,
    _teacher_prefill_comparison,
)


class _ToyBlock(torch.nn.Module):
    def __init__(self, leak_scale=0.0):
        super().__init__()
        self.leak_scale = leak_scale

    def forward(self, hidden_states, attention_mask):
        visible = hidden_states * attention_mask[..., None]
        causal = visible.cumsum(dim=1)
        # A deliberately broken future dependency tests rejection, including
        # differences smaller than the original BF16 allclose tolerance.
        return causal + self.leak_scale * visible.sum(dim=1, keepdim=True)


class _ToyCausal(torch.nn.Module):
    def __init__(self, leak_scale=0.0):
        super().__init__()
        self.config = SimpleNamespace(vocab_size=32)
        self.model = torch.nn.Module()
        self.model.layers = torch.nn.ModuleList([_ToyBlock(leak_scale)])
        self.shapes = []

    def forward(self, input_ids, attention_mask=None, use_cache=False):
        self.shapes.append(tuple(input_ids.shape))
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        values = input_ids.float()
        hidden = torch.stack((values, values.remainder(3), torch.ones_like(values)), dim=-1)
        for layer in self.model.layers:
            hidden = layer(hidden, attention_mask=attention_mask)
        return SimpleNamespace(logits=hidden)


class SameShapeCausalTests(unittest.TestCase):
    def setUp(self):
        examples = [
            {"input_ids": [1, 2, 3, 7, 8], "labels": [-100, -100, -100, 7, 8], "prompt_length": 3},
            {"input_ids": [4, 5, 9], "labels": [-100, -100, 9], "prompt_length": 2},
        ]
        self.batch = collate_encoded(examples, 0)
        self.ids = ["math:train:first", "math:train:second"]

    def check_model(self, model, representation=None, batch=None):
        batch = self.batch if batch is None else batch
        with torch.no_grad(), intervention_context(model, representation, batch["prompt_positions"]):
            teacher = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"], use_cache=False).logits
        return _same_shape_causal_checks(model, batch, representation, teacher,
            pad_token_id=0, condition="BASE" if representation is None else "SAVED_LTV", sample_ids=self.ids)

    def test_base_causal_visibility_and_future_changes_are_exact(self):
        model = _ToyCausal()
        records = self.check_model(model)
        self.assertEqual(model.shapes, [(2, 5), (2, 5), (2, 5)])
        for condition in ("prompt_visibility", "future_tokens"):
            self.assertEqual(len(records[condition]), 2)
            self.assertTrue(all(row["exact_logits"] and row["max_abs_logit_difference"] == 0 for row in records[condition]))

    def test_saved_ltv_retains_the_fixed_prompt_position(self):
        model = _ToyCausal()
        rep = Representation("activation", "math", 17,
            values={"model.layers.0": torch.tensor([0.25, -0.125, 0.5])})
        records = self.check_model(model, rep)
        self.assertTrue(all(row["condition"] == "SAVED_LTV" for row in records["prompt_visibility"]))
        self.assertTrue(all(row["max_abs_logit_difference"] == 0 for row in records["future_tokens"]))

    def test_real_future_leak_is_rejected_at_zero_tolerance(self):
        with self.assertRaises(AssertionError):
            self.check_model(_ToyCausal(leak_scale=0.0001))

    def test_masked_original_prompt_token_is_rejected(self):
        batch = {key: value.clone() for key, value in self.batch.items()}
        batch["attention_mask"][0, 1] = 0
        with self.assertRaisesRegex(ValueError, "Every original prompt token must be visible"):
            self.check_model(_ToyCausal(), batch=batch)


class CrossShapeComparisonTests(unittest.TestCase):
    def compare(self, prefill, teacher, *, bf16):
        return _teacher_prefill_comparison(prefill, teacher,
            atol=0.03125, rtol=0.015625, bf16=bf16)

    def test_bf16_shape_drift_is_reported_without_changing_tolerance(self):
        teacher = torch.tensor([1.0, 3.0, 2.0], dtype=torch.bfloat16)
        prefill = torch.tensor([1.25, 3.25, 2.25], dtype=torch.bfloat16)
        record = self.compare(prefill, teacher, bf16=True)
        self.assertFalse(record["within_original_tolerance"])
        self.assertEqual(record["max_abs_logit_difference"], 0.25)
        self.assertTrue(record["same_argmax"])
        self.assertEqual(record["allclose_role"], "cross_shape_BF16_diagnostic")

    def test_bf16_changed_greedy_decision_is_still_rejected(self):
        teacher = torch.tensor([1.0, 3.0, 2.0], dtype=torch.bfloat16)
        prefill = torch.tensor([4.0, 3.0, 2.0], dtype=torch.bfloat16)
        with self.assertRaisesRegex(AssertionError, "different first answer tokens"):
            self.compare(prefill, teacher, bf16=True)

    def test_nonfinite_logits_are_rejected_in_both_paths(self):
        for bf16, dtype in ((True, torch.bfloat16), (False, torch.float32)):
            for value in (float("nan"), float("inf"), -float("inf")):
                for bad_side in ("prefill", "teacher"):
                    with self.subTest(bf16=bf16, value=value, bad_side=bad_side):
                        good = torch.tensor([1.0, 3.0, 2.0], dtype=dtype)
                        bad = torch.tensor([value, 3.0, 2.0], dtype=dtype)
                        args = (bad, good) if bad_side == "prefill" else (good, bad)
                        with self.assertRaisesRegex(AssertionError, "nonfinite"):
                            self.compare(*args, bf16=bf16)

    def test_non_bf16_retains_original_allclose_hard_gate(self):
        teacher = torch.tensor([1.0, 3.0, 2.0])
        with self.assertRaises(AssertionError):
            self.compare(teacher + 0.25, teacher, bf16=False)
        record = self.compare(teacher + 0.0001, teacher, bf16=False)
        self.assertTrue(record["within_original_tolerance"])
        self.assertEqual(record["allclose_role"], "hard_gate")

    def test_engineering_protocol_is_explicit(self):
        self.assertEqual(SMOKE_PROTOCOL_VERSION, "same_shape_causal_prefill_v2")


if __name__ == "__main__":
    unittest.main(verbosity=2)
