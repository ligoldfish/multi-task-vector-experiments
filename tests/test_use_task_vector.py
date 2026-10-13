"""CPU equivalence checks against the repository's intervention implementation."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest

import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "portable"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from use_task_vector import load_plain, task_vector_context, verify_base_directory
from task_vector.ltv import intervention_context
from task_vector.representations import LowRankFactor, Representation


class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.norm = nn.RMSNorm(4)
        self.linear = nn.Linear(4, 4)

    def forward(self, hidden_states):
        return self.linear(self.norm(hidden_states)) + hidden_states


class ActivationModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Module()
        self.model.layers = nn.ModuleList([Block(), Block()])

    def forward(self, hidden_states, past_key_values=None, cache_position=None):
        for block in self.model.layers:
            hidden_states = block(hidden_states=hidden_states)
        return hidden_states


class ProjectionModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Module()
        self.model.layers = nn.ModuleList([nn.Module()])
        attention = nn.Module()
        self.model.layers[0].self_attn = attention
        for kind, output in (("q_proj", 6), ("k_proj", 3), ("v_proj", 3)):
            setattr(attention, kind, nn.Linear(4, output))

    def forward(self, hidden_states):
        attention = self.model.layers[0].self_attn
        return torch.cat([getattr(attention, kind)(hidden_states) for kind in ("q_proj", "k_proj", "v_proj")], dim=-1)


class OccupiedCache:
    def get_seq_length(self):
        return 5


class EquivalenceTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7021)

    def test_ltv_prefill_matches_before_rmsnorm_exactly_in_both_dtypes(self):
        for dtype in (torch.float32, torch.bfloat16):
            with self.subTest(dtype=dtype):
                model = ActivationModel().to(dtype)
                other = deepcopy(model)
                hidden = torch.randn(2, 5, 4).to(dtype)
                positions = torch.tensor([4, 2])
                rep = Representation("activation", "math", 17, values={"model.layers.0": torch.randn(4), "model.layers.1": torch.randn(4)})
                baseline = model(hidden)
                with intervention_context(model, rep, positions):
                    expected = model(hidden)
                with task_vector_context(other, rep.payload(), positions):
                    actual = other(hidden)
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)
                torch.testing.assert_close(other(hidden), baseline, rtol=0, atol=0)
                self.assertFalse(torch.equal(expected, baseline))

    def test_ltv_cache_detection_and_explicit_prefill_override(self):
        model = ActivationModel()
        other = deepcopy(model)
        hidden = torch.randn(2, 5, 4)
        positions = torch.tensor([4, 4])
        rep = Representation("activation", "logic", 17, values={"model.layers.0": torch.randn(4)})
        legacy_cache = ((torch.zeros(2, 1, 3, 4), torch.zeros(2, 1, 3, 4)),)
        arguments = ({}, {"cache_position": torch.tensor([5])}, {"past_key_values": OccupiedCache()}, {"past_key_values": legacy_cache})
        for kwargs in arguments:
            for force in (None, True, False):
                with self.subTest(kwargs=tuple(kwargs), force=force):
                    with intervention_context(model, rep, positions) as old:
                        old.set_prefill(force)
                        expected = model(hidden, **kwargs)
                    with task_vector_context(other, rep.payload(), positions) as new:
                        new.set_prefill(force)
                        actual = other(hidden, **kwargs)
                    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        with task_vector_context(other, rep.payload(), positions) as new:
            # A one-token decode with an occupied cache skips the original position.
            actual = other(hidden[:, :1], cache_position=torch.tensor([5]))
        torch.testing.assert_close(actual, other(hidden[:, :1]), rtol=0, atol=0)

    def test_ltv_requires_positions_and_restores_after_exception(self):
        model = ActivationModel()
        rep = Representation("activation", "math", 17, values={"model.layers.0": torch.randn(4)})
        with self.assertRaises(RuntimeError):
            with task_vector_context(model, rep.payload()):
                model(torch.randn(1, 5, 4))
        self.assertEqual(len(model._forward_pre_hooks), 0)
        self.assertEqual(len(model.model.layers[0]._forward_pre_hooks), 0)

    def test_loda_fp32_updates_match_original_exactly_with_cpu_autocast(self):
        for dtype in (torch.float32, torch.bfloat16):
            for autocast in (False, True):
                with self.subTest(dtype=dtype, autocast=autocast):
                    model = ProjectionModel().to(dtype)
                    other = deepcopy(model)
                    hidden = torch.randn(2, 5, 4).to(dtype)
                    terms = {}
                    originals = {}
                    for kind, output in (("q_proj", 6), ("k_proj", 3), ("v_proj", 3)):
                        key = "model.layers.0.self_attn." + kind
                        terms[key] = [LowRankFactor(torch.randn(output, rank), torch.randn(rank, 4)) for rank in (2, 1)]
                        originals[key] = other.get_submodule(key)
                    rep = Representation("parameter", "math", 17, factors=terms)
                    with torch.autocast("cpu", dtype=torch.bfloat16, enabled=autocast):
                        with intervention_context(model, rep):
                            expected = model(hidden)
                        with task_vector_context(other, rep.payload()):
                            actual = other(hidden)
                    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
                    for key, original in originals.items():
                        self.assertIs(other.get_submodule(key), original)

    def test_loda_partial_install_failure_restores_originals(self):
        model = ProjectionModel()
        key = "model.layers.0.self_attn.q_proj"
        original = model.get_submodule(key)
        rep = Representation("parameter", "math", 17, factors={
            key: [LowRankFactor(torch.randn(6, 2), torch.randn(2, 4))],
            "model.layers.9.self_attn.q_proj": [LowRankFactor(torch.randn(6, 2), torch.randn(2, 4))]})
        with self.assertRaises(AttributeError):
            with task_vector_context(model, rep.payload()):
                pass
        self.assertIs(model.get_submodule(key), original)

    def test_plain_load_uses_cpu_tensors(self):
        rep = Representation("activation", "math", 17, values={"model.layers.0": torch.randn(4)})
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "post.pt"
            torch.save(rep.payload(), path)
            payload = load_plain(path)
            self.assertEqual(payload["values"]["model.layers.0"].device.type, "cpu")
            with task_vector_context(ActivationModel(), path, torch.tensor([4])):
                pass

    def test_base_contract_checks_exact_names_contents_and_fingerprint(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "config.json").write_bytes(b'{"hidden_size":4}')
            (root / "model.safetensors").write_bytes(b"fake-weight-bytes")
            files = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in root.iterdir()}
            digest = hashlib.sha256(json.dumps(files, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
            contract = {"files": files, "fingerprint": digest}
            result = verify_base_directory(root, contract)
            self.assertTrue(result["verified"])
            self.assertEqual(result["fingerprint"], digest)
            (root / "extra.txt").write_bytes(b"extra")
            with self.assertRaisesRegex(ValueError, "file set differs"):
                verify_base_directory(root, contract)
            # Rename this test-created file outside the recognized suffix set.
            (root / "extra.txt").rename(root / "extra.md")
            (root / "model.safetensors").write_bytes(b"different")
            with self.assertRaisesRegex(ValueError, "content differs"):
                verify_base_directory(root, contract)
            with self.assertRaisesRegex(ValueError, "Invalid"):
                verify_base_directory(root, {"files": {"../config.json": "x"}})


if __name__ == "__main__":
    unittest.main(verbosity=2)
