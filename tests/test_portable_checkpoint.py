"""Portable checkpoint contracts; run directly with the existing torch.

Set TV_TEST_NPU=1 to also run the small real-NPU AdamW roundtrip.
"""
from __future__ import annotations

from collections import OrderedDict
import os
from pathlib import Path
import random
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import torch

from task_vector.io import atomic_torch_save


def assert_tree_equal(case, expected, observed):
    if isinstance(expected, torch.Tensor):
        case.assertIsInstance(observed, torch.Tensor)
        case.assertEqual(observed.device.type, "cpu")
        case.assertEqual(observed.dtype, expected.dtype)
        case.assertEqual(observed.shape, expected.shape)
        case.assertFalse(observed.requires_grad)
        torch.testing.assert_close(observed, expected.detach().cpu(), atol=0, rtol=0)
    elif isinstance(expected, dict):
        case.assertIs(type(observed), type(expected))
        case.assertEqual(list(observed), list(expected))
        for key in expected:
            case.assertIs(type(next(k for k in observed if k == key)), type(key))
            assert_tree_equal(case, expected[key], observed[key])
        if hasattr(expected, "_metadata"):
            case.assertTrue(hasattr(observed, "_metadata"))
            assert_tree_equal(case, expected._metadata, observed._metadata)
    elif isinstance(expected, (list, tuple)):
        case.assertIs(type(observed), type(expected))
        case.assertEqual(len(observed), len(expected))
        for left, right in zip(expected, observed):
            assert_tree_equal(case, left, right)
    else:
        case.assertEqual(observed, expected)


def adamw_step(parameter, optimizer):
    # A fixed gradient isolates checkpoint/resume from RNG and model kernels.
    parameter.grad = torch.full_like(parameter, 0.125)
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)


class PortableCheckpointTests(unittest.TestCase):
    def test_nested_optimizer_roundtrip_and_continuation(self):
        parameter = torch.nn.Parameter(torch.tensor([[1.0, -2.0], [3.0, 0.5]]))
        optimizer = torch.optim.AdamW([parameter], lr=0.003, foreach=False)
        adamw_step(parameter, optimizer)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: 1.0)
        state = {
            "adapter": OrderedDict(weight=parameter),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "rng": {"python": random.getstate(), "torch": torch.get_rng_state()},
            "nested": [{7: (torch.tensor([1.25, -0.5], dtype=torch.bfloat16), None)}],
        }
        state["adapter"]._metadata = OrderedDict([("", {"version": 1})])
        rng_before, python_before = torch.get_rng_state().clone(), random.getstate()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "state.pt"
            atomic_torch_save(path, state)
            loaded = torch.load(path, map_location="cpu", weights_only=True)
            assert_tree_equal(self, state, loaded)
            self.assertEqual(torch.serialization.get_unsafe_globals_in_checkpoint(path), [])
        self.assertTrue(torch.equal(torch.get_rng_state(), rng_before))
        self.assertEqual(random.getstate(), python_before)
        self.assertTrue(parameter.requires_grad)
        self.assertTrue(all(type(key) is int for key in loaded["optimizer"]["state"]))

        restored = torch.nn.Parameter(loaded["adapter"]["weight"].clone())
        restored_optimizer = torch.optim.AdamW([restored], lr=0.003, foreach=False)
        restored_optimizer.load_state_dict(loaded["optimizer"])
        adamw_step(parameter, optimizer)
        adamw_step(restored, restored_optimizer)
        torch.testing.assert_close(restored, parameter, atol=0, rtol=0)
        assert_tree_equal(self, optimizer.state_dict(), restored_optimizer.state_dict())

    def test_snapshot_is_detached_independent_and_preserves_metadata(self):
        original = torch.tensor([2.0, 4.0], requires_grad=True)
        state = OrderedDict([(3, {"values": [original], "rng": (5, original)})])
        state._metadata = {"nested_tensor": original, "version": 1}
        original_pointer = original.data_ptr()
        with tempfile.TemporaryDirectory() as directory:
            with patch("torch.save", wraps=torch.save) as save:
                atomic_torch_save(Path(directory) / "state.pt", state)
            snapshot = save.call_args.args[0]
        assert_tree_equal(self, state, snapshot)
        self.assertIsNot(snapshot, state)
        self.assertIsNot(snapshot[3], state[3])
        self.assertIsNot(snapshot[3]["values"], state[3]["values"])
        self.assertNotEqual(snapshot[3]["values"][0].data_ptr(), original_pointer)
        self.assertNotEqual(snapshot._metadata["nested_tensor"].data_ptr(), original_pointer)
        snapshot[3]["values"][0].add_(100)
        torch.testing.assert_close(original, torch.tensor([2.0, 4.0]), atol=0, rtol=0)
        self.assertEqual(original.data_ptr(), original_pointer)
        self.assertTrue(original.requires_grad)
        self.assertIsNone(original.grad)

    def test_tensor_copy_failure_preserves_existing_checkpoint(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "state.pt"
            atomic_torch_save(path, {"value": torch.tensor([1.0])})
            before = path.read_bytes()
            with patch.object(torch.Tensor, "cpu", side_effect=RuntimeError("copy failed")):
                with self.assertRaisesRegex(RuntimeError, "copy failed"):
                    atomic_torch_save(path, {"value": torch.tensor([2.0])})
            self.assertEqual(path.read_bytes(), before)

    @unittest.skipUnless(os.environ.get("TV_TEST_NPU") == "1", "set TV_TEST_NPU=1 for real NPU")
    def test_real_npu_adamw_portable_cpu_load_and_resume(self):
        import torch_npu  # noqa: F401: registers the NPU backend and serialization

        self.assertTrue(torch.npu.is_available(), "TV_TEST_NPU=1 requires an available NPU")
        device = torch.device("npu", torch.npu.current_device())
        parameter = torch.nn.Parameter(torch.tensor([[1.0, -2.0], [3.0, 0.5]], device=device))
        optimizer = torch.optim.AdamW([parameter], lr=0.003, foreach=False)
        adamw_step(parameter, optimizer)
        live_moment = optimizer.state[parameter]["exp_avg"]
        live_pointer, live_device = live_moment.data_ptr(), live_moment.device
        rng_cpu, rng_npu = torch.get_rng_state().clone(), torch.npu.get_rng_state().clone()
        state = {"adapter": {"weight": parameter}, "optimizer": optimizer.state_dict(),
                 "rng": {"torch": rng_cpu, "npu": {"device_index": device.index, "state": rng_npu}},
                 "bf16": torch.tensor([1.25, -0.5], dtype=torch.bfloat16, device=device)}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "npu.pt"
            atomic_torch_save(path, state)
            loaded = torch.load(path, map_location="cpu", weights_only=True)
            assert_tree_equal(self, state, loaded)
            self.assertEqual(torch.serialization.get_unsafe_globals_in_checkpoint(path), [])
        self.assertEqual(live_moment.device, live_device)
        self.assertEqual(live_moment.data_ptr(), live_pointer)
        self.assertEqual(live_moment.device.type, "npu")
        self.assertTrue(parameter.requires_grad)
        self.assertTrue(torch.equal(torch.get_rng_state(), rng_cpu))
        self.assertTrue(torch.equal(torch.npu.get_rng_state(), rng_npu))

        restored = torch.nn.Parameter(loaded["adapter"]["weight"].to(device))
        restored_optimizer = torch.optim.AdamW([restored], lr=0.003, foreach=False)
        restored_optimizer.load_state_dict(loaded["optimizer"])
        self.assertEqual(restored_optimizer.state[restored]["exp_avg"].device, device)
        self.assertEqual(restored_optimizer.state[restored]["exp_avg_sq"].device, device)
        adamw_step(parameter, optimizer)
        adamw_step(restored, restored_optimizer)
        torch.testing.assert_close(restored, parameter, atol=0, rtol=0)
        assert_tree_equal(self, optimizer.state_dict(), {
            "state": {key: {name: value.detach().cpu() if isinstance(value, torch.Tensor) else value
                            for name, value in moment.items()}
                      for key, moment in restored_optimizer.state_dict()["state"].items()},
            "param_groups": restored_optimizer.state_dict()["param_groups"],
        })


if __name__ == "__main__":
    unittest.main(verbosity=2)
