"""Independent stdlib tests for the external, read-only DPC monitor."""
from __future__ import annotations

import ast
from contextlib import redirect_stdout
from datetime import datetime, timezone
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

SCRIPT = Path(__file__).with_name("monitor_task_vector.py")
SPEC = importlib.util.spec_from_file_location("monitor_under_test", SCRIPT)
monitor = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(monitor)


def iso(value):
    return datetime.fromtimestamp(value, timezone.utc).isoformat()


def write_json(path, value, when=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")
    if when is not None:
        os.utime(path, (when, when))


class MonitorTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.cache = monitor.JsonCache()

    def test_training_plan_has_120_distinct_paths_and_reuses_ltv(self):
        plan = monitor.planned_training(self.root)
        paths = [entry[2] for entry in plan]
        self.assertEqual(len(plan), 120)
        self.assertEqual(len(set(paths)), 120)
        self.assertEqual(sum(route == "ltv" for route, _, _ in plan), 18)
        self.assertEqual(sum(route.startswith("ltv-layers-") for route, _, _ in plan), 36)
        self.assertEqual(sum(route == "loda" for route, _, _ in plan), 66)

    def test_pool_selection_does_not_reuse_previous_attempt(self):
        old = self.root / "scheduler/mvp-seed17-old/pool.json"
        new = self.root / "scheduler/mvp-seed17-new/pool.json"
        write_json(old, {"jobs": [{"started_utc": iso(99)}]})
        write_json(new, {"jobs": [{"started_utc": iso(105)}]})
        path, _ = monitor.select_pool(self.root, {"started_utc": iso(100)}, {}, self.cache)
        self.assertEqual(path, new)
        new.unlink()
        path, _ = monitor.select_pool(self.root, {"started_utc": iso(100)}, {}, self.cache)
        self.assertIsNone(path, "Even a prior attempt one second earlier must not be attributed to this worker")

    def test_pool_selection_respects_terminal_worker_end(self):
        future = self.root / "scheduler/mvp-seed17-future/pool.json"
        write_json(future, {"jobs": [{"started_utc": iso(201)}]})
        path, _ = monitor.select_pool(self.root, {"started_utc": iso(100)}, {"ended_utc": iso(200)}, self.cache)
        self.assertIsNone(path)

    def test_pool_unknown_without_worker_started_time(self):
        path = self.root / "scheduler/mvp-seed17-old/pool.json"
        write_json(path, {"jobs": [{"started_utc": iso(100)}]})
        selected, data = monitor.select_pool(self.root, {}, {}, self.cache)
        self.assertIsNone(selected)
        self.assertEqual(data, {})

    def test_selected_worker_remains_within_experiment(self):
        directory = self.root / "runs/platform_jobs/full-one"
        directory.mkdir(parents=True)
        self.assertEqual(monitor.select_job(self.root, "full-one"), directory)
        with self.assertRaises(ValueError):
            monitor.select_job(self.root, str(self.root))

    def test_latest_full_worker_selected(self):
        directory = self.root / "runs/platform_jobs"
        for name in ("full-20261011T000000Z", "check-20261012T000000Z", "full-20261012T000000Z"):
            (directory / name).mkdir(parents=True)
        self.assertEqual(monitor.select_job(self.root).name, "full-20261012T000000Z")

    def test_missing_and_invalid_json_are_recoverable(self):
        path = self.root / "temporary.json"
        self.assertEqual(self.cache.read(path, {}), {})
        path.write_text("{broken", encoding="utf-8")
        self.assertEqual(self.cache.read(path, {}), {})
        self.assertIn(path, self.cache.errors)
        write_json(path, {"recovered": True})
        self.assertEqual(self.cache.read(path, {}), {"recovered": True})
        self.assertNotIn(path, self.cache.errors)

    def test_wrong_json_shapes_do_not_crash_epoch_read(self):
        write_json(self.root / "training.json", [])
        write_json(self.root / "epochs.json", {"wrong": "shape"})
        progress = monitor.epoch_state(self.root, self.cache, 100)
        self.assertFalse(progress["committed"])
        self.assertEqual(progress["epochs"], 0)

    def test_missing_epochs_leave_eta_unknown(self):
        progress = monitor.epoch_state(self.root, self.cache, 100)
        self.assertIsNone(progress["eta"])

    def test_epoch_estimate_uses_initial_file_not_process_start(self):
        (self.root / "initial.pt").write_bytes(b"not-a-checkpoint")
        os.utime(self.root / "initial.pt", (100, 100))
        self.assertIsNone(monitor.epoch_state(self.root, self.cache, 110)["eta"])
        write_json(self.root / "epochs.json", [{"epoch": 1, "tune_answer_nll": 2.0}], 200)
        progress = monitor.epoch_state(self.root, self.cache, 225)
        self.assertEqual(progress["epoch_seconds"], 100)
        self.assertEqual(progress["eta"], 375)

    def test_midrun_monitor_waits_for_another_epoch_observation(self):
        (self.root / "initial.pt").write_bytes(b"not-a-checkpoint")
        os.utime(self.root / "initial.pt", (100, 100))
        write_json(self.root / "epochs.json", [{"epoch": 2}], 200)
        self.assertIsNone(monitor.epoch_state(self.root, self.cache, 210)["eta"])
        write_json(self.root / "epochs.json", [{"epoch": 3}], 300)
        progress = monitor.epoch_state(self.root, self.cache, 325)
        self.assertEqual(progress["epoch_seconds"], 100)
        self.assertEqual(progress["eta"], 175)

    def test_resume_rewritten_initial_resets_epoch_cadence(self):
        (self.root / "initial.pt").write_bytes(b"not-a-checkpoint")
        os.utime(self.root / "initial.pt", (100, 100))
        write_json(self.root / "epochs.json", [{"epoch": 2}], 200)
        monitor.epoch_state(self.root, self.cache, 210)
        write_json(self.root / "epochs.json", [{"epoch": 3}], 300)
        self.assertIsNotNone(monitor.epoch_state(self.root, self.cache, 310)["eta"])
        os.utime(self.root / "initial.pt", (400, 400))
        self.assertIsNone(monitor.epoch_state(self.root, self.cache, 410)["eta"])
        write_json(self.root / "epochs.json", [{"epoch": 4}], 500)
        self.assertIsNone(monitor.epoch_state(self.root, self.cache, 510)["eta"])

    def test_expired_epoch_forecast_becomes_unknown_not_zero(self):
        (self.root / "initial.pt").write_bytes(b"not-a-checkpoint")
        os.utime(self.root / "initial.pt", (100, 100))
        write_json(self.root / "epochs.json", [{"epoch": 1}], 200)
        self.assertIsNone(monitor.epoch_state(self.root, self.cache, 900)["eta"])

    def test_completed_training_has_zero_eta(self):
        write_json(self.root / "training.json", {"status": "complete", "completed_epochs": 3})
        progress = monitor.epoch_state(self.root, self.cache, 999)
        self.assertEqual(progress["eta"], 0)
        self.assertEqual(progress["epochs"], 3)

    def test_first_seed_plan_is_52_test_and_52_discovery(self):
        base = self.root / "runs/evaluation/mvp/17"
        expected = []
        for split, count in (("test", 256), ("discovery", 128)):
            for target in monitor.CORE:
                expected.append((base / split / "canonical/base" / target / "scores.json", count))
            for route in ("ltv", "loda"):
                for source in monitor.CORE:
                    for target in monitor.CORE:
                        expected.append((base / split / "canonical" / f"{route}-{source}-primary" / target / "scores.json", count))
                    for control in ("initial", "random-0"):
                        expected.append((base / split / "canonical" / f"{route}-{source}-{control}" / source / "scores.json", count))
        self.assertEqual(sum(count == 256 for _, count in expected), 52)
        self.assertEqual(sum(count == 128 for _, count in expected), 52)
        for path, count in expected:
            write_json(path, {"signature": "fixture", "count": count, "runtime_seconds": count * 2, "newly_generated": count})
        progress = monitor.first_seed_panel_eta(self.root, self.cache)
        self.assertEqual(progress["done"], 104)
        self.assertEqual(progress["total"], 104)
        self.assertEqual(progress["samples"], 52 * (256 + 128))
        self.assertEqual(progress["eta"], 0)

    def test_panel_eta_unknown_without_target_timing(self):
        progress = monitor.first_seed_panel_eta(self.root, self.cache)
        self.assertIsNone(progress["eta"])
        self.assertEqual(progress["unknown_target_panels"], 104)

    def test_read_only_terminal_gate_and_48_sequence_stages(self):
        job = self.root / "runs/platform_jobs/full-fixture"
        write_json(job / "metadata.json", {"status": "running", "started_utc": iso(100)})
        write_json(job / "result.json", {"status": "stopped_by_gate", "completed": False, "exit_code": 42, "ended_utc": iso(200)})
        for seed in monitor.SEEDS:
            for order in range(4):
                for stage in range(4):
                    write_json(self.root / "runs/sequential" / f"seed{seed}" / f"order{order}" / f"stage{stage}_math/stage-completed.json", {"status": "complete"})
        before = self.tree_hash(self.root)
        stream = io.StringIO()
        with redirect_stdout(stream):
            terminal = monitor.show(self.root, job, self.cache, 300)
        self.assertTrue(terminal)
        self.assertIn("stopped_by_gate", stream.getvalue())
        self.assertIn("exit_code=42", stream.getvalue())
        self.assertIn("48/48", stream.getvalue())
        self.assertEqual(before, self.tree_hash(self.root), "Monitor must not mutate experiment evidence")

    def test_partial_pool_is_not_whole_experiment_eta(self):
        job = self.root / "runs/platform_jobs/full-fixture"
        write_json(job / "metadata.json", {"status": "running", "started_utc": iso(100)})
        write_json(self.root / "scheduler/mvp-seed17-fixture/pool.json", {"status": "running", "jobs": [{"route": "ltv", "task": "math", "status": "running", "started_utc": iso(110)}]})
        stream = io.StringIO()
        with redirect_stdout(stream):
            monitor.show(self.root, job, self.cache, 200)
        self.assertIn("当前并行训练ETA=待估", stream.getvalue())
        self.assertIn("全程ETA=暂不可可靠估计", stream.getvalue())

    def test_tail_reads_only_last_32_kib(self):
        path = self.root / "large.log"
        path.write_text("old\n" * 20000 + "latest failure\n", encoding="utf-8")
        self.assertIn("latest failure", monitor.tail_error(path))
        self.assertLess(len(monitor.tail_error(path)), 1801)

    def test_no_framework_or_subprocess_imports(self):
        names = set()
        for node in ast.walk(ast.parse(SCRIPT.read_text(encoding="utf-8"))):
            if isinstance(node, ast.Import):
                names.update(alias.name.split(".")[0] for alias in node.names)
            elif isinstance(node, ast.ImportFrom):
                names.add((node.module or "").split(".")[0])
        self.assertFalse(names & {"torch", "torch_npu", "subprocess", "task_vector", "yaml"})

    def test_monitor_is_outside_hashed_project(self):
        project = Path(__file__).parents[1] / "task-vector"
        self.assertFalse(SCRIPT.resolve().is_relative_to(project.resolve()))
        self.assertFalse(SCRIPT.name.startswith("run_modelmate"))

    def test_actual_source_manifest_excludes_monitor_and_remains_identical(self):
        project = Path(__file__).parents[1] / "task-vector"
        parsed = ast.parse((project / "src/task_vector/config.py").read_text(encoding="utf-8"))
        function = next(node for node in parsed.body if isinstance(node, ast.FunctionDef) and node.name == "source_manifest")
        namespace = {"Path": Path, "_source_hash": lambda path: hashlib.sha256(path.read_bytes()).hexdigest()}
        exec(compile(ast.Module(body=[function], type_ignores=[]), "actual_source_manifest", "exec"), namespace)
        source_manifest = namespace["source_manifest"]
        before = source_manifest(project)
        self.assertTrue(before)
        self.assertFalse(any("monitor_task_vector" in name for name in before))
        job = self.root / "runs/platform_jobs/full-fixture"
        write_json(job / "metadata.json", {"status": "running", "started_utc": iso(100)})
        with redirect_stdout(io.StringIO()):
            monitor.show(self.root, job, self.cache, 200)
        self.assertEqual(before, source_manifest(project))

    def test_one_missing_worker_snapshot_does_not_sleep(self):
        with patch.object(monitor.time, "sleep", side_effect=AssertionError("must not sleep")), redirect_stdout(io.StringIO()):
            self.assertEqual(monitor.main(["--root", str(self.root), "--count", "1"]), 0)

    @staticmethod
    def tree_hash(root):
        return {str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
                for path in root.rglob("*") if path.is_file()}


if __name__ == "__main__":
    unittest.main(verbosity=2)
