"""Stdlib-only platform-adapter contracts; no model/runtime/training imports."""

import ast
from contextlib import contextmanager, redirect_stderr, redirect_stdout
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import shutil
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock


HERE = Path(__file__).resolve().parent
REPOSITORY = HERE.parent / "task-vector"
WRAPPER = HERE / "run_task_vector_full_modelmate.py"
SPEC = importlib.util.spec_from_file_location("platform_entry_under_test", WRAPPER)
adapter = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(adapter)


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def genuine_manifest(root):
    """Execute only the current project's actual manifest function AST."""
    source = REPOSITORY / "src" / "task_vector" / "config.py"
    tree = ast.parse(source.read_text(encoding="utf-8-sig"))
    function = next(node for node in tree.body
                    if isinstance(node, ast.FunctionDef) and node.name == "source_manifest")
    selected = ast.fix_missing_locations(ast.Module(body=[function], type_ignores=[]))
    namespace = {"Path": Path, "_source_hash": sha256}
    exec(compile(selected, str(source), "exec"), namespace)
    return namespace["source_manifest"](root)


@contextmanager
def isolated_ops_modules(replacement=None):
    keys = [name for name in sys.modules if name == "ops" or name.startswith("ops.")]
    saved = {name: sys.modules.pop(name) for name in keys}
    if replacement is not None:
        sys.modules["ops.modelmate_job"] = replacement
    try:
        yield
    finally:
        for name in list(sys.modules):
            if name == "ops" or name.startswith("ops."):
                sys.modules.pop(name)
        sys.modules.update(saved)


class Fixture(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.app = Path(self.temporary.name) / "relocated-app"
        self.project = self.app / adapter.PROJECT_DIRECTORY
        for name in ("src", "configs", "deploy", "third_party", "ops", "docs"):
            (self.project / name).mkdir(parents=True, exist_ok=True)
        self.experiment = Path(self.temporary.name) / "persistent-experiment"
        (self.experiment / "runs").mkdir(parents=True)
        shutil.copyfile(REPOSITORY / "run_modelmate_full.py",
                        self.project / "run_modelmate_full.py")
        (self.project / "ops" / "__init__.py").write_text("", encoding="utf-8")
        (self.project / "ops" / "modelmate_job.py").write_text("# fixture only\n", encoding="utf-8")
        self.settings_path = self.project / "ops" / "settings.local.json"
        self.settings_path.write_text(json.dumps(adapter.EXPECTED_SETTINGS), encoding="utf-8")
        (self.project / "configs" / "modelmate.local.yaml").write_text("fixture: true\n", encoding="utf-8")
        self.smoke_path = self.experiment / "runs" / "smoke.json"
        self.publish_smoke()

    def publish_smoke(self):
        self.smoke = {
            "passed": True,
            "source_manifest_sha256": adapter.object_sha256(adapter.source_manifest(self.project)),
            "checks": {
                "loda_true_gao_microbatch_step": True,
                "loda_checkpoint_resume_exact": True,
                "loda_resume_does_not_repeat_step": True,
            },
        }
        self.smoke_path.write_text(json.dumps(self.smoke), encoding="utf-8")

    def inspect(self):
        return adapter.inspect_project(self.project, self.experiment)


class ParserTests(unittest.TestCase):
    def test_known_metadata_aliases_separated_equal_and_empty(self):
        examples = [[], ["--checkpoint_url=", "--data_url=s3://platform/data"],
                    ["--checkpoint-url", "s3://platform/checkpoint", "--data-url", ""],
                    ["--checkpoint_url", "--data_url"],
                    ["--verify-only", "--checkpoint-url=", "--data-url=x"]]
        for arguments in examples:
            with self.subTest(arguments=arguments):
                adapter.parse_arguments(arguments)

    def test_unknown_abbreviated_positional_and_science_overrides_rejected(self):
        for arguments in (["--unknown=1"], ["--check=x"], ["--data=x"], ["--verify"],
                          ["--train_url=x"], ["--stage=formal"], ["--devices=1"],
                          ["positional"], ["--checkpoint_url", "x", "extra"]):
            with self.subTest(arguments=arguments), redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit) as caught:
                    adapter.parse_arguments(arguments)
                self.assertEqual(caught.exception.code, 2)


class VerificationTests(Fixture):
    def test_valid_saved_evidence_passes_without_changing_any_fixture_bytes(self):
        before = {str(path): path.read_bytes() for path in self.app.rglob("*") if path.is_file()}
        smoke_before = self.smoke_path.read_bytes()
        result = self.inspect()
        self.assertIsInstance(result, dict)
        self.assertEqual(before, {str(path): path.read_bytes() for path in self.app.rglob("*")
                                  if path.is_file()})
        self.assertEqual(smoke_before, self.smoke_path.read_bytes())

    def test_current_manifest_matches_real_project_function(self):
        self.assertEqual(adapter.source_manifest(REPOSITORY), genuine_manifest(REPOSITORY))

    def test_outer_adapter_and_pycache_excluded_inner_source_change_detected(self):
        baseline = adapter.source_manifest(self.project)
        (self.app / WRAPPER.name).write_text("# outer adapter\n", encoding="utf-8")
        cache = self.project / "ops" / "__pycache__"
        cache.mkdir()
        (cache / "fixture.pyc").write_bytes(b"ignored cache")
        self.assertEqual(baseline, adapter.source_manifest(self.project))
        self.inspect()
        (self.project / "ops" / "unexpected.py").write_text("# changed source\n", encoding="utf-8")
        self.assertNotEqual(baseline, adapter.source_manifest(self.project))
        with self.assertRaises((ValueError, RuntimeError)):
            self.inspect()

    def test_failed_or_incomplete_gao_smoke_rejected(self):
        mutations = [("passed", False)] + [(name, False) for name in self.smoke["checks"]]
        for name, value in mutations:
            with self.subTest(name=name):
                self.publish_smoke()
                if name == "passed":
                    self.smoke[name] = value
                else:
                    self.smoke["checks"][name] = value
                self.smoke_path.write_text(json.dumps(self.smoke), encoding="utf-8")
                with self.assertRaises((ValueError, RuntimeError)):
                    self.inspect()

    def test_settings_change_and_changed_entry_rejected(self):
        original = self.settings_path.read_bytes()
        changed = dict(adapter.EXPECTED_SETTINGS, expected_devices=1)
        self.settings_path.write_text(json.dumps(changed), encoding="utf-8")
        with self.assertRaises((ValueError, RuntimeError)):
            self.inspect()
        self.settings_path.write_bytes(original)
        with (self.project / "run_modelmate_full.py").open("a", encoding="utf-8") as handle:
            handle.write("\n# unexpected entry modification\n")
        with self.assertRaises((ValueError, RuntimeError)):
            self.inspect()

    def test_verify_only_writes_nothing_and_cannot_launch(self):
        record = self.inspect()
        before = {str(path): path.read_bytes() for path in Path(self.temporary.name).rglob("*")
                  if path.is_file()}
        with mock.patch.object(adapter, "inspect_project", return_value=record), \
             mock.patch.object(adapter, "EXPERIMENT_ROOT", self.experiment), \
             mock.patch.object(adapter, "run_original_entry", side_effect=AssertionError("would launch")) as launch, \
             redirect_stdout(io.StringIO()):
            self.assertEqual(adapter.main(["--verify-only", "--data_url=s3://platform/metadata"]), 0)
        launch.assert_not_called()
        self.assertEqual(before, {str(path): path.read_bytes()
                                  for path in Path(self.temporary.name).rglob("*") if path.is_file()})

    def test_metadata_receipt_snapshots_adapter_without_storing_raw_values(self):
        record = self.inspect()
        arguments = ["--checkpoint_url=s3://sensitive/checkpoint?token=secret",
                     "--data-url", "s3://sensitive/data?token=secret"]
        receipt_path = adapter.write_receipt(record, adapter.parse_arguments(arguments), arguments)
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        self.assertEqual(receipt["platform_argument_names"], ["--checkpoint_url", "--data-url"])
        self.assertEqual(receipt["wrapper_sha256"], sha256(WRAPPER))
        self.assertEqual((receipt_path.parent / WRAPPER.name).read_bytes(), WRAPPER.read_bytes())
        self.assertFalse(receipt["platform_values_used_as_experiment_inputs"])
        self.assertNotIn("s3://sensitive", receipt_path.read_text(encoding="utf-8"))
        self.assertNotIn("secret", receipt_path.read_text(encoding="utf-8"))
        self.assertTrue(receipt_path.is_relative_to(self.experiment / "environment" / "platform-entry"))
        self.assertFalse(receipt_path.is_relative_to(self.project))
        self.assertEqual(record["source_manifest_sha256"],
                         adapter.object_sha256(adapter.source_manifest(self.project)))

    def test_main_metadata_launches_exact_recorded_project_once(self):
        record = self.inspect()
        with mock.patch.object(adapter, "inspect_project", return_value=record), \
             mock.patch.object(adapter, "run_original_entry") as launch, \
             redirect_stdout(io.StringIO()) as output:
            self.assertEqual(adapter.main(["--checkpoint_url=", "--data_url=s3://platform/data"]), 0)
        launch.assert_called_once_with(self.project.resolve())
        self.assertIn("ORIGINAL_FULL_ENTRY=START", output.getvalue())
        self.assertNotIn("s3://platform/data", output.getvalue())
        self.assertEqual(len(list((self.experiment / "environment" / "platform-entry").glob("*/entry.json"))), 1)

    def test_failed_precheck_cannot_write_receipt_or_launch(self):
        with mock.patch.object(adapter, "inspect_project", side_effect=ValueError("mismatched source")), \
             mock.patch.object(adapter, "write_receipt") as receipt, \
             mock.patch.object(adapter, "run_original_entry") as launch, \
             redirect_stderr(io.StringIO()) as error:
            self.assertEqual(adapter.main(["--data_url="]), 2)
        receipt.assert_not_called()
        launch.assert_not_called()
        self.assertIn("PLATFORM_ENTRY_PRECHECK=FAIL", error.getvalue())


class DispatchTests(Fixture):
    def test_relocated_namespace_package_is_actually_imported_from_sibling(self):
        (self.project / "ops" / "__init__.py").unlink()
        (self.project / "ops" / "modelmate_job.py").write_text(
            "import sys\n"
            "OBSERVED = {}\n"
            "def launch(stage):\n"
            "    OBSERVED.update(stage=stage, argv=list(sys.argv), path=list(sys.path))\n"
            "    return 37\n", encoding="utf-8")
        with isolated_ops_modules():
            with self.assertRaises(SystemExit) as caught:
                adapter.run_original_entry(self.project)
            imported = sys.modules["ops.modelmate_job"]
            observed = imported.OBSERVED
            self.assertEqual(Path(imported.__file__), self.project / "ops" / "modelmate_job.py")
        self.assertEqual(caught.exception.code, 37)
        self.assertEqual(observed["stage"], "full")
        self.assertEqual(observed["argv"], [str(self.project / "run_modelmate_full.py")])
        self.assertEqual(observed["path"][0], str(self.project))

    def test_relocated_original_entry_imports_sibling_clean_argv_and_preserves_exit(self):
        observed = {}

        def launch(stage):
            observed.update(stage=stage, argv=list(sys.argv), path=list(sys.path))
            return 42

        module = SimpleNamespace(launch=launch)
        original_argv, original_path = list(sys.argv), list(sys.path)
        with isolated_ops_modules(module), mock.patch.object(sys, "argv", ["outer.py", "--data_url=x"]):
            with self.assertRaises(SystemExit) as caught:
                adapter.run_original_entry(self.project)
            self.assertEqual(caught.exception.code, 42)
            self.assertEqual(sys.argv, ["outer.py", "--data_url=x"])
            self.assertEqual(sys.path, original_path)
        self.assertEqual(sys.argv, original_argv)
        self.assertEqual(observed["stage"], "full")
        self.assertEqual(observed["argv"], [str(self.project / "run_modelmate_full.py")])
        self.assertEqual(observed["path"][0], str(self.project))

    def test_original_bootstrap_reaches_one_worker_command_without_real_process(self):
        spec = importlib.util.spec_from_file_location("genuine_bootstrap", REPOSITORY / "ops" / "modelmate_job.py")
        job = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(job)
        process = SimpleNamespace(stdout=iter([job.JOB_MARKER + "/persistent/fake-job\n"]),
                                  wait=lambda: 0, poll=lambda: 0)
        fake_os = SimpleNamespace(name="posix", environ={}, getpid=lambda: 12345)
        with isolated_ops_modules(job), mock.patch.object(job, "os", fake_os), \
             mock.patch.object(job, "_settings", return_value=adapter.EXPECTED_SETTINGS), \
             mock.patch.object(job.shutil, "which", return_value="/bin/bash"), \
             mock.patch.object(job.subprocess, "Popen", return_value=process) as popen, \
             mock.patch.object(job.signal, "signal", return_value=None), \
             redirect_stdout(io.StringIO()):
            with self.assertRaises(SystemExit) as caught:
                adapter.run_original_entry(self.project)
        self.assertEqual(caught.exception.code, 0)
        popen.assert_called_once()
        command = popen.call_args.args[0]
        self.assertEqual(command[0], "/bin/bash")
        self.assertEqual(command[-2], "full")
        self.assertEqual(command[-3], adapter.EXPECTED_SETTINGS["runtime_python"])
        self.assertEqual(command[-4], str(job.PROJECT_ROOT))
        self.assertEqual(command[-5], adapter.EXPECTED_SETTINGS["cann_env"])
        self.assertNotIn("--data_url", command)
        self.assertNotIn("--checkpoint_url", command)

    def test_keyboard_interrupt_propagates_and_restores_state(self):
        def interrupt(stage):
            raise KeyboardInterrupt

        old_path, old_argv = list(sys.path), list(sys.argv)
        with isolated_ops_modules(SimpleNamespace(launch=interrupt)):
            with self.assertRaises(KeyboardInterrupt):
                adapter.run_original_entry(self.project)
        self.assertEqual(sys.path, old_path)
        self.assertEqual(sys.argv, old_argv)


if __name__ == "__main__":
    unittest.main(verbosity=2)
