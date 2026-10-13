"""Stdlib-only publication checks, runnable without a local torch installation."""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest


SPEC = importlib.util.spec_from_file_location(
    "recover_prefill_smoke", Path(__file__).resolve().parents[1] / "ops" / "recover_prefill_smoke.py")
RECOVERY = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(RECOVERY)


def atomic_json(path, value):
    path.write_text(json.dumps(value), encoding="utf-8")


class PublicationTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="smoke-publish-")
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.canonical = self.root / "smoke.json"
        self.old = b'{"passed":false,"status":"failed"}\n'
        self.canonical.write_bytes(self.old)
        self.attempt = self.root / "attempt"
        self.attempt.mkdir()
        (self.attempt / "previous-smoke.json").write_bytes(self.old)
        self.expected = {"config_sha256": "config", "source_manifest_sha256": "new-source",
                         "data_manifest_sha256": "data", "base_fingerprint": "base"}
        self.candidate = {"passed": True, "status": "passed", **self.expected,
                          "checks": {"loda_true_gao_microbatch_step": True,
                                     "loda_checkpoint_resume_exact": True,
                                     "loda_resume_does_not_repeat_step": True}}

    def publish(self):
        (self.attempt / "smoke.json").write_text(json.dumps(self.candidate), encoding="utf-8")
        return RECOVERY.publish_passed_attempt(
            self.canonical, self.attempt, self.old, self.expected, atomic_json)

    def test_full_pass_publishes_and_keeps_original_evidence(self):
        record = self.publish()
        self.assertEqual(json.loads(self.canonical.read_bytes()), record)
        self.assertEqual((self.attempt / "previous-smoke.json").read_bytes(), self.old)
        self.assertEqual(record["previous_failed_smoke_sha256"], RECOVERY.sha256(self.old))
        self.assertEqual(record["engineering_attempt_directory"], str(self.attempt.resolve()))

    def test_failure_does_not_publish(self):
        self.candidate.update(passed=False, status="failed")
        with self.assertRaisesRegex(ValueError, "did not pass"):
            self.publish()
        self.assertEqual(self.canonical.read_bytes(), self.old)

    def test_changed_provenance_does_not_publish(self):
        self.candidate["source_manifest_sha256"] = "unexpected-source"
        with self.assertRaisesRegex(ValueError, "provenance mismatch"):
            self.publish()
        self.assertEqual(self.canonical.read_bytes(), self.old)

    def test_incomplete_loda_smoke_does_not_publish(self):
        self.candidate["checks"]["loda_checkpoint_resume_exact"] = False
        with self.assertRaisesRegex(ValueError, "LoDA"):
            self.publish()
        self.assertEqual(self.canonical.read_bytes(), self.old)

    def test_concurrent_record_change_is_preserved(self):
        replacement = b'{"newer":"record"}'
        self.canonical.write_bytes(replacement)
        with self.assertRaisesRegex(RuntimeError, "changed during retry"):
            self.publish()
        self.assertEqual(self.canonical.read_bytes(), replacement)


if __name__ == "__main__":
    unittest.main(verbosity=2)
