"""Retry the engineering smoke in a fresh attempt and publish only a full pass.

Configuration, prepared data, model weights and scientific training are unchanged.
The previous smoke and its saved engineering tensors remain recoverable.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace


def sha256(payload):
    return hashlib.sha256(payload).hexdigest()


def publish_passed_attempt(canonical, attempt, previous_bytes, expected, atomic_json):
    """Publish only a complete, provenance-matched attempt; retain failed evidence."""
    canonical, attempt = Path(canonical), Path(attempt)
    candidate = json.loads((attempt / "smoke.json").read_bytes())
    if candidate.get("passed") is not True or candidate.get("status") != "passed":
        raise ValueError("The new engineering smoke did not pass")
    for field, value in expected.items():
        if candidate.get(field) != value:
            raise ValueError(f"New smoke provenance mismatch: {field}")
    required = ("loda_true_gao_microbatch_step", "loda_checkpoint_resume_exact",
                "loda_resume_does_not_repeat_step")
    if any(candidate.get("checks", {}).get(key) is not True for key in required):
        raise ValueError("New smoke did not complete the LoDA training/resume checks")
    if canonical.read_bytes() != previous_bytes:
        raise RuntimeError("Canonical smoke changed during retry; refusing publication")
    candidate["engineering_attempt_directory"] = str(attempt.resolve())
    candidate["previous_failed_smoke_sha256"] = sha256(previous_bytes)
    candidate["previous_failed_smoke_backup"] = str((attempt / "previous-smoke.json").resolve())
    atomic_json(attempt / "smoke.json", candidate)
    atomic_json(canonical, candidate)
    return candidate


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    project = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(project / "src"))
    from task_vector.config import load_config, scientific_config, source_manifest
    from task_vector.io import atomic_json, object_sha256
    from task_vector.runner import ExperimentRunner
    from task_vector.smoke import run_smoke
    from task_vector.workflow import smoke_evidence

    config = load_config(args.config)
    root = Path(config["platform"]["persistent_root"])
    config.setdefault("runtime", {})
    config["runtime"].setdefault("data_root", str(root / "data"))
    config["runtime"].setdefault("output_root", str(root / "runs"))
    runner = ExperimentRunner(config, source_root=project)
    canonical = runner.output_root / "smoke.json"
    previous_bytes = canonical.read_bytes()
    previous = json.loads(previous_bytes)
    if previous.get("passed") is not False or previous.get("task") != "math":
        raise ValueError("This recovery requires the existing failed math smoke")
    expected = {"config_sha256": object_sha256(scientific_config(config)),
                "data_manifest_sha256": object_sha256(runner.manifest),
                "base_fingerprint": runner.base_fingerprint(),
                "source_manifest_sha256": object_sha256(source_manifest(project))}
    # Source changes are the declared engineering fix. Everything else must be
    # exactly the same as in the failed attempt, including the sample identities.
    for field in ("config_sha256", "data_manifest_sha256", "base_fingerprint"):
        if previous.get(field) != expected[field]:
            raise ValueError(f"Original smoke inputs changed: {field}")
    if previous.get("sample_ids") != [s.id for s in runner.samples("math", "train")[:2]]:
        raise ValueError("Original smoke training sample identities changed")
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    attempt = runner.output_root / "engineering_smoke" / f"prefill-fix-{stamp}-{os.getpid()}"
    attempt.mkdir(parents=True, exist_ok=False)
    (attempt / "previous-smoke.json").write_bytes(previous_bytes)
    receipt = {"kind": "engineering_smoke_recovery", "status": "running",
               "scientific_training": False, "formal_evaluation": False,
               "canonical_smoke": str(canonical), "attempt": str(attempt),
               "previous_failed_smoke_sha256": sha256(previous_bytes),
               "expected_provenance": expected}
    print(f"SMOKE_ATTEMPT={attempt}", flush=True)
    atomic_json(attempt / "recovery.json", receipt)
    try:
        print("STAGE=complete_real_model_smoke", flush=True)
        run_smoke(config, task="math", output_dir=attempt)
        attempt_view = SimpleNamespace(output_root=attempt, config=runner.config,
            manifest=runner.manifest, source_root=runner.source_root,
            base_fingerprint=runner.base_fingerprint, taskset=runner.taskset)
        pending_evidence = smoke_evidence(attempt_view, "mvp")
        if not pending_evidence["valid"]:
            raise ValueError(f"New smoke was not accepted by the workflow: {pending_evidence}")
        publish_passed_attempt(canonical, attempt, previous_bytes, expected, atomic_json)
        evidence = smoke_evidence(runner, "mvp")
        if not evidence["valid"]:
            raise ValueError(f"Published smoke was not accepted by the workflow: {evidence}")
        receipt.update(status="passed", workflow_smoke_evidence=evidence,
                       previous_artifacts_preserved=True)
        atomic_json(attempt / "recovery.json", receipt)
        record = json.loads(canonical.read_bytes())
        print("SMOKE_CHECKS=" + json.dumps(record["checks"], ensure_ascii=False), flush=True)
        print("REAL_MODEL_SMOKE=PASS", flush=True)
        print("WORKFLOW_SMOKE_EVIDENCE=PASS", flush=True)
        print(f"CANONICAL_SMOKE={canonical}", flush=True)
        print("NEXT=training-job configuration; no training job submitted", flush=True)
    except BaseException as exc:
        receipt.update(status="failed", error=f"{type(exc).__name__}: {exc}",
                       canonical_unchanged=canonical.read_bytes() == previous_bytes)
        atomic_json(attempt / "recovery.json", receipt)
        raise


if __name__ == "__main__":
    main()
