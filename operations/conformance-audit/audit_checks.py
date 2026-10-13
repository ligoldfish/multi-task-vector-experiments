"""Small conformance probes; no model, dataset, NPU, training or installation."""
from __future__ import annotations
import ast
from contextlib import redirect_stdout
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import io
import json
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace
import xml.etree.ElementTree as ET
import zipfile

import yaml

ROOT = Path(__file__).resolve().parent
DELIVERY = ROOT.parent
SOURCE = Path("D:/DataSpecificWork/task-vector")
PAYLOAD = DELIVERY / "modelmate-runbook/payload"
sys.path.insert(0, str(SOURCE / "src"))
from task_vector.config import validate_config


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    config = yaml.safe_load((SOURCE / "configs/experiment.yaml").read_text(encoding="utf-8"))
    validate_config(deepcopy(config))
    variants = {
        "training_count_512_to_1": ("data.counts.train", 1),
        "primary_layer_14_to_0": ("ltv.layers", [0]),
        "initialization_range_changed": ("ltv.init_high", 2.0),
        "a_scale_changed": ("loda.a_scale", 1.0),
        "seed17_removed": ("analysis.seeds", [29]),
        "heldout_folds_removed": ("analysis.heldout_tasks", []),
        "surface_controls_removed": ("evaluation.variants", ["canonical"]),
        "code_extension_removed": ("data.extension_tasks", []),
    }
    bypasses = {}
    for label, (dotted, value) in variants.items():
        changed = deepcopy(config)
        cursor = changed
        parts = dotted.split(".")
        for part in parts[:-1]:
            cursor = cursor[part]
        cursor[parts[-1]] = value
        try:
            validate_config(changed)
            bypasses[label] = {"field": dotted, "configured": value, "accepted_without_supplement_label": True}
        except Exception as exc:
            bypasses[label] = {"field": dotted, "rejected": str(exc)}

    # Replace only the heavy execution boundary. Exercise the real CLI main
    # with a legitimate stopped workflow result; no model/import is required.
    import task_vector.cli as cli
    fake_runner_module = ModuleType("task_vector.runner")
    fake_workflow_module = ModuleType("task_vector.workflow")

    class DummyRunner:
        def __init__(self, config, source_root=None):
            pass
        def release_model(self):
            pass

    fake_runner_module.ExperimentRunner = DummyRunner
    fake_workflow_module.run_phase = lambda *args: {"completed": False, "stop_reason": "audit_gate_fixture"}
    args = SimpleNamespace(command="run", config="unused", set=[], device=None,
                           phase="all", scope="mvp", split="test")
    old_parser, old_load = cli.parser, cli.load_config
    previous = {name: sys.modules.get(name) for name in ("task_vector.runner", "task_vector.workflow")}
    sys.modules.update({"task_vector.runner": fake_runner_module, "task_vector.workflow": fake_workflow_module})
    cli.parser = lambda: SimpleNamespace(parse_args=lambda argv: args)
    cli.load_config = lambda *args: {"platform": {"persistent_root": str(ROOT / "not-created")}}
    buffer = io.StringIO()
    try:
        with redirect_stdout(buffer):
            code = cli.main([])
        cli_probe = {"workflow_completed": False, "actual_cli_returncode": code,
                     "stdout": json.loads(buffer.getvalue()), "heavy_boundary_stubbed": True}
    finally:
        cli.parser, cli.load_config = old_parser, old_load
        for name, value in previous.items():
            if value is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = value

    syntax_files = sorted(SOURCE.glob("src/**/*.py")) + sorted(SOURCE.glob("tests/**/*.py"))
    syntax_files += sorted(PAYLOAD.glob("ops/*.py")) + sorted(PAYLOAD.glob("run_modelmate_*.py"))
    for path in syntax_files:
        text = path.read_text(encoding="utf-8-sig")
        ast.parse(text, filename=str(path), feature_version=(3, 10))
        compile(text, str(path), "exec")

    with zipfile.ZipFile(DELIVERY / "task-vector-modelmate.zip") as archive:
        manifest = json.loads(archive.read("PACKAGE_MANIFEST.json"))
    hashes = []
    for relative, expected in manifest["files"].items():
        actual_source, actual_payload = sha(SOURCE / relative), sha(PAYLOAD / relative)
        hashes.append({"file": relative, "source_sha256": actual_source,
                       "payload_sha256": actual_payload, "delivery_sha256": expected,
                       "identical": actual_source == actual_payload == expected})
    historical_tests = {}
    for name in ("validation-cpu28.xml", "validation-cpu214.xml"):
        path = DELIVERY / name
        if path.is_file():
            tree = ET.parse(path).getroot()
            historical_tests[name] = {"sha256": sha(path), "testcases": len(list(tree.iter("testcase"))),
                "failures": len(list(tree.iter("failure"))), "errors": len(list(tree.iter("error"))),
                "rerun_this_turn": False}
    record = {"checked_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "read-only source checks and small control-flow probes; only this report is written",
        "python": sys.executable, "torch_imported": "torch" in sys.modules,
        "real_model_data_npu_or_training_run": False,
        "original_document_sha256": sha(SOURCE / "docs/原始实验方案.md"),
        "default_config_validation_passed": True,
        "config_guard_probes": bypasses, "cli_gate_probe": cli_probe,
        "python_syntax_files_passed": len(syntax_files),
        "original_delivery_files_compared": len(hashes),
        "original_delivery_source_payload_all_identical": all(item["identical"] for item in hashes),
        "historical_test_reports": historical_tests,
        "file_hashes": hashes}
    (ROOT / "audit-checks.json").write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in record.items() if key != "file_hashes"}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
