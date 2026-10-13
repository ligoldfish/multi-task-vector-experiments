"""CPU contracts for preserving completed data during MBPP preparation recovery."""
import json

import pytest

from ops import recover_mbpp_prepare as recovery
from task_vector.data import Sample, _hash, audit_splits
from task_vector.io import file_sha256


def materialize(root, tasks, status="complete"):
    """Small prepared files; no experiment counts or source settings are changed."""
    manifest = {
        "schema_version": 1, "status": status, "split_seed": 17, "tasks": {},
        "length_policy": "filter_complete_answer_never_truncate",
        "target_eos_supervised": True,
    }
    for task in tasks:
        samples = {}
        for index, split in enumerate(("train", "tune", "discovery", "test", "test_confirmation")):
            program_id = str(601 + index)
            code_task = task in recovery.PENDING[:2]
            group = f"mbpp:{program_id}" if code_task else f"{task}:group:{index}"
            metadata = {"program_id": program_id} if code_task else {}
            samples[split] = [Sample(
                f"{task}:{split}:{program_id}", task, "prompt", "answer", group,
                "test" if split.startswith("test") else "train", metadata,
            )]
        samples["test_formal"] = samples["test"] + samples["test_confirmation"]
        ordered = {split: samples[split] for split in recovery.SPLITS}
        files = {}
        for split, rows in ordered.items():
            path = root / task / f"{split}.jsonl"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("".join(json.dumps(row.to_dict(), sort_keys=True) + "\n" for row in rows), encoding="utf-8")
            files[split] = {
                "path": str(path.resolve()), "sha256": file_sha256(path),
                "count": len(rows), "ids_sha256": _hash([row.id for row in rows]),
            }
        manifest["tasks"][task] = {"files": files, "audit": audit_splits(ordered), "source": {"files": []}}
    if status == "complete":
        manifest["manifest_content_sha256"] = _hash(manifest)
    (root / "manifest.json").write_text(json.dumps(manifest, sort_keys=True), encoding="utf-8")
    return manifest


@pytest.fixture
def interrupted(tmp_path):
    run = tmp_path / "experiment"
    root = run / "data"
    original = materialize(root, recovery.CORE, status="preparing")
    (root / "verification").mkdir()
    (root / "verification" / "mbpp-927.json").write_text('{"error":"old setup-order failure"}', encoding="utf-8")
    failed = root / "code_generation"
    failed.mkdir()
    (failed / "capacity_preflight.json").write_text(json.dumps({"eligible_train": 373, "source": {"files": []}}), encoding="utf-8")
    (failed / "rejected.jsonl").write_text('{"id":"code_generation:train:927"}\n', encoding="utf-8")
    config = {
        "platform": {"persistent_root": str(run)},
        "data": {
            "split_seed": 17, "tasks": list(recovery.CORE),
            "extension_tasks": list(recovery.PENDING[:2]), "auxiliary_tasks": ["summary"],
        },
    }
    config_path = tmp_path / "config.yaml"
    config_path.write_text("fixture configuration\n", encoding="utf-8")
    return root, original, config, config_path


def fake_prepare(config, tokenizer, stage):
    assert config["data"]["tasks"] == list(recovery.PENDING)
    assert not (stage / "verification").exists(), "fresh stage must not reuse old failure cache"
    pending = materialize(stage, recovery.PENDING)
    (stage / "verification").mkdir()
    (stage / "verification" / "mbpp-927.json").write_text('{"verification":{"fixed":true}}', encoding="utf-8")
    return pending


def test_recovery_keeps_core_bytes_and_publishes_verified_pending_paths(interrupted, monkeypatch):
    root, original, config, config_path = interrupted
    core_bytes = {path: path.read_bytes() for task in recovery.CORE for path in (root / task).glob("*.jsonl")}
    old_manifest = (root / "manifest.json").read_bytes()
    old_cache = (root / "verification" / "mbpp-927.json").read_bytes()
    monkeypatch.setattr(recovery, "prepare_dataset", fake_prepare)

    result = recovery.recover(config, object(), config_path)

    assert result["status"] == "complete"
    assert set(result["tasks"]) == set(recovery.CORE + recovery.PENDING)
    assert all(path.read_bytes() == content for path, content in core_bytes.items())
    assert all(result["tasks"][task] == original["tasks"][task] for task in recovery.CORE)
    assert result["manifest_content_sha256"] == _hash({k: v for k, v in result.items() if k != "manifest_content_sha256"})
    assert json.loads((root / "manifest.json").read_text()) == result
    for task in recovery.PENDING:
        for split, record in result["tasks"][task]["files"].items():
            assert record["path"] == str((root / task / f"{split}.jsonl").resolve())
            assert file_sha256(record["path"]) == record["sha256"]
    for split in recovery.SPLITS:
        program_ids = []
        for task in recovery.PENDING[:2]:
            rows = [json.loads(line) for line in (root / task / f"{split}.jsonl").read_text().splitlines()]
            program_ids.append([row["metadata"]["program_id"] for row in rows])
        assert program_ids[0] == program_ids[1]
    evidence, = (root.parent / "environment").glob("mbpp-recovery-*")
    assert (evidence / "before" / "manifest.json").read_bytes() == old_manifest
    assert (evidence / "before" / "verification" / "mbpp-927.json").read_bytes() == old_cache
    assert json.loads((root / "verification" / "mbpp-927.json").read_text())["verification"]["fixed"] is True
    assert json.loads((evidence / "recovery.json").read_text())["core_unchanged"] is True


def test_failed_pending_preparation_preserves_active_manifest_and_cache(interrupted, monkeypatch):
    root, original, config, config_path = interrupted
    active_before = (root / "manifest.json").read_bytes()
    cache_before = (root / "verification" / "mbpp-927.json").read_bytes()
    def fail(config, tokenizer, stage):
        stage.mkdir()
        (stage / "failure-evidence.json").write_text('{"eligible_train":373}')
        raise ValueError("pending capacity failure")
    monkeypatch.setattr(recovery, "prepare_dataset", fail)

    with pytest.raises(ValueError, match="pending capacity failure"):
        recovery.recover(config, object(), config_path)

    assert (root / "manifest.json").read_bytes() == active_before
    assert (root / "verification" / "mbpp-927.json").read_bytes() == cache_before
    assert recovery.verify_tasks(root, original, recovery.CORE)
    evidence, = (root.parent / "environment").glob("mbpp-recovery-*")
    assert (evidence / "pending-data" / "failure-evidence.json").is_file()
    assert json.loads((evidence / "recovery.json").read_text())["status"] == "failed"


@pytest.mark.parametrize("change", ["file", "complete", "missing_core", "seed"])
def test_recovery_refuses_changed_or_inapplicable_active_data(interrupted, monkeypatch, change):
    root, original, config, config_path = interrupted
    if change == "file":
        with (root / "math" / "train.jsonl").open("a", encoding="utf-8") as handle:
            handle.write("\n")
    elif change == "complete":
        original["status"] = "complete"
    elif change == "missing_core":
        original["tasks"].pop("logic")
    else:
        original["split_seed"] = 18
    if change != "file":
        (root / "manifest.json").write_text(json.dumps(original), encoding="utf-8")
    before = (root / "manifest.json").read_bytes()
    monkeypatch.setattr(recovery, "prepare_dataset", lambda *args: pytest.fail("invalid active data must be rejected first"))

    with pytest.raises(ValueError):
        recovery.recover(config, object(), config_path)

    assert (root / "manifest.json").read_bytes() == before
    assert not (root.parent / "environment").exists()


def test_pending_hash_mismatch_is_rejected_before_copy(interrupted, tmp_path):
    root, original, config, config_path = interrupted
    stage = tmp_path / "pending"
    pending = fake_prepare({"data": {"tasks": list(recovery.PENDING)}}, object(), stage)
    pending["tasks"]["summary"]["files"]["train"]["count"] = 9
    before = (root / "manifest.json").read_bytes()
    with pytest.raises(ValueError, match="Pending manifest hash mismatch"):
        recovery.publish_pending(root, original, stage, pending)
    assert (root / "manifest.json").read_bytes() == before
    assert not (root / "summary").exists()


def test_mismatched_mbpp_program_assignments_are_rejected_before_copy(interrupted, tmp_path):
    root, original, config, config_path = interrupted
    stage = tmp_path / "pending"
    pending = fake_prepare({"data": {"tasks": list(recovery.PENDING)}}, object(), stage)
    path = stage / "code_repair" / "train.jsonl"
    row = json.loads(path.read_text())
    row["metadata"]["program_id"] = "974"
    path.write_text(json.dumps(row) + "\n", encoding="utf-8")
    pending["tasks"]["code_repair"]["files"]["train"]["sha256"] = file_sha256(path)
    pending["manifest_content_sha256"] = _hash({k: v for k, v in pending.items() if k != "manifest_content_sha256"})
    before = (root / "manifest.json").read_bytes()

    with pytest.raises(ValueError, match="program assignments differ"):
        recovery.publish_pending(root, original, stage, pending)

    assert (root / "manifest.json").read_bytes() == before
    assert not (root / "code_repair").exists()


def test_recovery_refuses_changed_mbpp_source(interrupted, monkeypatch):
    root, original, config, config_path = interrupted
    before = (root / "manifest.json").read_bytes()
    def changed_source(config, tokenizer, stage):
        pending = fake_prepare(config, tokenizer, stage)
        pending["tasks"]["code_generation"]["source"]["revision"] = "changed-source"
        return pending
    monkeypatch.setattr(recovery, "prepare_dataset", changed_source)

    with pytest.raises(ValueError, match="Official MBPP source changed"):
        recovery.recover(config, object(), config_path)

    assert (root / "manifest.json").read_bytes() == before
    assert not (root / "code_repair").exists()
