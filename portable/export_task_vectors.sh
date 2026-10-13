#!/usr/bin/env bash
set -euo pipefail
"${TV_EXPORT_PYTHON:-python}" - <<'PY'
import hashlib
import json
import os
import uuid
import zipfile
from datetime import datetime, timezone
from pathlib import Path

code = Path(os.environ.get("TV_EXPORT_CODE", "/home/work/user-job-dir/app/task-vector-code-20261011T071308Z")).resolve()
experiment = Path(os.environ.get("TV_EXPORT_ROOT", "/opt/dpcvol/datasets/naie/task-vector/experiments/exp-20261011-01")).resolve()
destination = Path(os.environ.get("TV_EXPORT_DIR", "/home/work/user-job-dir/app")).resolve()
tasks = tuple(os.environ.get("TV_EXPORT_TASKS", "math,logic,translation,paraphrase").split(","))
seed = 17
if not tasks or len(set(tasks)) != len(tasks) or set(tasks) - {"math", "logic", "translation", "paraphrase"}:
    raise SystemExit("TV_EXPORT_TASKS must contain distinct original MVP task names")
for directory in (code, experiment, destination):
    if not directory.is_dir():
        raise SystemExit(f"Directory not found: {directory}")
if not (code / "src/task_vector/representations.py").is_file():
    raise SystemExit(f"Not a task-vector code directory: {code}")

def object_hash(value):
    data = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(data.encode()).hexdigest()

def file_hash(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

files = {}
def add(path, member, expected=None):
    path = Path(path)
    if path.is_symlink() or not path.is_file():
        raise RuntimeError(f"Missing file or symbolic link: {path}")
    if member in files:
        raise RuntimeError(f"Duplicate archive member: {member}")
    files[member] = (path, expected)

# Only code/configuration/documentation: never walk environments or run outputs.
excluded = {".git", "__pycache__", ".pytest_cache", ".mypy_cache", ".ruff_cache", ".venv", ".venv28", "node_modules", "dist", "build", "data", "runs", "outputs", "models", "checkpoints"}
binary = {".pyc", ".pyo", ".zip", ".whl", ".pt", ".pth", ".safetensors", ".bin"}
for directory, subdirs, names in os.walk(code, followlinks=False):
    subdirs[:] = sorted(name for name in subdirs if name not in excluded and not name.startswith(".venv") and not (Path(directory) / name).is_symlink())
    for name in sorted(names):
        path = Path(directory) / name
        if path.suffix.lower() not in binary:
            add(path, "code/" + path.relative_to(code).as_posix())

# Reproduce the existing project's source fingerprint without importing torch.
sources = {}
for folder in ("src", "configs", "deploy", "third_party", "ops"):
    for path in sorted((code / folder).rglob("*")):
        if path.is_file() and "__pycache__" not in path.parts and ".git" not in path.parts:
            sources[path.relative_to(code).as_posix()] = file_hash(path)
for relative in ["pyproject.toml", "docs/原始实验方案.md", "docs/补充协议与实现选择.md"] + [path.name for path in sorted(code.glob("run_modelmate*.py"))]:
    path = code / relative
    if path.is_file():
        sources[relative] = file_hash(path)
source_digest = object_hash(sources)

vectors = []
for route, family in (("ltv", "independent"), ("loda", "mvp")):
    for task in tasks:
        folder = experiment / "runs/training" / family / route / str(seed) / task
        record_path = folder / "training.json"
        record = json.loads(record_path.read_text(encoding="utf-8"))
        if record.get("status") != "complete" or record.get("completion_schema") != 2:
            raise RuntimeError(f"Training has not committed completed exports: {record_path}")
        prefix = "experiment/" + folder.relative_to(experiment).as_posix()
        add(record_path, prefix + "/training.json")
        exported = {}
        for field in ("raw_path", "post_path", "initial_path"):
            path = Path(record[field])
            if not path.is_absolute():
                path = folder / path.name
            if path.resolve().parent != folder.resolve():
                raise RuntimeError(f"Export is outside its completed training folder: {path}")
            expected = record.get("artifacts", {}).get(field, {}).get("sha256")
            if not expected:
                raise RuntimeError(f"Missing committed artifact hash: {record_path}, {field}")
            member = prefix + "/" + path.name
            add(path, member, expected)
            exported[field] = member
        for name in ("best-adapter.pt", "epochs.json", "steps.jsonl", "training-state.json"):
            path = folder / name
            if path.is_file():
                add(path, prefix + "/" + name)
        training_source = record.get("provenance", {}).get("source_manifest_sha256")
        vectors.append({"route": route, "task": task, "seed": seed,
            "primary": exported["post_path"], "exports": exported,
            "training_record": prefix + "/training.json", "best_nll": record.get("best_nll"),
            "completed_epochs": record.get("completed_epochs"), "stop_reason": record.get("stop_reason"),
            "training_source_sha256": training_source, "current_code_matches_training_source": training_source == source_digest})
        print(f"FOUND_COMPLETED={route}/{task} epochs={record.get('completed_epochs')}", flush=True)

# Preserve small immutable provenance files; exclude active evaluation/log files.
for relative in ("data/manifest.json", "runs/base_fingerprint.json", "runs/smoke.json", "runs/protocol/mvp.core-seed17.lock.json"):
    path = experiment / relative
    if path.is_file():
        add(path, "experiment/" + relative)

stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
archive_path = destination / f"task-vector-code-and-vectors-{stamp}-{uuid.uuid4().hex[:8]}.zip"
building = archive_path.with_name(archive_path.stem + ".building.zip")
index = {"schema_version": 1, "created_utc": datetime.now(timezone.utc).isoformat(),
    "original_code_root": str(code), "original_experiment_root": str(experiment),
    "seed": seed, "route_groups": ["ltv", "loda"], "tasks": list(tasks),
    "trained_representations": len(vectors), "current_source_sha256": source_digest,
    "current_sources": sources, "task_vectors": vectors,
    "base_model_required": "Qwen/Qwen3-1.7B", "full_training_resume_included": False,
    "original_absolute_paths_preserved": True}
readme = """代码与已完成任务表示归档\n
code/：Notebook当前代码、配置、文档、第三方源码及测试。\n
experiment/runs/training/independent/ltv/17/：LTV结果。\n
experiment/runs/training/mvp/loda/17/：LoDA-MT结果。\n
portable-index.json：任务列表及ZIP内的相对路径；primary指向post.pt。\n
raw.pt/post.pt是训练结果，initial.pt是未训练对照，不应多算任务向量。\n
原始training.json、配置和锁中的绝对路径原样保存；换机器后应先查看portable-index.json并配置新路径。\n
这是向量加载/推理与来源留存包，不是完整训练恢复包。best-adapter.pt不含可恢复的优化器状态。\n
未打包Qwen基座、语义模型、Python/CANN环境、数据集本体、last.pt/best.pt、subspaces.pt、稠密统计缓存、正在写入的评测结果。\n
向量加载仍需匹配的Qwen3-1.7B基座；评分与重建数据还需相应外部资产。\n
当前代码与训练源码是否一致见portable-index.json；如果不一致，原训练来源仍保留，不修改签名冒充一致。\n
EXPORT_MANIFEST.json记录本次包内文件哈希；code/PACKAGE_MANIFEST.json若存在，是原交付清单，不能替代本次清单。\n
本脚本只读源文件，新建归档，不停止实验，不删除文件。\n"""
digests = {}
with zipfile.ZipFile(building, "x", compression=zipfile.ZIP_DEFLATED, compresslevel=1, allowZip64=True) as archive:
    for member, (path, expected) in sorted(files.items()):
        before = path.stat()
        digest = hashlib.sha256()
        with path.open("rb") as source, archive.open(member, "w", force_zip64=True) as target:
            for chunk in iter(lambda: source.read(1024 * 1024), b""):
                digest.update(chunk)
                target.write(chunk)
        after = path.stat()
        if (before.st_size, before.st_mtime_ns, before.st_ctime_ns) != (after.st_size, after.st_mtime_ns, after.st_ctime_ns):
            raise RuntimeError(f"Source changed during export; partial archive retained: {path}")
        value = digest.hexdigest()
        if expected and value != expected:
            raise RuntimeError(f"Committed artifact SHA256 mismatch: {path}")
        relative = member.removeprefix("code/") if member.startswith("code/") else None
        if relative in sources and value != sources[relative]:
            raise RuntimeError(f"Code changed during export: {path}")
        digests[member] = value
    for member, value in (("portable-index.json", json.dumps(index, ensure_ascii=False, indent=2)), ("README.txt", readme)):
        payload = value.encode("utf-8")
        archive.writestr(member, payload)
        digests[member] = hashlib.sha256(payload).hexdigest()
    archive.writestr("EXPORT_MANIFEST.json", json.dumps({"schema_version": 1, "files": digests}, ensure_ascii=False, indent=2))
with zipfile.ZipFile(building) as archive:
    bad = archive.testzip()
    if bad:
        raise RuntimeError(f"ZIP CRC failed: {bad}")
building.rename(archive_path)
checksum_path = archive_path.with_name(archive_path.name + ".sha256.txt")
with checksum_path.open("x", encoding="utf-8", newline="\n") as handle:
    handle.write(file_hash(archive_path) + "  " + archive_path.name + "\n")
matching = sum(item["current_code_matches_training_source"] for item in vectors)
print("EXPORT_READY=PASS")
print("ROUTE_GROUPS=ltv,loda")
print(f"TRAINED_REPRESENTATIONS={len(vectors)}")
print(f"CURRENT_CODE_SOURCE_MATCH={matching}/{len(vectors)}")
if matching != len(vectors):
    print("NOTE=Current Notebook code differs from some training sources; original provenance is preserved in portable-index.json")
print(f"ARCHIVE_MIB={archive_path.stat().st_size / 1024**2:.2f}")
print(f"DOWNLOAD_ZIP={archive_path}")
print(f"DOWNLOAD_SHA256_TXT={checksum_path}")
PY
