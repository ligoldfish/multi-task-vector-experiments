"""Read-only ModelMate export: independent vectors plus a full experiment handoff.

Only PyYAML is required on the exporting machine; torch is never imported here.
The original experiment remains untouched, including during an active evaluation.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import uuid
import zipfile


TASKS = ("math", "logic", "translation", "paraphrase", "code_generation", "code_repair", "summary")
ROOT_CODE = "/home/work/user-job-dir/app/task-vector-code-20261011T071308Z"
ROOT_EXPERIMENT = "/opt/dpcvol/datasets/naie/task-vector/experiments/exp-20261011-01"


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def object_hash(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                    separators=(",", ":"), default=str).encode()).hexdigest()


def data_hash(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True).encode()).hexdigest()


def source_manifest(root):
    entries = {}
    for folder in ("src", "configs", "deploy", "third_party", "ops"):
        for path in sorted((root / folder).rglob("*")):
            if path.is_file() and "__pycache__" not in path.parts and ".git" not in path.parts:
                entries[path.relative_to(root).as_posix()] = file_hash(path)
    names = ["pyproject.toml", "docs/原始实验方案.md", "docs/补充协议与实现选择.md"]
    names += [p.name for p in sorted(root.glob("run_modelmate*.py"))]
    for name in names:
        if (root / name).is_file():
            entries[name] = file_hash(root / name)
    return entries


def scientific_config(config):
    result = deepcopy(config)
    result.get("platform", {}).pop("device", None)
    result.get("training", {}).pop("resume_from", None)
    return result


def request_hash(config, sources, manifest, base, item):
    config = deepcopy(config)
    if item["layers"] is not None:
        config["ltv"]["layers"] = item["layers"]
    return object_hash({"config": scientific_config(config), "route": item["route"],
                        "seed": item["seed"], "scope": item["scope"] if item["route"] == "loda" else "independent",
                        "task": item["task"], "data": manifest["manifest_content_sha256"],
                        "base": base, "sources": sources})


def add_file(files, path, member, expected=None):
    path = Path(path)
    if not path.is_file():
        raise ValueError(f"Required file is missing: {path}")
    if member in files:
        raise ValueError(f"Duplicate ZIP member: {member}")
    files[member] = (path, expected)


def tree_files(root, prefix, files):
    for directory, dirs, names in os.walk(root, followlinks=False):
        dirs[:] = sorted(d for d in dirs if d not in {".git", "__pycache__", ".cache", ".pytest_cache"})
        for name in sorted(names):
            path = Path(directory) / name
            if path.suffix not in {".pyc", ".pyo"}:
                add_file(files, path, prefix + path.relative_to(root).as_posix())


def collect(code, root, config_path, tools):
    import yaml
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    config.setdefault("runtime", {}).setdefault("data_root", str(root / "data"))
    config["runtime"].setdefault("output_root", str(root / "runs"))
    if Path(config["platform"]["persistent_root"]).resolve() != root:
        raise ValueError("Configured persistent_root differs from --experiment; select the actual worker configuration")
    if Path(config["runtime"]["data_root"]).resolve() != root / "data" or Path(config["runtime"]["output_root"]).resolve() != root / "runs":
        raise ValueError("The actual configured data/runs roots differ from this export")
    if config["model"]["id"] != "Qwen/Qwen3-1.7B":
        raise ValueError("Expected the experiment's Qwen3-1.7B base")
    manifest_path = root / "data/manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("status") != "complete" or set(manifest.get("tasks", {})) != set(TASKS):
        raise ValueError("All seven prepared tasks must be complete")
    if manifest.get("manifest_content_sha256") != data_hash({k: v for k, v in manifest.items() if k != "manifest_content_sha256"}):
        raise ValueError("Prepared manifest content hash is invalid")
    base_path = root / "runs/base_fingerprint.json"
    base = json.loads(base_path.read_text(encoding="utf-8"))
    if not base.get("sha256") or object_hash(base["sha256"]) != base.get("fingerprint"):
        raise ValueError("Saved base model fingerprint is invalid")
    model_root = Path(config["model"]["path"])
    model_names = {p.name for p in model_root.glob("*") if p.is_file() and p.suffix in {".json", ".safetensors", ".bin", ".model", ".txt"}}
    if model_names != set(base["sha256"]):
        raise ValueError("Model files have changed since the saved fingerprint")
    for name, expected in base["sha256"].items():
        if file_hash(model_root / name) != expected:
            raise ValueError(f"Base model changed: {name}")
        print(f"BASE_VERIFIED={name}", flush=True)
    sources = source_manifest(code)
    source_digest = object_hash(sources)
    full, vectors = {}, {}
    for folder in ("src", "configs", "deploy", "third_party", "ops", "tests", "docs"):
        if (code / folder).is_dir():
            tree_files(code / folder, "code/" + folder + "/", full)
    for path in sorted(code.iterdir()):
        if path.is_file() and (path.suffix.lower() in {".py", ".md", ".toml", ".txt", ".json", ".yaml", ".yml", ".sh"} or path.name in {".gitignore", ".gitattributes"}):
            add_file(full, path, "code/" + path.name)
    for name, expected in sources.items():
        member = "code/" + name
        if member not in full:
            raise ValueError(f"Source manifest file was excluded: {name}")
        full[member] = (full[member][0], expected)
    add_file(full, manifest_path, "experiment/data/manifest.json", file_hash(manifest_path))
    for task, description in manifest["tasks"].items():
        if not {"train", "tune", "discovery", "test", "test_formal", "test_confirmation"} <= set(description["files"]):
            raise ValueError(f"Prepared splits are missing: {task}")
        for split, entry in description["files"].items():
            path = root / "data" / task / f"{split}.jsonl"
            rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
            if len(rows) != entry["count"] or data_hash([row["id"] for row in rows]) != entry["ids_sha256"]:
                raise ValueError(f"Prepared sample IDs/count differ: {task}/{split}")
            add_file(full, path, "experiment/data/" + task + "/" + split + ".jsonl", entry["sha256"])
        for name in ("rejected.jsonl", "capacity_preflight.json", "group_audit.json"):
            path = root / "data" / task / name
            if path.is_file():
                add_file(full, path, "experiment/data/" + task + "/" + name)
    inventory = []
    completed = sorted((root / "runs/training").glob("*/*/*/*/training.json"))
    for record_path in completed:
        folder = record_path.parent
        family, route_folder, seed, task = folder.relative_to(root / "runs/training").parts
        if route_folder == "loda":
            route, layers = "loda", None
            scope = family
            if scope not in {"mvp", "expanded"} and not scope.startswith("loo-"):
                continue
        elif family == "independent" and re.fullmatch(r"ltv(?:-layers-\d+(?:-\d+)*)?", route_folder):
            route = "ltv"
            layers = [int(x) for x in route_folder.removeprefix("ltv-layers-").split("-")] if route_folder != "ltv" else None
            scope = "expanded" if task in {"code_generation", "code_repair"} else "mvp"
        else:
            continue
        record = json.loads(record_path.read_text(encoding="utf-8"))
        if record.get("status") != "complete" or record.get("completion_schema") != 2:
            continue
        relative_folder = "experiment/" + folder.relative_to(root).as_posix()
        item = {"directory": relative_folder, "route": route, "scope": scope,
                "seed": int(seed), "task": task, "layers": layers,
                "record": relative_folder + "/training.json", "best_nll": record.get("best_nll"),
                "completed_epochs": record.get("completed_epochs")}
        if record.get("provenance", {}).get("source_manifest_sha256") != source_digest:
            raise ValueError(f"Current code differs from the training source: {record_path}. Export the actual original worker code; do not forge provenance.")
        if record.get("request_signature") != request_hash(config, sources, manifest, base["fingerprint"], item):
            raise ValueError(f"Current config does not reproduce the completed training request: {record_path}")
        if record.get("provenance", {}).get("data_manifest_sha256") != object_hash(manifest):
            raise ValueError(f"Training data differs from current prepared data: {record_path}")
        add_file(full, record_path, item["record"], file_hash(record_path))
        vector_prefix = f"task_vectors/{family}/{route_folder}/{seed}/{task}/"
        add_file(vectors, record_path, vector_prefix + "training-origin.json", file_hash(record_path))
        exports = {}
        for field in ("raw_path", "post_path", "initial_path"):
            name = field.removesuffix("_path") + ".pt"
            path = folder / name
            if Path(record[field]).name != name:
                raise ValueError(f"Unexpected export name: {record_path}/{field}")
            artifact = record.get("artifacts", {}).get(field, {})
            expected = artifact.get("sha256")
            if not expected or not artifact.get("representation_digest"):
                raise ValueError(f"Missing committed export digest: {record_path}/{field}")
            add_file(full, path, relative_folder + "/" + name, expected)
            add_file(vectors, path, vector_prefix + name, expected)
            exports[field] = {"path": vector_prefix + name, "sha256": expected,
                              "representation_digest": artifact["representation_digest"]}
        for name in ("best-adapter.pt", "epochs.json", "steps.jsonl", "subspaces.pt"):
            path = folder / name
            if path.is_file():
                add_file(full, path, relative_folder + "/" + name)
        inventory.append(item)
        item["standalone_exports"] = exports
        print(f"COMPLETED={family}/{route_folder}/{seed}/{task} epochs={record.get('completed_epochs')}", flush=True)
    required = {(r, t) for r in ("ltv", "loda") for t in TASKS[:4]}
    found = {(i["route"], i["task"]) for i in inventory if i["seed"] == 17 and i["scope"] == "mvp" and i["layers"] is None}
    if not required <= found:
        raise ValueError(f"The original eight seed17 representations are not all complete: missing={required - found}")
    stats_folder = root / "runs/statistics/frozen_base"
    for path in sorted(stats_folder.glob("*.pt")):
        add_file(full, path, "experiment/runs/statistics/frozen_base/" + path.name)
    for task in TASKS[:4]:
        if not (stats_folder / f"{task}.pt").is_file():
            raise ValueError(f"Required frozen-base statistics are missing: {task}")
    add_file(full, base_path, "experiment/runs/base_fingerprint.json", file_hash(base_path))
    semantic = config.get("metrics", {}).get("paraphrase", {}).get("semantic_model_path") or config["metrics"].get("semantic_model_path")
    if not semantic or not Path(semantic).is_dir():
        raise ValueError("Configure the actual offline all-MiniLM-L6-v2 asset before export")
    tree_files(Path(semantic), "assets/semantic_model/", full)
    # Historical evidence is never installed as valid target smoke/locks/evaluations.
    for relative in ("runs/smoke.json", "runs/preflight.json", "runs/experiment_inventory.json"):
        path = root / relative
        if path.is_file():
            add_file(full, path, "evidence/" + relative)
    for path in sorted((root / "runs/protocol").glob("*.json")):
        add_file(full, path, "evidence/runs/protocol/" + path.name)
    for path in sorted((root / "environment").glob("*.json")):
        add_file(full, path, "evidence/environment/" + path.name)
    for name in ("portable_import.py", "portable_run.py", "use_task_vector.py", "requirements-portable.txt", "README_迁移与使用.md", "export_portable.py", "export_portable.sh"):
        add_file(full, tools / name, name)
    add_file(vectors, tools / "use_task_vector.py", "use_task_vector.py")
    index = {"schema_version": 2, "created_utc": datetime.now(timezone.utc).isoformat(),
             "original_code_root": str(code), "original_experiment_root": str(root),
             "source_manifest": sources, "source_manifest_sha256": source_digest,
             "config_used": config, "base_fingerprint": base,
             "training": inventory, "semantic_model": "assets/semantic_model",
             "reuse_policy": "completed_training_only_reevaluate",
             "optimizer_resume_supported": False, "base_model_included": False,
             "original_absolute_paths_are_audit_evidence": True,
             "evaluation_exposure": "Existing test samples were already evaluated; relocation does not make them unseen."}
    return full, vectors, index


VECTOR_README = """独立 task vector 包（无需整套实验代码）

每个目录是一项训练结果，目录区分路线、层、seed、任务。
post.pt：主结果，可直接使用。raw.pt：重标定前版本。initial.pt：未训练对照，不是额外训练结果。
training-origin.json：来源记录，其中绝对路径属于原运行位置。有效新位置见 VECTOR_INDEX.json。
BASE_MODEL_CONTRACT.json：相同基座所需文件及内容SHA256。不能只按模型名称认定兼容。

所有.pt均为普通字典和CPU张量，无需task_vector包即可读取：
  import torch
  d = torch.load('post.pt', map_location='cpu', weights_only=True)
  print(d['space'], d['task'], d['seed'], d['metadata'])

space='activation'（LTV）：d['values']保存各层向量，key为模型中的block路径。
在对应Transformer block输入、RMSNorm之前、每条输入的最后一个真实prompt token处
加向量，强度1；只在prefill加一次。
带KV cache续写时不要重复注入；无cache重算完整前缀时仍使用固定原prompt位置。
层号为0-based，默认14；请结合metadata.hook='decoder_block_input_before_rmsnorm'核对。

space='parameter'（LoDA）：d['factors']的key是Q/K/V原生线性投影模块名，
每个key对应多个 {'B': tensor, 'A': tensor}。实际更新为 sum(B @ A)。原实验前向时
在禁用autocast的FP32旁路加 sum((x.float() @ A.T) @ B.T)，再将delta转成原输出dtype。
直接把更新merge进BF16 weight可能产生不同数值；为保持原计算使用附带最小工具的旁路。
post已含重标定，不要再做lambda缩放，
不要额外套LoRA alpha/r。FP32因子迁移到目标设备时按原代码计算并与基座dtype匹配。
保留原始基座副本，卸载时恢复原weight，避免重复叠加。

直接拿到张量不等于任意模型都能使用：必须匹配相同基座权重、tokenizer、模块名、层与位置。
LTV不是可直接merge到weight的LoRA；LoDA也不提供标准PEFT adapter格式。
目标环境可自行实现等价注入；use_task_vector.py仅依赖torch，提供可选的最小注入上下文，
无需安装完整实验项目。示例：
  from use_task_vector import task_vector_context, verify_base_directory
  verify_base_directory('/path/to/Qwen3-1.7B', 'BASE_MODEL_CONTRACT.json')
  # model/tokenizer由你现有的程序加载，必须enable_thinking=False。
  # 左padding的batch中，每条prompt的最后位置都是input_ids.shape[1]-1。
  positions = torch.full((input_ids.shape[0],), input_ids.shape[1]-1, device=input_ids.device)
  with task_vector_context(model, 'path/to/post.pt', positions=positions):
      output = model.generate(input_ids=input_ids, attention_mask=attention_mask,
                              do_sample=False, use_cache=True, max_new_tokens=512)
完整包内原代码另有真实模型smoke。
CUDA/NPU、batch变化可能改变BF16生成细节，目标应重新评测；本包不宣称跨硬件逐token一致。
本包没有旧优化器状态、数据或原完整代码；后续完整实验请同时下载portable-full包。
EXPORT_MANIFEST.json记录本包文件SHA256；原来源记录未改写。
"""


def write_zip(path, files, generated):
    building = path.with_name(path.stem + ".building.zip")
    digests = {}
    total = sum(p.stat().st_size for p, _ in files.values())
    written = 0
    print(f"PACKAGING={path.name} source_GiB={total / 1024**3:.2f}", flush=True)
    with zipfile.ZipFile(building, "x", compression=zipfile.ZIP_DEFLATED, compresslevel=1, allowZip64=True) as archive:
        for member, (source_path, expected) in sorted(files.items()):
            before = source_path.stat()
            digest = hashlib.sha256()
            with source_path.open("rb") as source, archive.open(member, "w", force_zip64=True) as target:
                for chunk in iter(lambda: source.read(4 * 1024 * 1024), b""):
                    digest.update(chunk)
                    target.write(chunk)
            after = source_path.stat()
            if (before.st_size, before.st_mtime_ns, before.st_ctime_ns) != (after.st_size, after.st_mtime_ns, after.st_ctime_ns):
                raise ValueError(f"Source changed during export: {source_path}; partial archive is retained")
            sha = digest.hexdigest()
            if expected and expected != sha:
                raise ValueError(f"SHA256 differs from committed evidence: {source_path}")
            digests[member] = sha
            written += before.st_size
            if before.st_size > 64 * 1024**2:
                print(f"PACKED={member} progress_MiB={written / 1024**2:.1f}", flush=True)
        for member, content in generated.items():
            payload = (json.dumps(content, ensure_ascii=False, indent=2) if not isinstance(content, str) else content).encode()
            archive.writestr(member, payload)
            digests[member] = hashlib.sha256(payload).hexdigest()
        archive.writestr("EXPORT_MANIFEST.json", json.dumps({"schema_version": 1, "files": digests}, ensure_ascii=False, indent=2))
    print(f"CHECKING_ZIP={path.name}", flush=True)
    with zipfile.ZipFile(building) as archive:
        bad = archive.testzip()
        if bad:
            raise ValueError(f"ZIP CRC check failed: {bad}")
    building.rename(path)
    sha_path = path.with_name(path.name + ".sha256.txt")
    with sha_path.open("x", encoding="utf-8", newline="\n") as handle:
        handle.write(file_hash(path) + "  " + path.name + "\n")
    print(f"DOWNLOAD_ZIP={path}", flush=True)
    print(f"DOWNLOAD_SHA256_TXT={sha_path}", flush=True)
    print(f"ZIP_MiB={path.stat().st_size / 1024**2:.2f}", flush=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--code", default=os.environ.get("TV_EXPORT_CODE", ROOT_CODE))
    parser.add_argument("--experiment", default=os.environ.get("TV_EXPORT_ROOT", ROOT_EXPERIMENT))
    parser.add_argument("--output", default=os.environ.get("TV_EXPORT_DIR", "/home/work/user-job-dir/app"))
    parser.add_argument("--config", default=None)
    args = parser.parse_args(argv)
    code, root, output = (Path(x).resolve() for x in (args.code, args.experiment, args.output))
    if not all(p.is_dir() for p in (code, root, output)):
        raise ValueError("Code, experiment and download output directories must exist")
    tools = Path(__file__).resolve().parent
    config = Path(args.config).resolve() if args.config else code / "configs/modelmate.local.yaml"
    full, vectors, index = collect(code, root, config, tools)
    upper_bound = int(1.03 * sum(p.stat().st_size for p, _ in list(full.values()) + list(vectors.values()))) + 256 * 1024**2
    if shutil.disk_usage(output).free < upper_bound:
        raise ValueError(f"Download directory needs up to {upper_bound / 1024**3:.2f} GiB free before compression: {output}")
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ") + "-" + uuid.uuid4().hex[:8]
    vector_path = output / f"task-vectors-only-{stamp}.zip"
    full_path = output / f"task-vector-portable-full-{stamp}.zip"
    contract = {"model_id": index["config_used"]["model"]["id"], "fingerprint": index["base_fingerprint"]["fingerprint"],
                "files": index["base_fingerprint"]["sha256"], "expected_config": index["config_used"]["model"]["expected_config"],
                "enable_thinking": False, "intervention_scale": 1.0}
    standalone = [{"route": item["route"], "scope": item["scope"], "seed": item["seed"],
                   "task": item["task"], "layers": item["layers"],
                   "primary": item["standalone_exports"]["post_path"]["path"],
                   "exports": item["standalone_exports"],
                   "training_origin": item["standalone_exports"]["post_path"]["path"].rsplit("/", 1)[0] + "/training-origin.json"}
                  for item in index["training"]]
    vector_index = {"schema_version": 1, "base_contract": "BASE_MODEL_CONTRACT.json", "representations": standalone,
                    "primary_version": "post", "requires_experiment_code": False, "framework": "PyTorch tensor dictionary"}
    # Make the independently useful small archive available before the larger cache archive.
    write_zip(vector_path, vectors, {"VECTOR_INDEX.json": vector_index, "BASE_MODEL_CONTRACT.json": contract, "README.txt": VECTOR_README})
    write_zip(full_path, full, {"PORTABLE_INDEX.json": index, "BASE_MODEL_CONTRACT.json": contract})
    print("EXPORT_READY=PASS", flush=True)
    print(f"COMPLETED_TRAINING_REPRESENTATIONS={len(index['training'])}", flush=True)
    print("ORIGINAL_EXPERIMENT=READ_ONLY; NO_JOB_STOPPED; NO_FILE_DELETED", flush=True)


if __name__ == "__main__":
    main()
