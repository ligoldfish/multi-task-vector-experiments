"""Ascend placement, probes and run evidence; no CUDA substitution on NPU runs."""
from __future__ import annotations
from contextlib import nullcontext
from pathlib import Path
import importlib.metadata
import os
import platform
import random
import subprocess
import sys
import time
import numpy as np
import torch
from .io import atomic_json


def configure_device(name):
    if str(name).startswith("npu"):
        try:
            import torch_npu  # noqa: F401
        except ImportError as exc:
            raise RuntimeError("Use the supplied ModelMate torch/torch_npu image; torch_npu is missing") from exc
        if not torch.npu.is_available():
            raise RuntimeError("NPU requested but unavailable")
        torch.npu.set_device(name)
    elif str(name).startswith("cuda"):
        torch.cuda.set_device(name)
    elif str(name) != "cpu":
        raise ValueError(f"Unknown device {name}")
    return torch.device(name)


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if hasattr(torch, "npu") and torch.npu.is_available():
        torch.npu.manual_seed_all(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def synchronize(device):
    name = str(device)
    if name.startswith("npu"):
        torch.npu.synchronize()
    elif name.startswith("cuda"):
        torch.cuda.synchronize()


def autocast_context(device, dtype=torch.bfloat16):
    name = str(device).split(":")[0]
    return torch.autocast(name, dtype=dtype) if name in {"npu", "cuda"} else nullcontext()


def _command(command):
    try:
        result = subprocess.run(command, capture_output=True, text=True, timeout=20)
        return {"returncode": result.returncode, "stdout": result.stdout, "stderr": result.stderr}
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {"error": str(exc)}


def memory_evidence(device):
    kind = str(device).split(":")[0]
    api = getattr(torch, kind, None)
    result = {}
    if kind in {"npu", "cuda"} and api is not None:
        for name in ("memory_allocated", "memory_reserved", "max_memory_allocated", "max_memory_reserved"):
            try:
                result[name] = int(getattr(api, name)(device))
            except (AttributeError, RuntimeError, TypeError):
                result[name] = None
        try:
            free, total = api.mem_get_info(device)
            result.update(free_bytes=int(free), total_bytes=int(total))
        except (AttributeError, RuntimeError, TypeError):
            result["capacity_probe"] = "unavailable; inspect npu-smi output"
    return result


def preflight(config, output_dir):
    started = time.time()
    device = configure_device(config["platform"]["device"])
    packages = {}
    for name in ("torch", "torch-npu", "transformers", "datasets", "sacrebleu", "sentence-transformers", "vllm-ascend", "verl"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    root = Path(config["platform"]["persistent_root"])
    root.mkdir(parents=True, exist_ok=True)
    probe = root / f".write-probe-{os.getpid()}"
    probe.write_text("task-vector persistence probe", encoding="utf-8")
    probe.unlink()
    count = torch.npu.device_count() if str(device).startswith("npu") else (torch.cuda.device_count() if str(device).startswith("cuda") else 0)
    cards = []
    for index in range(count):
        dev = f"{str(device).split(':')[0]}:{index}"
        cards.append({"index": index, "memory": memory_evidence(dev)})
    record = {"python": sys.version, "host": platform.platform(), "packages": packages,
              "device": str(device), "device_count": count, "cards": cards,
              "user_reported": config["platform"], "npu_smi": _command(["npu-smi", "info"]) if str(device).startswith("npu") else None,
              "persistent_root": str(root.resolve()), "persistence_write_verified": True,
              "elapsed_seconds": time.time() - started}
    model_path = config["model"].get("path")
    if not model_path or not (Path(model_path) / "config.json").is_file():
        record["model_status"] = "missing explicit local model path; run assets or set model.path"
    else:
        record["model_status"] = "local config found"
    if str(device).startswith("npu"):
        if not packages["torch"].startswith("2.8.") or not (packages["torch-npu"] or "").startswith("2.8."):
            record["version_compatibility"] = "outside user-reported torch2.8/torch_npu2.8; explicitly review"
        else:
            record["version_compatibility"] = "matches reported major/minor; CANN validated by runtime smoke"
    atomic_json(Path(output_dir) / "preflight.json", record)
    return record
