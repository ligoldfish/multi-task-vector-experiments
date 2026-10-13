from __future__ import annotations

from collections import OrderedDict
import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Any


def file_sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(2**20), b""):
            h.update(chunk)
    return h.hexdigest()


def object_sha256(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str).encode()).hexdigest()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=path.name + ".", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as f:
            json.dump(value, f, ensure_ascii=False, indent=2, default=str)
            f.write("\n")
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def atomic_torch_save(path, value):
    import torch

    def portable(item):
        # Accelerator tensors can require device-specific pickle reducers even
        # when the reader requests map_location="cpu". Snapshot every tensor
        # before serialization without changing live optimizer or RNG state.
        if isinstance(item, torch.Tensor):
            return item.detach().cpu().clone()
        if isinstance(item, OrderedDict):
            result = OrderedDict((key, portable(part)) for key, part in item.items())
            # nn.Module.state_dict uses this attribute for load-time versions.
            if hasattr(item, "_metadata"):
                result._metadata = portable(item._metadata)
            return result
        if isinstance(item, dict):
            return {key: portable(part) for key, part in item.items()}
        if isinstance(item, list):
            return [portable(part) for part in item]
        if isinstance(item, tuple):
            parts = [portable(part) for part in item]
            return type(item)(*parts) if hasattr(item, "_fields") else tuple(parts)
        return item

    snapshot = portable(value)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=path.name + ".", dir=path.parent)
    os.close(fd)
    try:
        torch.save(snapshot, tmp)
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def read_json(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def read_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=path.name + ".", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)

