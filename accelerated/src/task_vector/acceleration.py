"""Explicit engineering choices; scientific hyperparameters remain unchanged."""
from __future__ import annotations
import re

DEFAULT_ACCELERATION = {
    "enabled": True,
    "evaluation_devices": [f"npu:{i}" for i in range(8)],
    "generation_batch_policy": "validated",
    "generation_batch_candidates": [1, 2, 4, 8],
    "calibration_max_samples": 8,
    "worker_cpu_threads": 2,
    "training_cpu_threads": 8,
    "bootstrap_workers": 1,
    "heartbeat_seconds": 30,
    "parallel_training": True,
    "parallel_sequential_orders": True,
}


def validate_acceleration(settings):
    if settings is None:
        return
    if not isinstance(settings, dict) or set(settings) - set(DEFAULT_ACCELERATION):
        raise ValueError("Unknown engineering acceleration settings")
    for key in ("enabled", "parallel_training", "parallel_sequential_orders"):
        if key in settings and type(settings[key]) is not bool:
            raise ValueError(f"acceleration.{key} must be boolean")
    devices = settings.get("evaluation_devices", DEFAULT_ACCELERATION["evaluation_devices"])
    if not isinstance(devices, list) or not devices or any(not isinstance(device, str) for device in devices) or len(devices) != len(set(devices)):
        raise ValueError("Evaluation devices must be a nonempty distinct list")
    for device in devices:
        if device != "cpu" and not re.fullmatch(r"(?:npu|cuda):[0-9]+", device):
            raise ValueError(f"Invalid evaluation device {device!r}")
    if len({device.split(":")[0] for device in devices}) != 1:
        raise ValueError("Evaluation workers cannot mix device backends")
    if settings.get("generation_batch_policy", "strict") not in {"strict", "validated", "audited"}:
        raise ValueError("Batch policy must be strict, validated or audited")
    candidates = settings.get("generation_batch_candidates", [1, 2, 4, 8])
    if not isinstance(candidates, list) or not candidates or candidates[0] != 1 or candidates != sorted(set(candidates)) or any(type(x) is not int or x < 1 or x > 32 for x in candidates):
        raise ValueError("Batch candidates must be sorted distinct integers in 1..32 starting at 1")
    for key in ("calibration_max_samples", "worker_cpu_threads", "training_cpu_threads", "bootstrap_workers", "heartbeat_seconds"):
        value = settings.get(key, DEFAULT_ACCELERATION[key])
        if type(value) is not int or value < 1:
            raise ValueError(f"acceleration.{key} must be a positive integer")
    if any(settings.get(key, DEFAULT_ACCELERATION[key]) > 32 for key in ("bootstrap_workers", "worker_cpu_threads", "training_cpu_threads")):
        raise ValueError("CPU worker/thread counts must be at most 32")
