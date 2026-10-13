"""Explicit scientific configuration and final-evaluation provenance lock."""
from __future__ import annotations
from copy import deepcopy
from pathlib import Path
import os
import math
import yaml
from .io import atomic_json, file_sha256, object_sha256, read_json

_SOURCE_HASH_CACHE = {}


def load_config(path, overrides=None):
    with open(path, encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    for dotted, value in (overrides or {}).items():
        cursor = config
        bits = dotted.split(".")
        for bit in bits[:-1]:
            cursor = cursor.setdefault(bit, {})
        cursor[bits[-1]] = value
    for field, env in (("model", "TV_MODEL_PATH"),):
        if os.environ.get(env):
            config[field]["path"] = os.environ[env]
    if os.environ.get("TV_PERSISTENT_ROOT"):
        config["platform"]["persistent_root"] = os.environ["TV_PERSISTENT_ROOT"]
    if os.environ.get("TV_SEMANTIC_MODEL_PATH"):
        config["metrics"]["semantic_model_path"] = os.environ["TV_SEMANTIC_MODEL_PATH"]
    validate_config(config)
    return config


def validate_config(config):
    from .acceleration import validate_acceleration
    validate_acceleration(config.get("acceleration"))
    if config.get("schema_version") != 1:
        raise ValueError("Expected schema_version=1")
    train, loda = config["training"], config["loda"]
    checks = {"learning_rate": .001, "epochs": 5, "micro_batch_size": 2,
              "effective_batch_size": 16, "gradient_accumulation_steps": 8,
              "weight_decay_ltv": .01, "weight_decay_loda": 0., "warmup_steps": 0,
              "max_grad_norm": 1., "patience": 2, "max_prompt_tokens": 512, "max_target_tokens": 512}
    # Variations require an explicit supplement label, never a silent fallback.
    deviations = {k: {"document": v, "configured": train.get(k)} for k, v in checks.items() if train.get(k) != v}
    if deviations and not config.get("supplement_label"):
        raise ValueError(f"Scientific changes require supplement_label: {deviations}")
    if config["model"].get("enable_thinking") is not False:
        raise ValueError("The protocol requires enable_thinking=False")
    if loda["rank_general"] != 8 or loda["rank_isolated"] != 8:
        if not config.get("supplement_label"):
            raise ValueError("The primary LoDA ranks are 8+8")
    if config["evaluation"].get("do_sample") is not False:
        raise ValueError("Primary evaluation is greedy")
    if config["evaluation"].get("use_cache") is not True:
        raise ValueError("Primary LTV generation uses the prefill KV cache")
    numerical = {"model.dtype": "bfloat16", "loda.rho_max": .05,
                 "loda.recalibration_lambda": 3., "loda.regularization": 1e-4,
                 "loda.statistics_samples": 128, "loda.max_statistics_tokens": 64,
                 "loda.master_dtype": "float32", "loda.primary_version": "post",
                 "evaluation.intervention_scale": 1.0,
                 "evaluation.max_new_tokens.math": 512, "evaluation.max_new_tokens.logic": 32,
                 "evaluation.max_new_tokens.translation": 256, "evaluation.max_new_tokens.paraphrase": 128,
                 "evaluation.max_new_tokens.code_generation": 512, "evaluation.max_new_tokens.code_repair": 512}
    other_deviations = {}
    for dotted, expected in numerical.items():
        actual = config
        for key in dotted.split("."):
            actual = actual.get(key) if isinstance(actual, dict) else None
        if actual != expected:
            other_deviations[dotted] = {"document": expected, "configured": actual}
    if other_deviations and not config.get("supplement_label"):
        raise ValueError(f"Scientific changes require supplement_label: {other_deviations}")
    if config["analysis"]["ranks"] != [0, 1, 2]:
        raise ValueError("Discovery must include q=0,1,2")
    required = {
        "model.id": "Qwen/Qwen3-1.7B", "ltv.layers": [14],
        "model.expected_config": {"num_hidden_layers": 28, "hidden_size": 2048, "num_attention_heads": 16,
                                  "num_key_value_heads": 8, "head_dim": 128},
        "ltv.init_low": -.1, "ltv.init_high": .1,
        "data.tasks": ["math", "logic", "translation", "paraphrase"],
        "data.extension_tasks": ["code_generation", "code_repair"], "data.auxiliary_tasks": ["summary"],
        "data.split_seed": 17, "data.counts": {"train": 512, "tune": 128, "discovery": 128, "test": 256},
        "data.counts_by_task": {t: {"train": 300, "tune": 37, "discovery": 37, "test": 256}
                                for t in ("code_generation", "code_repair")},
        "data.proofwriter_world": "OWA", "data.proofwriter_max_depth": 3,
        "analysis.seeds": [17, 29, 43], "analysis.methods": ["mean", "svd"],
        "analysis.heldout_tasks": ["math", "logic", "translation", "paraphrase"],
        "analysis.ltv_layer_ablations": [[6], [22], [6, 14, 22]],
        "analysis.baselines": ["ties", "tsv", "knots"],
        "evaluation.variants": ["canonical", "output_format", "prompt_paraphrase", "label_recode"],
        "loda.a_scale": 1 / math.sqrt(3), "loda.epsilon": 1e-12,
        "loda.statistics_seed": 17,
        "analysis.sequential_orders": [
            ["math", "logic", "translation", "paraphrase"],
            ["logic", "translation", "paraphrase", "math"],
            ["translation", "paraphrase", "math", "logic"],
            ["paraphrase", "math", "logic", "translation"]],
    }
    scope_deviations = {}
    for dotted, expected in required.items():
        actual = config
        for key in dotted.split("."):
            actual = actual.get(key) if isinstance(actual, dict) else None
        if actual != expected:
            scope_deviations[dotted] = {"document": expected, "configured": actual}
    if config["data"].get("formal_test_size") not in (1000, "all"):
        scope_deviations["data.formal_test_size"] = "must be 1000 or all"
    for key, expected in (("train_general_a", False), ("isolated_enabled", True)):
        if loda.get(key, expected) != expected:
            scope_deviations["loda." + key] = "sequential-only switch is not a primary MT setting"
    if scope_deviations and not config.get("supplement_label"):
        raise ValueError(f"Scientific scope changes require supplement_label: {scope_deviations}")
    return config


def scientific_config(config):
    result = deepcopy(config)
    # Runtime placement/resume are not scientific degrees of freedom.
    result.get("platform", {}).pop("device", None)
    result.get("training", {}).pop("resume_from", None)
    # Engineering settings are frozen separately, never omitted from evidence.
    result.pop("acceleration", None)
    return result


def source_manifest(root):
    root = Path(root)
    entries = {}
    for folder in ("src", "configs", "deploy", "third_party", "ops"):
        for path in sorted((root / folder).rglob("*")):
            if path.is_file() and "__pycache__" not in path.parts and ".git" not in path.parts:
                entries[path.relative_to(root).as_posix()] = _source_hash(path)
    names = ["pyproject.toml", "docs/原始实验方案.md", "docs/补充协议与实现选择.md"]
    names += [p.name for p in sorted(root.glob("run_modelmate*.py"))]
    for name in names:
        path = root / name
        if path.is_file():
            entries[name] = _source_hash(path)
    return entries


def _source_hash(path):
    """Hash changed files once; do not reread official snapshots for every panel."""
    path = Path(path)
    stat = path.stat()
    key = (str(path.resolve()), stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns)
    if key not in _SOURCE_HASH_CACHE:
        _SOURCE_HASH_CACHE[key] = file_sha256(path)
    return _SOURCE_HASH_CACHE[key]


def create_lock(config, data_manifest, discovery_decisions, output_path, source_root):
    if not discovery_decisions:
        raise ValueError("A final lock needs recorded source-only discovery choices")
    payload = {"schema_version": 1, "config": scientific_config(config),
               "config_sha256": object_sha256(scientific_config(config)),
               "data_manifest_sha256": object_sha256(data_manifest),
               "discovery_decisions": discovery_decisions, "sources": source_manifest(source_root)}
    if config.get("acceleration", {}).get("enabled"):
        payload["execution"] = deepcopy(config["acceleration"])
        payload["execution_sha256"] = object_sha256(payload["execution"])
    payload["lock_sha256"] = object_sha256(payload)
    output_path = Path(output_path)
    if output_path.exists():
        previous = read_json(output_path)
        if previous != payload:
            raise ValueError("A frozen protocol cannot be overwritten; use a new run and an unexposed confirmation panel")
        return previous
    atomic_json(output_path, payload)
    return payload


def verify_lock(path, config, data_manifest, source_root=None):
    payload = read_json(path)
    digest = payload.pop("lock_sha256")
    if object_sha256(payload) != digest:
        raise ValueError("Protocol lock was changed")
    if payload["config_sha256"] != object_sha256(scientific_config(config)):
        raise ValueError("Configuration differs from the frozen protocol")
    if payload["data_manifest_sha256"] != object_sha256(data_manifest):
        raise ValueError("Dataset differs from the frozen protocol")
    if payload.get("execution", {}) != (config.get("acceleration", {}) if config.get("acceleration", {}).get("enabled") else {}):
        raise ValueError("Engineering execution settings differ from the frozen protocol")
    if source_root is not None and payload["sources"] != source_manifest(source_root):
        raise ValueError("Source files differ from the frozen protocol; create a new run")
    payload["lock_sha256"] = digest
    return payload
