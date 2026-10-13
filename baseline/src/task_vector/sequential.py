"""Sequential LoDA with committed training and independently cached split evaluation.

Preserves first-stage A training and official double statistics accumulation.
Evaluation-only restores actual native-dtype merged QKV, without retraining.
"""
from __future__ import annotations
from copy import deepcopy
from pathlib import Path
import gc
import re
import torch
from .io import (atomic_json, atomic_torch_save, file_sha256, object_sha256,
                 read_json, write_jsonl)
from .loda import merge_representation_
from .representations import LowRankFactor, Representation, linear_combination
from .subspaces import (Statistics, build_subspaces, collect_statistics,
                        load_statistics, recalibration_diagonal, save_statistics)


def cyclic_orders(tasks):
    tasks = tuple(tasks)
    if len(tasks) != 4 or len(set(tasks)) != 4:
        raise ValueError("The specified sequential control uses four distinct MVP tasks")
    return [tasks[offset:] + tasks[:offset] for offset in range(4)]


def _sequential_recalibrate(raw, current, previous, config, first_stage):
    cfg = config.get("loda", {})
    factors, lambda_values = {}, {}
    for name, terms in raw.factors.items():
        layer = name.rsplit(".self_attn.", 1)[0]
        if len(terms) != 2:
            raise ValueError("Sequential LoDA export must retain ordered General/Isolated factors")
        diagonal = (torch.ones(terms[0].A.shape[0]) if first_stage else
                    recalibration_diagonal(terms[0].A, current[layer], previous[layer],
                                           float(cfg.get("recalibration_lambda", 3)),
                                           float(cfg.get("epsilon", 1e-12))))
        lambda_values[layer] = diagonal.tolist()
        factors[name] = [LowRankFactor(terms[0].B * diagonal.unsqueeze(0), terms[0].A), terms[1]]
    metadata = {**raw.metadata, "route": "sequential_LoDA", "version": "post",
                "lambda": lambda_values, "statistics_source": "post_training_adapted_model",
                "past_statistics_accumulation": "post_training_plus_post_merge",
                "first_stage_general_A_trainable": bool(first_stage)}
    return Representation("parameter", raw.task, raw.seed, factors=factors, metadata=metadata)


def _training_identity(config):
    """Evaluation split and operational placement do not change training."""
    train = deepcopy(config.get("training", {}))
    for key in ("resume_from", "save_every_steps"):
        train.pop(key, None)
    return {"training": train, "model": config.get("model", {}),
            "loda": config.get("loda", {}), "ltv": config.get("ltv", {}),
            "train_samples": config.get("sequential", {}).get("train_samples"),
            "implementation_sha256": object_sha256({path.name: file_sha256(path)
                for path in sorted(Path(__file__).resolve().parent.glob("*.py"))})}


def _sample_digest(samples):
    return object_sha256([s.to_dict() if hasattr(s, "to_dict") else vars(s) for s in samples])


def _artifact_hashes(directory, names):
    directory = Path(directory).resolve()
    result = {}
    for name in names:
        path = Path(name)
        path = path.resolve() if path.is_absolute() else (directory / path).resolve()
        if not path.is_relative_to(directory) or not path.is_file():
            raise ValueError(f"Sequential artifact is missing or outside its stage: {path}")
        result[path.relative_to(directory).as_posix()] = file_sha256(path)
    return result


def _verify_artifacts(directory, hashes):
    if not hashes:
        raise ValueError("Sequential completion has no artifact manifest")
    if _artifact_hashes(directory, hashes) != hashes:
        raise ValueError(f"Sequential artifact changed after completion: {directory}")


def _read_stage(stage_dir, expected):
    marker = read_json(stage_dir / "stage-completed.json")
    request = marker.get("request", {})
    if (marker.get("schema_version") != 2 or marker.get("status") != "complete"
            or marker.get("stage_signature") != object_sha256(request)
            or any(request.get(key) != value for key, value in expected.items())):
        raise ValueError(f"Sequential stage provenance differs: {stage_dir}")
    _verify_artifacts(stage_dir, marker.get("artifacts", {}))
    return marker


def _restore_weights(model, path, original_weights):
    """Copy recorded BF16/FP32 tensors; never reconstruct rounded merges."""
    weights = torch.load(path, map_location="cpu", weights_only=True)
    if set(weights) != set(original_weights):
        raise ValueError("Sequential merged weights do not cover exactly the base QKV projections")
    with torch.no_grad():
        for name, value in weights.items():
            parameter = model.get_submodule(name).weight
            if value.shape != parameter.shape or value.dtype != parameter.dtype:
                raise ValueError(f"Sequential weight shape/dtype differs for {name}")
            parameter.copy_(value.to(parameter.device))


def _stage_config(config, order, index, signature, merged_hash=None, split=None):
    result = deepcopy(config)
    result.setdefault("training", {}).pop("resume_from", None)
    result.setdefault("loda", {}).update(train_general_a=index == 0, isolated_enabled=index != 0)
    result.setdefault("sequential", {}).update(order=list(order), stage=index,
        stage_signature=signature, merged_weights_sha256=merged_hash, split=split)
    return result


def _train_stage(model, tokenizer, training, tuning, *, config, order, index,
                 seed, stage_dir, request, previous, updates, original_weights,
                 train_fn, device):
    from .evaluation import intervention_context
    signature = object_sha256(request)
    request_path = stage_dir / "stage-request.json"
    if request_path.exists():
        if read_json(request_path) != request:
            raise ValueError(f"Cannot resume a sequential stage with different inputs: {stage_dir}")
    else:
        if any(stage_dir.iterdir()):
            raise ValueError(f"Unversioned sequential stage exists; choose a new output root: {stage_dir}")
        atomic_json(request_path, request)
    stage_config = _stage_config(config, order, index, signature)
    current_before = collect_statistics(model, tokenizer, training, stage_config, device)
    save_statistics(current_before, stage_dir / "statistics_before_training.pt")
    if previous is None:
        previous = Statistics({k: torch.zeros_like(v) for k, v in current_before.items()},
                              metadata={"past_tasks": [], "accumulation": "none"})
    save_statistics(previous, stage_dir / "past_statistics.pt")
    stats_inputs = {order[index]: current_before}
    if index:
        stats_inputs["__past__"] = previous
    subspaces = build_subspaces(stats_inputs, order[index], stage_config, seen_tasks=list(stats_inputs))
    for spec in subspaces.values():
        spec["metadata"].update(route="sequential_LoDA", statistics_source="current_merged_model",
            past_tasks=list(order[:index]), stage=index,
            past_statistics_accumulation="post_training_plus_post_merge")
    train_provenance = {**request["provenance"], "route": "sequential_LoDA",
        "order": list(order), "stage": index, "stage_signature": signature,
        "parent_merge_sha256": request["parent_merge_sha256"],
        "train_samples_sha256": request["train_samples_sha256"],
        "tune_samples_sha256": request["tune_samples_sha256"]}
    result_path = stage_dir / "training-result.json"
    if result_path.exists():
        record = read_json(result_path)
        if record.get("stage_signature") != signature:
            raise ValueError("Sequential training export belongs to another stage")
        _verify_artifacts(stage_dir, record.get("artifacts", {}))
        trained = record["training"]
    else:
        state_path = stage_dir / "training-state.json"
        state = read_json(state_path) if state_path.exists() else {}
        if state.get("status") == "exported_uncommitted":
            bound = state.get("provenance", {})
            if (any(bound.get(key) != value for key, value in train_provenance.items())
                    or state.get("provenance_sha256") != object_sha256(bound)):
                raise ValueError("Uncommitted sequential training export provenance differs")
            trained = {key: value for key, value in state.items() if key != "status"}
        else:
            checkpoint = stage_dir / "last.pt"
            if checkpoint.exists():
                stage_config["training"]["resume_from"] = str(checkpoint)
            trained = train_fn(model, tokenizer, training, tuning, task=order[index], route="loda",
                seed=seed, output_dir=stage_dir, config=stage_config, device=device,
                subspaces=subspaces, provenance=train_provenance)
        raw = Representation.load(trained["raw_path"])
        raw.metadata.update({**request["provenance"], "route": "sequential_LoDA", "stage": index,
            "order": list(order), "past_tasks": list(order[:index]), "stage_signature": signature,
            "parent_merge_sha256": request["parent_merge_sha256"]})
        raw.save(trained["raw_path"])
        # The independent post export will be replaced by sequential lambda.
        names = [trained["raw_path"]] + [trained[k] for k in
            ("initial_path", "best_checkpoint", "last_checkpoint", "best_adapter_path") if trained.get(k)]
        atomic_json(result_path, {"stage_signature": signature, "training": trained,
                                 "artifacts": _artifact_hashes(stage_dir, names)})
    raw = Representation.load(trained["raw_path"])
    with intervention_context(model, raw):
        current_adapted = collect_statistics(model, tokenizer, training, stage_config, device)
    save_statistics(current_adapted, stage_dir / "statistics_after_training.pt")
    post = _sequential_recalibrate(raw, current_adapted, previous, stage_config, index == 0)
    post.save(trained["post_path"])
    merge_representation_(model, post)
    current_merged = collect_statistics(model, tokenizer, training, stage_config, device)
    save_statistics(current_merged, stage_dir / "statistics_after_merge.pt")
    previous = Statistics({k: previous[k] + current_adapted[k] + current_merged[k] for k in previous},
        metadata={"past_tasks": list(order[:index + 1]),
                  "accumulation": "official_post_training_plus_post_merge",
                  "source": "methods/dlora.py:194-219"})
    save_statistics(previous, stage_dir / "statistics_past_after_stage.pt")
    cumulative = linear_combination(updates + [post], [1.] * (len(updates) + 1),
                                    task=f"sequential_through_{order[index]}")
    cumulative.metadata.update({**request["provenance"], "route": "sequential_LoDA",
        "order": list(order), "stage": index, "stage_signature": signature,
        "relative_to": "original_theta0", "merge_dtype": str(next(model.parameters()).dtype),
        "merge_rounding": "native base dtype; see exact merged weight artifact"})
    cumulative.save(stage_dir / "cumulative_update.pt")
    atomic_torch_save(stage_dir / "merged_projection_weights.pt", {
        name: model.get_submodule(name).weight.detach().cpu().clone() for name in original_weights})
    atomic_torch_save(stage_dir / "actual_cumulative_update.pt", {
        "relative_to": "original_theta0", "includes_native_merge_rounding": True,
        "base_fingerprint": request["provenance"]["base_fingerprint"],
        "matrices": {name: model.get_submodule(name).weight.detach().float().cpu() - value
                     for name, value in original_weights.items()}})
    stage = {"stage": index, "task": order[index], "train_ids": request["train_ids"],
        "tune_ids": request["tune_ids"], "base_history": list(order[:index]),
        "stage_signature": signature, "training": trained,
        "statistics_protocol": "official_double_accumulation",
        "first_stage_general_A_trainable": index == 0, "isolated_enabled": index != 0}
    atomic_json(stage_dir / "stage.json", stage)
    # An exported adapter alone does not commit a stage: merge and stats must exist.
    atomic_json(stage_dir / "training.json", {**trained, "status": "complete",
        "stage_signature": signature, "provenance": train_provenance})
    names = ["stage-request.json", "training-result.json", "stage.json", "training.json",
        "statistics_before_training.pt", "past_statistics.pt", "statistics_after_training.pt",
        "statistics_after_merge.pt", "statistics_past_after_stage.pt", "cumulative_update.pt",
        "merged_projection_weights.pt", "actual_cumulative_update.pt",
        trained["raw_path"], trained["post_path"]]
    names += [trained[k] for k in ("initial_path", "best_checkpoint", "last_checkpoint", "best_adapter_path") if trained.get(k)]
    marker = {"schema_version": 2, "status": "complete", "request": request,
              "stage_signature": signature, "artifacts": _artifact_hashes(stage_dir, names)}
    atomic_json(stage_dir / "stage-completed.json", marker)
    return marker


def _evaluate_stage(model, tokenizer, samples, *, config, order, index, seed,
                    split, stage_dir, marker, evaluate_fn, access_fn, device):
    from .evaluation import generate_predictions
    from .metrics import score_predictions
    target = samples[0].task if samples else None
    if target is None:
        raise ValueError("Sequential evaluation split is empty")
    merge_hash = marker["artifacts"]["merged_projection_weights.pt"]
    context = {"route": "sequential_LoDA", "order": list(order), "stage": index,
        "seed": seed, "task": target, "split": split, "stage_signature": marker["stage_signature"],
        "merged_weights_sha256": merge_hash, "stage_dir": str(stage_dir),
        "base_fingerprint": marker["request"]["provenance"]["base_fingerprint"]}
    # Locks/access ledgers apply to cached test results too.
    if access_fn is not None:
        access_fn(target, split, samples, context)
    request = {**context, "stage_dir": None, "samples_sha256": _sample_digest(samples),
        "sample_ids_sha256": object_sha256([s.id for s in samples]),
        "evaluation": config.get("evaluation", {}), "metrics": config.get("metrics", {}),
        "provenance": config.get("sequential", {}).get("provenance", {})}
    signature = object_sha256(request)
    panel_dir = stage_dir / "evaluation" / split / target
    panel_dir.mkdir(parents=True, exist_ok=True)
    completed = panel_dir / "evaluation-completed.json"
    if completed.exists():
        cached = read_json(completed)
        if cached.get("schema_version") != 2 or cached.get("signature") != signature:
            raise ValueError(f"Sequential evaluation cache provenance differs: {panel_dir}")
        _verify_artifacts(panel_dir, cached.get("artifacts", {}))
        return read_json(panel_dir / "metrics.json")
    request_path = panel_dir / "evaluation-request.json"
    if request_path.exists() and read_json(request_path) != request:
        raise ValueError(f"Cannot overwrite an incompatible sequential evaluation: {panel_dir}")
    atomic_json(request_path, request)
    stage_config = _stage_config(config, order, index, marker["stage_signature"], merge_hash, split)
    stage_config["sequential"]["evaluation_dir"] = str(panel_dir)
    if evaluate_fn is None:
        rows = generate_predictions(model, tokenizer, samples, None, stage_config, device)
        result = {"metrics": score_predictions(target, rows, stage_config), "rows": rows,
                  "split": split, "evaluated_on_actual_merged_model": True}
    else:
        result = evaluate_fn(model, tokenizer, samples, stage_config, device)
    atomic_json(panel_dir / "metrics.json", result)
    names = ["evaluation-request.json", "metrics.json"]
    if isinstance(result, dict) and isinstance(result.get("rows"), list):
        write_jsonl(panel_dir / "predictions.jsonl", result["rows"])
        names.append("predictions.jsonl")
    atomic_json(completed, {"schema_version": 2, "status": "complete", "signature": signature,
        "request": request, "count": len(samples), "artifacts": _artifact_hashes(panel_dir, names)})
    return result


def run_sequential(config, *, device, model_loader=None, train_fn=None,
                   split="test", evaluation_only=False, evaluate_fn=None, access_fn=None):
    """Commit training, then evaluate a requested split independently.

    split=None commits training only. evaluation_only requires every stage commit
    and never loads train/tune data, collects statistics, or calls the trainer.
    evaluate_fn(model, tokenizer, samples, stage_config, device) receives actual
    split samples. access_fn(task, split, samples, context) runs on cache hits too.
    Supply immutable base/data/source fingerprints in sequential.provenance.
    """
    from filelock import FileLock
    from .data import load_samples
    from .modeling import load_model
    from .training import train_task
    if split is not None and not re.fullmatch(r"[A-Za-z0-9_]+", split):
        raise ValueError("Invalid sequential evaluation split")
    if evaluation_only and split is None:
        raise ValueError("Evaluation-only requires a split")
    model_loader, train_fn = model_loader or load_model, train_fn or train_task
    seq, runtime = config.get("sequential", {}), config.get("runtime", {})
    tasks = tuple(seq.get("tasks", config.get("tasks", ("math", "logic", "translation", "paraphrase"))))
    allowed = cyclic_orders(tasks)
    selected = tuple(seq["order"]) if seq.get("order") else None
    if selected is not None and selected not in allowed:
        raise ValueError("Sequential order must be one of the four locked cyclic rotations")
    orders = [selected] if selected else allowed
    seed = int(seq.get("seed", 17))
    data_root = seq.get("data_root", runtime.get("data_root", config.get("data", {}).get("prepared_root")))
    output_root = seq.get("output_root", runtime.get("output_root"))
    provenance = deepcopy(seq.get("provenance", {}))
    if not provenance.get("base_fingerprint"):
        raise ValueError("Sequential control requires immutable provenance.base_fingerprint")
    if data_root is None or output_root is None:
        raise ValueError("Sequential control requires prepared data_root and persistent output_root")
    output_root = Path(output_root).resolve()
    training_hash = object_sha256(_training_identity(config))
    reports = []
    for order in orders:
        order_dir = Path(output_root) / "sequential" / f"seed{seed}" / "__".join(order)
        order_dir.mkdir(parents=True, exist_ok=True)
        with FileLock(str(order_dir / ".sequential.lock")):
            identity = {"schema_version": 2, "route": "sequential_LoDA", "seed": seed,
                "order": list(order), "training_config_sha256": training_hash, "provenance": provenance}
            identity_path = order_dir / "order-request.json"
            if identity_path.exists():
                if read_json(identity_path) != identity:
                    raise ValueError(f"Sequential order provenance differs: {order_dir}")
            elif evaluation_only:
                raise ValueError(f"Sequential training has not been committed: {order_dir}")
            else:
                if any(p.name != ".sequential.lock" for p in order_dir.iterdir()):
                    raise ValueError(f"Unversioned sequential order exists; choose a new output root: {order_dir}")
                atomic_json(identity_path, identity)
            if evaluation_only and any(not (order_dir / f"stage{i}_{t}" / "stage-completed.json").is_file()
                                       for i, t in enumerate(order)):
                raise ValueError(f"Evaluation-only needs every completed sequential stage: {order_dir}")
            model, tokenizer = model_loader(config, device)
            original_weights = {name: (None if evaluation_only else module.weight.detach().float().cpu().clone())
                for name, module in model.named_modules()
                if name.endswith((".self_attn.q_proj", ".self_attn.k_proj", ".self_attn.v_proj"))}
            try:
                if not original_weights:
                    raise ValueError("Sequential model has no QKV projections")
                previous, updates, markers, stages, parent_hash = None, [], [], [], None
                for index, task in enumerate(order):
                    stage_dir = order_dir / f"stage{index}_{task}"
                    stage_dir.mkdir(parents=True, exist_ok=True)
                    expected = {"order": list(order), "stage": index, "task": task, "seed": seed,
                        "parent_merge_sha256": parent_hash, "training_config_sha256": training_hash,
                        "provenance": provenance}
                    if not evaluation_only:
                        training = load_samples(data_root, task, "train")
                        tuning = load_samples(data_root, task, "tune")
                        if seq.get("train_samples") is not None:
                            requested = int(seq["train_samples"])
                            if requested < int(config.get("loda", {}).get("statistics_samples", 128)) or len(training) < requested:
                                raise ValueError("Sequential train sample count is insufficient for calibration")
                            training = training[:requested]
                        expected.update(train_samples_sha256=_sample_digest(training),
                            tune_samples_sha256=_sample_digest(tuning),
                            train_ids=[s.id for s in training], tune_ids=[s.id for s in tuning])
                    if (stage_dir / "stage-completed.json").exists():
                        marker = _read_stage(stage_dir, expected)
                    else:
                        marker = _train_stage(model, tokenizer, training, tuning, config=config,
                            order=order, index=index, seed=seed, stage_dir=stage_dir, request=expected,
                            previous=previous, updates=updates, original_weights=original_weights,
                            train_fn=train_fn, device=device)
                    _restore_weights(model, stage_dir / "merged_projection_weights.pt", original_weights)
                    stage = read_json(stage_dir / "stage.json")
                    if not evaluation_only:
                        previous = load_statistics(stage_dir / "statistics_past_after_stage.pt")
                        updates.append(Representation.load(stage["training"]["post_path"]))
                    markers.append(marker)
                    stages.append(stage)
                    parent_hash = marker["artifacts"]["merged_projection_weights.pt"]
                training_report = {**identity, "status": "complete", "stages": stages,
                    "separate_from_independent_LoDA_MT": True}
                order_marker = order_dir / "order-completed.json"
                if not order_marker.exists():
                    atomic_json(order_dir / "report.json", training_report)
                    atomic_json(order_marker, {"schema_version": 2, "status": "complete",
                        "order_signature": object_sha256(identity),
                        "artifacts": _artifact_hashes(order_dir, ["order-request.json", "report.json"] +
                            [f"stage{i}_{t}/stage-completed.json" for i, t in enumerate(order)])})
                else:
                    completed = read_json(order_marker)
                    if (completed.get("schema_version") != 2 or completed.get("status") != "complete"
                            or completed.get("order_signature") != object_sha256(identity)):
                        raise ValueError("Sequential order completion provenance differs")
                    _verify_artifacts(order_dir, completed.get("artifacts", {}))
                evaluations = []
                if split is not None:
                    for index, (task, marker) in enumerate(zip(order, markers)):
                        stage_dir = order_dir / f"stage{index}_{task}"
                        _restore_weights(model, stage_dir / "merged_projection_weights.pt", original_weights)
                        panel = {}
                        for target in order[:index + 1]:
                            samples = load_samples(data_root, target, split)
                            panel[target] = _evaluate_stage(model, tokenizer, samples, config=config,
                                order=order, index=index, seed=seed, split=split, stage_dir=stage_dir,
                                marker=marker, evaluate_fn=evaluate_fn, access_fn=access_fn, device=device)
                        evaluations.append({"stage": index, "task": task, "metrics": panel})
                    atomic_json(order_dir / "evaluation" / split / "report.json", {
                        "route": "sequential_LoDA", "seed": seed, "order": list(order),
                        "split": split, "status": "complete", "stages": evaluations})
                reports.append({**training_report, "split": split,
                                "evaluation_only": evaluation_only, "evaluation": evaluations})
            finally:
                del model, original_weights
                gc.collect()
                if str(device).startswith("npu") and hasattr(torch, "npu"):
                    torch.npu.empty_cache()
    return reports
