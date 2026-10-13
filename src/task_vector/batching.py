"""Source-only generation batch benchmarks and explicit numerical policies.

Batch shape can change BF16 argmax decisions. A finite tune-set match is useful
engineering evidence, but never proves identical outputs on arbitrary held-out
prompts. The strict policy therefore uses batch one. The audited policy compares
every actual output against batch one and returns the reference on a mismatch.
The validated policy permits ordinary finite BF16 output differences and selects
the fastest source-calibrated valid batch. Token/stop differences are diagnostic;
its receipt explicitly disclaims universal equality.
"""
from __future__ import annotations

import hashlib
import json
import gc
import time

from .evaluation import DEFAULT_LIMITS, NonFiniteGenerationError, generate_predictions


MODES = {"strict", "validated", "audited"}
_SIGNATURE_FIELDS = ("id", "prediction_token_ids", "stop_reason", "truncated",
                     "generated_tokens", "max_new_tokens")


class GenerationContractError(ValueError):
    """A candidate violates coverage, generation budget or EOS semantics."""


def generation_signature(rows):
    """Hash the exact greedy tokens and stopping contract, without task scores."""
    content = [{key: row[key] for key in _SIGNATURE_FIELDS} for row in rows]
    return hashlib.sha256(json.dumps(content, ensure_ascii=False, sort_keys=True,
                                     separators=(",", ":")).encode("utf-8")).hexdigest()


def _validate_rows(model, tokenizer, samples, rows, config):
    if [row.get("id") for row in rows] != [sample.id for sample in samples]:
        raise GenerationContractError("Generated rows do not cover the source IDs exactly in order")
    if not rows:
        return
    eos = getattr(getattr(model, "generation_config", None), "eos_token_id", None)
    eos = eos if eos is not None else getattr(tokenizer, "eos_token_id", None)
    if eos is None:
        raise GenerationContractError("Generation validation needs the model/tokenizer EOS IDs")
    eos_ids = {int(eos)} if isinstance(eos, int) else set(map(int, eos))
    budgets = config.get("evaluation", {}).get("max_new_tokens", DEFAULT_LIMITS)
    for sample, row in zip(samples, rows):
        tokens = row.get("prediction_token_ids")
        budget = int(budgets.get(sample.task, DEFAULT_LIMITS.get(sample.task, 512)))
        if (not isinstance(tokens, list) or not tokens
                or any(type(token) is not int or token < 0 for token in tokens)
                or row.get("max_new_tokens") != budget or len(tokens) > budget
                or row.get("generated_tokens") != len(tokens)):
            raise GenerationContractError("Generated tokens violate the fixed generation budget")
        ended = tokens[-1] in eos_ids
        if (any(token in eos_ids for token in tokens[:-1])
                or row.get("stop_reason") != ("eos" if ended else "length")
                or row.get("truncated") is not (not ended)
                or (not ended and len(tokens) != budget)):
            raise GenerationContractError("Generated rows violate greedy EOS/length stopping semantics")


def _differences(reference, candidate):
    token_samples = stop_samples = positions = 0
    for left, right in zip(reference, candidate):
        a, b = left["prediction_token_ids"], right["prediction_token_ids"]
        token_samples += a != b
        stop_samples += (left["stop_reason"], left["truncated"], len(a)) != (right["stop_reason"], right["truncated"], len(b))
        positions += sum(x != y for x, y in zip(a, b)) + abs(len(a) - len(b))
    return {"token_difference_samples": token_samples, "stop_difference_samples": stop_samples,
            "different_token_positions": positions}


def _oom(error):
    # Do not convert arbitrary implementation or device errors into silent
    # fallbacks. torch_npu and torch can report OOM using different exception
    # classes; the standard diagnostic phrase is common to both.
    return "out of memory" in str(error).lower() or type(error).__name__ == "OutOfMemoryError"


def _device_backend(device):
    import torch
    return getattr(torch, str(device).split(":", 1)[0], None)


def _synchronize(device):
    backend = _device_backend(device)
    if backend is not None and hasattr(backend, "synchronize"):
        backend.synchronize()


def _empty_cache(device):
    gc.collect()
    backend = _device_backend(device)
    if backend is not None and hasattr(backend, "empty_cache"):
        backend.empty_cache()


def _source_subset(tokenizer, samples, *, task, variant, maximum):
    from .prompts import render_prompt
    if maximum < 1 or not samples:
        raise ValueError("Batch calibration requires a nonempty source sample pool")
    if any(sample.task != task for sample in samples):
        raise ValueError("Batch calibration must use one target task")
    if len({sample.id for sample in samples}) != len(samples):
        raise ValueError("Batch calibration source IDs must be unique")
    forbidden = {"test", "test_formal", "test_confirmation", "discovery"}
    if any(str(getattr(sample, "official_split", "")).lower() in forbidden for sample in samples):
        raise ValueError("Held-out examples cannot be used for generation batch calibration")
    by_length = sorted(samples, key=lambda sample: (
        len(tokenizer(render_prompt(tokenizer, sample, variant=variant),
                      add_special_tokens=False)["input_ids"]), sample.id))
    count = min(maximum, len(by_length))
    # Include short, median and long prompts deterministically. No reference
    # labels, test scores or generated answer quality enter the selection.
    indexes = [0] if count == 1 else [i * (len(by_length) - 1) // (count - 1) for i in range(count)]
    return [by_length[index] for index in indexes]


def calibrate_generation_batches(model, tokenizer, calibration_samples, config, device, *,
                                 task, variant="canonical", source_split="tune",
                                 candidates=(1, 2, 4, 8), mode="strict", max_samples=8,
                                 progress_callback=None):
    """Benchmark BASE on train/tune; return a JSON-serializable frozen policy.

    The coordinator must persist and reuse ONE policy for a task and prompt
    variant across routes, conditions, seeds and worker shards. Calibration is
    never repeated per condition. For a scientifically held-out target, use an
    explicit strict policy instead of reading that target's tune examples.
    """
    if source_split not in {"train", "tune"}:
        raise ValueError("Batch calibration is restricted to train or tune")
    if mode not in MODES:
        raise ValueError(f"Unknown generation batch policy: {mode}")
    values = sorted({int(candidate) for candidate in candidates} | {1})
    if any(value < 1 for value in values):
        raise ValueError("Generation batch candidates must be positive")
    selected = _source_subset(tokenizer, list(calibration_samples), task=task,
                              variant=variant, maximum=int(max_samples))
    # A candidate larger than the selected pool would never actually exercise
    # its advertised shape and must not be released as a calibrated batch.
    values = [value for value in values if value <= len(selected)]
    records = []
    reference_digest = None
    reference_rows = None
    for batch_size in values:
        _synchronize(device)
        started = time.perf_counter()
        try:
            rows = generate_predictions(model, tokenizer, selected, None, config,
                device, variant, batch_size_override=batch_size,
                progress_callback=progress_callback, check_finite_logits=True)
            _synchronize(device)
            elapsed = time.perf_counter() - started
            _validate_rows(model, tokenizer, selected, rows, config)
            digest = generation_signature(rows)
            if batch_size == 1:
                reference_digest = digest
                reference_rows = rows
            records.append({"batch_size": batch_size, "status": "completed",
                "elapsed_seconds": elapsed, "samples": len(rows),
                "generated_tokens": sum(row["generated_tokens"] for row in rows),
                "signature": digest, "exact_reference_match": digest == reference_digest,
                "finite_logits": True, "generation_contract_valid": True,
                **_differences(reference_rows, rows)})
        except (NonFiniteGenerationError, GenerationContractError) as error:
            if batch_size == 1:
                raise
            records.append({"batch_size": batch_size,
                "status": "nonfinite_logits" if isinstance(error, NonFiniteGenerationError) else "invalid_generation_contract",
                "elapsed_seconds": time.perf_counter() - started,
                "exact_reference_match": False, "error": str(error)})
        except RuntimeError as error:
            if batch_size == 1 or not _oom(error):
                raise
            detail = str(error)
            error.__traceback__ = None
            records.append({"batch_size": batch_size, "status": "out_of_memory",
                "elapsed_seconds": time.perf_counter() - started,
                "exact_reference_match": False, "error": detail})
            _empty_cache(device)
    matching = [record for record in records
                if record["status"] == "completed" and record["exact_reference_match"]]
    passing = [record for record in records if record["status"] == "completed"
               and (mode == "validated" or record["exact_reference_match"])]
    fastest = min(passing, key=lambda record: (record["elapsed_seconds"], record["batch_size"]))
    fastest_matching = min(matching, key=lambda record: (record["elapsed_seconds"], record["batch_size"]))
    effective = 1 if mode == "strict" else fastest["batch_size"]
    return {"schema_version": 1, "mode": mode, "task": task, "variant": variant,
        "effective_batch_size": effective, "fastest_matching_batch_size": fastest_matching["batch_size"],
        "fastest_valid_batch_size": min((record for record in records if record["status"] == "completed"),
            key=lambda record: (record["elapsed_seconds"], record["batch_size"]))["batch_size"],
        "reference_batch_size": 1, "calibration_representation": "BASE",
        "calibration_split": source_split, "calibration_ids": [sample.id for sample in selected],
        "calibration_selection": "deterministic_prompt_length_quantiles",
        "selection_uses_task_scores": False, "selection_uses_heldout_examples": False,
        "token_equality_required_for_selection": mode != "validated",
        "normal_finite_bf16_token_differences_permitted": mode == "validated",
        "calibration_signature": reference_digest, "candidates": records,
        "fixed_across_representations_conditions_seeds": True,
        "all_panel_token_equality_guaranteed": mode in {"strict", "audited"},
        "equality_basis": "batch_one" if mode == "strict" else
            "every_panel_reference_audit" if mode == "audited" else "method_data_scoring_preserved_normal_bf16_variation"}


def generate_with_batch_policy(model, tokenizer, samples, representation, config, device,
                               variant="canonical", *, policy, progress_callback=None):
    """Return (prediction rows, engineering execution receipt).

    Only strict and audited promise batch-one-equivalent scientific rows.
    Auditing is deliberately explicit: it costs an additional reference pass
    and must not be advertised as a guaranteed speed optimization.
    """
    mode = policy.get("mode", "strict")
    if mode not in MODES:
        raise ValueError(f"Unknown generation batch policy: {mode}")
    requested = int(policy.get("effective_batch_size", 1))
    if requested < 1:
        raise ValueError("Generation batch size must be positive")
    if policy.get("variant", variant) != variant:
        raise ValueError("Frozen batch policy prompt variant differs")
    task = policy.get("task")
    if task is not None and any(sample.task != task for sample in samples):
        raise ValueError("Frozen batch policy target task differs")
    effective = 1 if mode == "strict" else requested
    receipt = {"schema_version": 1, "mode": mode, "requested_batch_size": requested,
        "effective_batch_size": effective, "audited": False, "fallback": None,
        "all_panel_token_equality_guaranteed": mode in {"strict", "audited"},
        "normal_finite_bf16_token_differences_permitted": mode == "validated"}
    try:
        rows = generate_predictions(model, tokenizer, samples, representation, config,
            device, variant, batch_size_override=effective, progress_callback=progress_callback,
            check_finite_logits=True)
    except RuntimeError as error:
        invalid_logits = isinstance(error, NonFiniteGenerationError)
        if effective == 1 or not (_oom(error) or invalid_logits):
            raise
        detail = str(error)
        error.__traceback__ = None
        _empty_cache(device)
        rows = generate_predictions(model, tokenizer, samples, representation, config,
            device, variant, batch_size_override=1, progress_callback=progress_callback,
            check_finite_logits=True)
        receipt.update(effective_batch_size=1,
                       fallback="nonfinite_logits" if invalid_logits else "out_of_memory", error=detail)
    _validate_rows(model, tokenizer, samples, rows, config)
    if mode == "audited" and receipt["effective_batch_size"] != 1:
        reference = generate_predictions(model, tokenizer, samples, representation, config,
            device, variant, batch_size_override=1, progress_callback=progress_callback,
            check_finite_logits=True)
        _validate_rows(model, tokenizer, samples, reference, config)
        candidate_signature = generation_signature(rows)
        reference_signature = generation_signature(reference)
        matched = candidate_signature == reference_signature
        receipt.update(audited=True, exact_reference_match=matched,
                       candidate_signature=candidate_signature, reference_signature=reference_signature)
        if not matched:
            rows = reference
            receipt.update(effective_batch_size=1, fallback="token_or_stop_mismatch")
    receipt["returned_signature"] = generation_signature(rows)
    receipt.update(finite_logits_checked=True, generation_contract_valid=True)
    return rows, receipt
