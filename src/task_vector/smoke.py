"""Real-platform engineering checks on training samples, never test data."""
from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
import time
import torch

from .evaluation import greedy_generate
from .io import atomic_json, object_sha256, read_jsonl
from .losses import answer_nll, collate_encoded
from .ltv import LTVAdapter, intervention_context
from .representations import Representation


SMOKE_PROTOCOL_VERSION = "same_shape_causal_prefill_v2"


def _same_shape_causal_checks(model, batch, representation, teacher_logits, *,
                              pad_token_id, condition, sample_ids):
    """Separate causal/injection correctness from cross-shape BF16 rounding.

    Both comparisons retain the teacher batch, sequence width, and LM-head
    shape. Prompt-only visibility masks every token after the fixed original
    prompt position. A separate comparison changes future token contents while
    retaining the original attention mask. Neither path can read an answer.
    """
    ids, mask, positions = (batch[key] for key in ("input_ids", "attention_mask", "prompt_positions"))
    columns = torch.arange(ids.shape[1], device=ids.device)[None, :]
    prompt = columns <= positions[:, None]
    if not bool((mask[prompt] == 1).all()):
        raise ValueError("Every original prompt token must be visible in the teacher attention mask")
    future = ~prompt
    rows = torch.arange(ids.shape[0], device=ids.device)
    expected = teacher_logits[rows, positions]
    prompt_ids, prompt_mask = ids.clone(), mask.clone()
    prompt_ids.masked_fill_(future, int(pad_token_id))
    prompt_mask.masked_fill_(future, 0)
    changed_ids = ids.clone()
    changed_ids[future] = (changed_ids[future] + 17) % int(model.config.vocab_size)
    records = {"prompt_visibility": [], "future_tokens": []}
    with torch.no_grad():
        for label, current_ids, current_mask in (
            ("prompt_visibility", prompt_ids, prompt_mask),
            ("future_tokens", changed_ids, mask),
        ):
            with intervention_context(model, representation, positions):
                observed = model(input_ids=current_ids, attention_mask=current_mask,
                                 use_cache=False).logits[rows, positions]
            torch.testing.assert_close(observed, expected, atol=0, rtol=0)
            for index, sample_id in enumerate(sample_ids):
                records[label].append({"condition": condition, "id": sample_id,
                    "batch_shape": list(ids.shape), "exact_logits": True,
                    "max_abs_logit_difference": float((observed[index].float() - expected[index].float()).abs().max())})
    return records


def _teacher_prefill_comparison(prefill, teacher, *, atol, rtol, bf16):
    """Keep finite values and greedy decisions mandatory at every precision.

    BF16 batched/full-sequence and single-prefix kernels need not use identical
    floating-point reductions. Their original allclose result remains recorded;
    same-shape causal checks above independently require exact agreement.
    """
    left, right = prefill.detach().float().cpu(), teacher.detach().float().cpu()
    if not bool(torch.isfinite(left).all() and torch.isfinite(right).all()):
        raise AssertionError("Teacher forcing or prefill produced nonfinite logits")
    within = bool(torch.isclose(left, right, atol=atol, rtol=rtol).all())
    if not bf16:
        torch.testing.assert_close(prefill, teacher, atol=atol, rtol=rtol)
    if int(prefill.argmax()) != int(teacher.argmax()):
        raise AssertionError("Teacher forcing and prefill choose different first answer tokens")
    difference = left - right
    return {"max_abs_logit_difference": float(difference.abs().max()),
            "rmse": float(difference.square().mean().sqrt()),
            "within_original_tolerance": within,
            "allclose_role": "cross_shape_BF16_diagnostic" if bf16 else "hard_gate",
            "finite_logits": True, "same_argmax": True}


def _run_loda_training_smoke(model, tokenizer, samples, config, device, output_dir, base_fingerprint):
    """Exercise the actual trainer/GAO/checkpoint path with one engineering batch.

    Both calibration halves and the two diagnostic loss examples come from the
    same 16 source-task training examples. This is not a scientific task
    decomposition or an independent validation set, and cannot select a model.
    """
    from .subspaces import collect_statistics, build_subspaces, save_statistics
    from .training import train_task, capture_rng_state, _content_digest
    from .representations import norm
    from .prompts import encode_sample

    selected = list(samples[:16])
    if len(selected) != 16 or len({sample.id for sample in selected}) != 16:
        raise ValueError("LoDA engineering smoke needs 16 unique source-task training samples")
    if len({sample.task for sample in selected}) != 1:
        raise ValueError("LoDA engineering smoke must not mix or read other source/heldout tasks")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    cfg = deepcopy(config)
    cfg["supplement_label"] = "engineering_smoke_only_one_LoDA_logical_batch"
    cfg.setdefault("training", {}).update(epochs=1, micro_batch_size=2,
        effective_batch_size=16, gradient_accumulation_steps=8, warmup_steps=0,
        save_every_steps=1, weight_decay_loda=0.)
    cfg["training"].pop("resume_from", None)
    cfg.setdefault("loda", {}).update(statistics_samples=8, train_general_a=False,
                                      isolated_enabled=True)
    halves = {"engineering_train_half_a": selected[:8], "engineering_train_half_b": selected[8:]}
    stats = {}
    for label, half in halves.items():
        stats[label] = collect_statistics(model, tokenizer, half, cfg, device)
        stats[label].metadata.update(engineering_only=True, actual_source_task=selected[0].task,
            calibration_scope="two disjoint 8-sample halves of the same source task training pool")
        save_statistics(stats[label], output_dir / f"{label}_statistics.pt")
    spaces = build_subspaces(stats, "engineering_train_half_a", cfg, seen_tasks=list(halves))
    for spec in spaces.values():
        spec["metadata"].update(engineering_only=True, route="engineering_LoDA_GAO",
            statistics_source="frozen_base_current_task_train_halves", scientific_statistics=False)
    provenance = {"kind": "engineering_LoDA_GAO_smoke", "base_fingerprint": base_fingerprint,
        "source_task": selected[0].task, "source_split": "train", "scientific_training": False,
        "sample_ids": [sample.id for sample in selected],
        "diagnostic_loss_ids": [sample.id for sample in selected[:2]],
        "diagnostic_loss_scope": "training examples; no independent validation or scientific selection"}
    first = train_task(model, tokenizer, selected, selected[:2], task=selected[0].task,
        route="loda", seed=47017, output_dir=output_dir / "uninterrupted", config=cfg,
        device=device, subspaces=spaces, provenance=provenance)
    steps = read_jsonl(output_dir / "uninterrupted" / "steps.jsonl")
    if (first["steps"] != 1 or first["optimizer_steps"] != 2 or len(steps) != 1
            or steps[0].get("batch_optimizer_steps") != 2
            or steps[0].get("batch_scheduler_steps") != 2
            or len(steps[0].get("group1_ids", [])) != 8 or len(steps[0].get("group2_ids", [])) != 8):
        raise AssertionError("LoDA smoke did not execute exactly one full bidirectional GAO logical batch")
    if set(steps[0]["group1_ids"]) & set(steps[0]["group2_ids"]):
        raise AssertionError("LoDA smoke GAO halves overlap")
    if not -float(cfg["loda"].get("rho_max", .05)) <= steps[0]["rho"] <= 0:
        raise AssertionError("LoDA smoke rho differs from the single negative batch draw")
    initial = Representation.load(first["initial_path"])
    first_raw = Representation.load(first["raw_path"])
    if norm(initial) != 0 or norm(first_raw) <= 1e-12:
        raise AssertionError("LoDA smoke failed to update zero-initialized B tensors")
    checkpoint = torch.load(first["last_checkpoint"], map_location="cpu", weights_only=True)
    if checkpoint["optimizer_steps"] != 2 or checkpoint["scheduler"]["last_epoch"] != 2:
        raise AssertionError("LoDA optimizer/scheduler checkpoint counters differ")
    resumed_cfg = deepcopy(cfg)
    resumed_cfg["training"]["resume_from"] = first["last_checkpoint"]
    resumed = train_task(model, tokenizer, selected, selected[:2], task=selected[0].task,
        route="loda", seed=47017, output_dir=output_dir / "resumed", config=resumed_cfg,
        device=device, subspaces=spaces, provenance=provenance)
    restored = torch.load(resumed["last_checkpoint"], map_location="cpu", weights_only=True)
    if (resumed["steps"] != 1 or resumed["optimizer_steps"] != 2
            or _content_digest(checkpoint) != _content_digest(restored)):
        raise AssertionError("LoDA resume repeated a step or changed adapter/AdamW/scheduler/RNG state")
    if _content_digest(capture_rng_state()) != _content_digest(checkpoint["rng"]):
        raise AssertionError("LoDA resume did not restore the actual accelerator/CPU RNG state")
    if first["best_nll"] != resumed["best_nll"]:
        raise AssertionError("LoDA resume changed the engineering diagnostic loss")
    training = cfg["training"]
    encoded = [encode_sample(tokenizer, sample, max_prompt=training["max_prompt_tokens"],
                             max_target=training["max_target_tokens"]) for sample in selected[:2]]
    batch = collate_encoded(encoded, tokenizer.pad_token_id, device)
    was_training = model.training
    model.eval()
    versions = []
    try:
        with torch.no_grad():
            baseline = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"],
                             use_cache=False, logits_to_keep=1).logits
            for version in ("raw", "post"):
                original = Representation.load(first[f"{version}_path"])
                loaded = Representation.load(resumed[f"{version}_path"])
                if _content_digest(original.payload()) != _content_digest(loaded.payload()):
                    raise AssertionError(f"LoDA {version} resume export differs")
                with intervention_context(model, original):
                    before = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"],
                                   use_cache=False, logits_to_keep=1).logits
                with intervention_context(model, loaded):
                    after = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"],
                                  use_cache=False, logits_to_keep=1).logits
                torch.testing.assert_close(before, after, atol=0, rtol=0)
                versions.append({"version": version, "exact_logits": True,
                                 "max_abs_logit_difference": float((before.float() - after.float()).abs().max())})
            exited = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"],
                           use_cache=False, logits_to_keep=1).logits
            torch.testing.assert_close(exited, baseline, atol=0, rtol=0)
    finally:
        model.train(was_training)
    if any(parameter.requires_grad or parameter.grad is not None for parameter in model.parameters()):
        raise AssertionError("LoDA engineering training changed frozen-base gradient ownership")
    for result in (first, resumed):
        for field in ("raw_path", "post_path", "initial_path"):
            exported = Representation.load(result[field])
            exported.metadata.update(engineering_only=True, base_fingerprint=base_fingerprint,
                scientific_statistics=False, calibration_scope="same-source-task train halves")
            exported.save(result[field])
    protocol = {"engineering_only": True, "scientific_training": False, "formal_evaluation": False,
        "source_task": selected[0].task, "sample_split": "train", "sample_ids": [s.id for s in selected],
        "calibration_samples_total": 16, "calibration_samples_per_half": 8,
        "calibration_scope": "two disjoint halves of current source-task train; not scientific S_t",
        "diagnostic_loss_ids": [s.id for s in selected[:2]], "independent_validation": False,
        "micro_batch_size": 2, "effective_batch_size": 16, "epochs": 1,
        "rank_general": cfg["loda"].get("rank_general", 8),
        "rank_isolated": cfg["loda"].get("rank_isolated", 8),
        "logical_steps": 1, "optimizer_steps": 2, "scheduler_steps": 2,
        "rho": steps[0]["rho"], "group1_ids": steps[0]["group1_ids"], "group2_ids": steps[0]["group2_ids"],
        "checkpoint": first["last_checkpoint"], "resumed_checkpoint": resumed["last_checkpoint"],
        "checkpoint_state_exact": True, "resume_additional_optimizer_steps": 0,
        "rng_state_exact": True, "raw_post_resume": versions,
        "base_frozen_no_gradient": True, "context_exit_restores_base_exactly": True,
        "device": str(device), "memory": first["memory"],
        "training_runtime_seconds": first["runtime_seconds"], "resume_runtime_seconds": resumed["runtime_seconds"]}
    atomic_json(output_dir / "loda-engineering.json", protocol)
    return protocol


def run_smoke(config, *, task="math", output_dir=None):
    from .runner import ExperimentRunner
    from .prompts import encode_sample
    from .runtime import memory_evidence, synchronize
    runner = ExperimentRunner(config)
    model, tokenizer = runner.load_model()
    device = runner.device
    if output_dir is None:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
        output_dir = runner.root / "smoke" / stamp
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if (output_dir / "smoke.json").exists():
        raise FileExistsError("A completed smoke record already exists; choose a new output directory")
    available = runner.samples(task, "train")
    selected = available[:2]
    if len(selected) != 2:
        raise ValueError("Smoke needs two prepared training samples")
    cfg = config["training"]
    encoded = [encode_sample(tokenizer, sample, max_prompt=cfg["max_prompt_tokens"],
                             max_target=cfg["max_target_tokens"]) for sample in selected]
    batch = collate_encoded(encoded, tokenizer.pad_token_id, device)
    # BF16 tolerances are fixed before seeing results. Exact reload/restoration
    # use zero tolerance; cached-vs-full comparisons allow BF16 kernel rounding.
    bf16 = next(model.parameters()).dtype == torch.bfloat16
    atol, rtol = (0.03125, 0.015625) if bf16 else (2e-6, 2e-5)
    api = getattr(torch, str(device).split(":")[0], None)
    if hasattr(api, "reset_peak_memory_stats"):
        api.reset_peak_memory_stats(device)
    record = {"schema_version": 1, "protocol_version": SMOKE_PROTOCOL_VERSION,
              "kind": "engineering_smoke_only", "task": task,
              "sample_split": "train", "sample_ids": [s.id for s in selected],
              "device": str(device), "base_dtype": str(next(model.parameters()).dtype),
              "adapter_master_dtype": "torch.float32", "atol": atol, "rtol": rtol,
              "scientific_training": False, "formal_evaluation": False,
              "checks": {}, "started_utc": datetime.now(timezone.utc).isoformat()}
    from .config import scientific_config, source_manifest
    record["config_sha256"] = object_sha256(scientific_config(config))
    record["data_manifest_sha256"] = object_sha256(runner.manifest)
    record["base_fingerprint"] = runner.base_fingerprint()
    record["source_manifest_sha256"] = object_sha256(source_manifest(runner.source_root))
    if hasattr(api, "get_device_name"):
        record["device_name"] = api.get_device_name(device)
    started = time.perf_counter()
    print("SMOKE_STAGE=LTV_update_reload_and_causal_generation", flush=True)
    model.eval()
    adapter = LTVAdapter(model, config["ltv"]["layers"], seed=17)
    try:
        with torch.no_grad():
            baseline = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"], use_cache=False).logits.detach()
        # One explicit smoke optimizer update exercises a real answer loss. It
        # is saved only under smoke/, never in an experiment training directory.
        optimizer = torch.optim.AdamW(adapter.parameters(), lr=0.001, weight_decay=0.01, foreach=False)
        if cfg.get("gradient_checkpointing", True):
            model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
            record["gradient_checkpointing"] = "nonreentrant"
        else:
            record["gradient_checkpointing"] = "disabled"
        model.train()
        with adapter:
            adapter.set_positions(batch["prompt_positions"])
            adapter.set_prefill(True)
            train_logits = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"], use_cache=False).logits
            loss = answer_nll(train_logits, batch["labels"]).mean()
            loss.backward()
            norm = float(torch.nn.utils.clip_grad_norm_(adapter.parameters(), 1.0))
            if not torch.isfinite(torch.tensor(norm)) or norm <= 0:
                raise AssertionError("LTV gradient is zero or nonfinite")
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            record["smoke_answer_nll"] = float(loss.detach())
            record["smoke_grad_norm"] = norm
        model.eval()
        del train_logits, loss
        if any(p.grad is not None or p.requires_grad for p in model.parameters()):
            raise AssertionError("Frozen base acquired trainable parameters or gradients")
        record["checks"]["base_frozen_no_gradient"] = True
        representation = adapter.representation(task, 17)
        representation.save(output_dir / "smoke_ltv.pt")
        loaded = Representation.load(output_dir / "smoke_ltv.pt")
        with torch.no_grad():
            with intervention_context(model, representation, batch["prompt_positions"]):
                before = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"], use_cache=False).logits
            with intervention_context(model, loaded, batch["prompt_positions"]):
                after = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"], use_cache=False).logits
            torch.testing.assert_close(before, after, atol=0, rtol=0)
            record["checks"]["save_reload_exact_logits"] = True
            record["checks"]["teacher_prompt_visibility_same_shape"] = []
            record["checks"]["causal_future_tokens_same_shape"] = []
            for condition, current_representation, teacher_logits in (
                ("BASE", None, baseline), ("SAVED_LTV", loaded, after),
            ):
                causal = _same_shape_causal_checks(model, batch, current_representation,
                    teacher_logits, pad_token_id=tokenizer.pad_token_id,
                    condition=condition, sample_ids=[example["id"] for example in encoded])
                record["checks"]["teacher_prompt_visibility_same_shape"].extend(causal["prompt_visibility"])
                record["checks"]["causal_future_tokens_same_shape"].extend(causal["future_tokens"])
            record["checks"]["teacher_prompt_visibility_same_shape_exact"] = True
            record["checks"]["causal_future_tokens_same_shape_exact"] = True
            record["checks"]["teacher_forcing_prefill"] = []
            record["checks"]["cached_prefill_no_cache_exact_logits"] = []
            record["checks"]["cached_full_prefix_generation"] = []
            for index, example in enumerate(encoded):
                prompt_tokens = example["input_ids"][:example["prompt_length"]]
                prefix = torch.tensor([prompt_tokens], dtype=torch.long, device=device)
                pos = torch.tensor([len(prompt_tokens) - 1], dtype=torch.long, device=device)
                with intervention_context(model, loaded, pos):
                    prefill = model(input_ids=prefix, use_cache=False, logits_to_keep=1).logits[0, -1]
                teacher = after[index, example["prompt_length"] - 1]
                agreement = _teacher_prefill_comparison(prefill, teacher, atol=atol, rtol=rtol, bf16=bf16)
                record["checks"]["teacher_forcing_prefill"].append({"id": example["id"], **agreement})
                with intervention_context(model, loaded, pos):
                    cached_prefill = model(input_ids=prefix, use_cache=True, logits_to_keep=1)
                if cached_prefill.past_key_values is None:
                    raise AssertionError("Cached prefill did not return the required KV cache")
                torch.testing.assert_close(cached_prefill.logits[0, -1], prefill, atol=0, rtol=0)
                cache_length = int(cached_prefill.past_key_values.get_seq_length())
                if cache_length != len(prompt_tokens):
                    raise AssertionError("Cached prefill stored the wrong prompt length")
                record["checks"]["cached_prefill_no_cache_exact_logits"].append({"id": example["id"],
                    "exact_logits": True, "cache_length": cache_length,
                    "max_abs_logit_difference": float((cached_prefill.logits[0, -1].float() - prefill.float()).abs().max())})
                del cached_prefill
                eos = tokenizer.eos_token_id
                eos_ids = [eos] if isinstance(eos, int) else list(eos)
                with intervention_context(model, loaded) as ctx:
                    cached, _ = greedy_generate(model, prefix, torch.ones_like(prefix), max_new_tokens=3,
                        intervention=ctx, eos_token_ids=eos_ids, pad_token_id=tokenizer.pad_token_id)
                full_tokens = []
                for _ in range(3):
                    with intervention_context(model, loaded, pos):
                        logits = model(input_ids=prefix, use_cache=False, logits_to_keep=1).logits
                    token = int(logits[0, -1].argmax())
                    full_tokens.append(token)
                    if token in eos_ids:
                        break
                    prefix = torch.cat([prefix, torch.tensor([[token]], device=device)], dim=1)
                if full_tokens != cached[0]:
                    raise AssertionError("Cached and full-prefix decoding disagree at fixed prompt injection")
                record["checks"]["cached_full_prefix_generation"].append({"id": example["id"], "token_ids": full_tokens, "equal": True})
            del before, after
            restored = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"], use_cache=False).logits
            torch.testing.assert_close(restored, baseline, atol=0, rtol=0)
            record["checks"]["context_exit_restores_base_exactly"] = True
        # Verify native heterogeneous Q/K/V widths and a zero LoDA update on the
        # real model. These synthetic As are engineering fixtures, not S_t.
        print("SMOKE_STAGE=LoDA_native_projection_and_raw_post_exports", flush=True)
        from .loda import LoDAAdapter
        hidden = model.config.hidden_size
        rank = config["loda"]["rank_general"]
        isolated_rank = config["loda"]["rank_isolated"]
        eye = torch.eye(hidden)
        fixtures = {f"model.layers.{i}": {"A_G": eye[:rank] / 3**0.5,
                   "A_I": eye[rank:rank + isolated_rank] / 3**0.5,
                   "lambda": torch.linspace(0.3, 1.0, rank)} for i in range(model.config.num_hidden_layers)}
        loda = LoDAAdapter(model, fixtures)
        with torch.no_grad(), loda:
            zero = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"], use_cache=False).logits
        torch.testing.assert_close(zero, baseline, atol=0, rtol=0)
        record["checks"]["zero_loda_equals_base_exactly"] = True
        del zero, restored, baseline
        record["checks"]["nonzero_loda_saved_raw_post_equivalence"] = []
        fixture_rng = torch.Generator(device="cpu").manual_seed(47017)
        with torch.no_grad():
            for layer in loda.layers.values():
                for parameter in list(layer.B_G.values()) + list(layer.B_I.values()):
                    parameter.copy_(torch.randn(parameter.shape, generator=fixture_rng).to(parameter.device) * 0.005)
            for version in ("raw", "post"):
                loda.set_recalibrated(version == "post")
                with loda:
                    live = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"],
                                 use_cache=False, logits_to_keep=1).logits
                path = output_dir / f"smoke_loda_{version}.pt"
                loda.representation("synthetic_engineering_fixture", 47017, version).save(path)
                saved = Representation.load(path)
                with intervention_context(model, saved):
                    reloaded = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"],
                                     use_cache=False, logits_to_keep=1).logits
                torch.testing.assert_close(live, reloaded, atol=atol, rtol=rtol)
                same_argmax = bool(torch.equal(live.argmax(-1), reloaded.argmax(-1)))
                if not same_argmax:
                    raise AssertionError("Live LoDA and saved actual-BA intervention choose different tokens")
                record["checks"]["nonzero_loda_saved_raw_post_equivalence"].append({"version": version,
                    "max_abs_logit_difference": float((live.float() - reloaded.float()).abs().max()),
                    "same_argmax": same_argmax})
        record["loda_fixture_scope"] = "synthetic native-projection zero-update and nonzero raw/post export checks; no scientific subspaces or GAO training"
        # The synthetic export fixture above and actual GAO check below have
        # separate records. Neither contributes an experiment task vector.
        del loda, fixtures, live, reloaded
        print("SMOKE_STAGE=LoDA_statistics_GAO_update_and_checkpoint_resume", flush=True)
        record["loda_engineering"] = _run_loda_training_smoke(model, tokenizer, available, config,
            device, output_dir / "loda_training_smoke", record["base_fingerprint"])
        record["checks"].update(loda_true_gao_microbatch_step=True,
            loda_checkpoint_resume_exact=True, loda_resume_does_not_repeat_step=True)
        synchronize(device)
        record.update(status="passed", passed=True, memory=memory_evidence(device), runtime_seconds=time.perf_counter() - started)
    except BaseException as exc:
        record.update(status="failed", passed=False, error_type=type(exc).__name__, error=str(exc),
                      memory=memory_evidence(device), runtime_seconds=time.perf_counter() - started)
        atomic_json(output_dir / "smoke.json", record)
        raise
    finally:
        runner.release_model()
    atomic_json(output_dir / "smoke.json", record)
    return record
