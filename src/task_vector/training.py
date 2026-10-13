"""Shared sample-weighted training, GAO microbatching and resumable state."""
from __future__ import annotations

from contextlib import nullcontext
from collections.abc import Mapping
import copy
import hashlib
import json
import math
from pathlib import Path
import random
import time
import torch

from .evaluation import evaluate_nll, _set_positions
from .io import atomic_json, atomic_torch_save, file_sha256, object_sha256, write_jsonl
from .losses import answer_nll, collate_encoded
from .ltv import LTVAdapter


def capture_rng_state():
    state = {"python": random.getstate(), "torch": torch.get_rng_state()}
    try:
        import numpy as np
        name, keys, pos, has_gauss, cached = np.random.get_state()
        state["numpy"] = {"name": name, "keys": torch.from_numpy(keys.astype("int64")),
                          "pos": pos, "has_gauss": has_gauss, "cached": cached}
    except ImportError:
        pass
    if torch.cuda.is_available():
        state["cuda"] = {"device_index": torch.cuda.current_device(), "state": torch.cuda.get_rng_state()}
    npu = getattr(torch, "npu", None)
    if npu is not None and npu.is_available():
        state["npu"] = {"device_index": npu.current_device(), "state": npu.get_rng_state()}
    return state


def restore_rng_state(state):
    random.setstate(state["python"])
    torch.set_rng_state(state["torch"].cpu())
    if "numpy" in state:
        import numpy as np
        n = state["numpy"]
        np.random.set_state((n["name"], n["keys"].cpu().numpy().astype("uint32"),
                             n["pos"], n["has_gauss"], n["cached"]))
    if "cuda" in state and torch.cuda.is_available():
        cuda_state = state["cuda"]
        torch.cuda.set_rng_state(cuda_state["state"] if isinstance(cuda_state, dict) else cuda_state[torch.cuda.current_device()])
    npu = getattr(torch, "npu", None)
    if "npu" in state and npu is not None and npu.is_available():
        npu_state = state["npu"]
        npu.set_rng_state(npu_state["state"] if isinstance(npu_state, dict) else npu_state[npu.current_device()])


def _logical_batches(encoded, size, *, seed, epoch):
    order = torch.randperm(len(encoded), generator=torch.Generator().manual_seed(seed + 100003 * epoch)).tolist()
    chunks = [order[i:i + size] for i in range(0, len(order), size)]
    # Do not drop or duplicate a final singleton: move one sample from the
    # previous batch so GAO always has two nonempty, disjoint groups.
    if len(chunks) > 1 and len(chunks[-1]) == 1:
        chunks[-1].insert(0, chunks[-2].pop())
    return [[encoded[i] for i in chunk] for chunk in chunks]


def accumulated_backward(model, adapter, group, *, micro_batch_size, pad_token_id, device):
    """Backpropagate a complete group mean while freeing every microbatch graph."""
    if not group:
        raise ValueError("An optimization group cannot be empty")
    total = 0.0
    for start in range(0, len(group), micro_batch_size):
        micro = group[start:start + micro_batch_size]
        batch = collate_encoded(micro, pad_token_id, device)
        _set_positions(adapter, batch["prompt_positions"])
        output = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"], use_cache=False)
        losses = answer_nll(output.logits, batch["labels"])
        weighted_loss = losses.sum() / len(group)
        total += float(weighted_loss.detach())
        weighted_loss.backward()
        del weighted_loss, losses, output, batch
    return total


def _seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    try:
        import numpy as np
        np.random.seed(seed)
    except ImportError:
        pass
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    if hasattr(torch, "npu") and torch.npu.is_available():
        torch.npu.manual_seed(seed)


def _cpu_state(module):
    return {key: value.detach().cpu().clone() for key, value in module.state_dict().items()}


def _content_digest(value):
    """Hash tensor contents in bounded chunks, including native dtype and shape."""
    def normalize(item):
        if isinstance(item, torch.Tensor):
            digest = hashlib.sha256()
            flat = item.detach().reshape(-1)
            for start in range(0, flat.numel(), 2**20):
                chunk = flat[start:start + 2**20].contiguous().cpu().view(torch.uint8)
                digest.update(chunk.numpy().tobytes())
            return {"tensor_sha256": digest.hexdigest(), "shape": list(item.shape), "dtype": str(item.dtype)}
        if isinstance(item, Mapping):
            return {str(key): normalize(item[key]) for key in sorted(item, key=str)}
        if isinstance(item, (list, tuple)):
            return [normalize(part) for part in item]
        return item
    return object_sha256(normalize(value))


def _sample_content(samples):
    return [{name: getattr(sample, name, None) for name in
             ("id", "task", "prompt", "target", "group_id", "metadata")} for sample in samples]


def _checkpoint_provenance(model, train_samples, tune_samples, encoded, encoded_tune,
                           subspaces, provenance, initial_representation):
    # Callers with an already fingerprinted base avoid a second full weight read.
    # Standalone/tiny calls still bind the actual frozen model rather than its path.
    if provenance is not None and not isinstance(provenance, Mapping):
        raise TypeError("Training provenance must be a JSON object")
    bound = json.loads(json.dumps(dict(provenance or {}), allow_nan=False))
    if not isinstance(bound, dict):
        raise TypeError("Training provenance must be a JSON object")
    if not bound.get("base_fingerprint"):
        bound["base_fingerprint"] = _content_digest(model.state_dict())
    native_config = model.config.to_dict() if hasattr(model.config, "to_dict") else vars(model.config)
    bound["native_model_config_sha256"] = _content_digest(native_config)
    bound["sample_content_sha256"] = _content_digest({"train": _sample_content(train_samples),
                                                      "tune": _sample_content(tune_samples)})
    bound["encoded_samples_sha256"] = _content_digest({"train": encoded, "tune": encoded_tune})
    bound["subspaces_content_sha256"] = _content_digest(subspaces)
    bound["initial_override_sha256"] = None if initial_representation is None else _content_digest(initial_representation.payload())
    package = Path(__file__).resolve().parent
    bound["training_sources_sha256"] = object_sha256({path.name: file_sha256(path)
                                                       for path in sorted(package.glob("*.py"))})
    return bound


def _validate_resume_state(state, signature, provenance):
    if state.get("schema_version") != 2 or "initial_representation" not in state or "provenance" not in state:
        raise ValueError("Resume checkpoint lacks bound provenance or initial representation; use a new run")
    if state.get("signature") != signature:
        raise ValueError("Resume checkpoint protocol, sample content, base, source or subspaces differ")
    if (state["provenance"] != provenance or
            state.get("provenance_sha256") != object_sha256(provenance)):
        raise ValueError("Resume checkpoint provenance differs")
    if state.get("initial_representation_sha256") != _content_digest(state["initial_representation"]):
        raise ValueError("Resume checkpoint initial representation is corrupt")


def _initial_coordinates(payload):
    return {key: payload.get(key) for key in ("schema_version", "space", "task", "seed", "values", "factors")}


def train_task(model, tokenizer, train_samples, tune_samples, *, task, route, seed,
               output_dir, config, device, subspaces=None, initial_representation=None, provenance=None):
    """Train independently from the supplied base; never merge updates into it.

    For sequential controls the caller supplies the current merged base and
    enables training of A_G only at the first stage. The same checkpoint and
    answer-loss protocol applies, while its representations remain separately
    labeled by the sequential driver.

    This function exports an uncommitted training-state.json. The owning runner
    publishes training.json only after all representation provenance is complete.
    Checkpoints bind supplied run provenance as well as actual sample encodings,
    subspaces and training sources; legacy unbound checkpoints are rejected.
    """
    from .prompts import encode_sample
    cfg = config["training"]
    epochs = int(cfg.get("epochs", 5))
    micro = int(cfg.get("micro_batch_size", 2))
    effective = int(cfg.get("effective_batch_size", 16))
    accumulation = int(cfg.get("gradient_accumulation_steps", 8))
    if micro * accumulation != effective or micro < 1 or epochs < 1 or epochs > 5:
        raise ValueError("Invalid documented batch or epoch configuration")
    if cfg.get("warmup_steps", 0) != 0:
        raise ValueError("The experimental protocol specifies zero warmup")
    if not train_samples or not tune_samples:
        raise ValueError("Both training and independent tuning samples are required")
    if route not in {"ltv", "loda", "loda_mt", "sequential"}:
        raise ValueError(f"Unsupported route {route}")
    is_ltv = route == "ltv"
    if not is_ltv and len(train_samples) < 2:
        raise ValueError("GAO needs at least two disjoint training samples")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    _seed(int(seed))
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    if is_ltv:
        layers = config.get("ltv", {}).get("layers", [14])
        # Resetting the same seed for each task would create identical initial
        # vectors and a spurious cross-task initialization direction. Derive
        # independent, reproducible RNG streams for each task/layer setting.
        init_seed = int(object_sha256({"seed": int(seed), "task": task, "layers": layers})[:16], 16) % (2**63)
        adapter = LTVAdapter(model, layers, init_seed, initial_representation,
                             init_low=config.get("ltv", {}).get("init_low", -0.1),
                             init_high=config.get("ltv", {}).get("init_high", 0.1))
    else:
        from .loda import LoDAAdapter
        if subspaces is None:
            raise ValueError("LoDA training requires explicit frozen base statistics/subspaces")
        lc = config.get("loda", {})
        adapter = LoDAAdapter(model, subspaces, train_general_a=lc.get("train_general_a", False),
                              isolated_enabled=lc.get("isolated_enabled", True))
        adapter.set_recalibrated(False)
        if initial_representation is not None:
            raise ValueError("LoDA continuation must use a complete training checkpoint, preserving A and B")
    parameters = list(adapter.parameters())
    if not parameters or any(p.dtype != torch.float32 for p in parameters):
        raise ValueError("All optimized adapter master parameters must be FP32")
    weight_decay = cfg.get("weight_decay_ltv", 0.01) if is_ltv else cfg.get("weight_decay_loda", 0.0)
    optimizer = torch.optim.AdamW(parameters, lr=float(cfg.get("learning_rate", 0.001)),
        weight_decay=float(weight_decay), betas=tuple(cfg.get("betas", [0.9, 0.999])),
        eps=float(cfg.get("adam_epsilon", 1e-8)), foreach=False)
    updates_per_batch = 1 if is_ltv else 2
    total_updates = epochs * math.ceil(len(train_samples) / effective) * updates_per_batch
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: max(0.0, 1.0 - step / total_updates))
    encoded = [encode_sample(tokenizer, sample, max_prompt=int(cfg.get("max_prompt_tokens", 512)),
                             max_target=int(cfg.get("max_target_tokens", 512))) for sample in train_samples]
    encoded_tune = [encode_sample(tokenizer, sample, max_prompt=int(cfg.get("max_prompt_tokens", 512)),
                                  max_target=int(cfg.get("max_target_tokens", 512))) for sample in tune_samples]
    bound_provenance = _checkpoint_provenance(model, train_samples, tune_samples, encoded, encoded_tune,
                                              subspaces, provenance, initial_representation)
    signature = object_sha256({"task": task, "route": route, "seed": seed,
        "training": {k: v for k, v in cfg.items() if k not in {"resume_from", "save_every_steps"}},
        "model": config.get("model"), "ltv": config.get("ltv"), "loda": config.get("loda"),
        "train_ids": [s.id for s in train_samples], "tune_ids": [s.id for s in tune_samples],
        "provenance": bound_provenance})
    start_epoch, start_batch, logical_steps, optimizer_steps = 0, 0, 0, 0
    best_nll, bad_epochs = float("inf"), 0
    records, epoch_records = [], []
    best_state = None
    best_checkpoint_path = None
    resume = cfg.get("resume_from")
    if resume:
        state = torch.load(resume, map_location="cpu", weights_only=True)
        _validate_resume_state(state, signature, bound_provenance)
        initial_payload = state["initial_representation"]
        expected_initial = adapter.representation(task, seed, "raw").payload()
        if any(initial_payload.get(key) != expected_initial[key] for key in ("schema_version", "space", "task", "seed")):
            raise ValueError("Resume initial representation identity differs")
        initial_path = output_dir / "initial.pt"
        if initial_path.exists():
            previous_initial = torch.load(initial_path, map_location="cpu", weights_only=True)
            if _content_digest(_initial_coordinates(previous_initial)) != _content_digest(_initial_coordinates(initial_payload)):
                raise ValueError("Resume output initial representation differs")
        source_dir = Path(resume).resolve().parent
        copied_best = None
        if state["best_adapter"] is not None:
            source_best = source_dir / "best.pt"
            if source_best.is_file():
                copied_best = torch.load(source_best, map_location="cpu", weights_only=True)
                _validate_resume_state(copied_best, signature, bound_provenance)
                if (copied_best["best_nll"] != state["best_nll"] or
                        _content_digest(copied_best["adapter"]) != _content_digest(state["best_adapter"])):
                    # An explicitly selected older checkpoint can have a newer
                    # sibling best.pt. Its optimizer does not belong to our best.
                    copied_best = None
        atomic_torch_save(initial_path, initial_payload)
        if source_dir != output_dir.resolve():
            atomic_torch_save(output_dir / "last.pt", state)
            if copied_best is not None:
                atomic_torch_save(output_dir / "best.pt", copied_best)
        if copied_best is not None:
            best_checkpoint_path = output_dir / "best.pt"
        adapter.load_state_dict(state["adapter"])
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        restore_rng_state(state["rng"])
        start_epoch, start_batch = state["next_epoch"], state["next_batch"]
        logical_steps, optimizer_steps = state["logical_steps"], state["optimizer_steps"]
        best_nll, bad_epochs = state["best_nll"], state["bad_epochs"]
        records, epoch_records = state["records"], state["epoch_records"]
        best_state = state["best_adapter"]
    else:
        initial_payload = adapter.representation(task, seed, "raw").payload()
        atomic_torch_save(output_dir / "initial.pt", initial_payload)
    initial_payload_sha256 = _content_digest(initial_payload)
    was_training = model.training
    original_cache = model.config.use_cache
    original_checkpointing = bool(getattr(model, "is_gradient_checkpointing", False))
    gradient_checkpointing = bool(cfg.get("gradient_checkpointing", True))
    if gradient_checkpointing:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.config.use_cache = False
    accelerator = getattr(torch, str(device).split(":")[0], None)
    if hasattr(accelerator, "reset_peak_memory_stats"):
        accelerator.reset_peak_memory_stats(device)
    started = time.perf_counter()

    def save_state(path, next_epoch, next_batch):
        atomic_torch_save(path, {"schema_version": 2, "signature": signature, "task": task,
            "route": route, "seed": seed, "adapter": _cpu_state(adapter),
            "provenance": bound_provenance, "provenance_sha256": object_sha256(bound_provenance),
            "initial_representation": initial_payload,
            "initial_representation_sha256": initial_payload_sha256,
            "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
            "rng": capture_rng_state(), "next_epoch": next_epoch, "next_batch": next_batch,
            "logical_steps": logical_steps, "optimizer_steps": optimizer_steps,
            "best_nll": best_nll, "bad_epochs": bad_epochs, "best_adapter": best_state,
            "records": records, "epoch_records": epoch_records})

    try:
        with adapter:
            for epoch in range(start_epoch, epochs):
                if bad_epochs >= int(cfg.get("patience", 2)):
                    break
                model.train()
                batches = _logical_batches(encoded, effective, seed=int(seed), epoch=epoch)
                for batch_index, group in enumerate(batches):
                    if epoch == start_epoch and batch_index < start_batch:
                        continue
                    if is_ltv:
                        optimizer.zero_grad(set_to_none=True)
                        loss = accumulated_backward(model, adapter, group, micro_batch_size=micro,
                            pad_token_id=tokenizer.pad_token_id, device=device)
                        grad_norm = float(torch.nn.utils.clip_grad_norm_(parameters, float(cfg.get("max_grad_norm", 1.0))))
                        if not math.isfinite(loss) or not math.isfinite(grad_norm):
                            raise FloatingPointError("Nonfinite LTV loss or gradient")
                        optimizer.step()
                        scheduler.step()
                        detail = {"loss": loss, "grad_norm_before_clip": grad_norm}
                    else:
                        from .gao import gao_step
                        split = len(group) // 2
                        first, second = group[:split], group[split:]
                        if set(x["id"] for x in first) & set(x["id"] for x in second):
                            raise ValueError("GAO groups must be sample-disjoint")
                        def loss_fn(items):
                            return accumulated_backward(model, adapter, items, micro_batch_size=micro,
                                pad_token_id=tokenizer.pad_token_id, device=device)
                        detail = gao_step(parameters, optimizer, first, second, loss_fn,
                            rho_max=float(config.get("loda", {}).get("rho_max", 0.05)),
                            clip_norm=float(cfg.get("max_grad_norm", 1.0)), scheduler=scheduler)
                    logical_steps += 1
                    optimizer_steps += updates_per_batch
                    detail = dict(detail)
                    detail["batch_optimizer_steps"] = detail.pop("optimizer_steps", updates_per_batch)
                    if "scheduler_steps" in detail:
                        detail["batch_scheduler_steps"] = detail.pop("scheduler_steps")
                    records.append({"epoch": epoch + 1, "batch": batch_index, "logical_step": logical_steps,
                        "optimizer_steps": optimizer_steps, "batch_size": len(group),
                        "sample_ids": [x["id"] for x in group],
                        "learning_rate_next": optimizer.param_groups[0]["lr"], **detail})
                    if logical_steps % int(cfg.get("save_every_steps", 1)) == 0:
                        save_state(output_dir / "last.pt", epoch, batch_index + 1)
                tune_nll = evaluate_nll(model, tokenizer, tune_samples, adapter, config, device)
                if not math.isfinite(tune_nll):
                    raise FloatingPointError("Nonfinite tuning NLL")
                improved = tune_nll < best_nll
                if improved:
                    best_nll, bad_epochs = tune_nll, 0
                    best_state = _cpu_state(adapter)
                else:
                    bad_epochs += 1
                epoch_records.append({"epoch": epoch + 1, "tune_answer_nll": tune_nll,
                                      "best": improved, "bad_epochs": bad_epochs})
                if improved:
                    save_state(output_dir / "best.pt", epoch + 1, 0)
                    best_checkpoint_path = output_dir / "best.pt"
                save_state(output_dir / "last.pt", epoch + 1, 0)
                atomic_json(output_dir / "epochs.json", epoch_records)
                write_jsonl(output_dir / "steps.jsonl", records)
                start_batch = 0
        if best_state is None:
            raise RuntimeError("No validated checkpoint is available")
        adapter.load_state_dict(best_state)
        raw = adapter.representation(task, seed, "raw")
        post = adapter.representation(task, seed, "post")
        raw.save(output_dir / "raw.pt")
        post.save(output_dir / "post.pt")
        # last.pt carries the best adapter even when a separately copied best.pt
        # is unavailable. Never label its last-step optimizer as a best-step one.
        atomic_torch_save(output_dir / "best-adapter.pt", {"schema_version": 1,
            "kind": "adapter_only", "resume_supported": False, "adapter": best_state,
            "signature": signature, "provenance": bound_provenance, "best_nll": best_nll})
        from .runtime import memory_evidence, synchronize
        synchronize(device)
        result = {"raw_path": str(output_dir / "raw.pt"), "post_path": str(output_dir / "post.pt"),
                  "initial_path": str(output_dir / "initial.pt"),
                  "best_checkpoint": str(best_checkpoint_path) if best_checkpoint_path is not None else None,
                  "best_adapter_path": str(output_dir / "best-adapter.pt"),
                  "best_checkpoint_resume_supported": best_checkpoint_path is not None,
                  "last_checkpoint": str(output_dir / "last.pt"), "best_nll": best_nll,
                  "steps": logical_steps, "optimizer_steps": optimizer_steps,
                  "completed_epochs": len(epoch_records), "stop_reason": "patience" if bad_epochs >= int(cfg.get("patience", 2)) else "epochs",
                  "runtime_seconds": time.perf_counter() - started,
                  "selection": "raw_adapter_answer_nll_each_epoch", "signature": signature,
                  "provenance": bound_provenance, "provenance_sha256": object_sha256(bound_provenance),
                  "final_partial_batch": "retain_all_samples_split_disjoint_near_equal",
                  "precision": "BF16_base_FP32_adapter_master", "gradient_checkpointing": "nonreentrant" if gradient_checkpointing else "disabled"}
        result["memory"] = memory_evidence(device)
        result["device"] = str(device)
        atomic_json(output_dir / "training-state.json", {**result, "status": "exported_uncommitted"})
        return result
    except BaseException as exc:
        from .runtime import memory_evidence
        atomic_json(output_dir / "failure.json", {"type": type(exc).__name__, "message": str(exc),
            "logical_steps": logical_steps, "optimizer_steps": optimizer_steps,
            "last_checkpoint": str(output_dir / "last.pt"), "runtime_seconds": time.perf_counter() - started,
            "device": str(device), "memory": memory_evidence(device)})
        raise
    finally:
        optimizer.zero_grad(set_to_none=True)
        model.config.use_cache = original_cache
        if gradient_checkpointing and not original_checkpointing:
            model.gradient_checkpointing_disable()
        model.train(was_training)
