"""Deterministic full-answer generation and answer-only validation."""
from __future__ import annotations

from contextlib import nullcontext
import torch

from .losses import answer_nll, collate_encoded
from .ltv import intervention_context


DEFAULT_LIMITS = {"math": 512, "logic": 32, "translation": 256,
                  "paraphrase": 128, "code_generation": 512, "code_repair": 512,
                  "summary": 256, "samsum": 256, "joint": 1536}


def _active(adapter):
    return bool(getattr(adapter, "is_active", False) or getattr(adapter, "context", None) is not None)


def _set_positions(adapter, positions):
    if adapter is not None and hasattr(adapter, "set_positions"):
        adapter.set_positions(positions)
    if adapter is not None and hasattr(adapter, "set_prefill"):
        adapter.set_prefill(True)


def evaluate_nll(model, tokenizer, samples, adapter, config, device):
    from .prompts import encode_sample
    if not samples:
        raise ValueError("NLL validation set is empty")
    cfg = config.get("training", {})
    batch_size = int(cfg.get("micro_batch_size", 2))
    was_training = model.training
    context = nullcontext(adapter) if adapter is None or _active(adapter) else adapter
    total, count = 0.0, 0
    try:
        model.eval()
        with context, torch.no_grad():
            for start in range(0, len(samples), batch_size):
                encoded = [encode_sample(tokenizer, s,
                    max_prompt=cfg.get("max_prompt_tokens", 512),
                    max_target=cfg.get("max_target_tokens", 512)) for s in samples[start:start + batch_size]]
                batch = collate_encoded(encoded, tokenizer.pad_token_id, device)
                _set_positions(adapter, batch["prompt_positions"])
                out = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"], use_cache=False)
                losses = answer_nll(out.logits, batch["labels"])
                total += float(losses.sum())
                count += len(encoded)
    finally:
        model.train(was_training)
    return total / count


def _eos_ids(model, tokenizer):
    eos = getattr(model.generation_config, "eos_token_id", None)
    eos = eos if eos is not None else tokenizer.eos_token_id
    if eos is None:
        raise ValueError("Greedy generation requires a known EOS token")
    return [int(eos)] if isinstance(eos, int) else list(map(int, eos))


def greedy_generate(model, input_ids, attention_mask, *, max_new_tokens, intervention=None,
                    eos_token_ids, pad_token_id):
    """Greedy decoding with one prefill intervention and native model KV cache.

    No sampling, repetition penalty or implicit generation-config processor is
    used. Token count includes EOS. A length stop is never treated as EOS.
    """
    if max_new_tokens <= 0:
        raise ValueError("Generation limit must be positive")
    batch_size, prompt_width = input_ids.shape
    if not bool(attention_mask[:, -1].all()):
        raise ValueError("Generation batches must be left padded")
    positions = torch.full((batch_size,), prompt_width - 1, device=input_ids.device, dtype=torch.long)
    if intervention is not None and hasattr(intervention, "set_positions"):
        intervention.set_positions(positions)
    active = torch.ones(batch_size, dtype=torch.bool, device=input_ids.device)
    result = [[] for _ in range(batch_size)]
    ended = [False] * batch_size
    cache = None
    current = input_ids
    mask = attention_mask
    for step in range(max_new_tokens):
        if intervention is not None and hasattr(intervention, "set_prefill"):
            intervention.set_prefill(step == 0)
        position_ids = mask.long().cumsum(-1) - 1
        position_ids.masked_fill_(mask == 0, 0)
        position_ids = position_ids[:, -current.shape[1]:]
        cache_position = torch.arange(0 if step == 0 else prompt_width + step - 1,
                                      prompt_width if step == 0 else prompt_width + step,
                                      device=input_ids.device)
        output = model(input_ids=current, attention_mask=mask, position_ids=position_ids,
                       past_key_values=cache, cache_position=cache_position, use_cache=True,
                       logits_to_keep=1)
        if output.past_key_values is None:
            raise RuntimeError("Model failed to return the required KV cache")
        cache = output.past_key_values
        next_ids = output.logits[:, -1].argmax(dim=-1)
        eos = torch.zeros_like(active)
        for eos_id in eos_token_ids:
            eos |= next_ids.eq(eos_id)
        for i in range(batch_size):
            if bool(active[i]):
                result[i].append(int(next_ids[i]))
                ended[i] = bool(eos[i])
        active &= ~eos
        if not bool(active.any()):
            break
        current = torch.where(active, next_ids, torch.full_like(next_ids, pad_token_id))[:, None]
        mask = torch.cat([mask, active.long()[:, None]], dim=1)
    if intervention is not None and hasattr(intervention, "set_prefill"):
        intervention.set_prefill(None)
    return result, ended


def generate_predictions(model, tokenizer, samples, representation, config, device, variant="canonical"):
    from .prompts import render_prompt
    ecfg = config.get("evaluation", {})
    batch_size = int(ecfg.get("generation_batch_size", ecfg.get("batch_size", 2)))
    if batch_size < 1:
        raise ValueError("evaluation.batch_size must be positive")
    output_rows = []
    was_training = model.training
    try:
        model.eval()
        with torch.no_grad():
            for start in range(0, len(samples), batch_size):
                group = samples[start:start + batch_size]
                tasks = {sample.task for sample in group}
                if len(tasks) != 1:
                    raise ValueError("Generation batches must contain the same target task")
                task = group[0].task
                limits = ecfg.get("max_new_tokens", DEFAULT_LIMITS)
                limit = int(limits.get(task, DEFAULT_LIMITS.get(task, 512)))
                prompt_limits = ecfg.get("max_prompt_tokens", {})
                max_prompt = int(prompt_limits.get(task, 1536 if task == "joint" else 512)) if isinstance(prompt_limits, dict) else int(prompt_limits)
                prompts = [render_prompt(tokenizer, sample, variant=variant) for sample in group]
                enc = [tokenizer(prompt, add_special_tokens=False)["input_ids"] for prompt in prompts]
                if any(not ids or len(ids) > max_prompt for ids in enc):
                    raise ValueError(f"{task}: empty or overlong evaluation prompt; no silent truncation is allowed")
                width = max(map(len, enc))
                ids = torch.full((len(group), width), tokenizer.pad_token_id, dtype=torch.long, device=device)
                mask = torch.zeros_like(ids)
                for i, tokens in enumerate(enc):
                    ids[i, -len(tokens):] = torch.tensor(tokens, dtype=torch.long, device=device)
                    mask[i, -len(tokens):] = 1
                context = intervention_context(model, representation)
                with context as intervention:
                    generated, ended = greedy_generate(model, ids, mask, max_new_tokens=limit,
                        intervention=intervention, eos_token_ids=_eos_ids(model, tokenizer),
                        pad_token_id=tokenizer.pad_token_id)
                for sample, prompt, tokens, stopped, prompt_ids in zip(group, prompts, generated, ended, enc):
                    from .prompts import target_text
                    reference_ids = tokenizer.encode(target_text(sample, variant), add_special_tokens=False)
                    if not reference_ids or reference_ids[-1] != tokenizer.eos_token_id:
                        reference_ids = [*reference_ids, tokenizer.eos_token_id]
                    output_rows.append({"id": sample.id, "group_id": sample.group_id, "task": sample.task,
                        "prompt": sample.prompt, "rendered_prompt": prompt, "target": sample.target,
                        "prediction": tokenizer.decode(tokens, skip_special_tokens=True),
                        "prediction_token_ids": tokens, "generated_tokens": len(tokens),
                        "prompt_tokens": len(prompt_ids), "max_new_tokens": limit,
                        "reference_tokens": len(reference_ids),
                        "reference_exceeds_generation_budget": len(reference_ids) > limit,
                        "truncated": not stopped, "stop_reason": "eos" if stopped else "length",
                        "variant": variant, "metadata": sample.metadata,
                        "decoding": {"do_sample": False, "num_beams": 1, "use_cache": True}})
    finally:
        model.train(was_training)
    return output_rows
