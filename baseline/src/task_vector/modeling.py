"""Load the documented frozen Qwen3 base on the selected runtime device."""
from __future__ import annotations

import torch


def load_model(config, device):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    c = config["model"]
    if c.get("enable_thinking", False):
        raise ValueError("This experiment fixes enable_thinking=False")
    if c.get("dtype", "bfloat16") != "bfloat16":
        raise ValueError("The documented base model must use BF16")
    source = c.get("path") or c.get("id", "Qwen/Qwen3-1.7B")
    kwargs = {"revision": c.get("revision"), "local_files_only": c.get("local_files_only", True),
              "trust_remote_code": False}
    tokenizer = AutoTokenizer.from_pretrained(source, **kwargs)
    if tokenizer.eos_token_id is None:
        raise ValueError("Tokenizer has no EOS token")
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(source, torch_dtype=torch.bfloat16,
        attn_implementation=c.get("attention_implementation", "eager"), **kwargs).to(device)
    expected = {"model_type": "qwen3", "num_hidden_layers": 28, "hidden_size": 2048,
                "num_attention_heads": 16, "num_key_value_heads": 8, "head_dim": 128,
                **c.get("expected_config", {})}
    mismatches = {key: {"expected": value, "actual": getattr(model.config, key, None)}
                  for key, value in expected.items() if getattr(model.config, key, None) != value}
    if mismatches:
        raise ValueError(f"Main experiment requires the exact Qwen3-1.7B architecture: {mismatches}")
    for p in model.parameters():
        p.requires_grad_(False)
    model.eval()
    model.generation_config.do_sample = False
    model.generation_config.use_cache = True
    return model, tokenizer
