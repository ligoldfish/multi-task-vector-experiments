"""Causal answer-only likelihood with equal weight for every sample."""
from __future__ import annotations

import torch
import torch.nn.functional as F


def collate_encoded(examples, pad_token_id: int, device="cpu"):
    if not examples:
        raise ValueError("Cannot collate an empty batch")
    length = max(len(x["input_ids"]) for x in examples)
    ids = torch.full((len(examples), length), pad_token_id, dtype=torch.long)
    labels = torch.full_like(ids, -100)
    mask = torch.zeros_like(ids)
    positions = []
    for i, sample in enumerate(examples):
        n = len(sample["input_ids"])
        if n != len(sample["labels"]):
            raise ValueError("Input and label lengths differ")
        p = int(sample["prompt_length"])
        if p < 1 or p >= n:
            raise ValueError("A nonempty prompt and complete target are required")
        if any(y != -100 for y in sample["labels"][:p]):
            raise ValueError("Prompt tokens must not contribute to answer loss")
        if any(y == -100 for y in sample["labels"][p:]):
            raise ValueError("All answer tokens, including EOS, must be supervised")
        ids[i, :n] = torch.as_tensor(sample["input_ids"], dtype=torch.long)
        labels[i, :n] = torch.as_tensor(sample["labels"], dtype=torch.long)
        mask[i, :n] = 1
        positions.append(p - 1)
    return {"input_ids": ids.to(device), "attention_mask": mask.to(device),
            "labels": labels.to(device),
            "prompt_positions": torch.tensor(positions, dtype=torch.long, device=device)}


def answer_nll(logits: torch.Tensor, labels: torch.Tensor, chunk_size: int = 64) -> torch.Tensor:
    """Return a vector, not a token-weighted batch mean.

    Label at position k is predicted by logits at k-1. EOS is included by the
    data encoder; padding and the entire prompt carry label -100.
    """
    if logits.ndim != 3 or labels.shape != logits.shape[:2]:
        raise ValueError("Expected logits[batch,time,vocab] and labels[batch,time]")
    targets = labels[:, 1:].contiguous()
    valid = targets.ne(-100)
    counts = valid.sum(dim=1)
    if bool((counts == 0).any()):
        raise ValueError("Every sample must contain a supervised answer token")
    if chunk_size < 1:
        raise ValueError("Loss chunk size must be positive")
    # Qwen's large vocabulary makes a full sequence FP32 log-softmax a major
    # allocation. Chunking and recomputing CE preserve exactly the same answer
    # objective without retaining a second full vocabulary tensor for backward.
    if logits.shape[-1] <= 4096:
        token_losses = F.cross_entropy(logits[:, :-1].float().transpose(1, 2), targets,
                                      reduction="none", ignore_index=-100)
        return (token_losses * valid).sum(dim=1) / counts
    from torch.utils.checkpoint import checkpoint
    def chunk_sum(chunk_logits, chunk_labels):
        return F.cross_entropy(chunk_logits.float().transpose(1, 2), chunk_labels,
                               reduction="none", ignore_index=-100).sum(dim=1)
    sums = torch.zeros(logits.shape[0], device=logits.device, dtype=torch.float32)
    for start in range(0, targets.shape[1], chunk_size):
        stop = min(start + chunk_size, targets.shape[1])
        if not bool(valid[:, start:stop].any()):
            continue
        x, y = logits[:, start:stop], targets[:, start:stop]
        part = checkpoint(chunk_sum, x, y, use_reentrant=False) if torch.is_grad_enabled() and x.requires_grad else chunk_sum(x, y)
        sums = sums + part
    return sums / counts
