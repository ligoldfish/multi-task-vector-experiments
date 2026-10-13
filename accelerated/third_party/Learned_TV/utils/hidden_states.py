import torch
import json
import re
from pathlib import Path
from safetensors.torch import safe_open
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from typing import Optional, Dict, List, Tuple, Callable, Union
from collections import defaultdict
from copy import deepcopy

from tqdm import tqdm
import random
from utils.get_icl import *
from utils import utils


import time                   
from contextlib import contextmanager


sim = torch.nn.CosineSimilarity(dim=1)

def remove_final_ln(model,llm_name, hs, attention_mask, layers, batch_len):
    if llm_name != '2p8B':
        position_ids = torch.arange(attention_mask.shape[-1], dtype=model.dtype, device=model.device).unsqueeze(0)
        causal_mask = prepare_4d_mask(attention_mask, model.dtype, model.device,cache_position=position_ids.squeeze(0), batch_size=batch_len).to(model.dtype)
        if llm_name == '6p9B':
            pos_emb=model.gpt_neox.rotary_emb(hs[0].unsqueeze(0),position_ids)
        else:
            pos_emb=model.model.rotary_emb(hs[0].unsqueeze(0),position_ids)
        hs[-1]=layers[-1](hs[-2], position_ids=position_ids, attention_mask=causal_mask, position_embeddings=pos_emb)[0] #default hidden states from model(**tokenized_batch,output_hidden_states=True) automatically adds final layer norm
    else:
        hs[-1]=layers[-1](hs[-2])[0]
    return hs
@contextmanager
def track_section_noflops(
    stats_dict,
    key,
    *,
    method="none",        # "none", "prefix", "lora", "ltv"
    L=None,               # seq len (must be passed by evaluate_set functions)
    d=None,               # hidden size
    m=0,                  # prefix length
    r=0,                  # LoRA rank
    n_layers=1,           # number of affected layers
    device="cuda"
):
    """
    Latency + memory + fast analytical FLOPs (GBFLOPs).
    """
    torch.cuda.synchronize(device)
    start_t = time.perf_counter()
    torch.cuda.reset_peak_memory_stats(device)
    start_mem = torch.cuda.memory_allocated(device)

    yield

    torch.cuda.synchronize(device)
    end_t = time.perf_counter()

    peak = torch.cuda.max_memory_allocated(device)
    end_mem = torch.cuda.memory_allocated(device)

    flops = 0

    stats_dict[key] = {
        "latency_sec": end_t - start_t,
        "peak_GB": peak / 1024**3,
        "delta_GB": (end_mem - start_mem) / 1024**3,
        "GBFLOPs": flops / 1024**3,
        "method": method
    }


@contextmanager
def track_section(
    stats_dict,
    key,
    *,
    method="none",
    L=None,
    d=None,
    m=0,
    r=0,
    n_layers=1,
    device="cuda"
):
    """
    Faster profiler:
    - subsampling
    - avoids key_averages()
    - keeps CPU for FLOPs
    """

    try:
        step_id = int(key.split("_")[-1])
    except:
        step_id = 0

    torch.cuda.synchronize(device)
    start_t = time.perf_counter()

    torch.cuda.reset_peak_memory_stats(device)
    start_mem = torch.cuda.memory_allocated(device)

    prof = None
    prof = torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA
        ],
        with_flops=True,
        record_shapes=False,
        profile_memory=False,
        with_stack=False,     
        with_modules=False
    )
    prof.__enter__()

    yield

    if prof:
        prof.__exit__(None, None, None)

    torch.cuda.synchronize(device)
    end_t = time.perf_counter()

    peak = torch.cuda.max_memory_allocated(device)
    end_mem = torch.cuda.memory_allocated(device)

    # -------- FLOPs --------
    total_flops = 0
    if prof:
        events = prof.events()  # 🔥 faster than key_averages
        total_flops = sum(e.flops for e in events if e.flops)

    stats_dict[key] = {
        "latency_sec": end_t - start_t,
        "peak_GB": peak / 1024**3,
        "delta_GB": (end_mem - start_mem) / 1024**3,
        "GBFLOPs": total_flops / 1024**3,
        "method": method
    }


def polar_rotation_from_W(W):
    # SVD: W = U Σ V^T  -> orthogonal factor Q = U V^T
    U, S, Vh = torch.linalg.svd(W, full_matrices=False)
    Q = U @ Vh          # orthogonal
    H = Vh.T @ torch.diag(S) @ Vh   # symmetric PSD
    return Q, H



    
def generate_with_intervention(
    model,
    tokenizer,
    prompt: str,
    tv: torch.Tensor,          # shape == model.config.hidden_size
    layers,                    # int or list[int], 0-based indices into model.model.layers
    max_new_tokens: int = 50,
    device: str = "cuda",
    **gen_kwargs,
):
    """
    Add `tv` to the hidden state at the last token of `prompt` at the specified `layers`
    during generation. Preserves output structure to avoid IndexError in generation.
    """
    model.eval()
    tv = tv.to(device)

    blocks = model.model.layers
    layer_ids = [layers] if isinstance(layers, int) else list(layers)

    # Tokenize prompt and move to device
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    prompt_len = inputs["input_ids"].size(1)

    # We only want to apply the intervention on the *first* forward pass (full prompt)
    applied = {"flag": False}

    def make_hook(pos):
        def hook_fn(module, inputs_, output):
            # Only intervene on the first pass with the full prompt (no past_kv)
            if applied["flag"]:
                return output

            if isinstance(output, (tuple, list)):
                tup = list(output)
                hidden = tup[0]
                container_type = tuple if isinstance(output, tuple) else list
            else:
                tup = None
                hidden = output
                container_type = None

            # Only apply when sequence length >= pos (first pass has full prompt)
            if hidden.size(1) >= pos:
                h_mod = hidden.clone()
                h_mod[:, pos - 1, :] = h_mod[:, pos - 1, :] + tv
                applied["flag"] = True  # ensure we don't touch subsequent 1-token decode steps

                if tup is None:
                    return h_mod
                tup[0] = h_mod
                if container_type is tuple:
                    return tuple(tup)
                elif container_type is list:
                    return tup
                else:
                    # Rebuild same ModelOutput type
                    return container_type(*tup)
            else:
                # Not the initial pass—leave untouched
                return output
        return hook_fn

    # Register hooks
    handles = []
    for li in layer_ids:
        if li < 0 or li >= len(blocks):
            raise IndexError(f"Layer index {li} is out of range (0..{len(blocks)-1}).")
        handles.append(blocks[li].register_forward_hook(make_hook(prompt_len)))

    try:
        # Important: do NOT wrap in torch.no_grad(); generation handles that internally.
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            top_k=30,
            temperature=0.8
        )
    finally:
        for h in handles:
            h.remove()
    if 'Yi' not in model.name_or_path:
        return tokenizer.decode(output_ids[0], skip_special_tokens=True)
    else:
        return tokenizer.decode(output_ids[0][0], skip_special_tokens=True)


def inject_tv(hiddens, model, tv, tv_layer, tv_pos, position_ids, causal_mask, num_layer, pos_emb):
    with torch.inference_mode():
        tv_hiddens=hiddens.clone()
        x=hiddens[tv_layer].clone()
        x[tv_pos]+=tv
        tv_hiddens[tv_layer]=x
        x=x.unsqueeze(0)
        
        for i in range(tv_layer,num_layer):
            block=model.model.layers[i]
            x=block(x, position_ids=position_ids, attention_mask=causal_mask, position_embeddings=pos_emb)[0]
            tv_hiddens[i+1]=x.squeeze(0)[-1]
    return tv_hiddens

def layer_updates(hiddens, model, tv_layer, num_layer, causal_mask, position_ids, pos_emb):
    with torch.inference_mode():
        x=hiddens[tv_layer].clone().unsqueeze(0)
        for i in range(tv_layer,num_layer):
            block=model.model.layers[i]
            x_new=block(x, position_ids=position_ids, attention_mask=causal_mask, position_embeddings=pos_emb)[0]
            x=x_new
    return x_new
    
def ablate_PTH_with_intervention(model, batch, hiddens, heads_list, topk, inject_layers, num_layer, layers, final_ln, Hd,device):
    '''
    compute the hidden states of the delimiter tokens of the icl demonstrations with different labels and also the query
    parameters:
    model: the model to use to compute the hidden states
    batch: the batch of icl demonstrations to get the hidden states from
    heads_list: the list of attention heads to ablate while computing the hidden states
    topk: the number of attention heads to ablate
    num_layer: the number of layers in the model

    output:
    hiddens: hidden states to be used in subsequent interpolations
            shape = number of labels * (icl_num+1) * d_model
    '''

    with torch.no_grad():
        N=1
        if topk!=0:
            heads_list_filtered=torch.tensor(heads_list)[:topk] 
            T,H=hiddens.shape[1],hiddens.shape[2]
            x=hiddens[inject_layers[0]].unsqueeze(0)
            hiddens=hiddens[:inject_layers[0]+1]
            for j in range(inject_layers[0], num_layer):
                block=layers[j]
                a=block.input_layernorm(x) if 'gpt' not in model.name_or_path else block.ln_1(x)
                heads=heads_list_filtered[heads_list_filtered[:,0]==j]
                if 'gpt' in model.name_or_path:
                    query, key, value,_ = block.attention._attn_projections_and_rope(a,position_ids=torch.arange(T).unsqueeze(dim=0))
                    for k in range(heads.shape[0]):
                        value[:,heads[k,1]]=0
                    mask = torch.tril(torch.ones(T, T)).unsqueeze(0).unsqueeze(0).to(device)
                    mask = mask.masked_fill(mask == 0, float('-inf'))
                    attn_outputs = block.attention._attn(query, key, value, attention_mask=mask)
                    a = attn_outputs[0]
                    a = utils._merge_heads(a)
                    a = block.attention.dense(a)
                    b=block.post_attention_layernorm(x) if 'gpt' not in model.name_or_path else block.ln_2(x)
                    b = block.mlp(b)
                    x=x+a+b
                else:

                    query= block.self_attn.q_proj(a).view(N, T, -1, Hd).transpose(1, 2)
                    key= block.self_attn.k_proj(a).view(N, T, -1, Hd).transpose(1, 2)
                    value= block.self_attn.v_proj(a).view(N, T, -1, Hd).transpose(1, 2)

                    # if 'gemma-2b' in model.name_or_path:
                    #     for i in range(len(heads)):
                    #         value[:,:,:,heads[i,1]*Hdkv:(heads[i,1]+1)*Hdkv]=0
                    # else:
                    #     for i in range(len(heads)):
                    #         # value[:,heads[i,1]]=0
                    #         print(heads[i,1])
                    #         print(value.shape)
                    #         value[:, heads[i,1]] = 0
                    position_ids = torch.arange(a.shape[1], dtype=model.dtype, device=model.device).unsqueeze(0)
                    causal_mask = prepare_4d_mask(batch['attention_mask'], model.dtype, model.device,cache_position=position_ids.squeeze(0), batch_size=N).to(model.dtype)
                    pos_emb=model.model.rotary_emb(hiddens[0],position_ids)
                    cos, sin = pos_emb
                    query, key = apply_rotary_pos_emb(query, key, cos, sin)
                    attn_outputs = scaled_dot_product_attention(
                        query, key, value, attn_mask=causal_mask, dropout_p=0.0, enable_gqa=True, masked_attn_heads=heads, 
                    )
                    
                    attn_outputs= attn_outputs.transpose(1, 2).contiguous().view(N, T, -1)
                    a= block.self_attn.o_proj(attn_outputs)
                    x=x+a
                    b=block.post_attention_layernorm(x) if 'gpt' not in model.name_or_path else block.ln_2(x)
                    b = block.mlp(b)
                    x=x+b
                #if j == num_layer-1:
                    #x=model.gpt_neox.final_layer_norm(x)
                hiddens=torch.cat([hiddens,x],dim=0)
            hiddens=hiddens.view(-1,T,H)
        else:
            attention_mask=batch['attention_mask']
            position_ids = torch.arange(attention_mask.shape[-1], dtype=model.dtype, device=model.device).unsqueeze(0)
            causal_mask = prepare_4d_mask(attention_mask, model.dtype, model.device,cache_position=position_ids.squeeze(0), batch_size=N).to(model.dtype)
            pos_emb=model.model.rotary_emb(hiddens[0].unsqueeze(0),position_ids)
            hiddens[-1]=layers[-1](hiddens[-2].unsqueeze(0), position_ids=position_ids, attention_mask=causal_mask, position_embeddings=pos_emb)[0].squeeze(0)
        return hiddens
def get_last_token_head_outputs_with_bias_hooked(
    model,
    head_index: torch.Tensor,
    shard_reader,                        # ColumnShardReader for o_proj
    hiddens: torch.Tensor,
    layers: list[int],
    coef: torch.Tensor,
    attention_mask: torch.Tensor,
    position_ids: torch.Tensor,
    causal_mask: torch.Tensor,
    llm_name: str,
    pos_to_intervene: int = -1,
    dtype=torch.float32,
    device=None
) -> torch.Tensor:
    """
    Return outputs of selected attention heads at final token after bias is applied.
    Uses hooks on o_proj to exactly match head-specific computation.
    """
    cfg = model.config
    H = cfg.hidden_size
    n_head = cfg.num_attention_heads
    head_dim = H // n_head

    want, order = defaultdict(list), []
    for lid, h in head_index.tolist():
        want[lid].append(h)
        order.append((lid, h))

    if device is None:
        device = next(model.parameters()).device

    captured, handles = {}, []

    def make_hook(layer_id):
        proj = model.model.layers[layer_id].self_attn.o_proj
        bias = proj.bias
        def _hook(_mod, inputs, _out):
            x = inputs[0]                           # (1, seq_len, hidden_size)
            L = x.size(1)
            x = x.view(1, L, n_head, head_dim).transpose(1, 2)  # (1, n_head, seq_len, d_head)
            for h in want[layer_id]:
                s, e = h * head_dim, (h + 1) * head_dim
                W_o = shard_reader.slice(layer_id, s, e, dtype=x.dtype, device=x.device)  # (H, d_head)
                out = x[:, h, -1] @ W_o.t()  # (1, H)
                if bias is not None:
                    out = out + bias.to(dtype=out.dtype, device=out.device)
                captured[(layer_id, h)] = out.squeeze(0)
        return _hook

    # Register hooks
    for lid in want:
        handles.append(
            model.model.layers[lid].self_attn.o_proj.register_forward_hook(make_hook(lid))
        )

    # Run forward pass with bias
    try:
        with torch.no_grad():
            x, modified_hiddens,attn = forward_through_layers(
                hiddens=hiddens,
                layers=layers,
                num_layer=cfg.num_hidden_layers,
                model=model,
                attention_mask=attention_mask,
                position_ids=position_ids,
                causal_mask=causal_mask,
                llm_name=llm_name,
                coef=coef,
                pos_to_intervene=pos_to_intervene,
                dtype=dtype,
            )
    finally:
        for h in handles:
            h.remove()

    return x, modified_hiddens, attn,torch.stack([captured[(l, h)] for (l, h) in order])  # (N, hidden_size)

def _ensure_2d(t: torch.Tensor, rows: int) -> torch.Tensor:
    """
    If t is flat ([rows*cols] or [rows*cols, 1]), reshape to (rows, –1)
    else leave alone, then return a contiguous tensor.
    """
    if t.ndim == 1 or 1 in t.shape:
        t = t.view(rows, -1)
    return t.contiguous()


class ColumnWeightReader:
    def __init__(self, snapshot_dir: str, hidden_size: int, proj_type: str, transpose=True):
        self.snapshot_dir = Path(snapshot_dir)
        self.transpose = transpose
        self.rows = hidden_size
        self.proj_type = proj_type  # one of q_proj, k_proj, v_proj
        self.open_files = {}
        self.full_cache = {}

        # Load weight map
        idx_path = self.snapshot_dir / "model.safetensors.index.json"
        st_path = self.snapshot_dir / "model.safetensors"
        if idx_path.exists():
            wm = json.loads(idx_path.read_text())["weight_map"]
        elif st_path.exists():
            handle = safe_open(st_path, framework="pt", device="cpu")
            wm = {k: "model.safetensors" for k in handle.keys()}
        else:
            raise FileNotFoundError(f"No safetensors found in {snapshot_dir}")
        
        self.weight_map = wm
        self.layer_shards: dict[int, dict] = {}
        pattern = re.compile(
            rf"^(model\.layers\.(\d+)\.self_attn\.{proj_type}\.weight)(?:[._](\d+))?$"
        )
        for key, fname in wm.items():
            m = pattern.match(key)
            if not m:
                continue
            base, lid_str, shard_id = m.group(1), m.group(2), m.group(3)
            lid = int(lid_str)
            entry = self.layer_shards.setdefault(lid, {"full": None, "tp": {}})
            if shard_id is None:
                entry["full"] = (key, fname)
            else:
                entry["tp"][int(shard_id)] = (key, fname)

        # Sanity check
        for lid, entry in self.layer_shards.items():
            tp_keys = set(entry["tp"].keys())
            if entry["full"] is None and tp_keys != {0, 1}:
                raise KeyError(f"Layer {lid} missing shards: {tp_keys}")

    def _get_handle(self, fname: str):
        h = self.open_files.get(fname)
        if h is None:
            h = safe_open(self.snapshot_dir / fname, framework="pt", device="cpu")
            self.open_files[fname] = h
        return h

    def slice(
        self,
        layer_id: int,
        start: int,
        end: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> torch.Tensor:
        if layer_id not in self.full_cache:
            entry = self.layer_shards[layer_id]
            if entry["full"] is not None:
                key, fn = entry["full"]
                W = self._get_handle(fn).get_tensor(key)
                if self.transpose:
                    W = _ensure_2d(W, rows=self.rows).T
                else:
                    W = _ensure_2d(W, rows=self.rows)
            else:
                # two-shard TP case
                key0, fn0 = entry["tp"][0]
                key1, fn1 = entry["tp"][1]
                if self.transpose:
                    W0 = _ensure_2d(self._get_handle(fn0).get_tensor(key0), rows=self.rows).T  # (in_features, out//2)
                    W1 = _ensure_2d(self._get_handle(fn1).get_tensor(key1), rows=self.rows).T  # (in_features, out//2)
                else:
                    W0 = _ensure_2d(self._get_handle(fn0).get_tensor(key0), rows=self.rows)
                    W1 = _ensure_2d(self._get_handle(fn1).get_tensor(key1), rows=self.rows)
                W  = torch.cat([W0, W1], dim=1)  # (in_features, out_features)

            self.full_cache[layer_id] = W  # CPU float32

        W_full = self.full_cache[layer_id]
        sub = W_full[:, start:end]
        return sub.to(dtype=dtype, device=device, non_blocking=True)
    def return_full(
        self,
        layer_id: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> torch.Tensor:
        if layer_id not in self.full_cache:
            entry = self.layer_shards[layer_id]
            if entry["full"] is not None:
                key, fn = entry["full"]
                W = self._get_handle(fn).get_tensor(key)
                if self.transpose:
                    W = _ensure_2d(W, rows=self.rows).T
                else:
                    W = _ensure_2d(W, rows=self.rows)
            else:
                # two-shard TP case
                key0, fn0 = entry["tp"][0]
                key1, fn1 = entry["tp"][1]
                if self.transpose:
                    W0 = _ensure_2d(self._get_handle(fn0).get_tensor(key0), rows=self.rows).T  # (in_features, out//2)
                    W1 = _ensure_2d(self._get_handle(fn1).get_tensor(key1), rows=self.rows).T  # (in_features, out//2)
                else:
                    W0 = _ensure_2d(self._get_handle(fn0).get_tensor(key0), rows=self.rows)
                    W1 = _ensure_2d(self._get_handle(fn1).get_tensor(key1), rows=self.rows)
                W  = torch.cat([W0, W1], dim=1)  # (in_features, out_features)

            self.full_cache[layer_id] = W  # CPU float32

        W_full = self.full_cache[layer_id]
        return W_full.to(dtype=dtype, device=device, non_blocking=True)

    def clear_cache(self):
        self.full_cache.clear()

class QProjReader(ColumnWeightReader):
    def __init__(self, snapshot_dir: str, hidden_size: int):
        super().__init__(snapshot_dir, hidden_size, proj_type="q_proj")

class KProjReader(ColumnWeightReader):
    def __init__(self, snapshot_dir: str, hidden_size: int):
        super().__init__(snapshot_dir, hidden_size, proj_type="k_proj")

class VProjReader(ColumnWeightReader):
    def __init__(self, snapshot_dir: str, hidden_size: int):
        super().__init__(snapshot_dir, hidden_size, proj_type="v_proj")
        
class OProjReader(ColumnWeightReader):
    def __init__(self, snapshot_dir: str, hidden_size: int):
        super().__init__(snapshot_dir, hidden_size, proj_type="o_proj", transpose=False)
        
def create_qkvo_readers(snapshot_dir: str, hidden_size: int):
    """
    Creates Q, K, V, O projection readers from a model snapshot directory.

    Args:
        snapshot_dir: path to directory containing `model.safetensors` files
        hidden_size: model hidden size (e.g., 8192 for LLaMA-70B)

    Returns:
        Tuple of (q_reader, k_reader, v_reader, o_reader), each a ColumnWeightReader subclass
    """
    q_reader = QProjReader(snapshot_dir, hidden_size)
    k_reader = KProjReader(snapshot_dir, hidden_size)
    v_reader = VProjReader(snapshot_dir, hidden_size)
    o_reader = OProjReader(snapshot_dir, hidden_size)
    return q_reader, k_reader, v_reader, o_reader

def get_attention_pattern(
    model,
    input_ids: torch.Tensor,               # (1, L)
    head_index: torch.Tensor,              # (M, 2) – (layer_id, head_id) pairs
    attention_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:  # returns tensor of shape (M, L, L)
    """
    Returns an (M, L, L) tensor of attention patterns for selected heads,
    in the same order as `head_index`.

    Args:
        model: Transformer model (with model.model.layers[lid].self_attn)
        input_ids (1, L): token IDs
        head_index (M, 2): tensor of (layer_id, head_id) pairs to record
        attention_mask: optional attention mask

    Returns:
        Tensor of shape (M, L, L), where M = head_index.size(0).  Row i
        corresponds to the attention matrix (L×L) for head (layer_id, head_id) = head_index[i].
    """
    cfg    = model.config
    n_head = cfg.num_attention_heads
    L      = input_ids.shape[1]

    # Group heads by layer and remember the original order
    want, order = defaultdict(list), []
    for lid, h in head_index.tolist():
        want[lid].append(h)
        order.append((lid, h))

    captured, handles = {}, []

    # Hook factory: captures attn_probs for a given layer, then stores only the requested heads
    def make_hook(layer_id):
        def _hook(_mod, _inp, output):
            # output[1] is attn_probs of shape (B, n_head, L, L)
            attn_weights = output[1]  # (1, n_head, L, L)
            for h in want[layer_id]:
                # detach and move to CPU, storing shape (L, L)
                captured[(layer_id, h)] = attn_weights[0, h].detach().cpu()
        return _hook

    # Register one hook per layer that appears in head_index
    for lid in want:
        handle = model.model.layers[lid].self_attn.register_forward_hook(make_hook(lid))
        handles.append(handle)

    try:
        model.eval()
        device = next(model.parameters()).device
        ids    = input_ids.to(device)
        mask   = attention_mask.to(device) if attention_mask is not None else None

        with torch.no_grad():
            # We need output_attentions=True so attentions are returned to the hook
            model(input_ids=ids, attention_mask=mask, use_cache=False, output_attentions=True)
    finally:
        for h in handles:
            h.remove()

    # Now build an (M, L, L) tensor in the same order as head_index
    M = head_index.size(0)
    attn_tensor = torch.zeros((M, L, L), dtype=captured[order[0]].dtype)
    for i, (lid, h) in enumerate(order):
        attn_tensor[i] = captured[(lid, h)]

    return attn_tensor  # shape (M, L, L)
def get_head_outputs(
    model,
    input_ids: torch.Tensor,            # (1, L)
    head_index: torch.Tensor,           # (N, 2)  -> rows are (layer_id, head_id)
    shard_reader,                       # ColumnWeightReader
    attention_mask: torch.Tensor | None = None,
    pos_to_intervene: torch.Tensor | None = None,  # (K,) long tensor of positions
) -> torch.Tensor:
    """
    Return per-head outputs at specified token positions.

    Args:
        model: HF-style model with model.model.layers[*].self_attn.o_proj
        input_ids: (1, L)
        head_index: (N, 2) rows: (layer_id, head_id)
        shard_reader: ColumnWeightReader exposing .slice(layer_id, s, e, dtype, device) -> (H, head_dim)
        attention_mask: (1, L) or None
        pos_to_intervene: (K,) positions to read (0-based). If None, defaults to last position [L-1].

    Returns:
        outputs: (N, K, H) stacked in the same order as head_index
                 where H = hidden_size
    """
    cfg       = model.config
    H         = cfg.hidden_size
    n_head    = cfg.num_attention_heads
    head_dim  = H // n_head

    # group heads by layer for a single hook per layer
    want, order = defaultdict(list), []
    for lid, h in head_index.tolist():
        want[lid].append(h)
        order.append((lid, h))

    captured, handles = {}, []

    def make_hook(layer_id: int):
        proj = model.model.layers[layer_id].self_attn.o_proj
        bias = proj.bias
        def _hook(_mod, inputs, _out):
            # inputs[0]: (1, L, H) pre-o_proj activations
            x = inputs[0]
            B, L, HH = x.shape
            assert B == 1 and HH == H, "Unexpected input shape to o_proj."

            # Choose positions (default to last position if not provided)
            if pos_to_intervene is None:
                pos_idx = torch.tensor([L - 1], device=x.device, dtype=torch.long)
            else:
                pos_idx = pos_to_intervene.to(x.device).long()
                if torch.any((pos_idx >= L)):
                    raise ValueError(f"pos_to_intervene must be within [0, {L-1}]")
                if torch.any((pos_idx <0)):
                    pos_idx = pos_idx + L  # allow negative indices
                

            K = pos_idx.numel()

            # reshape to (1, n_head, L, head_dim)
            x = x.view(1, L, n_head, head_dim).transpose(1, 2)

            for h in want[layer_id]:
                s, e     = h * head_dim, (h + 1) * head_dim
                # (H, head_dim)
                w_slice  = shard_reader.slice(
                    layer_id, s, e, dtype=x.dtype, device=x.device
                )
                # select the K positions for this head: (1, K, head_dim)
                x_sel = x[:, h, pos_idx, :]  # (1, K, head_dim)

                # (1, K, head_dim) @ (head_dim, H) -> (1, K, H)
                out = torch.matmul(x_sel, w_slice.t())

                if bias is not None:
                    out = out + bias.to(dtype=out.dtype, device=out.device)

                # store as (K, H)
                captured[(layer_id, h)] = out.squeeze(0)
        return _hook

    # register one hook per needed layer
    for lid in want:
        handles.append(
            model.model.layers[lid]
                 .self_attn.o_proj
                 .register_forward_hook(make_hook(lid))
        )

    # forward
    try:
        dev  = next(model.parameters()).device
        ids  = input_ids.to(dev)
        mask = attention_mask.to(dev) if attention_mask is not None else None
        with torch.no_grad():
            model(input_ids=ids, attention_mask=mask, use_cache=False)
    finally:
        for h in handles:
            h.remove()

    # stack in the original order -> (N, K, H)
    return torch.stack([captured[(l, h)] for (l, h) in order], dim=0)
def get_last_token_head_outputs(
    model,
    input_ids: torch.Tensor,            # (1, L)
    head_index: torch.Tensor,           # (N, 2)
    shard_reader: ColumnWeightReader,
    attention_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    cfg       = model.config
    H         = cfg.hidden_size
    n_head    = cfg.num_attention_heads
    head_dim  = H // n_head

    # group heads by layer for a single hook per layer
    want, order = defaultdict(list), []
    for lid, h in head_index.tolist():
        want[lid].append(h)
        order.append((lid, h))

    captured, handles = {}, []

    def make_hook(layer_id: int):
        proj = model.model.layers[layer_id].self_attn.o_proj
        bias = proj.bias
        def _hook(_mod, inputs, _out):
            x = inputs[0]                           # (1, L, H)
            L = x.size(1)
            x = x.view(1, L, n_head, head_dim).transpose(1, 2)
            for h in want[layer_id]:
                s, e     = h * head_dim, (h + 1) * head_dim
                w_slice  = shard_reader.slice(
                             layer_id, s, e,
                             dtype=x.dtype, device=x.device
                           )                  # (H, head_dim)
                out      = x[:, h, -1] @ w_slice.t()  # (1, H)
                if bias is not None:
                    out = out + bias.to(dtype=out.dtype, device=out.device)
                captured[(layer_id, h)] = out.squeeze(0)
        return _hook

    # register one hook per needed layer
    for lid in want:
        handles.append(
            model.model.layers[lid]
                 .self_attn.o_proj
                 .register_forward_hook(make_hook(lid))
        )

    # forward
    try:
        dev  = next(model.parameters()).device
        ids  = input_ids.to(dev)
        mask = attention_mask.to(dev) if attention_mask is not None else None
        with torch.no_grad():
            model(input_ids=ids, attention_mask=mask, use_cache=False)
    finally:
        for h in handles:
            h.remove()

    # stack in the original order
    return torch.stack([captured[(l, h)] for (l, h) in order])  # (N, H)


def create_vector_database(model,llm_name,tokenizer,prompts,batch_size,device):
  '''
    get the vector representations of a series of prompts

    parameters:
    model: the model to use to compute the hidden states
    tokenizer: the tokenizer to use to encode the prompts
    d_model: the dimension of the hidden states
    prompts: the prompts to get the vector representations of
    device: the device to use for computation
    
    output:
    representations: the vector representations of the prompts
  '''
  all_embeddings=[]
  for i in tqdm(range(0,len(prompts),batch_size)):
      batch=prompts[i: i + batch_size]
      inputs=tokenizer(batch,return_tensors='pt',padding=True).to(device)
      with torch.no_grad():  # Disable gradient tracking for lower memory usage
          outputs = model(**inputs,output_hidden_states=True).hidden_states
      hidden_states = outputs[-1]  # Shape: (batch_size, seq_len, hidden_dim)

      # Get the last token's actual position (ignoring padding)
      input_ids = inputs["input_ids"]
      seq_lengths = (input_ids != tokenizer.pad_token_id).sum(dim=1) - 1  # Find last non-padding token
      if llm_name == 'opt':
          seq_lengths += 1
      # Extract embeddings from the correct positions
      last_token_embeddings = hidden_states[torch.arange(hidden_states.shape[0]), seq_lengths]
      all_embeddings.append(last_token_embeddings.cpu())
      del inputs,outputs,hidden_states, last_token_embeddings
  return torch.cat(all_embeddings)

def compute_statistics(representations):
  '''
    compute the mean covariance matrix of the vector representations of the prompts

    parameters:
    representations: the vector representations of the prompts
    
    output:
    mean: the mean of the vector representations
    cov: the covariance matrix of the vector representations
  '''
  mean=representations.mean(dim=0)
  d_model=representations.shape[1]
  for i in range(representations.shape[0]):
    if i==0:
      cov=torch.ger(representations[i]-mean,representations[i]-mean)
    else:
      cov+=torch.ger(representations[i]-mean,representations[i]-mean)
  cov=cov/representations.shape[0]
  return mean,cov

def compute_distance(query_hs, representations, sampled_indices=None, metric='euclidean'):
    '''
    Compute the distance or similarity between a single query hidden state or a batch of query 
    hidden states and the representation database.

    Parameters:
    query_hs: query hidden state with shape (embedding_dim) or (batch_size, embedding_dim)
    representations: database of embeddings with shape (num_samples, embedding_dim)
    sampled_indices: optional indices to subset the representations, shape (batch_size, num_samples)
    metric: either 'euclidean' for L2 distance or 'cosine' for cosine similarity

    Output:
    distance: tensor of shape (num_samples) or (batch_size, num_samples) containing distances or similarities
    '''
    if query_hs.ndim == 1:  # Single query
        query_hs = query_hs.unsqueeze(0)  # Add batch dimension: (1, embedding_dim)

    if sampled_indices is not None:
        if sampled_indices.ndim == 1:  # If 1D, reshape to (1, num_samples) for batch size 1
            sampled_indices = sampled_indices.unsqueeze(0)

        # Gather sampled representations for each batch
        sampled_hs = torch.stack([
            representations[sampled_indices[b]] for b in range(query_hs.shape[0])
        ], dim=0)  # Shape: (batch_size, num_samples, embedding_dim)
    else:
        sampled_hs = representations.unsqueeze(0).expand(query_hs.shape[0], -1, -1)

    if metric == 'euclidean':
        distance = torch.norm(query_hs.unsqueeze(1) - sampled_hs, dim=-1)
    elif metric == 'cosine':
        # Normalize vectors to unit norm
        query_hs_norm = F.normalize(query_hs, p=2, dim=-1)  # Shape: (batch_size, embedding_dim)
        sampled_hs_norm = F.normalize(sampled_hs, p=2, dim=-1)  # Shape: (batch_size, num_samples, embedding_dim)
        # Compute cosine similarity
        distance = -torch.bmm(sampled_hs_norm, query_hs_norm.unsqueeze(-1)).squeeze(-1)  # Shape: (batch_size, num_samples)
    return distance.squeeze(0) if query_hs.shape[0] == 1 else distance


def get_heads_token_contributions(
    model,
    input_ids: torch.Tensor,             # shape (1, L)
    head_index: torch.LongTensor,        # shape (n, 2), each row = [layer_id, head_id]
    shard_reader=None,                   # ColumnShardReader or None
    attention_mask: torch.Tensor | None = None,
    target_pos: int | None = None,
    v_group_size: int = 3,               # usually 4 in LLaMA‐3
) -> torch.Tensor:
    """
    Generalized version of get_head_token_contributions that accepts an (n, 2) tensor
    of (layer_id, head_id) pairs and returns an (n, L, H) tensor, where row i is
       contributions[i] = per‐token contributions for head (layer_i, head_i).

    If `shard_reader` is not None, we fetch each layer’s o_proj.weight via shard_reader.slice(...) 
    (shape H×H) and then slice per head. Otherwise, we index model.model.layers[layer_id].self_attn.o_proj.weight.

    Args:
      model:         HuggingFace LLaMA‐style model (e.g. LlamaForCausalLM).
      input_ids:     LongTensor of shape (1, L).
      head_index:    LongTensor of shape (n, 2); each row is [layer_id, head_id].
      shard_reader:  ColumnShardReader (pointing at a safetensors/TP‐sharded checkpoint) or None.
      attention_mask:(1, L) mask or None.
      target_pos:    Optional index in [0..L−1]. If None, defaults to L−1.
      v_group_size:  How many heads share one V (default=4 for LLaMA‐3).

    Returns:
      Tensor of shape (n, L, H), dtype = model.dtype, device = model.device, where
      contributions[i] is the (L, H) per‐token map for head (layer_i, head_i).
    """
    cfg      = model.config
    H        = cfg.hidden_size              # hidden dimension
    n_head   = cfg.num_attention_heads       # number of heads per layer
    L        = input_ids.size(1)             # sequence length
    head_dim = H // n_head

    if target_pos is None:
        target_pos = L - 1

    device = next(model.parameters()).device
    dtype  = next(model.parameters()).dtype

    # Step 1: Group requested heads by layer
    heads_by_layer: dict[int, list[int]] = defaultdict(list)
    for (l, h) in head_index.tolist():
        heads_by_layer[l].append(h)

    # Containers to store, per layer:
    ln1_outputs:  dict[int, torch.Tensor] = {}  # layer_id → x_ln (shape L×H)
    attn_outputs: dict[int, torch.Tensor] = {}  # layer_id → attn_probs (shape n_head×L×L)

    handles = []

    # Hook to capture ln1(x)  (input_layernorm output) at a given layer
    def make_ln1_hook(layer_id: int):
        def ln1_hook(_mod, _inputs, out):
            # out: shape (1, L, H); detach and store [0] → (L, H)
            ln1_outputs[layer_id] = out.detach()[0]
        return ln1_hook

    # Hook to capture attn_probs at a given layer
    def make_attn_hook(layer_id: int):
        def attn_hook(_mod, _inputs, out):
            # out[1] is attn_probs of shape (1, n_head, L, L)
            attn_outputs[layer_id] = out[1].detach()[0]  # → (n_head, L, L)
        return attn_hook

    # Register hooks for each unique layer in head_index
    for layer_id in heads_by_layer:
        ln1_handle = model.model.layers[layer_id].input_layernorm.register_forward_hook(
            make_ln1_hook(layer_id)
        )
        attn_handle = model.model.layers[layer_id].self_attn.register_forward_hook(
            make_attn_hook(layer_id)
        )
        handles.extend([ln1_handle, attn_handle])

    # Step 2: Run one forward pass (eval, no dropout) to populate ln1_outputs and attn_outputs
    try:
        model.eval()
        ids  = input_ids.to(device)
        mask = attention_mask.to(device) if (attention_mask is not None) else None

        with torch.no_grad():
            model(input_ids=ids, attention_mask=mask, output_attentions=True, use_cache=False)
    finally:
        for h in handles:
            h.remove()

    # Step 3: Build output tensor (n, L, H)
    n = head_index.size(0)
    contributions = torch.zeros((n, L, H), device=device, dtype=dtype)

    # Caches per layer:
    V_all_cache: dict[int, torch.Tensor]    = {}  # layer_id → (L, H)
    W_v_cache:   dict[int, torch.Tensor]    = {}
    W_o_cache:   dict[int, torch.Tensor]    = {}  # layer_id → full H×H (either from in‐memory or shard)
    W_o_from_shard: dict[int, torch.Tensor] = {}  # if we fetched from shard_reader

    for layer_id, head_list in heads_by_layer.items():
        # (a) Grab x_ln (L×H) and attn_probs (n_head×L×L)
        x_ln = ln1_outputs[layer_id]       # shape (L, H)
        attn = attn_outputs[layer_id]      # shape (n_head, L, L)

        # (b) Compute or fetch V_all = x_ln @ W_v^T  (→ L×H)
        if layer_id not in V_all_cache:
            W_v = model.model.layers[layer_id].self_attn.v_proj.weight  # (H, H)
            if W_v.dtype != dtype or W_v.device != device:
                W_v = W_v.to(device).to(dtype)
            W_v_cache[layer_id] = W_v
            V_all_cache[layer_id] = x_ln @ W_v.T  # (L, H)
        V_all = V_all_cache[layer_id]

        # (c) Compute or fetch full W_o (H×H)
        if layer_id not in W_o_cache:
            if shard_reader is None:
                # In‐memory: simply index the module weight
                W_o_full = model.model.layers[layer_id].self_attn.o_proj.weight  # (H, H)
                if W_o_full.dtype != dtype or W_o_full.device != device:
                    W_o_full = W_o_full.to(device).to(dtype)
                W_o_cache[layer_id] = W_o_full
                W_o_from_shard[layer_id] = None
            else:
                # Use shard_reader to assemble the full (H×H) from disk/CPU
                W_o_full = shard_reader.slice(
                    layer_id,
                    0,        # start column
                    H,        # end column (gives full H×H)
                    dtype = dtype,
                    device = device
                )  # → (H, H)
                W_o_cache[layer_id] = W_o_full
                W_o_from_shard[layer_id] = True

        W_o_full = W_o_cache[layer_id]  # shape (H, H)

        # Now compute each requested head in this layer
        for h in head_list:
            # Find all row‐indices i in head_index where (layer_id, head_id) == (layer_id, h)
            matches = (head_index[:, 0] == layer_id) & (head_index[:, 1] == h)
            idxs = torch.nonzero(matches, as_tuple=False).view(-1).tolist()

            # (1) α^h = attn[h, target_pos, :]  → (L,)
            alpha_h = attn[h, target_pos, :]   # (L,)

            # (2) V_shared^h = V_all[:, v_start:v_end] → (L, head_dim)
            v_group_id = h // v_group_size
            v_start = v_group_id * head_dim
            v_end   = (v_group_id + 1) * head_dim
            V_shared_h = V_all[:, v_start:v_end]  # (L, head_dim)

            # (3) W_o_head^h = slice rows [h*head_dim : (h+1)*head_dim] from W_o_full
            start_o = h * head_dim
            end_o   = (h + 1) * head_dim
            W_o_head_h = W_o_full[start_o:end_o, :]  # (head_dim, H)

            # (4) Compute per‐token contributions:  (L, head_dim) @ (head_dim, H) → (L, H)
            contr_h = (alpha_h.unsqueeze(1) * V_shared_h) @ W_o_head_h  # (L, H)

            # (5) If bias exists, add it. (Often bias=None for LLaMA‐style o_proj.)
            b_o = model.model.layers[layer_id].self_attn.o_proj.bias
            if b_o is not None:
                b = b_o.to(device).to(dtype)  # (H,)
                contr_h = contr_h + b.unsqueeze(0)

            # (6) Scatter into `contributions[i]` for each matching i
            for i in idxs:
                contributions[i] = contr_h.clone()

    return contributions  # shape (n, L, H)

def select_bad_neighbors(query_hs, representations, icl_num, coef=1, retain=4, metric='euclidean', reverse=True):
    """
    Selects icl_num hidden states from the vector database for each query hidden state
    such that they are close to the query while being minimally diverse.

    Args:
        query_hs (torch.Tensor): Tensor of shape (batch_size, d_model), representing the query hidden states.
        representations (torch.Tensor): Tensor of shape (demo_num, d_model), representing the vector database.
        icl_num (int): Number of neighbors to select.

    Returns:
        torch.Tensor: Indices of selected hidden states, shape (batch_size, icl_num).
    """
    batch_size, d_model = query_hs.shape
    demo_num, _ = representations.shape

    # Compute pairwise L2 distances between all query_hs and representations
    if metric=='euclidean':
        query_norms = torch.sum(query_hs**2, dim=1, keepdim=True)  # (batch_size, 1)
        repr_norms = torch.sum(representations**2, dim=1)  # (demo_num)
        distances = (query_norms + repr_norms - 2 * (query_hs @ representations.T))**0.5  # (batch_size, demo_num)
    if metric=='cosine':
        query_hs_temp=F.normalize(query_hs, p=2, dim=-1)
        representations_temp=F.normalize(representations.unsqueeze(0).expand(batch_size,-1,-1), p=2, dim=-1)
        distances=-torch.bmm(representations_temp, query_hs_temp.unsqueeze(-1)).squeeze(-1)

    # Sort distances to get the closest neighbors
    sorted_indices = torch.argsort(distances, dim=1)  # (batch_size, demo_num)

    # Initialize selection with the closest neighbor
    selected_indices = torch.zeros((batch_size, icl_num), dtype=torch.long, device=query_hs.device)
    selected_indices[:, :retain] = sorted_indices[:, :retain]  # First selection is the nearest point

    # Mask to track selected points
    selected_mask = torch.zeros((batch_size, demo_num), dtype=torch.bool, device=query_hs.device)
    selected_mask[:, :retain]=True
    for j in range(retain, icl_num):
        # Get currently selected representations (batch_size, j, d_model)
        selected_reprs = representations[selected_indices[:, :j]]

        # Compute pairwise distances among selected points (batch_size, j, j)
        pairwise_dists = torch.cdist(selected_reprs.to(torch.float32), selected_reprs.to(torch.float32), p=2).to(query_hs.dtype) if metric=='euclidean' else -torch.bmm(F.normalize(selected_reprs, p=2, dim=-1), F.normalize(selected_reprs, p=2, dim=-1).transpose(1,2))
        max_inner_dist = pairwise_dists.max(dim=2)[0].max(dim=1)[0]  # (batch_size,)

        # Compute max distance between query and selected points (batch_size,)
        max_query_dist = distances.gather(1, selected_indices[:, :j]).max(dim=1)[0]
        remaining_indices=sorted_indices[~selected_mask].view(batch_size,-1)
        a=sorted_indices[:, j:]
        remaining_dists = distances.gather(1, remaining_indices)

        # Compute max_query_dist for each candidate
        new_max_query_dist = torch.max(max_query_dist.unsqueeze(1), remaining_dists)
        # Compute new max_inner_dist when adding each candidate
        last_selected_reprs = representations[remaining_indices]  # (batch_size, remaining, d_model)
        new_inner_dists = torch.cdist(selected_reprs, last_selected_reprs, p=2) if metric=='euclidean' else -torch.bmm(F.normalize(selected_reprs, p=2, dim=-1), F.normalize(last_selected_reprs, p=2, dim=-1).transpose(1,2))
        new_max_inner_dist = torch.cat([pairwise_dists, new_inner_dists], dim=2).max(dim=1)[0][:,j:]

        # Compute the minimization criterion: (max_query_dist - max_inner_dist)
        scores = new_max_query_dist - new_max_inner_dist*coef  # (batch_size, remaining)
        # Select the best candidate that minimizes this score
        scores_indices=scores.argmin(dim=1) if reverse==False else scores.argmax(dim=1) 
        best_candidates = remaining_indices[torch.arange(batch_size), scores_indices]
        temp_indices = (sorted_indices == best_candidates.unsqueeze(1)).nonzero(as_tuple=True)[1]
        selected_indices[:, j] = best_candidates
        selected_mask[torch.arange(batch_size), temp_indices]=True
    return selected_indices  # Shape (batch_size, icl_num)

def compute_accuracy(probs, acc_list, answer_index):
    ''' 
    compute the accuracy of the model based on the predicted probabilities and the correct answer index

    parameters:
    probs: the predicted probabilities of the model
    acc_list: the list to store whether the model's prediction is correct for a range of prompts
    answer_index: index of the correct answer in the model's vocabulary
    '''
    if probs.argmax(dim=-1)==answer_index:
        acc_list.append(1)
    else:
        acc_list.append(0)
        
def prepare_4d_mask(
  attention_mask: torch.Tensor,
  dtype: torch.dtype,
  device: torch.device,
  cache_position: torch.Tensor,
  batch_size: int,
):

  if attention_mask is not None and attention_mask.dim() == 4:
      # In this case we assume that the mask comes already in inverted form and requires no inversion or slicing.
      causal_mask = attention_mask
  else:
      sequence_length=attention_mask.shape[1]
      target_length=sequence_length
      min_dtype = torch.finfo(dtype).min
      causal_mask = torch.full(
          (sequence_length, target_length), fill_value=min_dtype, dtype=dtype, device=device
      )
      if sequence_length != 1:
          causal_mask = torch.triu(causal_mask, diagonal=1)
      causal_mask *= torch.arange(target_length, device=device) > cache_position.reshape(-1, 1)
      causal_mask = causal_mask[None, None, :, :].expand(batch_size, 1, -1, -1)
      if attention_mask is not None:
          causal_mask = causal_mask.clone()  # copy to contiguous memory for in-place edit
          mask_length = attention_mask.shape[-1]
          padding_mask = causal_mask[:, :, :, :mask_length] + attention_mask[:, None, None, :]
          padding_mask = padding_mask == 0
          causal_mask[:, :, :, :mask_length] = causal_mask[:, :, :, :mask_length].masked_fill(
              padding_mask, min_dtype
          )

  return causal_mask




def compute_probs_and_accuracy(batch, model, tokenizer, mapping, answer_key, offset, acc_list):
    """Compute probabilities and update accuracy."""
    output = model(**batch, output_hidden_states=True)
    probs = output.logits[0][-1].softmax(dim=-1)
    answer_index = tokenizer(mapping[answer_key], return_tensors='pt')['input_ids'][0][offset]
    compute_accuracy(probs, acc_list, answer_index)
    return output, probs, answer_index

def forward_through_layers(
    hiddens, layers, num_layer, model,
    attention_mask, position_ids, causal_mask,
    llm_name, coef, pos_to_intervene, dtype
):
    """
    Memory-efficient forward pass through remaining layers.
    """

    x = hiddens[layers[0]].unsqueeze(0)  # (1, seq, d)
    

    for l in range(layers[0], num_layer):

        if llm_name == 'opt':
            block = model.model.decoder.layers[l]
        elif 'gpt' in llm_name:
            block = model.transformer.h[l]
        else:
            block = model.model.layers[l]

        if l in layers:
            i = layers.index(l)
            coef_i = coef[i].to(dtype)
            if coef_i.dim() == 2 and coef_i.shape[0] == 1:
                coef_i = coef_i.squeeze(0)
            x = x.clone()  # needed for autograd safety
            x[0, pos_to_intervene] += coef_i

        if 'gpt' in llm_name:
            x = block(x, attention_mask=attention_mask)[0]
        else:
            pos_emb = model.model.rotary_emb(
                hiddens[0].unsqueeze(0), position_ids
            )
            x = block(
                x,
                position_ids=position_ids,
                attention_mask=causal_mask,
                position_embeddings=pos_emb
            )[0]

    return x




def evaluate_set(
    data_set, model, tokenizer, mapping, dtype, num_layer, num_heads,
    llm_name, offset, layers, pos_to_intervene,
    acc_list, acc_mod_list, coef, final_ln,
    optimizer=None, lr_scheduler=None, training=False
):
    """Evaluate or train on a dataset (train or test)."""

    for k in tqdm(range(len(data_set))):

        batch = tokenizer(data_set['prompt'][k], return_tensors='pt')
        batch = {k_: v.to(model.device) for k_, v in batch.items()}

        input_ids = batch["input_ids"]
        batch_size, seq_length = input_ids.shape

        # ---- Masks (avoid unnecessary copies) ----
        position_ids = torch.arange(
            seq_length, device=model.device
        ).unsqueeze(0)

        attention_mask = batch["attention_mask"].to(model.device)
        attention_mask = (attention_mask != 0).to(dtype)

        causal_mask = prepare_4d_mask(
            attention_mask,
            dtype,
            model.device,
            cache_position=position_ids.squeeze(0),
            batch_size=1
        )

        # ---- Forward pass ----
        output, probs, answer_index = compute_probs_and_accuracy(
            batch, model, tokenizer, mapping,
            data_set['answer'][k], offset, acc_list
        )

        # ---- Stack hidden states efficiently ----
        hiddens = torch.stack(output.hidden_states, dim=0).squeeze(1)

        # free output ASAP
        del output

        # ---- Forward through layers ----
        x = forward_through_layers(
            hiddens, layers, num_layer, model,
            attention_mask, position_ids, causal_mask,
            llm_name, coef, pos_to_intervene,
            dtype=model.dtype
        )

        # free hiddens early
        del hiddens

        x = x.squeeze(0)

        # ---- Compute logits ----
        last_hidden = final_ln(x[-1])
        del x

        logits = model.lm_head(last_hidden)

        compute_accuracy(logits, acc_mod_list, answer_index)

        # ---- Loss ----
        log_probs = F.log_softmax(logits, dim=0)
        loss = -log_probs[answer_index]
        del logits, log_probs

        # ---- Training ----
        if training:
            loss.backward()
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad(set_to_none=True)

        # free loss
        del loss

        torch.cuda.empty_cache()
        
def evaluate_set_ltv(
    data_set,
    model,
    tokenizer,
    mapping,
    dtype,
    num_layer,
    num_heads,
    llm_name,
    offset,
    layers,
    pos_to_intervene,
    acc_list,
    acc_mod_list,
    coef,
    final_ln,
    optimizer=None,
    lr_scheduler=None,
    training=False,
    noflops=False,
    subsample=100
):
    from utils.hidden_states import track_section
    prof_stats = {}

    d = model.config.hidden_size
    ts = track_section_noflops if noflops else track_section

    subsample=min(len(data_set), subsample)
    for k in tqdm(range(subsample)):
        batch = tokenizer(
            data_set["prompt"][k],
            return_tensors='pt'
        ).to(model.device)

        input_ids = batch["input_ids"]
        L = input_ids.shape[1]

        last_pos = pos_to_intervene if pos_to_intervene.ndim == 0 else pos_to_intervene[-1]
        if input_ids.shape[1] <= last_pos:
            continue

        if pos_to_intervene.ndim > 0:
            tmp_pos = torch.arange(-L, 0, device=model.device) if len(pos_to_intervene) > L else pos_to_intervene
        else:
            tmp_pos = pos_to_intervene

        batch_size, seq_length = input_ids.shape

        position_ids = torch.arange(
            seq_length,
            device=model.device
        ).unsqueeze(0)

        attention_mask = (batch["attention_mask"] != 0).to(model.dtype)

        causal_mask = prepare_4d_mask(
            attention_mask,
            model.dtype,
            model.device,
            cache_position=position_ids.squeeze(0),
            batch_size=1
        ).to(dtype)

        with torch.no_grad():
            with ts(prof_stats, f"inference_{k}",
                    method="none", L=L, d=d, n_layers=num_layer,
                    device=model.device):

                output, probs, answer_index = compute_probs_and_accuracy(
                    batch, model, tokenizer, mapping,
                    data_set['answer'][k], offset, acc_list
                )

                hiddens = torch.stack(output.hidden_states, dim=0).squeeze(1)
        del output

        with ts(prof_stats, f"inference_ltv_{k}",
                method="ltv", L=L, d=d, n_layers=1,
                device=model.device):

            # slice coef once
            coef_slice = coef[:, :L, :]

            x = forward_through_layers(
                hiddens,
                layers,
                num_layer,
                model,
                attention_mask,
                position_ids,
                causal_mask,
                llm_name,
                coef_slice,
                tmp_pos,
                dtype=model.dtype
            )

            x = x.squeeze(0)

            logits = model.lm_head(final_ln(x[-1]))

        compute_accuracy(logits, acc_mod_list, answer_index)

        log_probs = F.log_softmax(logits, dim=0)
        loss = -log_probs[answer_index]


        if training:
            with ts(prof_stats, f"trainstep_{k}",
                    method="none", L=L, d=d, n_layers=num_layer,
                    device=model.device):

                loss.backward()
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

        del hiddens, x, logits, log_probs, loss, batch, attention_mask, causal_mask

    return prof_stats

def evaluate_set_prefix(
    data_set,
    model,
    tokenizer,
    mapping,
    dtype,
    num_layer,
    num_heads,
    llm_name,
    offset,
    layers,
    pos_to_intervene,
    acc_list,
    acc_mod_list,
    coef,
    final_ln,
    optimizer=None,
    lr_scheduler=None,
    training=False,
    noflops=False,
    subsample=100
):
    ts = track_section_noflops if noflops else track_section
    prof_stats = {}

    target_layer = layers[0]
    prefix_K, prefix_V = coef[0], coef[1]
    m = prefix_K.shape[0]
    d = model.config.hidden_size

    # move prefix once
    prefix_K = prefix_K.to(device=model.device)
    prefix_V = prefix_V.to(device=model.device)

    # get block
    if "gpt" in llm_name:
        block = model.transformer.h[target_layer]
    elif llm_name == "opt":
        block = model.model.decoder.layers[target_layer]
    else:
        block = model.model.layers[target_layer]

    orig_attn_forward = block.self_attn.forward

    # -------- prefix forward --------
    def prefix_attn_forward(self, hidden_states, attention_mask=None, position_ids=None,
                           past_key_value=None, output_attentions=False,
                           use_cache=False, cache_position=None,
                           position_embeddings=None, **kwargs):

        config = self.config
        H_q = config.num_attention_heads
        H_kv = config.num_key_value_heads
        d_head = config.hidden_size // H_q
        groups = H_q // H_kv

        bsz, q_len, _ = hidden_states.size()
        dtype_local = hidden_states.dtype

        Q = self.q_proj(hidden_states).view(bsz, q_len, H_q, d_head).transpose(1, 2)
        K = self.k_proj(hidden_states).view(bsz, q_len, H_kv, d_head).transpose(1, 2)
        V = self.v_proj(hidden_states).view(bsz, q_len, H_kv, d_head).transpose(1, 2)

        pk = prefix_K.to(dtype_local).unsqueeze(0).permute(0, 2, 1, 3).expand(bsz, H_kv, m, d_head)
        pv = prefix_V.to(dtype_local).unsqueeze(0).permute(0, 2, 1, 3).expand(bsz, H_kv, m, d_head)

        K = torch.cat([pk, K], dim=2)
        V = torch.cat([pv, V], dim=2)

        K = K.repeat_interleave(groups, dim=1)
        V = V.repeat_interleave(groups, dim=1)

        if attention_mask is not None:
            B, _, Lq, Lk = attention_mask.shape
            if Lk == q_len:
                prefix_mask = torch.zeros(B, 1, Lq, m, device=attention_mask.device, dtype=attention_mask.dtype)
                attention_mask = torch.cat([prefix_mask, attention_mask], dim=-1)

        out = F.scaled_dot_product_attention(
            Q, K, V,
            attn_mask=attention_mask,
            is_causal=self.is_causal,
            enable_gqa=False
        )
        out = out.transpose(1, 2).contiguous().view(bsz, q_len, -1)
        return self.o_proj(out), None

    subsample = min(len(data_set), subsample)

    for k in tqdm(range(subsample)):

        batch = tokenizer(data_set["prompt"][k], return_tensors="pt").to(model.device)
        input_ids = batch["input_ids"]
        if input_ids.shape[1] <= 1:
            continue

        L = input_ids.shape[1]

        with ts(prof_stats, f"inference_baseline_{k}", method="none", L=L, d=d, n_layers=num_layer, device=model.device):

            block.self_attn.forward = orig_attn_forward

            attention_mask = batch["attention_mask"].to(device=model.device, dtype=dtype)
            position_ids = torch.arange(L, device=model.device).unsqueeze(0)
            causal_mask = prepare_4d_mask(
                attention_mask, dtype, model.device,
                cache_position=position_ids.squeeze(0), batch_size=1
            )

            output, probs, answer_index = compute_probs_and_accuracy(
                batch, model, tokenizer, mapping,
                data_set["answer"][k], offset, acc_list
            )


            hiddens = torch.stack(output.hidden_states, dim=0).squeeze(1)
            del output.hidden_states  # free list immediately

            h0 = hiddens[0]
            h_target = hiddens[target_layer]
            del hiddens


        with ts(prof_stats, f"inference_prefix_{k}", method="prefix", m=m, L=L, d=d, n_layers=num_layer, device=model.device):

            block.self_attn.forward = prefix_attn_forward.__get__(block.self_attn, type(block.self_attn))

            x = h_target.unsqueeze(0)

            for l in range(target_layer, num_layer):
                if "gpt" in llm_name:
                    cur_block = model.transformer.h[l]
                    x, _ = cur_block(x, attention_mask=attention_mask, output_attentions=True)
                else:
                    cur_block = model.model.layers[l]
                    pos_emb = model.model.rotary_emb(h0.unsqueeze(0), position_ids)
                    x, _ = cur_block(
                        x,
                        position_ids=position_ids,
                        attention_mask=causal_mask,
                        position_embeddings=pos_emb,
                        output_attentions=True
                    )

            x = x.squeeze(0)
            logits = model.lm_head(final_ln(x[-1]))
            compute_accuracy(logits, acc_mod_list, answer_index)
            loss = -F.log_softmax(logits, dim=0)[answer_index]

        # -------- training --------
        if training:
            with ts(prof_stats, f"trainstep_{k}", method="none", L=L, d=d, n_layers=num_layer, device=model.device):
                loss.backward()
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

        # 🔥 aggressive cleanup
        del batch, x, logits, loss, h0, h_target, attention_mask, position_ids, causal_mask
        torch.cuda.empty_cache()

    block.self_attn.forward = orig_attn_forward
    return prof_stats

def evaluate_set_lora(
    data_set,
    model,
    tokenizer,
    mapping,
    dtype,
    num_layer,
    llm_name,
    offset,
    layers,
    acc_list,
    acc_mod_list,
    lora_A,
    lora_B,
    final_ln,
    optimizer=None,
    lr_scheduler=None,
    training=False,
    noflops=False,
    subsample=100
):
    prof_stats = {}
    ts = track_section_noflops if noflops else track_section

    target_layer = layers[0]
    d_model = model.config.hidden_size
    
    if "gpt" in llm_name:
        block = model.transformer.h[target_layer]
    elif llm_name == "opt":
        block = model.model.decoder.layers[target_layer]
    else:
        block = model.model.layers[target_layer]

    o_proj = block.self_attn.o_proj
    orig_o_proj_forward = o_proj.forward


    def lora_o_proj_forward(self, hidden_states, *args, **kwargs):
        orig_shape = hidden_states.shape
        device = hidden_states.device
        dtype_local = hidden_states.dtype

        base_out = orig_o_proj_forward(hidden_states, *args, **kwargs)

        A = lora_A.to(device=device, dtype=dtype_local)
        B = lora_B.to(device=device, dtype=dtype_local)

        x = hidden_states.view(-1, orig_shape[-1])  
        lora_out = x @ A @ B                         
        lora_out = lora_out.view(orig_shape)

        return base_out + lora_out

    subsample = min(len(data_set), subsample)

    for k in tqdm(range(subsample)):

        batch = tokenizer(data_set["prompt"][k], return_tensors="pt").to(model.device)
        input_ids = batch["input_ids"]

        if input_ids.shape[1] <= 1:
            continue

        L = input_ids.shape[1]


        with ts(prof_stats, f"inference_baseline_{k}", method="none",
                L=L, d=d_model, n_layers=num_layer, device=model.device):

            o_proj.forward = orig_o_proj_forward

            attention_mask = batch["attention_mask"].to(device=model.device, dtype=dtype)
            position_ids = torch.arange(L, device=model.device).unsqueeze(0)

            causal_mask = prepare_4d_mask(
                attention_mask, dtype, model.device,
                cache_position=position_ids.squeeze(0), batch_size=1
            )

            output, probs, answer_index = compute_probs_and_accuracy(
                batch, model, tokenizer, mapping,
                data_set["answer"][k], offset, acc_list
            )


            hiddens = torch.stack(output.hidden_states, dim=0).squeeze(1)
            del output.hidden_states


            h0 = hiddens[0]
            h_target = hiddens[target_layer]
            del hiddens


        with ts(prof_stats, f"inference_lora_{k}", method="lora",
                L=L, d=d_model, n_layers=num_layer, device=model.device):

            o_proj.forward = lora_o_proj_forward.__get__(o_proj, type(o_proj))

            x = h_target.unsqueeze(0)

            for l in range(target_layer, num_layer):
                if "gpt" in llm_name:
                    cur_block = model.transformer.h[l]
                    x, _ = cur_block(
                        x,
                        attention_mask=batch["attention_mask"].to(device=model.device, dtype=dtype),
                        output_attentions=True
                    )
                else:
                    cur_block = model.model.layers[l]

                    pos_emb = model.model.rotary_emb(h0.unsqueeze(0), position_ids)

                    x, _ = cur_block(
                        x,
                        position_ids=position_ids,
                        attention_mask=causal_mask,
                        position_embeddings=pos_emb,
                        output_attentions=True
                    )

            x = x.squeeze(0)
            logits = model.lm_head(final_ln(x[-1]))

            compute_accuracy(logits, acc_mod_list, answer_index)
            loss = -F.log_softmax(logits, dim=0)[answer_index]


        if training:
            with ts(prof_stats, f"trainstep_{k}", method="none",
                    L=L, d=d_model, n_layers=num_layer, device=model.device):
                loss.backward()
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

        del batch, x, logits, loss, h0, h_target, attention_mask, position_ids, causal_mask
        torch.cuda.empty_cache()

    o_proj.forward = orig_o_proj_forward
    return prof_stats



import math

def scaled_dot_product_attention(
    query, key, value,
    masked_attn_heads=None,
    attn_mask=None,
    dropout_p=0.0,
    is_causal=False,
    scale=None,
    enable_gqa=False
):
    L, S = query.size(-2), key.size(-2)
    dtype = query.dtype
    device = query.device

    # Cast prefix (kept in fp32) to correct dtype
    key   = key.to(dtype)
    value = value.to(dtype)

    scale_factor = 1 / math.sqrt(query.size(-1)) if scale is None else scale

    # Start with zero-bias
    attn_bias = torch.zeros(L, S, dtype=dtype, device=device)

    # -------------------------------------------------------
    # 1. causal mask (ALWAYS allowed, even if attn_mask exists)
    # -------------------------------------------------------
    if is_causal:
        temp_mask = torch.ones(L, S, dtype=torch.bool, device=device).tril(0)
        attn_bias = attn_bias.masked_fill(~temp_mask, float("-inf"))

    # -------------------------------------------------------
    # 2. Apply external attention mask (padding mask, prefix mask)
    # -------------------------------------------------------
    if attn_mask is not None:
        if attn_mask.dtype == torch.bool:
            attn_bias = attn_bias.masked_fill(~attn_mask, float("-inf"))
        else:
            attn_bias = attn_bias + attn_mask

    # -------------------------------------------------------
    # 3. Expand KV if GQA is enabled
    # -------------------------------------------------------
    if enable_gqa:
        key   = key.repeat_interleave(query.size(-3) // key.size(-3), dim= -3)
        value = value.repeat_interleave(query.size(-3) // value.size(-3), dim= -3)

    # -------------------------------------------------------
    # 4. SDPA: QK^T + masks → softmax → V
    # -------------------------------------------------------
    attn_weight = (query @ key.transpose(-2, -1)) * scale_factor
    attn_weight = attn_weight + attn_bias
    attn_weight = torch.softmax(attn_weight, dim=-1)

    if masked_attn_heads is not None:
        for i in range(len(masked_attn_heads)):
            attn_weight[:, masked_attn_heads[i,1], :, :] = 0

    attn_weight = torch.dropout(attn_weight, dropout_p, train=True)
    return attn_weight @ value


def ablate_PTH(model, batch,heads_list,topk,num_layer, layers,final_ln,device):
    '''
    compute the hidden states of the delimiter tokens of the icl demonstrations with different labels and also the query
s
    parameters:
    model: the model to use to compute the hidden states
    batch: the batch of icl demonstrations to get the hidden states from
    heads_list: the list of attention heads to ablate while computing the hidden states
    topk: the number of attention heads to ablate
    num_layer: the number of layers in the model

    output:
    hiddens: hidden states to be used in subsequent interpolations
            shape = number of labels * (icl_num+1) * d_model
    '''

    with torch.no_grad():
        N,T=batch['input_ids'].shape
        H=model.config.hidden_size
        hiddens = torch.stack(model(**batch, output_hidden_states=True).hidden_states,dim=0)
        if topk!=0:
            heads_list_filtered=torch.tensor(heads_list)[:topk]
            hiddens=hiddens[0]
            x=hiddens
            for j in range(num_layer):
                block=layers[j]
                a=block.input_layernorm(x) if 'gpt' not in model.name_or_path else block.ln_1(x)
                heads=heads_list_filtered[heads_list_filtered[:,0]==j]
                if 'gpt' in model.name_or_path:
                    query, key, value,_ = block.attention._attn_projections_and_rope(a,position_ids=torch.arange(T).unsqueeze(dim=0))
                    for k in range(heads.shape[0]):
                        value[:,heads[k,1]]=0
                    mask = torch.tril(torch.ones(T, T)).unsqueeze(0).unsqueeze(0).to(device)
                    mask = mask.masked_fill(mask == 0, float('-inf'))
                    attn_outputs = block.attention._attn(query, key, value, attention_mask=mask)
                    a = attn_outputs[0]
                    a = utils._merge_heads(a)
                    a = block.attention.dense(a)
                    b=block.post_attention_layernorm(x) if 'gpt' not in model.name_or_path else block.ln_2(x)
                    b = block.mlp(b)
                    x=x+a+b
                else:
                    Hd=model.config.head_dim
                    query= block.self_attn.q_proj(a).view(N, T, -1, Hd).transpose(1, 2)
                    key= block.self_attn.k_proj(a).view(N, T, -1, Hd).transpose(1, 2)
                    value= block.self_attn.v_proj(a).view(N, T, -1, Hd).transpose(1, 2)

                    # if 'gemma-2b' in model.name_or_path:
                    #     for i in range(len(heads)):
                    #         value[:,:,:,heads[i,1]*Hdkv:(heads[i,1]+1)*Hdkv]=0
                    # else:
                    #     for i in range(len(heads)):
                    #         # value[:,heads[i,1]]=0
                    #         print(heads[i,1])
                    #         print(value.shape)
                    #         value[:, heads[i,1]] = 0
                    position_ids = torch.arange(a.shape[1], dtype=model.dtype, device=model.device).unsqueeze(0)
                    causal_mask = prepare_4d_mask(batch['attention_mask'], model.dtype, model.device,cache_position=position_ids.squeeze(0), batch_size=N).to(model.dtype)
                    pos_emb=model.model.rotary_emb(hiddens[0],position_ids)
                    cos, sin = pos_emb
                    query, key = apply_rotary_pos_emb(query, key, cos, sin)
                    attn_outputs = scaled_dot_product_attention(
                        query, key, value, attn_mask=causal_mask, dropout_p=0.0, enable_gqa=True, masked_attn_heads=heads, 
                    )
                    
                    attn_outputs= attn_outputs.transpose(1, 2).contiguous().view(N, T, -1)
                    a= block.self_attn.o_proj(attn_outputs)
                    x=x+a
                    b=block.post_attention_layernorm(x) if 'gpt' not in model.name_or_path else block.ln_2(x)
                    b = block.mlp(b)
                    x=x+b
                #if j == num_layer-1:
                    #x=model.gpt_neox.final_layer_norm(x)
                hiddens=torch.cat([hiddens,x],dim=0)
            hiddens=hiddens.view(-1,N,T,H)
        else:
            attention_mask=batch['attention_mask']
            position_ids = torch.arange(attention_mask.shape[-1], dtype=model.dtype, device=model.device).unsqueeze(0)
            causal_mask = prepare_4d_mask(attention_mask, model.dtype, model.device,cache_position=position_ids.squeeze(0), batch_size=N).to(model.dtype)
            pos_emb=model.model.rotary_emb(hiddens[0].unsqueeze(0),position_ids)
            hiddens[-1]=layers[-1](hiddens[-2], position_ids=position_ids, attention_mask=causal_mask, position_embeddings=pos_emb)[0]
        return hiddens




      

def apply_rotary_pos_emb(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
    """Applies Rotary Position Embedding to the query and key tensors.

    Args:
        q (`torch.Tensor`): The query tensor.
        k (`torch.Tensor`): The key tensor.
        cos (`torch.Tensor`): The cosine part of the rotary embedding.
        sin (`torch.Tensor`): The sine part of the rotary embedding.
        position_ids (`torch.Tensor`, *optional*):
            Deprecated and unused.
        unsqueeze_dim (`int`, *optional*, defaults to 1):
            The 'unsqueeze_dim' argument specifies the dimension along which to unsqueeze cos[position_ids] and
            sin[position_ids] so that they can be properly broadcasted to the dimensions of q and k. For example, note
            that cos[position_ids] and sin[position_ids] have the shape [batch_size, seq_len, head_dim]. Then, if q and
            k have the shape [batch_size, heads, seq_len, head_dim], then setting unsqueeze_dim=1 makes
            cos[position_ids] and sin[position_ids] broadcastable to the shapes of q and k. Similarly, if q and k have
            the shape [batch_size, seq_len, heads, head_dim], then set unsqueeze_dim=2.
    Returns:
        `tuple(torch.Tensor)` comprising of the query and key tensors rotated using the Rotary Position Embedding.
    """
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed
def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def get_dataset_specific_IH(model, tokenizer, num_samp, prompts, answers, s, df, icl_num, instruction, prompt_format, delimiter, separator, natural_mapping,random_mapping, indices, offset, ratio, t):
    """
    get_dataset_specifc_IH calculates IH scores for a dataset-specific criterion and ranks the attention heads.
    Args:
        model: the model to get IHs from
        tokenizer: the tokenizer to encode the input sequences
        num_samp: the number of samples to use to calculate the attention scores
        topk: the number of top-scoring IHs to return
        prompts: the dataset-specific prompts (final queries) to sample from
        answers: the answers to the prompts
        IH_criterion: the criterion with which to calculate IH scores and rank the attention heads
         specifically: dataset_specific: choose the IHs that distribute most attention scores to the labels tokens at the final delimiter positions (for the prompts from the dataset)
                       highest_acc choose the IHs that distribute greatest proportion of attention scores (aggregated across many samples) to the correct label tokens at the final delimiter positions (for the prompts from the dataset)
                       highest_avg: choose the IHs that distribute greatest proportion of attention scores (averaged across many samples) to the correct label tokens at the final delimiter positions (for the prompts from the dataset)
    Returns:
        head_list: a list of [layer, head] pairs ranked by IH scores 
        sorted_scores: the corresponding IH scores
    """
    num_layer = model.config.num_hidden_layers
    num_heads = model.config.num_attention_heads
    induction_score=torch.zeros(num_layer,num_heads).to(model.device)
    for i in tqdm(range(num_samp)):
        icl, positions, labels,_ = icl_random_label(df, tokenizer, icl_num, 1, instruction, prompt_format, delimiter, separator, natural_mapping, random_mapping, indices=indices, offset=offset, ratio=ratio)
        positions=[i+1 for i in positions]
        prompt=icl+prompts[i]
        batch=tokenizer(prompt,return_tensors='pt').to(model.device)
        with torch.no_grad():
            attn=torch.stack(model(**batch,output_attentions=True).attentions,dim=0).squeeze(dim=1).to(model.device)
            #Or consider the false induction heads defined in Steinhardt's paper
            #same_pos=[positions[j] for j in range(len(positions)) if labels[j] == answers[i]]
            #diff_pos=[j for j in positions if j not in same_pos]
            #score=attn[:,:,-1, same_pos].sum(dim=-1)-attn[:,:,-1, diff_pos]
            score=attn[:,:,-1,positions].sum(dim=-1)
            induction_score=induction_score+score


    val,ind=torch.topk(induction_score,k=t,dim=-1)
    IH_list=torch.zeros(num_layer*t,2)
    for i in range(num_layer):
        IH_list[i*t:(i+1)*t,0]=i
        for j in range(t):
            IH_list[i*t+j,1]=ind[i,j]
        
    return torch.tensor(IH_list),ind

        


 
