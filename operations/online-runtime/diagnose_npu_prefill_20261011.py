"""Train-only, forward-only localization of failed teacher/prefill agreement.

Read the failed smoke and its saved LTV. Never change source, configuration,
data, parameters, tolerances, smoke records, or scientific training artifacts.
"""
from __future__ import annotations

import argparse
from contextlib import nullcontext
from datetime import datetime, timezone
import hashlib
import inspect
import json
import os
from pathlib import Path
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", required=True)
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    project = Path(args.project_root).resolve()
    sys.path.insert(0, str(project / "src"))

    import torch
    from task_vector.config import load_config, scientific_config, source_manifest
    from task_vector.io import atomic_json, object_sha256
    from task_vector.losses import collate_encoded
    from task_vector.ltv import intervention_context
    from task_vector.prompts import encode_sample
    from task_vector.representations import Representation
    from task_vector.runner import ExperimentRunner

    config = load_config(args.config)
    root = Path(config["platform"]["persistent_root"])
    config.setdefault("runtime", {})
    config["runtime"].setdefault("data_root", str(root / "data"))
    config["runtime"].setdefault("output_root", str(root / "runs"))
    runner = ExperimentRunner(config, source_root=project)
    smoke_path = runner.output_root / "smoke.json"
    previous_bytes = smoke_path.read_bytes()
    previous = json.loads(previous_bytes)
    if previous.get("passed") is not False or previous.get("task") != "math":
        raise ValueError("Expected the failed math smoke; this tool does not replace a passing record")
    for field, value in (
        ("config_sha256", object_sha256(scientific_config(config))),
        ("data_manifest_sha256", object_sha256(runner.manifest)),
        ("source_manifest_sha256", object_sha256(source_manifest(project))),
        ("base_fingerprint", runner.base_fingerprint()),
    ):
        if previous.get(field) != value:
            raise ValueError(f"Failed-smoke provenance differs: {field}")
    saved_path = runner.output_root / "smoke_ltv.pt"
    representation = Representation.load(saved_path)
    selected = runner.samples("math", "train")[:2]
    if [sample.id for sample in selected] != previous.get("sample_ids"):
        raise ValueError("The two training samples differ from the failed smoke")
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    evidence = root / "environment" / f"prefill-diagnostic-{stamp}-{os.getpid()}"
    evidence.mkdir(parents=True, exist_ok=False)
    report_path = evidence / "diagnostic.json"
    report = {"kind": "engineering_forward_diagnostic", "scientific_training": False,
              "formal_evaluation": False, "sample_split": "train", "sample_ids": previous["sample_ids"],
              "previous_smoke_sha256": hashlib.sha256(previous_bytes).hexdigest(),
              "saved_ltv_sha256": hashlib.sha256(saved_path.read_bytes()).hexdigest(),
              "unchanged_atol": previous["atol"], "unchanged_rtol": previous["rtol"],
              "previous_checks": previous.get("checks"), "comparisons": [], "status": "running"}
    atomic_json(report_path, report)
    print(f"DIAGNOSTIC_JSON={report_path}", flush=True)
    print("STAGE=load_original_frozen_model", flush=True)
    try:
        model, tokenizer = runner.load_model()
        model.eval()
        device = runner.device
        encoded = [encode_sample(tokenizer, sample,
                    max_prompt=config["training"]["max_prompt_tokens"],
                    max_target=config["training"]["max_target_tokens"]) for sample in selected]
        batch = collate_encoded(encoded, tokenizer.pad_token_id, device)
        module_path = Path(inspect.getfile(type(model)))
        metadata = {"device": str(device), "dtype": str(next(model.parameters()).dtype),
                    "torch": torch.__version__, "attention_implementation": model.config._attn_implementation,
                    "model_source": str(module_path), "model_source_sha256": hashlib.sha256(module_path.read_bytes()).hexdigest(),
                    "rope_scaling": getattr(model.config, "rope_scaling", None),
                    "use_sliding_window": getattr(model.config, "use_sliding_window", None),
                    "sliding_window": getattr(model.config, "sliding_window", None),
                    "attention_dropout": getattr(model.config, "attention_dropout", None),
                    "samples": [{"id": item["id"], "prompt_length": item["prompt_length"],
                                 "full_length": len(item["input_ids"])} for item in encoded]}
        for label, components in (
            ("allow_internal_format", ("npu", "config", "allow_internal_format")),
            ("matmul_allow_hf32", ("npu", "matmul", "allow_hf32")),
            ("conv_allow_hf32", ("npu", "conv", "allow_hf32")),
        ):
            try:
                value = torch
                for component in components:
                    value = getattr(value, component)
                metadata[label] = bool(value)
            except (AttributeError, RuntimeError) as exc:
                metadata[label] = f"unavailable: {type(exc).__name__}"
        report["runtime"] = metadata
        print("RUNTIME=" + json.dumps(metadata, ensure_ascii=False), flush=True)

        def measure(ids, mask, positions, rep, *, keep=0):
            trace, handles = {}, []
            context = intervention_context(model, rep, positions) if rep is not None else nullcontext()

            def capture(tensor, label):
                rows = torch.arange(tensor.shape[0], device=tensor.device)
                trace[label] = tensor[rows, positions].detach().float().cpu()

            def input_hook(label):
                def hook(module, args, kwargs):
                    capture(args[0] if args else kwargs["hidden_states"], label)
                return hook

            with torch.no_grad(), context:
                try:
                    for index, layer in enumerate(model.model.layers):
                        handles.append(layer.register_forward_pre_hook(input_hook(f"layer_{index}_input"), with_kwargs=True))
                    handles.append(model.model.norm.register_forward_hook(lambda module, args, output: capture(output, "final_norm")))
                    kwargs = {"input_ids": ids, "use_cache": False, "logits_to_keep": keep}
                    if mask is not None:
                        kwargs["attention_mask"] = mask
                    output = model(**kwargs)
                    rows = torch.arange(ids.shape[0], device=device)
                    logit_positions = torch.zeros_like(positions) if keep == 1 else positions
                    logits = output.logits[rows, logit_positions].detach().float().cpu()
                finally:
                    for handle in handles:
                        handle.remove()
            return {"logits": logits, "trace": trace}

        atol, rtol = float(previous["atol"]), float(previous["rtol"])

        def compare(condition, sample_index, name, left, right, *, exact=False):
            x, y = left["logits"][0], right["logits"][0]
            delta = (x - y).abs()
            outside = delta > (0 if exact else atol + rtol * y.abs())
            finite = bool(torch.isfinite(x).all() and torch.isfinite(y).all())
            top_x, top_y = x.topk(2), y.topk(2)
            layers = []
            for label in left["trace"]:
                a, b = left["trace"][label][0], right["trace"][label][0]
                diff = (a - b).abs()
                layers.append({"layer": label, "max_abs": float(diff.max()),
                               "outside_original_tolerance": bool((diff > atol + rtol * b.abs()).any())})
            row = {"condition": condition, "sample_id": encoded[sample_index]["id"], "comparison": name,
                   "finite": finite, "exact_comparison": exact, "max_abs": float(delta.max()),
                   "rmse": float((delta.square().mean()).sqrt()),
                   "mismatch_percent": float(outside.float().mean() * 100),
                   "within_original_tolerance": finite and not bool((delta > atol + rtol * y.abs()).any()),
                   "same_argmax": int(x.argmax()) == int(y.argmax()),
                   "top2_left": top_x.indices.tolist(), "top2_right": top_y.indices.tolist(),
                   "margin_left": float(top_x.values[0] - top_x.values[1]),
                   "margin_right": float(top_y.values[0] - top_y.values[1]),
                   "first_nonzero_hidden_layer": next((p["layer"] for p in layers if p["max_abs"] != 0), None),
                   "first_hidden_layer_outside_tolerance": next((p["layer"] for p in layers if p["outside_original_tolerance"]), None),
                   "final_hidden_max_abs": layers[-1]["max_abs"], "hidden_layers": layers}
            report["comparisons"].append(row)
            atomic_json(report_path, report)
            print("COMPARE=" + json.dumps({k: v for k, v in row.items() if k != "hidden_layers"}, ensure_ascii=False), flush=True)

        def one(result, index):
            return {"logits": result["logits"][index:index + 1],
                    "trace": {k: v[index:index + 1] for k, v in result["trace"].items()}}

        for condition, rep in (("BASE", None), ("SAVED_LTV", representation)):
            print(f"STAGE=forward_matrix_{condition}", flush=True)
            a = measure(batch["input_ids"], batch["attention_mask"], batch["prompt_positions"], rep)
            repeated = measure(batch["input_ids"], batch["attention_mask"], batch["prompt_positions"], rep)
            changed = batch["input_ids"].clone()
            for index, example in enumerate(encoded):
                p, n = example["prompt_length"], len(example["input_ids"])
                changed[index, p:n] = (changed[index, p:n] + 17) % model.config.vocab_size
            future = measure(changed, batch["attention_mask"], batch["prompt_positions"], rep)
            for index, example in enumerate(encoded):
                compare(condition, index, "same_shape_repeat", one(a, index), one(repeated, index), exact=True)
                compare(condition, index, "future_tokens_changed_same_shape", one(a, index), one(future, index), exact=True)
                full = torch.tensor([example["input_ids"]], dtype=torch.long, device=device)
                p = example["prompt_length"]
                prefix = full[:, :p]
                pos = torch.tensor([p - 1], dtype=torch.long, device=device)
                b = measure(full, torch.ones_like(full), pos, rep)
                c = measure(prefix, torch.ones_like(prefix), pos, rep)
                d = measure(prefix, None, pos, rep)
                e = measure(prefix, None, pos, rep, keep=1)
                compare(condition, index, "batch_full_vs_single_full", one(a, index), b)
                compare(condition, index, "single_full_vs_prefix_with_mask", b, c)
                compare(condition, index, "prefix_with_mask_vs_without_mask", c, d)
                compare(condition, index, "prefix_all_logits_vs_last_logit", d, e)
                compare(condition, index, "original_teacher_vs_prefill", one(a, index), e)
        if smoke_path.read_bytes() != previous_bytes:
            raise RuntimeError("Original smoke record changed during diagnosis")
        if any(parameter.requires_grad or parameter.grad is not None for parameter in model.parameters()):
            raise RuntimeError("Frozen-base ownership changed during diagnosis")
        report.update(status="complete", original_smoke_unchanged=True, base_frozen_no_gradient=True)
        atomic_json(report_path, report)
        print("FORWARD_DIAGNOSIS=COMPLETE (not a smoke pass)", flush=True)
        print(f"DIAGNOSTIC_JSON={report_path}", flush=True)
    except BaseException as exc:
        report.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        atomic_json(report_path, report)
        raise
    finally:
        runner.release_model()


if __name__ == "__main__":
    main()
