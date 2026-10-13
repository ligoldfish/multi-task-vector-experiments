"""One explicit entry point for preparation, independent jobs and frozen evaluation."""
from __future__ import annotations
import argparse
from pathlib import Path
import sys
import yaml
from .config import load_config
from .io import atomic_json


def parser():
    root = argparse.ArgumentParser(prog="task-vector", description="Document-faithful LTV / LoDA-MT experiment suite")
    sub = root.add_subparsers(dest="command", required=True)
    for name in ("preflight", "assets", "prepare", "inventory", "smoke", "train", "statistics", "run", "report", "package"):
        item = sub.add_parser(name)
        item.add_argument("--config", default="configs/experiment.yaml")
        item.add_argument("--device", help="Runtime placement only, e.g. npu:3")
        item.add_argument("--set", action="append", default=[], metavar="KEY=YAML", help="Explicit recorded dotted configuration override")
        if name in {"train", "statistics", "run", "prepare"}:
            item.add_argument("--scope", default="all" if name == "prepare" else "mvp", help="mvp, expanded, loo-TASK; prepare also accepts all")
        if name == "train":
            item.add_argument("--route", choices=["ltv", "loda"], required=True)
            item.add_argument("--seed", type=int, required=True)
            item.add_argument("--task", required=True)
            item.add_argument("--layers", type=int, nargs="+", default=None)
        if name == "smoke":
            item.add_argument("--task", default="math")
        if name == "run":
            item.add_argument("--phase", choices=["core", "discover", "lock", "final", "heldout", "ablations", "extended", "formal", "all"], required=True)
            item.add_argument("--split", choices=["test", "test_formal", "test_confirmation"], default="test")
        if name == "assets":
            item.add_argument("--hash-weights", action="store_true")
        if name == "package":
            item.add_argument("--output", default="dist/task-vector-modelmate.zip")
    return root


def main(argv=None):
    args = parser().parse_args(argv)
    overrides = {}
    for expression in args.set:
        if "=" not in expression:
            raise ValueError("--set requires dotted.key=YAML")
        name, raw = expression.split("=", 1)
        overrides[name] = yaml.safe_load(raw)
    if args.device:
        overrides["platform.device"] = args.device
    config = load_config(args.config, overrides)
    source_root = Path(__file__).resolve().parents[2]
    root = Path(config["platform"]["persistent_root"])
    config.setdefault("runtime", {})
    config["runtime"].setdefault("data_root", str(root / "data"))
    config["runtime"].setdefault("output_root", str(root / "runs"))
    if args.command == "package":
        from .packaging import create_package
        result = create_package(source_root, args.output)
    elif args.command == "inventory":
        from .suite import experiment_inventory
        result = experiment_inventory(config)
        atomic_json(Path(config["runtime"]["output_root"]) / "experiment_inventory.json", result)
    elif args.command == "preflight":
        from .runtime import preflight
        result = preflight(config, config["runtime"]["output_root"])
    elif args.command == "assets":
        from .assets import inspect_assets
        result = inspect_assets(config, output_path=Path(config["runtime"]["output_root"]) / "assets.json", hash_weights=args.hash_weights)
    elif args.command == "prepare":
        from transformers import AutoTokenizer
        from .data import prepare_dataset
        if (Path(config["runtime"]["data_root"]) / "manifest.json").exists():
            raise ValueError("Prepared data already has a manifest; use a new persistent root instead of overwriting split/lock evidence")
        if not config["model"].get("path"):
            raise ValueError("Set model.path or TV_MODEL_PATH to the local Qwen3-1.7B snapshot")
        tokenizer = AutoTokenizer.from_pretrained(config["model"]["path"], local_files_only=True)
        tasks = list(config["data"]["tasks"])
        if args.scope in {"all", "expanded"}:
            tasks += config["data"]["extension_tasks"]
        if args.scope in {"all", "expanded", "mvp"}:
            tasks += config["data"]["auxiliary_tasks"]
        if args.scope not in {"all", "expanded", "mvp", "core"}:
            raise ValueError("Prepare scope must be all, expanded, mvp or core; heldout folds reuse grouped source files")
        config["data"]["tasks"] = tasks
        result = prepare_dataset(config, tokenizer, config["runtime"]["data_root"])
    elif args.command == "smoke":
        from .smoke import run_smoke
        result = run_smoke(config, task=args.task, output_dir=config["runtime"]["output_root"])
    elif args.command == "report":
        from .reporting import build_report
        result = build_report(config["runtime"]["output_root"])
    else:
        from .runner import ExperimentRunner
        runner = ExperimentRunner(config, source_root=source_root)
        try:
            if args.command == "train":
                result = runner.train(args.scope, args.route, args.seed, args.task, args.layers)
            elif args.command == "statistics":
                result = {task: getattr(stat, "metadata", {}) for task, stat in runner.statistics(args.scope).items()}
            elif args.command == "run":
                from .workflow import run_phase
                result = run_phase(runner, args.phase, args.scope, args.split)
            else:
                raise ValueError(args.command)
        finally:
            runner.release_model()
    # Avoid flooding the terminal with per-example scores or full decision trees.
    import json
    print(json.dumps({"command": args.command, "result": _compact(result)}, ensure_ascii=False, indent=2, default=str))
    return 42 if isinstance(result, dict) and (result.get("completed") is False or result.get("passed") is False or result.get("passes") is False) else 0


def _compact(result):
    if not isinstance(result, dict):
        return {"items": len(result)} if isinstance(result, list) else result
    keys = {"completed", "passes", "passed", "stop_reason", "scope", "split", "status", "path", "sha256", "raw_path", "post_path", "steps", "optimizer_steps", "best_nll", "device", "device_count", "model_status", "lock_sha256"}
    compact = {k: v for k, v in result.items() if k in keys}
    return compact or {"artifact_sections": list(result)}


if __name__ == "__main__":
    sys.exit(main())
