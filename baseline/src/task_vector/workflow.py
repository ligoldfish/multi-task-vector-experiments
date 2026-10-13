"""Ordered scientific stages, independent of the single-card job executor."""
from __future__ import annotations
from copy import deepcopy
from pathlib import Path
import json
import numpy as np
from .io import atomic_json, read_json, read_jsonl, object_sha256, file_sha256
from .config import create_lock, verify_lock, scientific_config, source_manifest
from .runner import ExperimentRunner
from .analysis import relation_consistency, cosine_matrix, group_residual_decomposition, decompose


def directed_transfer_consistency(first, second, *, zero_tolerance=1e-12):
    """Compare every ordered off-diagonal E3 relation, never just its upper half.

    Rank correlations use baseline-std column-scaled effects, so chrF points
    cannot dominate binary accuracy simply through units. Signs are reported
    for all finite raw effects, including columns with zero baseline variance.
    """
    from scipy.stats import spearmanr
    if first["tasks"] != second["tasks"]:
        raise ValueError("Directed transfer comparisons require identical ordered task coordinates")
    tasks, tolerance = first["tasks"], float(zero_tolerance)
    if tolerance < 0:
        raise ValueError("Transfer sign tolerance must be nonnegative")
    raw_a, raw_b = np.asarray(first["raw"], dtype=float), np.asarray(second["raw"], dtype=float)
    scaled_a, scaled_b = np.asarray(first["standardized"], dtype=float), np.asarray(second["standardized"], dtype=float)
    shape = (len(tasks), len(tasks))
    if any(matrix.shape != shape for matrix in (raw_a, raw_b, scaled_a, scaled_b)):
        raise ValueError("Transfer matrix shape differs from its task coordinates")
    mask = ~np.eye(len(tasks), dtype=bool)
    raw_valid = mask & np.isfinite(raw_a) & np.isfinite(raw_b)
    sign_a = np.where(np.abs(raw_a) <= tolerance, 0, np.sign(raw_a))
    sign_b = np.where(np.abs(raw_b) <= tolerance, 0, np.sign(raw_b))
    records = [{"source": tasks[i], "target": tasks[j], "first_delta": float(raw_a[i, j]),
                "second_delta": float(raw_b[i, j]), "first_sign": int(sign_a[i, j]),
                "second_sign": int(sign_b[i, j]), "agrees": bool(sign_a[i, j] == sign_b[i, j])}
               for i, j in zip(*np.where(raw_valid))]
    variable_columns = ((np.asarray(first["column_std"], dtype=float) > tolerance) &
                        (np.asarray(second["column_std"], dtype=float) > tolerance))
    valid = raw_valid & np.isfinite(scaled_a) & np.isfinite(scaled_b) & variable_columns[None, :]
    values_a, values_b = scaled_a[valid], scaled_b[valid]
    constant_a = len(values_a) == 0 or float(np.std(values_a)) <= tolerance
    constant_b = len(values_b) == 0 or float(np.std(values_b)) <= tolerance
    if len(values_a) >= 3 and not constant_a and not constant_b:
        correlation, pvalue = spearmanr(values_a, values_b)
        reason = None
    else:
        correlation, pvalue = None, None
        reason = "fewer than three variable-column pairs or constant functional profile"
    nonzero = raw_valid & ((sign_a != 0) | (sign_b != 0))
    return {"spearman": None if correlation is None else float(correlation),
            "descriptive_pvalue": None if pvalue is None else float(pvalue),
            "ordered_pairs": len(records), "correlation_pairs": int(valid.sum()),
            "expected_ordered_pairs": len(tasks) * (len(tasks) - 1),
            "sign_agreement": float(np.mean((sign_a == sign_b)[raw_valid])) if raw_valid.any() else None,
            "nonzero_sign_agreement": float(np.mean((sign_a == sign_b)[nonzero])) if nonzero.any() else None,
            "both_zero_fraction": float(np.mean(((sign_a == 0) & (sign_b == 0))[raw_valid])) if raw_valid.any() else None,
            "both_profiles_constant": constant_a and constant_b, "zero_tolerance": tolerance,
            "pairs": records, "undefined_correlation_reason": reason,
            "direction": "all ordered source->target pairs excluding self-transfer",
            "correlation_scale": "target-baseline sample std",
            "interpretation": "descriptive; ordered task pairs are dependent observations"}


def functional_reproducibility(runner, scope, seeds, split="discovery"):
    cfg = runner.config["analysis"]
    tolerance = float(cfg.get("transfer_sign_tolerance", 1e-12))
    minimum_sign = float(cfg.get("transfer_min_sign_agreement", .8))
    minimum_correlation = float(cfg.get("transfer_min_spearman", .5))
    panels = {route: {seed: read_json(runner.output_root / "analysis" / scope / str(seed) /
                                     f"{route}-{split}-canonical-core.json")["matrix"]
                      for seed in seeds} for route in ("ltv", "loda")}
    output = {"routes": {}, "cross_space": {},
              "rule": {"min_sign_agreement": minimum_sign, "min_spearman_when_defined": minimum_correlation,
                       "constant_profile_rule": "both constant and all ordered signs agree",
                       "zero_tolerance": tolerance, "all_ordered_offdiagonal_pairs": True,
                       "cross_space_agreement_is_descriptive_only": True}}
    for route in ("ltv", "loda"):
        comparisons = {}
        for index, first_seed in enumerate(seeds):
            for second_seed in seeds[index + 1:]:
                evidence = directed_transfer_consistency(panels[route][first_seed], panels[route][second_seed], zero_tolerance=tolerance)
                complete = evidence["ordered_pairs"] == evidence["expected_ordered_pairs"]
                if evidence["spearman"] is not None:
                    stable = evidence["sign_agreement"] >= minimum_sign and evidence["spearman"] >= minimum_correlation
                else:
                    stable = evidence["both_profiles_constant"] and evidence["sign_agreement"] == 1.0
                evidence["passes_declared_reproducibility_rule"] = bool(complete and stable)
                comparisons[f"{first_seed}-{second_seed}"] = evidence
        output["routes"][route] = {"comparisons": comparisons,
                                    "passes": all(value["passes_declared_reproducibility_rule"] for value in comparisons.values()),
                                    "status": "pending_additional_seed" if len(seeds) < 2 else "assessed"}
    for seed in seeds:
        output["cross_space"][str(seed)] = directed_transfer_consistency(panels["ltv"][seed], panels["loda"][seed], zero_tolerance=tolerance)
    output["passes"] = all(value["passes"] for value in output["routes"].values())
    output["status"] = "pending_additional_seed" if len(seeds) < 2 else "assessed"
    return output


def smoke_evidence(runner, scope=None):
    path = runner.output_root / "smoke.json"
    record = read_json(path) if path.exists() else {}
    expected_backend = str(runner.config["platform"]["device"]).split(":")[0]
    checks = {"passed": bool(record.get("passed")),
              "true_loda_gao_passed": record.get("checks", {}).get("loda_true_gao_microbatch_step") is True,
              "loda_resume_exact": record.get("checks", {}).get("loda_checkpoint_resume_exact") is True,
              "loda_resume_no_repeat": record.get("checks", {}).get("loda_resume_does_not_repeat_step") is True,
              "configuration_matches": record.get("config_sha256") == object_sha256(scientific_config(runner.config)),
              "base_matches": record.get("base_fingerprint") == runner.base_fingerprint(),
              "dataset_matches": record.get("data_manifest_sha256") == object_sha256(runner.manifest),
              "source_matches": record.get("source_manifest_sha256") == object_sha256(source_manifest(runner.source_root)),
              "device_backend_matches": str(record.get("device", "")).split(":")[0] == expected_backend,
              "source_task_allowed": scope is None or record.get("task") in runner.taskset(scope)}
    return {"valid": all(checks.values()), "checks": checks, "path": str(path),
            "observed_device": record.get("device"), "expected_backend": expected_backend}


def assess_gate(runner, scope, seeds, split=None):
    split = split or ("test" if scope == "mvp" and (runner.output_root / "analysis" / scope / "first-round-completed.json").exists() else "discovery")
    tasks = runner.taskset(scope)
    requested = runner.config["analysis"].get("gate_min_positive_tasks", "all")
    required = len(tasks) if requested == "all" else int(requested)
    details, valid = {}, True
    for route in ("ltv", "loda"):
        by_task = {}
        for task in tasks:
            records = []
            for seed in seeds:
                panel = read_json(runner.output_root / "analysis" / scope / str(seed) / f"{route}-{split}-canonical-core.json")
                evidence = panel["E1"][task]
                own = evidence["own_task"]
                correct_score = own["condition"]
                controls = [v["delta"][evidence["primary_metric"]]["condition"] for v in evidence["controls"].values()]
                records.append({"seed": seed, "own_delta": own["delta"], "ci": own["ci"],
                                "beats_base": own["delta"] > 0,
                                "beats_controls": all(correct_score > x for x in controls),
                                "correct_score": correct_score, "control_scores": controls})
            by_task[task] = {"passes": all(r["beats_base"] and r["beats_controls"] for r in records), "seeds": records}
        passed = sum(x["passes"] for x in by_task.values())
        details[route] = {"tasks": by_task, "positive_tasks": passed, "required": required, "passes": passed >= required}
        valid = valid and passed >= required
    smoke = smoke_evidence(runner, scope)
    smoke_valid = smoke["valid"]
    functional = functional_reproducibility(runner, scope, seeds, split)
    complete_seeds = set(seeds) == set(runner.config["analysis"]["seeds"])
    result = {"scope": scope, "seeds": seeds, "evaluation_split": split, "passes": valid and smoke_valid and functional["passes"], "E1": details,
              "functional_reproducibility": functional,
              "fully_validated": bool(valid and smoke_valid and functional["passes"] and complete_seeds and len(seeds) >= 2),
              "smoke_passes": smoke_valid, "smoke_path": smoke["path"], "smoke_provenance": smoke,
              "criterion": "each required task beats BASE and all initialization/random own-task controls; all directed cross-task relations pass the locked cross-seed sign/rank rule; consistency smoke passed",
              "confidence_intervals": "reported for every effect; point-effect gate is a declared engineering stop rule, not a statistical discovery",
              "gate_failure": "retain every artifact and report; dependent stages stop without claiming absence/presence of shared capability"}
    atomic_json(runner.output_root / "analysis" / scope / "gate.json", result)
    return result


def run_core(runner, scope="mvp"):
    seeds = runner.config["analysis"]["seeds"]
    # Validate real-model intervention semantics and accelerator execution
    # before spending a single full training run.
    if not smoke_evidence(runner, scope)["valid"]:
        from .smoke import run_smoke
        runner.release_model()
        smoke_path = runner.output_root / "smoke.json"
        if smoke_path.exists():
            # Preserve the old evidence and execute into a fresh directory;
            # run_smoke itself correctly refuses overwriting any run record.
            import time
            old = read_json(smoke_path)
            atomic_json(runner.output_root / "engineering_smoke" / f"previous-{object_sha256(old)}.json", old)
            smoke_dir = runner.output_root / "engineering_smoke" / str(time.time_ns())
            record = run_smoke(runner.config, task=runner.taskset(scope)[0], output_dir=smoke_dir)
            atomic_json(smoke_path, record)
        else:
            run_smoke(runner.config, task=runner.taskset(scope)[0], output_dir=runner.output_root)
    smoke = smoke_evidence(runner, scope)
    if not smoke["valid"]:
        return {"completed": False, "stop_reason": "engineering_smoke_provenance", "smoke": smoke}
    for index, seed in enumerate(seeds):
        for route in ("ltv", "loda"):
            for task in runner.taskset(scope):
                runner.train(scope, route, seed, task)
        if scope == "mvp":
            runner.lock_core(scope, seed)
            with runner.protocol_phase("core"):
                for route in ("ltv", "loda"):
                    runner.core_panel(scope, route, seed, split="test")
                runner.complete_core_stage(scope, seed)
        for route in ("ltv", "loda"):
            runner.core_panel(scope, route, seed)
        gate = assess_gate(runner, scope, seeds[:index + 1], "test" if scope == "mvp" else "discovery")
        if not gate["passes"]:
            return {"completed": False, "stop_reason": "validity_or_consistency_gate", "gate": gate}
    return {"completed": True, "gate": gate}


def _require_gate(runner, scope):
    gate = assess_gate(runner, scope, runner.config["analysis"]["seeds"])
    if not gate["passes"] or not gate["fully_validated"]:
        raise RuntimeError(f"Dependent experiments stopped by documented validity gate: {runner.output_root / 'analysis' / scope / 'gate.json'}")


def run_discovery(runner, scope="mvp", *, source_fold=False):
    from .joint import calibrate_joint_thresholds
    if runner.lock_path(scope).is_file():
        # Frozen choices are reusable. Re-exporting identical tensors would
        # needlessly rewrite candidate files and invalidate their byte hashes.
        return runner.verify_evaluation_protocol(scope, runner.config["analysis"]["seeds"][0])["discovery_decisions"]
    # A leave-one-out fold is assessed only on its own source tasks. Reusing
    # the MVP gate would read the held-out task's discovery observations.
    _require_gate(runner, scope)
    tasks = runner.taskset(scope)
    decisions = {}
    for seed in runner.config["analysis"]["seeds"]:
        decisions[str(seed)] = {}
        baseline_rows = {t: runner.evaluate(None, t, "tune", label="base", scope=scope, seed=seed)["rows"] for t in tasks}
        thresholds = calibrate_joint_thresholds(baseline_rows, runner.config)
        decisions[str(seed)]["joint_thresholds"] = thresholds
        runner.geometry(scope, seed)
        for route in ("ltv", "loda"):
            core_path = runner.output_root / "analysis" / scope / str(seed) / f"{route}-discovery-canonical-core.json"
            core = read_json(core_path) if core_path.exists() else runner.core_panel(scope, route, seed)
            strengths = runner.tune_strengths(scope, route, seed)
            components = runner.component_panel(scope, route, seed)
            clusters = runner.functional_clustering(scope, route, seed, core)
            composition = runner.composition_candidates(scope, route, seed, components["selected"])
            decisions[str(seed)][route] = {"decomposition": components["selected"], "strengths": strengths,
                                          "clusters": clusters, "compositions": composition["choices"]}
    # A per-seed bootstrap-stable partition need not reproduce across seeds.
    # The six-task group stage requires the same unordered partition in every
    # seed; differing or absent partitions remain valid reported outcomes.
    group_stability = {}
    for route in ("ltv", "loda"):
        clusters = [decisions[str(seed)][route]["clusters"] for seed in runner.config["analysis"]["seeds"]]
        partitions = [sorted(tuple(sorted(group)) for group in cluster["groups"]) for cluster in clusters]
        stable = len(tasks) >= 6 and len(clusters) >= 2 and all(cluster["stable"] for cluster in clusters) and all(partition == partitions[0] for partition in partitions)
        group_stability[route] = {"stable_across_seeds": stable,
                                  "rule": "six tasks, bootstrap-stable partition in every seed, identical unordered source-task partition",
                                  "partitions": partitions}
        if not stable:
            continue
        for seed in runner.config["analysis"]["seeds"]:
            selected = decisions[str(seed)][route]["decomposition"]
            reps = runner.representations(scope, route, seed)
            global_decomp = decompose(reps, selected["method"], selected["q"])
            indices = [[tasks.index(task) for task in group] for group in decisions[str(seed)][route]["clusters"]["groups"]]
            if sorted(index for group in indices for index in group) != list(range(len(tasks))):
                raise ValueError("Cross-seed stable groups must partition all six source tasks")
            group_choices = {}
            base = {t: runner.evaluate(None, t, "discovery", label="base", scope=scope, seed=seed) for t in tasks}
            for group_index, members in enumerate(indices):
                source_reps = [global_decomp.residual[i] for i in members]
                records = []
                for q in runner.config["analysis"]["ranks"]:
                    d = decompose(source_reps, "svd", min(q, len(members)))
                    from .representations import linear_combination
                    group_shared = linear_combination(d.shared, [1 / len(members)] * len(members))
                    gains = {}
                    for target in tasks:
                        score = runner.evaluate(group_shared, target, "discovery", label=f"{route}-group{group_index}-q{q}-shared", scope=scope, seed=seed)
                        gains[target] = score["metrics"][score["primary_metric"]] - base[target]["metrics"][base[target]["primary_metric"]]
                    records.append({"q": q, "used_task_space_rank": min(q, len(members)), "gains": gains})
                from .analysis import choose_discovery_candidate
                chosen, candidates = choose_discovery_candidate(records, tasks, {t: float(np.std(base[t]["sample_scores"], ddof=1)) for t in tasks})
                group_choices[str(group_index)] = {"tasks": [tasks[i] for i in members], "selected": chosen, "candidates": candidates}
            decisions[str(seed)][route]["group_residual_svd"] = group_choices
    decisions["group_stability_across_seeds"] = group_stability
    # Geometry reproducibility is descriptive and uses every seed, including
    # negative/zero results. It does not substitute for functional transfer.
    reproducibility = {}
    seeds = runner.config["analysis"]["seeds"]
    for route in ("ltv", "loda"):
        reproducibility[route] = {}
        for i, first in enumerate(seeds):
            for second in seeds[i + 1:]:
                reproducibility[route][f"{first}-{second}"] = relation_consistency(cosine_matrix(runner.representations(scope, route, first)), cosine_matrix(runner.representations(scope, route, second)))
    decisions["geometry_across_seeds"] = reproducibility
    decisions["functional_across_seeds_and_spaces"] = functional_reproducibility(runner, scope, seeds)
    decisions["selection_audit"] = {"scope": scope, "source_tasks": tasks, "data_splits_used": ["train", "tune", "discovery"],
                                    "heldout": scope[4:] if scope.startswith("loo-") else None, "final_test_used": False}
    atomic_json(runner.output_root / "protocol" / f"{scope}.decisions.json", decisions)
    return decisions


def lock_protocol(runner, scope):
    decisions = read_json(runner.output_root / "protocol" / f"{scope}.decisions.json")
    decisions["asset_fingerprints"] = runner.asset_fingerprints()
    decisions["artifacts"] = runner.freeze_artifacts(scope, decisions=decisions)
    return create_lock(runner.config, runner.manifest, decisions, runner.lock_path(scope), runner.source_root)


def run_final(runner, scope="mvp", split="test"):
    lock = runner.verify_evaluation_protocol(scope, runner.config["analysis"]["seeds"][0])
    runner.verify_artifacts(lock["discovery_decisions"].get("artifacts", {}))
    decisions = lock["discovery_decisions"]
    results = {}
    for seed in runner.config["analysis"]["seeds"]:
        if scope.startswith("loo-"):
            results[str(seed)] = runner.heldout_final(scope[4:], seed, decisions, split)
            continue
        results[str(seed)] = {}
        for route in ("ltv", "loda"):
            route_decisions = decisions[str(seed)][route]
            final_core = runner.core_panel(scope, route, seed, split)
            from .analysis import confirm_transfer_relations
            discovery_core = read_json(runner.output_root / "analysis" / scope / str(seed) / f"{route}-discovery-canonical-core.json")
            confirmation = confirm_transfer_relations(discovery_core, final_core)
            confirmation["discovery_groups"] = route_decisions["clusters"]
            confirmation["groups_and_rank_reselected_on_test"] = False
            atomic_json(runner.output_root / "analysis" / scope / str(seed) / f"{route}-{split}-relation-confirmation.json", confirmation)
            runner.component_panel(scope, route, seed, split, selected=route_decisions["decomposition"])
            for variant in runner.config["evaluation"]["variants"]:
                if variant == "canonical":
                    continue
                runner.core_panel(scope, route, seed, split, variant=variant)
                runner.component_panel(scope, route, seed, split, selected=route_decisions["decomposition"], variant=variant)
            # Tuned strengths remain a separately labeled transfer matrix.
            tuned_matrix = {}
            from .representations import scaled
            for rep in runner.representations(scope, route, seed):
                strength = route_decisions["strengths"][rep.task]["selected"]["strength"]
                tuned_matrix[rep.task] = {}
                for target in runner.taskset(scope):
                    score = runner.evaluate(scaled(rep, strength), target, split, label=f"{route}-{rep.task}-tuned-strength", scope=scope, seed=seed)
                    tuned_matrix[rep.task][target] = {"strength": strength, "metrics": score["metrics"]}
            results[str(seed)][route] = {"tuned_transfer": tuned_matrix,
                                        "composition": runner.composition_final(scope, route, seed, route_decisions["compositions"], decisions[str(seed)]["joint_thresholds"], split)}
            if route_decisions.get("group_residual_svd"):
                runner.group_final(scope, route, seed, route_decisions, split)
        runner.branch_panel(scope, seed, split)
        runner.summary_final(scope, seed, decisions, split)
    atomic_json(runner.output_root / "analysis" / scope / f"{split}-completed.json", {"seeds": list(results), "protocol_lock": lock["lock_sha256"], "scope": scope})
    return {"completed": True, "scope": scope, "split": split}


def run_heldout_discovery(runner):
    results = {}
    for heldout in runner.config["analysis"]["heldout_tasks"]:
        scope = f"loo-{heldout}"
        if runner.lock_path(scope).is_file():
            runner.verify_evaluation_protocol(scope, runner.config["analysis"]["seeds"][0])
            results[heldout] = {"completed": True, "scope": scope, "stage": "source_only_discovery_locked",
                                "heldout_selection_or_test_accessed": False, "reused_locked_discovery": True}
            continue
        core = run_core(runner, scope)
        if not core["completed"]:
            results[heldout] = {"completed": False, "scope": scope, "stage": "source_only_core",
                                "evidence": core, "heldout_selection_or_test_accessed": False}
            atomic_json(runner.output_root / "analysis" / scope / "heldout-stage-status.json", results[heldout])
            continue
        run_discovery(runner, scope, source_fold=True)
        lock_protocol(runner, scope)
        results[heldout] = {"completed": True, "scope": scope, "stage": "source_only_discovery_locked",
                            "heldout_selection_or_test_accessed": False}
        atomic_json(runner.output_root / "analysis" / scope / "heldout-stage-status.json", results[heldout])
    return results


def _sequential_provenance(runner, main_lock):
    return {"base_fingerprint": runner.base_fingerprint(),
            "data_manifest_sha256": object_sha256(runner.manifest),
            "source_manifest_sha256": object_sha256(source_manifest(runner.source_root)),
            "protocol_lock_sha256": main_lock["lock_sha256"],
            "asset_fingerprints": runner.asset_fingerprints()}


def run_ablations(runner, split="test", *, evaluation_only=False):
    _require_gate(runner, "mvp")
    main_lock = runner.verify_evaluation_protocol("mvp", runner.config["analysis"]["seeds"][0])
    results = {}
    for seed in runner.config["analysis"]["seeds"]:
        layer_results = {}
        for layers in runner.config["analysis"]["ltv_layer_ablations"]:
            layer_tag = "-".join(map(str, layers))
            control_path = runner.output_root / "protocol" / f"mvp.layers-{layer_tag}-seed{seed}.lock.json"
            if not evaluation_only:
                for task in runner.taskset("mvp"):
                    runner.train("mvp", "ltv", seed, task, layers)
                create_lock(runner.config, runner.manifest,
                            {"phase": "layer_control", "seed": seed, "layers": layers,
                             "main_protocol_lock": main_lock["lock_sha256"],
                             "asset_fingerprints": runner.asset_fingerprints(),
                             "artifacts": runner.freeze_artifacts("mvp", [seed], layers=layers)},
                            control_path, runner.source_root)
            if not control_path.is_file():
                raise ValueError(f"Evaluation-only requires a completed layer control: {control_path}")
            control_lock = runner.verify_frozen_protocol(control_path)
            if control_lock["discovery_decisions"].get("main_protocol_lock") != main_lock["lock_sha256"]:
                raise ValueError("Layer control belongs to a different main protocol")
            layer_results[layer_tag] = runner.core_panel("mvp", "ltv", seed, split, layers)
        # Sequential evaluations use the actual merged BF16 model, preserving
        # rounding; they never evaluate algebraic factors as if no merge occurred.
        baseline = {t: runner.evaluate(None, t, split, label="base", seed=seed) for t in runner.taskset("mvp")}
        runner.release_model()
        config = deepcopy(runner.config)
        config["sequential"] = {"tasks": runner.taskset("mvp"), "seed": seed,
                                "data_root": str(runner.data_root), "output_root": str(runner.output_root),
                                "provenance": _sequential_provenance(runner, main_lock)}
        from .sequential import run_sequential
        from .evaluation import generate_predictions
        from .metrics import score_predictions
        from .runtime import synchronize, memory_evidence
        import time
        def access_merged(target, current_split, samples, context):
            lock = runner.verify_evaluation_protocol("mvp", seed)
            runner.record_final_access(target, current_split, samples, scope="mvp", seed=seed,
                                      label=f"sequential-order{context['order']}-stage{context['stage']}",
                                      variant="canonical", lock_hash=lock["lock_sha256"])
        def evaluate_merged(model, tokenizer, samples, stage_config, device):
            target = samples[0].task
            synchronize(device)
            started = time.perf_counter()
            seq = stage_config["sequential"]
            generation = {"provenance": seq["provenance"], "stage_signature": seq["stage_signature"],
                          "merged_weights_sha256": seq["merged_weights_sha256"],
                          "evaluation": stage_config["evaluation"], "metrics": stage_config["metrics"],
                          "max_prompt_tokens": stage_config["training"]["max_prompt_tokens"]}
            digests = {sample.id: object_sha256(sample.to_dict()) for sample in samples}
            reused = {}
            if split == "test_formal":
                old_dir = Path(seq["evaluation_dir"]).parent.parent / "test" / target
                completed = old_dir / "evaluation-completed.json"
                if completed.is_file():
                    previous = read_json(completed)
                    for name, expected in previous.get("artifacts", {}).items():
                        if not (old_dir / name).is_file() or file_sha256(old_dir / name) != expected:
                            raise ValueError(f"Sequential primary evaluation changed: {old_dir}")
                    previous_score = read_json(old_dir / "metrics.json")
                    if previous_score.get("generation_provenance") == generation:
                        old_digests = previous_score.get("generation_sample_digests", {})
                        reused = {row["id"]: row for row in read_jsonl(old_dir / "predictions.jsonl")
                                  if row["id"] in digests and old_digests.get(row["id"]) == digests[row["id"]]}
            remaining = [sample for sample in samples if sample.id not in reused]
            generated = generate_predictions(model, tokenizer, remaining, None, stage_config, device) if remaining else []
            by_id = {**reused, **{row["id"]: row for row in generated}}
            rows = [by_id[sample.id] for sample in samples]
            score = score_predictions(target, rows, stage_config)
            synchronize(device)
            score.update(split=split, runtime_seconds=time.perf_counter() - started,
                         memory=memory_evidence(device), newly_generated=len(generated), reused_predictions=len(samples)-len(generated),
                         generation_provenance=generation, generation_sample_digests=digests,
                         effect=runner.delta(score, baseline[target], target, seed),
                         evaluated_on_actual_merged_model=True)
            return score
        sequential = run_sequential(config, device=runner.device, split=split,
                                    evaluation_only=evaluation_only, evaluate_fn=evaluate_merged,
                                    access_fn=access_merged)
        results[str(seed)] = {"layers": list(layer_results), "sequential_orders": len(sequential)}
    result = {"completed": True, "scope": "mvp", "split": split,
              "protocol_lock": main_lock["lock_sha256"], "evaluation_only": evaluation_only, "seeds": results}
    atomic_json(runner.output_root / "analysis" / "controls" / f"{split}-completed.json", result)
    return result


def run_formal(runner):
    """All formally expanded panels; no training, tuning or discovery calls."""
    scopes = ["mvp", "expanded"] + [f"loo-{task}" for task in runner.config["analysis"]["heldout_tasks"]]
    # Validate every main lock before exposing any new formal test samples.
    locks = {}
    for scope in scopes:
        lock = runner.verify_evaluation_protocol(scope, runner.config["analysis"]["seeds"][0])
        locks[scope] = lock
        marker = runner.output_root / "analysis" / scope / "test-completed.json"
        record = read_json(marker) if marker.exists() else {}
        if (record.get("protocol_lock") != lock["lock_sha256"] or
                set(record.get("seeds", [])) != set(map(str, runner.config["analysis"]["seeds"]))):
            raise ValueError(f"Formal evaluation requires complete primary panels: {marker}")
    marker = runner.output_root / "analysis" / "controls" / "test-completed.json"
    record = read_json(marker) if marker.exists() else {}
    if (not record.get("completed") or record.get("protocol_lock") != locks["mvp"]["lock_sha256"] or
            set(record.get("seeds", {})) != set(map(str, runner.config["analysis"]["seeds"]))):
        raise ValueError(f"Formal evaluation requires complete primary controls: {marker}")
    from .sequential import cyclic_orders
    provenance = _sequential_provenance(runner, locks["mvp"])
    for seed in runner.config["analysis"]["seeds"]:
        for layers in runner.config["analysis"]["ltv_layer_ablations"]:
            layer_tag = "-".join(map(str, layers))
            control = runner.verify_frozen_protocol(runner.output_root / "protocol" / f"mvp.layers-{layer_tag}-seed{seed}.lock.json")
            if control["discovery_decisions"].get("main_protocol_lock") != locks["mvp"]["lock_sha256"]:
                raise ValueError("Layer control belongs to a different main protocol")
        for order in cyclic_orders(runner.taskset("mvp")):
            folder = runner.output_root / "sequential" / f"seed{seed}" / "__".join(order)
            required = [folder / "order-completed.json"] + [folder / f"stage{i}_{task}" / "stage-completed.json" for i, task in enumerate(order)]
            if any(not path.is_file() for path in required):
                raise ValueError(f"Formal evaluation requires completed sequential training: {folder}")
            identity = read_json(folder / "order-request.json")
            completed = read_json(folder / "order-completed.json")
            if (identity.get("provenance") != provenance or identity.get("order") != list(order) or
                    identity.get("seed") != seed or completed.get("status") != "complete" or
                    completed.get("order_signature") != object_sha256(identity)):
                raise ValueError(f"Sequential control provenance differs: {folder}")
            for path in required[1:]:
                stage = read_json(path)
                if (stage.get("status") != "complete" or stage.get("schema_version") != 2 or
                        stage.get("stage_signature") != object_sha256(stage.get("request", {})) or
                        stage.get("request", {}).get("provenance") != provenance):
                    raise ValueError(f"Sequential stage provenance differs: {path}")
    finals = {scope: run_final(runner, scope, "test_formal") for scope in scopes}
    controls = run_ablations(runner, "test_formal", evaluation_only=True)
    return {"completed": all(panel["completed"] for panel in finals.values()) and controls["completed"],
            "split": "test_formal", "evaluation_only": True, "scopes": finals, "controls": controls}


def run_phase(runner, phase, scope="mvp", split="test"):
    if phase == "core":
        return run_core(runner, scope)
    if phase == "discover":
        return run_discovery(runner, scope, source_fold=scope.startswith("loo-"))
    if phase == "lock":
        return lock_protocol(runner, scope)
    if phase == "final":
        return run_final(runner, scope, split)
    if phase == "formal":
        return run_formal(runner)
    if phase == "heldout":
        discovery = run_heldout_discovery(runner)
        folds = {task: run_final(runner, status["scope"], split) if status["completed"] else status
                 for task, status in discovery.items()}
        return {"completed": all(fold["completed"] for fold in folds.values()), "folds": folds}
    if phase == "ablations":
        return run_ablations(runner, split)
    if phase == "extended":
        return run_phase(runner, "all", "expanded", split)
    if phase == "all":
        if split == "test_formal":
            primary = run_phase(runner, "all", scope, "test")
            if not primary["completed"]:
                return primary
            formal = run_formal(runner) if scope == "mvp" else run_final(runner, scope, "test_formal")
            return {"completed": formal["completed"], "primary": primary, "formal": formal}
        if runner.lock_path(scope).is_file():
            runner.verify_evaluation_protocol(scope, runner.config["analysis"]["seeds"][0])
        else:
            core = run_core(runner, scope)
            if not core["completed"]:
                return core
            run_discovery(runner, scope)
            lock_protocol(runner, scope)
        final = run_final(runner, scope, split)
        if scope == "mvp":
            heldout = run_phase(runner, "heldout", scope, split)
            ablations = run_ablations(runner, split)
            expansion = run_phase(runner, "all", "expanded", split)
            return {"completed": bool(final["completed"] and ablations["completed"] and expansion["completed"]
                                       and heldout["completed"]),
                    "mvp": final, "heldout": heldout, "ablations": ablations, "expansion": expansion}
        return final
    raise ValueError(f"Unknown phase {phase}")
