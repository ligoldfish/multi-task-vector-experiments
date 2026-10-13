"""Reviewable experiment inventory, including the conditional expansion stages."""
from __future__ import annotations
from itertools import combinations


def experiment_inventory(config):
    core = config["data"]["tasks"]
    extended = core + config["data"]["extension_tasks"]
    seeds = config["analysis"]["seeds"]
    rows = []
    def add(stage, requirement, operation, *, tasks=None, seeds_=None, conditions=None, gate=None, note=""):
        rows.append({"stage": stage, "requirement": requirement, "operation": operation,
                     "tasks": tasks or core, "seeds": seeds_ or seeds,
                     "conditions": conditions or [], "gate": gate, "note": note})
    add("prepare", "R01-R16", "official_split_grouped_data_and_repair_validation", tasks=extended + ["summary"], seeds_=[17])
    add("diagnostics", "I01-I10", "tiny_and_real_model_consistency", seeds_=[17], conditions=["reload", "prefill_cached_vs_uncached", "teacher_forced_prompt_position", "frozen_base"])
    add("first_round", "E1/E3", "locked_core_test_before_structure_discovery", seeds_=[17],
        conditions=["four_LTV", "four_LoDA_MT", "BASE", "four_tasks_x256", "9216_core_predictions", "first_round_completed_ledger"],
        note="Train/tune-only core lock; decomposition/composition/E6 follow validity. Extra E1 controls are counted separately.")
    add("core", "E1", "independent_training", conditions=["LTV_layer14", "LoDA_MT_common_base_raw_and_post"])
    add("core", "E1", "own_task_controls", conditions=["base", "correct", "random_norm_matched", "untrained_initialization"])
    add("core", "E3", "directed_full_transfer_matrix", conditions=["scale1_primary", "tune_selected_scale_separate", "paired_group_bootstrap", "raw_metric_units", "column_standardization"])
    add("discovery", "E2", "actual_representation_geometry", conditions=["cosine", "LoDA_per_layer_input_output_angles", "initialization_controls", "cross_seed_and_cross_space_relations"], gate="effective_and_seed_stable")
    add("discovery", "E4", "component_interventions", conditions=["mean", "svd_q0", "svd_q1", "svd_q2", "shared", "residual", "reconstruction", "delete_shared_alias", "random_same_q_and_norm"], gate="effective_and_seed_stable")
    add("discovery", "E3", "functional_hierarchical_clustering", conditions=["bootstrap_coassignment", "single_group_allowed", "no_stable_group_allowed"], gate="effective_and_seed_stable")
    add("final", "E4", "format_and_label_controls", conditions=["output_format", "logic_label_recode", "prompt_paraphrase"], gate="protocol_locked")
    add("final", "E5", "pairwise_composition", conditions=["each_original", "sum", "mean", "weighted", "decomposed", "same_candidate_budget", "raw_and_same_norm", "original_task_panel", "joint_input_separate_scores_and_joint_criterion"], gate="protocol_locked", note=f"{len(list(combinations(core, 2)))} pairs; joint_count is per pair")
    add("final", "E6", "leave_one_task_out", conditions=["four_folds", "no_heldout_source_statistics", "retrain_LoDA_A_and_B", "source_only_rank_and_weights", "heldout_test_only"], gate="protocol_locked")
    add("final", "E6", "unseen_summary", tasks=["summary"], conditions=["SAMSum", "source_only_settings", "no_summary_tuning"], gate="protocol_locked")
    add("ablations", "LTV", "retrain_layer_ablations", conditions=["layer6", "layer22", "joint_layers6_14_22"], gate="effective_and_seed_stable")
    add("ablations", "LoDA", "branch_and_recalibration_controls", conditions=["general_only", "isolated_only", "raw", "post"], gate="effective_and_seed_stable")
    add("ablations", "LoDA", "sequential_four_cyclic_orders", conditions=config["analysis"]["sequential_orders"], gate="effective_and_seed_stable", note="Separate from common-base LoDA-MT; committed native BF16 merges reused for every evaluation split")
    add("formal", "E1-E6/controls", "all_locked_panels_evaluation_only", tasks=extended + ["summary"],
        conditions=["mvp", "six_tasks", "four_leave_one_out_folds", "layer_controls", "four_cyclic_orders"],
        gate="all_primary_panels_and_controls_complete", note="Full chains primary test then formal automatically; formal never retrains or reselects.")
    add("extension", "E1-E6", "six_task_repeat", tasks=extended, conditions=["MBPP_generation", "verified_MBPP_repair", "stable_group_residual_SVD", "ties", "tsv", "knots"], gate="effective_and_seed_stable")
    n, test = len(core), config["data"]["counts"]["test"]
    primary_states = 1 + 2 * n
    budget = {"core_test_primary_generations": primary_states * n * test,
              "core_discovery_primary_generations": primary_states * n * config["data"]["counts"]["discovery"],
              "E1_extra_random_own_task": 2 * n * test * config["evaluation"]["random_repeats"],
              "E1_extra_LTV_initial_own_task": n * test,
              "LoDA_initial_equals_base": "reuse only after numerical/output equivalence verified",
              "additional": "tuning, raw/post, branch, formats, decomposition, combinations, folds, seeds, sequential, extensions and formal panels are additional, never included in 9216"}
    return {"schema_version": 1, "inventory": rows, "first_seed": seeds[0],
            "conditional_seeds": seeds[1:], "budget": budget,
            "all_scope_implemented": True, "scientific_success_assumed": False}
