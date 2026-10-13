# Methods and experiment scope

The experiment asks whether task-specific model adaptations contain a reusable common component. It first learns separate adaptations for several tasks, checks that each adaptation is useful, and then tests whether a common component can transfer, survive controlled interventions, and help combine tasks. A geometric resemblance alone is insufficient evidence of a shared capability.

This repository contains one accelerated implementation. The original plan and implementation supplement are preserved, byte for byte, in [原始实验方案.md](原始实验方案.md) and [补充协议与实现选择.md](补充协议与实现选择.md). Read the supplement for choices left unspecified in the original plan, including MBPP counts, joint-task scoring, leave-one-out folds and numerical details.

## What is learned

The frozen backbone is Qwen3-1.7B, using BF16 and `enable_thinking=False`. The two adaptation routes are analyzed separately because their coordinates differ.

- **LTV:** a learned FP32 activation vector added before a decoder block's RMSNorm, at the last original prompt token. The main layer is index 14. Generation applies it during prefill; cached states carry its effect into later tokens.
- **LoDA-MT:** low-rank updates to native Q/K/V projections throughout the model. Fixed General and Isolated input subspaces each have rank 8. The multi-task comparison freezes the backbone and both A matrices and learns B matrices. Analysis uses the actual represented update `Σ(B @ A)`, including cross terms, rather than flattened A/B coordinates. General/Isolated are construction branches; they are not automatically the experimentally discovered shared/residual components.

LoDA statistics use 128 training prompts and up to 64 uniformly selected non-padding prompt tokens per example. Subspaces derive from the prompt second-moment matrices. The supplement defines the A scaling, GAO numerical choices and post-training recalibration. Both `raw.pt` and `post.pt` are exported; `post.pt` is the primary version and already contains recalibration.

## Tasks and data

| Role | Task | Source | Training / tuning / discovery / initial test |
|---|---|---|---|
| Core | Arithmetic reasoning | GSM8K | 512 / 128 / 128 / 256 |
| Core | Logical reasoning | ProofWriter, OWA, depths 0–3 | 512 / 128 / 128 / 256 |
| Core | English→German translation | IWSLT2017 | 512 / 128 / 128 / 256 |
| Core | Paraphrasing | QQP positive pairs | 512 / 128 / 128 / 256 |
| Extension | Code generation | MBPP | 300 / 37 / 37 / 256 |
| Extension | Code repair | Deterministic validated mutations of MBPP | 300 / 37 / 37 / 256 |
| Auxiliary held-out target | Summarization | SAMSum | 512 / 128 / 128 / 256 prepared; excluded from adaptation selection |

Official train/evaluation boundaries and group separation are preserved. Code generation and repair share original program IDs. Gold programs must pass official tests, mutations must fail, and the full prompts and targets must satisfy token limits. Samples are filtered rather than truncated; insufficient eligible capacity is reported instead of borrowing evaluation examples.

Training, tuning, discovery and test have distinct jobs. Training learns adaptations; tuning selects the lowest-NLL epoch and permitted inference settings; discovery selects decomposition and groups; test confirms a frozen choice. Formal evaluation uses 1000 examples or the available eligible pool. `test_formal` can contain the initial test, so it is an expansion rather than an independent new confirmation set. `test_confirmation` records a separate unexposed reserve where available. Exact IDs, group audits, accepted counts and content hashes are recorded in each run's data manifest.

## E1–E6

| Experiment | Question | Main comparison |
|---|---|---|
| E1: adaptation validity | Did training learn a useful effect? | Correct trained adaptation versus the unchanged backbone, same-norm random direction and untrained initial adaptation on the corresponding task |
| E2: geometry | Do different tasks have related adaptations? | Cosines, actual parameter-update input/output subspace angles, and cross-seed consistency within each route |
| E3: transfer | Does one task's adaptation help another? | Every source adaptation evaluated on each target task with the target prompt fixed; directed changes relative to the unchanged backbone |
| E4: component intervention | Does a candidate shared component actually do something useful? | Mean and normalized-column SVD candidates with q=0/1/2; shared-only, residual-only, reconstruction, shared deletion and matched random controls |
| E5: task composition | Can adaptations be combined without losing their original abilities? | Sum, mean, tuned weights and decomposition-based composition, with matched budgets/norm controls; both original-task retention and joint two-task inputs scored separately and jointly |
| E6: unseen-task transfer | Does a component generalize beyond the tasks used to discover it? | Four leave-one-out core-task folds, source-only selection and LoDA reconstruction/retraining; an additional SAMSum target excluded from selection |

E4's shared-plus-residual reconstruction checks the decomposition algebra; it does not establish functional utility. Deleting shared is the same intervention as using residual-only and is reported as an alias. Six-task stable groups, layer ablations, four cyclic sequential orders and official parameter-merging comparisons extend the main experiment. Dependent stages run only after the declared validity and reproducibility gates pass, and negative results remain part of the record.

## Training and evaluation

The answer-only objective includes EOS, averages token NLL within each sample, then averages samples. The learning rate is 0.001; micro-batch size is 2 and effective logical batch size is 16. LTV accumulates eight micro-batches. LoDA splits a logical batch into two disjoint groups and performs bidirectional sequential GAO optimizer updates. AdamW weight decay is 0.01 for LTV and 0 for LoDA, with linear decay, no warmup and outer gradient clipping at 1.

Training runs for at most five epochs with patience 2 and selects the best tuning NLL. Different task/route runs may stop at different epochs. `initial.pt`, `raw.pt`, `post.pt`, per-epoch/per-step records and checkpoint provenance support inspection. A complete optimizer checkpoint carries optimizer, scheduler, position, RNG and best-state evidence; a vector or `best-adapter.pt` alone is not a complete training checkpoint.

Evaluation is greedy with fixed task-specific generation budgets and metrics: numerical correctness, logical labels, corpus BLEU/chrF++, iBLEU and semantic/copy measures, official code-test success, and summarization ROUGE. Metrics retain their original units. Paired bootstrap resamples original groups and recomputes corpus metrics. Joint examples are grouped through their constituent original tasks rather than assumed independent.

The accelerated implementation increases inference throughput with persistent workers, device sharding, tune-selected inference batches, concurrent independent jobs within an allowed stage, and cached sufficient statistics for bootstrap. It preserves training batch semantics and scientific stage order. The default `validated` policy permits ordinary BF16 shape-dependent token differences and selects batch using source-task tuning inputs, not discovery/test scores. Held-out targets, summarization and joint inputs keep batch 1 while still using device sharding. Target hardware requires its own smoke checks and evaluation; CPU test results do not measure NPU speed.

## Code review map

| Component | Implementation | Relevant tests |
|---|---|---|
| Protocol validation and signatures | `src/task_vector/config.py`, `io.py` | `tests/test_cli.py`, `test_runner_repairs.py` |
| Data, prompts, token limits, MBPP setup | `data.py`, `prompts.py`, `code_execution.py` | `test_data.py`, `test_code_execution.py`, `test_mbpp_recovery.py` |
| Frozen model and intervention positions | `modeling.py`, `ltv.py`, `evaluation.py` | `test_ltv.py`, `test_smoke_prefill.py` |
| LoDA subspaces, recalibration and GAO | `subspaces.py`, `loda.py`, `gao.py` | `test_subspaces.py`, `test_loda.py`, `test_gao.py` |
| Training and checkpoint provenance | `training.py`, `losses.py`, `runner.py` | `test_training.py`, `test_portable_checkpoint.py`, `test_runner_repairs.py` |
| Geometry and shared/residual decomposition | `representations.py`, `analysis.py` | `test_analysis.py` |
| Composition, joint scoring, sequential controls | `joint.py`, `sequential.py`, `workflow.py` | `test_joint.py`, `test_sequential.py`, `test_workflow.py` |
| Metrics and bootstrap | `metrics.py`, `bootstrap.py` | `test_metrics.py`, `test_accelerated_bootstrap.py` |
| Inference batches and parallel workers | `acceleration.py`, `batching.py`, `parallel_evaluation.py` | `test_accelerated_generation.py`, `test_parallel_evaluation.py` |
| Parallel training and sequential orders | `training_pool.py`, `sequential_pool.py` | `test_pool.py`, `test_sequential_pool.py` |

The module names after the first column's full path are under `src/task_vector/`, and test names are under `tests/`. `cli.py` dispatches commands; `workflow.py` determines stage order; `runner.py` performs artifact-checked experiment operations. `ops/` and `deploy/` provide configuration and ModelMate launch helpers.

Run `python tools/verify_sources.py` to verify the preserved experiment source, protocol and configuration against `provenance/source-manifest.json`, without importing ML packages. This establishes content integrity of the published snapshot. To associate a particular `.pt` with training, separately compare the export's training provenance, base fingerprint and artifact digests; repository publication alone does not prove which run produced it.

The public configuration template explicitly enables the engineering settings defined by `DEFAULT_ACCELERATION`; its preexisting scientific values are unchanged. `provenance/source-manifest.json` records this configuration-only publication change, the original source mapping and the published source mapping separately. All training Python files and both original protocol documents retain their original bytes. Verify the publication before generating local configuration files, since those files intentionally become part of a new run's source signature.

`docs/validation/acceleration-evidence.json` is a preserved historical CPU engineering report from 2026-10-13. It is not a fresh execution, a complete scientific result or an NPU throughput benchmark. Third-party fixed commits and file hashes are in `third_party/official_sources.json` and `third_party/baseline_sources.json`; corresponding available license files are retained. The selected LoDA snapshot records a missing upstream LICENSE and does not provide a project-wide license grant.
