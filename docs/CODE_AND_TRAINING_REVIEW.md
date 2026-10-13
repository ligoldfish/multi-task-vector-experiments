# 如何审查代码、训练过程和 task vector

本仓库将实验方法、工程实现和已取得的运行证据分开。审查首先选择 `baseline/` 或 `accelerated/`，再核对向量来源。不要把两份实现的包混入同一个 `PYTHONPATH`；它们都提供名为 `task_vector` 的 Python 包。

## 一条训练链路

1. 从 `configs/experiment.yaml`、`docs/原始实验方案.md` 和 `docs/补充协议与实现选择.md` 核对协议；目标环境生成自己的本地路径配置。
2. `src/task_vector/data.py` 读取官方 split，按原 ID/group 做隔离、验证长度和代码测试，输出固定 JSONL 与 manifest。输入不截断以凑样本数；容量不足保留拒绝理由并失败。
3. `src/task_vector/modeling.py` 加载冻结 Qwen3-1.7B，核对架构与 BF16，关闭 thinking。`prompts.py` 负责模板、answer/EOS 编码及表面变体。
4. LTV 在 `ltv.py` 创建指定层的 FP32 参数，仅注入原 prompt 末 token 的 block 输入。LoDA 经 `subspaces.py` 从训练 prompt 的二阶矩构建子空间，`loda.py` 在原生 Q/K/V 输出加低秩更新，保留 Qwen 的 GQA、QKnorm 和 RoPE。
5. `training.py` 与 `losses.py` 训练。每条回答的 token NLL 先取均值，再按样本均值优化，包含 EOS，prompt/padding 不算 loss。`gao.py` 实现 LoDA 双向、顺序两次 optimizer step；这不是普通一次梯度累积。
6. 每个 epoch 计算 tune NLL，选最低点；最多五轮、patience=2 提前停止，所以各任务 epoch 不同可以正常。保存初始、raw、post、最佳 adapter、完整断点和逐步/逐轮记录。`runner.py` 核对来源及三份向量文件后，最后发布完整 `training.json`。
7. `workflow.py` 按首轮有效性、发现、冻结、测试确认、扩展/留出/对照、正式扩评顺序执行。门禁失败保留负结果；有依赖的实验不继续。

## 重点核对表

下列路径在两份快照中均存在，除注明的加速专用模块。具体源文件以仓库快照为准，不以这张表替代源码审查。

| 要求 | 主要实现与回归测试 | 审查要点 |
|---|---|---|
| 模型与精度 | `modeling.py`、`config.py`；`test_ltv.py`、`test_loda.py` | 同一内容基座，28 层、2048 hidden、16/8 heads，BF16 冻结基座，FP32 adapter master |
| 数据划分 | `data.py`；`test_data.py` | 核心四任务 512/128/128/256；MBPP 300/37/37/256；固定 seed17；test_formal 与 test_confirmation 另记，不能误称互斥 |
| MBPP setup 修复 | `code_execution.py`、`ops/recover_mbpp_prepare.py`；`test_code_execution.py`、`test_mbpp_recovery.py` | 官方程序 927 的 Node setup 按 candidate→setup→tests；生成/修复共用原程序 ID；不修改题目或补入测试集 |
| LTV 位置和 cache | `ltv.py`、`evaluation.py`、`smoke.py`；`test_ltv.py`、`test_smoke_prefill.py` | teacher forcing 固定原 prompt 末位置；生成只在 prefill 注入，KV cache 传播，不每 token 重复注入 |
| LoDA 统计 | `subspaces.py`、`runner.py`；`test_subspaces.py` | 训练128条、最多64个非 padding prompt token，二阶矩均值；固定基座统计；留出折不得使用留出任务统计 |
| LoDA 子空间与参数 | `subspaces.py`、`loda.py`；`test_subspaces.py`、`test_loda.py` | rank8+8，A=基转置/√3，MT 冻结基座和 A，B=0 初始；General 与 Isolated 分支不是研究中发现的 shared/residual |
| 训练超参数 | `config.py`、`training.py`、`gao.py`；`test_training.py`、`test_gao.py` | lr=.001，micro2/effective16，LTV 累积8，LoDA 16→不重叠8/8→micro2；AdamW、线性衰减、clip1；GAO rho上限.05 |
| raw/post | `loda.py`、`representations.py`；`test_loda.py` | post 已编码 `.5 B_G Λ A_G + B_I A_I`，默认主评 post；再次额外乘缩放会改变实验 |
| 完整断点与提交 | `training.py`、`runner.py`、`io.py`；`test_training.py`、`test_portable_checkpoint.py`、`test_runner_repairs.py` | `last.pt`/`best.pt` 包含优化器、调度、游标、RNG、初始化和最佳状态；`best-adapter.pt` 明确不可作完整 resume |
| 检查点兼容修复 | `ops/recover_portable_checkpoint.py`、`training.py`；`test_portable_recovery.py` | CPU 张量/标准容器，weights_only 加载；检查源码/模型/编码/子空间证据，不用任意 allowlist 掩盖错误 |
| NPU 冒烟与 BF16 | `smoke.py`、`ops/recover_prefill_smoke.py`；`test_smoke_prefill.py`、`test_smoke_recovery.py` | 同形状因果可见性精确相等；跨形状 BF16 另记录差异和 argmax；cache/no-cache 与重载检查仍有效 |
| E1 有效性 | `runner.core_panel`、`workflow.run_core`；`test_control_workflow.py` | BASE、训练表示、同范数随机方向、未训练初始表示；先确认效果，不能只看向量非零 |
| E2 几何关系 | `analysis.py`；`test_analysis.py` | LTV activation 与 LoDA parameter 分开；LoDA 对实际 BA 算几何，不能直接拼 A/B 算 cosine |
| E3 迁移 | `runner.core_panel`、`analysis.py`；`test_workflow.py` | 所有 source→target 有向关系，目标提示固定，分任务指标对 BASE 的变化；discovery 发现、test 确认 |
| E4 共享/残差干预 | `analysis.py`、`runner.component_panel`；`test_analysis.py` | mean 与归一化列 SVD，候选 q=0/1/2；shared/residual/reconstruction/delete alias/random；原 BA 精确组合，不悄悄压 rank |
| E5 组合 | `joint.py`、`runner.composition_candidates`；`test_joint.py` | 和/均值/调权/分解，相同候选预算与范数对照；原任务保留面板及双任务联合输入，分任务评分和联合成功率 |
| E6 未见任务 | `workflow.run_heldout_discovery`、`runner.py`；`test_workflow.py` | 四核心任务轮流留出，source-only 重建 LoDA/训练；SAMSum 不参与选择；不能拿删除旧向量冒充留出 |
| 扩展、消融、顺序 | `workflow.py`、`sequential.py`；`test_sequential.py`、`test_workflow.py` | 六任务、LTV 6/22/6+14+22、四循环顺序；formal 只评已训练结果，不偷偷重新训练 |
| 指标与统计 | `metrics.py`、`bootstrap.py`（加速版）、`analysis.py`；`test_metrics.py`、`test_analysis.py` | corpus BLEU/chrF/iBLEU 重算、语义模型指纹、代码真实测试、group 配对 bootstrap；不平均异质指标得“总分” |
| 官方合并对照 | `baselines.py`、`third_party/`；`test_baselines.py` | TIES/TSV/KnOTS 固定源码来源；参数空间适用，独立报告身份 |
| 工程并行加速 | 加速版 `parallel_evaluation.py`、`acceleration.py`、`batching.py`、`training_pool.py`、`sequential_pool.py`、`bootstrap.py`；`test_accelerated_*.py`、`test_parallel_evaluation.py` | 增大推理 batch、八卡分片、同阶段独立任务池、CPU 统计加速；训练 micro/effective batch 和方法未改 |

## 先看哪些训练证据

向量只表示训练后的作用，不包含所有训练历史。若导出包带有 `training.json`、`epochs.json`、`steps.jsonl`、配置与基座契约，应一起核对：

- `training.json` 的 `status=complete`、`completion_schema=2` 与 `artifacts` 逐文件 SHA；raw/post/initial 的 task、seed、route、base_fingerprint 和训练 request signature 是否一致。
- `completed_epochs`、`stop_reason`、`best_nll` 和逐轮 tune NLL，判断为什么提前停止、实际选了哪轮；LTV/LoDA 不应机械要求相同 epoch。
- `provenance.training_sources_sha256` 与本仓库 `provenance/source-snapshots.json` 的训练 Python 包摘要；基座、原样本/编码、子空间和科学超参数签名也需核对。
- `best_checkpoint_resume_supported`，以及完整 checkpoint 是否真的存在；仅有向量和 adapter 时不得声称已恢复原 optimizer/RNG。
- 日志中的设备数、实际显存、训练耗时和 NPU 利用率只按对应运行记录解释。Notebook 的一张 NPU 不代表训练 Worker 的八张卡。

旧评测已经看过的 test 不会因为换机器重新变成未见确认集。重新评测应记录新运行与硬件/推理 batch；跨 NPU/CUDA 允许出现 BF16 数值差异，不能承诺所有 token、score 或总耗时与原机逐位一致。vectors-only 可以继承训练效果，但单独它不能保证恢复完整科学工作流状态；固定划分、统计/子空间、锁与来源记录需要按续跑入口要求补齐。

## 建议审查顺序

先读原文和补充协议，再从 `cli.py` → `workflow.py` → `runner.py` 跟踪阶段，沿上表进入方法模块。对完整范围的需求，可继续查看快照 `docs/需求追踪与验收矩阵.md`（R01–R68 / I01–I10）、`docs/实现覆盖审计.md`、`docs/一致性修复与剩余差距.md`。加速实现另外阅读 `accelerated/docs/加速版部署指南-20261013.md` 和 `accelerated/docs/加速版验证报告-20261013.md`。历史文档的日期和局限应保留，不以本仓库上传成功替代正式实验验证。
