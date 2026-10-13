# Multi-Task Vector：LTV 与 LoDA-MT 全实验代码

本项目检验不同任务的表示是否包含**能够迁移、能够被干预验证、能够用于组合及未见任务的共享功能成分**。几何相似或任务矩阵低秩本身不构成这一结论。实验依据为 [原始实验方案](docs/原始实验方案.md)，SHA256 为 `D689BC2CB01A9DFE88B0F42CE2B385AC43AC0A2633DEA53E582882E3CCFD2A5C`；补充选择见 [补充协议](docs/补充协议与实现选择.md)。

实现包括四任务 MVP、六任务扩展、三 seed、E1–E6、LTV 层与多层消融、LoDA raw/post/分支功能、四个循环顺序对照、TIES/TSV/KnOTS、正式大测试面板。每阶段的入口均已实现。有效性门控失败会保存证据并停止依赖阶段，不会自动降低科学标准或宣称实验成功。

本地 CPU 单元测试和小模型验证覆盖公式与接线；本地没有 Ascend NPU，真实 Qwen3-1.7B 的 NPU 显存、算子兼容、完整训练和科学结论需要在 ModelMate 实测。

## 在 ModelMate 上运行

平台：用户提供 Ascend 910B 八卡、CANN 8.5.0-a3、PyTorch/torch_npu 2.8.0。用户描述的“八卡显存共64G”保留在配置中，`preflight` 读取实际设备和容量。所有数据、模型、checkpoint 与日志放在 `/opt/dpcvol/datasets/`。本项目默认写入独立命名空间 `/opt/dpcvol/datasets/task-vector/`。

先解压源码包并进入项目目录，按 [部署说明](deploy/README_ModelMate.md) 安装非平台依赖。安装脚本保留镜像自带 torch/torch_npu，不安装 CUDA、FlashAttention、bitsandbytes，也不依赖 vLLM 或 verl 进行本实验训练。

```bash
bash deploy/setup_modelmate.sh
export TV_MODEL_PATH=/opt/dpcvol/datasets/你的Qwen3-1.7B目录
export TV_SEMANTIC_MODEL_PATH=/opt/dpcvol/datasets/你的all-MiniLM-L6-v2目录
task-vector preflight --config configs/experiment.yaml
task-vector assets --config configs/experiment.yaml --hash-weights
```

在 `configs/experiment.yaml` 中为每个 `data.sources.<task>` 填写本地 `path`，或按模块文档提供明确官方 split 文件映射。离线 HF `save_to_disk` 目录、原始 JSON/JSONL 和 ProofWriter 官方 theory 文件均有读取适配。远程来源须显式 `allow_remote: true`，来源 revision、split、文件 SHA 与最终 ID 全部记录。不要将无 split 来源的混合文件冒充官方训练/测试划分。

```bash
# 完整研究使用同一份包含全部七任务的准备清单，后续不再改动。
task-vector prepare --config configs/experiment.yaml --scope all
task-vector inventory --config configs/experiment.yaml
task-vector smoke --config configs/experiment.yaml --device npu:0

# 完整流程：严格首轮256 -> 全部主评测/对照 -> 全部正式扩评。
task-vector run --config configs/experiment.yaml --phase all --split test_formal --device npu:0
task-vector report --config configs/experiment.yaml
```

推荐按 [Notebook 上传到作业提交指南](ModelMate_操作指南.md) 使用 `run_modelmate_check.py` 后接 `run_modelmate_full.py`；full 自动衔接上述全部阶段，full_formal 是兼容别名。首 seed 的八个独立训练可用八卡并行，每个任务仍保持 micro2、有效 batch16；后续阶段由流程自动衔接。统计和训练目录使用文件锁，完整逻辑 step 后保存 optimizer/scheduler/RNG/最佳点与下一批位置。恢复同一目录时不重置 seed，也不把 GAO 的两次 optimizer step 算成一次。

只做第一阶段时可以先 `prepare --scope core`（四任务）、`run --phase core`。`prepare --scope mvp` 还准备 E6 所需 SAMSum，能够执行 MVP 的最终面板。扩展准备会改变数据清单，不能继续使用已经锁定的旧运行目录；应先准备完整数据，或给新的完整实验设置独立 `platform.persistent_root`。

## 分阶段命令与结果

```bash
task-vector run --config configs/experiment.yaml --phase core
task-vector run --config configs/experiment.yaml --phase discover
task-vector run --config configs/experiment.yaml --phase lock
task-vector run --config configs/experiment.yaml --phase final
task-vector run --config configs/experiment.yaml --phase heldout
task-vector run --config configs/experiment.yaml --phase ablations
task-vector run --config configs/experiment.yaml --phase extended
# 对全部已完成的主实验、四折、六任务及层/顺序对照做纯正式扩评。
task-vector run --config configs/experiment.yaml --phase formal --split test_formal
```

`test_formal` 可以包含原256条，是同一锁定方案的扩评。若查看256条后修改方法，需新运行和未解封的 `test_confirmation`；准备程序按完整 group 从官方评测池排除原256条，再保留独立确认面板。解封 ledger 记录实际使用，不能将看过的数据重新称为独立确认。

发现集选择 q 与任务组；调参集选择 checkpoint、干预强度和同预算组合候选。首轮使用 train/tune-only core lock，完成后保存独立9216计数记录；最终 `lock` 固定配置、数据、源码、选择结果、训练表示和候选实际文件 SHA。最终测试会检查锁定一致性，拒绝在测试集上重新选秩、选系数或拟合判据。完整阶段可自动连续执行；已有锁定发现结果及同条件预测直接复用，正式面板只生成新增内容并重算完整指标。

结果目录主要内容：

| 目录/文件 | 证据 |
|---|---|
| `data/manifest.json`、各任务 JSONL | 官方来源、分组划分、ID、token 长度和拒绝理由 |
| `runs/preflight.json`、`assets.json`、`smoke.json` | 实际环境、模型指纹、训练/推理一致性 |
| `runs/training/` | 初始化、raw/post、最佳/最后 checkpoint、NLL 曲线、逐步日志 |
| `runs/statistics/` | 校准 ID/token 位置、各层 token 二阶矩与来源 |
| `runs/evaluation/` | 完整输出、截断、代码执行、逐样本指标和 corpus 指标签名 |
| `runs/analysis/` | 首轮计数、原单位迁移矩阵、发现到测试关系确认、共享/随机直接配对区间、聚类、成分、组合、留出结果 |
| `runs/protocol/` | 发现决策及锁定文件 |
| `runs/sequential/` | 与 LoDA-MT 分开的顺序模型、累计更新、合并舍入和遗忘证据 |
| `runs/report.md` / `.json` | 依据已有证据生成的执行报告 |

首阶段 `9216` 是 9 个主状态 ×4 个任务 ×256 条；初始化、随机对照、发现集、调参、重标定、格式、组合与后续扩展均另计，完整预算见 `inventory`。

## 核心实现保证

- LTV 在整个 decoder block 输入、RMSNorm 前的固定 prompt 末 token 注入；生成只在 prefill 注入；FP32 参数、完整回答监督和非重入梯度检查点采用相同语义。
- LoDA-MT 的 Q/K/V 共享输入子空间并保留不同输出宽度、原生 GQA/QK norm/RoPE/cache；只训练 FP32 B。A、白化、GAO 和重标定公式都有独立数值测试。
- 损失先按每个样本回答 token 平均，再按样本平均；完整答案包含 EOS；超长拒绝，不截断后训练。
- 参数几何与分解使用真实 BA，因子换基不改变结论；组合可增加矩阵秩，保存时不偷偷压回 rank16。
- 迁移保留准确率、chrF++、iBLEU 等原单位；聚类才用基座句级标准差标准化；配对 bootstrap 按 theory/重复问题组抽样，并重新计算 corpus 指标。
- E5 同时评原任务能力面板和双任务联合输入，保留各任务指标；连续指标联合达标判据必须事先在调参集定义并锁定。
- E6 中 LoDA 排除被留出任务的统计、重新构 A 并训练剩余 B；不会复用含被留出任务统计的主模型。SAMSum 不参与任何训练或选择。

## 代码测试

```bash
python -m pytest -q --basetemp outputs/test-temp
python -m task_vector.cli inventory --config configs/experiment.yaml
```

测试采用本地随机 tiny Qwen3 和可人工核算的矩阵/程序，不下载正式模型或用测试样本冒充正式数据。源码参考快照和许可证在 `third_party/`，其来源与改造边界在对应说明文档中。

MBPP 的 300/37/37 恰好用满官方 train374；若验证或长度过滤后容量不足，默认严格失败并保存拒绝记录。任何缩减数量或放宽长度须作为明确补充协议配置，不能混入 test 填数。
