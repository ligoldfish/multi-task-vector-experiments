# 运行实验

以下命令在仓库根目录执行，使用 Bash。路径仅为示例，应换成目标机器上的实际位置。开始前先安装与硬件相符的 PyTorch；Ascend 使用已配置好 CANN 与 `torch_npu` 的环境。

## 模型和数据

需要 Qwen3-1.7B 的完整模型目录，包括权重、配置和 tokenizer，以及 `sentence-transformers/all-MiniLM-L6-v2` 评分模型。数据适配器支持 Hugging Face `DatasetDict.save_to_disk()` 目录、明确声明官方 split 的 JSON/JSONL/CSV/TSV/Parquet，以及 ProofWriter、IWSLT、MBPP 的原始格式。

| 配置任务 | 数据来源 | 训练来源 | 评估来源 |
|---|---|---|---|
| math | GSM8K | train | test |
| logic | ProofWriter OWA，depth 0–3 | train | test |
| translation | IWSLT2017 en-de | train | test |
| paraphrase | GLUE QQP | train | validation |
| code_generation | MBPP full | train | test |
| code_repair | 同一份 MBPP full，按程序 ID 构造修复任务 | train | test |
| summary | SAMSum | train | test |

ProofWriter 使用原始 `meta-*` 记录，保留推理深度约束。MBPP 保留官方程序 ID、setup 和测试；不要将 sanitized 子集替代 full 数据。具体加载逻辑在 `src/task_vector/data.py`。

## 配置路径与设备

复制完整模板到实验目录，再编辑其中已有字段：

```bash
mkdir -p /work/task-vector
cp configs/experiment.yaml /work/experiment.yaml
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
```

例如以下内容表示需要修改的字段，**不是用来替换整个配置文件**：

```yaml
platform:
  device: npu:0
  persistent_root: /work/task-vector
  expected_devices: 8
model:
  path: /models/Qwen3-1.7B
metrics:
  semantic_model_path: /models/all-MiniLM-L6-v2
data:
  sources:
    math: {path: /datasets/gsm8k}
    logic: {path: /datasets/proofwriter-owa}
    translation: {path: /datasets/iwslt2017-en-de}
    paraphrase: {path: /datasets/qqp}
    code_generation: {path: /datasets/mbpp}
    code_repair: {path: /datasets/mbpp}
    summary: {path: /datasets/samsum}
```

保留每个来源原有的 `dataset_id`、`config`、`train_split` 和 `eval_split`。如果提供分离的原始文件，可在对应任务来源中将 `path` 设为 `null`，并声明：

```yaml
files:
  train: /datasets/gsm8k/train.jsonl
  test: /datasets/gsm8k/test.jsonl
```

七个任务都要配置，才能运行全部实验。`prepare` 会按官方 split 筛选完整样本、隔离 group、验证程序及长度，并保存固定 JSONL 与 manifest。容量不足会报告原因，不通过截断答案或补入测试样本凑数。

## 设备并行与推理 batch

模板已包含以下工程配置，默认使用八张 NPU；按实际分配的设备修改：

```yaml
acceleration:
  enabled: true
  evaluation_devices: [npu:0, npu:1, npu:2, npu:3, npu:4, npu:5, npu:6, npu:7]
  generation_batch_policy: validated
  generation_batch_candidates: [1, 2, 4, 8]
  calibration_max_samples: 8
  worker_cpu_threads: 2
  training_cpu_threads: 8
  bootstrap_workers: 1
  heartbeat_seconds: 30
  parallel_training: true
  parallel_sequential_orders: true
```

设备编号是容器内可见编号。单张 NPU 将设备列表缩为 `[npu:0]`。CUDA 将 `platform.device` 和设备列表改为相应的 `cuda:N`，并把 `parallel_training`、`parallel_sequential_orders` 设为 `false`：这两个训练池目前要求 NPU。生成评测可以使用 CUDA 设备列表。CPU 建议用于测试与张量分析；若进行模型检查，应关闭设备并行。

`validated` 仅在 `tune` 样本上检查候选 batch 的有限数值、因果位置与生成行为，并记录所选 batch；允许正常 BF16 形状差异。`strict` 使用 batch=1。两者均不改变训练的 micro batch=2、有效 batch=16 或优化方法，也不使用 test 选择 batch。

## 执行顺序

```bash
python -m task_vector.cli preflight --config /work/experiment.yaml
python -m task_vector.cli assets --config /work/experiment.yaml --hash-weights
python -m task_vector.cli prepare --config /work/experiment.yaml --scope all
python -m task_vector.cli inventory --config /work/experiment.yaml
python -m task_vector.cli smoke --config /work/experiment.yaml
python -m task_vector.cli run --config /work/experiment.yaml --phase all --scope mvp --split test_formal
```

`preflight` 检查设备与持久化路径；`assets` 记录模型与来源文件；`prepare` 创建七任务固定数据；`smoke` 检查真实模型上的 LTV/LoDA 注入、因果可见性、缓存、重载与训练恢复。最后一条命令执行首轮、发现、锁定、确认及满足门禁的后续实验，并正式扩大评测范围。

完整流程有科学门禁。若必要效果或稳定性不足，会保留负结果，停止依赖该前提的实验；不能将这种停止解释为程序遗漏。状态与输出位于配置中的 `persistent_root`。

只训练一个任务时：

```bash
python -m task_vector.cli train --config /work/experiment.yaml --scope mvp --route ltv --task math --seed 17
python -m task_vector.cli train --config /work/experiment.yaml --scope mvp --route loda --task math --seed 17
```

LoDA 需要的基座统计和子空间会由流程生成。独立训练命令本身不会完成迁移、成分干预等科学验证。

## 输出与继续运行

主要输出包括 `data/manifest.json`、各任务固定划分、`runs/training/`、`runs/evaluation/`、协议锁和统计报告。训练目录中的 `training.json` 最后提交完成状态，绑定三个表示文件与来源摘要。`raw.pt` 是训练更新，`post.pt` 是主评表示，`initial.pt` 是未训练对照。

在来源、配置、基座和数据未改变时，重复实验命令会核验并复用已完成产物。移动到另一位置时使用 [接续工具](RESUME.md)，保留原来源记录。精确恢复一个未完成训练的优化器步骤，还需要完整的 `last.pt`/`best.pt`；轻量向量不包含这些状态。

## ModelMate

通用 CLI 可以直接在已分配的训练容器中运行。使用平台的 Python 启动文件时，先根据 `ops/paths.example.json` 配置实际持久化路径，运行 `ops/configure_local.py --help` 查看生成配置和运行设置的参数；该工具专用于八卡 ModelMate，要求资源位于 `/opt/dpcvol/datasets`。

`run_modelmate_full_formal.py` 对应按阶段执行全部流程并正式扩评；`run_modelmate_check.py` 用于环境和工程检查。平台入口不会自行创建云端训练资源。
