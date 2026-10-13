# Multi-task vector experiments

完整的 LTV / LoDA-MT 实验实现、独立向量分析工具、训练审查资料和续跑入口。对应 Qwen3-1.7B 冻结基座。

## 拿到代码后能做什么

| 手上资料 | 可以进行的工作 |
|---|---|
| 本仓库 | 审查原方案、补充协议、所有训练/评测/分析代码与工程修复；运行 CPU 测试 |
| 本仓库 + 独立向量包 | 校验向量和来源记录，查看形状/范数/训练摘要，计算同路线 Gram/余弦、均值和 SVD 共享/残差 |
| 再加相同 Qwen3 基座 | 使用已有向量推理；在已有评测数据上重新评测 |
| 再加原固定数据划分 | 核验训练源码及数据后继承已完成训练，重新评测，并继续未完成任务/种子/后续阶段 |
| 完整迁移包 + 相同基座 | 使用原 Worker 代码快照、全部固定划分和缓存完成严格迁移 |

独立包里的 `post.pt` 是主结果，`raw.pt` 是重标定前版本，`initial.pt` 是未训练对照。首轮两条路线各有四项任务，共八项训练结果。已有向量不包含旧 optimizer/RNG 状态，不能恢复旧训练中的某个优化器步骤。

## 仓库结构

```text
baseline/        当前旧版完整实现：方案、代码、数据适配、评测、测试、第三方快照
accelerated/     加速版完整实现：推理批量/分片、任务池、统计加速、部署与验证
portable/        导出、加载、完整迁移、目标环境续跑工具
tools/           独立向量分析、固定数据接续、源码摘要核验
operations/      ModelMate 平台入口、监控及环境/资产准备工具
docs/            远端分析、代码审查、续跑和来源范围说明
provenance/      两版科学源码的 SHA256 清单及发布验证记录
tests/           远端分析与续跑桥接的 CPU 测试
```

`baseline/` 与 `accelerated/` 分别保留完整版本，不混合改写实验实现。原始设计在 [原始实验方案](baseline/docs/原始实验方案.md)，已接受的数值和范围选择在 [补充协议](baseline/docs/补充协议与实现选择.md)。原有 README/部署指南中的旧路径属于原运行位置；本页及 `docs/` 是远端接入入口。

## 1. 下载与准备

```bash
git clone https://github.com/ligoldfish/multi-task-vector-experiments.git
cd multi-task-vector-experiments
python tools/verify_sources.py
```

私有仓库需先用自己的获授权 GitHub 账号登录。也可以从仓库网页 Download ZIP；源文件摘要按内容校验。使用 Python 3.11 和适合目标硬件的 torch 2.8 环境；NPU 另需匹配 CANN/驱动的 torch_npu。仅查看源码和校验源码摘要无需安装模型或 PyTorch。

CPU 张量分析使用已安装 torch 的环境，再安装实验依赖：

```bash
python -m pip install -r portable/requirements-portable.txt
```

不要对冻结的 `baseline/` 或 `accelerated/` 执行原地 `pip install`，构建工具可能在源码里增加 egg-info 文件并改变来源摘要。工具会直接从对应 `src/` 导入模块。模型、固定数据和运行配置应放在仓库外部。

## 2. 只拿独立向量包做分析

不用模型、不用原数据、不用 NPU：

```bash
python tools/analyze_vectors.py \
  --bundle /assets/task-vectors-only-实际时间.zip \
  --output /results/vector-analysis-01
```

工具会核对归档和向量 SHA256、原训练完成记录、表示摘要及元数据，再在 CPU 上输出结构检查、训练摘要和同路线几何分析。LoDA 按实际参数更新 `sum(B @ A)` 计算几何，不能把 A/B 拼起来当作向量。LTV 和 LoDA、不同 seed/层/训练范围分别分析。

主报告为输出目录中的 `analysis.json`。加 `--save-decomposition` 可保存均值/SVD 分解的共享与残差 `.pt`；只检查张量和来源时使用 `--skip-geometry`。默认分析 `post`，也可指定 `--version raw` 或 `--version initial`。这些几何结果不是已经通过评测的最终共享方案。

分析结果只能说明张量结构、来源一致性和几何关系。向量是否提升任务表现、迁移是否有效，需要相同基座上的固定数据评测，不能从余弦相似度直接推断。详见 [代码与训练审查](docs/CODE_AND_TRAINING_REVIEW.md)。

独立加载只需 PyTorch 与 `portable/use_task_vector.py`：

若只收到一份 `.pt`，可先查看内部任务、seed、形状和精度：

```bash
python portable/use_task_vector.py --inspect /vectors/post.pt
```

完整来源和跨任务几何分析使用上面的独立包入口；单份 `.pt` 缺少包内校验清单及其他任务向量。

```python
import sys
sys.path.insert(0, "/path/to/repository/portable")
from use_task_vector import load_plain, task_vector_context, verify_base_directory

payload = load_plain("/path/to/post.pt")
verify_base_directory("/models/Qwen3-1.7B", "/vectors/BASE_MODEL_CONTRACT.json")
with task_vector_context(model, payload, positions=original_prompt_positions):
    output = model.generate(...)
```

LTV 注入 Transformer block 输入、RMSNorm 之前，在原 prompt 最后有效 token 处且仅 prefill 生效。LoDA 保留 FP32 旁路更新；`post.pt` 已包含重标定，不要再乘 lambda 或 LoRA alpha/r。这两种文件不是标准 PEFT adapter。

## 3. 继承已完成训练，继续后续实验

如果有完整迁移包，先解压到新位置：

```bash
python portable/portable_import.py \
  --bundle /assets/full-bundle-unpacked \
  --destination /experiments/imported-01 \
  --model /models/Qwen3-1.7B \
  --device cuda:0

python portable/portable_run.py --destination /experiments/imported-01 inventory
python portable/portable_run.py --destination /experiments/imported-01 preflight
python portable/portable_run.py --destination /experiments/imported-01 smoke
python portable/portable_run.py --destination /experiments/imported-01 all --scope mvp --split test_formal
```

NPU 将设备改为 `npu:0`。完整迁移使用包中原 Worker 源码，保持原向量和准备数据字节不变，重新形成目标环境的 smoke、评测和协议锁。已完成训练核验后复用，未训练的任务、种子、层和 leave-one-out 方案仍需按原流程训练。

如果只有独立向量包，还需提供原固定数据目录和目标配置，走 [独立向量接续说明](docs/RESUME_FROM_VECTORS.md)。工具检查训练 Python 源码摘要和原数据摘要；不会把路径变化伪装成原训练，也不会拿新抽样数据冒充原固定划分。

目标硬件、库、batch 或解码器变化可能产生正常 BF16 生成差异，目标环境需要重新评测。同一已曝光 test 集的重评，不重新获得独立确认资格。

## 4. 审查与验证

- [完整代码范围与来源边界](docs/SOURCE_COVERAGE.md)
- [代码与训练流程审查入口](docs/CODE_AND_TRAINING_REVIEW.md)
- [独立向量接续与必要资料](docs/RESUME_FROM_VECTORS.md)
- [原有需求追踪与验收矩阵](baseline/docs/需求追踪与验收矩阵.md)
- [加速版工程边界和部署](accelerated/docs/加速版部署指南-20261013.md)
- [加速版已有 CPU 验证证据](accelerated/docs/validation/acceleration-evidence.json)

已有验证报告有各自明确范围；CPU 测试通过不等于真实 NPU/CUDA 实验结果成立，也不证明加速吞吐。新增工具的发布验证见 `provenance/publication-validation.json`。
