# Multi-Task Vector Experiments

研究语言模型在不同任务中学到的轻量更新，是否包含可以跨任务复用的共享成分。

项目使用冻结的 **Qwen3-1.7B**，分别训练激活空间的 **Learned Task Vector（LTV）** 与参数空间的 **LoDA-MT** 更新。随后通过迁移评测、共享/残差分解、成分删除与组合实验，检验向量的几何关系是否具有实际功能意义。

代码覆盖训练、数据划分、生成评测、统计分析、向量导出及后续实验。主要运行平台为 Ascend 910B；CPU 可用于测试和向量分析。CUDA 设备入口也可用，运行实验前需在目标环境完成模型冒烟检查。

## 实验研究什么

| 实验 | 核心问题 |
|---|---|
| 有效性 | 训练得到的任务向量，是否优于冻结基座、初始化向量和同范数随机向量？ |
| 几何关系 | 不同任务向量之间的相似结构是否稳定？ |
| 迁移 | 一个任务的向量，能否改善另一个任务的表现？ |
| 成分干预 | 分解得到的共享成分能否单独发挥作用，删除后是否造成相应损失？ |
| 组合 | 向量组合是否优于公平预算下的加和、平均与调权对照？ |
| 未见任务 | 仅由已见任务确定的结构，能否迁移到未参与构建的任务？ |

核心任务包括 GSM8K 数学推理、ProofWriter 逻辑推理、IWSLT2017 英德翻译和 QQP 改写。扩展实验包含 MBPP 代码生成/修复、SAMSum 摘要、层消融、顺序训练和 TIES/TSV/KnOTS 合并对照。几何相似度本身不构成共享能力的证据；结论需要受控评测支持。详见 [实验方法](docs/METHODS.md)。

## 安装

使用 Python 3.11，以及适合目标硬件的 PyTorch 2.8。Ascend 还需要匹配的 `torch_npu`、CANN 和驱动；已有可用框架时，直接安装其余依赖：

```bash
git clone https://github.com/ligoldfish/multi-task-vector-experiments.git
cd multi-task-vector-experiments
python tools/verify_sources.py
python -m pip install -r requirements.txt
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
python -m task_vector.cli --help
```

以上为 Bash 命令。PowerShell 设置模块路径可用 `$env:PYTHONPATH = "$PWD\src"`。项目通过 `src/` 直接导入，方便保留训练来源校验；模型、数据和实验输出建议放在代码目录之外。

## 从头运行实验

先准备 Qwen3-1.7B、`sentence-transformers/all-MiniLM-L6-v2` 和对应官方数据。在 `configs/experiment.yaml` 的副本中填写设备、模型、数据和输出路径，并配置参与计算的设备。完整示例见 [运行指南](docs/RUNNING.md)。

```bash
python -m task_vector.cli preflight --config /work/experiment.yaml
python -m task_vector.cli prepare --config /work/experiment.yaml --scope all
python -m task_vector.cli smoke --config /work/experiment.yaml
python -m task_vector.cli run --config /work/experiment.yaml --phase all --scope mvp --split test_formal
```

`mvp` 表示四个核心任务的首轮研究范围；`all` 阶段会按依赖关系执行后续实验，并在科学门禁不满足时记录结果、停止依赖该结论的阶段。需要单独训练一个任务时：

```bash
python -m task_vector.cli train --config /work/experiment.yaml --route ltv --task math --seed 17
python -m task_vector.cli train --config /work/experiment.yaml --route loda --task math --seed 17
```

实现支持独立训练任务并行、生成评测按设备分片，以及仅使用 `tune` 样本选择推理 batch。训练的 micro batch、有效 batch 和优化方法按实验协议保持固定。BF16 下推理 batch 的形状变化可能带来少量 token 差异，运行记录会保存采用的策略。

## 使用已训练向量

`.pt` 文件保存轻量更新，不包含完整基座模型。无需加载模型即可查看任务、seed、张量形状和精度：

```bash
python portable/use_task_vector.py --inspect /artifacts/post.pt
```

对带有文件清单及训练来源记录的向量归档，可以校验内容并计算范数、Gram/余弦、均值与 SVD 共享/残差分解：

```bash
python tools/analyze_vectors.py --archive /artifacts/vectors.zip --output /results/vector-analysis --save-decomposition
```

结果写入 `analysis.json`，可选保存分解得到的 `.pt`。分析在 CPU 上进行，不需要基座权重或原数据。LTV 与 LoDA 分开分析；LoDA 几何基于实际参数更新 `sum(B @ A)`，不将低秩因子直接拼接作为向量。

在模型上使用向量，需要与训练时内容相同的 Qwen3-1.7B。加载示例、文件结构及 `raw`/`post`/`initial` 的含义见 [向量使用指南](docs/VECTORS.md)。已有结果的核验、重新评测与后续训练见 [恢复与继续实验](docs/RESUME.md)。

## 代码结构

```text
src/task_vector/  模型干预、训练、生成评测、分解、统计与实验调度
configs/          实验协议配置
tests/            方法实现与工程正确性测试
tools/            向量分析、来源核验和结果接续工具
portable/         轻量向量加载、导出与迁移入口
deploy/           多设备任务调度和 ModelMate 部署
ops/              运行环境配置、检查与恢复工具
docs/             方法、运行、向量格式与复现实验说明
third_party/      固定版本的上游参考实现及来源清单
provenance/       发布源码摘要与验证记录
```

训练流程主要从 `training.py`、`ltv.py`、`loda.py` 和 `gao.py` 阅读；实验编排从 `workflow.py`、`runner.py` 阅读；数据与评分分别位于 `data.py` 和 `metrics.py`。

## 测试与可复现性

```bash
python -m pip install -r requirements-test.txt
python -m pytest tests -q
python -m unittest discover -s portable -p 'test_*.py' -v
```

每次运行记录基座、数据、配置、源码及向量摘要。训练、调参、结构发现和测试分别使用固定划分；重新评测已经使用过的测试集，不会使它重新成为独立确认集。CPU 测试检查实现与产物一致性，实际模型正确性及吞吐需要在目标硬件上验证。

实验协议见 [原始方案](docs/原始实验方案.md) 和 [补充协议](docs/补充协议与实现选择.md)。模型权重、数据、训练向量和优化器检查点由使用者另行准备，不存放在本仓库。第三方参考文件保留各自来源与已有许可证，具体见 `third_party/`。
