# ModelMate Ascend 运行说明

本代码面向用户指定的 Ascend 910B 八卡镜像：CANN `8.5.0-a3`、PyTorch / torch_npu `2.8.0`，vLLM-Ascend `0.13.0`、verl `0.7.1`。训练与生成使用 Transformers `4.51.3` 原生 Qwen3，保持 LTV 指定层输入干预语义。vLLM 和 verl 不参与本实验执行。

用户报告“八卡显存共 64G”。这是待验证的资源信息；`preflight` 写入训练容器实际可见卡数、每卡容量及 `npu-smi`，`smoke` 写入真实模型的峰值显存。Notebook 的资源与训练容器不同，应在实际训练容器里运行这些检查。

## 1. 放置代码并安装附加依赖

把完整 `task-vector` 项目放入持久化工作区。模型、原始数据、语义评分模型、划分清单与结果均需位于 `/opt/dpcvol/datasets/` 的明确子路径。代码不搜索猜测现有模型和数据目录。

在镜像终端执行；若镜像尚未激活 CANN，先把 `TV_CANN_ENV` 设置为该镜像实际存在的激活脚本路径：

```bash
cd /opt/dpcvol/datasets/task-vector-code
export TV_PERSISTENT_ROOT=/opt/dpcvol/datasets/task-vector
bash deploy/setup_modelmate.sh
```

安装脚本保存安装前后框架版本、精确约束和 `pip freeze`，只补充项目依赖；不会安装 CUDA、替换 torch 或 torch_npu。若当前框架不是 2.8 系列或 NPU 不可见，会立即停止。环境证据保存在 `$TV_PERSISTENT_ROOT/environment/`。

## 2. 配置真实资产路径

复制 `configs/experiment.yaml` 为自己的运行配置，例如 `configs/modelmate.local.yaml`。填写：

- `model.path`：包含 Qwen3-1.7B `config.json`、tokenizer 与所有权重分片的本地目录。
- `data.sources.<task>.path`：每任务的本地原始数据或 Hugging Face `save_to_disk` 目录。原始 JSON/JSONL 必须保留官方划分来源；不能把自制训练划分标为官方测试。
- `metrics.semantic_model_path`：完整的 `sentence-transformers/all-MiniLM-L6-v2` 本地目录，用于 QQP 语义保持补充指标。
- `platform.persistent_root`：本次实验的独立持久化根目录，默认 `/opt/dpcvol/datasets/task-vector`。
- `platform.device`：单进程所用卡，例如 `npu:0`。

扩展阶段还需填写 MBPP 和 SAMSum 的本地来源。代码修复构造会验证原程序通过、错误程序失败；数据准备会拒绝缺少这些证据的样本。

```bash
python -m task_vector.cli assets --config configs/modelmate.local.yaml
python -m task_vector.cli preflight --config configs/modelmate.local.yaml
python -m task_vector.cli prepare --config configs/modelmate.local.yaml
python -m task_vector.cli smoke --config configs/modelmate.local.yaml
```

`assets` 检查本地路径与模型分片；`prepare` 完成固定划分、长度和组级防泄漏审计。`smoke` 的LTV/解码检查使用两条训练样本；LoDA使用16条当前任务train，拆成8+8工程校准半池，真实执行micro2、有效16的一次双向GAO（两次AdamW/scheduler），再恢复checkpoint并确认没有多做一步、状态和raw/post logits一致。校准半池不代表科学S_t，不读tune/test或其他任务；合成零/非零导出检查仍保留。全部工程产物独立保存，不能用于E1–E6科学结论。

`prepare` 默认 `--scope all`，一次准备四个主任务、两个代码任务和摘要任务，便于后续冻结同一份 manifest。若当前只有四任务资产，可先使用 `prepare --scope mvp`；加入代码或摘要时应选择新的持久化实验根目录并重新准备完整范围，保留原运行的 manifest 和锁定产物。不得在已锁定的运行中覆盖数据划分。

如需完整权重 SHA256 清单，可独立运行：

```bash
python -m task_vector.assets --config configs/modelmate.local.yaml --output /opt/dpcvol/datasets/task-vector/assets-full.json --hash-weights
```

## 3. 单卡完整流程或八卡独立训练池

每个实验实例始终保持 micro-batch 2、effective batch 16。八张卡用于独立训练实例；不会把一个实验自动改为八卡数据并行。

单卡执行首轮工程检查、四任务训练和发现集 E1/E3：

```bash
bash deploy/run_modelmate.sh configs/modelmate.local.yaml core
```

若已完成 `prepare` 与 `smoke`，可在已分配的八卡训练容器里并行训练 seed 17 的八个独立实例：

```bash
python deploy/launch_pool.py --config configs/modelmate.local.yaml --scope mvp --seed 17 --devices 0,1,2,3,4,5,6,7
python -m task_vector.cli run --config configs/modelmate.local.yaml --phase core
```

第二条命令复用已完成的训练产物，再执行核心面板。池脚本只启动当前容器内的子进程，不提交 ModelMate 网页作业。日志、配置、PID、退出码和调度清单保存在持久化根目录的 `scheduler/`。首次失败后停止启动新任务，等待已运行任务退出并保留检查点。存在池锁时拒绝重复启动，不自动删锁。

后续阶段由正式流程门控：

```bash
python -m task_vector.cli run --config configs/modelmate.local.yaml --phase discover
python -m task_vector.cli run --config configs/modelmate.local.yaml --phase lock
python -m task_vector.cli run --config configs/modelmate.local.yaml --phase final
python -m task_vector.cli run --config configs/modelmate.local.yaml --phase ablations
python -m task_vector.cli run --config configs/modelmate.local.yaml --phase extended
python -m task_vector.cli run --config configs/modelmate.local.yaml --phase heldout
python -m task_vector.cli report --config configs/modelmate.local.yaml
```

`discover`、`lock` 和 `final` 分别承担发现选择、协议冻结与最终确认。最终测试不能用于选择秩、分组或组合系数；功能门控未通过时，依赖它的后续科学阶段停止并记录原因。`all` 可作为统一入口，仍遵循相同门控。

## 4. 恢复与资源限制

训练保存 `last.pt` 的优化器、调度器、RNG、最佳参数和逻辑 batch 位置。相同配置与样本 ID 的重跑会恢复已有完整逻辑步；中途某个 GAO 方向未完成时，从上一完整逻辑步重新执行双向更新。配置或样本不一致时拒绝恢复。

LoDA 每逻辑 batch 分为两组，采用官方顺序的两次 AdamW 更新；微批即时反传，不保留整组八样本计算图。MVP 的 512 条可整除 16；MBPP 的 300 条保留最后 12 条，并分为两组各 6 条，不丢样本、不重复补样本。

单卡显存不足时先查看 `failure.json`、`smoke.json`、`preflight.json` 和峰值记录；保持 BF16、micro-batch、有效 batch、序列上限和实验超参数。不得为了运行成功而静默改成更小模型、更短答案或缺失任务。当前工程采用非重入 checkpoint、仅末 token 生成 logits，以及分块重算交叉熵降低临时显存；真实可运行性仍需平台 smoke 验证。

本地 CPU tiny Qwen3 测试与真实 NPU 验证分开记录。环境检查通过、工程 smoke 通过、训练完成、产物完整和科学效果成立，是不同的证据状态。
