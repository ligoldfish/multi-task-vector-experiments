# 已训练 task vector 的导出与迁移

此工具只读取当前代码和实验目录，新建下载文件。它不停止训练、不删除文件、不改写原实验。

## 两种下载包

1. `task-vectors-only-时间.zip`：独立向量包。按路线、seed、任务存放原样的 `post.pt`、`raw.pt`、`initial.pt`，以及训练来源、基座文件校验清单和一个可选的最小注入工具。只拿某个任务的 `post.pt` 也可以用，无需安装完整实验项目。它是普通 CPU 张量字典，可用 `torch.load(..., weights_only=True)` 读取。
2. `task-vector-portable-full-时间.zip`：完整迁移包。包含当前完整源码、全部七任务的固定划分、已完成训练导出、统计与子空间、语义评分模型和迁移入口。可以复用训练结果，在目标环境重新评测，并按原有科学门禁继续后续阶段。

首轮通常是 LTV 四个任务、LoDA 四个任务，共八项训练结果。`raw/post` 是同一训练结果的版本，`initial` 是未训练对照，不是另外训练了三份。后续若已有其他训练完成，也会逐项导出。未完成训练不会被冒充为已完成。

完整包保留较大的 LoDA 统计与子空间，可能有数 GiB。它们不属于“仅使用向量”所需的文件；只要推理效果就下载第一个包。完整包不携带 Qwen 基座、Python/CANN 环境、旧优化器状态或有效的新评测结果。

## 在当前 Notebook 导出

把本工具包解压到任意新目录，运行：

```bash
bash /path/to/export-kit/export_portable.sh
```

默认读取：

- 代码：`/home/work/user-job-dir/app/task-vector-code-20261011T071308Z`
- 实验：`/opt/dpcvol/datasets/naie/task-vector/experiments/exp-20261011-01`
- 下载文件保存到：`/home/work/user-job-dir/app`

也可以显式指定：

```bash
bash /path/to/export-kit/export_portable.sh \
  --code /path/to/actual-worker-code \
  --experiment /path/to/original-experiment \
  --output /path/to/download-directory
```

导出会核对原基座、完整数据划分、训练提交记录以及向量 SHA256，并检查当前源码与训练来源一致。若当前代码已经换成另一版本，需使用实际训练时的代码快照，不能修改签名冒充一致。原始路径和记录保留为来源证据。

两个 ZIP 分别配有 `.sha256.txt` 文件。日志先出现 `task-vectors-only` 的 `DOWNLOAD_ZIP` 时，小包已可下载；完整包完成后出现 `EXPORT_READY=PASS`。任何失败的 `.building.zip` 都不是正式交付文件，工具会保留它以免自行删除。

## 只使用向量

解压独立向量包，阅读其中 `README.txt`。主结果是每个任务目录的 `post.pt`。

- LTV 必须注入对应 Transformer block 的**输入、RMSNorm 之前**，仅作用于原 prompt 最后一个有效 token；有 KV cache 时只在 prefill 注入。
- LoDA 按原生 Q/K/V 模块名保存 FP32 的低秩 A/B 因子，表示实际更新 `sum(B @ A)`；`post` 已做重标定。默认用 FP32 旁路计算并转回输出 dtype，避免 BF16 权重合并引入额外数值变化。
- `use_task_vector.py` 只依赖 torch，使用方式见独立包 README；不依赖实验项目、数据集或统计矩阵。已有自己的等价注入代码时可以完全忽略它。
- LTV 和 LoDA 是两种不同接口，不能把 LTV 当标准 LoRA adapter 加载。仅有相同维度或相同模型名称也不能保证兼容，需匹配相同权重和 tokenizer 等基座文件。

## 在另一台机器继续完整实验

使用 Python 3.10+；建议保持原 Python 3.11、torch 2.8 及实验依赖版本。目标是 CUDA 时配置对应的 PyTorch；目标是 NPU 时安装与驱动/CANN 匹配的 torch_npu。不能复制原 aarch64 NPU 虚拟环境到其他架构。原 `code/pyproject.toml` 列出了 Python 实验依赖；其安装不负责安装驱动或选择设备版 torch。

先解压完整迁移包到一个新位置，例如 `/workspace/task-vector-bundle`；把**完全相同内容**的 Qwen3-1.7B 基座放到另一目录。压缩包自带 all-MiniLM-L6-v2 语义模型。

```bash
python -m pip install -r /workspace/task-vector-bundle/requirements-portable.txt

python /workspace/task-vector-bundle/portable_import.py \
  --bundle /workspace/task-vector-bundle \
  --destination /workspace/task-vector-imported-01 \
  --model /models/Qwen3-1.7B \
  --device cuda:0
```

若是 NPU，最后一项改为 `--device npu:0`。CPU 可用于检查文件和小规模调试，不建议在 CPU 上跑完整实验。目的地必须是尚不存在的新目录；出错的半成品也会保留，不覆盖旧证据。

冻结的 `code/` 不要直接执行 `pip install`，以免构建工具在源码目录生成额外文件，改变来源摘要。迁移入口会从该目录直接导入模块；仅向目标 Python 环境安装依赖。先确认环境已安装适合目标硬件的 torch，再安装上述 Python 依赖。

导入后使用目的地的外置配置和专用入口：

```bash
python /workspace/task-vector-bundle/portable_run.py \
  --destination /workspace/task-vector-imported-01 \
  preflight

python /workspace/task-vector-bundle/portable_run.py \
  --destination /workspace/task-vector-imported-01 \
  smoke

python /workspace/task-vector-bundle/portable_run.py \
  --destination /workspace/task-vector-imported-01 \
  core --scope mvp --split test
```

完成 core 后可以按需继续原流程：

```bash
python /workspace/task-vector-bundle/portable_run.py \
  --destination /workspace/task-vector-imported-01 \
  all --scope mvp --split test_formal
```

`all` 会按原 workflow 和科学门禁执行：发现/锁定/正式评测、补充种子、留出任务、扩展、消融与顺序对照等依赖阶段。已导入完成的训练会核验后复用；没有训练过的任务、种子、层或留出方案仍需训练，迁移不能省掉这些科学要求。

请以入口的 `--help` 为准。不要使用原 ModelMate 专用八卡入口在 CUDA 上启动，也不要用普通 `task-vector` CLI 绕过导入校验。目标环境变化需要重新 smoke，原 NPU smoke 仅保留为历史证据。完整包中保持的是当前旧版代码，不会自动改成另一套加速实现。

## 继承与证据边界

向量 `.pt`、准备数据、原 manifest 和原训练记录保留原始字节。目标训练完成记录显式标为 `imported_completed`，同时绑定原请求与新路径配置，由专用入口校验。它们不会伪装成目标机器重新训练的结果，也不支持恢复旧 optimizer/RNG 状态。

统计若通过原输入、源码、基座和内容校验，可作为固定的原始校准矩阵继承，并登记导入来源；不会宣称在新硬件重新计算过。新方案仍按其 seen-task 范围重建子空间，不能把 MVP 的 LoDA 向量当作 leave-one-out 训练。

目标不会继承旧 evaluation/analysis/gate/lock 为当前有效记录，需重新评测和发现。旧测试集已经被运行过，迁移重评不会让它变成未见过的独立确认集。目标硬件、库或 batch 改变时，BF16 生成 token 可能不同；重新评测是重新建立目标环境的结果。

本工具的本地测试使用合成数据和 CPU 验证导出/导入与张量一致性；真实目标设备的兼容性以目标 preflight/smoke 为准。
