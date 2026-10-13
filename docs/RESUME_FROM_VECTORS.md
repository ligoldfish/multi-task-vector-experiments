# 从独立向量包接着做实验

`task-vectors-only-时间.zip` 加上本仓库，足够读取张量、接到相同基座推理、进行向量几何分析、审查原实现和训练来源。继续**同一套完整实验**还需要原始固定数据划分和相同基座：向量文件无法恢复当时的数据、数据 ID 或优化器状态。

本仓库提供两条接续路径。两条都保留已训练的效果，在目标环境重新评测，不继承旧预测、协议锁或硬件验证；迁移不意味着原测试集变成未见数据。

## 路径一：有完整迁移包

解压 `task-vector-portable-full-时间.zip`，按 `portable/README_迁移与使用.md` 使用 `portable_import.py`。完整包包括原实际运行源码及配置、八份向量、全部固定数据、LoDA 统计/子空间和语义评分模型，因此证据最完整，通常也最快。

## 路径二：独立向量包加仓库

还需准备：

1. 内容完全相同的 Qwen3-1.7B 权重、配置和 tokenizer。工具逐文件比对独立向量包的 `BASE_MODEL_CONTRACT.json`，不会只按模型名称判断。
2. **原实验整个 `data/` 目录**，包括原字节不变的 `manifest.json`，七个任务的 train/tune/discovery/test/test_formal/test_confirmation JSONL。可以从完整迁移包的 `experiment/data/` 取出，或在原机器另行拷贝。只复制文件，保留 manifest 内旧路径字段作为历史证据；加载器会按新目录定位 JSONL。
3. 目标环境的 all-MiniLM-L6-v2 评分模型，以及 PyTorch、Transformers 和其余实验依赖。NPU 另需匹配 `torch_npu`/CANN；CUDA 使用对应 PyTorch。BF16 跨硬件不保证逐 token 相同，需在目标重新 smoke 和评测。

仅重新下载原始数据并运行 `prepare`，不等于已经证明得到原来那份划分：源文件、筛选环境及 manifest 中路径可能变化。本入口要求原 prepared manifest 的哈希与训练来源一致；不接受猜测或改写训练来源以绕过检查。如果只有独立向量、暂时没有原数据，先进行推理/几何分析和代码审查，取得原 `data/` 后再接续。

### 1. 在目标创建运行环境和配置

以下是 Linux 终端示例。选择 NPU 或 CUDA 环境原生提供的 PyTorch；`baseline/requirements.txt`/`baseline/pyproject.toml` 与 `portable/requirements-portable.txt` 记录其余依赖。不要用 CPU wheel 覆盖已经可用的加速框架。

把 `baseline/configs/experiment.yaml` 复制到仓库外的 `target-config.yaml`，只改以下内容：

```yaml
platform:
  device: cuda:0                 # 昇腾用 npu:0；CPU 可审查但全流程会很慢
  persistent_root: /work/unused-placement
model:
  path: /models/Qwen3-1.7B
metrics:
  semantic_model_path: /models/all-MiniLM-L6-v2
```

这是**修改已有字段的示意**，不要把整份 YAML 替换为这三段。其余科学配置保持原样。`--destination` 会设置新的实际数据/结果根目录，不依赖 `persistent_root` 的示意值。固定准备好的数据供后续工作使用，不需要把原始下载数据路径改成新路径。

### 2. 审计并挂接已完成的八份向量

```bash
python tools/attach_vectors.py \
  --bundle /downloads/task-vectors-only-时间.zip \
  --config /work/target-config.yaml \
  --prepared-data /downloads/original-data \
  --destination /work/task-vector-resumed
```

也接受解压后的独立包目录。目标目录必须不存在；工具不会覆盖旧实验或删除任何文件。它会：

- 校验包文件、基座、原数据 manifest、全部 JSONL，以及已训练任务实际 train/tune 内容。
- 对比训练来源中绑定的 **`task_vector/*.py` 全部科学实现文件**内容哈希。此入口默认使用仓库的 `baseline/`，不能直接改成 accelerated 源码来冒充原实现。
- 原样保留 `raw.pt`、`post.pt`、`initial.pt`，创建新位置的完成记录和 `IMPORT_RECEIPT.json`，明确标记为 imported_completed，没有重新训练。
- 在 `vector-origin-archive/` 保留原训练来源、原数据 manifest、向量和目标使用的源码快照。

独立包没有原 `ops/` 和动态配置的逐文件清单，因此此入口**不声称核实了原运行目录的每一个文件**；收据明确记录这一证据边界。它核实的是绑定在训练记录中的完整科学 Python 包与原训练数据，而目标运行位置另行记录。原语义评分模型的权重哈希也不在独立包里：目标评分模型会记录新哈希；需要精确核实原评分模型时，使用完整包内的评分模型。

### 3. 确认八份训练被复用，验证目标，然后续跑

```bash
python portable/portable_run.py --destination /work/task-vector-resumed inventory
python portable/portable_run.py --destination /work/task-vector-resumed preflight
python portable/portable_run.py --destination /work/task-vector-resumed smoke
python portable/portable_run.py --destination /work/task-vector-resumed all --scope mvp --split test_formal
```

`inventory` 应显示 `completed_imports: 8`。这里的八份是 LTV/LoDA 两条路线，各有 math、logic、translation、paraphrase 四个任务，seed=17；不是八个全量模型。`all --split test_formal` 会跳过这些已完成训练，先重新做目标环境的 256 条首轮评测，再按科学门禁继续后续阶段并为全部控制组做正式扩评。只写默认的 `all` 使用 `test`，不等于已经要求正式扩评。后续尚未完成的 seed、层消融、留出、扩展任务和顺序对照仍需训练，这是原实验范围，不能用已有八份向量替代。

独立包不含 LoDA 冻结基座统计和子空间缓存，后续需要时会在目标重新计算；不会因此重训已挂接的八份已完成向量。完整包可复用已有缓存。如果目标只能分配单卡，这条入口顺序执行相应任务；独立向量本身无需八卡，八卡并行是 ModelMate 的运行安排。

### 能续什么，不能续什么

可以复用完成的训练效果并接续后面的实验阶段；不能从旧优化器的某一个 step 精确续接，因为独立包没有 optimizer、scheduler、RNG checkpoint。`best-adapter.pt` 也不是完整断点。原来的 `post.pt`/`raw.pt` 不会被重新命名为新的训练证据，原请求签名和来源会保留在归档与导入收据中。

加载目标实验使用 `portable/portable_run.py`，不要直接运行普通 `task_vector` CLI 或 ModelMate 原入口：普通入口会按旧的严格来源绑定拒绝已迁移记录；迁移入口显式验证导入收据后才接受这些记录。目标目录建立后保持位置和源码快照稳定；移动它或修改科学配置时，重新从原独立包审计导入新的目标目录。
