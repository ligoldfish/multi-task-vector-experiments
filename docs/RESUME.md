# 恢复与继续实验

项目区分两件事：复用已经训练完成的表示，和恢复一个尚未完成训练的优化器步骤。前者需要向量及来源记录；后者还需要完整 checkpoint，不能只靠 `post.pt` 完成。

## 在原位置继续

保持模型、数据、科学配置和源码不变，重复相同实验命令。调度器会核验已经提交完成的训练与评测，复用有效结果，并继续尚未完成的部分。若修改推理设备或工程配置，应按目标运行形成新的评测记录，而不改写原训练来源。

实验目录中的 `last.pt`/`best.pt` 是包含优化器、调度器、游标、RNG 等状态的完整训练断点。精确训练恢复按 checkpoint 绑定的数据、编码、模型、子空间与源码进行校验。`best-adapter.pt` 和导出的表示不是完整断点。

## 导出已完成结果

```bash
mkdir -p /artifacts
python portable/export_portable.py --code . --experiment /work/task-vector --config /work/experiment.yaml --output /artifacts
```

导出器接受一个或多个已完成任务，不要求先完成全部研究。它会校验现有七任务 prepared data、基座记录和已完成训练：

- `vectors-时间.zip` 保存表示、训练来源、基座契约和校验清单，适合分发及 CPU 分析。
- `experiment-时间.zip` 另保存固定数据、运行源码、配置、已有统计/子空间和评分模型，适合实验迁移。

Qwen3 基座权重和优化器/RNG checkpoint 不包含在这些导出物中。历史运行日志与协议记录作为证据保留，迁移时不会被当作新环境已经通过验证。

## 将已有向量接入当前实现

完整实验接续还需以下资料：

1. 与训练时逐文件内容相同的 Qwen3-1.7B。
2. 原固定 `data/` 目录，包含原字节不变的 `manifest.json` 和七任务全部 JSONL。重新下载并抽样，不自动等于原来的划分。
3. 目标环境中的 all-MiniLM-L6-v2 评分模型及依赖。
4. 一份采用本项目协议、填写目标设备与模型路径的配置，参见 [运行指南](RUNNING.md)。

```bash
python tools/attach_vectors.py --archive /artifacts/vectors.zip --config /work/experiment.yaml --prepared-data /artifacts/original-data --destination /work/resumed-experiment
python portable/portable_run.py --destination /work/resumed-experiment inventory
python portable/portable_run.py --destination /work/resumed-experiment preflight
python portable/portable_run.py --destination /work/resumed-experiment smoke
python portable/portable_run.py --destination /work/resumed-experiment all --scope mvp --split test_formal
```

向量归档也可以使用解压后的目录。`--destination` 必须是新目录。工具核对基座、原数据和 train/tune 实际内容、表示的 hash、task/seed/层/训练范围，以及训练来源。核验通过的完成表示被原样保留；缺少的任务、种子或消融仍由当前实现训练。

来源完全相同时，收据记录精确代码匹配。对于 `provenance/representation-compatibility.json` 明确列出的实现变更，工具仅接受固定的来源/目标代码摘要，并核对十五个训练、数据和干预方法文件未变：这允许复用完成表示，但不声称两份源码完全相同。新评测、新训练和新运行位置独立记录；原向量与原训练来源不修改。任意未知代码变更不会自动获准。

使用 `inventory` 查看已导入训练数量，再完成目标环境的真实模型检查。旧预测与协议锁不作为有效新结果继承。不同设备或推理 batch 下，BF16 生成可能略有差异；重新评测已经接触过的 test，不会获得新的独立确认资格。

## 恢复带固定数据的实验归档

将 `experiment-时间.zip` 解压后，使用其中的源码和固定证据进行迁移：

```bash
python portable/portable_import.py --bundle /artifacts/experiment-extracted --destination /work/restored-experiment --model /models/Qwen3-1.7B --device npu:0
python portable/portable_run.py --destination /work/restored-experiment inventory
python portable/portable_run.py --destination /work/restored-experiment preflight
python portable/portable_run.py --destination /work/restored-experiment smoke
python portable/portable_run.py --destination /work/restored-experiment all --scope mvp --split test_formal
```

这种恢复保留归档中的源码快照，支持审查与复现该次实验。若希望将另一实现修订训练的表示用于当前实现，应使用上面的向量接续入口，让代码兼容性与新评测边界得到显式核验。

## 只有单个 PT 文件时

可以检查张量，或把它接到相同基座上推理，见 [向量指南](VECTORS.md)。单个 PT 不包含原固定数据、完整训练来源清单和优化器状态；这些资料缺失时，可以重新开展一项明确记录的新评测，但不能声称恢复了原完整实验。
