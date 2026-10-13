# 本仓库包含哪些完整代码

本次发布于 2026-10-13，保留两份独立的实验实现，便于审查旧实验和使用新实现。两个快照不混装，模型、数据、task vector 和运行目录不随 Git 上传。实验原文与已批准补充协议均在每个快照的 `docs/` 内。

| 路径 | 内容 | 使用边界 |
|---|---|---|
| `baseline/` | 本地旧实现全部 `src/`、`configs/`、`deploy/`、`ops/`、`tests/`、`docs/`、`third_party/`，四个 ModelMate 入口、项目依赖、README 与操作指南 | 审查用户已运行实验对应的本地代码；实际 Worker 的本地配置仍须从导出包核对 |
| `accelerated/` | 加速实现的同等完整范围，另含批量生成、八卡分片、任务池、顺序池、bootstrap 加速、监控及针对性测试 | 保持实验协议的工程加速；默认允许 BF16 batch 形状引起的正常 token 差异，需在目标设备重新验证 |
| `portable/` | task-vector 独立读取、接入、导出和迁移工具 | 根据工具说明核对包类型、基座与可继承能力 |
| `provenance/source-snapshots.json` | 两份快照每个文件 SHA256、原实验 source_manifest 和训练 Python 包摘要 | 在分析 `.pt` 的训练来源时核对，不把路径相同当作内容相同 |
| `provenance/curate_sources.py` | 创建快照的复制与校验工具 | 只操作新的发布目录，拒绝覆盖已有快照 |

源目录中的模型、数据集、输出、虚拟环境、Git 历史、Python 缓存、pytest 临时数据与临时安装/恢复沙箱没有复制。单元测试的固定 `tests/fixtures/mbpp_927.json` 保留，它是官方程序 setup 顺序回归用例，不是正式训练数据。

快照保留原文件字节；唯一发布属性调整是将原 `.gitattributes` 原字节保存为 `.gitattributes.snapshot-original`，实际属性使用 `* -text`，避免 Git 自动将 CRLF 改为 LF，导致训练来源摘要失效。该属性文件不属于实验 `source_manifest`。精确文件数和字节数以 `provenance/source-snapshots.json` 为准。

## 历史验证材料的含义

`accelerated/validation-*.xml`、`accelerated/docs/validation/` 及两份快照的本地验证报告记录历史工程测试。它们不是本次远端训练结果，不证明共享成分、迁移收益或组合收益成立。标题中的测试时间、运行平台与结果只能按报告本身解释；失败后修复的中间报告也保留供审查，不能只挑通过报告。

原代码中的报告有明确历史日期，文内“尚未正式运行”描述的是编写该报告时的状态，不能当作当前 Worker 状态。用户提供过真实 NPU smoke 和首轮训练完成日志，但本仓库不伪造没有拿到的远端日志、配置、模型或训练张量。

## 来源证明的边界

`baseline/` 对本地 `task-vector` 目录逐字节复制验证；`accelerated/` 对本地加速目录逐字节复制验证。`source_manifest_sha256` 使用原 `config.py` 的精确算法：覆盖 `src/configs/deploy/third_party/ops`、`pyproject.toml`、原文、补充协议、`run_modelmate*.py`；`training_package_sha256` 使用原 `training._checkpoint_provenance` 的精确算法，即 `src/task_vector/*.py` 的文件名到内容 SHA256 映射摘要。

实际 ModelMate Worker 还生成了 `configs/modelmate.local.yaml` 和 `ops/settings.local.json`，本地发布源中没有这两份远端文件。因此完整 `source_manifest` 不能仅凭本仓库与 Worker 路径相似就宣称一致。应读取用户导出包的 `training.json`、向量元数据、基座契约和训练 provenance，核对 `training_sources_sha256`；若有原远端源码清单，再核对完整映射。缺少对应证据时必须标记“来源未验证”，不要修改原签名来假装匹配。

## 第三方来源与许可

原 `third_party/official_sources.json` 和 `third_party/baseline_sources.json` 保留固定 upstream commit、下载 URL、每个选取文件的 SHA256。它们是所需官方文件的固定快照，不是完整 upstream 仓库。Learned_TV、TIES、TSV、KnOTS 的原许可证与 README 均随相应文件保留。

LoDA 固定 upstream 快照的 `LICENSE` 下载当时返回 404，清单保留该失败记录；本仓库没有替其编造许可，也没有将整个项目擅自声明为 MIT/Apache。根项目未新增开源许可证。私有发布用于用户授权的实验协作与审查；若未来公开或重新分发，应先确认各第三方许可，尤其是 LoDA 的许可状态。

## 只拿到 vectors-only 时

可以读取向量、比较其范数与几何关系、核对训练摘要、在同一权重的 Qwen3-1.7B 上重新推理与评测。代码提供完整审查入口；具体可重现程度取决于导出包里实际存在的内容。独立向量不是 AdamW 状态，无法凭 `post.pt` 恢复训练到一半的优化器、epoch/batch 游标和所有 RNG。当前 portable-full 导出器也明确不包含 optimizer/RNG checkpoint，它支持继承已完成训练的效果并重新评测；精确断点续训必须另存完整 `last.pt`/`best.pt` 及其绑定的样本、子空间、配置和源码证据，并通过来源校验。若重新下载数据，必须核对原划分的 ID 与内容，不能把新抽样称作续跑。
