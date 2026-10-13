# ModelMate 运维与修复资料

这些文件保留原运行环境的工具和问题修复过程，供代码审查、环境准备及运行监控使用。脚本中的旧绝对路径是历史部署值；迁移目标优先使用仓库根 README 和 `docs/RESUME_FROM_VECTORS.md`。

| 目录 | 用途 |
|---|---|
| `platform-entry/` | 修复 ModelMate 自动注入启动参数时原入口拒绝运行的问题，含测试和说明 |
| `monitor/` | 根据训练容器落盘证据监控阶段、评测进展及有条件的 ETA |
| `online-runtime/` | 复用已有 Ascend 运行时、依赖修复、镜像下载模型/数据、BF16 诊断和 checkpoint 检查 |
| `online-runtime/*-fix/` | MBPP setup、prefill smoke、可移植 checkpoint 修复包的构建/安装/验证工具；修复已进入科学源码快照 |
| `conformance-audit/` | 2026-10-10 原文与旧实现逐项审计的脚本和历史报告 |

历史一致性审计记录的是修复前后的特定版本，不能把其中旧缺口直接当作当前代码缺口。当前需求矩阵、修复及验证在 `baseline/docs/` 和 `accelerated/docs/` 中。运维脚本和历史报告不构成真实训练结果或科学结论。

没有复制虚拟环境、模型、数据下载内容、测试临时目录或磁盘清理资料。完整向量导出工具在 `portable/`，真实训练产物由用户单独提供。
