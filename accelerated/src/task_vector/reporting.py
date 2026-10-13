"""Evidence-based reports; implementation and actual scientific results stay distinct."""
from __future__ import annotations
from pathlib import Path
from .io import read_json, file_sha256, atomic_json


def build_report(output_root):
    root = Path(output_root)
    root.mkdir(parents=True, exist_ok=True)
    training = sorted(root.glob("training/**/training.json"))
    sequential_training = sorted(root.glob("sequential/**/training.json"))
    scores = sorted(root.glob("evaluation/**/scores.json"))
    sequential_scores = sorted(root.glob("sequential/**/evaluation/**/metrics.json"))
    confirmations = sorted(root.glob("analysis/**/*-relation-confirmation.json"))
    first_round = root / "analysis" / "mvp" / "first-round-completed.json"
    failures = sorted(root.glob("**/failure.json"))
    gates = sorted(root.glob("analysis/**/gate.json"))
    completed = sorted(root.glob("analysis/**/*-completed.json"))
    records = {"training": [{"path": str(p), "sha256": file_sha256(p), "record": read_json(p)} for p in training],
               "failures": [{"path": str(p), "record": read_json(p)} for p in failures],
               "gates": [{"path": str(p), "record": read_json(p)} for p in gates],
               "completed_panels": [{"path": str(p), "record": read_json(p)} for p in completed],
               "evaluation_files": len(scores),
               "sequential_training": [{"path": str(p), "sha256": file_sha256(p), "record": read_json(p)} for p in sequential_training],
               "sequential_evaluation_files": len(sequential_scores),
               "first_round": read_json(first_round) if first_round.exists() else None,
               "relation_confirmations": [{"path": str(p), "record": read_json(p)} for p in confirmations],
               "scientific_panel_artifacts": [{"path": str(p), "sha256": file_sha256(p)} for p in sorted(root.glob("analysis/**/*.json"))
                                               if any(tag in p.name for tag in ("components", "groups", "composition", "unseen", "summary"))]}
    generated, truncations, seconds = 0, 0., 0.
    for p in scores + sequential_scores:
        record = read_json(p)
        count = record.get("count", 0)
        generated += count
        truncations += record.get("truncation_rate", 0) * count
        seconds += record.get("runtime_seconds", 0)
    records["generation_totals"] = {"samples_in_saved_panels": generated, "truncated": truncations, "evaluation_seconds": seconds,
                                     "note": "Counts include cached artifacts once, not necessarily the number of physical forward executions"}
    atomic_json(root / "report.json", records)
    lines = ["# 实验执行报告", "", f"已保存独立训练记录：{len(training)}；顺序训练记录：{len(sequential_training)}；独立评测面板：{len(scores)}；顺序评测面板：{len(sequential_scores)}；失败记录：{len(failures)}。", ""]
    if records["first_round"]:
        first = records["first_round"]
        lines += [f"首轮 seed={first['seed']}：{first['core_states']} 个核心状态，{first['core_generations']} 条核心预测；随机/初始化及调参/发现开销单独记录。", ""]
    if not training:
        lines += ["尚无正式模型训练证据。代码完成、CPU 小模型测试或环境预检均不能作为共享能力结论。", ""]
    for item in records["gates"]:
        gate = item["record"]
        lines += [f"- {gate['scope']}：有效性门控 {'通过' if gate['passes'] else '未通过'}；seed={gate['seeds']}。证据：`{item['path']}`。"]
    if completed:
        lines += ["", "完成的锁定评测面板："]
        lines += [f"- {item['record']['scope']} / {Path(item['path']).name}；锁定哈希 `{item['record']['protocol_lock']}`。" for item in records["completed_panels"]]
    if confirmations:
        lines += ["", "发现关系的锁定测试确认（不重新选择分组或 q）："]
        for item in records["relation_confirmations"]:
            summary = item["record"]["off_diagonal"]
            lines += [f"- `{item['path']}`：有向非对角关系 {summary['relations']} 条，符号一致率 {summary['sign_agreement']}，标准化秩相关 {summary['standardized_spearman']}。"]
    lines += ["", "原始指标、逐样本预测、配对 bootstrap 区间及负结果见 analysis/evaluation/sequential；不同任务指标保留原单位。", "components/groups 面板包含共享成分与同秩随机对照的直接配对差值区间；只有一个联合依赖簇时CI不可识别，退化区间不得视作独立统计证据。", "无稳定分组、低秩或门控未通过均须保留；这些结果本身不证明存在共享能力。", "运行时间和显存是工程证据，不替代 E1–E6 的科学证据。", ""]
    path = root / "report.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    return {"path": str(path), "training_records": len(training), "evaluation_panels": len(scores)}
