#!/usr/bin/env python3
"""Read small JSON progress records only; never imports torch or loads tensors."""
from __future__ import annotations
import argparse
from datetime import datetime, timezone, timedelta
import json
from pathlib import Path
import time


def read(path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def snapshot(root):
    root = Path(root)
    runs = root / "runs"
    print("=" * 72)
    print(datetime.now(timezone(timedelta(hours=8))).isoformat(timespec="seconds"))
    print("ROOT=" + str(root))
    metadata = sorted((runs / "platform_jobs").glob("*/metadata.json"), key=lambda p: p.stat().st_mtime)
    if metadata:
        state = read(metadata[-1]) or {}
        print("Worker记录状态=", state.get("status"), "平台网页状态=UNKNOWN")
        print("WORKER_EVIDENCE=" + str(metadata[-1].parent))
    pools = sorted((root / "scheduler").glob("*/pool.json"), key=lambda p: p.stat().st_mtime)
    if pools:
        pool = read(pools[-1]) or {}
        jobs = pool.get("jobs", [])
        print("最近训练池=", pool.get("status"), "记录=", str(pools[-1]))
        for status in ("pending", "running", "completed", "failed", "cancelled"):
            print("  ", status, sum(job.get("status") == status for job in jobs))
        for job in jobs:
            if job.get("status") == "running":
                print("  活动训练=", job.get("device"), job.get("route"), job.get("task"),
                      "seed=", job.get("seed", pool.get("seed")), "order=", job.get("order"))
    completed_training = list((runs / "training").glob("*/*/*/*/training.json"))
    completed_sequential = list((runs / "sequential").glob("*/*/stage*/stage-completed.json"))
    print("已提交独立/层消融训练记录=", len(completed_training), "顺序阶段完成标记=", len(completed_sequential))
    print("记录计数仅供进度参考，未重新校验所有张量哈希；条件范围由科学门禁决定。")
    generation = read(runs / "engineering/generation/progress.json")
    if generation:
        print("生成阶段=", generation.get("status"), "operation=", generation.get("operation"))
        print("完成分片=", generation.get("completed_workers"), "/", generation.get("expected_workers"))
        print("完成样本=", generation.get("completed_samples"), "/", generation.get("total_samples"))
        eta = generation.get("panel_eta_seconds")
        print("当前面板生成ETA=", "待估" if eta is None else str(timedelta(seconds=round(eta))))
        print("生成心跳UTC=", generation.get("updated_utc"))
        for worker in generation.get("workers", []):
            print("  ", worker.get("device"), worker.get("status"), worker.get("panel"),
                  f"{worker.get('completed_samples', '?')}/{worker.get('total_samples', '?')}")
    bootstrap = read(runs / "engineering/bootstrap-progress.json")
    if bootstrap:
        print("统计阶段=", bootstrap.get("status"), bootstrap.get("task"), bootstrap.get("condition"))
    for scope in ("mvp", "loo-math", "loo-logic", "loo-translation", "loo-paraphrase", "expanded"):
        directory = runs / "analysis" / scope
        print("阶段=", scope, "锁=", (runs / "protocol" / f"{scope}.lock.json").is_file(),
              "主评=", (directory / "test-completed.json").is_file(),
              "扩评=", (directory / "test_formal-completed.json").is_file())
    print("当前面板ETA不含评分/统计；全程ETA仍取决于科学门禁和后续范围。")
    print("本指令只读DPC记录；Notebook资源与训练容器资源隔离。")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True)
    parser.add_argument("--interval", type=int, default=60)
    parser.add_argument("--count", type=int, default=1, help="0 repeats until Ctrl-C")
    args = parser.parse_args()
    if args.interval < 1 or args.count < 0:
        parser.error("interval must be positive; count must be nonnegative")
    index = 0
    try:
        while args.count == 0 or index < args.count:
            snapshot(args.root)
            index += 1
            if args.count == 0 or index < args.count:
                time.sleep(args.interval)
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
