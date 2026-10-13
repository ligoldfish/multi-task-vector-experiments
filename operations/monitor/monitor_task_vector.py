"""Read-only DPC monitor. Standard library only; no model/checkpoint loading."""
from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timezone, timedelta
import json
import math
from pathlib import Path
import statistics
import sys
import time

ROOT = Path("/opt/dpcvol/datasets/naie/task-vector/experiments/exp-20261011-01")
CORE = ("math", "logic", "translation", "paraphrase")
EXTENDED = CORE + ("code_generation", "code_repair")
SEEDS = (17, 29, 43)
SCOPES = ("mvp", "loo-math", "loo-logic", "loo-translation", "loo-paraphrase", "expanded")
CN = timezone(timedelta(hours=8))
TERMINAL = {"completed", "failed", "interrupted", "stopped_by_gate", "report_failed"}


def duration(value):
    if value is None or not math.isfinite(value):
        return "待估"
    value = max(0, round(value))
    days, value = divmod(value, 86400)
    hours, value = divmod(value, 3600)
    minutes, seconds = divmod(value, 60)
    return (f"{days}天" if days else "") + f"{hours:02d}:{minutes:02d}:{seconds:02d}"


def timestamp(value):
    if not isinstance(value, str) or not value:
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        return parsed.replace(tzinfo=timezone.utc).timestamp() if parsed.tzinfo is None else parsed.timestamp()
    except (ValueError, TypeError):
        return None


def mtime(path):
    try:
        return Path(path).stat().st_mtime
    except OSError:
        return None


class JsonCache:
    def __init__(self):
        self.values = {}
        self.errors = {}
        self.epoch_observations = {}

    def read(self, path, default=None):
        path = Path(path)
        try:
            state = path.stat()
            key = (state.st_mtime_ns, state.st_size)
            if state.st_size > 32 * 1024 * 1024:
                raise ValueError("JSON exceeds 32 MiB read limit")
            if path not in self.values or self.values[path][0] != key:
                result = json.loads(path.read_text(encoding="utf-8-sig"))
                if isinstance(default, (dict, list)) and not isinstance(result, type(default)):
                    raise ValueError(f"JSON must contain {type(default).__name__}")
                self.values[path] = key, result
            self.errors.pop(path, None)
            return self.values[path][1]
        except FileNotFoundError:
            self.values.pop(path, None)
            return default
        except (OSError, ValueError) as exc:
            self.errors[path] = str(exc)
            return default


def select_job(root, selected=None):
    folder = Path(root) / "runs/platform_jobs"
    if selected:
        candidate = Path(selected)
        candidate = candidate if candidate.is_absolute() else folder / candidate
        if not candidate.is_dir():
            raise ValueError(f"Worker evidence directory not found: {candidate}")
        if not candidate.resolve().is_relative_to(folder.resolve()):
            raise ValueError("--job must stay within this experiment's runs/platform_jobs")
        return candidate
    candidates = [p for p in folder.glob("full-*") if p.is_dir()]
    # Compare the timestamp portions, so a legacy full-formal prefix cannot
    # lexicographically outrank a newer ordinary full worker.
    return max(candidates, key=lambda p: p.name.removeprefix("full-formal-").removeprefix("full-")) if candidates else None


def select_pool(root, job_metadata, job_result, cache):
    lower = timestamp(job_metadata.get("started_utc"))
    upper = timestamp(job_result.get("ended_utc"))
    if lower is None:
        return None, {}
    candidates = []
    for path in (Path(root) / "scheduler").glob("mvp-seed17-*/pool.json"):
        data = cache.read(path, {})
        starts = [timestamp(item.get("started_utc")) for item in data.get("jobs", [])]
        starts = [value for value in starts if value is not None]
        # Child timestamps are ISO UTC; the scheduler's own stamp is compact.
        start = min(starts) if starts else None
        if start is not None and start >= lower and (upper is None or start <= upper):
            candidates.append((start, path, data))
    if not candidates:
        return None, {}
    _, path, data = max(candidates, key=lambda item: item[0])
    return path, data


def planned_training(root):
    """Current authorized full protocol, deduplicating reused LTV coordinates."""
    base = Path(root) / "runs/training"
    records = []
    for seed in SEEDS:
        for task in EXTENDED:
            records.append(("ltv", task, base / "independent/ltv" / str(seed) / task))
        for layer in ("6", "22", "6-14-22"):
            for task in CORE:
                records.append((f"ltv-layers-{layer}", task,
                                base / "independent" / f"ltv-layers-{layer}" / str(seed) / task))
        for scope in SCOPES:
            tasks = EXTENDED if scope == "expanded" else tuple(t for t in CORE if scope != "loo-" + t)
            for task in tasks:
                records.append(("loda", task, base / scope / "loda" / str(seed) / task))
    return records


def epoch_state(folder, cache, now, max_epochs=5):
    folder = Path(folder)
    completed = cache.read(folder / "training.json", {})
    epochs = cache.read(folder / "epochs.json", [])
    epochs = [item for item in epochs if isinstance(item, dict)
              and type(item.get("epoch")) is int and 0 <= item["epoch"] <= max_epochs]
    count = max((item["epoch"] for item in epochs), default=0)
    checkpoint = mtime(folder / "last.pt")
    initial, boundary = mtime(folder / "initial.pt"), mtime(folder / "epochs.json")
    committed = completed.get("status") == "complete"
    eta, rate = None, None
    observed = cache.epoch_observations.get(folder)
    if observed is None or observed["initial"] != initial:
        observed = {"initial": initial, "samples": []}
        cache.epoch_observations[folder] = observed
        # A fresh start actually observed at epoch zero is a valid time anchor.
        if count == 0 and initial is not None:
            observed["samples"].append((0, initial))
    samples = observed["samples"]
    if count and boundary is not None and (initial is None or boundary >= initial):
        if samples and count < samples[-1][0]:
            samples.clear()
        if not samples or (count > samples[-1][0] and boundary > samples[-1][1]):
            samples.append((count, boundary))
            del samples[:-6]
    if committed:
        count, eta = int(completed.get("completed_epochs", count)), 0.0
    elif 0 < count < max_epochs and len(samples) >= 2:
        first, last = samples[-2:]
        rate = (last[1] - first[1]) / (last[0] - first[0])
        # Includes train, tuning and epoch checkpoint time; early stop can shorten it.
        forecast = boundary + rate * (max_epochs - count) - now
        eta = forecast if forecast > 0 else None
    return {"committed": committed, "record": completed, "epochs": count,
            "checkpoint": checkpoint, "initial": initial, "epoch_boundary": boundary,
            "eta": eta, "epoch_seconds": rate,
            "last_nll": epochs[-1].get("tune_answer_nll") if epochs else None}


def tail_error(path):
    try:
        with Path(path).open("rb") as handle:
            handle.seek(0, 2)
            handle.seek(max(0, handle.tell() - 32768))
            lines = handle.read(32768).decode("utf-8", errors="replace").splitlines()
        return " | ".join(lines[-6:])[-1800:]
    except OSError:
        return "日志当前不可读取"


def first_seed_panel_eta(root, cache):
    """Known 52 test + 52 discovery panels; excludes bootstrap/analysis costs."""
    base = Path(root) / "runs/evaluation/mvp/17"
    expected, costs = [], defaultdict(list)
    for split, count in (("test", 256), ("discovery", 128)):
        for target in CORE:
            expected.append((split, target, count, base / split / "canonical/base" / target / "scores.json"))
        for route in ("ltv", "loda"):
            for source in CORE:
                for target in CORE:
                    expected.append((split, target, count, base / split / "canonical" /
                                     f"{route}-{source}-primary" / target / "scores.json"))
                for control in ("initial", "random-0"):
                    expected.append((split, source, count, base / split / "canonical" /
                                     f"{route}-{source}-{control}" / source / "scores.json"))
    done, pending, rows, seconds = 0, [], 0, 0.0
    for split, task, count, path in expected:
        record = cache.read(path, {})
        if record.get("signature") and record.get("count") == count:
            done += 1
            newly = record.get("newly_generated", record.get("count", 0))
            elapsed = record.get("runtime_seconds", 0)
            if newly > 0 and elapsed > 0:
                costs[task].append(elapsed / newly)
            rows += count
            seconds += max(0, elapsed)
        else:
            pending.append((task, count))
    unknown = sum(task not in costs for task, _ in pending)
    # Only return a complete cost projection if every remaining target has timing.
    eta = sum(count * statistics.median(costs[task]) for task, count in pending) if not unknown else None
    return {"done": done, "total": len(expected), "samples": rows, "eta": eta,
            "unknown_target_panels": unknown, "measured_seconds": seconds}


def show(root, job, cache, now):
    root = Path(root)
    runs = root / "runs"
    metadata = cache.read(job / "metadata.json", {})
    result = cache.read(job / "result.json", {})
    state = result.get("status", metadata.get("status", "UNKNOWN"))
    started = timestamp(metadata.get("started_utc"))
    ended = timestamp(result.get("ended_utc"))
    elapsed = None if started is None else (ended if ended is not None else now) - started
    print("\n" + "=" * 84)
    print(datetime.fromtimestamp(now, CN).strftime("%Y-%m-%d %H:%M:%S +08:00"))
    print(f"WORKER_EVIDENCE={job}")
    print(f"Worker记录状态={state}  已用时={duration(elapsed)}  平台网页状态=UNKNOWN（请在网页核对）")
    context = cache.read(job / "context.json", {})
    devices = context.get("visible_devices", context.get("device_count", "待记录"))
    print(f"训练容器已记录NPU数={devices}；本监控不读取Notebook的NPU或进程作为训练状态")
    pool_path, pool = select_pool(root, metadata, result, cache)
    reusable = cache.read(job / "pool_reuse.json", {})
    active = []
    if pool_path:
        print(f"首轮pool={pool.get('status', 'UNKNOWN')}  最近调度写入距今={duration(now - (mtime(pool_path) or now))}")
        print("路线/任务                       pool状态     epoch(最多5)   NLL          checkpoint距今   训练ETA")
        estimates, unknown, complete = [], False, 0
        for item in pool.get("jobs", []):
            route, task = item.get("route", "?"), item.get("task", "?")
            family = "independent" if route == "ltv" else "mvp"
            folder = runs / "training" / family / route / "17" / task
            progress = epoch_state(folder, cache, now)
            complete += int(progress["committed"])
            job_state = item.get("status", "UNKNOWN")
            if not progress["committed"]:
                active.append(folder)
                unknown |= progress["eta"] is None or job_state not in {"running", "completed"}
                if progress["eta"] is not None:
                    estimates.append(progress["eta"])
            nll = progress["last_nll"]
            nll_text = f"{nll:.4f}" if isinstance(nll, (int, float)) else "--"
            age = duration(None if progress["checkpoint"] is None else now - progress["checkpoint"])
            print(f"{route + '/' + task:30s} {job_state:12s} {progress['epochs']:2d}/5         {nll_text:11s} {age:15s} {duration(progress['eta'])}")
            if job_state in {"failed", "launch_failed", "cancelled_after_failure", "interrupted"} and item.get("log"):
                print("  错误日志尾部：" + tail_error(item["log"]))
        pool_eta = None if unknown else max(estimates, default=0.0)
        print(f"首轮正式训练完成={complete}/{len(pool.get('jobs', []))}；当前并行训练ETA={duration(pool_eta)}")
        print("ETA按监控到的epoch间隔外推到5轮，早停可缩短；采样/LoDA前置统计/特征分解期间待估。")
    elif reusable.get("pool_skipped"):
        print("首轮pool已复用完成记录；后续训练/评估继续按原full流程。")
    else:
        print("本worker尚无匹配的首轮pool记录：可能在前置检查，或入口尚未进入调度。")

    planned = planned_training(root)
    completed, costs, pending = [], defaultdict(list), []
    for route, task, folder in planned:
        progress = epoch_state(folder, cache, now)
        if progress["committed"]:
            completed.append(folder)
            cost = progress["record"].get("runtime_seconds", 0)
            if cost > 0:
                costs[(route, task)].append(cost)
        else:
            pending.append((route, task, folder, progress))
            if progress["initial"] is not None:
                active.append(folder)
    unknown_jobs, remaining_work = 0, 0.0
    for route, task, folder, progress in pending:
        if costs[(route, task)]:
            # Cumulative training work, not eight-card wall time; excludes calibration.
            estimate = statistics.median(costs[(route, task)])
            remaining_work += estimate
        else:
            unknown_jobs += 1
    sequential = [p for p in (runs / "sequential").glob("seed*/*/stage*/stage-completed.json")
                  if cache.read(p, {}).get("status") == "complete"]
    print(f"全范围独立训练完成记录={len(completed)}/{len(planned)}；顺序对照阶段完成记录={len(sequential)}/48")
    print("上述分母是全部科学门禁通过时的条件计划；记录计数未重新校验所有张量产物哈希。")
    print(f"未提交任务按同类完整训练耗时投影≈{duration(remaining_work)}；尚缺同类计时={unknown_jobs}项")
    print("该投影未扣除正在训练部分，不含统计/分解、顺序对照、生成评估及分析，不是全程ETA。")
    for stage in (runs / "sequential").glob("seed*/*/stage*"):
        if stage.is_dir() and not (stage / "stage-completed.json").exists() and (stage / "initial.pt").exists():
            active.append(stage)
    recent_active = sorted(set(active), key=lambda p: mtime(p / "last.pt") or mtime(p / "initial.pt") or 0,
                           reverse=True)
    if recent_active:
        print("最近未提交训练目录（依文件记录；不代表已确认进程存活）：")
        for folder in recent_active[:6]:
            progress = epoch_state(folder, cache, now)
            print(f"  {folder.relative_to(root)} epoch={progress['epochs']}/5 ETA={duration(progress['eta'])}")

    print("阶段记录（完成标记有/无）：")
    for scope in SCOPES:
        lock = (runs / "protocol" / f"{scope}.lock.json").is_file()
        primary = (runs / "analysis" / scope / "test-completed.json").is_file()
        formal = (runs / "analysis" / scope / "test_formal-completed.json").is_file()
        gate = cache.read(runs / "analysis" / scope / "gate.json", {})
        gate_text = "待记录" if not gate else f"passes={gate.get('passes')} seeds={gate.get('seeds')}"
        print(f"  {scope:18s} 锁定={'有' if lock else '无'} 主评={'有' if primary else '无'} 扩评={'有' if formal else '无'} {gate_text}")
    for split in ("test", "test_formal"):
        print(f"  控制组/{split}: " + ("已落盘" if (runs / "analysis/controls" / f"{split}-completed.json").is_file() else "未落盘"))
    first = first_seed_panel_eta(root, cache)
    print(f"首seed主评+发现面板={first['done']}/{first['total']}（含随机/初始化控制）；已保存样本={first['samples']}")
    print(f"这组面板的剩余生成/评分成本≈{duration(first['eta'])}（不含bootstrap/分析；无逐样本实时进度）")
    if state in TERMINAL:
        print(f"全程剩余时间：作业已记录终态 {state}；completed={result.get('completed')} exit_code={result.get('exit_code')}")
    else:
        print("全程ETA=暂不可可靠估计；后续生成量/条件分组及科学门禁未形成完整运行预算。")

    relevant = []
    for tree in ("training", "evaluation", "analysis", "protocol", "statistics", "sequential"):
        for path in (runs / tree).rglob("*"):
            if path.suffix in {".json", ".jsonl", ".pt"}:
                changed = mtime(path)
                if changed is not None and path.is_file():
                    relevant.append((changed, path))
    relevant.sort(key=lambda item: item[0], reverse=True)
    print("最近实验产物（只读取文件属性，绝不加载.pt）：")
    for changed, path in relevant[:5]:
        print(f"  距今{duration(now - changed)}  {path.relative_to(root)}")
    if relevant and now - relevant[0][0] > 1800 and state not in TERMINAL:
        print("提示：30分钟未见上述产物更新；长生成/CPU分解也会这样，请结合网页日志判断，不能据此判定卡死。")
    for folder in sorted(set(active))[:8]:
        failure = cache.read(folder / "failure.json", {})
        if failure:
            print(f"任务失败记录（可能来自之前attempt，需结合当前日志）：{folder}: {failure.get('type')} {failure.get('message')}")
    if result.get("error"):
        print("本worker错误：" + str(result["error"]))
    failure = cache.read(job / "failure.json", {})
    if failure.get("error") and not result.get("error"):
        print("本worker失败记录：" + str(failure["error"]))
    for path, error in list(cache.errors.items())[:4]:
        print(f"读取暂失败（下一轮重试）：{path}: {error}")
    return state in TERMINAL


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--job", help="Worker full-... ID or its absolute evidence path; defaults to latest full worker")
    parser.add_argument("--interval", type=int, default=60)
    parser.add_argument("--count", type=int, default=0, help="Snapshots; 0 means continuously until terminal or Ctrl-C")
    args = parser.parse_args(argv)
    if args.interval < 10 or args.count < 0:
        parser.error("--interval must be >=10 seconds; --count must be >=0")
    cache, job, snapshots = JsonCache(), None, 0
    print("READ_ONLY_MONITOR=YES；仅从DPC读取；Ctrl-C只停止监控，不发送任何训练信号。", flush=True)
    try:
        while True:
            if job is None:
                job = select_job(args.root, args.job)
            if job is None:
                print("尚无full worker证据。若平台入口失败，需查看网页日志；否则等待worker创建目录。", flush=True)
                terminal = False
            else:
                terminal = show(args.root, job, cache, time.time())
            sys.stdout.flush()
            snapshots += 1
            if terminal or (args.count and snapshots >= args.count):
                return 0
            time.sleep(args.interval)
    except KeyboardInterrupt:
        print("\n监控已停止；训练作业不受影响。", flush=True)
        return 0
    except (OSError, ValueError) as exc:
        print(f"MONITOR_ERROR={exc}", file=sys.stderr, flush=True)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
