"""Print new completed answers from the existing read-only cluster transport."""
import argparse
from collections import Counter
import fcntl
import json
from pathlib import Path
import subprocess
import sys
import time


def follow(args):
    with Path(str(args.log) + ".lock").open("a+b") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        seen = set()
        counts = {}
        with args.log.open("a", encoding="utf-8", buffering=1) as log:
            def emit(line):
                print(line, flush=True)
                print(line, file=log, flush=True)

            emit("CV3D 实时结果：累计准确率按已完成题目计算；错误和非法答案计入分母。")
            emit("只读取已冻结实验的结果，不调用模型、不修改实验配置。")
            last_summary = None
            last_heartbeat = 0.0
            while True:
                result = subprocess.run(
                    [sys.executable, str(args.transport), "titan", str(args.snapshot)],
                    capture_output=True, text=True, timeout=130)
                if result.returncode:
                    emit("只读连接中断，实时打印已停止；服务器实验保持原有状态。")
                    return 2
                value = json.loads(result.stdout)
                for row in value["results"]:
                    if row["event_id"] in seen:
                        continue
                    seen.add(row["event_id"])
                    key = (row["phase"], row["arm"])
                    count = counts.setdefault(key, Counter())
                    count["completed"] += 1
                    count["correct"] += int(row["correct"])
                    phase = "训练核验" if row["phase"] == "protocol_train" else "测试"
                    arm = "工具组" if row["arm"] == "tools" else "直接组"
                    verdict = "对" if row["correct"] else "错"
                    if row["status"] != "answered":
                        verdict = row["status"]
                    total = value["summary"][row["phase"]][row["arm"]]["total"]
                    accuracy = count["correct"] / count["completed"] * 100
                    seconds = row["elapsed_seconds"]
                    elapsed = f"{seconds:.1f}s" if seconds is not None else "—"
                    emit(
                        f'{row["at"][11:19]} [{phase}/{arm}] {row["sample_id"]} '
                        f'{row["task_type"]} 预测={row["prediction"] or "—"} '
                        f'GT={row["gt"]} {verdict} | '
                        f'完成 {count["completed"]}/{total} '
                        f'正确 {count["correct"]}/{count["completed"]} ({accuracy:.2f}%) | '
                        f'工具 {row["successful_tools"]}/{row["tool_calls"]} '
                        f'模型请求 {row["requests"]} 用时 {elapsed}')
                current = (value["summary"], value["budget_used"])
                if current != last_summary or time.monotonic() - last_heartbeat >= 30:
                    info = value["summary"]["test"]
                    parts = []
                    for arm, name in (("tools", "工具组"), ("direct", "直接组")):
                        row = info[arm]
                        accuracy = (f'{row["accuracy_recorded"] * 100:.2f}%'
                                    if row["accuracy_recorded"] is not None else "—")
                        parts.append(
                            f'{name} {row.get("completed", 0)}/{row["total"]}，'
                            f'答对 {row.get("correct", 0)}，准确率 {accuracy}')
                    emit(f'{value["at"][11:19]} [测试汇总] ' + "；".join(parts)
                         + f'；请求 {value["budget_used"]}/{value["budget_limit"]}')
                    last_summary, last_heartbeat = current, time.monotonic()
                if value["finished"]:
                    emit("实验已完成。" if value["completed"] else
                         "实验已停止；结果和请求历史保留。")
                    return 0 if value["completed"] else 2
                time.sleep(args.interval)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--transport", type=Path, required=True)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--log", type=Path, required=True)
    parser.add_argument("--interval", type=float, default=10)
    args = parser.parse_args()
    if args.interval < 5 or not args.log.parent.is_dir():
        parser.error("interval must be at least 5 seconds and log directory must exist")
    return follow(args)


if __name__ == "__main__":
    raise SystemExit(main())
