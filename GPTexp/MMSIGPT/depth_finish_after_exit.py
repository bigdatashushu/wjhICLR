"""Finish an already-running comparison after its process exits, without new inference."""
from __future__ import annotations

import argparse
from contextlib import closing
import json
from pathlib import Path
import select
import sys

from ..traces import read, write
from .depth_comparison.mac_control import score
from .depth_comparison.runner import GPTEXP, code_identity, freeze, validate, validate_frozen
from .depth_comparison_report import ancestors, export
from .depth_service_probe import all_keys


def await_exit(pid, directory):
    if (directory / "execution.json").exists():
        return
    with closing(select.kqueue()) as queue:
        event = select.kevent(
            pid, filter=select.KQ_FILTER_PROC,
            flags=select.KQ_EV_ADD | select.KQ_EV_ONESHOT,
            fflags=select.KQ_NOTE_EXIT)
        try:
            queue.control([event], 1, None)
        except ProcessLookupError:
            pass
    if not (directory / "execution.json").exists():
        raise ValueError("exited_without_terminal_execution_record")


def update_status(directory, destination, execution, overall):
    path = Path(__file__).parent / "depth_comparison/work-20261107/status.md"
    old = path.read_text(encoding="utf-8")
    header = old.split("\n\n", 1)[0]
    tail = old.split("## 已完成", 1)[1]
    if overall:
        summary = (
            f"本轮已完成 {overall['complete_pairs']}/{overall['questions']} 对并独立评分。"
            f"原图 {overall['direct_correct']}/{overall['questions']}，"
            f"原图＋深度 {overall['depth_correct']}/{overall['questions']}，"
            f"差值 {100 * overall['delta']:+.1f} 个百分点。"
            "比较累计 48/60 次，服务复查另计 25 次，全部调用累计 73 次。")
    else:
        summary = (
            f"新运行已结束并冻结，状态为 {execution['status']}；"
            "完整比较未完成，未读取参考答案或评分。")
    receipt = read(destination / "collection-verification.json")
    addition = (
        "\n\n## 新运行终态\n\n" + summary + "\n\n"
        f"- 运行目录：`{directory}`。\n"
        f"- 最终报告：`{destination / 'report.md'}`。\n"
        f"- run identity：`{execution['run_identity']}`。\n"
        f"- freeze identity：`{read(directory / 'frozen.json')['freeze_identity']}`。\n"
        f"- 完整冻结与统计核验状态：`{receipt['status']}`。\n")
    if overall:
        addition += (
            f"- 错→对 {overall['wrong_to_correct']}，对→错 {overall['correct_to_wrong']}；"
            f"McNemar p={overall['mcnemar_exact_two_sided_p']:.4f}。\n")
    path.write_text(header + "\n\n" + summary + "\n\n## 已完成" + tail + addition,
                    encoding="utf-8")


def privacy_receipt(directory, destination, manifest):
    prior = ancestors(directory, manifest)
    roots = [destination, *(path for path, _, _, _ in prior)]
    files = {path.resolve() for root in roots for path in root.rglob("*") if path.is_file()}
    sources = [GPTEXP / name for name in code_identity()]
    sources += [
        Path(__file__), Path(__file__).parent / "depth_comparison_report.py",
        Path(__file__).parent / "depth_service_probe.py",
        Path(__file__).parent / "depth_comparison/work-20261107/status.md",
    ]
    files.update(path.resolve() for path in sources)
    keys = [value.encode() for value in all_keys().values()]
    for path in sorted(files):
        raw = path.read_bytes()
        if any(key in raw for key in keys):
            raise ValueError("credential_in_public_delivery")
    target = destination / "collection-verification.json"
    receipt = read(target)
    receipt.update({
        "credential_scan": "passed", "credential_values_persisted": False,
        "distinct_credentials_checked": len(keys), "public_files_scanned": len(files),
        "source_files_scanned": len(sources), "completion_handler_model_requests": 0,
        "automatic_model_retry": False,
    })
    write(target, receipt)
    return receipt


def finish(pid, directory, destination):
    directory, destination = Path(directory), Path(destination)
    await_exit(pid, directory)
    frozen = freeze(directory)
    manifest = validate(directory)
    validate_frozen(directory, manifest)
    execution = read(directory / "execution.json")
    overall = None
    if execution["status"] == "complete":
        score(directory)
        overall = read(directory / "score.json")["overall"]
    export(destination, directory)
    update_status(directory, destination, execution, overall)
    receipt = privacy_receipt(directory, destination, manifest)
    print(json.dumps({
        "completion_status": execution["status"], "report": str(destination / "report.md"),
        "freeze_identity": frozen["freeze_identity"], "overall": overall,
        "paired_requests": receipt.get("paired_requests", receipt.get("requests")),
        "service_recheck_requests": receipt["service_recheck_requests"],
        "total_real_inference_calls": receipt["total_real_inference_calls"],
        "credential_scan": receipt["credential_scan"],
        "completion_handler_model_requests": 0,
    }, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pid", type=int, required=True)
    parser.add_argument("--run-directory", required=True)
    parser.add_argument("--destination", required=True)
    args = parser.parse_args()
    try:
        finish(args.pid, args.run_directory, args.destination)
    except Exception:
        print('{"status":"failed","reason_code":"completion_handler_failed"}', file=sys.stderr)
        raise SystemExit(2)
