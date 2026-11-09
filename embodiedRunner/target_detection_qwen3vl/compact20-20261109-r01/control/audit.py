"""Offline artifact/action audit and comparison against the exact original list."""
import base64
from collections import Counter
from datetime import datetime, timezone
import hashlib
import io
import json
from pathlib import Path
import sys

OUT = Path(__file__).resolve().parents[1]
ROOT = OUT.parents[1]
frozen = json.loads((OUT / "frozen.json").read_text())
sys.path.insert(0, frozen["source_root"])
from target_detection_qwen3vl.config import load_config
from target_detection_qwen3vl.detector import Detector
from target_detection_qwen3vl.sources import Sources, canonical, file_digest, implementation_hashes
from target_detection_r05.__main__ import check
from target_detection_r05.lifecycle import remote
from target_detection_r05.protocol import Task


def load(path):
    return json.loads(Path(path).read_text())


def events(folder):
    return [json.loads(line) for line in (folder / "trajectory.jsonl").read_text().splitlines()]


def total_usage(rows):
    return {k: sum(row.get("usage", {}).get(k, 0) for row in rows)
            for k in ("prompt_tokens", "completion_tokens", "total_tokens")}


def first_public(folder):
    return next((e["public"] for e in events(folder) if e["type"] == "observation"), None)


def success(result):
    return int(result["status"] not in ("not_started", "operational_error", "stopped_after_peer_failure")
               and bool(result.get("official")) and result["official"].get("task_success") == 1)


def audit_episode(folder, config):
    from PIL import Image
    result, trace = load(folder / "result.json"), events(folder)
    assert trace[-1]["type"] == "episode_end" and trace[-1]["result"] == result
    sent, acks, model_sent, model_received, detects, detect_sent = [], [], [], [], [], []
    public_by_serial = {}
    for e in trace:
        kind = e["type"]
        if kind == "observation":
            public = e["public"]
            public_by_serial[e["serial"]] = public
            raw = base64.b64decode(public["raw_rgb"]["base64"])
            assert hashlib.sha256(raw).hexdigest() == public["raw_rgb"]["sha256"]
            assert (folder / f'observation-{e["serial"]:03d}-raw.png').read_bytes() == raw
        elif kind == "model_request":
            payload = load(folder / e["request_file"])
            assert hashlib.sha256(canonical(payload)).hexdigest() == e["payload_sha256"]
            assert payload["model"] == config["model"]["name"]
            assert payload["temperature"] == config["model"]["temperature"]
            assert payload["max_tokens"] == config["model"]["max_tokens"]
            for key, value in config["model"]["extra_body"].items():
                assert payload[key] == value
            assert e["reference_answer_visible"] is False
            assert payload["messages"][0]["role"] == "system"
            model_sent.append(e)
        elif kind == "model_response_received":
            response = load(folder / e["response_file"])
            assert response["usage"] == e["usage"]
            model_received.append(e)
        elif kind == "detector_request_sent":
            detect_sent.append(e)
        elif kind == "detector_result":
            report = e["report"]
            assert report["image_sha256"] == public_by_serial[e["observation_serial"]]["raw_rgb"]["sha256"]
            assert report["model"] == config["detection"]["service"]["model"]
            assert report["service_identity_sha256"] == hashlib.sha256(canonical(config["detection"]["service"])).hexdigest()
            assert report["automatic_retry"] is False and report["reference_answer_visible"] is False
            detects.append(report)
        elif kind == "perception_tool_result":
            for image in e.get("images", []):
                assert file_digest(folder / image["file"]) == image["sha256"]
        elif kind == "action_sent":
            sent.append(e)
        elif kind == "action_ack":
            assert len(acks) < len(sent)
            assert e["action"] == sent[len(acks)]["action"]
            acks.append(e)
    assert len(model_sent) == result["metrics"]["planner_steps"]
    assert len(model_sent) <= config["max_requests"][result["task"]["environment"]]
    assert len(detects) == result["perception"]["detector_requests"]
    if result["status"] != "operational_error":
        assert len(sent) == len(acks)
    assert trace[0]["reference_answer_visible"] is False
    return {"trajectory_sha256": file_digest(folder / "trajectory.jsonl"),
            "result_sha256": file_digest(folder / "result.json"),
            "planner_requests": len(model_sent), "planner_responses": len(model_received),
            "planner_usage": total_usage(model_received),
            "detector_requests_sent": len(detect_sent), "detector_requests": len(detects),
            "detector_usage": total_usage(detects), "actions": len(sent), "acks": len(acks),
            "unknown_action_outcomes": len(sent) - len(acks),
            "invalid_actions": result["metrics"]["num_invalid_actions"],
            "host_elapsed_seconds": result["host_elapsed_seconds"],
            "verification": "artifact hashes, returned usage, fixed request parameters and action ACKs"}


def after(config):
    code = r'''
import datetime,hashlib,json,os
from pathlib import Path
root=Path(ROOT_VALUE)
processes=[]
for proc in Path("/proc").iterdir():
    if not proc.name.isdigit(): continue
    try:
        if proc.stat().st_uid!=os.getuid(): continue
        raw=(proc/"cmdline").read_bytes()
        cwd=str((proc/"cwd").resolve())
        ours=any(a.startswith(str(root).encode()+b"/") for a in raw.split(b"\0"))
        if not ours and not cwd.startswith(str(root)+"/"): continue
        stat=(proc/"stat").read_text().split(") ",1)[1].split()
        processes.append({"pid":int(proc.name),"start_ticks":int(stat[19]),
                          "command_sha256":hashlib.sha256(raw).hexdigest(),"cwd":cwd})
    except (FileNotFoundError,PermissionError,ProcessLookupError): pass
pins={str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest()
      for folder in ("target_detection_r05","target_detection_qwen3vl")
      for p in sorted((root/folder).glob("*.py"))}
print(json.dumps({"checked_at":datetime.datetime.now(datetime.timezone.utc).isoformat(),
 "owned_run_processes":processes,"implementation_hashes":pins}))
'''.replace("ROOT_VALUE", repr(config["runtime"]["code_root"]))
    state = remote(config, code, timeout=45)
    assert state["implementation_hashes"] == frozen["implementation_hashes"]
    state["detector"] = Detector(config).status()
    (OUT / "verification/server-after.json").write_bytes(canonical(state) + b"\n")
    return state


def finish():
    from PIL import Image
    config, tasks = load_config(OUT / "config.json")
    for name, pin in frozen["snapshot_files"].items():
        assert file_digest(OUT / name) == pin, "frozen_snapshot_changed"
    assert implementation_hashes() == frozen["implementation_hashes"]
    sources = Sources(frozen["input_root"], OUT / "inputs/manipulation_assets.json")
    check(config, tasks, sources)
    assert sources.hashes == frozen["source_hashes"]
    preserved = load(OUT / "verification/preserved-history.json")
    for name, pin in preserved.items():
        assert file_digest(ROOT / name) == pin, "history_changed"
    baseline = load(OUT / "baseline.json")
    previous = {p["identity"]: p for p in baseline["pairs"]}
    pairs, audits = [], []
    for task in tasks:
        folder = OUT / "run" / task.directory
        old = previous[task.identity]
        old_result = load(Path(old["directory"]) / "result.json")
        new = load(folder / "result.json") if (folder / "result.json").exists() else None
        row = {"identity": task.identity, "task": task.as_dict(),
               "previous_success": old["current_success"],
               "previous_status": old_result["status"], "directory": str(folder),
               "current_success": success(new) if new else None,
               "status": new["status"] if new else "not_started"}
        if new:
            audit = audit_episode(folder, config)
            row.update(error=new.get("error"), metrics=new["metrics"], audit=audit)
            audits.append(audit)
            a, b = first_public(folder), first_public(Path(old["directory"]))
            initial = {"available": a is not None and b is not None}
            if initial["available"]:
                initial.update(
                    instruction_equal=a["instruction"] == b["instruction"],
                    actions_equal=a["actions"] == b["actions"],
                    rgb_pixels_equal=Image.open(io.BytesIO(base64.b64decode(a["raw_rgb"]["base64"]))).tobytes()
                    == Image.open(io.BytesIO(base64.b64decode(b["raw_rgb"]["base64"]))).tobytes(),
                    depth_png_equal=a["depth"]["image"]["sha256"] == b["depth"]["image"]["sha256"])
            row["initial_identity"] = initial
            (OUT / "verification" / (task.directory + "-audit.json")).write_bytes(canonical(audit) + b"\n")
        pairs.append(row)
    server = after(config)
    summary = load(OUT / "run/summary.json")
    completed = sum(p["status"] != "not_started" for p in pairs)
    full = (completed == len(tasks) and summary["status"] == "finished"
            and not server["owned_run_processes"]
            and all(p.get("audit", {}).get("unknown_action_outcomes", 0) == 0 for p in pairs))
    report = {
        "checked_at": datetime.now(timezone.utc).isoformat(), "selected": len(tasks),
        "completed": completed, "final_results_available": full,
        "status": "completed" if full else "interrupted",
        "current_successes": sum(p["current_success"] or 0 for p in pairs),
        "previous_successes": baseline["current_successes"], "pairs": pairs,
        "changed_to_success": [p["identity"] for p in pairs if p["current_success"] == 1 and p["previous_success"] == 0],
        "changed_to_failure": [p["identity"] for p in pairs if p["current_success"] == 0 and p["previous_success"] == 1],
        "planner_requests": sum(a["planner_requests"] for a in audits),
        "planner_usage": {k: sum(a["planner_usage"][k] for a in audits) for k in baseline["planner_usage"]},
        "detector_requests": sum(a["detector_requests"] for a in audits),
        "detector_usage": {k: sum(a["detector_usage"][k] for a in audits) for k in baseline["detector_usage"]},
        "actions": sum(a["actions"] for a in audits), "acks": sum(a["acks"] for a in audits),
        "unknown_action_outcomes": sum(a["unknown_action_outcomes"] for a in audits),
        "host_elapsed_seconds": sum(a["host_elapsed_seconds"] for a in audits),
        "status_counts": dict(Counter(p["status"] for p in pairs)),
        "initial_identity_counts": {k: sum(p.get("initial_identity", {}).get(k, False) for p in pairs)
                                  for k in ("instruction_equal", "actions_equal", "rgb_pixels_equal", "depth_png_equal")},
        "history_files_verified": len(preserved), "frozen_snapshot_files_verified": len(frozen["snapshot_files"]),
        "owned_run_processes": server["owned_run_processes"], "automatic_retry": False,
        "by_environment": {}}
    for env in ("alfred", "habitat", "manipulation", "navigation"):
        rows = [p for p in pairs if p["task"]["environment"] == env]
        report["by_environment"][env] = {
            "selected": len(rows), "completed": sum(p["status"] != "not_started" for p in rows),
            "successes": sum(p["current_success"] or 0 for p in rows),
            "previous_successes": sum(p["previous_success"] for p in rows)}
    (OUT / "results.json").write_bytes(canonical(report) + b"\n")
    lines = [
        "# 上下文压缩与合并规划：原四环境 20 题复跑", "",
        f"状态：{'全部完成并核验' if full else '运行中断，尚非完整结果'}。已结束 {completed}/20 题。",
        f"新版成功 {report['current_successes']}/20；原 Qwen3-VL 流程成功 {baseline['current_successes']}/20。",
        "采用原题单：四环境 base 0–4，各 5 题。主模型 qwen3.8-flash，辅助检测 Qwen3-VL-8B-Instruct-FP8。",
        "模型采样参数、种子、环境动作上限及规划请求预算保持原配置；few-shot 降至最多 3。",
        "本轮测试当前开发版：工具协议仅置于 system，自动准备检测与最多两张裁剪，视觉复核与完整动作计划合并输出，补充 ALFRED 单手规则。", "",
        "| 环境 | 原流程 | 新版 | 已结束 |", "| --- | --- | --- | --- |"]
    for env, data in report["by_environment"].items():
        lines.append(f"| {env} | {data['previous_successes']}/5 | {data['successes']}/5 | {data['completed']}/5 |")
    lines += ["", f"规划请求：原 {baseline['planner_requests']} 次，新 {report['planner_requests']} 次。",
              f"规划 tokens：原 {baseline['planner_usage']['total_tokens']:,}，新 {report['planner_usage']['total_tokens']:,}。",
              f"检测返回：原 {baseline['detector_requests']} 次，新 {report['detector_requests']} 次。",
              f"整题运行累计：原 {baseline['host_elapsed_seconds']/60:.1f} 分钟，新 {report['host_elapsed_seconds']/60:.1f} 分钟。",
              "", "| 题目 | 原结果 | 新结果 | 停止状态 | 错误代码 |",
              "| --- | --- | --- | --- | --- |"]
    for p in pairs:
        lines.append(f"| {p['identity']} | {p['previous_success']} | {p['current_success'] if p['current_success'] is not None else '未启动'} | {p['status']} | {(p.get('error') or {}).get('code', '—')} |")
    lines += ["", "转为成功：" + (", ".join(report["changed_to_success"]) or "无") + "。",
              "转为失败：" + (", ".join(report["changed_to_failure"]) or "无") + "。",
              f"动作 ACK：{report['acks']}/{report['actions']}；未知动作结果 {report['unknown_action_outcomes']}。",
              "初始观测一致性：" + "，".join(f"{k}={v}/20" for k, v in report["initial_identity_counts"].items()) + "。",
              f"核验 {len(preserved)} 个历史文件及 {len(frozen['snapshot_files'])} 个冻结文件；本轮远端残留 {len(server['owned_run_processes'])}。",
              "独立运行仅执行一次，不自动重试。差值反映这组样本上整条流程的表现，包含提示改动、模型输出和仿真观测波动。",
              "原始请求、响应、观测与工具轨迹位于 run/；逐题核验回执位于 verification/。", ""]
    (OUT / "实验结果.md").write_text("\n".join(lines))
    return report


if __name__ == "__main__":
    print(json.dumps({k: v for k, v in finish().items() if k in
                     ("status", "selected", "completed", "current_successes", "final_results_available")}))
