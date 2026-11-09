"""Read-only run monitoring, public-trajectory collection and archived comparison."""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import shlex
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT.parent / "src"))
REMOTE = "/home/cvailab/wangjh/embodied-action-v2-task-card-15-20261106-r01/embodiedRunner"
STEM = "task-card-15-full-r01"
SSH = ["ssh", "-T", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10",
       "-S", str(ROOT / "checks/ssh4090")]
HOST = "cvailab@10.10.0.198"


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b""):
            h.update(block)
    return h.hexdigest()


def save(path, value):
    with Path(path).open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, sort_keys=True, indent=2)
        stream.write("\n")


def status():
    source = f"""
import pathlib,json,hashlib,collections,time
root=pathlib.Path({REMOTE!r})/"runs"
stem={STEM!r}
start=json.loads((root/(stem+"-startup.json")).read_text())
p=pathlib.Path("/proc")/str(start["pid"])
alive=p.exists()
identity=False
if alive:
    stat=(p/"stat").read_text().rsplit(")",1)[1].split()
    identity=(int(stat[19])==start["start_ticks"] and
              hashlib.sha256((p/"cmdline").read_bytes()).hexdigest()==start["cmdline_sha256"])
completion=root/(stem+"-completion.json")
rows=[]
for f in sorted((root/(stem+"-episodes")).glob("attempt-*.json")):
    r=json.loads(f.read_text())
    rows.append({{k:r.get(k) for k in ("episode_id","status","terminal_reason",
                                    "task_success","task_progress","environment_steps")}})
journals=sorted((root/(stem+"-episodes")).glob("trajectory-*/events.jsonl"))
active=None
if journals:
    f=journals[-1]
    events=[json.loads(x) for x in f.read_text().splitlines()]
    active={{"directory":f.parent.name,"event_counts":dict(collections.Counter(e["kind"] for e in events)),
             "last_kind":events[-1]["kind"],"event_age_s":round(time.time()-f.stat().st_mtime,1),
             "card_chars":[e["receipt"].get("task_card_chars") for e in events if e["kind"]=="model_request"]}}
files={{}}
for name in (stem+".json",stem+"-startup.json",stem+"-completion.json",stem+"-log.jsonl"):
    f=root/name
    if f.exists():
        files[name]={{"sha256":hashlib.sha256(f.read_bytes()).hexdigest(),"bytes":f.stat().st_size}}
print(json.dumps({{"checked_at_unix":time.time(),"node":"4090","remote_root":{REMOTE!r},
                  "process_alive":alive,"process_identity_match":identity,"attempts":rows,
                  "completion":json.loads(completion.read_text()) if completion.exists() else None,
                  "active":active,"files":files}}))
"""
    result = subprocess.run(SSH + [HOST, "python3 -c " + shlex.quote(source)],
                            capture_output=True, check=True, timeout=30)
    return json.loads(result.stdout)


def collect():
    receipt = status()
    assert receipt["completion"] and not receipt["process_alive"], "run_not_completed"
    assert receipt["completion"]["exit_code"] == 0, "worker_failed"
    patterns = [STEM + suffix for suffix in
                (".json", "-startup.json", "-completion.json", "-log.jsonl")]
    directory = STEM + "-episodes"
    patterns += [directory + "/", directory + "/manifest.json",
                 directory + "/trajectories.json", directory + "/attempt-*.json",
                 directory + "/trajectory-*-Full/",
                 directory + "/trajectory-*-Full/trajectory.json",
                 directory + "/trajectory-*-Full/events.jsonl",
                 directory + "/trajectory-*-Full/assets/",
                 directory + "/trajectory-*-Full/assets/*"]
    command = ["rsync", "-a", "--ignore-existing",
               "-e", shlex.join(SSH)]
    command += ["--include=/" + p for p in patterns]
    command += ["--exclude=*", HOST + ":" + REMOTE + "/runs/", str(ROOT / "runs") + "/"]
    subprocess.run(command, check=True, timeout=60)
    for name, identity in receipt["files"].items():
        path = ROOT / "runs" / name
        assert path.stat().st_size == identity["bytes"] and digest(path) == identity["sha256"]
    save(ROOT / "checks/task-card-15-final-status-r01.json", receipt)
    return {"status": "collected", "files": len(receipt["files"])}


def collect_stopped():
    receipt = status()
    assert receipt["completion"] and not receipt["process_alive"], "worker_still_running"
    directory = STEM + "-episodes"
    patterns = [STEM + suffix for suffix in ("-startup.json", "-completion.json", "-log.jsonl")]
    patterns += [directory + "/", directory + "/manifest.json", directory + "/attempt-*.json",
                 directory + "/trajectory-*-Full/", directory + "/trajectory-*-Full/trajectory.json",
                 directory + "/trajectory-*-Full/events.jsonl", directory + "/trajectory-*-Full/assets/",
                 directory + "/trajectory-*-Full/assets/*"]
    command = ["rsync", "-a", "--ignore-existing", "-e", shlex.join(SSH)]
    command += ["--include=/" + p for p in patterns]
    command += ["--exclude=*", HOST + ":" + REMOTE + "/runs/", str(ROOT / "runs") + "/"]
    subprocess.run(command, check=True, timeout=60)
    for name, identity in receipt["files"].items():
        path = ROOT / "runs" / name
        assert path.stat().st_size == identity["bytes"] and digest(path) == identity["sha256"]
    save(ROOT / "checks/task-card-15-stopped-collection.json", receipt)
    return {"status": "stopped_trajectories_collected", "completed_tasks": len(receipt["attempts"])}


def wait_progress():
    current = status()
    count = len(current["attempts"])
    deadline = time.monotonic() + 35
    while not current["completion"] and time.monotonic() < deadline:
        time.sleep(min(5, max(0, deadline - time.monotonic())))
        current = status()
        if len(current["attempts"]) != count:
            break
    return current


def stats(report, trajectory, directory):
    from action_host import FORMAT_ERRORS, PROGRAM_ERRORS
    from action_audit import record_hash
    from action_contract import public_hash
    from action_trajectory import restore
    assert report["sha256"] == record_hash(report)
    assert trajectory["events"][-1]["report"] == report
    assets = trajectory["assets"]
    for asset in assets:
        p = (directory / asset["path"]).resolve()
        assert p.is_relative_to(directory.resolve())
        assert p.stat().st_size == asset["bytes"] and digest(p) == asset["sha256"]
    events = restore(trajectory["events"], directory)
    requests = [e for e in events if e["kind"] == "model_request"]
    assert len(requests) == len(report["model_requests"])
    for event, row in zip(requests, report["model_requests"]):
        assert public_hash(event["messages"]) == row["messages_sha256"]
    initial = report["observations"][0] if report["observations"] else {}
    codes = Counter(e["code"] for e in report["errors"])
    submissions = [e for e in events if e["kind"] == "action_submission"]
    actions = [e["action_name"] for e in submissions]
    proposals = [e for e in events if e["kind"] == "model_proposal"]
    perception_keys = [(e["receipt"]["decision_id"], public_hash(e["perception_request"]))
                       for e in proposals if e.get("perception_request") is not None]
    reserved = report["budget"]["reserved"]
    return {
        "status": report["status"], "success": report["task_success"],
        "progress": report["task_progress"], "terminal_reason": report["terminal_reason"],
        "steps": report["environment_steps"], "invalid_actions": report["invalid_actions"],
        "qwen_requests": reserved.get("qwen", 0),
        "qwen_tokens": report["budget"]["qwen_tokens_observed"],
        "format_errors": sum(codes[k] for k in FORMAT_ERRORS),
        "program_errors": sum(codes[k] for k in PROGRAM_ERRORS), "error_codes": dict(codes),
        "requests_by_dimension": reserved,
        "request_statuses": dict(Counter(e["status"] for e in report["model_requests"])),
        "elapsed_s": events[-1]["elapsed_ms"] / 1000,
        "consecutive_same_action": sum(a == b for a, b in zip(actions, actions[1:])),
        "perception_decision_requests": len(perception_keys),
        "repeated_identical_perception_same_decision": len(perception_keys) - len(set(perception_keys)),
        "catalog_decision_requests": sum(e.get("catalog_request") is not None for e in proposals),
        "initial_input_sha256": initial.get("input_sha256"),
        "initial_rgb_sha256": initial.get("rgb_sha256"),
        "source_sha256": report["source_sha256"],
        "report_sha256": report["sha256"],
        "trajectory_path": str(directory / "trajectory.json"),
        "trajectory_file_sha256": digest(directory / "trajectory.json"),
        "assets_verified": len(assets), "asset_bytes_verified": sum(a["bytes"] for a in assets),
        "request_text_chars": [e["receipt"]["text_chars"] for e in requests],
        "system_text_chars": [len(e["messages"][0]["content"]) for e in requests],
        "task_card_chars": [e["receipt"]["task_card_chars"] for e in requests
                            if "task_card_chars" in e["receipt"]],
        "task_card_over_budget": sum(e["receipt"].get("task_card_over_budget", False) for e in requests),
    }


def aggregate(rows):
    chars = [c for r in rows for c in r["task_card_chars"]]
    texts = [c for r in rows for c in r["request_text_chars"]]
    systems = [c for r in rows for c in r["system_text_chars"]]
    return {
        "selected": len(rows), "successes": sum(r["success"] for r in rows),
        "success_rate": sum(r["success"] for r in rows) / len(rows),
        **{k: sum(r[k] for r in rows) for k in (
            "steps", "qwen_requests", "qwen_tokens", "invalid_actions", "format_errors",
            "program_errors", "assets_verified", "asset_bytes_verified", "consecutive_same_action",
            "perception_decision_requests", "repeated_identical_perception_same_decision",
            "catalog_decision_requests")},
        "elapsed_s": round(sum(r["elapsed_s"] for r in rows), 3),
        "zero_step_failures": sum(not r["success"] and r["steps"] == 0 for r in rows),
        "terminal_reasons": dict(Counter(r["terminal_reason"] for r in rows)),
        "task_card_requests": len(chars), "task_card_max_chars": max(chars, default=0),
        "task_card_mean_chars": sum(chars) / len(chars) if chars else None,
        "task_card_over_budget": sum(r["task_card_over_budget"] for r in rows),
        "request_text_max_chars": max(texts, default=0),
        "request_text_mean_chars": sum(texts) / len(texts) if texts else None,
        "system_text_max_chars": max(systems, default=0),
        "requests_by_dimension": dict(sum((Counter(r["requests_by_dimension"]) for r in rows), Counter())),
    }


def finalize():
    from action_audit import audit_run, record_hash
    from audit_action_trajectory import audit
    from task_card import validate_card
    from action_trajectory import restore
    run_path = ROOT / "runs" / (STEM + ".json")
    run = read(run_path)
    run_audit, trajectory_audit = audit_run(run), audit(run_path)
    assert run_audit["status"] == trajectory_audit["status"] == "passed"
    reference_path = ROOT / "checks/task-card-15-archive-reference-r01.json"
    reference = read(reference_path)
    assert reference["status"] == "verified" and reference["baseline_rerun_required"] is False
    assert run["manifest"]["selection"] == reference["selection"]
    assert run["manifest"]["arms"] == ["Full"] and run["manifest"]["task_cards"] is True
    old = {(r["episode_id"], r["arm"]): r for r in reference["archived_instances"]}
    index = read(run_path.with_name(STEM + "-episodes") / "trajectories.json")
    comparisons = []
    cards_checked = 0
    for entry, report in zip(index["episodes"], run["attempts"]):
        directory = run_path.with_name(STEM + "-episodes") / entry["directory"]
        trajectory = read(directory / "trajectory.json")
        new = stats(report, trajectory, directory)
        for event in restore(trajectory["events"], directory):
            if event["kind"] == "model_request":
                wire = event["messages"][1]["content"][0]["text"]
                validate_card(wire)
                assert len(wire) == event["receipt"]["task_card_chars"]
                assert event["receipt"]["task_card_complete"] is True
                cards_checked += 1
        row = {"episode_id": report["episode_id"], "environment": report["environment"], "task_card": new}
        for arm, key in (("Direct", "direct"), ("Full", "old_full")):
            ref = old[(report["episode_id"], arm)]
            path = Path(ref["trajectory_path"])
            assert digest(path) == ref["trajectory_sha256"], "archived_trajectory_changed"
            archived = read(path)
            r = archived["events"][-1]["report"]
            assert r["sha256"] == ref["report_sha256"]
            previous = stats(r, archived, path.parent)
            for field in ("success", "progress", "steps", "qwen_requests", "qwen_tokens",
                          "format_errors", "program_errors", "initial_input_sha256", "initial_rgb_sha256"):
                assert previous[field] == ref[field], "archive_reference_changed:" + field
            assert previous["source_sha256"] == new["source_sha256"]
            assert previous["initial_input_sha256"] == new["initial_input_sha256"]
            archived_run = read(path.parent.parent.parent / "comparison.json")
            assert archived_run["sha256"] == record_hash(archived_run)
            assert r in archived_run["attempts"] and r["prepared_config_sha256"] == archived_run["manifest"]["configuration"]["sha256"]
            old_config = archived_run["manifest"]["configuration"]
            current_config = run["manifest"]["configuration"]
            assert old_config["model"] == current_config["model"]
            assert old_config["budgets"][report["environment"]] == current_config["budgets"][report["environment"]]
            assert old_config["method_library_sha256"] == current_config["method_library_sha256"]
            row[key] = previous
            row["initial_rgb_matches_" + key] = previous["initial_rgb_sha256"] == new["initial_rgb_sha256"]
        comparisons.append(row)
    summaries = {k: aggregate([r[k] for r in comparisons]) for k in ("direct", "old_full", "task_card")}
    per_environment = {env: {k: aggregate([r[k] for r in comparisons if r["environment"] == env])
                              for k in summaries} for env in sorted({r["environment"] for r in comparisons})}
    changes = {key: {
        "improved": sum(r["task_card"]["success"] and not r[key]["success"] for r in comparisons),
        "regressed": sum(not r["task_card"]["success"] and r[key]["success"] for r in comparisons),
        "unchanged": sum(r["task_card"]["success"] == r[key]["success"] for r in comparisons),
    } for key in ("direct", "old_full")}
    result = {
        "schema": "embodied-task-card-15-comparison/1", "run_sha256": run["sha256"],
        "archive_reference_file_sha256": digest(reference_path), "new_direct_requests": 0,
        "comparison_scope": "15 previously used development tasks; archived Direct and old Full",
        "paired_comparison_eligible": False, "summaries": summaries, "per_environment": per_environment,
        "task_cards_checked_complete": cards_checked, "comparisons": comparisons,
        "success_changes": changes,
        "run_audit": run_audit, "trajectory_audit": trajectory_audit,
        "initial_rgb_matches": {key: sum(r["initial_rgb_matches_" + key] for r in comparisons)
                                for key in ("direct", "old_full")},
    }
    save(ROOT / "runs" / (STEM + "-comparison.json"), result)
    names = {"direct": "Qwen 直答（存档）", "old_full": "旧版 Full（存档）", "task_card": "新版任务卡 Full"}
    text = ["# 15 题任务卡复跑结果", "",
            "复用了之前的 15 个任务和 Qwen 直答存档，新增直答调用为 0。", "",
            "| 方案 | 成功数 | 成功率 | 环境步数 | Qwen 调用 | Qwen token | 格式错误 | 程序错误 |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for key, label in names.items():
        s = summaries[key]
        text.append(f"| {label} | {s['successes']}/15 | {s['success_rate']:.2%} | {s['steps']} | "
                    f"{s['qwen_requests']} | {s['qwen_tokens']} | {s['format_errors']} | {s['program_errors']} |")
    text += ["", "| 环境 | 直答 | 旧版 Full | 新版任务卡 |",
             "| --- | ---: | ---: | ---: |"]
    for env, values in per_environment.items():
        text.append("| " + env + " | " + " | ".join(f"{values[k]['successes']}/5" for k in names) + " |")
    s = summaries["task_card"]
    text += ["", f"共核验 {cards_checked} 张完整任务卡，最大 {s['task_card_max_chars']} 字符，"
             f"平均 {s['task_card_mean_chars']:.1f} 字符；必要字段超出 4000 字符目标的请求有 "
             f"{s['task_card_over_budget']} 次，均按完整卡片核验。",
             "", f"4000 字符目标指任务卡正文。加上固定系统说明和完整环境 Skill，"
             f"本轮请求文本最大 {s['request_text_max_chars']} 字符，"
             f"平均 {s['request_text_mean_chars']:.1f} 字符；图片 token 另计。",
             "", "终止原因：" + "；".join(f"{k}：{v} 题" for k, v in s["terminal_reasons"].items()) + "。",
             "", f"有 {s['zero_step_failures']} 题在执行第一个环境动作前失败。"
             "调用量和 token 减少需结合任务提前结束解释。",
             "", f"模型提出 {s['perception_decision_requests']} 次感知请求和 "
             f"{s['catalog_decision_requests']} 次目录请求，其中同一决策内完全重复的感知请求有 "
             f"{s['repeated_identical_perception_same_decision']} 次。感知/目录选择也计入每轮三次决策调用预算。",
             "", "相对旧 Full：" + f"{changes['old_full']['improved']} 题由失败变成功，"
             f"{changes['old_full']['regressed']} 题由成功变失败，{changes['old_full']['unchanged']} 题持平。"
             "相对直答：" + f"{changes['direct']['improved']} 题提高，"
             f"{changes['direct']['regressed']} 题下降，{changes['direct']['unchanged']} 题持平。",
             "", "本轮模型、预算、方法库和任务输入身份与存档一致。初始 RGB 与直答/旧 Full 一致的任务分别为 "
             f"{result['initial_rgb_matches']['direct']}/15、{result['initial_rgb_matches']['old_full']}/15。"
             "导航环境的独立渲染存在差异；本轮为已用开发任务上的诊断比较。",
             "", "| 任务 | 直答 | 旧 Full | 任务卡 | 任务卡步数 | 任务卡终止原因 |",
             "| --- | --- | --- | --- | ---: | --- |"]
    for row in comparisons:
        outcomes = ["成功" if row[k]["success"] else "失败" for k in names]
        text.append(f"| {row['episode_id']} | " + " | ".join(outcomes) +
                    f" | {row['task_card']['steps']} | {row['task_card']['terminal_reason']} |")
    text += ["", "完整轨迹：" + str(run_path.with_name(STEM + "-episodes")),
             "", "逐题指标、初始图像身份、调用状态和审计结果见同名 comparison.json。", ""]
    report_path = ROOT / "runs" / (STEM + "-comparison.md")
    with report_path.open("x", encoding="utf-8") as stream:
        stream.write("\n".join(text))
    assert "\ufffd" not in report_path.read_text(encoding="utf-8")
    save(ROOT / "checks/task-card-15-acceptance-r01.json", {
        "schema": "embodied-task-card-15-acceptance/1",
        "status": "completed" if all(r["status"] == "finished" for r in run["attempts"]) else "operational_stop",
        "selected": 15, "attempts": len(run["attempts"]), "new_direct_requests": 0,
        "run_file_sha256": digest(run_path), "comparison_file_sha256": digest(
            ROOT / "runs" / (STEM + "-comparison.json")),
        "run_audit": run_audit["status"], "trajectory_audit": trajectory_audit["status"],
        "task_cards_checked_complete": cards_checked, "summaries": summaries})
    return {"status": "finalized", "summaries": summaries, "initial_rgb_matches": result["initial_rgb_matches"]}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("operation", choices=("status", "wait", "collect", "collect-stopped", "finalize"))
    args = parser.parse_args()
    value = {"status": status, "wait": wait_progress, "collect": collect,
             "collect-stopped": collect_stopped, "finalize": finalize}[args.operation]()
    print(json.dumps(value, ensure_ascii=False))
