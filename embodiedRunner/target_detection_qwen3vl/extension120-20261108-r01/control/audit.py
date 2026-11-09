"""Audit immutable recorded outputs and aggregate the selected 120 + existing 20."""
from collections import Counter
from datetime import datetime, timezone
import importlib.util
import json
from pathlib import Path
import sys

TMP = Path(__file__).parent
ROOT = Path("/Users/bytedance/wangjh/wjhICLR/embodiedRunner")
BASE = ROOT / "target_detection_qwen3vl/twenty-20261108-r01"
OUT = ROOT / "target_detection_qwen3vl/extension120-20261108-r01"
sys.path.insert(0, str(OUT / "source_frozen/embodiedRunner"))
from target_detection_qwen3vl.sources import Sources, canonical, file_digest, implementation_hashes
from target_detection_r05.protocol import Task

spec = importlib.util.spec_from_file_location("previous_audit", BASE / "verification/report.py")
previous = importlib.util.module_from_spec(spec)
spec.loader.exec_module(previous)
previous.OUT = OUT
# The audited method code is unchanged. Supply the new selected-input catalogs.
previous.Sources = lambda *args, **kwargs: Sources(
    OUT / "inputs/embodiedRunner", OUT / "inputs/manipulation_assets.json")


def load(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def save(path, value):
    Path(path).write_bytes(canonical(value) + b"\n")


def roots():
    return [(OUT, OUT / "run")] + [
        (p.parent, p) for p in sorted((OUT / "continuations").glob("*/run"))]


def folders():
    result = {}
    for _, run in roots():
        for path in run.glob("*/result.json"):
            assert path.parent.name not in result, "task_started_twice"
            with (path.parent / "trajectory.jsonl").open("rb") as stream:
                stream.seek(max(0, stream.seek(0, 2) - 64))
                tail = stream.read()
            if tail.endswith(b',"type":"episode_end"}\n'):
                result[path.parent.name] = path.parent
    return result


def started():
    values = [p.parent.name for _, run in roots() for p in run.glob("*/trajectory.jsonl")]
    assert len(values) == len(set(values)), "task_started_twice"
    return set(values)


def status():
    completed = [load(p / "result.json") for p in folders().values()]
    completed_names = {Task(**r["task"]).directory for r in completed}
    active = []
    for _, run in roots():
        for trace in run.glob("*/trajectory.jsonl"):
            if trace.parent.name in completed_names:
                continue
            raw = trace.read_bytes()
            rows = [json.loads(line) for line in raw.split(b"\n")[:-1] if line]
            public = next((e["public"] for e in reversed(rows) if e["type"] == "observation"), None)
            active.append({"task": trace.parent.name, "last_event": rows[-1]["type"] if rows else None,
                           "step": public["step"] if public else None,
                           "planning_requests": sum(e["type"] == "model_request" for e in rows),
                           "detector_receipts": sum(e["type"] == "detector_result" for e in rows)})
    return {
        "completed": len(completed), "selected": 120,
        "successes": sum(r["status"] != "operational_error"
                         and r["metrics"]["task_success"] == 1 for r in completed),
        "by_environment": {
            env: {"completed": sum(r["task"]["environment"] == env for r in completed),
                  "successes": sum(r["task"]["environment"] == env
                                   and r["status"] != "operational_error"
                                   and r["metrics"]["task_success"] == 1 for r in completed)}
            for env in ("alfred", "habitat", "manipulation", "navigation")},
        "active": active, "runs": len(roots()),
        "controller": load(OUT / "controller-status.json") if (OUT / "controller-status.json").exists() else None,
        "final_results_available": (OUT / "results.json").exists(),
    }


def completed_audits(names=None):
    config = load(OUT / "config.json")
    rows = []
    for name, folder in sorted(folders().items()):
        if names is not None and name not in names:
            continue
        identity = {"trajectory_sha256": file_digest(folder / "trajectory.jsonl"),
                    "result_sha256": file_digest(folder / "result.json"),
                    "audit_script_sha256": file_digest(__file__),
                    "reused_audit_script_sha256": file_digest(BASE / "verification/report.py")}
        target = OUT / "verification" / (name + "-audit.json")
        cached = load(target) if target.exists() else None
        if cached and cached["identity"] == identity:
            audit = cached["audit"]
        else:
            audit = previous.audit_episode(folder, config)
            save(target, {"identity": identity, "audit": audit})
        result = load(folder / "result.json")
        task = Task(**result["task"])
        rows.append({
            "identity": task.identity, "task": result["task"],
            "success": int(result["status"] != "operational_error" and result["metrics"]["task_success"] == 1),
            "status": result["status"], "error": result["error"], "metrics": result["metrics"],
            "targets": result["perception"]["targets"], "audit": audit,
            "directory": str(folder.resolve()), "cohort": "new120",
        })
    return rows


def aggregate(rows, selected):
    return {
        "selected": selected, "completed": len(rows),
        "successes": sum(p["success"] for p in rows),
        "success_rate": sum(p["success"] for p in rows) / selected,
        "operational_failures": sum(p["status"] == "operational_error" for p in rows),
        "status_counts": dict(Counter(p["status"] for p in rows)),
        "planner_requests": sum(p["audit"]["model_requests"] for p in rows),
        "detector_requests": sum(p["audit"]["detector_requests"] for p in rows),
        "detector_requests_sent": sum(p["audit"]["detector_requests_sent"] for p in rows),
        "detector_requests_without_result": sum(p["audit"]["detector_requests_without_result"] for p in rows),
        "actions": sum(p["audit"]["actions"] for p in rows),
        "acks": sum(p["audit"]["acks"] for p in rows),
        "planner_usage": {k: sum(p["audit"]["planner_usage"][k] for p in rows)
                          for k in ("prompt_tokens", "completion_tokens", "total_tokens")},
        "detector_usage": {k: sum(p["audit"]["detector_usage"][k] for p in rows)
                           for k in ("prompt_tokens", "completion_tokens", "total_tokens")},
        "host_elapsed_seconds": sum(p["audit"]["host_elapsed_seconds"] for p in rows),
    }


def preservation():
    before = load(OUT / "verification/original20-preserved.json")
    assert all((BASE / name).is_file() and file_digest(BASE / name) == value
               for name, value in before.items()), "original_twenty_changed"
    frozen = load(OUT / "frozen.json")
    assert implementation_hashes() == frozen["implementation_hashes"]
    assert all(file_digest(OUT / n) == h for n, h in frozen["snapshot_files"].items())
    assert file_digest(OUT / "config.json") == frozen["config_sha256"]
    assert file_digest(OUT / "sampling.json") == frozen["sampling_sha256"]
    sources = previous.Sources()
    counts = sources.verify_all()
    assert sources.hashes == frozen["official_source_hashes"]
    for metadata, run in roots():
        if run.exists():
            assert load(run / "identity.json")["implementation_hashes"] == frozen["implementation_hashes"]
            assert load(run / "identity.json")["source_hashes"] == sources.hashes
        if metadata != OUT:
            assert file_digest(metadata / "config.json") == load(metadata / "prepared.json")["config_sha256"]
    return {"checked_at": datetime.now(timezone.utc).isoformat(),
            "original20_preserved_files": len(before), "official_source_counts": counts,
            "snapshot_changes": [], "input_identity_verified": True,
            "model_requests": 0, "simulator_launches": 0}


def finish():
    frozen, config = load(OUT / "frozen.json"), load(OUT / "config.json")
    assert len(folders()) == 120 and started() == set(folders())
    assert all((metadata / "exit.json").exists() for metadata, _ in roots())
    preserved = preservation()
    save(OUT / "verification/preservation-after.json", preserved)
    state = load(OUT / "verification/server-after.json")
    assert not state["owned_trial_workers"]
    assert (OUT / "verification/detector-after.json").is_file()
    new = completed_audits()
    old_report = load(BASE / "results.json")
    old = [{"identity": p["identity"], "task": p["task"], "success": p["current_success"],
            "status": p["status"], "error": p["error"], "metrics": p["metrics"],
            "targets": p["targets"], "audit": p["audit"], "directory": p["directory"],
            "cohort": "existing20"} for p in old_report["pairs"]]
    combined = old + new
    assert len({p["identity"] for p in combined}) == 140
    assert {p["identity"] for p in new} == {Task(**t).identity for t in config["episodes"]}
    summaries = [load(run / "summary.json") for _, run in roots()]
    new_summary = aggregate(new, 120)
    assert sum(s["http_requests"] for s in summaries) == new_summary["planner_requests"]
    assert sum(s["detector_requests"] for s in summaries) == new_summary["detector_requests"]
    assert all(s["peak_http_concurrency"] <= 1 for s in summaries)
    assert new_summary["planner_requests"] <= 3450
    environments = {}
    for env in ("alfred", "habitat", "manipulation", "navigation"):
        old_rows = [p for p in old if p["task"]["environment"] == env]
        new_rows = [p for p in new if p["task"]["environment"] == env]
        assert len(old_rows) == 5 and len(new_rows) == 30
        environments[env] = {"existing20": aggregate(old_rows, 5), "new120": aggregate(new_rows, 30),
                             "combined140": aggregate(old_rows + new_rows, 35)}
    report = {
        "checked_at": datetime.now(timezone.utc).isoformat(),
        "existing20": aggregate(old, 20), "new120": new_summary,
        "combined140": aggregate(combined, 140), "by_environment": environments,
        "episodes": combined, "sampling_seed": 137, "new_per_environment": 30,
        "same_method_as_original20": frozen["same_method_as_original20"],
        "planning_model_request_cap_new120": 3450,
        "operational_failure_score": 0, "automatic_retry": False, "generative_audit_requests": 0,
        "peak_planner_concurrency": max(s["peak_http_concurrency"] for s in summaries),
        "runs": [{"run": str(run), "summary": summary} for (_, run), summary in zip(roots(), summaries)],
        "original20_results_sha256": file_digest(BASE / "results.json"),
        "final_verification": preserved,
        "paired_dino_comparison_scope": "existing20_only",
        "unknown_action_outcomes": [p["identity"] for p in combined if p["audit"]["unknown_action_outcome"]],
    }
    save(OUT / "results.json", report)
    lines = [
        "# Qwen3-VL 检测流程：140 题结果", "",
        f"新增 120 题已完成，成功 {new_summary['successes']}/120（{new_summary['success_rate']:.2%}）；"
        f"原 20 题成功 {report['existing20']['successes']}/20。"
        f"合并成功 {report['combined140']['successes']}/140（{report['combined140']['success_rate']:.2%}）。", "",
        "| 环境 | 原 5 题 | 新增 30 题 | 合并 35 题 | 合并成功率 |",
        "| --- | --- | --- | --- | --- |",
    ]
    for env, row in environments.items():
        a, b, c = row["existing20"], row["new120"], row["combined140"]
        lines.append(f"| {env} | {a['successes']}/5 | {b['successes']}/30 | "
                     f"{c['successes']}/35 | {c['success_rate']:.2%} |")
    total = report["combined140"]
    lines += [
        "", "新增样本使用 seed=137，按四类环境的官方 base 池分别无放回随机抽取 30 题，"
        "排除已有 20 题。模型仍为 Qwen3.8-Flash 规划、Qwen3-VL-8B-Instruct-FP8 检测；"
        "提示源码、few-shot、temperature=0、4096 token 输出上限、关闭 thinking、"
        "仿真种子、逐题请求预算和串行执行配置均沿用原轮。",
        "新增资产按固定仓库 revision 安装在独立目录，并核验 SHA256 或 Git blob 身份。", "",
        f"合并规划请求 {total['planner_requests']:,} 次，"
        f"{total['planner_usage']['total_tokens']:,} tokens；"
        f"检测返回 {total['detector_requests']:,} 次，"
        f"{total['detector_usage']['total_tokens']:,} tokens，"
        f"另有 {total['detector_requests_without_result']} 次检测无返回回执。"
        f"动作发送 {total['actions']} 次、ACK {total['acks']} 次。",
        f"新增运营失败 {new_summary['operational_failures']} 题，合并运营失败 "
        f"{total['operational_failures']} 题，均计 0 分且保留原始轨迹。"
        "停止后只继续尚未启动的题目，不重试已开始的失败题、不补样。",
        "全部新增题已离线重放请求、目标提取、检测、裁剪复核与动作执行顺序，"
        "原 20 题结果按已核验的文件哈希保留。", "",
        "原 GroundingDINO 只有同组 20 题对照；140 题结果是本检测流程的分层样本分数，"
        "没有新增 120 题的 DINO 成对对照。", "",
        "| 题目 | 批次 | 成功 | 状态 | 错误代码 |", "| --- | --- | --- | --- | --- |",
    ]
    for p in combined:
        lines.append(f"| {p['identity']} | {p['cohort']} | {p['success']} | {p['status']} | "
                     f"{p['error']['code'] if p['error'] else '—'} |")
    (OUT / "实验结果.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {"status": "finished", "new120": new_summary, "combined140": report["combined140"]}


if __name__ == "__main__":
    command = sys.argv[1] if len(sys.argv) > 1 else "status"
    result = (status() if command == "status" else
              {"verified": len(completed_audits()), "generative_audit_requests": 0}
              if command == "audit" else finish())
    print(canonical(result).decode(), flush=True)
