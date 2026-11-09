"""Verify completed episode receipts and publish the fixed 20-task baseline."""
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from official_direct.planner import OfficialPlanner
from official_direct.protocol import Task
from official_direct.sources import Sources, canonical, digest, file_digest

BASE = ROOT / "official_direct"
TERMINAL = {"official_done", "invalid_action_limit", "empty_plan", "planner_error"}


def load(path):
    return json.loads(Path(path).read_text())


def audit_episode(folder, config, sources):
    folder = Path(folder).resolve()
    row = load(folder / "result.json")
    task = Task(**row["task"])
    events = [json.loads(line) for line in (folder / "trajectory.jsonl").read_text().splitlines()]
    packet, planner, sent, ack, requests, responses = None, None, [], [], 0, 0
    for event in events:
        kind = event["type"]
        if kind == "observation":
            packet = event
            if planner is None:
                planner = OfficialPlanner(sources, task, config["model"]["name"],
                                          config["n_shots"][task.environment], packet["public"]["actions"],
                                          config["model"]["response_format"])
        elif kind == "model_request":
            requests += 1
            assert event["request_index"] == requests and event["reference_answer_visible"] is False
            payload = load(folder / event["request_file"])
            assert digest(canonical(payload)) == event["payload_sha256"]
            assert payload["messages"] == planner.messages(packet["public"])
            assert payload["model"] == "qwen3.8-flash"
            assert payload["seed"] == 137 and payload["temperature"] == 0
            assert payload["max_tokens"] == 4096 and payload["enable_thinking"] is False
            assert payload["response_format"] == {"type": "json_object"}
        elif kind == "model_response":
            responses += 1
            response = load(folder / f"response-{responses:03d}.json")
            assert response == event["response"] and response["model"] == "qwen3.8-flash"
            assert vars(planner.parse(response["text"])) == event["plan"]
        elif kind == "action_sent":
            sent.append(event)
        elif kind == "action_ack":
            ack.append(event)
            assert event["action"] == sent[-1]["action"] and event["official"] == packet["official"]
            planner.feedback(event["action"], packet["public"])
    assert len(sent) == len(ack) == row["metrics"]["num_steps"]
    assert requests == responses == row["metrics"]["planner_steps"]
    assert requests <= config["max_requests"][task.environment]
    assert row["status"] in TERMINAL
    assert row["official"]["task_success"] == packet["official"]["task_success"]
    assert events[-1]["result"] == row
    assert all(event["type"] != "failure" for event in events)
    return {"request_message_replay_matches_official_public_inputs": True,
            "all_sent_actions_acknowledged_once": True, "request_and_step_counts_match": True,
            "actual_returned_model_matches": True, "no_operational_failure": True,
            "reference_answer_visible": False, "automatic_retry": False,
            "files": {str(p.relative_to(ROOT)): file_digest(p) for p in folder.iterdir() if p.is_file()}}


def main():
    run_names = sys.argv[1:]
    if not run_names:
        raise ValueError("explicit_run_names_required")
    reused = load(BASE / "verification/reuse-20261107.json")
    for value in reused["evidence"].values():
        for path, sha in value["files"].items():
            assert file_digest(ROOT / path) == sha
    rows = {Task(**row["task"]).identity: dict(row, completed=True) for row in reused["results"]}
    audits, attempts = {}, []
    sources = Sources(manipulation_catalog=BASE / "assets/manipulation_assets.json")
    sources.verify_all()
    for name in run_names:
        folder = BASE / "runs" / name
        config, identity, summary = load(folder / "config.json"), load(folder / "identity.json"), load(folder / "summary.json")
        attempts.append({"run": name, "status": summary["status"], "http_requests": summary["http_requests"],
                         "peak_http_concurrency": summary["peak_http_concurrency"],
                         "source_identity": identity, "summary": summary})
        for result in load(folder / "results.json"):
            task = Task(**result["task"])
            if result["status"] in TERMINAL:
                if task.identity in rows and rows[task.identity]["completed"]:
                    raise ValueError("duplicate_completed_baseline_task")
                episode = folder / task.directory
                audits[task.identity] = audit_episode(episode, config, sources)
                rows[task.identity] = {
                    "task": task.as_dict(), "origin": "new_real_official_baseline", "completed": True,
                    "status": result["status"], "source_result": str(episode / "result.json"),
                    "source_trace": str(episode / "trajectory.jsonl"),
                    "task_success": int(result["official"]["task_success"]),
                    "num_steps": result["metrics"]["num_steps"],
                    "model_requests": result["metrics"]["planner_steps"], "original_result": result}
            elif not rows.get(task.identity, {}).get("completed"):
                rows[task.identity] = {
                    "task": task.as_dict(), "origin": "new_attempt", "completed": False,
                    "status": result["status"], "source_result": str(folder / "results.json"),
                    "task_success": 0, "original_result": result}
    selected = [Task(kind, "base", i) for kind in ("navigation", "alfred", "habitat", "manipulation")
                for i in range(5)]
    ordered = [rows.get(t.identity, {"task": t.as_dict(), "status": "not_started", "completed": False,
                                    "task_success": 0, "origin": "unstarted"}) for t in selected]
    stats = {}
    for kind in ("navigation", "alfred", "habitat", "manipulation"):
        group = [r for r in ordered if r["task"]["environment"] == kind]
        done = sum(r["completed"] for r in group)
        success = sum(r["task_success"] for r in group if r["completed"])
        stats[kind] = {"selected": 5, "completed": done, "successes": success, "success_rate_selected": success / 5,
                       "model_plan_failures": sum(r["status"] == "planner_error" for r in group),
                       "official_done": sum(r["origin"] == "reused_real_official_baseline" or
                                            bool(r.get("original_result", {}).get("official", {}).get("done"))
                                            for r in group if r["completed"]),
                       "reused": sum(r["origin"] == "reused_real_official_baseline" for r in group),
                       "new": sum(r["origin"] == "new_real_official_baseline" for r in group)}
    completed = sum(r["completed"] for r in ordered)
    report = {"schema": "official-baseline-five/1", "status": "complete" if completed == 20 else "incomplete",
              "model": "qwen3.8-flash", "selected": 20, "completed": completed, "by_environment": stats,
              "results": ordered, "attempts": attempts, "episode_audits": audits,
              "api_concurrency_limit": 32, "gpu_concurrency_limit": 1,
              "reference_answer_visible": False, "automatic_retry": False,
              "scope": "official base indexes 0-4 per environment; this is a small baseline, not a full benchmark score",
              "historical_trace_limits": reused["limitations"]}
    (BASE / "baseline-five-results-20261107.json").write_bytes(canonical(report) + b"\n")
    text = [f"# 四环境官方直接测评基线\n\n模型：qwen3.8-flash。完成 {completed}/20 道，"
            "每环境固定选择官方 base 索引 0–4。采用官方提示、示例、公开观测和完整动作计划；"
            "无 Skill、额外工具或参考答案。\n",
            "| 环境 | 已完成 | 成功 | 成功率（固定分母 5） | 复用 | 新测 |",
            "|---|---:|---:|---:|---:|---:|---:|"]
    for kind, stat in stats.items():
        text.append(f"| {kind} | {stat['completed']}/5 | {stat['successes']} | "
                    f"{stat['success_rate_selected']:.0%} | {stat['reused']} | {stat['new']} |")
    text += ["\n复用 9 道历史官方多动作测评，重新核对源码哈希、逐步轨迹、图像、原始审计和新旧提示一致性。"
             "旧的 15 道首动作 Direct 与 Manipulation 首动作验收不计入本次成绩。",
             "\nAPI 并发上限 32；仿真总上限 8，ALFRED 上限 2、Manipulation 上限 3，"
             "Navigation/Habitat 在 GPU 7 上合计同时 1 道，启动前至少保留 4096 MiB 可用显存。"
             "实际 HTTP 峰值：" + str(max(a["peak_http_concurrency"] for a in attempts)) + "。",
             "\n首轮因 PyRep 原生库路径在 Python 启动后设置而停止，0 次模型请求；"
             "第二轮 Manipulation 请求返回 HTTP 400，其 JSON-object 输入缺少 JSON 格式指令。"
             "修复采用固定版本官方 template_manip，恢复原 schema 提供的输出格式说明。"
             "每次修复后使用新部署和新运行身份，保留旧失败轨迹。所有运行均无自动重试。",
             "\n模型参数：temperature=0、seed=137、max_tokens=4096、enable_thinking=false、JSON-object。"
             "官方动作上限：Navigation 20，ALFRED/Habitat 30（累计无效动作 10），Manipulation 15。",
             "\n新测每条模型消息均通过公开观测重放比对，动作发送与回执一一对应，"
             "实际返回模型与请求、步数核对见 JSON 的 episode_audits。"
             "历史轨迹保留消息/响应哈希和逐步动作回执，未保留原始模型请求/响应正文。",
             "\n| 题目 | 来源 | 状态 | 成功 | 步数 | 模型请求 | 结果 |",
             "|---|---|---|---:|---:|---:|---|"]
    for row in ordered:
        task = Task(**row["task"])
        path = row.get("source_result", "")
        link = f"[记录](file://{path})" if path else "—"
        text.append(f"| {task.identity} | {'复用' if row['origin']=='reused_real_official_baseline' else '新测'} | "
                    f"{row['status']} | {row['task_success']} | {row.get('num_steps','—')} | "
                    f"{row.get('model_requests','—')} | {link} |")
    text.append("\n模型计划无效而提前终止的题目数：" + str(sum(s["model_plan_failures"] for s in stats.values())) +
                "，保留原始输出并计失败。官方终止回执覆盖 " +
                str(sum(s["official_done"] for s in stats.values())) + "/20 道。"
                "最终补跑操作性失败数：" + str(sum(
                    s["operational_failures"] for s in attempts[-1]["summary"]["by_environment"].values())) + "。")
    text.append("\n这是 20 道固定抽样基线，不能作为全量 EmbodiedBench 成绩。")
    (BASE / "baseline-five-report-20261107.md").write_text("\n".join(text) + "\n")
    print(json.dumps({"status": report["status"], "completed": completed, "by_environment": stats}))


if __name__ == "__main__":
    main()
