"""Replay public model inputs, action receipts and compare frozen five-task runs."""
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from manipulation_minimal.client import ModelClient
from manipulation_minimal.planner import OfficialPlanner
from manipulation_minimal.protocol import Task, validate_packet
from manipulation_minimal.sources import Sources, canonical, digest, file_digest, implementation_hashes

BASE = ROOT / "official_direct/runs/baseline-five-20261107-r03"
PACKAGE = ROOT / "manipulation_minimal"
TRIAL = PACKAGE / "runs/single-step-visual-five-20261107-r01"


def load(path):
    return json.loads(Path(path).read_text())


def events(folder):
    return [json.loads(line) for line in (folder / "trajectory.jsonl").read_text().splitlines()]


def audit(folder, config, sources, frozen):
    result = load(folder / "result.json")
    task = Task(**result["task"])
    trace = events(folder)
    packet, planner, plan = None, None, None
    requests, responses, sends, acks = [], [], [], []
    decisions, frame_history, notes, observations = [], [], [], []
    usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    builder = ModelClient(config["model"])
    for event in trace:
        kind = event["type"]
        if kind == "episode_start":
            assert event["reference_answer_visible"] is False and event["automatic_retry"] is False
        elif kind == "runtime_preflight":
            report = event["report"]
            assert report["simulator_launches"] == report["model_requests"] == 0
            assert report["implementation_hashes"] == frozen["implementation_hashes"]
            assert report["source_hashes"] == frozen["source_hashes"]
        elif kind == "observation":
            packet = {k: event[k] for k in ("task", "public", "official", "input_sha256")}
            validate_packet(packet, task)
            observations.append(packet)
            if planner is None:
                planner = OfficialPlanner(sources, task, config["model"]["name"],
                                          config["n_shots"]["manipulation"], [],
                                          config["model"]["response_format"])
        elif kind == "model_request":
            requests.append(event)
            assert event["request_index"] == len(requests)
            assert event["reference_answer_visible"] is False and event["model_tool_calls"] == 0
            payload = load(folder / event["request_file"])
            assert digest(canonical(payload)) == event["payload_sha256"]
            replay = builder.payload(planner.messages(packet["public"]), planner.schema)
            assert replay == payload
            decisions.append(packet["public"]["step"])
            image_hashes = []
            for message in payload["messages"]:
                for part in message["content"]:
                    if part["type"] == "image_url":
                        import base64
                        image_hashes.append(digest(base64.b64decode(part["image_url"]["url"].split(",", 1)[1])))
            current_hash = packet["public"]["images"][0]["sha256"]
            expected = [frame_history[-1], current_hash] if frame_history else [current_hash]
            assert image_hashes == expected
            frame_history.append(current_hash)
        elif kind == "model_response":
            responses.append(event)
            response = load(folder / f"response-{len(responses):03d}.json")
            assert response == event["response"]
            assert response["model"] == config["model"]["name"]
            plan = planner.parse(response["text"])
            assert vars(plan) == event["plan"]
            for key in usage:
                usage[key] += response["usage"].get(key, 0)
            value = json.loads(response["text"])
            notes.append({
                "request": len(responses), "environment_step": decisions[-1],
                "visual_state": value.get("visual_state_description"),
                "reasoning": value.get("reasoning_and_reflection"),
                "language_plan": value.get("language_plan"), "proposed_actions": plan.actions})
        elif kind == "plan_execution":
            assert event["executed_prefix"] == plan.actions[:1]
        elif kind == "action_sent":
            sends.append(event)
            assert event["index_in_plan"] == 0
            assert event["request_index"] == len(requests)
            assert event["action"] == plan.actions[0]
            assert event["expected_step"] == len(sends)
        elif kind == "action_ack":
            acks.append(event)
            assert event["action"] == sends[-1]["action"]
            assert event["official"] == packet["official"]
            planner.feedback(event["action"], packet["public"])
        elif kind in ("failure", "cleanup_failure"):
            raise AssertionError(f"operational_failure_preserved:{task.identity}")
    assert result["status"] in ("official_done", "empty_plan", "planner_error", "request_budget")
    assert len(sends) == len(acks) == result["metrics"]["num_steps"] <= 15
    assert len(requests) == len(responses) == result["metrics"]["planner_steps"] <= 15
    assert len({event["request_index"] for event in sends}) == len(sends)
    expected_official = dict(observations[-1]["official"])
    if result["status"] == "empty_plan":
        expected_official.update(task_success=0, episode_elapsed_seconds=0)
    assert result["official"] == expected_official
    assert trace[-1]["type"] == "episode_end" and trace[-1]["result"] == result
    return {
        "public_request_replay_exact": True, "adjacent_image_hashes_match": True,
        "one_action_per_request": True, "all_actions_acknowledged_once": True,
        "actual_model_matches": True, "automatic_retry": False,
        "reference_answer_visible": False, "decisions_at_steps": decisions,
        "token_usage": usage, "model_decision_notes": notes,
        "trajectory_sha256": file_digest(folder / "trajectory.jsonl"),
        "result_sha256": file_digest(folder / "result.json")}


def main():
    frozen = load(PACKAGE / "verification/frozen-manifest.json")
    config, identity, summary = [load(TRIAL / name) for name in ("config.json", "identity.json", "summary.json")]
    assert summary["status"] == "finished" and summary["selected"] == 5
    assert summary["by_environment"]["manipulation"]["started"] == 5
    assert summary["by_environment"]["manipulation"]["operational_failures"] == 0
    assert identity["implementation_hashes"] == implementation_hashes() == frozen["implementation_hashes"]
    assert identity["source_hashes"] == frozen["source_hashes"]
    assert identity["reference_answer_visible"] is False and identity["automatic_retry"] is False
    assert file_digest(PACKAGE / "experiment-20261107-r01.json") == frozen["config_sha256"]
    assert config == load(PACKAGE / "experiment-20261107-r01.json")
    parent = load(BASE / "config.json")
    for key in ("model", "n_shots", "max_requests", "execution"):
        assert config[key] == parent[key]
    assert config["episodes"] == [t for t in parent["episodes"] if t["environment"] == "manipulation"]
    for key in ("source_root", "manipulation_catalog", "manipulation_dataset",
                "python", "gpu", "timeout_s", "environment_seed", "gpu_min_free_mb"):
        assert config["runtime"][key] == parent["runtime"][key]
    assert identity["source_hashes"] == load(BASE / "identity.json")["source_hashes"]
    preserved = load(PACKAGE / "verification/preserved-official-direct.json")
    assert all(file_digest(ROOT / name) == sha for name, sha in preserved.items())
    assert sorted(p for p in preserved) == sorted(
        str(p.relative_to(ROOT)) for p in (ROOT / "official_direct").rglob("*") if p.is_file())
    sources = Sources(manipulation_catalog=ROOT / "official_direct/assets/manipulation_assets.json")
    sources.verify_all()
    pairs, audits = [], {}
    for i in range(5):
        task = Task("manipulation", "base", i)
        old, new = BASE / task.directory, TRIAL / task.directory
        before, after = load(old / "result.json"), load(new / "result.json")
        first_old = next(e for e in events(old) if e["type"] == "observation")
        first_new = next(e for e in events(new) if e["type"] == "observation")
        checks = {
            "task": before["task"] == after["task"] == task.as_dict(),
            "instruction": first_old["public"]["instruction"] == first_new["public"]["instruction"],
            "input_sha256": first_old["input_sha256"] == first_new["input_sha256"],
            "initial_rgb_sha256": first_old["public"]["images"][0]["sha256"] == first_new["public"]["images"][0]["sha256"],
            "initial_anonymous_coordinates": first_old["public"]["manipulation"] == first_new["public"]["manipulation"]}
        pairs.append({
            "task": task.as_dict(), "instruction": after["metrics"]["instruction"],
            "baseline_success": int(before["official"]["task_success"]),
            "trial_success": int(after["official"]["task_success"]),
            "baseline_steps": before["metrics"]["num_steps"], "trial_steps": after["metrics"]["num_steps"],
            "baseline_requests": before["metrics"]["planner_steps"], "trial_requests": after["metrics"]["planner_steps"],
            "baseline_invalid_actions": before["metrics"]["num_invalid_actions"],
            "trial_invalid_actions": after["metrics"]["num_invalid_actions"],
            "status": after["status"], "initial_input_checks": checks,
            "initial_inputs_identical": all(checks.values()),
            "baseline_result": str(old / "result.json"), "trial_result": str(new / "result.json"),
            "trial_trace": str(new / "trajectory.jsonl")})
        audits[task.identity] = audit(new, config, sources, frozen)
    report = {
        "schema": "manipulation-single-step-visual-comparison/1", "status": "verified",
        "selected": 5, "completed": 5, "baseline_successes": sum(p["baseline_success"] for p in pairs),
        "trial_successes": sum(p["trial_success"] for p in pairs),
        "identical_initial_inputs": sum(p["initial_inputs_identical"] for p in pairs),
        "all_frozen_parameters_match": True, "old_files_unchanged": len(preserved),
        "http_requests": summary["http_requests"], "peak_http_concurrency": summary["peak_http_concurrency"],
        "automatic_retry": False, "reference_answer_visible": False,
        "pairs": pairs, "audits": audits,
        "scope": "Fixed five-task diagnostic; single-step and visual policy tested together.",
        "summary_sha256": file_digest(TRIAL / "summary.json")}
    (PACKAGE / "comparison-20261107.json").write_bytes(canonical(report) + b"\n")
    text = [
        "# Manipulation 单步执行与抓取视觉核验试验",
        "",
        f"原基线成功 {report['baseline_successes']}/5，独立试验成功 {report['trial_successes']}/5。"
        f"完成 5/5 道，初始公开图像、匿名坐标、指令和题目身份逐题一致的有 "
        f"{report['identical_initial_inputs']}/5 道。",
        "",
        "固定 qwen3.8-flash、temperature=0、seed=137、max_tokens=4096、"
        "enable_thinking=false、JSON-object、官方四条文本示例、每题 15 步。"
        "运行目录和源码独立冻结，官方环境、所选资产和环境随机种子保持一致。",
        "",
        "本次同时改变：①每次只执行第一个动作并重新观察；②增加抓取视觉核验指引；"
        "③附带前一次决策的公开图像、匿名坐标和已执行动作。"
        "宿主没有抓取状态传感器；核验仍依赖模型看图。",
        "",
        "| 官方 base 索引 | 基线成功 | 新版成功 | 基线步数 | 新版步数 | 基线请求 | 新版请求 | 新版无效动作 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for pair in pairs:
        text.append(
            f"| {pair['task']['index']:03d} | {pair['baseline_success']} | {pair['trial_success']} | "
            f"{pair['baseline_steps']} | {pair['trial_steps']} | {pair['baseline_requests']} | "
            f"{pair['trial_requests']} | {pair['trial_invalid_actions']} |")
    baseline_requests = sum(p["baseline_requests"] for p in pairs)
    text += [
        "",
        f"实际模型请求：基线 {baseline_requests} 次，新版 {report['http_requests']} 次。"
        f"新版 API 并发上限 32，实际峰值 {report['peak_http_concurrency']}；仿真并发上限 3。"
        "没有自动重试或动作重发，没有操作性失败。",
        "",
        f"审计通过：逐条请求从公开观测精确重放，前后帧图像哈希匹配；"
        f"每个请求最多执行一个动作；动作发送与回执一一对应；"
        f"实际返回模型及参数匹配；原 official_direct 的 {len(preserved)} 个文件内容和文件集合未变。",
        "",
        "五题为诊断样本；本次组合试验不能拆分判定单步执行、提示指引和相邻帧的各自作用。"
        "模型采样与仿真仍可能有重复运行波动。",
        "",
        "[五题最终公开图像对照](file://" + str(PACKAGE / "verification/final-comparison.png") + ")",
        "",
        "## 轨迹判断",
        "",
        "- 新增成功为 base:004：基线首先把青色星形物体作为三棱柱，"
        "新版首先选中蓝色三棱柱，随后下降、闭合、抬升核验、运送并释放，6 步获官方成功。"
        "请求 5 的核验图像显示该物体随夹爪升高，原始坐标从约 z=18 变为 z=25。"
        "本题从第一次决策就改变了目标选择，因此不能把新增成功仅归因于抓取核验。",
        "- base:001 保持成功，5 步完成；抬升后模型查看新图像再运送。",
        "- base:000 成功抬起橙色星形物体，但在两个容器的语义与坐标之间反复切换；"
        "后续声称任务已完成并反复开夹爪、移动。最终官方失败。",
        "- base:002 在星形、三棱柱和圆柱之间反复误认、改选目标。"
        "请求 7 在闭合后尚未执行核验抬升，就因物体仍在桌面而打开夹爪，"
        "说明短指引仍不能保证模型持续遵循核验流程。15 步后官方失败。",
        "- base:003 把左侧圆柱误认成三棱柱，抬起圆柱后又据其移动确认抓取成功；"
        "真正的三棱柱持续留在桌面右侧。动作均有效但目标错误，官方失败。"
        "本题无效动作从基线 6 次降为 0 次，没有转化为任务成功。",
        "",
        "本轮增加了可检查的动作后观察，并改善一题；三道遗留失败仍显示"
        "物体形状、容器语义与匿名坐标的绑定不稳定。后续若继续，应优先检验"
        "目标识别与坐标对应关系，再评价抓取后的恢复策略。",
        "",
        "## 逐题轨迹证据"]
    for pair in pairs:
        task = Task(**pair["task"])
        text += [
            "",
            f"### {task.subset}:{task.index:03d} — {pair['instruction']}",
            "",
            f"成功：{pair['baseline_success']} → {pair['trial_success']}；"
            f"新版状态：{pair['status']}。"
            f"[动作与图像回执](file://{pair['trial_trace']}) · "
            f"[官方结果](file://{pair['trial_result']})",
            ""]
        for note in audits[task.identity]["model_decision_notes"]:
            text.append(
                f"- 请求 {note['request']}（环境步 {note['environment_step']}）："
                f"动作 {note['proposed_actions']}。模型推断：{note['reasoning']}")
    text += [
        "",
        f"[完整结构化对照与核验](file://{PACKAGE / 'comparison-20261107.json'})",
        "",
        "上述“模型推断”是模型原始轨迹中的判断，不能单独证明真实抓取状态。"
        "实际行为还应结合每步公开图像与官方任务终态。"]
    (PACKAGE / "comparison-20261107.md").write_text("\n".join(text) + "\n")
    print(json.dumps({k: report[k] for k in (
        "status", "completed", "baseline_successes", "trial_successes",
        "identical_initial_inputs", "http_requests", "peak_http_concurrency", "old_files_unchanged")}))


if __name__ == "__main__":
    main()
