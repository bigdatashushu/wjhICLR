"""Audit this run without changing any previously frozen source or result file."""
import base64
from datetime import datetime, timezone
import importlib
import io
import json
from pathlib import Path
import re
import sys

import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
OUT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from rgbd_three_envs.sources import canonical, digest, file_digest

KINDS = ("navigation", "alfred", "habitat", "manipulation")
TERMINAL = ("official_done", "planner_error", "empty_plan", "request_budget", "invalid_action_limit")


def load(path):
    return json.loads(Path(path).read_text())


def events(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines()]


def audit_episode(folder, config, sources, pin, package):
    module = lambda name: importlib.import_module(package + "." + name)
    row = load(folder / "result.json")
    task = module("protocol").Task(**row["task"])
    client = module("client").ModelClient(config["model"])
    planner, packet, first, plan = None, None, None, None
    sent, ack, decisions, observations = [], [], [], []
    requests, responses = 0, 0
    usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    trace = events(folder / "trajectory.jsonl")
    for event in trace:
        kind = event["type"]
        if kind == "episode_start":
            assert event["task"] == task.as_dict()
            assert event["automatic_retry"] is False and event["reference_answer_visible"] is False
        elif kind == "runtime_preflight":
            report = event["report"]
            assert report["implementation_hashes"] == pin["implementation_hashes"]
            assert report["source_hashes"] == pin["source_hashes"]
            assert report["model_requests"] == report["simulator_launches"] == 0
        elif kind == "observation":
            packet = {k: event[k] for k in ("task", "public", "official", "input_sha256")}
            module("protocol").validate_packet(packet, task)
            public, serial = packet["public"], event["serial"]
            depth = public["depth"]
            assert serial == len(observations)
            assert file_digest(folder / f"observation-{serial:03d}-0.png") == public["images"][0]["sha256"]
            assert file_digest(folder / f"observation-{serial:03d}-depth.png") == depth["image"]["sha256"]
            raw = np.load(folder / f"observation-{serial:03d}-depth-meters.npy", allow_pickle=False)
            assert raw.shape == (500, 500) and raw.dtype == np.dtype("<f4")
            assert digest(raw.tobytes()) == depth["raw"]["sha256"]
            assert np.array_equal(raw, module("depth").decode_raw(depth), equal_nan=True)
            observations.append({
                "serial": serial, "step": public["step"], "same_event": True,
                "rgb_sha256": public["images"][0]["sha256"],
                "depth_png_sha256": depth["image"]["sha256"],
                "raw_depth_sha256": depth["raw"]["sha256"], "stats": depth["stats"],
            })
            if first is None:
                first = packet
                assert public["step"] == 0 and not packet["official"]["done"]
                planner = module("planner").OfficialPlanner(
                    sources, task, config["model"]["name"], config["n_shots"][task.environment],
                    public["actions"], config["model"]["response_format"])
        elif kind == "model_request":
            requests += 1
            assert event["request_index"] == requests
            assert event["reference_answer_visible"] is False and event["model_tool_calls"] == 0
            payload = load(folder / event["request_file"])
            assert digest(canonical(payload)) == event["payload_sha256"]
            assert payload == client.payload(planner.messages(packet["public"]), planner.schema)
            hashes = [digest(base64.b64decode(part["image_url"]["url"].split(",", 1)[1]))
                      for message in payload["messages"] for part in message["content"]
                      if part["type"] == "image_url"]
            assert hashes == [packet["public"]["images"][0]["sha256"],
                              packet["public"]["depth"]["image"]["sha256"]]
            assert packet["public"]["depth"]["raw"]["base64"] not in canonical(payload).decode()
            decisions.append({"request": requests, "step": packet["public"]["step"], "image_hashes": hashes})
        elif kind == "model_response":
            responses += 1
            response = load(folder / f"response-{responses:03d}.json")
            assert event["request_index"] == responses
            assert response == event["response"] and response["model"] == config["model"]["name"]
            plan = planner.parse(response["text"])
            assert vars(plan) == event["plan"]
            for key in usage:
                usage[key] += response["usage"].get(key, 0)
        elif kind == "plan_execution":
            remaining = module("sources").LIMITS[task.environment] - packet["public"]["step"]
            assert event["executed_prefix"] == plan.actions[:remaining]
            assert event["proposed_count"] == plan.proposed_count
        elif kind == "action_sent":
            assert len(sent) == len(ack), "unacknowledged_action_before_next_send"
            assert event["expected_step"] == len(sent) + 1
            assert event["action"] == plan.actions[event["index_in_plan"]]
            sent.append(event)
        elif kind == "action_ack":
            assert len(sent) == len(ack) + 1
            assert event["action"] == sent[-1]["action"] and event["official"] == packet["official"]
            assert packet["public"]["step"] == len(sent)
            ack.append(event)
            planner.feedback(event["action"], packet["public"])
        elif kind in ("failure", "cleanup_failure"):
            raise AssertionError("operational_failure_preserved")
        elif kind == "episode_end":
            assert event["result"] == row
    assert row["status"] in TERMINAL
    assert requests == responses == row["metrics"]["planner_steps"] <= config["max_requests"][task.environment]
    assert len(sent) == len(ack) == row["metrics"]["num_steps"] <= module("sources").LIMITS[task.environment]
    if row["status"] == "empty_plan" and task.environment == "manipulation":
        packet["official"].update(task_success=0, episode_elapsed_seconds=0)
    assert row["official"] == packet["official"]
    assert trace[-1]["type"] == "episode_end" and first is not None and requests > 0
    return {
        "task": task.as_dict(), "status": row["status"], "success": int(row["official"]["task_success"]),
        "steps": len(sent), "requests": requests, "observations": len(observations), "usage": usage,
        "all_requests_have_rgb_then_depth": True, "public_message_replay_exact": True,
        "all_depth_displays_match_raw_values": True, "all_actions_acknowledged_once": True,
        "full_native_plan_execution_preserved": True, "reference_answer_visible": False,
        "automatic_retry": False, "first_input_sha256": first["input_sha256"],
        "first_rgb_sha256": first["public"]["images"][0]["sha256"],
        "first_instruction_sha256": digest(first["public"]["instruction"].encode()),
        "observation_receipts": observations, "decision_receipts": decisions,
        "result_sha256": file_digest(folder / "result.json"),
        "trajectory_sha256": file_digest(folder / "trajectory.jsonl"),
    }, first, row


def original_first(old):
    task = old["task"]
    trace = events(old["source_trace"])
    if old["origin"] == "new_real_official_baseline":
        return next(e for e in trace if e["type"] == "observation")
    selected = next(e for e in trace if e.get("environment_step") == 0 and e.get("request_index") == 1
                    and (task["environment"] != "navigation" or e.get("task_id") == f"base:{task['index']:03d}"))
    return {"legacy": True, "rgb_sha256": selected["image_sha256"], "input_sha256": selected.get("input_sha256"),
            "action_space_sha256": selected.get("action_space_sha256")}


def render(initials, pairs):
    try:
        font = ImageFont.truetype("/System/Library/Fonts/Supplemental/Arial.ttf", 17)
    except OSError:
        font = ImageFont.load_default()
    width, band = 1250, 570
    canvas = Image.new("RGB", (width, band * 4), "white")
    draw = ImageDraw.Draw(canvas)
    for row, environment in enumerate(KINDS):
        draw.text((8, row * band + 8), environment.upper() + " | RGB above, depth below (white near / black >=10m)",
                  fill="#18212a", font=font)
        for column in range(5):
            key = f"{environment}/base:{column:03d}"
            public, pair = initials[key]["public"], pairs[key]
            for index, value in enumerate((public["images"][0], public["depth"]["image"])):
                with Image.open(io.BytesIO(base64.b64decode(value["base64"]))) as image:
                    resized = image.convert("RGB").resize((250, 250))
                    canvas.paste(resized, (column * 250, row * band + 36 + index * 250))
            draw.text((column * 250 + 6, row * band + 543),
                      f"base:{column:03d}  RGB {pair['baseline_success']} -> RGB-D {pair['rgbd_success']}",
                      fill="#18212a", font=font)
    canvas.save(OUT / "rgbd20-initial-pairs.png")
    for row, environment in enumerate(KINDS):
        canvas.crop((0, row * band, width, (row + 1) * band)).save(OUT / f"rgbd20-{environment}.png")


def main():
    frozen = load(OUT / "frozen.json")
    controller = load(OUT / "controller.json")
    assert controller["status"] == "finished"
    baseline_path = ROOT / "official_direct/baseline-five-results-20261107.json"
    assert file_digest(baseline_path) == frozen["baseline_sha256"]
    baseline = load(baseline_path)
    old = {f"{r['task']['environment']}/base:{r['task']['index']:03d}": r for r in baseline["results"]}
    preserved = load(OUT / "preserved.json")
    assert all(file_digest(ROOT / relative) == sha for relative, sha in preserved.items())
    assert load(OUT / "server-after.json")["status"] == "verified"
    audits, pairs, initials, summaries = {}, {}, {}, {}
    for package, pin in frozen["packages"].items():
        config = load(OUT / pin["config"])
        assert file_digest(OUT / pin["config"]) == pin["config_sha256"]
        source_module = importlib.import_module(package + ".sources")
        assert source_module.implementation_hashes() == pin["implementation_hashes"]
        sources = source_module.Sources(manipulation_catalog=ROOT / "official_direct/assets/manipulation_assets.json")
        sources.verify_all()
        assert sources.hashes == pin["source_hashes"]
        run = ROOT / package / "runs" / pin["run_id"]
        identity, summary = load(run / "identity.json"), load(run / "summary.json")
        assert load(run / "config.json") == config
        assert summary["status"] == "finished" and summary["selected"] == len(pin["episodes"])
        assert identity["implementation_hashes"] == pin["implementation_hashes"]
        assert identity["source_hashes"] == pin["source_hashes"]
        assert identity["selected_tasks"] == pin["episodes"]
        assert identity["reference_answer_visible"] is False and identity["automatic_retry"] is False
        summaries[package] = summary
        for selected in pin["episodes"]:
            task = importlib.import_module(package + ".protocol").Task(**selected)
            audit, first, result = audit_episode(run / task.directory, config, sources, pin, package)
            audits[task.identity], initials[task.identity] = audit, first
            previous = old[task.identity]
            original = original_first(previous)
            old_rgb = original["rgb_sha256"] if original.get("legacy") else original["public"]["images"][0]["sha256"]
            same_instruction = None if original.get("legacy") else original["public"]["instruction"] == first["public"]["instruction"]
            instruction_identity_method = "direct_text"
            if same_instruction is not None:
                assert same_instruction
            if task.environment == "navigation" and original.get("legacy"):
                assert load(previous["source_result"])["instruction_sha256"] == audit["first_instruction_sha256"]
                same_instruction = True
                instruction_identity_method = "historical_instruction_sha256"
            elif original.get("legacy"):
                path = ("embodiedbench/envs/eb_alfred/data/splits/splits.json" if task.environment == "alfred"
                        else f"embodiedbench/envs/eb_habitat/datasets/{task.subset}.pickle")
                expected = digest(canonical({
                    "environment": task.environment, "subset": task.subset, "index": task.index,
                    "dataset_sha256": digest(sources.read(task.environment, path)),
                    "instruction": first["public"]["instruction"],
                }))
                assert original["input_sha256"] == expected
                same_instruction = True
                instruction_identity_method = "historical_dataset_task_instruction_sha256"
            pixel_match, pixel_diff = (True, 0.0) if old_rgb == audit["first_rgb_sha256"] else (None, None)
            if not original.get("legacy"):
                def pixels(image):
                    with Image.open(io.BytesIO(base64.b64decode(image["base64"]))) as decoded:
                        return np.asarray(decoded.convert("RGB")).astype(np.int16)
                before = pixels(original["public"]["images"][0])
                after = pixels(first["public"]["images"][0])
                pixel_match = np.array_equal(before, after)
                pixel_diff = float(np.mean(np.abs(before - after)))
            action_space_match = (
                original["action_space_sha256"] == digest(canonical(first["public"]["actions"]))
                if original.get("legacy") and original.get("action_space_sha256") else
                original["public"]["actions"] == first["public"]["actions"]
                if not original.get("legacy") else None)
            manipulation_input_match = (
                original["public"]["manipulation"] == first["public"]["manipulation"]
                if task.environment == "manipulation" else None)
            pairs[task.identity] = {
                "task": selected, "instruction": result["metrics"]["instruction"],
                "baseline_success": previous["task_success"], "rgbd_success": audit["success"],
                "baseline_status": previous["status"], "rgbd_status": audit["status"],
                "baseline_steps": previous["num_steps"], "rgbd_steps": audit["steps"],
                "baseline_requests": previous["model_requests"], "rgbd_requests": audit["requests"],
                "same_official_task_index": True, "initial_instruction_text_matches": same_instruction,
                "instruction_identity_method": instruction_identity_method,
                "baseline_first_rgb_sha256": old_rgb, "rgbd_first_rgb_sha256": audit["first_rgb_sha256"],
                "initial_rgb_hash_matches": old_rgb == audit["first_rgb_sha256"],
                "initial_rgb_pixels_match": pixel_match, "initial_rgb_pixel_mean_abs_difference": pixel_diff,
                "initial_action_space_matches": action_space_match,
                "initial_official_manipulation_input_matches": manipulation_input_match,
                "baseline_origin": previous["origin"], "baseline_result": previous["source_result"],
                "rgbd_result": str(run / task.directory / "result.json"),
                "rgbd_trace": str(run / task.directory / "trajectory.jsonl"),
                "official_done": result["official"]["done"], "invalid_actions": result["metrics"]["num_invalid_actions"],
            }
        assert sum(a["requests"] for a in audits.values() if a["task"] in pin["episodes"]) == summary["http_requests"]
    assert len(audits) == len(pairs) == len(frozen["selected"]) == 20
    pattern = re.compile(rb"(?<![A-Za-z0-9])sk-[A-Za-z0-9_.+/-]{16,}")
    checked = 0
    for directory in [OUT] + [ROOT / p / "runs" / pin["run_id"] for p, pin in frozen["packages"].items()]:
        for path in directory.rglob("*"):
            if path.is_file() and path.suffix in (".json", ".jsonl", ".md"):
                assert pattern.search(path.read_bytes()) is None, "credential_pattern_in_artifact"
                checked += 1
    statistics = {}
    for environment in KINDS:
        group = [pair for pair in pairs.values() if pair["task"]["environment"] == environment]
        statistics[environment] = {
            "selected": 5, "completed": 5, "baseline_successes": sum(p["baseline_success"] for p in group),
            "rgbd_successes": sum(p["rgbd_success"] for p in group),
            "baseline_requests": sum(p["baseline_requests"] for p in group),
            "baseline_steps": sum(p["baseline_steps"] for p in group),
            "rgbd_requests": sum(p["rgbd_requests"] for p in group),
            "rgbd_steps": sum(p["rgbd_steps"] for p in group),
            "changed_to_success": sum(p["baseline_success"] == 0 and p["rgbd_success"] == 1 for p in group),
            "changed_to_failure": sum(p["baseline_success"] == 1 and p["rgbd_success"] == 0 for p in group),
            "initial_rgb_hash_matches": sum(p["initial_rgb_hash_matches"] for p in group),
        }
    report = {
        "schema": "synchronized-rgbd-twenty/1", "status": "verified", "selected": 20, "completed": 20,
        "checked_at": datetime.now(timezone.utc).isoformat(), "model": config["model"],
        "baseline_successes": sum(p["baseline_success"] for p in pairs.values()),
        "rgbd_successes": sum(p["rgbd_success"] for p in pairs.values()),
        "rgbd_requests": sum(a["requests"] for a in audits.values()),
        "observations": sum(a["observations"] for a in audits.values()),
        "rgbd_steps": sum(a["steps"] for a in audits.values()),
        "official_done": sum(p["official_done"] for p in pairs.values()),
        "planner_terminal_failures": sum(p["rgbd_status"] in ("empty_plan", "planner_error", "request_budget")
                                        for p in pairs.values()),
        "changed_to_success": sum(p["baseline_success"] == 0 and p["rgbd_success"] == 1 for p in pairs.values()),
        "changed_to_failure": sum(p["baseline_success"] == 1 and p["rgbd_success"] == 0 for p in pairs.values()),
        "total_usage": {k: sum(a["usage"][k] for a in audits.values())
                        for k in ("prompt_tokens", "completion_tokens", "total_tokens")},
        "initial_rgb_hash_matches": sum(p["initial_rgb_hash_matches"] for p in pairs.values()),
        "by_environment": statistics, "pairs": pairs, "episode_audits": audits, "summaries": summaries,
        "executed_package_order": [phase["package"] for phase in controller["phases"]],
        "scheduler_note": "Canonical JSON sorted package keys: three environments ran before Navigation; "
                          "all twenty episodes remained sequential with the frozen settings.",
        "reference_answer_visible": False, "automatic_retry": False, "operational_failures": 0,
        "all_requests_have_two_synchronized_images": True, "old_files_unchanged": len(preserved),
        "credential_pattern_scan_files": checked, "active_owned_workers": [],
        "depth_shape": [500, 500], "depth_display_range_meters": [0, 10], "depth_raw_unit": "meters",
        "comparison_scope": "Same twenty official indexes and model parameters; descriptive small-sample comparison. "
                            "Initial RGB hash differences and reused historical traces prevent a strict paired causal claim.",
    }
    render(initials, pairs)
    (OUT / "rgbd20-results.json").write_bytes(canonical(report) + b"\n")
    text = [
        "# 原 20 题同步 RGB-D 复测\n",
        f"完成并审计 20/20 题。原 RGB 成功 {report['baseline_successes']}/20（{report['baseline_successes']/20:.0%}），"
        f"本次 RGB-D 成功 {report['rgbd_successes']}/20（{report['rgbd_successes']/20:.0%}）。"
        "四环境均使用官方 base 索引 0–4；本轮全部重新运行。\n",
        "| 环境 | 原 RGB | RGB-D | 成功率变化 | 模型请求 | 动作步数 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for environment, stat in statistics.items():
        text.append(f"| {environment} | {stat['baseline_successes']}/5 | {stat['rgbd_successes']}/5 | "
                    f"{(stat['rgbd_successes']-stat['baseline_successes'])*20:+d} 个百分点 | "
                    f"{stat['rgbd_requests']} | {stat['rgbd_steps']} |")
    text += [
        "\n模型：qwen3.8-flash；temperature=0，seed=137，max_tokens=4096，enable_thinking=false，"
        "JSON-object。沿用官方提示、示例和完整动作计划。仿真串行运行，GPU 渲染使用物理卡 7，"
        "ALFRED/Manipulation 使用 CPU 软件渲染。API 并发允许 32，本轮实际峰值为 1。\n",
        "实际执行顺序为 ALFRED → Habitat → Manipulation → Navigation，全部串行。"
        "准备清单写了 Navigation 先执行，但 canonical JSON 的包名排序使三环境包先执行；"
        "保留原清单，实际顺序见 controller.json。\n",
        "每次模型决策输入两张 500×500 图：当前 RGB 与同一观测的原生仿真深度。"
        "固定 0–10 米反向灰度：白色近，黑色表示 10 米及以上或无效像素。"
        "原始 float32 米制深度另存 NPY。模型消息只含 RGB 和深度可视化图，"
        "没有额外工具调用或参考答案。\n",
        f"审计覆盖 {report['rgbd_requests']} 次模型请求、{report['observations']} 次同步观测和 "
        f"{report['rgbd_steps']} 个动作回执；逐请求重放一致、深度 PNG 与原始数值一致、动作发送与 ACK 一一对应。"
        f"操作性失败 0，自动重试 0。{len(preserved)} 个原实验文件哈希保持不变，远程自有 worker 已全部退出。\n",
        f"相对原 RGB，{report['changed_to_success']} 题由失败变成功，"
        f"{report['changed_to_failure']} 题由成功变失败。"
        "成功以官方环境回执为准；模型自行声称任务已完成仍可能计为失败。\n",
        f"官方环境终止回执覆盖 {report['official_done']}/20 题，"
        f"模型空计划、无效计划或请求预算触发的提前终止共 {report['planner_terminal_failures']} 题，均计失败。\n",
        f"初始 RGB 图像哈希与基线一致 {report['initial_rgb_hash_matches']}/20 题。"
        "原基线包含 9 题历史复用记录，其中没有原始模型请求/响应正文；"
        "当前结果用于观察同一批题的表现，尚不能把差异全部归因于深度输入。\n",
        "| 题目 | 原 RGB → RGB-D | 当前状态 | 步数 | 请求数 | 初始 RGB 相同 | 结果 |",
        "|---|---|---|---:|---:|---|---|",
    ]
    for key, pair in pairs.items():
        text.append(f"| {key} | {pair['baseline_success']} → {pair['rgbd_success']} | {pair['rgbd_status']} | "
                    f"{pair['rgbd_steps']} | {pair['rgbd_requests']} | "
                    f"{'是' if pair['initial_rgb_hash_matches'] else '否'} | [记录](file://{pair['rgbd_result']}) |")
    text += [
        f"\n[初始 RGB/深度图总览](file://{OUT / 'rgbd20-initial-pairs.png'}) · "
        f"[逐题结果和完整审计](file://{OUT / 'rgbd20-results.json'}) · "
        f"[远程最终检查](file://{OUT / 'server-after.json'})\n",
        "深度来源：Navigation 使用 AI2-THOR 同一事件的米制深度；ALFRED 原生毫米除以 1000；"
        "Habitat 使用与 RGB 相同位置、朝向及视场角的深度传感器；"
        "Manipulation 使用同一 front 相机观测，并按 near + depth × (far − near) 转换为米。"
        "这是仿真原生深度试验；迁移至真实 RGB 输入时需要另行接入深度估计。",
    ]
    (OUT / "rgbd20-report.md").write_text("\n".join(text) + "\n")
    print(json.dumps({k: report[k] for k in ("status", "completed", "baseline_successes", "rgbd_successes",
                                          "rgbd_requests", "observations", "rgbd_steps",
                                          "initial_rgb_hash_matches", "by_environment")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
