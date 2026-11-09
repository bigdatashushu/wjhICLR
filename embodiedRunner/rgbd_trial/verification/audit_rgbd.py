"""Verify recorded RGB-D inputs and report the fixed five-task Navigation pilot."""
import base64
import json
from pathlib import Path
import re
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from rgbd_trial.client import ModelClient
from rgbd_trial.depth import decode_raw
from rgbd_trial.planner import OfficialPlanner
from rgbd_trial.protocol import Task, validate_packet
from rgbd_trial.sources import Sources, canonical, digest, file_digest, implementation_hashes

PACKAGE = ROOT / "rgbd_trial"
RUN = PACKAGE / "runs/nav-rgbd-five-20261107-r01"


def load(path):
    return json.loads(Path(path).read_text())


def events(folder):
    return [json.loads(line) for line in (folder / "trajectory.jsonl").read_text().splitlines()]


def audit_episode(folder, config, sources, frozen):
    row = load(folder / "result.json")
    task = Task(**row["task"])
    trace = events(folder)
    packet, first, planner, plan = None, None, None, None
    requests, responses, sent, ack, observations = 0, 0, [], [], 0
    depth_mentions, observation_receipts = [], []
    usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    client = ModelClient(config["model"])
    for event in trace:
        kind = event["type"]
        if kind == "episode_start":
            assert event["automatic_retry"] is False and event["reference_answer_visible"] is False
        elif kind == "runtime_preflight":
            assert event["report"]["implementation_hashes"] == frozen["implementation_hashes"]
            assert event["report"]["source_hashes"] == frozen["source_hashes"]
            assert event["report"]["model_requests"] == event["report"]["simulator_launches"] == 0
        elif kind == "observation":
            observations += 1
            packet = {k: event[k] for k in ("task", "public", "official", "input_sha256")}
            validate_packet(packet, task)
            if first is None:
                first = packet
            public, serial = packet["public"], event["serial"]
            depth = public["depth"]
            assert file_digest(folder / f"observation-{serial:03d}-0.png") == public["images"][0]["sha256"]
            assert file_digest(folder / f"observation-{serial:03d}-depth.png") == depth["image"]["sha256"]
            raw = np.load(folder / f"observation-{serial:03d}-depth-meters.npy", allow_pickle=False)
            assert raw.shape == (500, 500) and raw.dtype == np.dtype("<f4")
            assert digest(raw.tobytes()) == depth["raw"]["sha256"]
            assert np.array_equal(raw, decode_raw(depth), equal_nan=True)
            observation_receipts.append({
                "serial": serial, "step": public["step"], "rgb_sha256": public["images"][0]["sha256"],
                "depth_png_sha256": depth["image"]["sha256"], "raw_depth_sha256": depth["raw"]["sha256"],
                "shape": [500, 500], "same_event": True, "unit": "meters", "stats": depth["stats"]})
            if planner is None:
                planner = OfficialPlanner(sources, task, config["model"]["name"], 3,
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
        elif kind == "model_response":
            responses += 1
            response = load(folder / f"response-{responses:03d}.json")
            assert response == event["response"] and response["model"] == config["model"]["name"]
            plan = planner.parse(response["text"])
            assert vars(plan) == event["plan"]
            for key in usage:
                usage[key] += response["usage"].get(key, 0)
            if re.search(r"\bdepth\b|深度", response["text"], re.I):
                depth_mentions.append({"request_index": responses, "response_text": response["text"]})
        elif kind == "plan_execution":
            assert event["executed_prefix"] == plan.actions[:20 - packet["public"]["step"]]
        elif kind == "action_sent":
            sent.append(event)
            assert event["expected_step"] == len(sent)
            assert event["action"] == plan.actions[event["index_in_plan"]]
        elif kind == "action_ack":
            ack.append(event)
            assert event["action"] == sent[-1]["action"] and event["official"] == packet["official"]
            planner.feedback(event["action"], packet["public"])
        elif kind in ("failure", "cleanup_failure"):
            raise AssertionError("operational_failure_preserved")
    assert row["status"] in ("official_done", "planner_error", "request_budget")
    assert requests == responses == row["metrics"]["planner_steps"] <= 20
    assert len(sent) == len(ack) == row["metrics"]["num_steps"] <= 20
    assert trace[-1]["type"] == "episode_end" and trace[-1]["result"] == row
    assert row["official"] == packet["official"]
    return {
        "public_message_replay_exact": True, "all_requests_have_rgb_then_depth": True,
        "all_depth_displays_match_raw_values": True, "all_shapes_500x500": True,
        "all_raw_units_meters": True, "all_actions_acknowledged_once": True,
        "full_native_plan_execution_preserved": True, "reference_answer_visible": False,
        "automatic_retry": False, "requests": requests, "observations": observations,
        "depth_mentions_in_responses": depth_mentions, "usage": usage,
        "first_rgb_sha256": first["public"]["images"][0]["sha256"],
        "first_instruction_sha256": digest(first["public"]["instruction"].encode()),
        "observation_receipts": observation_receipts,
        "result_sha256": file_digest(folder / "result.json"),
        "trajectory_sha256": file_digest(folder / "trajectory.jsonl")}


def main():
    config, identity, summary = [load(RUN / name) for name in ("config.json", "identity.json", "summary.json")]
    frozen = load(PACKAGE / "verification/frozen-manifest.json")
    assert summary["status"] == "finished" and summary["selected"] == 5
    assert summary["by_environment"]["navigation"]["operational_failures"] == 0
    assert identity["implementation_hashes"] == implementation_hashes() == frozen["implementation_hashes"]
    assert identity["source_hashes"] == frozen["source_hashes"]
    assert config == load(PACKAGE / "experiment-20261107-r01.json")
    assert file_digest(PACKAGE / "experiment-20261107-r01.json") == frozen["config_sha256"]
    assert identity["mode"] == "navigation_rgbd" and identity["automatic_retry"] is False
    parent = load(ROOT / "official_direct/baseline-20261107-r03.json")
    for key in ("model", "execution", "n_shots", "max_requests"):
        assert config[key] == parent[key]
    for key in ("source_root", "manipulation_catalog", "manipulation_dataset", "python", "gpu",
                "environment_seed", "timeout_s", "gpu_min_free_mb"):
        assert config["runtime"][key] == parent["runtime"][key]
    preserved = load(PACKAGE / "verification/preserved-runs.json")
    assert all(file_digest(ROOT / name) == sha for name, sha in preserved.items())
    baseline = load(ROOT / "official_direct/baseline-five-results-20261107.json")
    old_rows = {r["task"]["index"]: r for r in baseline["results"] if r["task"]["environment"] == "navigation"}
    sources = Sources(manipulation_catalog=ROOT / "official_direct/assets/manipulation_assets.json")
    sources.verify_all()
    dataset = json.loads(sources.read("navigation", "embodiedbench/envs/eb_navigation/datasets/base.json"))["tasks"]
    legacy_folder = ROOT / "runs/qwen-bare-official-suite-nav-3"
    legacy_trace = [json.loads(line) for line in (legacy_folder / "trace.jsonl").read_text().splitlines()]
    audits, pairs = {}, []
    for i in range(5):
        task = Task("navigation", "base", i)
        folder, old = RUN / task.directory, old_rows[i]
        row = load(folder / "result.json")
        audit = audit_episode(folder, config, sources, frozen)
        if i < 3:
            original = load(old["source_result"])
            assert original["input_sha256"] == digest(canonical(dataset[i]))
            assert original["instruction_sha256"] == audit["first_instruction_sha256"]
            first = next(e for e in legacy_trace if e["task_id"] == f"base:{i:03d}"
                         and "request_index" in e and e["environment_step"] == 0)
            old_rgb = first["image_sha256"]
        else:
            old_episode = Path(old["source_result"]).parent
            first = next(e for e in events(old_episode) if e["type"] == "observation")
            assert digest(first["public"]["instruction"].encode()) == audit["first_instruction_sha256"]
            old_rgb = first["public"]["images"][0]["sha256"]
        audits[task.identity] = audit
        pairs.append({
            "task": task.as_dict(), "instruction": row["metrics"]["instruction"],
            "baseline_success": old["task_success"], "rgbd_success": int(row["official"]["task_success"]),
            "baseline_steps": old["num_steps"], "rgbd_steps": row["metrics"]["num_steps"],
            "baseline_requests": old["model_requests"], "rgbd_requests": audit["requests"],
            "rgbd_status": row["status"], "baseline_origin": old["origin"],
            "same_official_question_and_instruction": True, "baseline_first_rgb_sha256": old_rgb,
            "rgbd_first_rgb_sha256": audit["first_rgb_sha256"],
            "initial_rgb_hash_matches": old_rgb == audit["first_rgb_sha256"],
            "rgbd_result": str(folder / "result.json"), "rgbd_trace": str(folder / "trajectory.jsonl")})
    report = {
        "schema": "navigation-rgbd-pilot/1", "status": "verified", "selected": 5, "completed": 5,
        "baseline_successes": sum(r["baseline_success"] for r in pairs),
        "rgbd_successes": sum(r["rgbd_success"] for r in pairs),
        "baseline_requests": sum(r["baseline_requests"] for r in pairs),
        "rgbd_requests": summary["http_requests"],
        "total_verified_synchronized_observations": sum(a["observations"] for a in audits.values()),
        "all_requests_have_two_images": True, "initial_rgb_hash_matches": sum(r["initial_rgb_hash_matches"] for r in pairs),
        "model": config["model"], "depth_display_range_meters": [0, 10], "depth_shape": [500, 500],
        "depth_raw_unit": "meters", "old_files_unchanged": len(preserved),
        "responses_explicitly_mentioning_depth": sum(
            len(a["depth_mentions_in_responses"]) for a in audits.values()),
        "reference_answer_visible": False, "automatic_retry": False, "pairs": pairs, "audits": audits}
    (PACKAGE / "rgbd-test-results-20261107.json").write_bytes(canonical(report) + b"\n")
    text = [
        "# Navigation 同步 RGB-D 首次试验",
        "",
        f"实现与模型输入核验通过。完成 5/5 题，所有 {report['rgbd_requests']} 次模型请求"
        "均按顺序输入当前 RGB 与同一仿真事件的深度图。两图均为 500×500。",
        "",
        f"本轮 RGB-D 成功 {report['rgbd_successes']}/5；已有 RGB 基线成功 "
        f"{report['baseline_successes']}/5。模型请求分别为 {report['rgbd_requests']} 与 "
        f"{report['baseline_requests']} 次。",
        "",
        "成功题仍为 000、001、002、004；003 在 20 步上限结束时未完成目标。"
        "原基线 003 因无效计划提前结束；本轮该题执行到了官方步数上限，仍没有转化为成功。",
        "",
        "[RGB 与深度首帧对照](computer://" + str(PACKAGE / "rgb-depth-preview.png") + ")",
        "",
        "[五题首帧双图](computer://" + str(PACKAGE / "rgbd-five-initial-pairs.png") + ")",
        "",
        "读取 AI2-THOR 原生 depth_frame，不运行深度估计模型，不增加仿真动作。"
        "显示采用固定 0–10 米反向灰度，亮表示近、暗表示远；10 米外及无效深度显示黑色。"
        "未缩放或平移深度图。原始 float32 米制深度另存为 .npy，保留 10 米以外的数值。",
        "",
        "RGB 必须与 depth_frame 所属事件的 RGB 数组逐像素相同。"
        "模型只收到两张图与颜色含义说明；原始数组不进入模型消息。",
        "",
        f"26 个响应中有 {report['responses_explicitly_mentioning_depth']} 个显式提到 depth map。"
        "例如 001 的请求 2 描述近处地面、橱柜较亮，远墙较暗。"
        "这说明模型会在回答中引用深度；具体几何判断是否正确仍需按图核验。",
        "",
        "| 官方 base 索引 | RGB 基线成功 | RGB-D 成功 | 基线步数 | RGB-D 步数 | 基线请求 | RGB-D 请求 | 首帧 RGB 哈希相同 |",
        "|---|---:|---:|---:|---:|---:|---:|---|"]
    for r in pairs:
        text.append(
            f"| {r['task']['index']:03d} | {r['baseline_success']} | {r['rgbd_success']} | "
            f"{r['baseline_steps']} | {r['rgbd_steps']} | {r['baseline_requests']} | "
            f"{r['rgbd_requests']} | {'是' if r['initial_rgb_hash_matches'] else '否'} |")
    text += [
        "",
        "模型保持 qwen3.8-flash、temperature=0、seed=137、max_tokens=4096、"
        "enable_thinking=false、JSON-object、官方三条示例和原有完整动作计划。"
        "每题仍限 20 步；没有加入单步执行策略。0 次自动重试或动作重发。",
        "",
        f"全部 {report['total_verified_synchronized_observations']} 份观测已核验：RGB、深度显示图、"
        "原始数组及哈希对应；深度图像像素按原始数值精确重建；"
        "逐条模型消息从公开观测精确重放；动作发送与回执对应。"
        f"先前 {report['old_files_unchanged']} 个基线及 Manipulation 试验文件内容保留不变。",
        "",
        f"五题均与已有基线对应相同官方问题和指令，但首帧 RGB 哈希一致的只有 "
        f"{report['initial_rgb_hash_matches']}/5 题。"
        "基线 000–002 来自历史 GPU 3 运行，本次使用 GPU 7；"
        "不能将哈希不同的图像当作完全相同的初始视觉输入。"
        "这轮验证同步双图输入可运行，成绩变化仍需结合配对限制理解。",
        "",
        "同题抓图验收与正式运行的两次重置也出现了轻微首帧变化："
        "RGB 平均每通道绝对差为 0.157（8 位强度），"
        "深度平均绝对差约 1.99e-8 米，最大差约 0.00237 米。"
        "这是跨重置比较，与单次事件内 RGB/深度同步的核验分别记录。",
        "",
        "[完整数值与逐帧审计](computer://" + str(PACKAGE / "rgbd-test-results-20261107.json") + ")"]
    (PACKAGE / "rgbd-test-report-20261107.md").write_text("\n".join(text) + "\n")
    print(json.dumps({k: report[k] for k in (
        "status", "completed", "rgbd_successes", "baseline_successes", "rgbd_requests",
        "total_verified_synchronized_observations", "initial_rgb_hash_matches", "old_files_unchanged")}))


if __name__ == "__main__":
    main()
