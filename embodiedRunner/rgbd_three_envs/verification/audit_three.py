"""Replay every saved public RGB-D request and verify simulator/action receipts."""
import base64
import io
import json
from pathlib import Path
import re
import sys

import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from rgbd_three_envs.client import ModelClient
from rgbd_three_envs.depth import decode_raw
from rgbd_three_envs.planner import OfficialPlanner
from rgbd_three_envs.protocol import Task, validate_packet
from rgbd_three_envs.sources import LIMITS, Sources, canonical, digest, file_digest, implementation_hashes

PACKAGE = ROOT / "rgbd_three_envs"
RUN = PACKAGE / "runs/three-rgbd-smoke-20261107-r02"
RECOVERY = PACKAGE / "runs/three-rgbd-alfred-recovery-20261107-r02"


def load(path):
    return json.loads(Path(path).read_text())


def events(folder):
    return [json.loads(line) for line in (folder / "trajectory.jsonl").read_text().splitlines()]


def audit_observation(folder, event, task):
    packet = {k: event[k] for k in ("task", "public", "official", "input_sha256")}
    validate_packet(packet, task)
    public, serial = packet["public"], event["serial"]
    depth = public["depth"]
    assert file_digest(folder / f"observation-{serial:03d}-0.png") == public["images"][0]["sha256"]
    assert file_digest(folder / f"observation-{serial:03d}-depth.png") == depth["image"]["sha256"]
    raw = np.load(folder / f"observation-{serial:03d}-depth-meters.npy", allow_pickle=False)
    assert raw.dtype == np.dtype("<f4") and raw.shape == (500, 500)
    assert digest(raw.tobytes()) == depth["raw"]["sha256"]
    assert np.array_equal(raw, decode_raw(depth), equal_nan=True)
    return packet


def image(value):
    return Image.open(io.BytesIO(base64.b64decode(value["base64"]))).convert("RGB")


def render_pairs(kind, decisions):
    selected = decisions if len(decisions) <= 6 else decisions[:4] + decisions[-2:]
    canvas = Image.new("RGB", (1000, 530 * len(selected)), "white")
    draw = ImageDraw.Draw(canvas)
    for row, (request, public) in enumerate(selected):
        canvas.paste(image(public["images"][0]), (0, row * 530))
        canvas.paste(image(public["depth"]["image"]), (500, row * 530))
        label = f"{kind} request {request} / step {public['step']}"
        draw.text((8, row * 530 + 505), label + " RGB", fill="black")
        draw.text((508, row * 530 + 505), label + " depth 0-10m", fill="black")
    path = PACKAGE / f"rgbd-{kind}-preview.png"
    canvas.save(path)
    return str(path)


def capture_audit(task, frozen):
    revision = "r02" if task.environment == "habitat" else "r01"
    if revision == "r01":
        frozen = load(PACKAGE / "verification/frozen-manifest.json")
    folder = PACKAGE / f"runs/capture-{task.environment}-20261107-{revision}"
    report = load(folder / "capture.json")
    trace = events(folder)
    assert report["status"] == "synchronized_capture_verified"
    assert report["task"] == task.as_dict()
    assert report["model_requests"] == report["environment_steps"] == 0
    observed = [e for e in trace if e["type"] == "observation"]
    assert len(observed) == 1
    packet = audit_observation(folder, observed[0], task)
    assert trace[0]["report"]["implementation_hashes"] == frozen["implementation_hashes"]
    assert trace[0]["report"]["source_hashes"] == frozen["source_hashes"]
    return report, packet


def audit_episode(folder, config, sources, frozen):
    row = load(folder / "result.json")
    task = Task(**row["task"])
    packet, planner, plan = None, None, None
    requests, responses, observations, sent, ack = 0, 0, 0, [], []
    decisions, mentions, receipts = [], [], []
    usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    client = ModelClient(config["model"])
    trace = events(folder)
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
            packet = audit_observation(folder, event, task)
            public, depth = packet["public"], packet["public"]["depth"]
            receipts.append({"serial": event["serial"], "step": public["step"],
                             "rgb_sha256": public["images"][0]["sha256"],
                             "depth_png_sha256": depth["image"]["sha256"],
                             "raw_depth_sha256": depth["raw"]["sha256"],
                             "stats": depth["stats"], "provenance": depth["provenance"]})
            if planner is None:
                planner = OfficialPlanner(sources, task, config["model"]["name"],
                                          config["n_shots"][task.environment], public["actions"],
                                          config["model"]["response_format"])
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
            decisions.append((requests, packet["public"]))
        elif kind == "model_response":
            responses += 1
            response = load(folder / f"response-{responses:03d}.json")
            assert response == event["response"] and response["model"] == config["model"]["name"]
            plan = planner.parse(response["text"])
            assert vars(plan) == event["plan"]
            for key in usage:
                usage[key] += response["usage"].get(key, 0)
            if re.search(r"\bdepth\b|深度", response["text"], re.I):
                mentions.append({"request_index": responses, "response_text": response["text"]})
        elif kind == "plan_execution":
            assert event["executed_prefix"] == plan.actions[:LIMITS[task.environment] - packet["public"]["step"]]
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
        elif kind == "episode_end":
            assert event["result"] == row
    assert row["status"] in ("official_done", "empty_plan", "planner_error", "request_budget", "invalid_action_limit")
    assert requests == responses == row["metrics"]["planner_steps"] <= config["max_requests"][task.environment]
    assert len(sent) == len(ack) == row["metrics"]["num_steps"] <= LIMITS[task.environment]
    if row["status"] == "empty_plan" and task.environment == "manipulation":
        packet["official"].update(task_success=0, episode_elapsed_seconds=0)
    assert row["official"] == packet["official"]
    assert trace[-1]["type"] == "episode_end"
    assert requests > 0 and len(sent) > 0
    return {"task": task.as_dict(), "status": row["status"],
            "success": int(row["official"]["task_success"]), "steps": len(sent), "requests": requests,
            "observations": observations, "usage": usage,
            "all_requests_have_rgb_then_depth": True, "public_message_replay_exact": True,
            "all_depth_displays_match_raw_values": True, "all_actions_acknowledged_once": True,
            "full_native_plan_execution_preserved": True, "automatic_retry": False,
            "depth_mentions": mentions, "observation_receipts": receipts,
            "preview": render_pairs(task.environment, decisions),
            "instruction": row["metrics"]["instruction"]}


def main():
    config, identity, summary = [load(RUN / name) for name in ("config.json", "identity.json", "summary.json")]
    frozen = load(PACKAGE / "verification/frozen-manifest-r02.json")
    assert summary["selected"] == 3
    recovered = summary["status"] == "stopped_on_operational_failure"
    assert summary["status"] in ("finished", "stopped_on_operational_failure")
    total_requests = summary["http_requests"]
    startup_failure = None
    if recovered:
        failed = load(RUN / "alfred-base-000/result.json")
        assert failed["status"] == "operational_error" and failed["metrics"]["planner_steps"] == 0
        assert not any(e["type"] in ("observation", "model_request")
                       for e in events(RUN / "alfred-base-000"))
        recovery_config = load(RECOVERY / "config.json")
        recovery_identity = load(RECOVERY / "identity.json")
        recovery_summary = load(RECOVERY / "summary.json")
        assert recovery_config == config
        assert recovery_summary["status"] == "finished" and recovery_summary["selected"] == 1
        assert recovery_identity["implementation_hashes"] == frozen["implementation_hashes"]
        assert recovery_identity["source_hashes"] == frozen["source_hashes"]
        assert recovery_identity["selected_tasks"] == [{"environment": "alfred", "subset": "base", "index": 0}]
        assert recovery_identity["automatic_retry"] is False
        total_requests += recovery_summary["http_requests"]
        startup_failure = {"phase": "Unity graphics initialization", "model_requests": 0,
                           "original_error": failed["error"],
                           "result_sha256": file_digest(RUN / "alfred-base-000/result.json"),
                           "trajectory_sha256": file_digest(RUN / "alfred-base-000/trajectory.jsonl"),
                           "recovery": "separate ALFRED-only run; unchanged frozen code/model/settings"}
    assert identity["mode"] == "three_environments_rgbd" and identity["automatic_retry"] is False
    assert identity["implementation_hashes"] == implementation_hashes() == frozen["implementation_hashes"]
    assert identity["source_hashes"] == frozen["source_hashes"]
    assert config == load(PACKAGE / "experiment-20261107-r02.json")
    assert file_digest(PACKAGE / "experiment-20261107-r02.json") == frozen["config_sha256"]
    preserved = load(PACKAGE / "verification/preserved-files-r02.json")
    assert all(file_digest(ROOT / name) == sha for name, sha in preserved.items())
    credential_pattern = re.compile(rb"(?<![A-Za-z0-9])sk-[A-Za-z0-9_.+/-]{16,}")
    checked_text_files = 0
    for path in PACKAGE.rglob("*"):
        if path.is_file() and path.suffix in (".json", ".jsonl", ".md"):
            assert credential_pattern.search(path.read_bytes()) is None, "credential_pattern_in_artifact"
            checked_text_files += 1
    sources = Sources(manipulation_catalog=ROOT / "official_direct/assets/manipulation_assets.json")
    sources.verify_all()
    assert sources.hashes == frozen["source_hashes"]
    server_audit = load(PACKAGE / "verification/server-final-audit.json")
    assert server_audit["status"] == "verified" and not server_audit["active_owned_workers"]
    assert server_audit["deployed_r01_unchanged"] and server_audit["deployed_r02_unchanged"]
    tasks = [Task(**t) for t in config["episodes"]]
    captures, audits = [], []
    initial = Image.new("RGB", (1000, 530 * len(tasks)), "white")
    draw = ImageDraw.Draw(initial)
    for i, task in enumerate(tasks):
        report, packet = capture_audit(task, frozen)
        captures.append(report)
        origin = RECOVERY if recovered and task.environment == "alfred" else RUN
        audit = audit_episode(origin / task.directory, config, sources, frozen)
        audit["run_directory"] = str(origin)
        audits.append(audit)
        first = next(e for e in events(origin / task.directory) if e["type"] == "observation")["public"]
        initial.paste(image(first["images"][0]), (0, i * 530))
        initial.paste(image(first["depth"]["image"]), (500, i * 530))
        draw.text((8, i * 530 + 505), task.identity + " RGB", fill="black")
        draw.text((508, i * 530 + 505), task.identity + " synchronized depth", fill="black")
    initial.save(PACKAGE / "rgbd-three-initial-pairs.png")
    report = {"schema": "three-environments-rgbd-smoke/1", "status": "verified", "selected": 3,
              "model": config["model"], "model_requests": total_requests,
              "synchronized_observations": sum(a["observations"] for a in audits),
              "successes": sum(a["success"] for a in audits),
              "all_shapes_500x500": True, "all_raw_units_meters": True,
              "all_model_requests_have_two_images": True, "old_files_unchanged": len(preserved),
              "deployed_sources_and_assets_verified": True, "active_owned_workers": [],
              "credential_pattern_matches": 0, "credential_checked_text_files": checked_text_files,
              "automatic_retry": False, "captures": captures, "audits": audits,
              "initial_alfred_startup_failure": startup_failure,
              "scope": "one selected task per environment; functional verification"}
    (PACKAGE / "rgbd-three-results-20261107.json").write_bytes(canonical(report) + b"\n")
    assert sum(a["requests"] for a in audits) == total_requests
    text = ["# 三个环境同步 RGB-D 验收", "",
            f"ALFRED、Habitat、Manipulation 均已接入并实际运行。完成三题功能验收，"
            f"全部 {report['model_requests']} 次模型请求包含当前 RGB 和同步深度两张图；"
            f"{report['synchronized_observations']} 份观测完成数值、显示图、同步绑定及消息重放核验。",
            "",
            "| 环境 | 读取方式 | 换算为米 |",
            "|---|---|---|",
            "| ALFRED | 同一 last_event 的 RGB / depth_frame；开启辅助动作的深度渲染 | 原生毫米除以 1000 |",
            "| Habitat | 增加 head_depth_sensor，复制 RGB 相机位置、方向、视场角和投影类型 | normalize_depth=False，直接读取米 |",
            "| Manipulation | 同一观测的 front_rgb / front_depth | near + depth × (far − near) |",
            "",
            "三种深度图均保留 500×500 像素网格，固定 0–10 米反向灰度：亮表示近，暗表示远。"
            "无效像素及显示范围外的远处像素为黑色。显示图是可视化；原始 float32 米制数组保存在 .npy，"
            "不进入模型消息。Manipulation 保留官方 RGB 坐标轴和物体框，深度对应相同相机网格。",
            "",
            "模型附带的简短说明：第一张是 RGB，第二张是同像素坐标的同步深度；白色接近 0 米，"
            "黑色为 10 米以上或无效值；用 RGB 识别物体，用深度判断可见表面。"
            "黑色本身不能证明可通行，深度也不能直接当作世界坐标动作。",
            "",
            "[三环境首帧双图](computer://" + str(PACKAGE / "rgbd-three-initial-pairs.png") + ")",
            "",
            "| 环境 / 官方题目 | 成功 | 动作步数 | 模型请求 | 结束原因 |",
            "|---|---:|---:|---:|---|"]
    for audit in audits:
        task = Task(**audit["task"])
        text.append(f"| {task.identity} | {audit['success']} | {audit['steps']} | "
                    f"{audit['requests']} | {audit['status']} |")
    text += ["",
             "模型为 qwen3.8-flash，temperature=0、seed=137、max_tokens=4096、enable_thinking=false、"
             "JSON-object。保留各环境官方示例、动作反馈、完整计划执行与步数上限；"
             "Manipulation 未加入单步重规划版本。无自动服务重试、动作重发或额外模型工具调用。",
             "",
             f"三个环境均先完成零模型首帧验收，再各运行一题；成功 {report['successes']}/3。"
             "这是接口与运行链路验收，不能据此判断整体成功率或深度带来的增益。",
             "",
             "动作失败时，Manipulation 官方环境可能返回上一次观测；"
             "此时 RGB 与深度仍来自同一份观测，不能将其描述为新物理帧。",
             "",
             f"新代码位于独立 rgbd_three_envs 包与新的服务器目录。"
             f"旧基线、Navigation RGB-D 和 Manipulation 试验的 {len(preserved)} 个文件哈希保持不变。",
             "服务器上的 r01/r02 源码、官方快照及选定任务资产核验通过，已无本轮工作进程运行。",
             "",
             "[完整数值与逐帧审计](computer://" + str(PACKAGE / "rgbd-three-results-20261107.json") + ")"]
    if recovered:
        text += ["",
                 "首次三环境并行启动时，ALFRED 停在 Unity 图形初始化，600 秒超时退出，"
                 "尚无首帧、动作或模型请求。该失败记录完整保留。清理本轮残留进程后，"
                 "在其他两环境已结束的条件下另建 ALFRED 单独运行目录完成验证；"
                 "冻结代码、模型参数、任务和提示均保持一致。这里采用独立启动恢复，"
                 "没有自动重发 HTTP 请求或原仿真动作；是否由并行图形初始化触发仍未做因果验证。"]
    (PACKAGE / "rgbd-three-report-20261107.md").write_text("\n".join(text) + "\n")
    print(json.dumps({k: report[k] for k in ("status", "selected", "successes", "model_requests",
                                           "synchronized_observations", "old_files_unchanged")}))


if __name__ == "__main__":
    main()
