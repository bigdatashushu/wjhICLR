"""Offline audit and historical comparison for the frozen twenty-task run."""
import base64
from datetime import datetime, timezone
import io
import json
from pathlib import Path
import re
import sys

import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from target_detection_r05.audit import episode_audit
from target_detection_r05.depth import decode_raw
from target_detection_r05.protocol import Task, validate_packet
from target_detection_r05.sources import canonical, digest, file_digest
from target_detection_twenty.control import (
    CONFIG, ENV_ORDER, OUT, ROOT, RUN_ID, VERIFY, attempted, load, save, verify_identity)


def events(folder):
    return [json.loads(s) for s in (folder / "trajectory.jsonl").read_text(encoding="utf-8").splitlines()]


def first_observation(trace):
    return next((json.loads(s) for s in Path(trace).read_text(encoding="utf-8").splitlines()
                 if json.loads(s)["type"] == "observation"), None)


def operational_audit(folder, config):
    """Keep service failures and a possible lost action ACK explicit."""
    rows, result = events(folder), load(folder / "result.json")
    task = Task(**result["task"])
    assert result["status"] == "operational_error"
    assert rows[-1]["type"] == "episode_end"
    observations = [e for e in rows if e["type"] == "observation"]
    for event in observations:
        packet = {k: event[k] for k in ("task", "public", "official", "input_sha256")}
        validate_packet(packet, task)
        serial, public = event["serial"], event["public"]
        for suffix, image in (("0", public["images"][0]), ("raw", public["raw_rgb"]),
                              ("depth", public["depth"]["image"])):
            assert (folder / f"observation-{serial:03d}-{suffix}.png").read_bytes() == base64.b64decode(image["base64"])
        depth = np.load(folder / f"observation-{serial:03d}-depth-meters.npy", allow_pickle=False)
        assert np.array_equal(depth, decode_raw(public["depth"]), equal_nan=True)
    requests = [e for e in rows if e["type"] == "model_request"]
    sent, acks = ([e for e in rows if e["type"] == kind] for kind in ("action_sent", "action_ack"))
    assert len(requests) <= config["max_requests"][task.environment]
    assert 0 <= len(sent) - len(acks) <= 1
    assert [e["action"] for e in sent[:len(acks)]] == [e["action"] for e in acks]
    assert [e["expected_step"] for e in sent] == list(range(1, len(sent) + 1))
    assert all(e.get("automatic_retry") is not True for e in rows)
    for event in requests:
        payload = load(folder / event["request_file"])
        assert digest(canonical(payload)) == event["payload_sha256"]
        assert payload["model"] == config["model"]["name"]
    tools = [e for e in rows if e["type"] == "perception_tool_result"]
    by_serial = {e["serial"]: e for e in observations}
    for event in tools:
        assert event["simulator_steps"] == event["simulator_observations"] == 0
        public = by_serial[event["result"]["observation_serial"]]["public"]
        assert event["result"]["raw_rgb_sha256"] == public["raw_rgb"]["sha256"]
        for meta in event["images"]:
            assert file_digest(folder / meta["file"]) == meta["sha256"]
        if event["tool_request"]["name"] == "inspect_candidates":
            raw = Image.open(io.BytesIO(base64.b64decode(public["raw_rgb"]["base64"]))).convert("RGB")
            for crop, meta in zip(event["result"]["crops"], event["images"]):
                assert np.array_equal(np.asarray(Image.open(folder / meta["file"]).convert("RGB")),
                                      np.asarray(raw.crop(crop["bounds"])))
    perception = result["perception"]
    return {"task": task.as_dict(), "status": result["status"],
            "official_success": result["metrics"]["task_success"],
            "model_requests": len(requests), "tool_calls": perception["tool_calls"],
            "completed_tool_calls": len(tools), "detector_requests": perception["detector_requests"],
            "crops": perception["crop_count"], "actions": len(sent), "acks": len(acks),
            "targets": perception["targets"], "partial_operational_evidence_verified": True,
            "request_replay_verified": False, "pending_action_outcome_unknown": len(sent) != len(acks),
            "tool_steps_zero": True}


def compare_observation(current, previous):
    if current is None:
        return {"initial_observation_available": False}
    a, b = current["public"], previous["public"]
    x = np.asarray(Image.open(io.BytesIO(base64.b64decode(a["images"][0]["base64"]))).convert("RGB"))
    y = np.asarray(Image.open(io.BytesIO(base64.b64decode(b["images"][0]["base64"]))).convert("RGB"))
    return {
        "initial_observation_available": True,
        "initial_instruction_equal": a["instruction"] == b["instruction"],
        "initial_actions_equal": a["actions"] == b["actions"],
        "initial_official_manipulation_equal": a.get("manipulation") == b.get("manipulation"),
        "initial_rgb_sha256_equal": a["images"][0]["sha256"] == b["images"][0]["sha256"],
        "initial_rgb_pixels_equal": bool(np.array_equal(x, y)),
        "initial_rgb_pixel_mean_absolute_difference": float(np.abs(x.astype(float) - y.astype(float)).mean()),
        "initial_depth_png_equal": a["depth"]["image"]["sha256"] == b["depth"]["image"]["sha256"],
    }


def legacy_full():
    path = ROOT / "runs/task-card-full-r06.json"
    document = load(path)
    rows = {}
    for attempt in document["attempts"]:
        environment = attempt["environment"].removeprefix("eb_")
        episode = attempt["episode_id"]
        if not isinstance(episode, str):
            continue
        match = re.search(r"(\d+)$", episode)
        if not match or environment not in ENV_ORDER:
            continue
        task = Task(environment, "base", int(match.group(1)))
        requests = attempt["model_requests"]
        rows[task.identity] = {"success": attempt["task_success"], "status": attempt["status"],
                               "steps": attempt["environment_steps"],
                               "requests": len(requests) if isinstance(requests, list) else requests,
                               "source": str(path), "source_sha256": file_digest(path)}
    return rows


def font(size):
    return ImageFont.truetype("/System/Library/Fonts/Supplemental/Arial.ttf", size)


def screenshots(environment, pairs):
    canvas = Image.new("RGB", (1480, 105 + len(pairs) * 445 + 44), "#E2E8F0")
    draw = ImageDraw.Draw(canvas)
    draw.text((28, 22), environment.upper() + " | Question-directed perception, base 0-4",
              font=font(28), fill="#0F172A")
    draw.text((28, 66), "Actual initial observations | RGB -> candidate groups -> original-pixel crops",
              font=font(18), fill="#475569")
    receipt = []
    for i, pair in enumerate(pairs):
        folder = Path(pair["current_directory"])
        trajectory = events(folder)
        raw = folder / "observation-000-raw.png"
        tools = [e for e in trajectory if e["type"] == "perception_tool_result"
                 and e["result"]["observation_serial"] == 0]
        detector = next((e for e in reversed(tools) if e["tool_request"]["name"] == "detect_targets"), None)
        reviews = [e for e in trajectory if e["type"] == "perception_action_review" and e["observation_serial"] == 0]
        judgments = {r["group_id"]: r["verdict"] for r in reviews[-1]["perception_review"]} if reviews else {}
        selected = {r["group_id"] for r in reviews[-1]["selected_targets"]} if reviews else set()
        crops = {}
        for event in tools:
            if event["tool_request"]["name"] == "inspect_candidates":
                for crop, meta in zip(event["result"]["crops"], event["images"]):
                    crops[crop["group_id"]] = meta
        chosen = sorted(crops, key=lambda gid: (gid not in selected, gid))[:2]
        y = 105 + i * 445
        draw.rectangle((16, y, 1464, y + 427), fill="white")
        success = pair["current_success"]
        counts = (f'{detector["result"]["proposal_count"]} proposals / '
                  f'{detector["result"]["group_count"]} groups') if detector else "no completed detection"
        draw.text((30, y + 12), f'base {pair["task"]["index"]} | {counts} | success={success} | {pair["current_status"]}',
                  font=font(21), fill="#0F172A")
        targets = ", ".join(t["category"] for t in (pair["audit"]["targets"] or []))
        draw.text((30, y + 44), "Targets: " + targets, font=font(17), fill="#475569")
        if raw.exists():
            image = ImageOps.contain(Image.open(raw).convert("RGB"), (354, 324))
            draw.text((30, y + 66), "Raw RGB", font=font(16), fill="#475569")
            canvas.paste(image, (30, y + 88))
        else:
            draw.text((30, y + 110), "No model-visible observation was returned.", font=font(18), fill="#475569")
        if detector:
            image = ImageOps.contain(Image.open(folder / detector["images"][0]["file"]).convert("RGB"), (574, 324))
            draw.text((411, y + 66), "Numbered proposal groups", font=font(16), fill="#475569")
            canvas.paste(image, (411, y + 88))
        for j, gid in enumerate(chosen):
            x = 1009 + j * 224
            draw.text((x, y + 66), gid + " raw crop", font=font(16), fill="#475569")
            image = ImageOps.contain(Image.open(folder / crops[gid]["file"]).convert("RGB"), (204, 290))
            canvas.paste(image, (x + (204 - image.width) // 2, y + 88 + (290 - image.height) // 2))
            if gid in judgments:
                color = {"match": "#047857", "not_match": "#B91C1C", "uncertain": "#B45309"}[judgments[gid]]
                draw.text((x, y + 384), "model verdict: " + judgments[gid], font=font(14), fill=color)
        receipt.append({"task": pair["task"], "raw_image": str(raw) if raw.exists() else None,
                        "proposal_count": detector["result"]["proposal_count"] if detector else None,
                        "group_count": detector["result"]["group_count"] if detector else None,
                        "crop_groups": chosen})
    draw.text((28, canvas.height - 32), "Proposal groups and model judgments are not independently verified object identities.",
              font=font(16), fill="#475569")
    path = OUT / f"results-{environment}.png"
    canvas.save(path)
    return {"file": str(path), "sha256": file_digest(path), "rows": receipt}


def main():
    config, tasks, sources = verify_identity()
    after = load(VERIFY / "server-after.json")
    assert not after["owned_trial_workers"] and after["local_old_files_unchanged"] and after["old_evidence_unchanged"]
    folders = attempted()
    assert len(folders) == 20 and all((folder / "result.json").exists() for folder in folders.values())
    historical = load(VERIFY / "historical-index.json")
    old_tools = legacy_full()
    pairs, total_usage = [], {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    for task in tasks:
        folder = folders[task.identity]
        outcome = load(folder / "result.json")
        audit = (operational_audit(folder, config) if outcome["status"] == "operational_error"
                 else episode_audit(folder, config, sources))
        prior = historical[task.identity]
        assert file_digest(prior["result"]) == prior["result_sha256"]
        assert file_digest(prior["trace"]) == prior["trace_sha256"]
        first = next((e for e in events(folder) if e["type"] == "observation"), None)
        observation = compare_observation(first, first_observation(prior["trace"]))
        if observation["initial_observation_available"]:
            assert observation["initial_instruction_equal"] and observation["initial_actions_equal"]
        for path in folder.glob("response-*.json"):
            usage = load(path).get("usage", {})
            for key in total_usage:
                total_usage[key] += usage.get(key, 0)
        success = int(outcome["status"] == "official_done" and outcome["metrics"]["task_success"] == 1)
        pairs.append({"task": task.as_dict(), "identity": task.identity,
                      "current_success": success, "current_status": outcome["status"],
                      "current_directory": str(folder), "current_result": str(folder / "result.json"),
                      "rgbd_success": prior["rgbd_success"], "rgb_success": prior["rgb_success"],
                      "rgbd_requests": prior["rgbd_requests"], "rgbd_steps": prior["rgbd_steps"],
                      "rgbd_result": prior["result"], "legacy_full": old_tools.get(task.identity),
                      "initial_identity": observation, "audit": audit})
    by_environment = {}
    for environment in ENV_ORDER:
        rows = [p for p in pairs if p["task"]["environment"] == environment]
        by_environment[environment] = {
            "selected": len(rows), "current_successes": sum(p["current_success"] for p in rows),
            "rgbd_successes": sum(p["rgbd_success"] for p in rows),
            "model_requests": sum(p["audit"]["model_requests"] for p in rows),
            "detector_requests": sum(p["audit"]["detector_requests"] for p in rows),
            "tool_calls": sum(p["audit"]["tool_calls"] for p in rows),
            "actions": sum(p["audit"]["actions"] for p in rows),
            "acks": sum(p["audit"]["acks"] for p in rows),
        }
    screenshots_receipts = [screenshots(e, [p for p in pairs if p["task"]["environment"] == e]) for e in ENV_ORDER]
    result = {
        "status": "verified", "checked_at": datetime.now(timezone.utc).isoformat(),
        "selected": 20, "completed": 20, "implementation_package": "target_detection_r05",
        "current_successes": sum(p["current_success"] for p in pairs),
        "rgbd_successes": sum(p["rgbd_success"] for p in pairs),
        "rgb_successes": sum(p["rgb_success"] for p in pairs),
        "changed_to_success": [p["identity"] for p in pairs if p["current_success"] == 1 and p["rgbd_success"] == 0],
        "changed_to_failure": [p["identity"] for p in pairs if p["current_success"] == 0 and p["rgbd_success"] == 1],
        "initial_rgb_pixels_equal": sum(p["initial_identity"].get("initial_rgb_pixels_equal", False) for p in pairs),
        "legacy_full_coverage": sum(p["legacy_full"] is not None for p in pairs),
        "operational_failures": sum(p["current_status"] == "operational_error" for p in pairs),
        "unknown_action_outcomes": sum(p["audit"].get("pending_action_outcome_unknown", False) for p in pairs),
        "model_requests": sum(p["audit"]["model_requests"] for p in pairs),
        "detector_requests": sum(p["audit"]["detector_requests"] for p in pairs),
        "tool_calls": sum(p["audit"]["tool_calls"] for p in pairs),
        "actions": sum(p["audit"]["actions"] for p in pairs),
        "acks": sum(p["audit"]["acks"] for p in pairs),
        "by_environment": by_environment, "total_usage": total_usage, "pairs": pairs,
        "screenshots": screenshots_receipts, "automatic_retry": False, "live_requests_in_audit": 0,
        "comparison_scope": "Same historical twenty tasks, RGB-D versus question-directed R05 perception; historical Full is supplementary.",
    }
    save(OUT / "results.json", result)
    lines = [
        "# 原 20 题：问题定向检测与裁剪复核实验",
        "",
        f'完成并审计 20/20 题。历史 RGB-D 成功 {result["rgbd_successes"]}/20，'
        f'本轮 R05 成功 {result["current_successes"]}/20；'
        f'由失败变成功 {len(result["changed_to_success"])} 题，由成功变失败 {len(result["changed_to_failure"])} 题。',
        "",
        "| 环境 | 历史 RGB-D | 本轮 R05 | Qwen 请求 | 检测推理 | 工具调用 | 动作/ACK |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for environment, row in by_environment.items():
        lines.append(f'| {environment} | {row["rgbd_successes"]}/5 | {row["current_successes"]}/5 | '
                     f'{row["model_requests"]} | {row["detector_requests"]} | {row["tool_calls"]} | '
                     f'{row["actions"]}/{row["acks"]} |')
    lines.extend([
        "",
        "四环境均为官方 base 索引 0–4。使用冻结的 target_detection_r05，"
        "Qwen3.8-Flash、temperature=0、seed=137、max_tokens=4096、enable_thinking=false，"
        "JSON-object。官方提示、示例和完整动作执行保持；RGB 与原生深度同步输入。",
        "",
        "新增流程是同一模型提取目标、逐类别 GroundingDINO、IoU≥0.8 完整链接分组、"
        "图外 G 编号、原始 RGB 裁剪和明确的视觉复核。模型请求上限仍为 Navigation 20、"
        "ALFRED/Habitat 40、Manipulation 15，检测与复核轮次计入该预算。"
        "四环境与每题均串行，检测复用物理 GPU1，GPU 渲染用 GPU7，其余软件渲染。",
        "",
        f'合计 {result["model_requests"]} 次 Qwen 请求、{result["detector_requests"]} 次检测、'
        f'{result["tool_calls"]} 次工具调用、{result["actions"]} 个动作和 {result["acks"]} 个 ACK。'
        f'操作性失败 {result["operational_failures"]}，动作结果未知 {result["unknown_action_outcomes"]}。',
        "",
        f'与历史 RGB-D 的初始 RGB 像素一致 {result["initial_rgb_pixels_equal"]}/20 题；'
        "逐题记录同时比较公开指令、动作列表、官方 Manipulation 输入和深度图身份。"
        "这是同一批题的历史运行对照，感知材料、阶段提示和多轮模型交互共同变化。",
        "",
        f'旧 Full 工具版本可匹配 {result["legacy_full_coverage"]}/20 题，逐题附在结果 JSON 中。'
        "历史 RGB-D 自身没有额外工具调用；旧 Full 使用不同的子目标与执行协议，作为补充参考。",
        "",
        "## 逐题结果",
        "",
        "| 题目 | RGB-D→R05 | 结束状态 | 模型请求 | 工具调用 | 裁剪 | 动作/ACK | 初始 RGB 相同 |",
        "|---|---|---|---:|---:|---:|---:|---|",
    ])
    for pair in pairs:
        row = pair["audit"]
        same = pair["initial_identity"].get("initial_rgb_pixels_equal")
        lines.append(f'| {pair["identity"]} | {pair["rgbd_success"]}→{pair["current_success"]} | '
                     f'{pair["current_status"]} | {row["model_requests"]} | {row["tool_calls"]} | '
                     f'{row["crops"]} | {row["actions"]}/{row["acks"]} | '
                     f'{"是" if same else "否" if same is False else "无观测"} |')
    lines.extend(["", "## 实际截图与记录", ""])
    for environment in ENV_ORDER:
        lines.append(f'- [{environment}：五题原图、编号检测图和裁剪](results-{environment}.png)')
    lines.extend([
        "- [逐题结果、输入身份与离线审计](results.json)",
        "- [运行清单与续跑记录](verification/run-records.json)",
        "- [源码与配置身份](verification/frozen.json)",
        "- [最终服务器与原文件保留检查](verification/server-after.json)",
        "- [运行说明](README.md)",
        "",
        "正常结束题逐请求重放模型消息、工具结果和动作计划，验证裁剪像素、编号失效及 ACK。"
        "操作性失败单列，保留已有观测、请求身份与未确认动作，不将其写成模型的负向判断。"
        "原实验文件和冻结源码保持字节一致，最终没有自有 worker。",
        "",
    ])
    report = OUT / "report.md"
    report.write_text("\n".join(lines), encoding="utf-8")
    print(canonical({"status": "verified", "selected": 20,
                     "current_successes": result["current_successes"], "rgbd_successes": result["rgbd_successes"],
                     "model_requests": result["model_requests"], "actions": result["actions"], "acks": result["acks"],
                     "report": str(report)}).decode())


if __name__ == "__main__":
    main()
