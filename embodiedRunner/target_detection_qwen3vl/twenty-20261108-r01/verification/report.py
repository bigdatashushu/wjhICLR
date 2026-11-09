"""Offline evidence checks and a compact paired report; never make model requests."""
import base64
from collections import Counter, deque
from datetime import datetime, timezone
import hashlib
import io
import json
from pathlib import Path
import sys
import tempfile

import numpy as np
from PIL import Image

ROOT = Path("/Users/bytedance/wangjh/wjhICLR/embodiedRunner")
OUT = ROOT / "target_detection_qwen3vl/twenty-20261108-r01"
sys.path.insert(0, str(OUT / "source_frozen/embodiedRunner"))
from target_detection_qwen3vl.detector import detection_prompt, pixel_box
from target_detection_qwen3vl.sources import canonical, digest, file_digest, Sources, OperationalError
from target_detection_qwen3vl.perception import crop_bounds, Perception, PlannerError
from target_detection_r05.planner import TargetPlanner
from target_detection_r05.depth import decode_raw
from target_detection_r05.protocol import Task, validate_packet


def load(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def events(path):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines()]


def run_roots():
    return [(OUT, OUT / "run")] + [
        (path.parent, path) for path in sorted((OUT / "continuations").glob("*/run")) if path.is_dir()]


def task_folders():
    folders = {}
    for _, run in run_roots():
        for result in run.glob("*/result.json"):
            assert result.parent.name not in folders, "task_was_repeated"
            folders[result.parent.name] = result.parent
    return folders


def status():
    roots = run_roots()
    completed = [load(folder / "result.json") for folder in task_folders().values()]
    active = []
    for trace in [p for _, run in roots for p in run.glob("*/trajectory.jsonl")]:
        if trace.with_name("result.json").exists():
            continue
        rows = events(trace)
        if rows:
            public = next((e["public"] for e in reversed(rows) if e["type"] == "observation"), None)
            active.append({
                "task": trace.parent.name, "last_event": rows[-1]["type"],
                "step": public["step"] if public else None,
                "planning_requests": sum(e["type"] == "model_request" for e in rows),
                "detector_requests_completed": sum(e["type"] == "detector_result" for e in rows),
            })
    return {
        "completed": len(completed), "selected": 20,
        "successes": sum(r["status"] != "operational_error" and r["metrics"]["task_success"] == 1
                         for r in completed),
        "by_environment": {
            env: {"completed": sum(r["task"]["environment"] == env for r in completed),
                  "successes": sum(r["task"]["environment"] == env and r["status"] != "operational_error"
                                   and r["metrics"]["task_success"] == 1 for r in completed)}
            for env in ("alfred", "habitat", "manipulation", "navigation")},
        "active": active,
        "exit": load(roots[-1][0] / "exit.json") if (roots[-1][0] / "exit.json").exists() else None,
        "run_count": len(roots),
    }


def first_observation(folder):
    for row in events(Path(folder) / "trajectory.jsonl"):
        if row["type"] == "observation":
            return row
    return None


def replay_episode(folder, config, rows, result):
    """Rebuild the actual prompt/tool/action sequence using recorded outputs only."""
    task = Task(**result["task"])
    observations = {e["serial"]: e["public"] for e in rows if e["type"] == "observation"}
    if not observations:
        assert result["status"] == "operational_error"
        return {"request_replay_verified": 0, "action_plans_replayed": 0,
                "planner_failures_replayed": 0, "target_and_review_replay_verified": True}
    sources = Sources(ROOT, ROOT / "official_direct/assets/manipulation_assets.json")
    primary_failure = next((e["code"] for e in rows if e["type"] == "failure"), None)
    recorded_detectors = deque(e for e in rows if e["type"] == "detector_result")
    class ReplayDetector:
        def detect(self, image, category):
            if not recorded_detectors:
                assert result["status"] == "operational_error"
                raise OperationalError(primary_failure or result["error"]["code"])
            event = recorded_detectors.popleft()
            assert event["query"] == category
            assert event["report"]["image_sha256"] == image["sha256"]
            # Canonical journals sort keys; the actual detector creates proposals
            # in this order, which is visible when tool state becomes JSON text.
            return dict(event["report"], detections=[
                {key: proposal[key] for key in ("id", "label", "bbox_1000", "bbox")}
                for proposal in event["report"]["detections"]])
    class ReplayJournal:
        def __init__(self):
            self.events = []
        def append(self, value):
            self.events.append(json.loads(canonical(value)))
    journal = ReplayJournal()
    finalized = {e["request_index"] for e in rows if e["type"] == "model_response"}
    replayed_requests, replayed_plans, replayed_failures = 0, 0, 0
    event_types = {"perception_bind", "targets_extracted", "detector_request_sent",
                   "detector_result", "perception_tool_result", "perception_action_review",
                   "explicit_review_wrapper_normalized"}
    with tempfile.TemporaryDirectory(prefix="qwen3vl-offline-replay-", dir="/private/tmp") as tmp:
        perception = Perception(config, tmp, journal, ReplayDetector())
        planner = TargetPlanner(sources, task, config["model"]["name"],
                                config["n_shots"][task.environment], observations[0]["actions"],
                                config["model"]["response_format"])
        latest_public = observations[0]
        for event in rows:
            kind = event["type"]
            if kind == "observation":
                latest_public = event["public"]
            elif kind == "model_request":
                public = observations[event["observation_serial"]]
                perception.bind(public, event["observation_serial"])
                payload = load(folder / event["request_file"])
                actual_messages = planner.messages(public, perception)
                if payload["messages"] != actual_messages:
                    def difference(a, b, path="messages"):
                        if isinstance(a, dict) and isinstance(b, dict):
                            for key in set(a) | set(b):
                                if a.get(key) != b.get(key):
                                    return difference(a.get(key), b.get(key), path + "." + str(key))
                        elif isinstance(a, list) and isinstance(b, list):
                            if len(a) != len(b):
                                return {"path": path, "recorded_length": len(a), "replayed_length": len(b)}
                            for index, (x, y) in enumerate(zip(a, b)):
                                if x != y:
                                    return difference(x, y, f"{path}[{index}]")
                        return {"path": path, "recorded_sha256": digest(canonical(a)),
                                "replayed_sha256": digest(canonical(b)),
                                "recorded_excerpt": str(a)[:180] if "url" not in path else None,
                                "replayed_excerpt": str(b)[:180] if "url" not in path else None}
                    raise AssertionError(json.dumps({"task": task.identity,
                                                     "request": event["request_index"],
                                                     "difference": difference(payload["messages"], actual_messages)}))
                replayed_requests += 1
            elif kind == "model_response":
                response = load(folder / f'response-{event["request_index"]:03d}.json')
                assert response == event["response"]
                plan = planner.decision(response["text"], perception)
                assert (vars(plan) if plan else None) == event["plan"]
                replayed_plans += int(plan is not None)
            elif kind == "model_response_received" and event["request_index"] not in finalized:
                response = load(folder / event["response_file"])
                try:
                    planner.decision(response["text"], perception)
                except PlannerError as exc:
                    assert result["status"] == "planner_error"
                    assert str(exc) == result["error"]["code"]
                    replayed_failures += 1
                except OperationalError as exc:
                    assert result["status"] == "operational_error"
                    assert exc.code == (primary_failure or result["error"]["code"])
                else:
                    raise AssertionError("recorded_failure_not_reproduced")
            elif kind == "action_ack":
                planner.feedback(event["action"], latest_public)
        assert not recorded_detectors
        recorded_events = [e for e in rows if e["type"] in event_types]
        if journal.events != recorded_events:
            mismatch = next(((i, a, b) for i, (a, b) in enumerate(zip(recorded_events, journal.events))
                             if a != b), None)
            raise AssertionError(json.dumps({
                "task": task.identity, "recorded_events": len(recorded_events),
                "replayed_events": len(journal.events),
                "first_difference": {"index": mismatch[0], "recorded": mismatch[1],
                                     "replayed": mismatch[2]} if mismatch else None}))
        for event in journal.events:
            if event["type"] == "perception_tool_result":
                for image in event["images"]:
                    assert (folder / image["file"]).read_bytes() == (Path(tmp) / image["file"]).read_bytes()
        assert result["perception"]["targets"] == perception.targets
        assert result["perception"]["tool_calls"] == perception.tool_calls
        assert result["perception"]["detector_requests"] == perception.detector_calls
        assert result["perception"]["crop_count"] == perception.crop_count
    return {"request_replay_verified": replayed_requests, "action_plans_replayed": replayed_plans,
            "planner_failures_replayed": replayed_failures, "target_and_review_replay_verified": True}


def audit_episode(folder, config):
    result, rows = load(folder / "result.json"), events(folder / "trajectory.jsonl")
    task = Task(**result["task"])
    assert rows[-1]["type"] == "episode_end" and rows[-1]["result"] == result
    assert rows[0]["reference_answer_visible"] is False and rows[0]["automatic_retry"] is False
    by_serial, requested_queries = {}, []
    detectors, detector_attempts, crops, requests, actions, acks = [], [], [], [], [], []
    phases = Counter()
    planner_usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    detector_usage = dict(planner_usage)
    for row in rows:
        kind = row["type"]
        if kind == "observation":
            validate_packet({k: row[k] for k in ("task", "public", "official", "input_sha256")}, task)
            public = row["public"]
            raw = base64.b64decode(public["raw_rgb"]["base64"])
            assert digest(raw) == public["raw_rgb"]["sha256"]
            assert (folder / f'observation-{row["serial"]:03d}-raw.png').read_bytes() == raw
            assert (folder / f'observation-{row["serial"]:03d}-0.png').read_bytes() == base64.b64decode(
                public["images"][0]["base64"])
            assert (folder / f'observation-{row["serial"]:03d}-depth.png').read_bytes() == base64.b64decode(
                public["depth"]["image"]["base64"])
            depth = np.load(folder / f'observation-{row["serial"]:03d}-depth-meters.npy', allow_pickle=False)
            assert np.array_equal(depth, decode_raw(public["depth"]), equal_nan=True)
            by_serial[row["serial"]] = public
        elif kind == "model_request":
            payload = load(folder / row["request_file"])
            assert digest(canonical(payload)) == row["payload_sha256"]
            assert payload["model"] == config["model"]["name"]
            for key in ("temperature", "max_tokens"):
                assert payload[key] == config["model"][key]
            for key, value in config["model"]["extra_body"].items():
                assert payload[key] == value
            assert row["reference_answer_visible"] is False
            requests.append(row)
            phase_text = payload["messages"][0]["content"]
            phase = phase_text.rsplit("CURRENT PHASE: ", 1)[1].split(".", 1)[0]
            phases[phase] += 1
        elif kind == "model_response_received":
            response = load(folder / row["response_file"])
            assert response["response_sha256"] == row["response_sha256"]
            for key in planner_usage:
                planner_usage[key] += response["usage"].get(key, 0)
        elif kind == "detector_request_sent":
            assert row["automatic_retry"] is False and row["backend"] == "qwen3vl8b"
            key = (row["observation_serial"], row["target_id"], row["query"])
            assert key not in requested_queries, "duplicate_same_frame_detection"
            requested_queries.append(key)
            detector_attempts.append(row)
        elif kind == "detector_result":
            public, report = by_serial[row["observation_serial"]], row["report"]
            assert report["model"] == config["detection"]["service"]["model"]
            assert report["image_sha256"] == public["raw_rgb"]["sha256"]
            assert report["service_identity_sha256"] == digest(canonical(config["detection"]["service"]))
            assert report["automatic_retry"] is False and report["reference_answer_visible"] is False
            raw = Image.open(io.BytesIO(base64.b64decode(public["raw_rgb"]["base64"]))).convert("RGB")
            assert report["image_size"] == list(raw.size)
            payload = {
                "model": report["model"], "messages": [{"role": "user", "content": [
                    {"type": "image_url", "image_url": {
                        "url": "data:image/png;base64," + public["raw_rgb"]["base64"]}},
                    {"type": "text", "text": detection_prompt(row["query"])}]}],
                "temperature": config["detection"]["temperature"], "top_p": 1,
                "max_tokens": config["detection"]["max_tokens"],
                "response_format": {"type": "json_object"},
            }
            assert digest(canonical(payload)) == report["request_sha256"]
            output = {"detections": [{"label": r["label"], "bbox_2d": r["bbox_1000"]}
                                     for r in report["detections"]]}
            assert digest(canonical(output)) == report["output_sha256"]
            for proposal in report["detections"]:
                assert proposal["label"] == row["query"]
                assert proposal["bbox"] == pixel_box(proposal["bbox_1000"], raw.size)
                assert "confidence" not in proposal
            for key in detector_usage:
                detector_usage[key] += report["usage"].get(key, 0)
            detectors.append(row)
        elif kind == "perception_tool_result":
            assert row["simulator_steps"] == row["simulator_observations"] == 0
            public = by_serial[row["result"]["observation_serial"]]
            assert row["result"]["raw_rgb_sha256"] == public["raw_rgb"]["sha256"]
            for image in row["images"]:
                assert file_digest(folder / image["file"]) == image["sha256"]
            if row["tool_request"]["name"] == "inspect_candidates":
                raw = Image.open(io.BytesIO(base64.b64decode(public["raw_rgb"]["base64"]))).convert("RGB")
                groups = {g["id"]: g for g in row["result"]["groups"]}
                for crop, meta in zip(row["result"]["crops"], row["images"]):
                    assert crop["bounds"] == crop_bounds(groups[crop["group_id"]]["bbox"], raw.size)
                    actual = Image.open(folder / meta["file"]).convert("RGB")
                    assert actual.size == raw.crop(crop["bounds"]).size
                    assert actual.tobytes() == raw.crop(crop["bounds"]).tobytes()
                    crops.append(crop)
        elif kind == "action_sent":
            actions.append(row)
        elif kind == "action_ack":
            acks.append(row)
    assert len(requests) <= config["max_requests"][task.environment]
    assert len(detectors) == result["perception"]["detector_requests"]
    assert len(crops) == result["perception"]["crop_count"]
    assert 0 <= len(actions) - len(acks) <= 1
    assert [a["action"] for a in actions[:len(acks)]] == [a["action"] for a in acks]
    plans = {e["request_index"]: e["executed_prefix"] for e in rows if e["type"] == "plan_execution"}
    for action in actions:
        assert action["action"] == plans[action["request_index"]][action["index_in_plan"]]
    replay = replay_episode(folder, config, rows, result)
    query_counts = {}
    for row in detectors:
        counts = query_counts.setdefault(row["query"], {"requests": 0, "proposals": 0, "empty": 0})
        counts["requests"] += 1
        counts["proposals"] += len(row["report"]["detections"])
        counts["empty"] += int(not row["report"]["detections"])
    reviews = [e for e in rows if e["type"] == "perception_action_review"]
    verdicts = Counter(item["verdict"] for e in reviews for item in e["perception_review"])
    last_response = next((load(folder / e["response_file"])
                          for e in reversed(rows) if e["type"] == "model_response_received"), None)
    last_keys, last_tool = [], None
    if last_response:
        try:
            value = json.loads(last_response["text"])
            if isinstance(value, dict):
                last_keys = sorted(value)
                if isinstance(value.get("tool_request"), dict):
                    last_tool = value["tool_request"].get("name")
        except (ValueError, TypeError):
            pass
    return {"model_requests": len(requests), "detector_requests": len(detectors), "crops": len(crops),
            "detector_requests_sent": len(detector_attempts),
            "detector_requests_without_result": len(detector_attempts) - len(detectors),
            "actions": len(actions), "acks": len(acks), "planner_usage": planner_usage,
            "detector_usage": detector_usage, "unknown_action_outcome": len(actions) != len(acks),
            "coordinates_and_crops_verified": True, "model_requests_by_phase": dict(phases),
            "host_elapsed_seconds": result["host_elapsed_seconds"], "queries": query_counts,
            "review_verdict_counts": dict(verdicts), "last_response_root_keys": last_keys,
            "last_tool_name": last_tool,
            "primary_operational_failure": next((e["code"] for e in rows if e["type"] == "failure"), None),
            "cleanup_failures": [e["code"] for e in rows if e["type"] == "cleanup_failure"], **replay}


def completed_audits():
    config = load(OUT / "config.json")
    audited = []
    for folder in sorted(task_folders().values()):
        path = folder / "result.json"
        rows = events(folder / "trajectory.jsonl")
        if rows[-1]["type"] != "episode_end":
            continue
        identity = {"trajectory_sha256": file_digest(folder / "trajectory.jsonl"),
                    "result_sha256": file_digest(path), "audit_script_sha256": file_digest(__file__)}
        destination = OUT / "verification" / f"{folder.name}-audit.json"
        cached = load(destination) if destination.exists() else None
        if cached and cached["identity"] == identity:
            audit = cached["audit"]
        else:
            audit = audit_episode(folder, config)
            destination.write_bytes(canonical({"identity": identity, "audit": audit}) + b"\n")
        audited.append({"task": folder.name, "status": load(path)["status"],
                        "model_requests": audit["model_requests"],
                        "detector_requests": audit["detector_requests"],
                        "actions": audit["actions"], "acks": audit["acks"]})
    return {"verified": len(audited), "generative_audit_requests": 0, "episodes": audited}


def finish():
    assert all((metadata / "exit.json").exists() for metadata, _ in run_roots()), "suite_has_not_exited"
    frozen, config = load(OUT / "frozen.json"), load(OUT / "config.json")
    for name, expected in frozen["snapshot_files"].items():
        assert file_digest(OUT / name) == expected
    for _, run in run_roots():
        assert load(run / "identity.json")["implementation_hashes"] == frozen["implementation_hashes"]
    old = load(ROOT / "target_detection_twenty/results.json")
    old_pairs = {p["identity"]: p for p in old["pairs"]}
    pairs, by_env = [], {}
    folders = task_folders()
    for task_row in config["episodes"]:
        task = Task(**task_row)
        folder = folders.get(task.directory)
        if folder is None:
            continue
        result = load(folder / "result.json")
        audit = audit_episode(folder, config)
        previous = old_pairs[task.identity]
        a, b = first_observation(folder), first_observation(previous["current_directory"])
        initial = {"available": a is not None and b is not None}
        if initial["available"]:
            x, y = a["public"], b["public"]
            ax = np.asarray(Image.open(io.BytesIO(base64.b64decode(x["raw_rgb"]["base64"]))).convert("RGB"))
            ay = np.asarray(Image.open(io.BytesIO(base64.b64decode(y["raw_rgb"]["base64"]))).convert("RGB"))
            initial.update(instruction_equal=x["instruction"] == y["instruction"],
                           actions_equal=x["actions"] == y["actions"],
                           rgb_pixels_equal=bool(np.array_equal(ax, ay)),
                           depth_png_equal=x["depth"]["image"]["sha256"] == y["depth"]["image"]["sha256"])
        success = int(result["status"] != "operational_error" and result["metrics"]["task_success"] == 1)
        pairs.append({"identity": task.identity, "task": task_row, "current_success": success,
                      "dino_success": previous["current_success"], "rgbd_success": previous["rgbd_success"],
                      "dino_status": previous["current_status"],
                      "status": result["status"], "error": result["error"],
                      "instruction": result["metrics"]["instruction"], "steps": result["metrics"]["num_steps"],
                      "metrics": result["metrics"],
                      "targets": result["perception"]["targets"], "audit": audit,
                      "initial_identity": initial, "directory": str(folder.resolve())})
    for env in ("alfred", "habitat", "manipulation", "navigation"):
        rows = [p for p in pairs if p["task"]["environment"] == env]
        previous = [p for p in old["pairs"] if p["task"]["environment"] == env]
        by_env[env] = {"selected": 5, "completed": len(rows),
                       "current_successes": sum(p["current_success"] for p in rows),
                       "dino_successes": sum(p["current_success"] for p in previous),
                       "rgbd_successes": sum(p["rgbd_success"] for p in previous)}
    report = {
        "checked_at": datetime.now(timezone.utc).isoformat(), "selected": 20, "completed": len(pairs),
        "current_successes": sum(p["current_success"] for p in pairs), "dino_successes": old["current_successes"],
        "rgbd_successes": old["rgbd_successes"], "by_environment": by_env, "pairs": pairs,
        "changed_to_success": [p["identity"] for p in pairs if p["current_success"] > p["dino_success"]],
        "changed_to_failure": [p["identity"] for p in pairs if p["current_success"] < p["dino_success"]],
        "planner_requests": sum(p["audit"]["model_requests"] for p in pairs),
        "detector_requests": sum(p["audit"]["detector_requests"] for p in pairs),
        "detector_requests_sent": sum(p["audit"]["detector_requests_sent"] for p in pairs),
        "detector_requests_without_result": sum(p["audit"]["detector_requests_without_result"] for p in pairs),
        "planner_usage": {k: sum(p["audit"]["planner_usage"][k] for p in pairs)
                          for k in ("prompt_tokens", "completion_tokens", "total_tokens")},
        "detector_usage": {k: sum(p["audit"]["detector_usage"][k] for p in pairs)
                           for k in ("prompt_tokens", "completion_tokens", "total_tokens")},
        "model_requests_by_phase": dict(sum(
            (Counter(p["audit"]["model_requests_by_phase"]) for p in pairs), Counter())),
        "actions": sum(p["audit"]["actions"] for p in pairs),
        "acks": sum(p["audit"]["acks"] for p in pairs),
        "crops": sum(p["audit"]["crops"] for p in pairs),
        "status_counts": dict(Counter(p["status"] for p in pairs)),
        "error_code_counts": dict(Counter(p["error"]["code"] for p in pairs if p["error"])),
        "host_elapsed_seconds": sum(p["audit"]["host_elapsed_seconds"] for p in pairs),
        "initial_identity_counts": {k: sum(p["initial_identity"].get(k, False) for p in pairs)
                                    for k in ("instruction_equal", "actions_equal",
                                              "rgb_pixels_equal", "depth_png_equal")},
        "operational_failures": [p["identity"] for p in pairs if p["status"] == "operational_error"],
        "unknown_action_outcomes": [p["identity"] for p in pairs if p["audit"]["unknown_action_outcome"]],
        "automatic_retry": False, "generative_audit_requests": 0,
    }
    assert report["completed"] == 20, "twenty_tasks_not_yet_complete"
    summaries = [load(run / "summary.json") for _, run in run_roots()]
    assert summaries[-1]["status"] == "finished"
    assert sum(summary["http_requests"] for summary in summaries) == report["planner_requests"]
    assert sum(summary["detector_requests"] for summary in summaries) == report["detector_requests"]
    assert all(summary["peak_http_concurrency"] == 1 for summary in summaries)
    assert report["planner_requests"] <= frozen["planning_model_request_cap"]
    report["request_cap"] = frozen["planning_model_request_cap"]
    report["peak_planner_concurrency"] = max(summary["peak_http_concurrency"] for summary in summaries)
    report["runs"] = [{"directory": str(run), "status": summary["status"],
                      "selected": summary["selected"],
                      "started": sum(r["started"] for r in summary["by_environment"].values())}
                     for (_, run), summary in zip(run_roots(), summaries)]
    preservation = load(OUT / "verification/preservation-after.json")
    remote_state = load(OUT / "verification/server-after.json")
    assert not preservation["missing"] and not preservation["changed"]
    assert not preservation["snapshot_changes"] and not preservation["continuation_config_changes"]
    assert not remote_state["owned_trial_workers"]
    assert (OUT / "verification/detector-after.json").is_file()
    report["final_verification"] = {
        "preserved_files": preservation["preserved_files"],
        "official_source_files": sum(preservation["official_source_counts"].values()),
        "owned_trial_workers": 0,
        "receipts": ["verification/preservation-after.json", "verification/server-after.json",
                     "verification/detector-after.json"],
    }
    (OUT / "results.json").write_bytes(canonical(report) + b"\n")
    lines = [
        "# Qwen3-VL-8B 检测器：同组 20 题结果", "",
        f"已完成 {len(pairs)}/20 题；新版成功 {report['current_successes']}/20，"
        f"原 GroundingDINO 流程 {old['current_successes']}/20，历史 RGB-D {old['rgbd_successes']}/20。", "",
        "| 环境 | 新版 Qwen3-VL | 原 DINO | 历史 RGB-D |",
        "| --- | --- | --- | --- |",
    ]
    lines += [f"| {env} | {r['current_successes']}/5 | {r['dino_successes']}/5 | {r['rgbd_successes']}/5 |"
              for env, r in by_env.items()]
    lines += ["", "## 配置与实际调用", "",
              "主模型为 Qwen3.8-Flash；检测器为 Qwen3-VL-8B-Instruct-FP8。"
              "题目选择、规划模型参数、few-shot、环境种子和规划请求预算与原 DINO 二十题一致。"
              "每题由主模型从任务指令提取类别，检测器使用同帧原始 RGB；"
              "0–1000 框转换到原图像素后分组、裁剪和复核。",
              f"规划请求 {report['planner_requests']} 次，{report['planner_usage']['total_tokens']:,} tokens；"
              f"检测请求已返回 {report['detector_requests']} 次，"
              f"未收到结果 {report['detector_requests_without_result']} 次；"
              f"已返回检测累计 {report['detector_usage']['total_tokens']:,} tokens。",
              f"20 题累计整题运行时长 {report['host_elapsed_seconds'] / 60:.1f} 分钟。"
              f"原 DINO 轮次 {old['diagnostics']['costs']['current']['host_elapsed_seconds'] / 60:.1f} 分钟，"
              f"规划 {old['model_requests']} 次／"
              f"{old['diagnostics']['costs']['current']['total_usage']['total_tokens']:,} tokens，"
              f"检测 {old['detector_requests']} 次。原 DINO 检测调用没有模型 token 用量。",
              "", "## 逐题对比", "",
              "| 题目 | 原 DINO | 新版 | RGB-D | 状态 | 失败代码 |",
              "| --- | --- | --- | --- | --- | --- |"]
    for p in pairs:
        code = p["error"]["code"] if p["error"] else "—"
        lines.append(f"| {p['identity']} | {p['dino_success']} | {p['current_success']} | "
                     f"{p['rgbd_success']} | {p['status']} | {code} |")
    lines += ["", "相较原 DINO 转为成功：" + (", ".join(report["changed_to_success"]) or "无") + "。",
              "相较原 DINO 转为失败：" + (", ".join(report["changed_to_failure"]) or "无") + "。",
              "", "## 失败与审计", "",
              "本轮状态分布：" + json.dumps(report["status_counts"], ensure_ascii=False) + "。",
              "有明确错误代码的题：" + json.dumps(report["error_code_counts"], ensure_ascii=False) + "。",
              "", "| 未成功题目 | 直接停止原因 |",
              "| --- | --- |"]
    for p in pairs:
        if p["current_success"]:
            continue
        code = p["error"]["code"] if p["error"] else None
        if p["status"] == "request_budget":
            reason = (f"用满 {p['audit']['model_requests']} 次规划请求，"
                      f"执行 {p['steps']} 个动作后未达到官方成功判定")
            if "task_progress" in p["metrics"]:
                reason += f"；官方任务进度 {p['metrics']['task_progress']:.0%}"
        elif p["status"] == "operational_error":
            reason = (f"原始错误 `{p['audit']['primary_operational_failure']}`；"
                      f"清理错误 {p['audit']['cleanup_failures']}；"
                      f"检测已发送但无结果 {p['audit']['detector_requests_without_result']} 次")
        elif code == "first_response_requires_targets":
            reason = "主模型首次响应没有提供必需的检测目标 targets"
        elif code == "frozen_targets_changed":
            reason = "主模型后续响应改写已冻结的 targets，触发协议校验"
        elif code == "detection_required_before_action":
            reason = "主模型尚未完成必需的检测便请求执行动作，触发协议校验"
        elif p["status"] == "official_done":
            reason = "环境结束，但官方 task_success=0"
        else:
            reason = f"{p['status']}；错误代码 `{code}`"
        lines.append(f"| {p['identity']} | {reason} |")
    lines += ["",
              f"运行分为 {len(report['runs'])} 个独立目录。运营失败题保留原结果，"
              "续跑仅包含尚未启动的题，未重试已请求过的失败题。"
              "运营原始错误和清理错误分别保存在逐题 audit 中。",
              f"原 DINO 轮次有 {old['operational_failures']} 次运营失败及"
              f" {old['diagnostics']['status_counts'].get('planner_error', 0)} 次规划协议错误；"
              "该对照衡量整条流程，成功率差值不能单独归因于检测器。",
              f"动作 ACK {report['acks']}/{report['actions']}，裁剪 {report['crops']} 个；"
              "请求内容、冻结参数、目标提取来源、候选分组、裁剪像素、复核和动作计划均已离线回放。",
              "与原 DINO 的初始观测匹配：" +
              ", ".join(f"{k}={v}/20" for k, v in report["initial_identity_counts"].items()) + "。",
              f"最终核验：原有 {preservation['preserved_files']} 个文件及冻结源码未变；"
              f"核验上游源文件 {report['final_verification']['official_source_files']} 个；"
              "本轮远端任务残留为 0。检测服务身份核验回执见 `verification/detector-after.json`。",
              "", "这是一轮固定配置的历史逐题对照；差值包含模型输出和仿真器观测波动。",
              "原始请求、响应、图像和工具轨迹分布在 `run/` 与 `continuations/remaining01/run/`；"
              "坐标、输入哈希和裁剪像素已离线核对。",
              "逐题初始 RGB/深度一致性及全部数值见 `results.json`。", ""]
    (OUT / "实验结果.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k not in ("pairs",)}, ensure_ascii=False))


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "status":
        print(json.dumps(status(), ensure_ascii=False))
    elif len(sys.argv) > 1 and sys.argv[1] == "audit-completed":
        print(json.dumps(completed_audits(), ensure_ascii=False))
    else:
        finish()
