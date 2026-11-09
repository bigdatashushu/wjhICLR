"""Audit the completed pilot, actual RGB pairing and preserved runtime identities."""
from collections import Counter
import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(ROOT), str(ROOT.parent / "src")]
from action_contract import plain, public_hash
from action_identity import source_identity
from action_trajectory import restore
from audit_action_trajectory import audit
from protocol import canonical, sha256


def initial(record):
    path = Path(record["trajectory_path"])
    trace = json.loads(path.read_bytes())
    event = next(e for e in trace["events"] if e["kind"] == "observation")
    event = restore(event, path.parent)
    import io
    return np.asarray(Image.open(io.BytesIO(event["png"])).convert("RGB")), event["observation"]


def main():
    path = ROOT / "runs/task-card-pilot-r05.json"
    run = json.loads(path.read_bytes())
    result = audit(path)
    assert result["status"] == "passed", result
    assert run["audit"]["status"] == "passed", run["audit"]
    manifest = run["manifest"]
    assert manifest["implementation_source"] == source_identity()
    assert manifest["implementation_source"] == json.loads((HERE / "deployment.json").read_bytes())["source"]
    assert manifest["automatic_retry"] is False and manifest["reference_answer_visible"] is False
    old_run = json.loads((ROOT / "runs/task-card-15-full-context-r04.json").read_bytes())
    old_manifest = old_run["manifest"]
    unchanged = ["model_config_sha256", "budget_sha256_by_environment", "method_sha256_by_environment",
                 "method_library_sha256", "evidence_config_sha256", "input_policy", "execution_policy"]
    assert all(manifest[k] == old_manifest[k] for k in unchanged)
    inventory = json.loads((HERE / "embodied_diagnosis_inventory_20261106.json").read_bytes())["records"]
    old = {(r["episode_id"], r["version"]): r for r in inventory}
    index = json.loads((ROOT / "runs/task-card-pilot-r05-episodes/trajectories.json").read_bytes())
    lookup = {r["episode_id"]: ROOT / "runs/task-card-pilot-r05-episodes" / r["directory"] / "trajectory.json"
              for r in index["episodes"]}
    rows = []
    for attempt in run["attempts"]:
        ep = attempt["episode_id"]
        trace_path = lookup[ep]
        trace = json.loads(trace_path.read_bytes())
        events = [restore(e, trace_path.parent) for e in trace["events"]]
        assert trace["events"][-1]["kind"] == "episode_result"
        assert trace["reference_answer_visible"] is False and trace["automatic_retry"] is False
        first = next(e for e in events if e["kind"] == "observation")["observation"]
        current, _ = initial({"trajectory_path": str(trace_path)})
        comparisons = {}
        for v in ("Direct", "Full", "r02", "r04"):
            original, obs = initial(old[(ep, v)])
            delta = np.abs(current.astype(np.int16) - original.astype(np.int16))
            assert obs["instruction"] == first["instruction"]
            assert obs["action_table_sha256"] == first["action_table_sha256"]
            comparisons[v] = {
                "pixel_equal": bool(np.array_equal(current, original)),
                "mean_channel_abs_difference": float(delta.mean()),
                "max_channel_abs_difference": int(delta.max()),
                "instruction_equal": True, "initial_action_table_equal": True}
        requests = [e for e in events if e["kind"] == "model_request"]
        for e in requests:
            assert e["receipt"]["reference_answer_visible"] is False
            assert e["receipt"]["model_config_sha256"] == manifest["model_config_sha256"]
        responses = [e for e in events if e["kind"] == "model_response"]
        cards = {e["receipt"]["decision_id"]: [] for e in requests}
        for e in requests:
            cards[e["receipt"]["decision_id"]].append(json.loads(e["messages"][1]["content"][0]["text"]))
        repair_counts = Counter(
            json.loads(e["messages"][1]["content"][0]["text"])["repair"].get("code", "none")
            for e in requests)
        significant = []
        for e in requests:
            card = json.loads(e["messages"][1]["content"][0]["text"])
            if card["repair"]:
                significant.append({"event": e["index"], "decision_id": e["receipt"]["decision_id"],
                                    "goal": card["subgoal"], "facts": card["facts"],
                                    "latest": card["latest"], "repair": card["repair"]})
        actions = [{"event": e["index"], "id": e["action_id"], "name": e["action_name"]}
                   for e in events if e["kind"] == "action_submission"]
        rows.append({
            "episode_id": ep, "success": attempt["task_success"], "progress": attempt["task_progress"],
            "steps": attempt["environment_steps"], "qwen": len(requests),
            "r04_success": old[(ep, "r04")]["success"],
            "initial_pixel_equal": comparisons["r04"]["pixel_equal"], "initial_comparisons": comparisons,
            "terminal_reason": attempt["terminal_reason"], "status": attempt["status"],
            "trajectory_path": str(trace_path), "trajectory_sha256": sha256(trace_path.read_bytes()),
            "actions": actions, "repair_prompt_codes": dict(repair_counts),
            "significant_repair_inputs": significant,
            "responses": [{"event": e["index"], "decision_id": e["decision_id"],
                           "output": json.loads(e["text"])} for e in responses],
            "all_request_text_over_target": sum(e["receipt"]["text_chars"] > 16000 for e in requests)})
    assert len(rows) == 7
    assert all(r["status"] == "finished" for r in rows)
    successes = sum(r["success"] for r in rows)
    before = sum(r["r04_success"] for r in rows)
    report = {
        "schema": "embodied-r05-pilot-verification/1", "status": "passed",
        "task_denominator": len(rows), "successes": successes, "r04_selected_successes": before,
        "improved_on_selected_tasks": successes > before,
        "run_sha256": sha256(path.read_bytes()), "trajectory_audit": result,
        "runtime_audit": run["audit"], "unchanged_identity_fields": unchanged,
        "source_identity_matches_deployment": True, "automatic_retry": False,
        "reference_answer_visible": False, "operational_failure_attempts": 0,
        "episodes": rows,
        "narrative": f"r05独立七题pilot：{successes}/7成功；同七题r04为{before}/7。"
                     "模型、Skill、预算、后端和官方任务来源身份保持一致。"
                     "实际初始RGB差异逐题列于JSON，导航不据任务ID直接宣称严格像素配对。"
                     "这是联合控制/提示修复的开发集结果，不是全15题或最终测试结果。",
        "unresolved": "失败题的完整提案、修复输入与动作链见episodes；"
                      "ALFRED000/004的整任务visual子目标被复制并反复advance，是需要单独修复的生命周期问题。"}
    with (HERE / "pilot-verification.json").open("xb") as f:
        f.write(canonical(report) + b"\n")
    print(json.dumps({k: v for k, v in report.items() if k not in {"episodes", "trajectory_audit", "runtime_audit"}}))


if __name__ == "__main__":
    main()
