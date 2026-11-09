"""Audit saved real official baselines without simulator or model requests."""
import base64
import json
from pathlib import Path
from types import SimpleNamespace

from .planner import OfficialPlanner
from .protocol import Task
from .sources import ROOT, REVISION, Sources, canonical, digest, file_digest


def parity(kind, sources):
    # These old modules compile verified prompt methods; no provider is created.
    if kind == "navigation":
        from official_bare import OfficialBarePlanner
        from protocol import ACTIONS
        old = OfficialBarePlanner(None)
        old.reset()
        actions = list(ACTIONS)
    else:
        from household_planner import OfficialHouseholdPlanner
        actions = ["Find a table", "Pick up a cup", "Put the cup"]
        old = OfficialHouseholdPlanner(None, kind, "base", actions)
        old.reset(actions)
    new = OfficialPlanner(sources, Task(kind, "base", 0), "qwen3.8-flash",
                          3 if kind == "navigation" else 10, actions)
    # Same image bytes, action space, instruction and public feedback.
    png = b"offline-message-parity"
    obs = SimpleNamespace(png=png, instruction="Move the cup to the table.", actions=actions)
    public = {"instruction": obs.instruction, "images": [{"base64": base64.b64encode(png).decode()}]}
    for feedback in (None, "Last action executed.", "The action could not execute."):
        if feedback is not None:
            if kind == "navigation":
                old.update_info(action=0, feedback=feedback, last_action_success=True, step=1)
            else:
                old.update(0, feedback)
            new.feedback(0, {"env_feedback": feedback})
        prior = old.messages(obs, obs.instruction) if kind == "navigation" else old.messages(obs)
        if new.messages(public) != prior:
            raise ValueError("old_and_new_official_messages_differ")
    return True


def main():
    sources = Sources()
    sources.verify_all()
    results, evidence = [], {}
    for kind, alias in (("navigation", "nav"), ("alfred", "alfred"), ("habitat", "habitat")):
        folder = ROOT / "runs" / f"qwen-bare-official-suite-{alias}-3"
        startup = json.loads((folder / "startup.json").read_text())
        summary = json.loads((folder / "summary.json").read_text())
        config = json.loads((folder / "config.json").read_text())
        audit = json.loads((folder / "audit.json").read_text())
        assert summary["status"] == "complete" and audit["status"] == "pass"
        assert all(audit["checks"].values())
        assert summary["upstream_revision"] == REVISION
        assert startup["reference_answer_visible"] is False and startup["automatic_retry"] is False
        assert startup["uses_official_messages"] and startup["uses_official_prompt"]
        assert config["model"] == "qwen3.8-flash" and config["max_tokens"] == 4096
        assert config["seed"] == 137 and config["enable_thinking"] is False
        for name, sha in startup["source_sha256"].items():
            assert file_digest(ROOT / name) == sha, name
        assert parity(kind, sources)
        files = [folder / f for f in ("startup.json", "config.json", "summary.json", "audit.json")]
        for index in range(3):
            task = Task(kind, "base", index)
            if kind == "navigation":
                path = folder / "episodes" / f"base-{index:03d}.json"
                row = json.loads(path.read_text())
                trace = [json.loads(line) for line in (folder / "trace.jsonl").read_text().splitlines()
                         if json.loads(line)["task_id"] == f"base:{index:03d}"]
                trace_path = folder / "trace.jsonl"
                steps = [e for e in trace if e.get("event") == "environment_step"]
                calls = [e for e in trace if "request_index" in e]
                assert len(steps) == row["environment_steps"] and len(calls) == row["planner_requests"]
                assert steps[-1]["done"] and bool(steps[-1]["task_success"]) == row["task_success"]
            else:
                episode = folder / f"episode-{index:03d}"
                path, trace_path = episode / "result.json", episode / "trace.jsonl"
                row = json.loads(path.read_text())
                trace = [json.loads(line) for line in trace_path.read_text().splitlines()]
                steps = [e for e in trace if e["kind"] == "environment_step"]
                calls = [e for e in trace if e["kind"] == "model_request"]
                assert row == summary["results"][index]
                assert len(steps) == row["num_steps"] and len(calls) == row["model_requests"]
                assert row["task_success"] or row["num_invalid_actions"] >= 10 or row["num_steps"] >= 30
                for e in steps + calls:
                    step = e.get("step", e.get("environment_step"))
                    image = episode / f"step-{step:03d}.png"
                    assert file_digest(image) == e["image_sha256"]
                    files.append(image)
                files.append(episode / "input.json")
            assert all(e["reference_answer_visible"] is False and e["model_tool_calls"] == 0
                       and e["reported_model_sha256"] == digest(b"qwen3.8-flash") for e in calls)
            files.extend([path, trace_path])
            results.append({
                "task": task.as_dict(), "origin": "reused_real_official_baseline",
                "source_result": str(path), "source_trace": str(trace_path),
                "status": "official_done", "task_success": int(row["task_success"]),
                "num_steps": len(steps), "model_requests": len(calls),
                "original_result": row})
        evidence[kind] = {
            "initial_and_feedback_message_parity": True, "original_audit_sha256": file_digest(folder / "audit.json"),
            "original_source_identity_verified": True,
            "files": {str(p.relative_to(ROOT)): file_digest(p) for p in set(files)}}
    report = {"schema": "official-baseline-reuse/1", "status": "pass", "reused": 9,
              "results": results, "evidence": evidence, "model_requests": 0, "simulator_launches": 0,
              "limitations": ["Historical runs retain message/response hashes and every action receipt; "
                              "they do not retain raw model request/response bodies."],
              "excluded": {"old_single_action_direct": 15, "manipulation_first_action_smoke": True}}
    output = ROOT / "official_direct/verification/reuse-20261107.json"
    if output.exists():
        raise ValueError("reuse_receipt_already_exists")
    output.write_bytes(canonical(report) + b"\n")
    print(json.dumps({"status": "pass", "reused": 9, "model_requests": 0, "simulator_launches": 0}))


if __name__ == "__main__":
    main()
