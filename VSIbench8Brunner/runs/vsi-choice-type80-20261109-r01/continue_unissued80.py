#!/usr/bin/env python3
"""Finish only never-issued episodes, retaining all original native evidence."""
import hashlib
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
ARM = ROOT / "harness"
REC = ROOT / "continuation_unissued"
sys.path[:0] = [str(ROOT), str(ROOT / "source_frozen/scripts"),
                str(ROOT / "source_frozen/src")]


def read(p):
    return json.loads(p.read_text())


def sha(p):
    h = hashlib.sha256()
    with p.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def save(p, value):
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def original_exited():
    launch = read(ROOT / "execute.pid.json")
    p = Path("/proc") / str(launch["pid"]) / "stat"
    if p.exists():
        fields = p.read_text().rsplit(")", 1)[1].split()
        assert fields[19] == launch["process_identity"]["start_ticks"]
        assert fields[0] == "Z", "original execution is still active"
    assert Path("/proc/sys/kernel/random/boot_id").read_text().strip() == launch[
        "process_identity"]["boot_id"]


def verify_preserved(plan):
    for name, identity in plan["preserved_files"].items():
        p = ARM / name
        assert p.stat().st_size == identity["bytes"] and sha(p) == identity["sha256"]
    old = read(REC / "original_request_budget.json")
    budget = read(ARM / "request_budget.json")
    assert sha(REC / "original_request_budget.json") == plan["original_budget_sha256"]
    assert budget["max_requests"] == old["max_requests"] == 7988
    assert budget["used"] == len(budget["requests"]) <= 7988
    assert budget["requests"][:old["used"]] == old["requests"]
    original_index = read(REC / "original_trajectory_index.json")["questions"]
    current = read(ARM / "trajectory_index.json")["questions"]
    assert all(current[qa] == row for qa, row in original_index.items())
    journal = (ARM / "parallel_events.jsonl").read_bytes()
    prefix = (REC / "original_parallel_events.jsonl").read_bytes()
    assert journal.startswith(prefix)


def prepare_plan():
    from type80_adapter import verify_sources
    verify_sources()
    original_exited()
    assert not REC.exists(), "continuation plan already exists"
    selected = {str(q["qa_id"]) for q in read(ROOT / "sampling_manifest.json")["questions"]}
    completed = {p.stem for p in (ARM / "outcomes").glob("*.json")}
    assert len(completed) == 49 and len(selected - completed) == 31
    assert read(ARM / "live_status.json")["status"] == "stopped"
    for p in (ARM / "solver_requests").glob("*/*.json"):
        assert p.parent.name in completed
        assert read(p)["status"] in ("returned", "unavailable")
    # Refuse partial preparation that would resend already issued work.
    scopes = {q["qa_id"]: q["scene_id"] for q in read(
        ARM / "preflight_verified.json")["questions"]}
    for qa in selected - completed:
        binding = ARM / "bindings" / (qa + ".json")
        if binding.exists():
            assert (ARM / "scene_audits" / (scopes[qa] + ".json")).exists()
            assert (ARM / "sam2_receipts" / (qa + ".json")).exists()
        else:
            assert not list((ARM / "preparation" / qa).rglob("*"))
            assert not list((ARM / "parser_requests" / qa).rglob("*.json"))
    preserved = {}
    for directory in ("outcomes", "solver_requests", "parser_requests", "host_errors",
                      "bindings", "sam2_receipts", "scene_audits", "trajectories",
                      "program_executions", "solver_trace"):
        for p in (ARM / directory).rglob("*"):
            if p.is_file() and not p.name.endswith(".lock"):
                preserved[str(p.relative_to(ARM))] = {
                    "bytes": p.stat().st_size, "sha256": sha(p)}
    # Native trajectory filenames can use a different directory; pin from index.
    for row in read(ARM / "trajectory_index.json")["questions"].values():
        p = ARM / row["file"]
        preserved[row["file"]] = {"bytes": p.stat().st_size, "sha256": sha(p)}
    REC.mkdir()
    for name in ("request_budget.json", "trajectory_index.json", "live_status.json",
                 "parallel_execution.json", "parallel_events.jsonl"):
        (REC / ("original_" + name)).write_bytes((ARM / name).read_bytes())
    plan = {
        "schema": "vsi-type80-never-issued-continuation/1",
        "preserved_questions": sorted(completed),
        "pending_qa_ids": sorted(selected - completed),
        "preserved_files": preserved,
        "original_budget_sha256": sha(REC / "original_request_budget.json"),
        "original_used": read(ARM / "request_budget.json")["used"],
        "preparation_request_cap": 7988,
        "sampling_sha256": sha(ROOT / "sampling_manifest.json"),
        "source_manifest_sha256": sha(ROOT / "source_manifest.json"),
        "script_sha256": sha(Path(__file__)),
        "original_execution_receipt_sha256": sha(ROOT / "execute.pid.json"),
        "original_terminal_service_failure_qa_ids": ["5105"],
        "service_failure_policy": "retain terminal unavailable outcome; never retry; continue unissued tasks",
        "issued_model_requests_repeated": 0, "automatic_retry": False,
        "source_frozen_modified": False, "scientific_configuration_changed": False,
        "reference_answers_read": False, "reference_answer_visible": False,
        "at": time.time(),
    }
    save(REC / "plan.json", plan)
    verify_preserved(plan)
    print(json.dumps({k: v for k, v in plan.items() if k != "preserved_files"}))


def execute():
    import type80_adapter as adapter
    adapter.verify_sources()
    original_exited()
    plan = read(REC / "plan.json")
    assert sha(Path(__file__)) == plan["script_sha256"]
    assert sha(ROOT / "sampling_manifest.json") == plan["sampling_sha256"]
    assert sha(ROOT / "source_manifest.json") == plan["source_manifest_sha256"]
    verify_preserved(plan)
    import pilot40 as pilot
    pilot.base_config = adapter.base_config
    pilot.initialize = adapter.initialize
    import harness_runtime_identity_guard
    harness_runtime_identity_guard.install()
    original_write = pilot.write
    original_export = pilot.export_question
    original_solve = pilot.solve_question
    preserved = set(plan["preserved_questions"])

    def write(p, value):
        p = Path(p)
        if p.parent == ARM and p.name in (
                "parallel_execution.json", "live_status.json", "parallel_failure.json"):
            p = REC / p.name
        original_write(p, value)

    def export(root, qa, *args, **kwargs):
        if str(qa) in preserved:
            return
        return original_export(root, qa, *args, **kwargs)

    def solve(root, cfg, provider, overview, question, qa, **kwargs):
        assert qa in plan["pending_qa_ids"]
        assert not (root / "outcomes" / (qa + ".json")).exists()
        assert not list((root / "solver_requests" / qa).glob("*.json"))
        try:
            return original_solve(root, cfg, provider, overview, question, qa, **kwargs)
        except RuntimeError as exc:
            if str(exc) != "main solver operational failure; no automatic retry":
                raise
            result = read(root / "outcomes" / (qa + ".json"))
            out = result["outcome"]
            reqs = [read(p) for p in sorted((root / "solver_requests" / qa).glob("*.json"))]
            assert out["failure_code"] == "service_error" and not out["legal_answer"]
            assert out["answer"] is None
            assert reqs and reqs[-1]["status"] == "unavailable"
            assert all(r["status"] == "returned" for r in reqs[:-1])
            assert reqs[-1]["error_type"] in ("TimeoutError", "HTTPError", "URLError")
            assert all(r["automatic_retry"] is False for r in reqs)
            save(REC / "terminal_service_failures" / (qa + ".json"), {
                "qa_id": qa, "outcome_sha256": sha(root / "outcomes" / (qa + ".json")),
                "unavailable_request_sha256": sha(sorted(
                    (root / "solver_requests" / qa).glob("*.json"))[-1]),
                "error_type": reqs[-1]["error_type"], "at": time.time(),
                "automatic_retry": False,
            })
            return result

    pilot.write, pilot.export_question, pilot.solve_question = write, export, solve
    import pilot40_parallel
    try:
        pilot40_parallel.main()
        verify_preserved(plan)
        assert {p.stem for p in (ARM / "outcomes").glob("*.json")} == set(
            plan["pending_qa_ids"]) | preserved
        save(REC / "verified.json", {
            "status": "complete", "completed": 80,
            "preserved_questions": 49, "never_issued_questions_completed": 31,
            "plan_sha256": sha(REC / "plan.json"),
            "script_sha256": sha(Path(__file__)),
            "final_budget_sha256": sha(ARM / "request_budget.json"),
            "final_index_sha256": sha(ARM / "trajectory_index.json"),
            "issued_model_requests_repeated": 0, "automatic_retry": False,
            "scientific_configuration_changed": False, "source_frozen_modified": False,
            "reference_answers_read": False, "at": time.time(),
        })
    except Exception as exc:
        save(REC / "failed.json", {"error_type": type(exc).__name__,
                                  "automatic_retry": False, "at": time.time()})
        raise


if __name__ == "__main__":
    phase = sys.argv.pop(1)
    if phase == "plan":
        prepare_plan()
    elif phase == "execute":
        execute()
    else:
        raise ValueError("unknown phase")
