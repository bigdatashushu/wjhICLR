"""Explicit, sealed tool-arm correction; reuse direct outputs without paid replay."""
from pathlib import Path
import shutil

import sr
from skill3d.small_image.blink_protocol import sha_value, strict_json


def state_at(root):
    return strict_json((Path(root) / "budget/state.json").read_text())


def seed(root, parent, seal_path, jobs):
    root, parent = Path(root), Path(parent).resolve()
    seal = sr.read(seal_path)
    old = sr.read(parent / "manifest.json")
    completed = sr.read(parent / "completed.json")
    state = state_at(parent)
    policy = {"max_requests": 80, "max_inflight": 8, "requests_per_second": 0.0}
    if (completed["manifest_sha256"] != old["content_sha256"]
            or completed["episodes"] != 40 or state["policy"] != policy
            or any(r["status"] != "completed" for r in state["requests"])
            or completed["budget"]["reserved_requests"] != len(state["requests"])
            or any(completed["budget"][k] for k in
                   ("error_requests", "unknown_requests", "active_requests"))
            or seal["parent_manifest_sha256"] != old["content_sha256"]
            or seal["verified_episodes"] != 20 or seal["model_requests"] != 20
            or seal["reference_answers_read"] is not False):
        raise ValueError("sr_continuation_parent_not_complete")
    old_jobs = {j["operation_id"]: j for j in old["jobs"]}
    direct = [j for j in jobs if j["spec"]["arm"] == "direct"]
    sealed = {r["operation_id"]: r for r in seal["episodes"]}
    if len(direct) != 20 or len(sealed) != 20 or set(sealed) != {
            j["operation_id"] for j in direct}:
        raise ValueError("sr_continuation_direct_denominator")
    files, requests = {}, 0
    for job in direct:
        op = job["operation_id"]
        receipt = sr.read(parent / "episodes" / (op + ".json"))
        result = receipt["result"]
        row = sealed[op]
        if (old_jobs[op] != job or sha_value(job) != row["job_sha256"]
                or receipt["job_sha256"] != row["job_sha256"]
                or receipt["content_sha256"] != row["episode_sha256"]
                or result["request_count"] != 1 or len(result["request_trace"]) != 1
                or result["status"] != "answered"
                or result["reference_answer_visible"] is not False
                or result["attempted_tool_calls"] or result["execution_trace"]
                or sha_value(result["request_trace"][0]["payload"]) != row["payload_sha256"]
                or sum(r["context"].get("variant_id") == op for r in state["requests"]) != 1):
            raise ValueError("sr_continuation_direct_identity_changed")
        paths = [parent / "episodes" / (op + ".json"),
                 *sorted((parent / "requests" / op).glob("*.json"))]
        if len(paths) != 5:
            raise ValueError("sr_continuation_direct_receipts_incomplete")
        for path in paths:
            sr.read(path)
            files[str(path.relative_to(parent))] = sr.file_sha(path)
        requests += result["request_count"]
    prior_count = sum(sr.read(parent / "episodes" / (j["operation_id"] + ".json"))[
        "result"]["request_count"] for j in old["jobs"])
    new_ops = {j["operation_id"] for j in jobs if j["spec"]["arm"] == "skill_tools"}
    if (prior_count != len(state["requests"]) or len(new_ops) != 20
            or new_ops & set(old_jobs)
            or len(state["requests"]) >= 80):
        raise ValueError("sr_continuation_budget_or_operation_identity")
    if any((root / name).exists() for name in
           ("budget", "requests", "episodes", "continuation.json")):
        raise ValueError("sr_continuation_destination_not_empty")
    for name, digest in files.items():
        target = root / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(parent / name, target)
        if sr.file_sha(target) != digest:
            raise ValueError("sr_continuation_copy_mismatch")
    shutil.copytree(parent / "budget", root / "budget")
    value = {
        "schema": "blink-sr-explicit-tool-correction/1",
        "reason": "initial solver exposed pair_point_distance outside the frozen SR catalog",
        "parent_root": str(parent), "parent_manifest_sha256": old["content_sha256"],
        "parent_completion_sha256": completed["content_sha256"],
        "direct_seal": seal, "copied_files": files,
        "prior_requests": len(state["requests"]), "reused_direct_episodes": 20,
        "reused_direct_requests": requests,
        "excluded_parent_tool_requests": len(state["requests"]) - requests,
        "prior_budget_state_sha256": sr.file_sha(parent / "budget/state.json"),
        "prior_request_rows_sha256": sha_value(state["requests"]),
        "new_operations": sorted(new_ops), "request_cap": 80,
        "reference_answers_read": False, "automatic_retry": False,
        "old_tool_outputs_included_in_comparison": False,
    }
    return sr.write_once(root / "continuation.json", value)


def verify(root, manifest, *, initial=False):
    """Use copied receipts and a frozen budget prefix; parent is never needed online."""
    root = Path(root)
    value = sr.read(root / "continuation.json")
    if manifest.get("continuation_sha256") != value["content_sha256"]:
        raise ValueError("sr_continuation_binding_changed")
    for name, digest in value["copied_files"].items():
        if sr.file_sha(root / name) != digest:
            raise ValueError("sr_continuation_reused_receipt_changed")
    state = state_at(root)
    count = value["prior_requests"]
    if (state["policy"] != {"max_requests": 80, "max_inflight": 8, "requests_per_second": 0.0}
            or sha_value(state["requests"][:count]) != value["prior_request_rows_sha256"]):
        raise ValueError("sr_continuation_prior_budget_changed")
    new_ops = set(value["new_operations"])
    if any(r["context"].get("variant_id") not in new_ops for r in state["requests"][count:]):
        raise ValueError("sr_continuation_unexpected_paid_operation")
    if initial:
        if (len(state["requests"]) != count
                or sr.file_sha(root / "budget/state.json") != value["prior_budget_state_sha256"]
                or {p.stem for p in (root / "episodes").glob("*.json")} != {
                    j["operation_id"] for j in manifest["jobs"] if j["spec"]["arm"] == "direct"}
                or {p.name for p in (root / "requests").iterdir()} != {
                    j["operation_id"] for j in manifest["jobs"] if j["spec"]["arm"] == "direct"}):
            raise ValueError("sr_continuation_paid_replay_forbidden")
    return value
