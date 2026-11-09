"""Explicit, label-free recovery of the first stopped BLINK direct phase.

Preparation never sends requests. A separate frozen runner executes the prepared
child campaign only after the operator authorizes the timeout change and replay.
"""
from __future__ import annotations

from contextlib import ExitStack
from copy import deepcopy
import fcntl
import json
from pathlib import Path
import shutil
import subprocess

from . import blink_experiment as experiment
from skill3d.small_image.blink_protocol import sha_value, strict_json

SCHEMA = "harness3d-blink-recovery/1"
ALLOWED_SOURCE_CHANGES = {
    "configs/experiments/blink.json",
    "src/skill3d/synthesis/bailian_client.py",
    "src/skill3d/small_image/blink_protocol.py",
    "src/skill3d/evaluation/blink_experiment.py",
    "src/skill3d/evaluation/blink_recovery.py",
}


def _legacy_files(parent):
    files = [parent/"manifest.json", parent/"budget/state.json"]
    for directory in ("results", "requests", "intents", "skills"):
        files.extend(sorted((parent/directory).rglob("*.json")))
    return {str(p.relative_to(parent)): experiment.file_sha(p) for p in files}


def inspect(parent):
    """Inspect one stopped direct prefix, with no labels or model calls."""
    parent = Path(parent).resolve()
    manifest = experiment.read(parent/"manifest.json")
    if "recovery" in manifest or (parent/"completion.json").exists():
        raise ValueError("requires_original_incomplete_campaign")
    if experiment.source_files(parent/"code/source") != manifest["source_files"]:
        raise ValueError("parent_frozen_source_changed")
    if manifest["profile"]["model"]["timeout_s"] != 180:
        raise ValueError("unexpected_parent_timeout")
    # Preflight checks public inputs and GT file metadata, never GT contents.
    if experiment.preflight(manifest["profile"]) != manifest["preflight"]:
        raise ValueError("parent_input_or_runtime_changed")
    files = _legacy_files(parent)
    if any((parent/"skills").rglob("*.json")):
        raise ValueError("recovery_requires_direct_phase_only")
    rows = strict_json((parent/"budget/state.json").read_text())
    policy = manifest["profile"]["runtime"]
    if rows["policy"] != {
        "max_requests": policy["max_requests"],
        "max_inflight": policy["request_concurrency"], "requests_per_second": 0.0,
    }:
        raise ValueError("parent_budget_policy_changed")
    requests = rows["requests"]
    if len(requests) != 25 or [r["status"] for r in requests] != ["completed"]*24+["error"]:
        raise ValueError("requires_24_completed_and_one_failed_request")
    if [r["request_id"] for r in requests] != list(range(25)):
        raise ValueError("parent_request_ids_changed")
    operations = {p.stem: experiment.read(p) for p in (parent/"results/direct").glob("*.json")}
    if len(operations) != 25 or len(list((parent/"results").rglob("*.json"))) != 25:
        raise ValueError("unexpected_parent_operation_count_or_stage")
    tests = [p for p in manifest["preflight"]["problems"] if p["source_split"] == "test"]
    reused, failures = {}, []
    for index, request in enumerate(requests):
        op_id = request["context"]["variant_id"]
        operation = operations[op_id]
        identity = operation["identity"]
        cache = manifest["preflight"]["cache_identities"][tests[index]["sample_id"]]
        expected_identity = {
            "stage": "direct", "input_sha256": tests[index]["input_sha256"],
            "skill_sha256": sha_value(""), "seed": 137,
            "solver": manifest["profile"]["solver"], "images": cache["images"],
            "artifact_sha256": cache["artifact_sha256"], "model": manifest["profile"]["model"],
            "campaign": manifest["content_sha256"], "mode": "direct", "allowed_tools": [],
        }
        if (sha_value(identity) != op_id or identity != expected_identity
                or request["category"] != "solver"
                or operation["status"] != ("complete" if index < 24 else "stopped")):
            raise ValueError("parent_operation_or_request_binding_changed")
        intent_path = f"intents/{op_id}.json"
        if experiment.read(parent/intent_path)["identity"] != identity:
            raise ValueError("parent_operation_intent_changed")
        records = {f"requests/{op_id}/{name}": files[f"requests/{op_id}/{name}"]
                   for name in ("001.intent.json", "001.response.json")}
        if len(list((parent/"requests"/op_id).glob("*.json"))) != 2:
            raise ValueError("requires_single_request_direct_operations")
        response = experiment.read(parent/f"requests/{op_id}/001.response.json")
        request_intent = experiment.read(parent/f"requests/{op_id}/001.intent.json")
        expected_parameters = {
            "max_tokens": 8192, "seed": 137, "temperature": 0, "enable_thinking": True,
        }
        if (request_intent["parameters"] != expected_parameters
                or response["request_identity"]["parameters"] != {
                    "model": "qwen3.8-flash", "top_p": None, "response_format": None,
                    **expected_parameters}):
            raise ValueError("parent_actual_request_parameters_changed")
        if index < 24:
            if (operation["request_records"] != records or response["status"] != "received"
                    or operation["result"]["request_count"] != 1
                    or operation["result"]["sample_id"] != tests[index]["sample_id"]
                    or operation["result"]["source_split"] != "test"
                    or response["response_identity"].get("model") != "qwen3.8-flash"
                    or response["response_identity"].get("model_reported") is not True):
                raise ValueError("parent_completed_response_changed")
            reused[identity["input_sha256"]] = f"results/direct/{op_id}.json"
        else:
            if (operation.get("error_type") != "ServiceUnavailable"
                    or response["status"] != "error"
                    or response.get("error_type") != "ServiceUnavailable"
                    or response.get("model_response") is not None
                    or response.get("reasoning_content") is not None
                    or response.get("usage") or response.get("response_identity")):
                raise ValueError("failed_request_has_output_or_unknown_failure")
            failures.append(op_id)
    if len(files) != 102:
        # manifest + budget + 25 operation intents + 25 results + 50 request records
        raise ValueError("unexpected_legacy_records")
    if _legacy_files(parent) != files:
        raise ValueError("parent_changed_during_inspection")
    return {
        "schema": SCHEMA, "parent_campaign_sha256": manifest["content_sha256"],
        "parent_manifest_sha256": files["manifest.json"], "parent_root": str(parent),
        "legacy_files": files, "reused_direct_operations": reused,
        "authorized_failure_operation": failures[0],
        "authorized_failure_result_sha256": files[f"results/direct/{failures[0]}.json"],
        "approved_error_request_id": 24, "carried_requests": 25,
        "remaining_requests": policy["max_requests"]-25,
        "timeout_change": {"before": 180, "after": 600},
        "model_requests": 0, "reference_answers_read": False,
    }


def prepare(repo, parent, child, *, authorized_failure_sha256):
    """Freeze a separately identified child; preserve the parent's bytes and budget."""
    repo, parent, child = Path(repo).resolve(), Path(parent).resolve(), Path(child).resolve()
    if child.exists() or parent == child or parent in child.parents or child in parent.parents:
        raise ValueError("recovery_requires_separate_new_directory")
    if repo == child or repo in child.parents or child in repo.parents:
        raise ValueError("recovery_output_must_be_outside_source")
    # Lock existing files read-only. No parent records or lock contents are changed.
    with ExitStack() as locks:
        lock_paths = [parent/"operator.lock"]
        lock_paths.extend(sorted((parent/"budget").glob("slot-*.lock")))
        for path in lock_paths:
            stream = locks.enter_context(path.open("rb"))
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        plan = inspect(parent)
        if authorized_failure_sha256 != plan["authorized_failure_result_sha256"]:
            raise ValueError("explicit_failed_operation_authorization_required")
        original = experiment.read(parent/"manifest.json")
        source = experiment.source_files(repo)
        delta = {p: {"before": original["source_files"].get(p), "after": source.get(p)}
                 for p in sorted(set(source) | set(original["source_files"]))
                 if source.get(p) != original["source_files"].get(p)}
        if not set(delta) <= ALLOWED_SOURCE_CHANGES or any(
                row["after"] is None for row in delta.values()):
            raise ValueError("repair_changes_exceed_reviewed_scope")
        profile = deepcopy(original["profile"])
        profile["model"]["timeout_s"] = 600
        commit = subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"],
                                         text=True).strip()
        if commit != original["git_commit"]:
            raise ValueError("repair_must_be_based_on_parent_commit")
        dirty = bool(subprocess.check_output(
            ["git", "-C", str(repo), "status", "--porcelain"], text=True).strip())
        plan.update(source_changes=delta, source_patch_sha256=sha_value(delta),
                    repair_source_base_commit=commit, repair_source_uncommitted=dirty)
        for relative, digest in source.items():
            target = child/"code/source"/relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(repo/relative, target)
            if experiment.file_sha(target) != digest:
                raise ValueError("repair_source_changed_during_freeze")
        for relative, digest in plan["legacy_files"].items():
            target = child/"origin"/relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(parent/relative, target)
            if experiment.file_sha(target) != digest:
                raise ValueError("legacy_record_changed_during_copy")
        (child/"budget").mkdir()
        shutil.copyfile(parent/"budget/state.json", child/"budget/state.json")
        if _legacy_files(parent) != plan["legacy_files"] or experiment.source_files(repo) != source:
            raise ValueError("parent_or_repair_source_changed_during_preparation")
        manifest = experiment.write_once(child/"manifest.json", {
            "schema": original["schema"], "profile": profile,
            "preflight": original["preflight"], "source_files": source,
            "git_commit": original["git_commit"],
            "reference_file_sha256": original["reference_file_sha256"],
            "recovery": plan,
        })
    return manifest


def verify(campaign):
    """Accept only the explicitly carried error; any new failure still stops."""
    recovery = campaign.manifest["recovery"]
    if recovery["schema"] != SCHEMA:
        raise ValueError("unsupported_recovery_schema")
    for relative, digest in recovery["legacy_files"].items():
        if experiment.file_sha(campaign.root/"origin"/relative) != digest:
            raise ValueError("legacy_recovery_record_changed")
    original = experiment.read(campaign.root/"origin/manifest.json")
    expected_profile = deepcopy(original["profile"])
    expected_profile["model"]["timeout_s"] = 600
    if (campaign.profile != expected_profile or campaign.inputs != original["preflight"]
            or recovery["parent_campaign_sha256"] != original["content_sha256"]
            or recovery["parent_manifest_sha256"] != recovery["legacy_files"]["manifest.json"]):
        raise ValueError("recovery_identity_changed")
    old = strict_json((campaign.root/"origin/budget/state.json").read_text())
    current = strict_json((campaign.root/"budget/state.json").read_text())
    carried = recovery["carried_requests"]
    if (carried != 25 or current["policy"] != old["policy"]
            or current["requests"][:carried] != old["requests"]
            or len(old["requests"]) != carried):
        raise ValueError("recovery_budget_prefix_changed")
    errors = [r["request_id"] for r in current["requests"] if r["status"] == "error"]
    if (errors != [recovery["approved_error_request_id"]]
            or recovery["approved_error_request_id"] != 24):
        raise RuntimeError("unresolved_or_failed_prior_requests")


def reused_operation(campaign, problem, identity):
    recovery = campaign.manifest["recovery"]
    relative = recovery["reused_direct_operations"].get(problem.input_sha256)
    if identity["stage"] != "direct" or relative is None:
        return None
    original = experiment.read(campaign.root/"origin/manifest.json")
    expected = {**identity, "campaign": original["content_sha256"],
                "model": original["profile"]["model"]}
    row = experiment.read(campaign.root/"origin"/relative)
    if row["identity"] != expected or row["status"] != "complete":
        raise ValueError("reused_operation_identity_changed")
    return row


def main(argv=None):
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--authorize-failure-sha256")
    parser.add_argument("--prepare", action="store_true")
    args = parser.parse_args(argv)
    if args.prepare:
        if not args.output_dir or not args.authorize_failure_sha256:
            parser.error("--prepare requires output and explicit failed-operation SHA-256")
        result = prepare(Path(__file__).resolve().parents[3], args.parent, args.output_dir,
                         authorized_failure_sha256=args.authorize_failure_sha256)["recovery"]
    else:
        result = inspect(args.parent)
    print(json.dumps({key: value for key, value in result.items() if key not in (
        "legacy_files", "reused_direct_operations", "parent_root")},
        ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(json.dumps({"status": "stopped", "error_type": type(exc).__name__}))
        raise SystemExit(2)
