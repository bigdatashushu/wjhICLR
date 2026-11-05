"""Independently check final BLINK pair identities, effective requests and scores."""
from __future__ import annotations

import argparse
from collections import Counter
import json
import os
from pathlib import Path
import sys


def audit(root):
    sys.path.insert(0, str(root / "driver"))
    import paired
    exp = paired.load_exp(root / "code/source")
    manifest = exp.read(root / "manifest.json")
    completion = exp.read(root / "completion.json")
    frozen = exp.read(root / "prediction_freeze.json")
    comparison = exp.read(root / "comparison.json")
    paired.verify(exp, root, manifest)
    if (completion["status"] != "complete"
            or completion["manifest_sha256"] != manifest["content_sha256"]
            or completion["prediction_freeze_sha256"] != frozen["content_sha256"]
            or completion["comparison_sha256"] != comparison["content_sha256"]
            or comparison["prediction_freeze_sha256"] != frozen["content_sha256"]):
        raise ValueError("completion_binding_changed")
    errors = []
    arms = {arm: [] for arm in paired.ARMS}
    all_request_paths = set()
    fingerprints = Counter()
    phases = Counter()
    for job in manifest["jobs"]:
        arm = job["identity"]["arm"]
        destination = root / "results" / arm / (job["operation_id"] + ".json")
        receipt = exp.read(destination)
        paired.validate_receipt(exp, root, job, receipt)
        if frozen["receipts"].get(job["operation_id"]) != receipt["content_sha256"]:
            raise ValueError("prediction_freeze_changed")
        arms[arm].append(receipt)
        rounds = receipt["result"]["rounds"]
        responses = [exp.read(root / relative) for relative in sorted(receipt["request_records"])
                     if relative.endswith(".response.json")]
        if len(rounds) != len(responses):
            raise ValueError("request_round_binding_mismatch")
        for round_, response in zip(rounds, responses):
            phases[arm + "/" + round_["phase"]] += 1
            actual = response["request_identity"]
            params = actual["parameters"]
            is_category = round_["phase"] == "category"
            if (params["model"] != "qwen3.8-flash" or params["seed"] != 137
                    or params["temperature"] != 0 or actual["parameter_status"] != "effective_payload"
                    or params["max_tokens"] != (256 if is_category else 8192)
                    or params["enable_thinking"] != (not is_category)):
                errors.append({"operation_id": job["operation_id"], "code": "effective_parameters"})
            expected_images = [] if is_category else [
                image["sha256"] for image in job["identity"]["images"]]
            if [image["jpeg_sha256"] for image in actual["images_in_order"]] != expected_images:
                errors.append({"operation_id": job["operation_id"], "code": "actual_image_order"})
            if round_.get("request_identity") != actual:
                errors.append({"operation_id": job["operation_id"], "code": "round_request_identity"})
            if round_.get("model_response") != response["model_response"]:
                errors.append({"operation_id": job["operation_id"], "code": "round_response_binding"})
            fingerprint = response["response_identity"].get("backend_fingerprint")
            fingerprints[str(fingerprint)] += 1
            if arm == "fixed_skill" and not is_category:
                if round_.get("delivered_skill_sha256") != job["identity"]["skill_sha256"]:
                    errors.append({"operation_id": job["operation_id"], "code": "actual_skill_delivery"})
            if arm == "direct":
                worker = job["worker"]
                if ("prepared_root" in worker or "artifact_sha256" in worker
                        or worker["skill_text"] or job["identity"]["allowed_tools"]):
                    raise ValueError("direct_geometry_boundary_changed")
        for path in receipt["request_records"]:
            if path in all_request_paths:
                raise ValueError("shared_request_record")
            all_request_paths.add(path)
    if all_request_paths != {str(p.relative_to(root)) for p in (root / "requests").rglob("*.json")}:
        raise ValueError("unaccounted_request_record")
    for arm, receipts in arms.items():
        if len(receipts) != 93 or len({r["result"]["sample_id"] for r in receipts}) != 93:
            raise ValueError("final_denominator_changed")
    # Independent scorer, only after workers exited and predictions were frozen.
    path = Path(manifest["profile"]["inputs"]["host_answers"])
    if (exp.file_identity(path) != manifest["preflight"]["inputs"]["host_answers"]
            or exp.file_sha(path) != manifest["reference_file_sha256"]):
        raise ValueError("reference_identity_changed")
    reference_rows = exp.strict_json(path.read_text())
    references = {r["sample_id"]: r["reference_answer"] for r in reference_rows
                  if r["benchmark"] == "blink" and r["source_split"] == "test"}
    independent = {}
    for arm, receipts in arms.items():
        correct = sum(r["result"]["status"] == "answered"
                      and r["result"]["answer"] == references[r["result"]["sample_id"]]
                      for r in receipts)
        independent[arm] = {"correct": correct, "questions": len(receipts),
                            "accuracy": correct / len(receipts)}
        if any(independent[arm][k] != completion["scores"][arm][k]
               for k in independent[arm]):
            raise ValueError("independent_score_mismatch")
    if len(comparison["rows"]) != 93 or sum(completion["paired"].values()) != 93:
        raise ValueError("paired_comparison_denominator")
    lookup = {arm: {r["result"]["sample_id"]: r["result"] for r in receipts}
              for arm, receipts in arms.items()}
    independent_pairs = Counter()
    for row in comparison["rows"]:
        sample_id = row["sample_id"]
        flags = {}
        for arm in paired.ARMS:
            result = lookup[arm][sample_id]
            flags[arm] = (result["status"] == "answered" and result["answer"] == references[sample_id])
            if (row[arm]["correct"] != flags[arm] or row[arm]["answer"] != result["answer"]
                    or row["reference_answer"] != references[sample_id]):
                raise ValueError("comparison_row_changed")
        outcome = ("both_correct" if all(flags.values()) else "skill_only" if flags["fixed_skill"]
                   else "direct_only" if flags["direct"] else "both_wrong")
        if outcome != row["outcome"]:
            raise ValueError("comparison_outcome_changed")
        independent_pairs[outcome] += 1
    if dict(independent_pairs) != completion["paired"]:
        raise ValueError("paired_score_changed")
    budget = exp.RequestGovernor(root / "budget", exp.RequestGovernorConfig(paired.CAP, 4)).summary()
    if (budget != completion["request_usage"]
            or any(budget[k] for k in ("error_requests", "unknown_requests", "active_requests"))):
        raise ValueError("final_budget_changed")
    exit_record = json.loads((root / "exit.json").read_text())
    startup = json.loads((root / "startup.json").read_text())
    proc = Path("/proc") / str(startup["runner_pid"])
    owned_alive = proc.exists() and (
        (proc / "stat").read_text().split(") ", 1)[1].split()[19] == startup["runner_start_ticks"])
    if exit_record["returncode"] != 0 or owned_alive:
        raise ValueError("runner_not_successfully_exited")
    if errors:
        raise ValueError("effective_request_audit_failed")
    record = exp.write_once(root / "final_audit.json", {
        "status": "passed", "at": paired.now(), "manifest_sha256": manifest["content_sha256"],
        "completion_sha256": completion["content_sha256"],
        "prediction_freeze_sha256": frozen["content_sha256"],
        "auditor_sha256": exp.file_sha(Path(__file__)), "episodes": 186, "pairs": 93,
        "requests": budget["reserved_requests"], "phases": dict(phases), "errors": errors,
        "scores_independently_recomputed": independent,
        "paired_independently_recomputed": dict(independent_pairs),
        "actual_original_images_and_order_verified": True, "effective_parameters_verified": True,
        "fixed_skill_delivery_verified": True, "direct_geometry_boundary_verified": True,
        "reference_answer_visible_to_solver": False, "runner_exited": True,
        "backend_fingerprints": dict(fingerprints),
        "backend_fingerprint_observed": "None" not in fingerprints and "" not in fingerprints,
    })
    print(json.dumps(record, sort_keys=True))


if __name__ == "__main__":
    try:
        parser = argparse.ArgumentParser(description=__doc__)
        parser.add_argument("--root", type=Path, required=True)
        args = parser.parse_args()
        for name in list(os.environ):
            if any(t in name.upper() for t in ("API_KEY", "TOKEN", "SECRET", "PASSWORD", "MODELS_KEY")):
                os.environ.pop(name, None)
        audit(args.root.resolve())
    except BaseException as exc:
        print(json.dumps({"status": "failed", "error_type": type(exc).__name__}))
        raise SystemExit(1)
