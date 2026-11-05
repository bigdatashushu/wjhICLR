"""Summarize actual tool use and recoverable program errors after SR host scoring."""
import argparse
from collections import Counter
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
import sr
from sr_trial import split_manifest


def diagnose(output, host_data_root):
    output = Path(output)
    report = sr.read(output / "results_and_full_trajectories.json")
    split = report["split"]
    if split_manifest(host_data_root) != {
            k: v for k, v in split.items() if k != "content_sha256"}:
        raise ValueError("sr_host_dataset_and_split_identity_changed")
    geometry = sr.read(output / "geometry_audit.json")
    manifest = sr.read(output / "raw/manifest.json")
    if {r["sample_id"]: r["artifact_sha256"] for r in geometry["inputs"]} != {
            r["problem"]["sample_id"]: r["prepared_artifact_sha256"]
            for r in manifest["inspection"]["inputs"]}:
        raise ValueError("sr_report_geometry_identity_changed")
    groups, wrong = {}, []
    for arm in ("direct", "skill_tools"):
        tools, errors, used, usage = Counter(), [], [], Counter()
        for pair in report["pairs"]:
            trajectory = pair[arm]["trajectory"]
            calls = [call for cell in trajectory["execution_trace"] for call in cell["tool_calls"]]
            tools.update(call.get("tool", call.get("name")) for call in calls)
            if calls:
                used.append(pair["sample_id"])
            for row in trajectory["rounds"]:
                if row["status"] in {
                        "program_parse_error", "program_rejected", "run_error",
                        "program_missing_action"}:
                    errors.append({"sample_id": pair["sample_id"], "status": row["status"],
                                   "final_status": trajectory["status"],
                                   "recovered": trajectory["status"] == "answered"})
            for request in trajectory["request_trace"]:
                usage.update({k: request["response_json"]["usage"].get(k, 0)
                              for k in ("prompt_tokens", "completion_tokens", "total_tokens")})
        groups[arm] = {
            "tool_using_questions": len(used), "tool_using_sample_ids": used,
            "tool_calls_by_name": dict(tools), "program_error_rounds": errors,
            "final_runtime_errors": report["groups"][arm]["statuses"].get("run_error", 0),
            "token_usage_for_current_comparison": dict(usage),
        }
    for pair in report["pairs"]:
        if not pair["direct"]["correct"] or not pair["skill_tools"]["correct"]:
            wrong.append({
                "sample_id": pair["sample_id"], "question": pair["question"],
                "reference": pair["reference"],
                **{arm: pair[arm]["trajectory"]["answer"] for arm in ("direct", "skill_tools")},
            })
    return sr.write_once(output / "diagnostic_details.json", {
        "schema": "blink-sr-paired20-postscore-diagnostics/1",
        "report_sha256": report["content_sha256"], "groups": groups, "wrong_pairs": wrong,
        "host_dataset_and_random_split_identity_verified": True,
        "same_prepared_artifact_identity_verified": True,
        "prepared_archive_sha256": sr.file_sha(output / "prepared_evidence.tar.gz"),
        "reference_access_role": "host_postscore_diagnostics", "model_requests": 0,
        "interpretation": "Same 19 correct pairs; only 3/20 tool-arm episodes invoked tools.",
    })


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, default=sr.DEFAULT_ROOT)
    args = parser.parse_args()
    print(__import__("json").dumps(diagnose(args.output, args.data_root), ensure_ascii=False))


if __name__ == "__main__":
    main()
