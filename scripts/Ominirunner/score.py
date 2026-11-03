"""Host-only implementation of the pinned VADAR summary scoring rules."""
from __future__ import annotations

import argparse
from collections import Counter
import math
from pathlib import Path
import sys

if not __package__:
    root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(root / "scripts"))
    __package__ = "Ominirunner"

from .common import digest, read, write

ENGINE_COMMIT = "56018ebc2ecf7a430fc94e64e80e2768b0931739"
ENGINE_SHA = "952daaf1441d7d9677488e04fe4fb1fa1218fb4e9caa53d425b2c1978feb78a0"
THRESHOLDS = [0.5, 0.45, 0.40, 0.35, 0.3, 0.25, 0.2, 0.15, 0.1, 0.05]


def evaluate(kind, gt, pred):
    group = kind
    if kind == "str":
        group = "yes_no" if gt in ("yes", "no") else "other_text"
        return group, float(gt == pred.lower()), bool(pred)
    if kind == "int":
        try:
            value = int(pred)
        except (ValueError, TypeError):
            return group, 0.0, False
        return group, float(int(gt) == value), True
    if kind == "float":
        if not math.isfinite(float(gt)) or float(gt) <= 0:
            raise ValueError("official_float_gt_must_be_positive")
        try:
            value = float(pred)
        except (ValueError, TypeError):
            return group, 0.0, False
        return group, sum(abs(float(gt)-value)/float(gt) < t
                          for t in THRESHOLDS)/len(THRESHOLDS), math.isfinite(value)
    raise ValueError("unsupported_official_answer_type")


def score(root, annotation_path):
    # Loaded only after online worker completion; never passed into any child.
    if read(root / "completed.json")["status"] != "complete":
        raise ValueError("incomplete_trial")
    frozen = read(root / "frozen.json")
    inputs = read(root / "inputs.json")
    annotations = read(annotation_path)["questions"]
    selected = {}
    for item in inputs:
        row = annotations[int(item["sample_id"])]
        if (str(row["question_index"]) != item["sample_id"] or row["question"] != item["question"]
                or row["answer_type"] != item["task_type"]):
            raise ValueError("official_question_binding_changed")
        selected[item["sample_id"]] = row
    table, scored = {}, []
    for arm in ("direct", "skill"):
        groups, statuses, usage, times = {}, Counter(), {}, []
        legal = requests = tool_calls = 0
        for item in inputs:
            result = read(root / "episodes" / (item["sample_id"]+"-"+arm) / "result.json")
            reuse = frozen.get("reused_direct", {}).get(item["sample_id"]) if arm == "direct" else None
            expected_identity = reuse["source_run_identity"] if reuse else digest(frozen)
            if reuse:
                audited = read(root / "audit.json")
                if audited["status"] != "passed" or audited["run_identity"] != digest(frozen):
                    raise ValueError("reuse_requires_completed_audit")
            if (result["receipt_sha256"] != digest({
                    k: v for k, v in result.items() if k != "receipt_sha256"})
                    or result["run_identity"] != expected_identity):
                raise ValueError("result_identity_changed")
            label = selected[item["sample_id"]]
            pred = result["answer"] if result["status"] == "answered" else ""
            group, value, valid = evaluate(label["answer_type"], label["answer"], pred)
            legal += int(valid)
            groups.setdefault(group, []).append(value)
            statuses[result["status"]] += 1
            requests += result["request_count"]
            tool_calls += result["tool_counts"]["calls"]
            times.append(result["elapsed_seconds"])
            for key, value_usage in result["usage"].items():
                usage[key] = usage.get(key, 0) + value_usage
            scored.append({"sample_id": item["sample_id"], "arm": arm, "group": group,
                           "score": value, "legal": valid, "prediction": pred,
                           "status": result["status"], "requests": result["request_count"],
                           "reused": bool(reuse), "source_run_identity": expected_identity,
                           "tool_calls": result["tool_counts"]["calls"]})
        table[arm] = {
            "metrics": {g: {"n": len(v), "value": sum(v)/len(v)} for g, v in groups.items()},
            "legal": legal, "n": len(inputs), "statuses": dict(statuses),
            "requests": requests, "tool_calls": tool_calls, "usage": usage,
            "mean_episode_seconds": sum(times)/len(times),
            "sample_weighted_score": sum(sum(v) for v in groups.values())/len(inputs)}
    summary = {"schema": "omni-official-host-score/1", "engine_commit": ENGINE_COMMIT,
               "engine_source_sha256": ENGINE_SHA, "thresholds": THRESHOLDS,
               "run_identity": digest(frozen), "pairs": len(inputs), "table": table,
               "scored": scored, "ground_truth_in_report": False}
    summary["new_requests"] = sum(r["requests"] for r in scored if not r["reused"])
    summary["reused_requests"] = sum(r["requests"] for r in scored if r["reused"])
    write(root / "scores.json", summary, exclusive=True)
    print({arm: value["metrics"] for arm, value in table.items()})
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--annotations", type=Path, required=True)
    args = parser.parse_args()
    score(args.root, args.annotations)
