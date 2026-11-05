"""Export full inference trajectories with prompt text and image identities."""
from __future__ import annotations

import base64
import hashlib
from collections import Counter
from pathlib import Path

from .common import digest, read, write


def export(root):
    root = Path(root)
    frozen, audited = read(root / "frozen.json"), read(root / "audit.json")
    if audited["status"] != "passed" or audited["run_identity"] != digest(frozen):
        raise ValueError("trajectories_require_audit")
    cases = []
    for item in read(root / "inputs.json"):
        arms = {}
        for arm in ("skill", "direct"):
            directory = root / "episodes" / (item["sample_id"] + "-" + arm)
            result = read(directory / "result.json")
            prompts = []
            for row in result["rounds"]:
                name = f'{row["round"]:02d}.messages.json'
                payload = read(directory / name)
                if digest(payload) != row["request"]["messages_sha256"]:
                    raise ValueError("trajectory_prompt_changed")
                parts = []
                for part in payload[0]["content"]:
                    if part["type"] == "text":
                        parts.append(part)
                    else:
                        raw = base64.b64decode(part["image_url"]["url"].split(",", 1)[1])
                        parts.append({"type": "image", "sha256": hashlib.sha256(raw).hexdigest(),
                                      "bytes": len(raw)})
                prompts.append({"round": row["round"], "role": "user", "content": parts,
                                "original_messages_file": str(directory.relative_to(root) / name)})
            arms[arm] = {"result": result, "prompts": prompts,
                         "reuse": frozen.get("reused_direct", {}).get(item["sample_id"])
                         if arm == "direct" else None}
        cases.append({"sample_id": item["sample_id"], "question": item["question"],
                      "task_type": item["task_type"], "image_sha256": item["image_sha256"],
                      "input_sha256": item["input_sha256"], "arms": arms})
    threshold = frozen["config"].get("text_record_threshold", 10000)
    oversized = [{"sample_id": case["sample_id"], "arm": arm, "round": row["round"],
                  "text_chars": row["request"]["text_chars"],
                  "over_threshold_chars": row["request"]["text_chars"]-threshold}
                 for case in cases for arm, data in case["arms"].items()
                 for row in data["result"]["rounds"] if row["request"]["text_chars"] > threshold]
    activity = {}
    for arm in ("direct", "skill"):
        counts, usage = Counter(), Counter()
        for case in cases:
            result = case["arms"][arm]["result"]
            counts.update(result.get("evidence_activity") or {})
            usage.update(result.get("usage", {}))
        activity[arm] = {"evidence_activity": dict(counts), "usage": dict(usage)}
    value = {"schema": "omni-full-inference-trajectories/3", "run_identity": digest(frozen),
             "pairs": len(cases), "ground_truth_included": False,
             "text_record_threshold": threshold, "over_threshold_requests": oversized,
             "new_requests": audited["charged_requests"],
             "reused_direct": audited.get("reused_direct", 0),
             "activity_by_arm": activity, "cases": cases}
    write(root / "trajectories.json", value, exclusive=True)
    return value
