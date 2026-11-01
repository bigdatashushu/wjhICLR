"""Read-only completed-answer snapshots; no calls to the model or its tools."""
from collections import Counter
from datetime import datetime
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path("/home/wangy/wangjh/harness3d/four-datasets-20261031/cv3d-simple-paired-20261101-r02")


def snapshot(root=ROOT):
    root = Path(root)
    sys.path.insert(0, str(root / "source_frozen/src"))
    from skill3d.small_image.datasets import normalize_answer

    manifest = json.loads((root / "manifest.json").read_text())
    answers = {
        (r["source_split"], r["sample_id"]): r["reference_answer"]
        for r in json.loads(Path(manifest["config"]["answers_path"]).read_text())
        if r["benchmark"] == "cv3d"
    }
    out = {
        "at": datetime.now().astimezone().isoformat(),
        "inspection_model_requests": 0, "results": [], "summary": {},
    }
    for phase, split, id_field in (
        ("protocol_train", "train", "train_protocol_ids"),
        ("test", "test", "test_ids"),
    ):
        selected = set(manifest[id_field])
        counts = {arm: Counter() for arm in ("tools", "direct")}
        for arm in counts:
            for path in (root / phase / "episodes" / arm).glob("*/result.json"):
                try:
                    row = json.loads(path.read_text())
                except json.JSONDecodeError:
                    continue  # The writer has not finished publishing this receipt.
                data = {k: v for k, v in row.items() if k != "receipt_sha256"}
                encoded = json.dumps(
                    data, ensure_ascii=False, sort_keys=True,
                    allow_nan=False, separators=(",", ":")).encode()
                if hashlib.sha256(encoded).hexdigest() != row["receipt_sha256"]:
                    raise ValueError("completed result receipt mismatch")
                key = row["sample_id"]
                if key not in selected or row["source_split"] != split:
                    raise ValueError("result outside the frozen phase selection")
                options = row["input_identity"]["problem"]["options"]
                gt = normalize_answer(answers[(split, key)], tuple(options))
                correct = row["status"] == "answered" and row["answer"] == gt
                counts[arm]["completed"] += 1
                counts[arm]["correct"] += int(correct)
                out["results"].append({
                    "event_id": f"{phase}:{arm}:{key}",
                    "at": datetime.fromtimestamp(path.stat().st_mtime).astimezone().isoformat(),
                    "phase": phase, "arm": arm, "sample_id": key,
                    "task_type": row["task_type"], "prediction": row["answer"],
                    "gt": gt, "correct": correct, "status": row["status"],
                    "requests": row.get("request_count", 0),
                    "tool_calls": row.get("attempted_tool_calls", 0),
                    "successful_tools": row.get("succeeded_tool_calls", 0),
                    "elapsed_seconds": row.get("elapsed_seconds"),
                })
        out["summary"][phase] = {
            arm: {
                **count, "total": len(selected),
                "accuracy_recorded": count["correct"] / count["completed"]
                if count["completed"] else None,
                "accuracy_full": count["correct"] / len(selected),
            } for arm, count in counts.items()
        }
    out["results"].sort(key=lambda r: (r["at"], r["event_id"]))
    budget = json.loads((root / "budget/state.json").read_text())
    out["budget_used"] = len(budget["requests"])
    out["budget_limit"] = budget["policy"]["max_requests"]
    out["finished"] = (root / "exit.json").exists()
    out["failed"] = (root / "failure.json").exists()
    out["completed"] = (root / "completion.json").exists()
    return out


if __name__ == "__main__":
    print(json.dumps(snapshot(), ensure_ascii=False))
