#!/usr/bin/env python3
"""Preserve the stopped preparation attempt and seal its reusable responses."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import time

from skill3d.infra.m5_identity import file_identity


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    root = args.run_dir
    history = root / "attempt_history"
    history.mkdir(exist_ok=True)
    replay = root / "review_replay"
    replay.mkdir(exist_ok=False)
    (replay / "visual").mkdir()
    records, successful, failures = {}, [], []
    for source in (root / "preparation").glob("**/visual/*.json"):
        value = json.loads(source.read_text())
        if value.get("target_description", {}).get("status") != "described":
            continue
        target = replay / "visual" / source.name
        shutil.copyfile(source, target)
        records[value["candidate_id"]] = {
            "path": target.relative_to(replay).as_posix(),
            "identity": file_identity(target), "source": str(source),
        }
        (successful if value["status"] == "observed" else failures).append(value["candidate_id"])
    if len(records) != 2 or len(successful) != 1 or len(failures) != 1:
        raise ValueError("unexpected first-attempt response count")
    (replay / "manifest.json").write_text(json.dumps({
        "schema": "vsi-review-replay/1", "files": records,
        "reference_answer_visible": False}, indent=2) + "\n")
    for source in (root / "bindings").glob("*.json"):
        value = json.loads(source.read_text())
        if value["stats"].get("binding_error"):
            source.rename(history / ("attempt01-binding-" + source.name))
    budget = root / "request_budget.json"
    value = json.loads(budget.read_text())
    if value["max_requests"] != 3884 or value["used"] != 4:
        raise ValueError("first-attempt budget changed")
    shutil.copyfile(budget, history / "attempt01-request-budget.json")
    value["max_requests"] = 3885
    value["admission_failure_allowance"] = 1
    budget.write_text(json.dumps(value, indent=2) + "\n")
    (history / "admission_repair.json").write_text(json.dumps({
        "schema": "vsi-pilot-admission-repair/1", "repaired_at": time.time(),
        "reason": "original reviewer has one nonblocking inference slot",
        "description_concurrency": 2, "reviewer_concurrency": 1,
        "preserved_description_receipts": 2, "preserved_successful_review_receipts": 1,
        "preserved_unavailable_review_receipts": 1,
        "original_request_cap": 3884, "request_cap_with_admission_failure": 3885,
        "automatic_retry": False, "reference_answer_visible": False,
    }, indent=2) + "\n")
    print(json.dumps({"descriptions_reused": 2, "successful_reviews_reused": 1,
                      "failed_reviews_preserved": 1, "max_preparation_requests": 3885}))


if __name__ == "__main__":
    main()
