#!/usr/bin/env python3
"""Wait for both frozen arms and seal their shared forty-question completion."""

import argparse
import ctypes
import hashlib
import json
import os
from pathlib import Path
import select
import shutil
import time


def read(path):
    return json.loads(path.read_text())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--direct-continuation", type=Path, required=True)
    args = parser.parse_args()
    root, extra = args.run_root.resolve(), args.direct_continuation.resolve()
    assert root != extra and not (root / "pair_completion.json").exists()
    libc = ctypes.CDLL(None, use_errno=True)
    fd = libc.inotify_init1(os.O_NONBLOCK)
    assert fd >= 0
    for directory in (root, extra):
        assert libc.inotify_add_watch(fd, os.fsencode(directory), 0x80 | 0x8 | 0x100) >= 0
    try:
        while not ((root / "recovery_receipt.json").exists() and (extra / "exit.json").exists()):
            select.select([fd], [], [])
            os.read(fd, 65536)
    finally:
        os.close(fd)
    assert read(root / "system_exit.json")["exit_code"] == 0
    assert read(extra / "exit.json")["status"] == "complete"
    assert read(root / "solve_receipt.json")["status"] == "complete"
    assert read(extra / "direct_receipt.json")["status"] == "complete"
    experiment, continuation = read(root / "diagnostic.json"), read(extra / "diagnostic.json")
    pinned = continuation["direct_continuation"]
    assert pinned["origin_identity"] == experiment["identity_sha256"]
    assert pinned["origin_source_sha256"] == experiment["source_sha256"]
    assert pinned["origin_native_root"] == str(root)
    assert continuation["input_files"] == experiment["input_files"]
    assert continuation["runtime"] == experiment["runtime"]
    for relative, sha in pinned["files"].items():
        assert hashlib.sha256((root / relative).read_bytes()).hexdigest() == sha
    expected = {row["qa_id"] for row in experiment["variants"]}
    groups = []
    for directory, identity, wanted in (
        (root, experiment["identity_sha256"], set(pinned["completed_qa_ids"])),
        (extra, continuation["identity_sha256"], set(pinned["remaining_qa_ids"]))):
        rows = [read(p) for p in (directory / "direct/responses").glob("*.json")]
        assert len(rows) == len(wanted) and {row["qa_id"] for row in rows} == wanted
        assert len(list((directory / "direct/intents").glob("*.json"))) == len(wanted)
        assert all(row["campaign_identity"] == identity and row["status"] == "completed"
                   and row["reference_answer_visible"] is False and row["tool_calls"] == 0
                   and row["actual_request_identity"]["parameters"]["enable_thinking"] is True
                   and len(row["actual_request_identity"]["images_in_order"]) == 32
                   for row in rows)
        groups.append(wanted)
    assert not groups[0] & groups[1] and groups[0] | groups[1] == expected and len(expected) == 40
    summary = read(root / "system/summary.json")
    assert summary["status"] == "complete" and summary["completed_variants"] == 40
    requests = read(root / "requests/state.json")
    assert all(row["status"] == "completed" for row in requests["requests"])
    assert len(requests["requests"]) <= experiment["runtime"]["max_requests"]
    assert requests["peak_inflight"] <= requests["policy"]["max_inflight"] == 8
    direct_requests = [row for row in requests["requests"]
                       if row["context"]["variant_id"].startswith("baseline:")]
    assert len(direct_requests) == 40
    for ids, identity in zip(groups, (experiment["identity_sha256"], continuation["identity_sha256"])):
        wanted = {"baseline:" + hashlib.sha256(
            (json.dumps(qa, sort_keys=True, ensure_ascii=False, allow_nan=False, indent=2)
             + "\n").encode()
        ).hexdigest()[:24] for qa in ids}
        observed = [row for row in direct_requests if row["context"]["campaign_id"] == identity]
        assert len(observed) == len(wanted) and {row["context"]["variant_id"] for row in observed} == wanted
    destination = root / "direct_continuation"
    assert not destination.exists()
    destination.mkdir()
    for name in ("diagnostic.json", "direct_start.json", "direct_receipt.json", "direct_health.json",
                 "launcher_start.json", "exit.json", "code/code_identity.json"):
        path = destination / name
        path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(extra / name, path)
    shutil.copytree(extra / "direct", destination / "direct")
    files = {str(path.relative_to(destination)): hashlib.sha256(path.read_bytes()).hexdigest()
             for path in destination.rglob("*.json")}
    receipt = {
        "status": "complete", "finished_at": time.time(),
        "diagnostic_identity": experiment["identity_sha256"],
        "continuation_identity": continuation["identity_sha256"],
        "system_source_sha256": experiment["source_sha256"],
        "original_direct_source_sha256": experiment["source_sha256"],
        "continuation_source_sha256": continuation["source_sha256"],
        "system_questions": 40, "original_direct_questions": len(groups[0]),
        "continuation_direct_questions": len(groups[1]), "direct_questions": 40,
        "direct_requests": 40, "total_requests": len(requests["requests"]),
        "peak_inflight": requests["peak_inflight"], "both_enable_thinking": True,
        "automatic_model_retry": False, "original_attempt_history_preserved": True,
        "continuation_files": files,
    }
    temporary = root / "pair_completion.tmp"
    temporary.write_text(json.dumps(receipt, sort_keys=True, indent=2) + "\n")
    temporary.replace(root / "pair_completion.json")
    print(json.dumps(receipt, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
