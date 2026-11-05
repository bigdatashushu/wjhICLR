"""Reuse only completed direct answers, preserving original receipts and traces."""
from __future__ import annotations

from pathlib import Path
import shutil

from .common import Config, IDS, digest, read, selected_ids, sha, write
from .data import pair_identity


def inspect(source, item, expected_messages):
    source = Path(source)
    directory = source / "episodes" / (item["sample_id"] + "-direct")
    result_path = directory / "result.json"
    if not result_path.exists():
        return None
    result, frozen = read(result_path), read(source / "frozen.json")
    if result["status"] != "answered":
        return None
    identity = digest(frozen)
    rounds = result["rounds"]
    if (result["arm"] != "direct" or result["sample_id"] != item["sample_id"]
            or result["run_identity"] != identity
            or result["pair_identity"] != pair_identity(item)
            or result["receipt_sha256"] != digest({
                k: v for k, v in result.items() if k != "receipt_sha256"})
            or result["reference_answer_visible"] or result["real_labels_available"]
            or result["automatic_retry"] or result["request_count"] != 1 or len(rounds) != 1
            or result["tool_counts"]["calls"] or rounds[0].get("tool_receipts")
            or rounds[0]["parameters"] != Config().kwargs()
            or rounds[0]["response_identity"]["model"] != Config().model
            or not rounds[0]["response_identity"]["model_reported"]):
        raise ValueError("historical_direct_receipt_changed")
    payload = read(directory / "00.messages.json")
    if payload != expected_messages or digest(payload) != rounds[0]["request"]["messages_sha256"]:
        raise ValueError("historical_direct_prompt_changed")
    pair = next(p for p in frozen["pairs"] if p["sample_id"] == item["sample_id"])
    if (pair["pair_identity"] != pair_identity(item)
            or rounds[0]["request"] != pair["initial_receipts"]["direct"]
            or read(source / "preflight.json")["run_identity"] != identity):
        raise ValueError("historical_direct_preflight_changed")
    local = read(source / "requests/state.json")["requests"]
    shared = read(Path(frozen["shared_budget_root"]) / "state.json")["requests"]
    charges = []
    for ledger in (local, shared):
        rows = [r for r in ledger if r["context"].get("run_identity") == identity
                and r["context"].get("slot_key") == item["sample_id"]
                and r["context"].get("variant_id") == "direct"]
        if (len(rows) != 1 or rows[0]["status"] != "completed"
                or rows[0].get("usage") != result["usage"]):
            raise ValueError("historical_direct_charge_changed")
        charges.append(digest(rows[0]))
    files = {p.name: sha(p) for p in sorted(directory.iterdir()) if p.is_file()
             and p.suffix in (".json", ".png", ".jpg", ".jpeg")}
    return {"sample_id": item["sample_id"], "source_root": str(source),
            "source_run_identity": identity, "source_frozen_sha256": sha(source / "frozen.json"),
            "result_sha256": sha(result_path), "receipt_sha256": result["receipt_sha256"],
            "messages_sha256": digest(payload), "files": files,
            "historical_charge_sha256": charges, "new_requests": 0}


def validate(entry, item, expected_messages):
    actual = inspect(entry["source_root"], item, expected_messages)
    if actual != entry:
        raise ValueError("frozen_reuse_provenance_changed")
    return actual


def materialize(entry, root):
    target = Path(root) / "episodes" / (entry["sample_id"] + "-direct")
    target.mkdir()
    source = Path(entry["source_root"]) / "episodes" / target.name
    for name, expected in entry["files"].items():
        if sha(source / name) != expected:
            raise ValueError("reuse_trace_changed")
        shutil.copyfile(source / name, target / name)
    write(target / "reuse.json", entry, exclusive=True)
    return read(target / "result.json")


def request_cap(reused, sample_ids=IDS, measurement_reviewer=None):
    ids = selected_ids({"sample_ids": sample_ids})
    if not set(reused) <= set(ids):
        raise ValueError("reuse_outside_selection")
    config = Config()
    extra = 0
    if measurement_reviewer is not None:
        from .jev import Policy
        extra = len(ids) * Policy.parse(measurement_reviewer).max_requests
    return (len(ids) * (config.request_limit("direct") + config.request_limit("skill"))
            - len(reused) + extra)


def accounted_requests(results, reused):
    return sum(r["request_count"] for r in results
               if not (r["arm"] == "direct" and r["sample_id"] in reused))
