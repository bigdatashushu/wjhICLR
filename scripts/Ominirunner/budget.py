"""Read-only accounting across nonduplicated historical small-image ledgers."""
from __future__ import annotations

from collections import Counter
from pathlib import Path
import time

from .common import Config, digest, read, sha

# origin copies and MMSI mirrored test_budget are deliberately excluded.
LEDGERS = (
    "smallimage/requests", "blink-formal-20261031-14cf22c/budget",
    "blink-recovery-20261031-1ee33f28c07e/budget",
    "blink_direct_qwen_20261031/requests", "blink_direct_qwen_desktop_20261031_r02/requests",
    "cv3d-mmsi-thinking-pilot-20261101/budget",
    "cv3d-840-test-thinking-20261101/budget",
    "cv3d-840-test-thinking-20261101-dispatch-fixed/budget",
    "mmsi-continuation-20261101/requests", "mmsi-continuation-20261101/train_budget",
    "mmsi-365-test-multiagent-20261101/requests",
    "mmsi-365-test-rerun-20261101/requests",
    "mmsi-skill-service-retry-20261102-r01/requests",
    "cv3d-simple-paired-20261101-r02/budget",
    "cv3d-program-skill-paired-20261101-r01/budget",
    "cv3d-program-skill-paired-20261101-r02/budget",
    "cv3d-context-v2-trial-20261101-r01/budget",
    "cv3d-box-review10-20261101-r01/budget",
    "cv3d-box-review840-20261101-r01/budget",
)

# These two archived campaigns wrapped the same call in local and shared
# governors. Exclude a local copy only when the full numeric record matches.
MIRRORS = {
    "cv3d-mmsi-thinking-pilot-20261101/budget": "other-datasets-thinking-20261101",
    "cv3d-840-test-thinking-20261101-dispatch-fixed/budget":
        "cv3d-840-test-thinking-20261101",
}

RELEASE_PATH = "smallimage/mmsi_completed_reservation_release.json"
MMSI_LEDGERS = ("mmsi-365-test-rerun-20261101/requests",
                "mmsi-skill-service-retry-20261102-r01/requests")
MMSI_COMPLETIONS = (
    "mmsi-365-test-rerun-20261101/completion.json",
    "mmsi-365-test-rerun-20261101/exit.json",
    "mmsi-skill-service-retry-20261102-r01/completion.json",
    "mmsi-skill-service-retry-20261102-r01/final_acceptance.json",
    "mmsi-skill-service-retry-20261102-r01/exit.json",
)
CV_LEDGER = "cv3d-box-review840-20261101-r01/budget"
CV_COMPLETIONS = ("cv3d-box-review840-20261101-r01/completion.json",
                  "cv3d-box-review840-20261101-r01/manifest.json")
CV_RELEASE_PATH = "smallimage/cv3d_completed_reservation_release.json"
EXTERNAL_PATH = "smallimage/omni_external_review_accounting.json"


UNLIMITED_POLICY = {"global_limit": None, "authorization": "explicit_user",
                    "instruction": "别管这个预算，没有预算上限"}


def admission(accounting, requested_cap, policy=None):
    """Reserve the complete new run once, including every outstanding policy."""
    if type(requested_cap) is not int or requested_cap < 0:
        raise ValueError("invalid_requested_cap")
    upper = (accounting["used"] + accounting["mmsi_remaining_cap"]
             + accounting["mindcube_remaining_cap"] + accounting["cv3d_remaining_policy_cap"]
             + requested_cap)
    if policy is not None and policy != UNLIMITED_POLICY:
        raise ValueError("aggregate_budget_policy_invalid")
    if policy is None and upper > accounting["global_cap"]:
        raise ValueError("aggregate_reserved_budget_insufficient")
    return {**accounting, "requested_cap": requested_cap, "admission_upper_bound": upper,
            "aggregate_budget_policy": policy}


def completed_cv_release(base, records, policy):
    path = Path(base) / CV_RELEASE_PATH
    if not path.exists():
        return None
    receipt = read(path)
    rows = records.get(CV_LEDGER, [])
    if (receipt.get("schema") != "omni-cv3d-completed-reservation-release/1"
            or receipt.get("status") != "released" or receipt.get("authorization") != "explicit_user"
            or receipt.get("active_budget_slots") != 0 or receipt.get("operator_lock_free") is not True
            or receipt.get("original_process_absent") is not True
            or receipt.get("model_requests") != 0 or receipt.get("reference_answers_returned") is not False
            or receipt.get("policy") != policy
            or set(receipt.get("completion_files", {})) != set(CV_COMPLETIONS)
            or receipt.get("ledger_snapshot") != {"count": len(rows), "requests_sha256": digest(rows)}
            or not rows or any(r["status"] != "completed" for r in rows)
            or receipt.get("released_unused_requests") != max(0, policy["max_requests"] - len(rows))):
        raise ValueError("cv_completed_reservation_release_invalid")
    for name in CV_COMPLETIONS:
        if sha(Path(base) / name) != receipt["completion_files"][name]:
            raise ValueError("cv_completed_reservation_evidence_changed")
    return {"receipt_sha256": sha(path),
            "released_unused_requests": receipt["released_unused_requests"],
            "historical_charges_preserved": True}


def external_ledgers(base):
    """Immutable copies of host-only review charges; never invent historical calls."""
    path = Path(base) / EXTERNAL_PATH
    if not path.exists():
        return {}, None
    receipt = read(path)
    if (receipt.get("schema") != "omni-external-review-accounting/1"
            or receipt.get("model_requests") != 0 or not receipt.get("ledgers")):
        raise ValueError("external_review_accounting_invalid")
    states = {}
    for name, expected in receipt["ledgers"].items():
        relative = Path(name)
        if (relative.is_absolute() or ".." in relative.parts
                or relative.parent != Path("smallimage/omni_external_review")):
            raise ValueError("external_review_path_invalid")
        state_path = Path(base) / relative / "state.json"
        state = read(state_path)
        if (sha(state_path) != expected["file_sha256"]
                or len(state["requests"]) != expected["count"]
                or any(r["category"] != "gate" or r["status"] != "completed"
                       for r in state["requests"])):
            raise ValueError("external_review_snapshot_changed")
        states[name] = state
    return states, {"receipt_sha256": sha(path), "ledgers": len(states),
                    "historical_snapshot_copies": True}


def completed_mmsi_release(base, records):
    """Validate an explicit host release; never read old answer-bearing bodies."""
    path = Path(base) / RELEASE_PATH
    if not path.exists():
        return None
    receipt = read(path)
    if (receipt.get("schema") != "omni-completed-reservation-release/1"
            or receipt.get("status") != "released" or receipt.get("active_budget_slots") != 0
            or receipt.get("model_requests") != 0 or receipt.get("reference_answers_returned") is not False
            or set(receipt.get("ledger_snapshots", {})) != set(MMSI_LEDGERS)
            or set(receipt.get("completion_files", {})) != set(MMSI_COMPLETIONS)):
        raise ValueError("completed_reservation_release_invalid")
    for name in MMSI_LEDGERS:
        rows = records.get(name)
        saved = receipt["ledger_snapshots"][name]
        if rows is None or saved != {"count": len(rows), "requests_sha256": digest(rows)}:
            raise ValueError("completed_reservation_ledger_changed")
    for name in MMSI_COMPLETIONS:
        if sha(Path(base) / name) != receipt["completion_files"][name]:
            raise ValueError("completed_reservation_evidence_changed")
    return {"receipt_sha256": sha(path), "released_unused_requests": receipt["released_unused_requests"],
            "prior_completion_hashes_verified": True, "historical_charges_preserved": True}


def mirror_receipt(name, local, shared):
    campaign = MIRRORS[name]
    common = [r for r in shared if r.get("context", {}).get("campaign_id") == campaign]
    def projected(r):
        return digest({k: r.get(k) for k in ("category", "context", "usage", "status")})
    available = Counter(projected(r) for r in common)
    excluded = []
    for row in local:
        key = projected(row)
        if (row.get("context", {}).get("campaign_id") == campaign
                and available[key] > 0):
            available[key] -= 1
            excluded.append(key)
    return {
        "ledger": name, "shared_ledger": "smallimage/requests", "campaign": campaign,
        "local_count": len(local), "shared_count": len(common),
        "matched_duplicate_count": len(excluded),
        "unmatched_local_count": len(local)-len(excluded),
        "projection_fields": ["category", "context", "usage", "status"],
        "matched_projection_multiset_sha256": digest(sorted(excluded)),
        "complete_mirror": len(local) == len(common) == len(excluded),
    }


def audit(base):
    base = Path(base)
    rows = []
    records = {}
    common = None
    external, external_receipt = external_ledgers(base)
    for name in (*LEDGERS, *external):
        path = base / name / "state.json"
        if not path.exists():
            continue
        state = external[name] if name in external else read(path)
        requests = state["requests"]
        records[name] = requests
        rows.append({
            "ledger": name, "count": len(requests), "policy": state["policy"],
            "statuses": dict(Counter(r["status"] for r in requests)),
            "request_prefix_sha256": digest(requests),
            # started rows may later finish; admission identity remains immutable.
            "admission_prefix_sha256": digest([
                {k: r[k] for k in ("request_id", "category", "context", "started_at")}
                for r in requests])})
        if name == "smallimage/requests":
            common = state
    if common is None:
        raise ValueError("shared_ledger_missing")
    mirrors = [mirror_receipt(name, records[name], common["requests"])
               for name in MIRRORS if name in records]
    duplicates = sum(r["matched_duplicate_count"] for r in mirrors)
    # Compare immutable admissions, rather than mutable usage/finished fields.
    admitted = set()
    external_duplicates = 0
    for name, requests in records.items():
        for row in requests:
            key = digest({k: row[k] for k in ("request_id", "category", "context", "started_at")})
            if name in external and key in admitted:
                external_duplicates += 1
            admitted.add(key)
    duplicates += external_duplicates
    raw_count = sum(r["count"] for r in rows)
    used = raw_count-duplicates
    mind_cap = read(base / "mindcube-perview-20261101-r01/manifest.json")["model_category_request_cap"]
    mind_used = sum(r.get("context", {}).get("campaign_id") == "mindcube-perview-20261101-r01"
                    for r in common["requests"])
    mmsi = next(r for r in rows if r["ledger"] == "mmsi-365-test-rerun-20261101/requests")
    retry_count = len(records.get("mmsi-skill-service-retry-20261102-r01/requests", []))
    reserve_mmsi = max(0, mmsi["policy"]["max_requests"] - mmsi["count"] - retry_count)
    release = completed_mmsi_release(base, records)
    if release is not None:
        if release["released_unused_requests"] != reserve_mmsi:
            raise ValueError("completed_reservation_amount_changed")
        reserve_mmsi = 0
    reserve_mind = max(0, mind_cap-mind_used)
    cv = next((r for r in rows if r["ledger"] == "cv3d-box-review840-20261101-r01/budget"), None)
    reserve_cv = max(0, cv["policy"]["max_requests"]-cv["count"]) if cv else 0
    cv_release = completed_cv_release(base, records, cv["policy"]) if cv else None
    if cv_release is not None:
        reserve_cv = 0
    omni_cap = Config().trial_request_cap
    return {
        "at": time.time(), "global_cap": 20000, "used": used, "ledgers": rows,
        "raw_ledger_count": raw_count, "mirrored_request_copies_excluded": duplicates,
        "shared_campaign_counts": dict(Counter(
            r.get("context", {}).get("campaign_id", "") for r in common["requests"])),
        "mirror_deduplication": mirrors, "mmsi_service_retry_count": retry_count,
        "external_review_accounting": external_receipt,
        "external_review_duplicate_copies_excluded": external_duplicates,
        "completed_mmsi_reservation_release": release,
        "completed_cv3d_reservation_release": cv_release,
        "mmsi_remaining_cap": reserve_mmsi, "mindcube_remaining_cap": reserve_mind,
        "omni_requested_cap": omni_cap,
        "upper_bound_with_mmsi_mindcube_and_omni_caps": used+reserve_mmsi+reserve_mind+omni_cap,
        "all_active_policy_upper_bound": used+reserve_mmsi+reserve_mind+reserve_cv+omni_cap,
        "cv3d_remaining_policy_cap": reserve_cv,
        "pending_cv3d_full_cap": 5880,
        "pending_cv3d_full_has_requests": any(
            r["count"] for r in rows if r["ledger"] == "cv3d-box-review840-20261101-r01/budget"),
        "duplicate_origin_and_mmsi_test_ledgers_excluded": True}
