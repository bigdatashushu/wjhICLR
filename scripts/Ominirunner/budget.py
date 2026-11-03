"""Read-only accounting across nonduplicated historical small-image ledgers."""
from __future__ import annotations

from collections import Counter
from pathlib import Path
import time

from .common import Config, digest, read

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
    for name in LEDGERS:
        path = base / name / "state.json"
        if not path.exists():
            continue
        state = read(path)
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
    raw_count = sum(r["count"] for r in rows)
    used = raw_count-duplicates
    mind_cap = read(base / "mindcube-perview-20261101-r01/manifest.json")["model_category_request_cap"]
    mind_used = sum(r.get("context", {}).get("campaign_id") == "mindcube-perview-20261101-r01"
                    for r in common["requests"])
    mmsi = next(r for r in rows if r["ledger"] == "mmsi-365-test-rerun-20261101/requests")
    retry_count = len(records.get("mmsi-skill-service-retry-20261102-r01/requests", []))
    reserve_mmsi = max(0, mmsi["policy"]["max_requests"] - mmsi["count"] - retry_count)
    reserve_mind = max(0, mind_cap-mind_used)
    cv = next((r for r in rows if r["ledger"] == "cv3d-box-review840-20261101-r01/budget"), None)
    reserve_cv = max(0, cv["policy"]["max_requests"]-cv["count"]) if cv else 0
    omni_cap = Config().trial_request_cap
    return {
        "at": time.time(), "global_cap": 20000, "used": used, "ledgers": rows,
        "raw_ledger_count": raw_count, "mirrored_request_copies_excluded": duplicates,
        "mirror_deduplication": mirrors, "mmsi_service_retry_count": retry_count,
        "mmsi_remaining_cap": reserve_mmsi, "mindcube_remaining_cap": reserve_mind,
        "omni_requested_cap": omni_cap,
        "upper_bound_with_mmsi_mindcube_and_omni_caps": used+reserve_mmsi+reserve_mind+omni_cap,
        "all_active_policy_upper_bound": used+reserve_mmsi+reserve_mind+reserve_cv+omni_cap,
        "cv3d_remaining_policy_cap": reserve_cv,
        "pending_cv3d_full_cap": 5880,
        "pending_cv3d_full_has_requests": any(
            r["count"] for r in rows if r["ledger"] == "cv3d-box-review840-20261101-r01/budget"),
        "duplicate_origin_and_mmsi_test_ledgers_excluded": True}
