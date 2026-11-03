"""Historical aggregate accounting, without importing another runner's config."""
from __future__ import annotations

from collections import Counter
from contextlib import ExitStack, contextmanager
import fcntl
from pathlib import Path

from skill3d.synthesis.request_governor import RequestBudgetExhausted
from .common import digest, read

# Authoritative histories from the 2026-11-03 audit. Archived BLINK origins,
# MMSI test_budget and Omni local governors are copies, not new requests.
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
MIRRORS = {
    "cv3d-mmsi-thinking-pilot-20261101/budget": "other-datasets-thinking-20261101",
    "cv3d-840-test-thinking-20261101-dispatch-fixed/budget": "cv3d-840-test-thinking-20261101",
}


def audit(base):
    base = Path(base)
    records = {name: read(base / name / "state.json") for name in LEDGERS}
    common = records["smallimage/requests"]
    if common["policy"]["max_requests"] != 20000:
        raise ValueError("global_budget_policy_changed")
    excluded, mirrors = 0, []
    for name, campaign in MIRRORS.items():
        def projected(row):
            return digest({k: row.get(k) for k in ("category", "context", "usage", "status")})
        available = Counter(projected(row) for row in common["requests"]
                            if row.get("context", {}).get("campaign_id") == campaign)
        count = 0
        for row in records[name]["requests"]:
            key = projected(row)
            if row.get("context", {}).get("campaign_id") == campaign and available[key]:
                available[key] -= 1
                count += 1
        excluded += count
        mirrors.append({"ledger": name, "matched_duplicate_count": count})
    raw = sum(len(state["requests"]) for state in records.values())
    return {"global_cap": 20000, "used": raw - excluded, "remaining": 20000 - raw + excluded,
            "raw_count": raw, "excluded_mirror_count": excluded, "mirrors": mirrors,
            "ledgers": [{"name": name, "count": len(state["requests"]),
                         "state_sha256": digest(state)}
                        for name, state in records.items()]}


class DualBudget:
    """All new calls enter the shared primary ledger and the run-local ledger."""
    def __init__(self, local, shared, base, stop_file, *, campaign, arm, sample_id, run_identity):
        self.local, self.shared = local, shared
        self.base, self.stop = Path(base), Path(stop_file)
        if shared.root.resolve() != (self.base / "smallimage/requests").resolve():
            raise ValueError("primary_budget_root_required")
        self.context = {"campaign_id": campaign, "variant_id": arm, "slot_key": sample_id,
                        "run_identity": run_identity}

    @contextmanager
    def request(self):
        # Serialize our aggregate admission with other MindcubeRunner workers.
        # The primary governor separately locks shared admissions for all users.
        with ExitStack() as stack:
            with (self.base / "mindcube-admission.lock").open("a+b") as lock:
                fcntl.flock(lock, fcntl.LOCK_EX)
                if self.stop.exists() or audit(self.base)["remaining"] <= 0:
                    raise RequestBudgetExhausted("aggregate_admission_stopped")
                local = stack.enter_context(self.local.request("solver", self.context))
                shared = stack.enter_context(self.shared.request("solver", self.context))
            # Admission is atomic for our workers; inference may now overlap.
            try:
                yield local
            finally:
                shared["usage"] = local["usage"]
