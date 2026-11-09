from concurrent.futures import ThreadPoolExecutor
import threading

import pytest

from MindcubeRunner.budget import audit, DualBudget
from MindcubeRunner.common import digest, sha, write
from MindcubeRunner.run import bounded_pairs
from skill3d.synthesis.request_governor import RequestGovernor, RequestGovernorConfig


def campaign(tmp_path):
    root = tmp_path / "run"
    base = root / "campaign-accounting"
    base.mkdir(parents=True)
    ids = [f"q{index:04d}" for index in range(700)]
    write(root / "selection.json", {"selection": ids})
    policy = {
        "schema": "mindcube-explicit-campaign-budget/1",
        "questions": 700, "arms": ["skill", "direct"],
        "request_limits": {"skill": 6, "direct": 1},
        "max_requests": 4900, "max_inflight": 16, "selection_sha256": digest(ids),
    }
    write(base / "authorization.json", policy)
    shared = RequestGovernor(base / "smallimage/requests", RequestGovernorConfig(4900, 16))
    local = RequestGovernor(root / "results/requests", RequestGovernorConfig(4900, 16))
    frozen = {
        "schema": "mindcube-tinybench-preflight/1",
        "items": [{"sample_id": sample} for sample in ids],
        "arms": ["skill", "direct"],
        "settings": {"budget_authorization_sha256": sha(base / "authorization.json")},
    }
    frozen["run_identity"] = digest(frozen)
    write(root / "results/preflight.json", frozen)
    return root, base, local, shared, frozen


def test_campaign_separate_budget_and_sanitized_matching_ledgers(tmp_path):
    root, base, local, shared, frozen = campaign(tmp_path)
    assert audit(base)["remaining"] == 4900
    governor = DualBudget(local, shared, base, root / "STOP.json",
                          campaign="campaign", arm="skill", sample_id="q0000",
                          run_identity=frozen["run_identity"])
    with governor.request() as receipt:
        receipt["usage"] = {"total_tokens": 37}
    assert audit(base)["remaining"] == 4899
    assert local.summary()["usage"] == shared.summary()["usage"] == {"total_tokens": 37}
    assert not (tmp_path / "smallimage/requests").exists()


def test_campaign_rejects_changed_selection_and_policy(tmp_path):
    root, base, local, shared, frozen = campaign(tmp_path)
    write(root / "selection.json", {"selection": ["q0000"]})
    with pytest.raises(ValueError, match="authorization_changed"):
        audit(base)


@pytest.mark.parametrize("arm,sample,identity", [
    ("gate", "q0000", None),
    ("skill", "foreign-question", None),
    ("direct", "q0000", "foreign-run"),
])
def test_campaign_rejects_out_of_scope_admission(tmp_path, arm, sample, identity):
    root, base, local, shared, frozen = campaign(tmp_path)
    with pytest.raises(ValueError, match="outside_authorized_campaign"):
        DualBudget(local, shared, base, root / "STOP.json",
                   campaign="campaign", arm=arm, sample_id=sample,
                   run_identity=identity or frozen["run_identity"])
    assert not shared.summary()["reserved_requests"]


def test_bounded_pairs_overlap_without_duplicate_dispatch():
    barrier = threading.Barrier(2)
    lock = threading.Lock()
    seen = []
    def execute_pair(index, item):
        with lock:
            seen.append(index)
        if index < 2:
            barrier.wait(timeout=5)
        return [{"index": index, "item": item}]
    rows = bounded_pairs(list(range(7)), 2, execute_pair, lambda: False)
    assert sorted(seen) == list(range(7))
    assert sorted(row["index"] for row in rows) == list(range(7))


def test_bounded_pairs_stop_then_drain_only_admitted_pairs():
    stopped = threading.Event()
    barrier = threading.Barrier(2)
    def execute_pair(index, item):
        barrier.wait(timeout=5)
        stopped.set()
        return [{"index": index}]
    rows = bounded_pairs(list(range(100)), 2, execute_pair, stopped.is_set)
    assert sorted(row["index"] for row in rows) == [0, 1]
