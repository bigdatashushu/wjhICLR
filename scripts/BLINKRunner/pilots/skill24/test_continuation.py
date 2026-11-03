"""Offline checks for the new continuation boundary."""
from copy import deepcopy
import pytest
from continuation import verify_budget_prefix


def baseline():
    return {"policy": {"max_requests": 1341, "max_inflight": 4},
            "requests": [{"request_id": 0, "status": "completed"},
                         {"request_id": 1, "status": "error"}]}


def test_carried_failure_retained_and_new_completed_request_accepted():
    old = baseline()
    current = deepcopy(old)
    current["requests"].append({"request_id": 2, "status": "completed"})
    verify_budget_prefix(current, old)
    assert old["requests"][1]["status"] == "error"


def test_changed_prefix_or_erased_failure_rejected():
    old = baseline()
    changed = deepcopy(old)
    changed["requests"][1]["status"] = "completed"
    with pytest.raises(ValueError):
        verify_budget_prefix(changed, old)


def test_new_error_or_active_request_stops_without_acceptance():
    old = baseline()
    for state in ("error", "started"):
        current = deepcopy(old)
        current["requests"].append({"request_id": 2, "status": state})
        with pytest.raises(RuntimeError):
            verify_budget_prefix(current, old)
