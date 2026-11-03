"""Pure append-aware audit for an independently shared request budget."""


def audit_shared_state(current, legacy, retained, *, mmsi_operations, mmsi_run_identity,
                       mmsi_requests, mmsi_request_cap=1900):
    expected = {"max_inflight": 4, "max_requests": 20000, "requests_per_second": 0.0}
    assert current["policy"] == legacy["policy"] == expected
    rows = current["requests"]
    assert rows[:len(legacy["requests"])] == legacy["requests"], "legacy_budget_rows_changed"
    if retained is not None:
        assert retained["policy"] == expected
        assert all(r["status"] in {"completed", "error"} for r in retained["requests"])
        assert rows[:len(retained["requests"])] == retained["requests"], "retained_budget_rows_changed"
    assert [r["request_id"] for r in rows] == list(range(len(rows))), "budget_request_ids_changed"
    assert all(r["status"] in {"started", "completed", "error"} for r in rows)
    assert current["peak_inflight"] <= 4
    assert sum(r["status"] == "started" for r in rows) <= 4
    assert 0 <= mmsi_requests <= mmsi_request_cap == 1900
    assert len(rows) + mmsi_requests <= 20000, "combined_budget_exceeded"
    # Maintain enough headroom for the entire independent MMSI budget.
    assert len(rows) + mmsi_request_cap <= 20000, "mmsi_budget_headroom_exhausted"
    for row in rows[len(legacy["requests"]):]:
        context = row.get("context", {})
        assert context.get("variant_id") not in mmsi_operations, "mmsi_request_in_other_ledger"
        assert context.get("run_identity") != mmsi_run_identity, "mmsi_identity_in_other_ledger"
        assert isinstance(context.get("campaign_id"), str) and context["campaign_id"]
        assert row["category"] in {"solver", "reviser", "gate"}
    return {
        "legacy_prefix_requests": len(legacy["requests"]),
        "retained_prefix_requests": len(retained["requests"]) if retained is not None else 0,
        "original_requests": len(rows),
        "other_requests_added_since_initial_freeze": len(rows) - len(legacy["requests"]),
        "other_requests_added_since_resume": len(rows) - len(retained["requests"]) if retained is not None else 0,
        "other_active_requests": sum(r["status"] == "started" for r in rows),
        "original_plus_mmsi_requests": len(rows) + mmsi_requests,
        "combined_cap": 20000, "mmsi_cap": mmsi_request_cap,
        "historical_rows_unchanged": True,
    }
