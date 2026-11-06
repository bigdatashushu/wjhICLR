"""Permit-before-call, shared role caps and conservative unknown-execution costs."""
from types import SimpleNamespace

import pytest

from budget import BudgetExhausted, RunBudget
from protocol import ContractError


def episode(environment="eb_navigation", clock=lambda: 0):
    run = RunBudget("run", [environment], clock=clock)
    return run, run.episode("episode", environment)


def test_every_role_reserves_before_call_and_failed_request_remains_charged():
    run, budget = episode()
    for stage in ("synthesis", "repair", "category_parse", "category_review", "dino", "sam", "moge"):
        with pytest.raises(RuntimeError):
            with budget.request(stage, decision_id="decision", view_id="current") as permit:
                assert budget.receipts[-1]["status"] == "started"
                assert run.inflight
                raise RuntimeError("synthetic-private-service-message")
    assert budget.counts["models"] == 4 and budget.counts["qwen"] == 3
    assert budget.counts["reviews"] == 1 and budget.counts["dino"] == 1
    assert all(row["status"] == "error" for row in budget.receipts)
    assert "synthetic-private" not in repr(budget.summary())
    assert run.inflight is False


def test_decision_and_episode_caps_include_repair_and_category_parse():
    _run, budget = episode()
    for stage in ("synthesis", "repair", "repair"):
        with budget.request(stage, decision_id="d"):
            pass
    with pytest.raises(BudgetExhausted, match="decision_synthesis"):
        with budget.request("repair", decision_id="d"):
            pytest.fail("called beyond decision cap")
    for _ in range(17):
        with budget.request("category_parse"):
            pass
    with pytest.raises(BudgetExhausted, match="qwen"):
        with budget.request("category_parse"):
            pytest.fail("called beyond Qwen cap")
    assert budget.counts["qwen"] == 20
    # Unused Qwen cannot be replenished by a different role.
    with budget.request("category_review"):
        pass
    assert budget.counts["qwen"] == 20 and budget.counts["models"] == 21


def test_governed_role_context_is_sanitized_and_usage_missing_is_explicit(tmp_path):
    from skill3d.synthesis.request_governor import RequestGovernor, RequestGovernorConfig
    governor = RequestGovernor(tmp_path / "qwen", RequestGovernorConfig(20, 1))
    run = RunBudget("run", ["eb_navigation"], qwen_governor=governor)
    budget = run.episode("episode", "eb_navigation")
    with budget.request("synthesis", decision_id="d") as permit:
        assert governor.summary()["reserved_requests"] == 1
        permit["usage"] = {"prompt_tokens": 5, "completion_tokens": 2, "total_tokens": 7,
                           "provider_url": "synthetic-private"}
    with budget.request("repair", decision_id="d"):
        pass
    assert budget.qwen_tokens == 7
    assert budget.receipts[0]["usage_missing"] is False
    assert budget.receipts[1]["usage_missing"] is True
    text = governor.state_path.read_text()
    assert "synthetic-private" not in text
    assert governor.summary()["categories"] == {"solver": 2}


def test_usage_soft_cap_stops_next_request_and_no_double_service_or_nested_call():
    _run, budget = episode()
    with budget.request("category_parse") as permit:
        permit["usage"] = {"total_tokens": 150001}
    with pytest.raises(BudgetExhausted, match="qwen_tokens_soft"):
        with budget.request("synthesis", decision_id="d"):
            pytest.fail("soft threshold did not stop next call")
    with budget.request("dino", view_id="v"):
        with pytest.raises(ContractError, match="nested"):
            with budget.request("sam", view_id="v"):
                pass
    with pytest.raises(ContractError, match="duplicate"):
        with budget.request("dino", view_id="v"):
            pass
    with pytest.raises(ContractError, match="unknown_request_stage"):
        with budget.request("reviser"):
            pass


def test_sandbox_unknown_keeps_entire_tranche_then_success_releases_only_unused():
    _run, budget = episode()
    first = budget.reserve_segment("d")
    assert first.limits["tool_calls"] == 12 and first.limits["reads"] == 3
    first.settle(SimpleNamespace(receipt={}, tool_calls=0, reader_calls=0))
    assert budget.counts["tools"] == 12 and budget.counts["reads"] == 3
    second = budget.reserve_segment("d")
    assert second.limits["tool_calls"] == 0 and second.limits["reads"] == 3
    second.settle(SimpleNamespace(receipt={"isolation_sealed": True}, tool_calls=0, reader_calls=1))
    assert budget.counts["tools"] == 12 and budget.counts["reads"] == 4
    third = budget.reserve_segment("d")
    assert third.limits["reads"] == 2
    with pytest.raises(BudgetExhausted, match="decision_segments"):
        budget.reserve_segment("d")
    with pytest.raises(ContractError, match="already_settled"):
        second.settle(SimpleNamespace())


@pytest.mark.parametrize("environment,steps,invalid,qwen", [
    ("eb_navigation", 20, 0, 20), ("eb_alfred", 30, 10, 40), ("eb_habitat", 30, 10, 40)])
def test_actual_caps_and_fixed_denominator(environment, steps, invalid, qwen):
    run, budget = episode(environment)
    assert budget.policy["qwen"] == qwen
    for _ in range(steps):
        budget.reserve_mutation("step")
    with pytest.raises(BudgetExhausted, match="steps"):
        budget.reserve_mutation("step")
    for _ in range(invalid):
        budget.reserve_mutation("invalid")
    with pytest.raises(BudgetExhausted, match="invalid"):
        budget.reserve_mutation("invalid")
    with pytest.raises(ContractError):
        run.episode("another", environment)
    assert run.summary()["planned_episodes"] == 1


def test_total_wall_time_before_request_and_action():
    now = [0]
    _run, budget = episode(clock=lambda: now[0])
    now[0] = 600
    with pytest.raises(BudgetExhausted, match="wall"):
        budget.reserve_mutation("step")
    with pytest.raises(BudgetExhausted, match="wall"):
        with budget.request("synthesis", decision_id="d"):
            pass
    assert budget.counts == {}
