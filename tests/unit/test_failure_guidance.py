"""Protective errors retain their meaning while exposing authorized safe actions."""

import pytest

from skill3d.online.recovery import failed_call_feedback
from skill3d.sandbox.kernel import RestrictedNamespaceKernel
from skill3d.tools import REGISTRY
from skill3d.tools.failure_guidance import failure_guidance
from skill3d.schemas.tool import FailureCause
from test_v9_active_vision import _handle


@pytest.mark.parametrize("code,expected", [
    ('inspect_frames([0, 1], [[0, 0, 20, 20]])', "crop_frame_count_mismatch"),
    ('inspect_frames([0], [[2000, 2000, 2020, 2020]])', "invalid_crop_box"),
    ('inspect_frames([1000])', "frame_slot_out_of_range"),
    ('inspect_frames([0], [[0, 0, float("inf"), 20]])', "invalid_crop_box"),
])
def test_crop_rejection_has_argument_reason_scope_and_available_actions(code, expected):
    handle = _handle(n_frames=32)
    kernel = RestrictedNamespaceKernel(REGISTRY, handle)
    cell = kernel.run_cell(code)
    assert cell.error_code == "tool_contract"  # Historical cell bucket stays unchanged
    result = kernel.cell_results[-1]
    assert result.error_code == "domain_value"
    assert result.status == "failed" and result.failure_cause.code == expected
    assert result.failure_guidance["reason_code"] == expected
    assert result.failure_guidance["scope"] == "local_call"
    assert "inspect_frames" in result.failure_guidance["safe_available_actions"]
    assert "object_centroid" not in result.failure_guidance["safe_available_actions"]
    assert result.failure_guidance["retry_requires_change"]
    assert "available_next_steps" in failed_call_feedback([result])


def test_collection_guidance_does_not_invent_a_shared_invalidation_cause():
    handle = _handle(n_frames=32)
    cause = FailureCause(code="object_binding_unavailable", scope="local_call")
    guide = failure_guidance(REGISTRY, handle, name="count_objects", cause=cause)
    assert guide["scope"] == "local_collection"
    assert guide["missing_preconditions"] == ["requested_collection_coverage_and_usable_bindings"]
    assert cause.scope == "local_call" and not cause.shared_premises
    guide = failure_guidance(REGISTRY, handle, name="object_visible_frames",
        cause=FailureCause(code="object_mask_unavailable", scope="local_call"))
    assert guide["scope"] == "local_object"
    assert guide["missing_preconditions"] == ["readable_consistent_object_mask"]
    assert failure_guidance(REGISTRY, handle, name="inspect_frames", tools_enabled=False)[
        "safe_available_actions"] == []
