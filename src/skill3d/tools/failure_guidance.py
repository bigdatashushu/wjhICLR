"""Typed denial guidance from actual failure facts and the authorization catalog."""

from __future__ import annotations

GUIDANCE_VERSION = "tool-failure-guidance/1.0"
_LOCAL_REQUIREMENTS = {
    "object_mask_unavailable": ["readable_consistent_object_mask"],
    "object_binding_unavailable": ["verified_object_binding"],
    "semantic_label_mask_conflict": ["nonconflicting_object_binding"],
    "invalid_tool_arguments": ["valid_argument_schema"],
    "invalid_crop_box": ["finite_pixel_box_with_minimum_side"],
    "crop_frame_count_mismatch": ["one_box_per_requested_frame"],
    "frame_slot_out_of_range": ["frame_slot_in_frozen_frameset"],
}


def failure_guidance(registry, scene, *, name, cause=None, missing=(),
                     tools_enabled=True) -> dict:
    """Safe actions remain subject to arguments and execution-time validation."""
    from skill3d.tools.contract import authorize_tool_call

    code = str(getattr(cause, "code", "") or "unlocated_tool_failure")
    scope = str(getattr(cause, "scope", "") or "unknown")
    if name in {"count_objects", "relative_distance_rank"} and code == "object_binding_unavailable":
        scope = "local_collection"
        requirements = ["requested_collection_coverage_and_usable_bindings"]
    else:
        requirements = list(_LOCAL_REQUIREMENTS.get(code, []))
        if code == "object_mask_unavailable":
            # Display scope is descriptive; it does not infer an invalidation dependency.
            scope = "local_object"
    actions = []
    for candidate in ("inspect_frames", "list_objects", "detect_objects"):
        if candidate not in registry.names():
            continue
        spec = registry.spec(candidate)
        decision = authorize_tool_call(
            candidate, spec, scope=scene.question_tool_scope,
            profile=scene.evidence_profile, available_artifacts=scene.available_artifacts,
            gate=scene.metric_evidence_gate_result, gate_passed=scene.metric_gate_passed,
            supported_metric_tasks=spec.supported_metric_tasks,
            allowed_metric_tasks=scene.allowed_metric_tasks, question_type=scene.question_type,
            local_facts=scene.local_facts, tools_enabled=tools_enabled)
        if decision.allowed:
            actions.append(candidate)
    return {
        "schema_version": GUIDANCE_VERSION, "reason_code": code, "scope": scope,
        "missing_preconditions": sorted(set(requirements) | set(map(str, missing))),
        "safe_available_actions": actions,
        "action_availability": "catalog authorization; arguments and service availability still checked",
        "retry_requires_change": code not in {"service_error", "detector_service_fault", "service_unavailable"},
    }
