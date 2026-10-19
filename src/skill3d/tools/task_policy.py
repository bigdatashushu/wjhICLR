"""Fixed task-level model interfaces; technical authorization remains separate."""

from types import MappingProxyType

from skill3d.routing.task_classifier import UnknownQuestionTypeError, canonical_task

TOOL_TASK_POLICY_VERSION = "tool-task-policy-v11.8"
COMMON_TOOLS = frozenset({
    "inspect_frames", "detect_objects", "list_objects", "object_visible_frames",
})
SPATIAL_HELPERS = frozenset({"object_centroid", "reproject"})
_SPATIAL = COMMON_TOOLS | SPATIAL_HELPERS
TASK_TOOLS = MappingProxyType({
    "object_counting": _SPATIAL | {"count_objects"},
    "object_abs_distance": _SPATIAL | {"object_surface_distance"},
    "object_rel_distance": _SPATIAL | {
        "relative_distance_rank", "object_surface_distance"},
    "object_size_estimation": _SPATIAL | {"object_3d_extent"},
    "room_size_estimation": _SPATIAL | {
        "plane_fit_room_size", "connectivity_graph"},
    "object_rel_direction": _SPATIAL | {"relative_direction_of"},
    "obj_appearance_order": COMMON_TOOLS,
    "route_planning": _SPATIAL | {
        "connectivity_graph", "relative_direction_of"},
})
MODEL_TOOLS = frozenset().union(*TASK_TOOLS.values())


def task_allows(tool: str, question_type: str) -> bool:
    """Unknown tasks retain only common observation interfaces."""
    try:
        task = canonical_task(question_type)
    except UnknownQuestionTypeError:
        return tool in COMMON_TOOLS
    return tool in TASK_TOOLS[task]
