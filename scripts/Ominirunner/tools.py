"""Immutable current-question candidate view and unbounded call accounting."""
from __future__ import annotations

import re
from types import SimpleNamespace

from skill3d.small_image.tools import build_registry as shared_registry
from skill3d.tools.contract import DomainValueError
from .sampling import VERSION as SAMPLING_VERSION, EDGE_NOTE, point_depth, pair_point_distance
from .coordinates import resolve, intent
from .instances import candidate_facts
from .recovery import input_error

VERSION = "omni-question-scoped-tools/6"
ALLOWED = {"view_candidates", "view_depth_summary", "point_depth", "pair_point_distance"}
MEASUREMENTS = {"point_depth", "pair_point_distance"}
SYNONYMS = {
    "sofa": ("couch",), "couch": ("sofa",), "armchair": ("chair",),
    "chair": ("armchair",), "tv": ("television",), "television": ("tv",),
    "cabinet": ("cupboard",), "cupboard": ("cabinet",),
}


def empty_counts():
    return {"calls": 0, "measurements": 0, "attempted": 0, "rejected": 0,
            "attempted_measurements": 0, "rejected_measurements": 0}


def phase_for(config, round_index):
    return "final" if round_index == config.max_program_rounds else "program"


def words(text):
    result = []
    for token in re.findall(r"[a-z0-9]+", text.lower()):
        if token.endswith("ies") and len(token) > 4:
            token = token[:-3] + "y"
        elif token.endswith("s") and not token.endswith(("ss", "us")) and len(token) > 3:
            token = token[:-1]
        result.append(token)
    return result


def relevant(category, question):
    phrase = " " + " ".join(words(question)) + " "
    variants = (category, *SYNONYMS.get(category.lower(), ()))
    return any(" " + " ".join(words(v)) + " " in phrase for v in variants)


def build_registry(cache, question, config, measurement_reviewer=None):
    if measurement_reviewer is not None:
        from .jev import Policy
        Policy.parse(measurement_reviewer)
    candidates = [dict(c) for c in cache.candidates if relevant(c["category"], question.question)]
    view = SimpleNamespace(
        frames=cache.frames, candidates=candidates, geometries=cache.geometries,
        provenance={"scope": VERSION, "input_sha256": question.input_sha256})
    registry, scene = shared_registry(view, question)
    scene.question_type = "omni_native_scalar"
    scene.delivered_images = {}
    scene.delivered_measurement_ids = set()
    scene.delivered_reselect_ids = set()

    def coordinate_point(handle, view_index, x, y, image_id=None,
                         coordinate_space="original_pixel", targets=None, replaces=None,
                         target=None, quantity=None):
        targets = sampling_targets(targets, target, quantity, 1)
        validate_sampling_inputs(handle, view_index, targets, 1, replaces)
        xy, transform = resolve([x, y], image_id, coordinate_space, handle.delivered_images)
        data = point_depth(handle, view_index, *xy)
        return annotate(handle, data, [transform], targets, 1, replaces, quantity)

    def coordinate_pair(handle, view_index, a_xy, b_xy, image_id=None,
                        coordinate_space="original_pixel", targets=None, replaces=None,
                        target=None, quantity=None):
        targets = sampling_targets(targets, target, quantity, 2)
        validate_sampling_inputs(handle, view_index, targets, 2, replaces)
        a, at = resolve(a_xy, image_id, coordinate_space, handle.delivered_images, "a_xy")
        b, bt = resolve(b_xy, image_id, coordinate_space, handle.delivered_images, "b_xy")
        data = pair_point_distance(handle, view_index, a, b)
        return annotate(handle, data, [at, bt], targets, 2, replaces, quantity)

    def sampling_targets(targets, target, quantity, count):
        if measurement_reviewer is None:
            if target is not None or quantity is not None:
                raise input_error("omni_sampling", "arguments", "targets in legacy mode")
            return targets
        from .jev import sampling_intent
        if targets is not None:
            raise input_error("omni_sampling", "targets", "target and quantity in Jev mode")
        try:
            return sampling_intent(target, quantity, count)
        except ValueError:
            raise input_error("omni_sampling", "target/quantity", "a valid object and measurement quantity") from None

    def validate_sampling_inputs(handle, view_index, targets, count, replaces):
        if type(view_index) is not int or not 0 <= view_index < len(handle.frames):
            raise input_error("omni_sampling", "view_index", [0], view_index)
        intent(targets, count)
        replaceable = handle.delivered_measurement_ids
        if measurement_reviewer is not None:
            replaceable = replaceable | handle.delivered_reselect_ids
        if replaces is not None and (not isinstance(replaces, str) or replaces not in replaceable):
            raise input_error("omni_sampling", "replaces", "a currently delivered measurement ID", replaces)

    def annotate(handle, data, transforms, targets, count, replaces, quantity):
        data.update(coordinate_transforms=transforms, intended_targets=intent(targets, count),
                    replaces_result_id=replaces)
        if measurement_reviewer is not None:
            data["jev_quantity"] = quantity
        return data

    def candidates_view(handle, view_index=-1, category=""):
        if type(view_index) is not int or not -1 <= view_index < len(handle.frames):
            raise input_error("view_candidates", "view_index", [-1, 0], view_index)
        return candidate_facts(handle, view_index, category)

    candidate_tool = registry.get("view_candidates")
    candidate_tool.fn = candidates_view
    registry.public_fields["view_candidates"] = frozenset({
        "candidates", "count", "coverage", "exhaustive", "overlap_groups",
        "lineage_version", "instance_rule"})
    for name, fn, description, fields in (
        ("point_depth", coordinate_point,
         "Raw camera Z at the rounded/clamped pixel and each valid 5x5 pixel, null if invalid. "
         "depth_samples has origin_xy and row-major values; no averaging. " + EDGE_NOTE,
         {"view_index", "pixel_xy", "depth", "depth_samples", "n_valid", "unit"}),
        ("pair_point_distance", coordinate_pair,
         "Same-view chord from raw 3D points at the rounded/clamped endpoint pixels; "
         "null if either center is invalid. Both endpoint depth matrices are returned. "
         "Call again for different endpoints. " + EDGE_NOTE,
         {"view_index", "a_xy", "b_xy", "distance", "a_depth_samples", "b_depth_samples", "unit"}),
    ):
        tool = registry.get(name)
        tool.fn = fn
        tool.spec = tool.spec.model_copy(update={
            "description": description, "args_schema_ref": SAMPLING_VERSION,
            "returns_schema_ref": SAMPLING_VERSION})
        registry.public_fields[name] = frozenset(fields | {
            "coordinate_transforms", "intended_targets", "replaces_result_id"} |
            ({"jev_quantity"} if measurement_reviewer is not None else set()))
    registry.decision_version = VERSION
    original = registry.call_tool
    counters = empty_counts()
    def counted(name, args, *rest, **kwargs):
        if name not in ALLOWED:
            raise DomainValueError(name, "omni_tool_scope_denied")
        measurement = int(name in MEASUREMENTS)
        counters["attempted"] += 1
        counters["attempted_measurements"] += measurement
        counters["calls"] += 1
        counters["measurements"] += measurement
        if name in MEASUREMENTS and (type(args.get("view_index")) is not int
                                    or not 0 <= args["view_index"] < len(scene.frames)):
            raise input_error(name, "view_index", [0], args.get("view_index"))
        return original(name, args, *rest, **kwargs)

    registry.call_tool = counted
    return registry, scene, counters
