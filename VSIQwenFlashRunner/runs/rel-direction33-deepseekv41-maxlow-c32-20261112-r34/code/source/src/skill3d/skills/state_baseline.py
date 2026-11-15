"""Deterministic 24-slot format migration; no model calls or learning feedback."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from skill3d.schemas.evidence import SKILL_EVIDENCE_STATES
from skill3d.skills.v11_library import (
    V11LibraryError, _write_create_once, build_v11_snapshot, load_skill_source_v11,
)

STATE_SNAPSHOT_ID = "S0-v11-initial-state-skills"
ORIGIN_SNAPSHOT_ID = "S0-v11-english"

_VISUAL_METHODS = {
    "object_counting": (
        "Count distinct visible instances",
        "Extract the target category, requested spatial scope and exclusions. Inspect the "
        "frozen views and build an instance table using appearance, neighboring structures "
        "and co-occurrence. Keep similar objects visible simultaneously as separate "
        "instances. Match reappearances across views using multiple identity cues; never "
        "sum frame detections or use the candidate-list length as the answer. Review "
        "occlusion and missed detections that could change the count. Return the best "
        "supported non-negative integer; failed detections do not mean zero."),
    "object_abs_distance": (
        "Estimate the nearest separation visually",
        "Identify both specific object instances using the question's modifiers and shared "
        "structures. Compare their nearest visible boundaries across views, distinguishing "
        "contact, foreground/background ordering and perspective. The reference is not "
        "the camera. If the images support a physical magnitude, give a finite non-negative "
        "visual estimate in the requested unit. Do not claim that pixel gaps are meters or "
        "invent a scale measurement. Record the result as visual_estimate."),
    "object_rel_distance": (
        "Compare all options in a visual layout",
        "Extract the named reference object and every option category. Disambiguate the "
        "reference in the images. For each candidate category, consider the instance "
        "nearest to that reference as required by the question. Compare shared views, "
        "occlusion, depth cues and neighboring structures. A missing candidate remains "
        "unknown rather than infinitely distant. Do not rank by pixel gaps across different "
        "depths. Compare the complete option set and return its best supported letter."),
    "object_size_estimation": (
        "Estimate the complete object's longest dimension",
        "Identify the specified instance and its complete outline across views. Decide "
        "which of length, width and height is largest. Assess occlusion, omitted parts "
        "and perspective before giving a single visual estimate. Category size knowledge "
        "is an uncertain cue, not a fixed constant. Express the finite non-negative "
        "number in the requested unit, commonly centimeters; mark visual_estimate."),
    "room_size_estimation": (
        "Estimate floor area from a visual floor plan",
        "Identify the requested room or union of spaces using walls, corners and doorways. "
        "Build a simplified layout across views. Do not count furniture-occluded floor as "
        "missing area, sum repeated views twice or include an adjacent room without "
        "support from the question. Estimate supported dimensions and decompose irregular "
        "regions before estimating their union. Return a finite positive visual estimate "
        "in square meters, not floor pixel count or normalized area."),
    "object_rel_direction": (
        "Judge direction from the specified observer",
        "Extract observer, facing object and target as distinct roles and disambiguate "
        "their instances. Build a layout from the original views, place the observer at "
        "the requested location facing the facing object, and compare the target against "
        "that heading. Camera-image left/right is not automatically observer left/right. "
        "Use the options' direction vocabulary, consider mirrored layouts and boundaries, "
        "and return the supported option letter."),
    "obj_appearance_order": (
        "Find earliest visible evidence in frame order",
        "Read all categories and complete option sequences. Use frozen frame slots in "
        "their original order; crops are not new time points. Inspect early frames and "
        "preceding intervals for partial appearances, using clearer later views to "
        "recognize the same category. Consider every instance's earliest visibility. "
        "First detection is an upper bound, not proof of true first appearance; an empty "
        "or failed detection does not prove absence. Match the complete option sequence "
        "without inventing subframe timestamps."),
    "route_planning": (
        "Rehearse turns along the full specified route",
        "Parse the start, initial facing object, ordered landmarks and each complete "
        "option action sequence. Disambiguate repeated landmarks using neighboring "
        "structures and observation order. Move an imagined observer through the image "
        "layout, updating heading after every segment and judging each turn from the "
        "arrival heading. Check passages, walls and doorways; nearby object centers do "
        "not prove traversability. Follow the specified route rather than a shorter "
        "alternative and return the supported complete sequence."),
}

_NO_SCALE_METHODS = {
    "object_abs_distance": (
        "Use relative geometry to support an absolute visual estimate",
        "Disambiguate both specific objects in the images and pass explicit IDs to "
        "available nonmetric tools. surface_distance_between_objects can provide "
        "distance_normalized and surface/point-set diagnostics from valid objects. "
        "Use these only to check relative separation and surface coverage; there is no "
        "valid conversion from this value to meters. If a metric magnitude is required, "
        "use image evidence for an explicitly marked visual estimate. Do not call "
        "object_distance_m or fabricate scale. Check both identities, nearest-surface "
        "definition and the requested unit before returning a finite number."),
    "object_size_estimation": (
        "Use shape evidence without fabricating metric size",
        "Disambiguate the complete target and inspect its outline and principal axes "
        "across images. Available object_centroid results may help identify its "
        "reconstructed position, but a centroid does not measure object size. "
        "object_3d_extent requires metric authorization and is not available in this "
        "initial state. Determine the longest complete dimension visually and give an "
        "explicit visual estimate in the requested unit. Do not convert normalized "
        "geometry to centimeters or write a visual scale guess into the tool path."),
    "room_size_estimation": (
        "Use geometric layout without a metric area claim",
        "Identify the room boundaries and requested spatial scope in the original "
        "images. Available geometric layout and connectivity observations may support "
        "the arrangement when their local world conventions are valid, but they do not "
        "supply metric floor area. plane_fit_room_size requires scale and is not "
        "available in this initial state. Build a supported visual floor plan, avoid "
        "double-counting repeated or overlapping regions, and give an explicitly "
        "marked visual area estimate in square meters."),
}

_COMMON_CHECKS = """## Candidate and failure checks

- M5 provides candidates and technical validity checks. Even a unique category match does not prove the question's referent. Inspect frames/crops and select explicit object IDs.
- list_objects exposes candidates; an empty list does not prove scene-wide absence. detect_objects can supplement frozen views when authorized, but new box-only records do not have 3D geometry or valid masks.
- object_visible_frames is independent of 3D capability and requires a readable, consistent mask. Check the original images when masks are unavailable; do not equate failed calls with absence.
- Failed calls, invalid payloads and invalidated result IDs cannot support the answer. Keep valid observations from feedback; do not repeat deterministically invalid calls unchanged.
- The method stays fixed for the whole episode. Use only the currently authorized tools, including after a shared capability is withdrawn; continue with supported visual reasoning when necessary.
- Match the question's object roles, scope, complete options and units. Visual estimates are not tool measurements. Submit the requested answer with its actual evidence basis and references.
"""


def _state_source(entry: dict, original: str, state: str) -> str:
    task = entry["question_type"]
    from skill3d.schemas.skill import parse_skill_markdown

    metadata, body = parse_skill_markdown(original)
    name = metadata["name"] + "-" + state
    condition = {
        "g0m0": "Initial geometry_3d=unavailable and metric_scale=unavailable.",
        "g1m0": "Initial geometry_3d=available and metric_scale=unavailable.",
        "g1m1": "Initial geometry_3d=available and metric_scale=available. Metric tools still require question authorization.",
    }[state]
    if state == "g0m0" or state == "g1m0" and task in _NO_SCALE_METHODS:
        title, method = (_VISUAL_METHODS if state == "g0m0" else _NO_SCALE_METHODS)[task]
        body = f"# {title}\n\n## Method\n\n{method}\n\n"
    else:
        body = body.replace("`evidence_degraded`", "`diagnostic_warnings`")
        body = body.replace("or evidence is degraded", "or its payload is unusable")
        body = body.replace("geometry and world-frame evidence are authorized",
                            "geometry is authorized and local world conventions are valid")
        # Preserve complete task examples and lessons while correcting retired fields.
        body += "\n"
    return (
        f"---\nname: {name}\ndescription: Method for {task} in initial evidence state {state}.\n"
        f"---\n\n{condition}\n\n{body.strip()}\n\n{_COMMON_CHECKS}"
    )


def build_initial_state_baseline(library_root: str | Path, *, write_sources: bool = True):
    """Create or verify immutable sources and return the complete migration snapshot."""
    root = Path(library_root)
    origin = json.loads(
        (root / "snapshots" / f"snapshot_{ORIGIN_SNAPSHOT_ID}.json").read_bytes())
    sources, origins = [], {}
    for entry in origin["entries"].values():
        path = root / entry["source_ref"]
        original = path.read_text(encoding="utf-8")
        digest = hashlib.sha256(original.encode()).hexdigest()
        if digest != entry["content_sha256"]:
            raise V11LibraryError("英文迁移源 hash 不一致")
        for state in SKILL_EVIDENCE_STATES:
            identity = f"{entry['skill_id']}-{state}"
            text = _state_source(entry, original, state)
            from skill3d.schemas.skill import SkillSpecV11

            spec = SkillSpecV11(skill_id=identity, version="1.0.0",
                                question_type=entry["question_type"],
                                evidence_state=state, skill_md=text)
            ref = f"versions/{identity}/1.0.0/{spec.name}/SKILL.md"
            if write_sources:
                _write_create_once(root / ref, text.encode())
            elif not (root / ref).is_file() or (root / ref).read_bytes() != text.encode():
                raise V11LibraryError(f"状态方法缺失或过期: {ref}")
            sources.append(load_skill_source_v11(
                root, source_ref=ref, skill_id=identity, version="1.0.0",
                question_type=entry["question_type"], evidence_state=state))
            origins[identity] = {
                "origin_skill_version": f"{entry['skill_id']}@{entry['version']}",
                "origin_content_sha256": digest,
            }
    return build_v11_snapshot(
        sources, snapshot_id=STATE_SNAPSHOT_ID, parent_snapshot_id=ORIGIN_SNAPSHOT_ID,
        generation=0, created_by="initial_state_format_migration",
        origins=origins, require_complete=True)
