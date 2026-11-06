---
name: estimate-object-distance-g1m1
description: Method for object_abs_distance in initial evidence state g1m1.
---

Initial geometry_3d=available and metric_scale=available. Metric tools still require question authorization.

# Estimate absolute distance between objects

## Method

1. Extract both objects' roles and disambiguating descriptions. The question asks for the nearest distance between objects, not camera-to-object distance or a default centroid distance.
2. Identify the two specific instances. For categories with multiple instances, use question modifiers, co-occurring structures, and visible frames to disambiguate them, then prefer explicit `obj_id` values in the distance call.
3. When metric geometry is authorized, call `object_surface_distance(obj_a, obj_b, unit="m")`. Check that returned `a` and `b` match the intended instances, and inspect `distance`, `definition`, `n_valid_points`, and `degradation_flags`.
4. `distance` is already in meters; use it directly as a metric candidate without applying scale again. Use `distance_normalized` only for auditing or consistency checks; do not independently convert it to meters without valid scale.
5. Check whether occlusion, point-set contamination, partial visibility, or duplicate instances affect the nearest surfaces. The tool uses a robust low-quantile point-cloud surface-distance proxy; do not claim exact ground truth for complete invisible surfaces.
6. If tool measurements are unavailable, compare object boundaries, adjacency, and depth cues across original views. Use supported reference sizes to estimate the metric magnitude, then return the best-supported finite non-negative number.

## Branches and checks

- Without valid scale, 3D geometry, or object grounding, do not call metric distance tools or fabricate scale authorization from a visual guess.
- Tool failures, invalid numbers, empty point sets, and invalidated results do not mean zero distance; use visual estimation instead.
- Do not reconvert `distance`, which is already in meters. Convert to another unit only when explicitly required by the question.
- Resolving both `obj_a` and `obj_b` to the same object indicates a grounding error; disambiguate again.
- Visual reference objects should be at similar depths and have reasonable size ranges. Pixel ratios across different depths are not metric ratios.
- Before submission, check object identities, distance definition, units, and evidence basis. Do not record a visual estimate as a tool measurement.

## Candidate and failure checks

- M5 provides candidates and technical validity checks. Even a unique category match does not prove the question's referent. Inspect frames/crops and select explicit object IDs.
- list_objects exposes candidates; an empty list does not prove scene-wide absence. detect_objects can supplement frozen views when authorized, but new box-only records do not have 3D geometry or valid masks.
- object_visible_frames is independent of 3D capability and requires a readable, consistent mask. Check the original images when masks are unavailable; do not equate failed calls with absence.
- Failed calls, invalid payloads and invalidated result IDs cannot support the answer. Keep valid observations from feedback; do not repeat deterministically invalid calls unchanged.
- The method stays fixed for the whole episode. Use only the currently authorized tools, including after a shared capability is withdrawn; continue with supported visual reasoning when necessary.
- Match the question's object roles, scope, complete options and units. Visual estimates are not tool measurements. Submit the requested answer with its actual evidence basis and references.
