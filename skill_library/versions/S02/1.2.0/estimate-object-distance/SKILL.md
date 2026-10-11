---
name: estimate-object-distance
description: Use for object_abs_distance to estimate the nearest distance between two named objects using authorized metric surface distances, retaining visual estimation when scale or grounding is invalid.
---

# Estimate absolute distance between objects

## Method

1. Extract both objects' roles and disambiguating descriptions. The question asks for the nearest distance between objects, not camera-to-object distance or a default centroid distance.
2. Identify the two specific instances. For categories with multiple instances, use question modifiers, co-occurring structures, and visible frames to disambiguate them, then prefer explicit `obj_id` values in the distance call.
3. When metric geometry is authorized, call `object_distance_m(obj_a, obj_b)`. Check that returned `a` and `b` match the intended instances, and inspect `distance_m`, `definition`, `n_valid_points`, and `degradation_flags`.
4. `distance_m` is already in meters; use it directly as a metric candidate without applying scale again. Use `distance_normalized` only for auditing or consistency checks; do not independently convert it to meters without valid scale.
5. Check whether occlusion, point-set contamination, partial visibility, or duplicate instances affect the nearest surfaces. The tool uses a robust low-quantile point-cloud surface-distance proxy; do not claim exact ground truth for complete invisible surfaces.
6. If tool measurements are unavailable, compare object boundaries, adjacency, and depth cues across original views. Use supported reference sizes to estimate the metric magnitude, then return the best-supported finite non-negative number.

## Branches and checks

- Without valid scale, 3D geometry, or object grounding, do not call metric distance tools or fabricate scale authorization from a visual guess.
- Tool failures, invalid numbers, empty point sets, and invalidated results do not mean zero distance; use visual estimation instead.
- Do not reconvert `distance_m`, which is already in meters. Convert to another unit only when explicitly required by the question.
- Resolving both `obj_a` and `obj_b` to the same object indicates a grounding error; disambiguate again.
- Visual reference objects should be at similar depths and have reasonable size ranges. Pixel ratios across different depths are not metric ratios.
- Before submission, check object identities, distance definition, units, and evidence basis. Do not record a visual estimate as a tool measurement.
