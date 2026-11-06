---
name: estimate-object-size-g1m1
description: Method for object_size_estimation in initial evidence state g1m1.
---

Initial geometry_3d=available and metric_scale=available. Metric tools still require question authorization.

# Estimate an object's longest dimension

## Method

1. Identify the target instance and the object's intended extent from the question. The canonical task asks for the maximum of the complete object's length, width, and height, not pixel length, volume, or a prominent part.
2. Select original images showing the overall outline and principal axes. Check occlusion, truncation, attachments, and multiple instances of the same category; disambiguate the target when necessary.
3. When metric geometry is authorized, call `object_3d_extent(obj_id)`. Check that `extent_metric` contains three finite non-negative lengths, and read `n_valid_points` and `degradation_flags`.
4. Take the maximum of `extent_metric` as the longest dimension, cross-checking `extent_longest_metric` when present. Tool lengths are in meters. Multiply by 100 when centimeters are requested; do not apply scale or unit conversion twice.
5. If `bbox_fallback_no_pointcloud` was used or the point set covers only the visible portion, use original images to assess axis-aligned-box overestimation, object rotation, underestimation from partial point clouds, and omitted attachments.
6. If measurements are unavailable, first determine which dimension is longest. Then estimate visually using complete views, reference objects at similar depths, and category size ranges. Return the best-supported single number.

## Branches and checks

- Without valid scale, geometry, or object grounding, do not call metric size tools or write an image-based scale guess back into the tool path.
- `extent_metric` is a three-element list; select its largest finite value instead of submitting the whole list.
- A partial outline or point cloud does not establish the exact size of the complete object.
- An axis-aligned bounding box may overestimate a rotated object; partial surfaces may underestimate its complete size. Interpret these effects using `degradation_flags` and original images.
- Category knowledge provides an estimation range, not a fixed category constant that replaces scene-specific judgment.
- Before submission, check target identity, the longest-dimension choice, and a single meters-to-centimeters conversion.

## Candidate and failure checks

- M5 provides candidates and technical validity checks. Even a unique category match does not prove the question's referent. Inspect frames/crops and select explicit object IDs.
- list_objects exposes candidates; an empty list does not prove scene-wide absence. detect_objects can supplement frozen views when authorized, but new box-only records do not have 3D geometry or valid masks.
- object_visible_frames is independent of 3D capability and requires a readable, consistent mask. Check the original images when masks are unavailable; do not equate failed calls with absence.
- Failed calls, invalid payloads and invalidated result IDs cannot support the answer. Keep valid observations from feedback; do not repeat deterministically invalid calls unchanged.
- The method stays fixed for the whole episode. Use only the currently authorized tools, including after a shared capability is withdrawn; continue with supported visual reasoning when necessary.
- Match the question's object roles, scope, complete options and units. Visual estimates are not tool measurements. Submit the requested answer with its actual evidence basis and references.
