---
name: estimate-object-size-g1m1
description: Method for object_size_estimation in initial evidence state g1m1.
---

Initial geometry_3d=available and metric_scale=available. Metric tools still require question authorization.

# Estimate an object's longest dimension

## Read usable evidence

- Follow the current round's action format. Program rounds use Python and documented tool names; preserve reserved variables and do not invent `scene` methods or XML calls. When finalization requests only JSON, return only that JSON.
- A tool returns a dictionary envelope. Its decoded payload is `r["data"]`, and its ID is `r["result_id"]`. For example, object rows are `list_objects(category_filter=category)["data"]`. Do not use `.value`, decode `data` again, or read payload keys from the envelope root.
- Reuse valid result IDs. For omitted fields or rows, use `ReadObservation`; advance by its returned `next_cursor`, and confirm the union of read ranges covers the relevant records. A page may contain fewer rows than the requested limit. Print output is diagnostic; use `show` and `YieldObservations` to expose evidence for another round. Rebuild variables from retained observations after yielding.
- Check inner status, local binding diagnostics, and current authorization before using a value. A failed object binding stops that object's geometry branch. A duplicate warning requires identity checks; it is not a blanket reason to discard every candidate. Report the evidence actually used.

## Method

1. Identify the target instance and the object's intended extent from the question. The canonical task asks for the maximum of the complete object's length, width, and height, not pixel length, volume, or a prominent part.
2. Select original images showing the overall outline and principal axes. Check occlusion, truncation, attachments, and multiple instances of the same category; disambiguate the target when necessary.
3. When metric geometry is authorized, call `object_3d_extent(obj_id)`. Check that `extent_metric` contains three finite non-negative lengths, and read `n_valid_points` and `degradation_flags`.
4. Take the maximum of `extent_metric` as the longest dimension, cross-checking `extent_longest_metric` when present. Tool lengths are in meters. Multiply by 100 when centimeters are requested; do not apply scale or unit conversion twice.
5. If `bbox_fallback_no_pointcloud` was used or the point set covers only the visible portion, use original images to assess axis-aligned-box overestimation, object rotation, underestimation from partial point clouds, and omitted attachments.
   The current three axes describe world-axis bounds, not automatically the object's own length, width, and height. Inspect all relevant same-category candidates, image-edge truncation, rotation, and complete outline coverage before treating the maximum as a complete-object estimate. Sufficient points, category validation, or empty flags do not establish surface completeness.
6. If measurements are unavailable, first determine which dimension is longest. Then estimate visually using complete views, reference objects at similar depths, and category size ranges. Return the best-supported single number.

## Branches and checks

- Without valid scale, geometry, or object grounding, do not call metric size tools or write an image-based scale guess back into the tool path.
- `extent_metric` is a three-element list; select its largest finite value instead of submitting the whole list.
- For centimeters derived from `extent_longest_metric`, use the registered `convert` derivation with `parameters={"field":"extent_longest_metric","from_unit":"m","to_unit":"cm"}` and the original result ID. Submit the unrounded converted value. A meter-field `field` derivation cannot support a centimeter answer. Do not change a pure conversion to `mixed` to bypass a unit or value mismatch.
- A partial outline or point cloud does not establish the exact size of the complete object.
- A crop must include the intended whole object and useful context. Check that the image was actually delivered and observed before using it; a crop or resize is not a new metric calibration.
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
