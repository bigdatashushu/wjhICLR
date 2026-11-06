---
name: estimate-room-area-g1m1
description: Method for room_size_estimation in initial evidence state g1m1.
---

Initial geometry_3d=available and metric_scale=available. Metric tools still require question authorization.

# Estimate room boundaries and floor area

## Read usable evidence

- Follow the current round's action format. Program rounds use Python and documented tool names; preserve reserved variables and do not invent `scene` methods or XML calls. When finalization requests only JSON, return only that JSON.
- A tool returns a dictionary envelope. Its decoded payload is `r["data"]`, and its ID is `r["result_id"]`. For example, object rows are `list_objects(category_filter=category)["data"]`. Do not use `.value`, decode `data` again, or read payload keys from the envelope root.
- Reuse valid result IDs. For omitted fields or rows, use `ReadObservation`; advance by its returned `next_cursor`, and confirm the union of read ranges covers the relevant records. A page may contain fewer rows than the requested limit. Print output is diagnostic; use `show` and `YieldObservations` to expose evidence for another round. Rebuild variables from retained observations after yielding.
- Check inner status, local binding diagnostics, and current authorization before using a value. A failed object binding stops that object's geometry branch. A duplicate warning requires identity checks; it is not a blanket reason to discard every candidate. Report the evidence actually used.

## Method

1. Determine whether the requested space is a single room, a connected region, or a union of multiple spaces. Use doorways, walls, corners, and camera motion to distinguish spaces; count repeated views of the same region only once.
2. Use original images and the scene summary to check floor, wall, and boundary coverage. Furniture-occluded floor is not missing area. Floor pixel count, wall area, and room diagonal are not the requested floor area.
3. When metric plane tools are authorized, call `plane_fit_room_size()` and read `room_area_m2`, `room_diagonal_normalized`, `plane_inlier_ratio`, `fit_quality`, and `degradation_flags`.
4. `room_area_m2` is already in square meters; use it directly as an area candidate without reapplying scale or squared conversion. `fit_quality` is audit information, not an independent reason to abstain outside the frozen thresholds.
   The current estimator uses a robust enclosing rectangle of the point cloud's horizontal projection. It does not measure an identified floor-region union. Check walls, doorways, recesses, adjacent spaces, and repeated coverage in original views before using this proxy for the requested room. A positive area, empty flags, or high fit quality does not establish the spatial scope. A saturated `floor_fit_ratio=1` does not prove complete floor coverage: inspect its documented numerator and denominator when available, and keep missing coverage evidence unknown.
5. Check that fitted coverage matches the requested space. Interpret nonrectangular or multi-region scenes through supported decomposition and unions. Do not fill recesses, space beyond doors, or gaps between regions using an enclosing rectangle or convex hull.
6. If plane measurements are unavailable, construct a simplified floor layout across original views. Estimate major dimensions from wall segments, doorways, and size references at similar depths, then compute area from a supported combination of regions.
7. Check whether unobserved boundaries, repeated regions, or spatial scope dominate the uncertainty. Return the best-supported finite positive area in square meters.

## Branches and checks

- Without valid scale, a point cloud, or metric authorization, do not call `plane_fit_room_size` or present normalized area as square meters.
- Tool failures, invalid numbers, and invalidated results do not mean zero area; use visual layout estimation.
- `room_area_m2` already includes area-scale conversion. Do not multiply by scale squared again.
- When submitting the direct area candidate, use the original unrounded `room_area_m2`, `unit="m2"`, and a `field` derivation from that result. Code-only rounding does not change the registered replay operation. Use a visual layout correction only with actual additional evidence and a matching evidence basis.
- Furniture occlusion does not automatically reduce room area. Do not sum repeated views or overlapping regions twice.
- Combining rooms must follow the requested scope. Visibility of an adjacent space does not justify expanding the boundary.
- Before submission, check spatial scope, square-meter units, and consistency with the measurement or visual basis.

## Candidate and failure checks

- M5 provides candidates and technical validity checks. Even a unique category match does not prove the question's referent. Inspect frames/crops and select explicit object IDs.
- list_objects exposes candidates; an empty list does not prove scene-wide absence. detect_objects can supplement frozen views when authorized, but new box-only records do not have 3D geometry or valid masks.
- object_visible_frames is independent of 3D capability and requires a readable, consistent mask. Check the original images when masks are unavailable; do not equate failed calls with absence.
- Failed calls, invalid payloads and invalidated result IDs cannot support the answer. Keep valid observations from feedback; do not repeat deterministically invalid calls unchanged.
- The method stays fixed for the whole episode. Use only the currently authorized tools, including after a shared capability is withdrawn; continue with supported visual reasoning when necessary.
- Match the question's object roles, scope, complete options and units. Visual estimates are not tool measurements. Submit the requested answer with its actual evidence basis and references.
