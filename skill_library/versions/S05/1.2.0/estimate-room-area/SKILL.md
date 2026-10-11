---
name: estimate-room-area
description: Use for room_size_estimation to identify the requested spatial scope and estimate floor area from authorized plane fitting or visual layout, handling multiple rooms, occlusion, and square meters.
---

# Estimate room boundaries and floor area

## Method

1. Determine whether the requested space is a single room, a connected region, or a union of multiple spaces. Use doorways, walls, corners, and camera motion to distinguish spaces; count repeated views of the same region only once.
2. Use original images and the scene summary to check floor, wall, and boundary coverage. Furniture-occluded floor is not missing area. Floor pixel count, wall area, and room diagonal are not the requested floor area.
3. When metric plane tools are authorized, call `plane_fit_room_size()` and read `room_area_m2`, `room_diagonal_normalized`, `plane_inlier_ratio`, `fit_quality`, and `degradation_flags`.
4. `room_area_m2` is already in square meters; use it directly as an area candidate without reapplying scale or squared conversion. `fit_quality` is audit information, not an independent reason to abstain outside the frozen thresholds.
5. Check that fitted coverage matches the requested space. Interpret nonrectangular or multi-region scenes through supported decomposition and unions. Do not fill recesses, space beyond doors, or gaps between regions using an enclosing rectangle or convex hull.
6. If plane measurements are unavailable, construct a simplified floor layout across original views. Estimate major dimensions from wall segments, doorways, and size references at similar depths, then compute area from a supported combination of regions.
7. Check whether unobserved boundaries, repeated regions, or spatial scope dominate the uncertainty. Return the best-supported finite positive area in square meters.

## Branches and checks

- Without valid scale, a point cloud, or metric authorization, do not call `plane_fit_room_size` or present normalized area as square meters.
- Tool failures, invalid numbers, and invalidated results do not mean zero area; use visual layout estimation.
- `room_area_m2` already includes area-scale conversion. Do not multiply by scale squared again.
- Furniture occlusion does not automatically reduce room area. Do not sum repeated views or overlapping regions twice.
- Combining rooms must follow the requested scope. Visibility of an adjacent space does not justify expanding the boundary.
- Before submission, check spatial scope, square-meter units, and consistency with the measurement or visual basis.
