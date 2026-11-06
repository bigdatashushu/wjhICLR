---
name: estimate-object-distance-g1m0
description: Method for object_abs_distance in initial evidence state g1m0.
---

Initial geometry_3d=available and metric_scale=unavailable.

# Use relative geometry to support an absolute visual estimate

## Method

Disambiguate both specific objects in the images and pass explicit IDs to available nonmetric tools. surface_distance_between_objects can provide distance_normalized and surface/point-set diagnostics from valid objects. Use these only to check relative separation and surface coverage; there is no valid conversion from this value to meters. If a metric magnitude is required, use image evidence for an explicitly marked visual estimate. Do not call object_distance_m or fabricate scale. Check both identities, nearest-surface definition and the requested unit before returning a finite number.

## Candidate and failure checks

- M5 provides candidates and technical validity checks. Even a unique category match does not prove the question's referent. Inspect frames/crops and select explicit object IDs.
- list_objects exposes candidates; an empty list does not prove scene-wide absence. detect_objects can supplement frozen views when authorized, but new box-only records do not have 3D geometry or valid masks.
- object_visible_frames is independent of 3D capability and requires a readable, consistent mask. Check the original images when masks are unavailable; do not equate failed calls with absence.
- Failed calls, invalid payloads and invalidated result IDs cannot support the answer. Keep valid observations from feedback; do not repeat deterministically invalid calls unchanged.
- The method stays fixed for the whole episode. Use only the currently authorized tools, including after a shared capability is withdrawn; continue with supported visual reasoning when necessary.
- Match the question's object roles, scope, complete options and units. Visual estimates are not tool measurements. Submit the requested answer with its actual evidence basis and references.
