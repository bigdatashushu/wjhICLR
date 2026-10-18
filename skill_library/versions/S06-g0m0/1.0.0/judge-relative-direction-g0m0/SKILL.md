---
name: judge-relative-direction-g0m0
description: Method for object_rel_direction in initial evidence state g0m0.
---

Initial geometry_3d=unavailable and metric_scale=unavailable.

# Judge direction from the specified observer

## Method

Extract observer, facing object and target as distinct roles and disambiguate their instances. Build a layout from the original views, place the observer at the requested location facing the facing object, and compare the target against that heading. Camera-image left/right is not automatically observer left/right. Use the options' direction vocabulary, consider mirrored layouts and boundaries, and return the supported option letter.

## Candidate and failure checks

- M5 provides candidates and technical validity checks. Even a unique category match does not prove the question's referent. Inspect frames/crops and select explicit object IDs.
- list_objects exposes candidates; an empty list does not prove scene-wide absence. detect_objects can supplement frozen views when authorized, but new box-only records do not have 3D geometry or valid masks.
- object_visible_frames is independent of 3D capability and requires a readable, consistent mask. Check the original images when masks are unavailable; do not equate failed calls with absence.
- Failed calls, invalid payloads and invalidated result IDs cannot support the answer. Keep valid observations from feedback; do not repeat deterministically invalid calls unchanged.
- The method stays fixed for the whole episode. Use only the currently authorized tools, including after a shared capability is withdrawn; continue with supported visual reasoning when necessary.
- Match the question's object roles, scope, complete options and units. Visual estimates are not tool measurements. Submit the requested answer with its actual evidence basis and references.
