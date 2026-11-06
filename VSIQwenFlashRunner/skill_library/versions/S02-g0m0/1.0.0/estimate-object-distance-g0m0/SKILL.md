---
name: estimate-object-distance-g0m0
description: Method for object_abs_distance in initial evidence state g0m0.
---

Initial geometry_3d=unavailable and metric_scale=unavailable.

# Estimate the nearest separation visually

## Method

Identify both specific object instances using the question's modifiers and shared structures. Compare their nearest visible boundaries across views, distinguishing contact, foreground/background ordering and perspective. The reference is not the camera. If the images support a physical magnitude, give a finite non-negative visual estimate in the requested unit. Do not claim that pixel gaps are meters or invent a scale measurement. Record the result as visual_estimate.

## Candidate and failure checks

- M5 provides candidates and technical validity checks. Even a unique category match does not prove the question's referent. Inspect frames/crops and select explicit object IDs.
- list_objects exposes candidates; an empty list does not prove scene-wide absence. detect_objects can supplement frozen views when authorized, but new box-only records do not have 3D geometry or valid masks.
- object_visible_frames is independent of 3D capability and requires a readable, consistent mask. Check the original images when masks are unavailable; do not equate failed calls with absence.
- Failed calls, invalid payloads and invalidated result IDs cannot support the answer. Keep valid observations from feedback; do not repeat deterministically invalid calls unchanged.
- The method stays fixed for the whole episode. Use only the currently authorized tools, including after a shared capability is withdrawn; continue with supported visual reasoning when necessary.
- Match the question's object roles, scope, complete options and units. Visual estimates are not tool measurements. Submit the requested answer with its actual evidence basis and references.
