---
name: estimate-object-size-g0m0
description: Method for object_size_estimation in initial evidence state g0m0.
---

Initial geometry_3d=unavailable and metric_scale=unavailable.

# Estimate the complete object's longest dimension

## Method

Identify the specified instance and its complete outline across views. Decide which of length, width and height is largest. Assess occlusion, omitted parts and perspective before giving a single visual estimate. Category size knowledge is an uncertain cue, not a fixed constant. Express the finite non-negative number in the requested unit, commonly centimeters; mark visual_estimate.

## Candidate and failure checks

- M5 provides candidates and technical validity checks. Even a unique category match does not prove the question's referent. Inspect frames/crops and select explicit object IDs.
- list_objects exposes candidates; an empty list does not prove scene-wide absence. detect_objects can supplement frozen views when authorized, but new box-only records do not have 3D geometry or valid masks.
- object_visible_frames is independent of 3D capability and requires a readable, consistent mask. Check the original images when masks are unavailable; do not equate failed calls with absence.
- Failed calls, invalid payloads and invalidated result IDs cannot support the answer. Keep valid observations from feedback; do not repeat deterministically invalid calls unchanged.
- The method stays fixed for the whole episode. Use only the currently authorized tools, including after a shared capability is withdrawn; continue with supported visual reasoning when necessary.
- Match the question's object roles, scope, complete options and units. Visual estimates are not tool measurements. Submit the requested answer with its actual evidence basis and references.
