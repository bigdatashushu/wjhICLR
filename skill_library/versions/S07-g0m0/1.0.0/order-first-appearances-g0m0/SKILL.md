---
name: order-first-appearances-g0m0
description: Method for obj_appearance_order in initial evidence state g0m0.
---

Initial geometry_3d=unavailable and metric_scale=unavailable.

# Find earliest visible evidence in frame order

## Method

Read all categories and complete option sequences. Use frozen frame slots in their original order; crops are not new time points. Inspect early frames and preceding intervals for partial appearances, using clearer later views to recognize the same category. Consider every instance's earliest visibility. First detection is an upper bound, not proof of true first appearance; an empty or failed detection does not prove absence. Match the complete option sequence without inventing subframe timestamps.

## Candidate and failure checks

- M5 provides candidates and technical validity checks. Even a unique category match does not prove the question's referent. Inspect frames/crops and select explicit object IDs.
- list_objects exposes candidates; an empty list does not prove scene-wide absence. detect_objects can supplement frozen views when authorized, but new box-only records do not have 3D geometry or valid masks.
- object_visible_frames is independent of 3D capability and requires a readable, consistent mask. Check the original images when masks are unavailable; do not equate failed calls with absence.
- Failed calls, invalid payloads and invalidated result IDs cannot support the answer. Keep valid observations from feedback; do not repeat deterministically invalid calls unchanged.
- The method stays fixed for the whole episode. Use only the currently authorized tools, including after a shared capability is withdrawn; continue with supported visual reasoning when necessary.
- Match the question's object roles, scope, complete options and units. Visual estimates are not tool measurements. Submit the requested answer with its actual evidence basis and references.
