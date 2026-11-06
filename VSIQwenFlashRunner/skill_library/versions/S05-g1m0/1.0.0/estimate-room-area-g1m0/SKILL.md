---
name: estimate-room-area-g1m0
description: Method for room_size_estimation in initial evidence state g1m0.
---

Initial geometry_3d=available and metric_scale=unavailable.

# Use geometric layout without a metric area claim

## Method

Identify the room boundaries and requested spatial scope in the original images. Available geometric layout and connectivity observations may support the arrangement when their local world conventions are valid, but they do not supply metric floor area. plane_fit_room_size requires scale and is not available in this initial state. Build a supported visual floor plan, avoid double-counting repeated or overlapping regions, and give an explicitly marked visual area estimate in square meters.

## Candidate and failure checks

- M5 provides candidates and technical validity checks. Even a unique category match does not prove the question's referent. Inspect frames/crops and select explicit object IDs.
- list_objects exposes candidates; an empty list does not prove scene-wide absence. detect_objects can supplement frozen views when authorized, but new box-only records do not have 3D geometry or valid masks.
- object_visible_frames is independent of 3D capability and requires a readable, consistent mask. Check the original images when masks are unavailable; do not equate failed calls with absence.
- Failed calls, invalid payloads and invalidated result IDs cannot support the answer. Keep valid observations from feedback; do not repeat deterministically invalid calls unchanged.
- The method stays fixed for the whole episode. Use only the currently authorized tools, including after a shared capability is withdrawn; continue with supported visual reasoning when necessary.
- Match the question's object roles, scope, complete options and units. Visual estimates are not tool measurements. Submit the requested answer with its actual evidence basis and references.
