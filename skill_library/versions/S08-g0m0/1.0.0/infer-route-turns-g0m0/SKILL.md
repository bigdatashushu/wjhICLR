---
name: infer-route-turns-g0m0
description: Method for route_planning in initial evidence state g0m0.
---

Initial geometry_3d=unavailable and metric_scale=unavailable.

# Rehearse turns along the full specified route

## Method

Parse the start, initial facing object, ordered landmarks and each complete option action sequence. Disambiguate repeated landmarks using neighboring structures and observation order. Move an imagined observer through the image layout, updating heading after every segment and judging each turn from the arrival heading. Check passages, walls and doorways; nearby object centers do not prove traversability. Follow the specified route rather than a shorter alternative and return the supported complete sequence.

## Candidate and failure checks

- M5 provides candidates and technical validity checks. Even a unique category match does not prove the question's referent. Inspect frames/crops and select explicit object IDs.
- list_objects exposes candidates; an empty list does not prove scene-wide absence. detect_objects can supplement frozen views when authorized, but new box-only records do not have 3D geometry or valid masks.
- object_visible_frames is independent of 3D capability and requires a readable, consistent mask. Check the original images when masks are unavailable; do not equate failed calls with absence.
- Failed calls, invalid payloads and invalidated result IDs cannot support the answer. Keep valid observations from feedback; do not repeat deterministically invalid calls unchanged.
- The method stays fixed for the whole episode. Use only the currently authorized tools, including after a shared capability is withdrawn; continue with supported visual reasoning when necessary.
- Match the question's object roles, scope, complete options and units. Visual estimates are not tool measurements. Submit the requested answer with its actual evidence basis and references.
