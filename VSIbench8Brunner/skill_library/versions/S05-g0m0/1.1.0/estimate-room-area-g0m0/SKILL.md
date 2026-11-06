---
name: estimate-room-area-g0m0
description: Method for room_size_estimation in initial evidence state g0m0.
---

Initial geometry_3d=unavailable and metric_scale=unavailable.

# Estimate floor area from a visual floor plan

## Method

Identify the requested room or union of spaces using walls, corners and doorways. Build a simplified layout across views. Do not count furniture-occluded floor as missing area, sum repeated views twice or include an adjacent room without support from the question. Estimate supported dimensions and decompose irregular regions before estimating their union. Return a finite positive visual estimate in square meters, not floor pixel count or normalized area.

## Task decision sheet

Sketch a working floor layout of the requested room or space union. Give each region a temporary label, boundary cues, supporting frozen slots, and an inclusion reason. A doorway into an adjacent room is a boundary cue rather than automatic permission to include that room.

For a rectangular region, assign separate visual evidence and uncertain estimates to its length and width. For an irregular layout, decompose it into supported rectangles or polygons and subtract shared overlap once. Match repeated views to the same physical region. Do not fill a recess or an unrequested neighboring room using an enclosing rectangle.

Furniture-covered floor remains part of the room. Keep visible floor, free walking area, normalized scene span and total room area conceptually separate. A door or furnishing size can supply an uncertain visual assumption but cannot calibrate runtime metric scale. When a valid metric area is available, check its boundary coverage against the sheet and do not reconvert square-meter output. Finish with one supported positive estimate for the requested union.

## Candidate and failure checks

- M5 provides candidates and technical validity checks. Even a unique category match does not prove the question's referent. Inspect frames/crops and select explicit object IDs.
- list_objects exposes candidates; an empty list does not prove scene-wide absence. detect_objects can supplement frozen views when authorized, but new box-only records do not have 3D geometry or valid masks.
- object_visible_frames is independent of 3D capability and requires a readable, consistent mask. Check the original images when masks are unavailable; do not equate failed calls with absence.
- Failed calls, invalid payloads and invalidated result IDs cannot support the answer. Keep valid observations from feedback; do not repeat deterministically invalid calls unchanged.
- The method stays fixed for the whole episode. Use only the currently authorized tools, including after a shared capability is withdrawn; continue with supported visual reasoning when necessary.
- Match the question's object roles, scope, complete options and units. Visual estimates are not tool measurements. Submit the requested answer with its actual evidence basis and references.
