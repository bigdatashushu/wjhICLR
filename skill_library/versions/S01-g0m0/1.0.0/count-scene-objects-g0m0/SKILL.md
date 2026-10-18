---
name: count-scene-objects-g0m0
description: Method for object_counting in initial evidence state g0m0.
---

Initial geometry_3d=unavailable and metric_scale=unavailable.

# Count distinct visible instances

## Method

Extract the target category, requested spatial scope and exclusions. Inspect the frozen views and build an instance table using appearance, neighboring structures and co-occurrence. Keep similar objects visible simultaneously as separate instances. Match reappearances across views using multiple identity cues; never sum frame detections or use the candidate-list length as the answer. Review occlusion and missed detections that could change the count. Return the best supported non-negative integer; failed detections do not mean zero.

## Candidate and failure checks

- M5 provides candidates and technical validity checks. Even a unique category match does not prove the question's referent. Inspect frames/crops and select explicit object IDs.
- list_objects exposes candidates; an empty list does not prove scene-wide absence. detect_objects can supplement frozen views when authorized, but new box-only records do not have 3D geometry or valid masks.
- object_visible_frames is independent of 3D capability and requires a readable, consistent mask. Check the original images when masks are unavailable; do not equate failed calls with absence.
- Failed calls, invalid payloads and invalidated result IDs cannot support the answer. Keep valid observations from feedback; do not repeat deterministically invalid calls unchanged.
- The method stays fixed for the whole episode. Use only the currently authorized tools, including after a shared capability is withdrawn; continue with supported visual reasoning when necessary.
- Match the question's object roles, scope, complete options and units. Visual estimates are not tool measurements. Submit the requested answer with its actual evidence basis and references.
