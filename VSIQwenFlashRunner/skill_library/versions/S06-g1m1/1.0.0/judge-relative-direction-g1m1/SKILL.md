---
name: judge-relative-direction-g1m1
description: Method for object_rel_direction in initial evidence state g1m1.
---

Initial geometry_3d=available and metric_scale=available. Metric tools still require question authorization.

# Judge direction from a specified location and heading

## Method

1. Extract `observer`, `facing_at`, and `target` separately, retaining the original options. Facing an object means the observer looks toward it; it does not refer to the furniture's own front.
2. Disambiguate each role's instance. For categories with multiple objects, use question modifiers, co-occurring structures, and visible frames to identify the specific `obj_id`. Do not use the first default category match.
3. Choose `difficulty` from the option set: use `easy` for left/right only, `medium` for left/right/back, and `hard` for the four front/back and left/right quadrants.
4. When geometry is authorized and local world conventions are valid, call `relative_direction_of(observer_id, facing_at_id, target_id, difficulty)`. Read and check `direction`, `difficulty`, `world_up_used`, and `handedness_used`.
5. Map the returned direction strictly to the current options. easy permits only left/right; medium permits left/right/back; hard permits front-left, front-right, back-left, and back-right.
6. If the tool is unavailable, build a local layout of the three roles from original images. Place the observer at an imagined origin facing facing_at, then determine which side contains target. Do not directly reuse left/right from the camera image.

## Branches and checks

- Do not interchange `observer`, `facing_at`, and `target`, or treat `facing_at` as a furniture orientation property.
- Relative direction does not require metric scale; missing metric scale alone must not block this method.
- Without a valid world frame, up direction, or object grounding, use the visual path rather than fabricating a tool direction.
- Returned `difficulty` must match the option vocabulary. A direction absent from the options indicates incorrect arguments or mapping.
- Near direction boundaries, with degenerate horizontal vectors, or with a mirrored visual-layout ambiguity, prioritize frames and instance bindings that could change the selected option.
- Before submission, check all three roles, difficulty vocabulary, and option mapping. Return a valid option for the current question.

## Candidate and failure checks

- M5 provides candidates and technical validity checks. Even a unique category match does not prove the question's referent. Inspect frames/crops and select explicit object IDs.
- list_objects exposes candidates; an empty list does not prove scene-wide absence. detect_objects can supplement frozen views when authorized, but new box-only records do not have 3D geometry or valid masks.
- object_visible_frames is independent of 3D capability and requires a readable, consistent mask. Check the original images when masks are unavailable; do not equate failed calls with absence.
- Failed calls, invalid payloads and invalidated result IDs cannot support the answer. Keep valid observations from feedback; do not repeat deterministically invalid calls unchanged.
- The method stays fixed for the whole episode. Use only the currently authorized tools, including after a shared capability is withdrawn; continue with supported visual reasoning when necessary.
- Match the question's object roles, scope, complete options and units. Visual estimates are not tool measurements. Submit the requested answer with its actual evidence basis and references.
