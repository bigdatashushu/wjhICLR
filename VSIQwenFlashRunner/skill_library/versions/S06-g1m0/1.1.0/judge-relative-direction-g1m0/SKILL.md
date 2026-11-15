---
name: judge-relative-direction-g1m0
description: Method for object_rel_direction in initial evidence state g1m0.
---

Initial geometry_3d=available and metric_scale=unavailable.

# Judge direction from a specified location and heading

## Method

1. Extract `observer`, `facing_at`, and `target` separately, retaining the original options. Facing an object means the observer looks toward it; it does not refer to the furniture's own front.
2. Fix each role to a visible instance before geometry. Record `role=obj_id @ image_id + region` in a brief program comment. Check that candidate's box/mask itself, not another object in the same frame. Use modifiers and fixed structures to distinguish instances across rooms or sofas.
3. Choose `difficulty` from the option set: use `easy` for left/right only, `medium` for left/right/back, and `hard` for the four front/back and left/right quadrants.
4. With authorized geometry, valid local world conventions and usable bindings for the confirmed instances, call `relative_direction_of(observer_id, facing_at_id, target_id, difficulty)`. Check `direction`, `difficulty`, `world_up_used` and `handedness_used`. Never borrow another object's centroid for a missing role.
5. Map the returned direction strictly to the current options. easy permits only left/right; medium permits left/right/back; hard permits front-left, front-right, back-left, and back-right.
6. For visual reasoning, keep O, F and T in one local horizontal layout and compare O->F with O->T. For medium, back requires a turn >=135 degrees; opposite room ends alone do not establish it. Preserve the same instances across views; do not reuse camera left/right.

## Branches and checks

- Do not interchange `observer`, `facing_at`, and `target`, or treat `facing_at` as a furniture orientation property.
- Relative direction does not require metric scale; missing metric scale alone must not block this method.
- Without a valid world frame, up direction, or object grounding, use the visual path rather than fabricating a tool direction.
- Returned `difficulty` must match the option vocabulary. A direction absent from the options indicates incorrect arguments or mapping.
- If plausible instances change the option, or the layout is mirrored or near a boundary, inspect the distinguishing views before accepting a direction. Do not vote over candidates or select the first successful result.
- Before submission, check all three roles, difficulty vocabulary, and option mapping. Return a valid option for the current question.

## Candidate and failure checks

- M5 offers candidates and technical checks, not referent certification. Missing bindings use visual locations.
- list_objects exposes candidates; an empty list does not prove scene-wide absence. detect_objects can supplement frozen views when authorized, but new box-only records do not have 3D geometry or valid masks.
- object_visible_frames is independent of 3D capability and requires a readable, consistent mask. Check the original images when masks are unavailable; do not equate failed calls with absence.
- Failed calls, invalid payloads and invalidated result IDs cannot support the answer. Keep valid observations from feedback; do not repeat deterministically invalid calls unchanged.
- The method stays fixed for the whole episode. Use only the currently authorized tools, including after a shared capability is withdrawn; continue with supported visual reasoning when necessary.
- Match the question's object roles, scope, complete options and units. Visual estimates are not tool measurements. Submit the requested answer with its actual evidence basis and references.
