---
name: infer-route-turns-g1m1
description: Method for route_planning in initial evidence state g1m1.
---

Initial geometry_3d=available and metric_scale=available. Metric tools still require question authorization.

# Determine turns along a specified route

## Read usable evidence

- Follow the current round's action format. Program rounds use Python and documented tool names; preserve reserved variables and do not invent `scene` methods or XML calls. When finalization requests only JSON, return only that JSON.
- A tool returns a dictionary envelope. Its decoded payload is `r["data"]`, and its ID is `r["result_id"]`. For example, object rows are `list_objects(category_filter=category)["data"]`. Do not use `.value`, decode `data` again, or read payload keys from the envelope root.
- Reuse valid result IDs. For omitted fields or rows, use `ReadObservation`; advance by its returned `next_cursor`, and confirm the union of read ranges covers the relevant records. A page may contain fewer rows than the requested limit. Print output is diagnostic; use `show` and `YieldObservations` to expose evidence for another round. Rebuild variables from retained observations after yielding.
- Check inner status, local binding diagnostics, and current authorization before using a value. A failed object binding stops that object's geometry branch. A duplicate warning requires identity checks; it is not a blanket reason to discard every candidate. Report the evidence actually used.

## Method

1. Expand the question into a start location, initial facing object, ordered landmarks, straight segments, and missing turns. Preserve each option's complete action sequence. Follow the specified route rather than replacing it with a shorter route.
2. Use original images and object records to confirm landmark instances and their order. For repeated categories, disambiguate using question descriptions, doorways, neighboring structures, and observation order. Revisiting the same location does not create a new landmark.
3. At each turn, construct a triple of previous position, current position, and next position. The arrival heading is inherited from the preceding segment; judge the next action relative to that heading. For the first turn at the start, use initial facing object minus start as the forward vector and first landmark minus start as the target vector. Reverse the initial heading when the question specifies facing away. A corner, a point beyond a landmark, or a stop after a stated travel distance is a path position; do not replace it automatically with an object centroid. Record the stop position and heading before evaluating the next turn.
4. When geometry is authorized, obtain coordinates from `object_centroid(obj_id)` using `centroid_normalized`, and read `world_up_used` and `handedness_used` from the same return. Compute only when all points share a coordinate frame and orientation metadata is valid and consistent. With unit up vector u, incoming f = current - previous, and outgoing g = next - current, first project onto the horizontal plane: f_h = f - dot(f,u)u and g_h = g - dot(g,u)u. Then compute the signed turn angle. Do not guess axes, handedness, or angles if either horizontal vector is near zero or orientation metadata is None.
5. The function below returns degrees: positive is right, negative is left, near zero is straight, and an absolute angle near 180 is a U-turn. Flip the sign for left handedness. For nearly collinear vectors, use the dot product to distinguish same from opposite directions rather than relying only on the cross-product sign. Interpret vocabulary, angle tolerances, and action sequences using the question and visual layout; do not reuse the direction task's medium 135-degree threshold.

```python
import numpy as np
import math

def route_turn_angle(previous, current, following, world_up, handedness):
    if world_up is None or handedness not in ("right", "left"):
        return None
    u = np.asarray(world_up, dtype=float)
    if u.shape != (3,) or not np.all(np.isfinite(u)) or np.linalg.norm(u) < 1e-9:
        return None
    u = u / np.linalg.norm(u)
    f = np.asarray(current, dtype=float) - np.asarray(previous, dtype=float)
    g = np.asarray(following, dtype=float) - np.asarray(current, dtype=float)
    if f.shape != (3,) or g.shape != (3,):
        return None
    f = f - np.dot(f, u) * u
    g = g - np.dot(g, u) * u
    if not np.all(np.isfinite(f)) or not np.all(np.isfinite(g)):
        return None
    if np.linalg.norm(f) < 1e-9 or np.linalg.norm(g) < 1e-9:
        return None
    f = f / np.linalg.norm(f)
    g = g / np.linalg.norm(g)
    side = float(np.dot(g, np.cross(f, u)))
    if handedness == "left":
        side = -side
    return math.degrees(math.atan2(side, float(np.dot(g, f))))
```

6. Use `connectivity_graph()` only to check supported reachability relations and impossible segments. Small Euclidean object distance, being in the same room, or graph adjacency alone does not prove traversability.
7. If geometry or world-orientation conventions are unavailable, move an imagined observer segment by segment along the route in the original-image layout. Check turns against walls, doorways, passages, and nearby objects. Update the heading after every segment.
8. Rehearse each candidate action sequence completely in the question's order, checking that its segments reach the specified landmarks and destination. Select the best-supported valid option.

## Branches and checks

- Relative turns do not require metric scale; missing metric scale alone must not block route judgment.
- In `relative_direction_of(observer_id=previous_position, facing_at_id=current_position, target_id=next_position)`, the observer is still at the previous position. This can help identify the side of a polyline, but cannot replace the turn angle at the current position, especially for U-turns.
- `connectivity_graph` requires a world frame and point cloud. Missing evidence or a failed call does not make unknown traversability equivalent to unreachable.
- Without clear turn points or traversable boundaries, a polyline of object centers is only a cue; interpret it together with the original-image layout.
- Before submission, check each action relative to the heading on arrival at that landmark and match the current question's complete action-sequence options.

## Candidate and failure checks

- M5 provides candidates and technical validity checks. Even a unique category match does not prove the question's referent. Inspect frames/crops and select explicit object IDs.
- list_objects exposes candidates; an empty list does not prove scene-wide absence. detect_objects can supplement frozen views when authorized, but new box-only records do not have 3D geometry or valid masks.
- object_visible_frames is independent of 3D capability and requires a readable, consistent mask. Check the original images when masks are unavailable; do not equate failed calls with absence.
- Failed calls, invalid payloads and invalidated result IDs cannot support the answer. Keep valid observations from feedback; do not repeat deterministically invalid calls unchanged.
- The method stays fixed for the whole episode. Use only the currently authorized tools, including after a shared capability is withdrawn; continue with supported visual reasoning when necessary.
- Match the question's object roles, scope, complete options and units. Visual estimates are not tool measurements. Submit the requested answer with its actual evidence basis and references.
