---
name: count-scene-objects-g1m1
description: Method for object_counting in initial evidence state g1m1.
---

Initial geometry_3d=available and metric_scale=available. Metric tools still require question authorization.

# Count distinct objects across frames

## Read usable evidence

- Follow the current round's action format. Program rounds use Python and documented tool names; preserve reserved variables and do not invent `scene` methods or XML calls. When finalization requests only JSON, return only that JSON.
- A tool returns a dictionary envelope. Its decoded payload is `r["data"]`, and its ID is `r["result_id"]`. For example, object rows are `list_objects(category_filter=category)["data"]`. Do not use `.value`, decode `data` again, or read payload keys from the envelope root.
- Reuse valid result IDs. For omitted fields or rows, use `ReadObservation`; advance by its returned `next_cursor`, and confirm the union of read ranges covers the relevant records. A page may contain fewer rows than the requested limit. Print output is diagnostic; use `show` and `YieldObservations` to expose evidence for another round. Rebuild variables from retained observations after yielding.
- Check inner status, local binding diagnostics, and current authorization before using a value. A failed object binding stops that object's geometry branch. A duplicate warning requires identity checks; it is not a blanket reason to discard every candidate. Report the evidence actually used.

## Method

1. Extract the target category, spatial scope, and inclusion or exclusion relations from the question. Distinguish whole objects from parts; do not count an object's attachments as additional instances.
2. Inspect the original images and organize target candidates. `list_objects(category_filter)` returns object records containing `obj_id`, `category_name`, `visible_frames`, `track_id`, and `duplicate_suspect`. The record count is only the candidate count, not the final answer.
3. Build an instance table for this question with each candidate's co-occurrence frames, track, neighboring structures, and appearance. Require multiple identity cues before merging across frames. Keep simultaneously visible, spatially separate whole objects distinct. Co-occurring records may still be overlapping masks or part-whole splits; distinguish those from separate objects before grouping. Appearance similarity alone does not justify merging. Read frame/category values from retained observations instead of reconstructing them from memory.
4. When `count_objects(category_name)` is authorized and collection binding is usable, read `count`, `n_records`, `n_distinct_tracks`, `n_geometric_merges`, `n_without_track`, `diagnostic_warnings`, and `instance_consolidation.method`. The tool groups records with the same nonempty track; untracked records form their own groups. Under `verified_m5_instances`, geometric merging is skipped and the count relies on the established instance bindings. The legacy `track_then_bidirectional_nn_pointcloud_overlap_v11` branch can merge track groups geometrically. Read the actual method instead of assuming the legacy branch ran. `count = n_distinct_tracks - n_geometric_merges`; cross-check it against the instance table and original images.
   Different track IDs alone do not prove distinct physical entities. If the collection precondition rejects unavailable bindings, no count was produced: do not infer its hypothetical value. Continue the question-scoped instance table from actual visual identity and inclusion evidence.
5. If omissions could change the count, call `detect_objects` for the target category and relevant frozen frames. Distinguish `status="fault"`, `status="empty"`, and `status="ok"`, and check `error` for partial frame failures. New detections are 2D candidates added to this question's inventory; they do not automatically acquire 3D grounding or cross-frame tracks. Do not sum detections of the same object across frames or add them directly to the old `count`. Recheck identities and duplicates using co-occurrence, appearance, and position.
6. Resolve specific conflicts between tool results and visual observations, then choose the best-supported non-negative integer. A zero count must be supported by valid observations or healthy empty detections.

## Branches and checks

- If `count_objects` is unavailable or its payload is unusable, continue using the original images and instance table; do not equate candidate records with the answer.
- A detector fault, an unexecuted tool, or an invalid return does not imply zero. Healthy empty detections establish only that the specified frames yielded no detection, not that the object is absent from the entire video.
- If detector feedback establishes an unconfigured or unhealthy service, stop that detection branch and continue from usable original frames and retained observations. More calls to the same unavailable service cannot fill coverage.
- Reappearance after occlusion, fragmented tracks, and duplicate detections may overestimate the count. Limited frame coverage and fully invisible instances may underestimate it.
- Geometric positions can assist deduplication only with valid evidence and grounding. Do not present 2D detection boxes as 3D instances.
- Before submission, check that each entity is counted once across frames, similar but distinct instances have not been merged, and the result is the non-negative integer requested by the question.

## Candidate and failure checks

- M5 provides candidates and technical validity checks. Even a unique category match does not prove the question's referent. Inspect frames/crops and select explicit object IDs.
- list_objects exposes candidates; an empty list does not prove scene-wide absence. detect_objects can supplement frozen views when authorized, but new box-only records do not have 3D geometry or valid masks.
- object_visible_frames is independent of 3D capability and requires a readable, consistent mask. Check the original images when masks are unavailable; do not equate failed calls with absence.
- Failed calls, invalid payloads and invalidated result IDs cannot support the answer. Keep valid observations from feedback; do not repeat deterministically invalid calls unchanged.
- The method stays fixed for the whole episode. Use only the currently authorized tools, including after a shared capability is withdrawn; continue with supported visual reasoning when necessary.
- Match the question's object roles, scope, complete options and units. Visual estimates are not tool measurements. Submit the requested answer with its actual evidence basis and references.
