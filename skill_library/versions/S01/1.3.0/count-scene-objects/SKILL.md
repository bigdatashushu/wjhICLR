---
name: count-scene-objects
description: Use for object_counting to count a target category using cross-frame visibility, instance identity, and tool deduplication, accounting for occlusion, missed detections, duplicate tracks, and empty detections.
---

# Count distinct objects across frames

## Method

1. Extract the target category, spatial scope, and inclusion or exclusion relations from the question. Distinguish whole objects from parts; do not count an object's attachments as additional instances.
2. Inspect the original images and organize target candidates. `list_objects(category_filter)` returns object records containing `obj_id`, `category_name`, `visible_frames`, `track_id`, and `duplicate_suspect`. The record count is only the candidate count, not the final answer.
3. Build an instance table for this question with each candidate's co-occurrence frames, track, neighboring structures, and appearance. Require multiple identity cues before merging across frames. Keep similar objects visible simultaneously in one frame as separate instances; appearance similarity alone does not justify merging.
4. When `count_objects(category_name)` is authorized, read `count`, `n_records`, `n_distinct_tracks`, `n_geometric_merges`, `n_without_track`, `duplicate_suspect`, and `evidence_degraded`. The tool first groups records with the same nonempty track; untracked records each form their own group. It then deduplicates groups by geometric overlap. A group without a point cloud still counts once. `count = n_distinct_tracks - n_geometric_merges`; it is not the raw record count. Cross-check against the instance table and coverage in the original images.
5. If omissions could change the count, call `detect_objects` for the target category and relevant frozen frames. Distinguish `status="fault"`, `status="empty"`, and `status="ok"`, and check `error` for partial frame failures. New detections are 2D candidates added to this question's inventory; they do not automatically acquire 3D grounding or cross-frame tracks. Do not sum detections of the same object across frames or add them directly to the old `count`. Recheck identities and duplicates using co-occurrence, appearance, and position.
6. Resolve specific conflicts between tool results and visual observations, then choose the best-supported non-negative integer. A zero count must be supported by valid observations or healthy empty detections.

## Branches and checks

- If `count_objects` is unavailable or evidence is degraded, continue using the original images and instance table; do not equate candidate records with the answer.
- A detector fault, an unexecuted tool, or an invalid return does not imply zero. Healthy empty detections establish only that the specified frames yielded no detection, not that the object is absent from the entire video.
- Reappearance after occlusion, fragmented tracks, and duplicate detections may overestimate the count. Limited frame coverage and fully invisible instances may underestimate it.
- Geometric positions can assist deduplication only with valid evidence and grounding. Do not present 2D detection boxes as 3D instances.
- Before submission, check that each entity is counted once across frames, similar but distinct instances have not been merged, and the result is the non-negative integer requested by the question.
