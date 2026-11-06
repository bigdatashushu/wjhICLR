---
name: rank-object-distances
description: Use for object_rel_distance to compare distances from the named reference object to candidate categories, handling multiple instances, occlusion, missed detections, and ambiguous grounding without metric scale.
---

# Reason about relative distances between objects

## Method

1. Extract the reference object, candidate categories, and original option mapping. The reference is the object named in the question, not the camera. The candidate list must cover all options.
2. Establish the reference and candidate instances. If the reference category has multiple objects, use the question description, co-occurrence structure, and visible frames to select the specific instance, preferably passing its `obj_id` to distance tools. Do not rely on a default ordering when a category matches multiple instances. A candidate category may contain multiple instances; compare its instance nearest to the reference as required by the question.
3. When geometric tools are authorized, call `relative_distance_rank(reference, candidate_categories)`. Check that returned `reference` matches the intended instance, and inspect `ranking`, `closest_category`, `categories_without_detection`, `definition`, and `degradation_flags`.
4. Rank using `distance_normalized` values from the same call. A shared scale cancels in relative ranking, so metric scale is unnecessary. Do not fabricate scale to obtain meters or substitute camera depth or 2D pixel gaps for object-to-object distances.
5. Focus on incorrect bindings, point-set contamination, occlusion, duplicate instances, and coverage gaps that could change the top two candidates. When additional evidence is needed, prioritize frozen frames or candidate categories directly relevant to these ambiguities.
6. Map the nearest category back to a valid option in the current question. Recheck reference identity, candidate coverage, distance definition, and option mapping before submission.

## Branches and checks

- Categories in `categories_without_detection` are unknown, not infinitely distant, and must not be silently removed. Detect them when available; otherwise retain incomplete coverage as uncertainty in the final comparison.
- An unauthorized tool, failed execution, empty ranking, or invalidated result does not imply zero distance or object absence. Compare visually using shared structures, occlusion, relative depth, and local layouts in the original images.
- The result uses a robust low-quantile surface-distance proxy. Do not relabel it as exact nearest-point distance, centroid distance, or official 3D ground truth.
- 2D image gaps are only visual cues. Do not rank directly by pixel gaps when candidates differ in depth or perspective distortion is substantial.
- Aggregate candidate instances of the same category by the nearest instance. Disambiguate multiple instances of the reference category first. These are distinct rules for multiple instances.
- The final answer must use the current option mapping and agree with the valid tool results or visual evidence actually used.
