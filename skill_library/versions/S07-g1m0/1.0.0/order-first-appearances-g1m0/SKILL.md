---
name: order-first-appearances-g1m0
description: Method for obj_appearance_order in initial evidence state g1m0.
---

Initial geometry_3d=available and metric_scale=unavailable.

# Determine the order of first appearances

## Method

1. List all categories to order and the complete option sequences. Use frozen FrameSet slots 0..31 as the timeline; do not reorder time by filenames, interface layout, or video frames that were not provided.
2. Inspect original images and use `list_objects(category_filter)` to collect known instances of each category. Every instance of a category may provide earlier visible evidence.
3. When `object_visible_frames(obj_id)` is authorized, obtain an ascending visible-frame list for each instance. For each category, take the minimum frame slot across all nonempty instance lists as its earliest visibility under current evidence.
4. Record the earliest evidence for each category and the preceding frame range already checked. A tool's first detection is only an upper bound under current observations; it is not automatically the true first appearance.
5. Prioritize early frames and adjacent slots that could change the option ordering. For suspected missed categories, detect relevant frozen frames or inspect crops, and use clearer later views to verify partial early appearances.
6. Compare the categories' ordering constraints against each option. If order cannot be resolved within one sampling interval, do not invent subframe timestamps. Choose the complete valid sequence best supported by visible evidence.

## Branches and checks

- An empty `object_visible_frames` list means the instance has no valid visible mask; it does not mean the category never appears in the entire video.
- Detector failure, an unexecuted detector, or an empty return cannot prove earlier absence. Healthy empty detections constrain only inspected frames.
- Aggregate multiple instances of a category by earliest visibility; do not inspect only an arbitrary default instance.
- Repeated placeholder images, missing frames, and derived crops are not new real time points.
- Relative appearance order does not require 3D geometry, a world frame, or metric scale. Missing capabilities must not block visual ordering.
- Before submission, check category coverage, frame-slot chronology, and the complete option mapping. Do not fabricate observations outside the frozen set.

## Candidate and failure checks

- M5 provides candidates and technical validity checks. Even a unique category match does not prove the question's referent. Inspect frames/crops and select explicit object IDs.
- list_objects exposes candidates; an empty list does not prove scene-wide absence. detect_objects can supplement frozen views when authorized, but new box-only records do not have 3D geometry or valid masks.
- object_visible_frames is independent of 3D capability and requires a readable, consistent mask. Check the original images when masks are unavailable; do not equate failed calls with absence.
- Failed calls, invalid payloads and invalidated result IDs cannot support the answer. Keep valid observations from feedback; do not repeat deterministically invalid calls unchanged.
- The method stays fixed for the whole episode. Use only the currently authorized tools, including after a shared capability is withdrawn; continue with supported visual reasoning when necessary.
- Match the question's object roles, scope, complete options and units. Visual estimates are not tool measurements. Submit the requested answer with its actual evidence basis and references.
