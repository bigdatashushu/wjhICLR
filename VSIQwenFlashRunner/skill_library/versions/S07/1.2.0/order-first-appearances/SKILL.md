---
name: order-first-appearances
description: Use for obj_appearance_order to compare earliest visible evidence by frozen frame-slot order, distinguishing actual first appearance, first detection, missed detections, and missing valid masks.
---

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
