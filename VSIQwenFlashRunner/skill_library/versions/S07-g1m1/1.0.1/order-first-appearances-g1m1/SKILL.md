---
name: order-first-appearances-g1m1
description: Method for obj_appearance_order in initial evidence state g1m1.
---

Initial geometry_3d=available and metric_scale=available. Metric tools still require question authorization.

# Determine the order of first appearances

## Read usable evidence

- Follow the current round's action format. Program rounds use Python and documented tool names; preserve reserved variables and do not invent `scene` methods or XML calls. When finalization requests only JSON, return only that JSON.
- A tool returns a dictionary envelope. Its decoded payload is `r["data"]`, and its ID is `r["result_id"]`. For example, object rows are `list_objects(category_filter=category)["data"]`. Do not use `.value`, decode `data` again, or read payload keys from the envelope root.
- Reuse valid result IDs. For omitted fields or rows, use `ReadObservation`; advance by its returned `next_cursor`, and confirm the union of read ranges covers the relevant records. A page may contain fewer rows than the requested limit. Print output is diagnostic; use `show` and `YieldObservations` to expose evidence for another round. Rebuild variables from retained observations after yielding.
- Check inner status, local binding diagnostics, and current authorization before using a value. A failed object binding stops that object's geometry branch. A duplicate warning requires identity checks; it is not a blanket reason to discard every candidate. Report the evidence actually used.

## Method

1. List all categories to order and the complete option sequences. Use frozen FrameSet slots 0..31 as the timeline; do not reorder time by filenames, interface layout, or video frames that were not provided.
2. Inspect original images and use `list_objects(category_filter)` to collect known instances of each category. Every instance of a category may provide earlier visible evidence.
3. When `object_visible_frames(obj_id)` is authorized, obtain an ascending visible-frame list for each instance. For each category, take the minimum frame slot across all nonempty instance lists as its earliest visibility under current evidence.
   A geometry duplicate flag is not a reason to discard earlier category-visible pixels. Verify the category and include its temporal evidence when the mask is usable. Missing categories remain unknown; do not insert infinity and stable-sort it into a complete sequence.
4. Record the earliest evidence for each category and the preceding frame range already checked. A tool's first detection is only an upper bound under current observations; it is not automatically the true first appearance.
5. Prioritize early frames and adjacent slots that could change the option ordering. For suspected missed categories, detect relevant frozen frames or inspect crops, and use clearer later views to verify partial early appearances.
6. Compare the categories' ordering constraints against each option. Equal earliest slots remain tied: do not order them by object ID, dictionary order, or insertion order. Keep a partial order, inspect earlier relevant original frames when available, and choose the complete option that best satisfies the supported relations. Do not invent subframe timestamps. If a computed sequence matches no option, do not default to a letter. Any visual correction must identify the changed relation and its visible basis; rematch the entire sequence before returning the answer.

## Branches and checks

- An empty `object_visible_frames` list means the instance has no valid visible mask; it does not mean the category never appears in the entire video.
- Detector failure, an unexecuted detector, or an empty return cannot prove earlier absence. Healthy empty detections constrain only inspected frames.
- Check inner detector status before using detections. A fault must not enter time minima, sorting, or supporting result IDs. Stop unchanged calls to an unconfigured service. If original frames conflict with recorded visibility, state which relation is corrected and the visible basis; do not silently relabel a mask's first slot.
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
