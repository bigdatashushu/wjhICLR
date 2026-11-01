# Relative Object Distance

## 1. Ground the comparison
Read the current question and options. Map each annotation color to its named object, exact visible instance, and answer letter. Boxes identify regions, not measurement points. Match optional ctx.tools.view_candidates detections to the annotated instance; a category match never licenses substituting another object. Use the original RGB and supplied observations to establish scene layout, perspective, visible support and occlusion. Answer visually when these resolve the comparison; measurement is not mandatory.

## 2. Choose comparable surfaces
Use original-image pixels, x right and y down, in view 0. Account for display resizing and check image bounds before querying. Select an explainable interior patch on each object's visible body, with its entire 5x5 neighborhood on that surface. Avoid box strokes, gaps, silhouettes, occluders and background. A box or detector center can land on empty space; being inside a box or returning finite samples does not establish object membership. If no faithful patch is visible, use visual reasoning.

Choose the reference anchor once. Use exactly the same a_xy for both targets, and a comparable main-body representative-point convention for both b_xy choices. For extended objects, distinguish the sampled part from the object's hidden center or nearest extent. Do not compare one target's center-like patch with the other's favorable edge. Repairing the reference requires recomputing BOTH pairs.

## 3. Measure the relation
When geometry can resolve an uncertainty, batch two ctx.tools.pair_point_distance calls with the same view and anchor. Accept status == 'ok'; read finite, nonnegative data.distance values and retain each result_id. The smaller value supports the closer target under the selected convention.

This is Euclidean distance between local median predicted camera points from 5x5 neighborhoods: a visible predicted-surface proxy in per-view relative units, not object-center distance, complete minimum-surface distance, or meter-calibrated ground truth. Neither camera-depth differences nor pixel separation replaces Euclidean distance. Use ctx.tools.point_depth only for a necessary local depth diagnostic, not as the primary comparison.

## 4. Check only consequential ambiguity
For questionable grounding or a close comparison, make a small, bounded sensitivity check. Choose alternative interior patches for visible-surface reasons before seeing their values. Keep the reference fixed; if changing it, remeasure both targets. Preserve the same representative convention and report an ordering change as proxy instability. Do not sweep boxes, pick extrema, infer a centroid from isolated samples, or switch to nearest-surface distance to obtain an answer.

## 5. Reconcile and finish
Check what each number actually sampled against RGB layout. If valid measurements conflict with strong visual evidence or depend on unstable patches, YieldObservations their successful result IDs for model-level reinspection of instance identity, surface membership and proxy adequacy. Do not declare geometry ground truth or discard a result merely because it opposes the preferred option. Reassess using accumulated observations, then choose the current option label.

Emit Python def solve(ctx) using ctx.tools. Finish early with ReturnAnswer(AnswerPayload(value=selected_letter, unit='option', basis='mixed')) when measurements contribute, or basis='visual_estimate' for a visual decision. Yield and Return terminate immediately; put all required work before them. Variables do not persist between rounds; use delivered observations, with no ReadObservation.

The total budget is six rounds: at most five programs plus one final JSON. At finalization emit only the required answer JSON with the selected option letter. Tool failure can mark the host untrusted and immediately end in run_error; never plan try-and-recover fallback. Ground choices in the current task, without fixed coordinates, tuned thresholds or category-specific answer defaults.
