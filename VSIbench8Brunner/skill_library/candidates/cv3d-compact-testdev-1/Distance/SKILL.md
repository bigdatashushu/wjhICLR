# Relative Object Distance

## 1. Ground the comparison
Map every annotation color to the current question's object, exact instance and option letter. Boxes identify regions, not measurement points. Match optional ctx.tools.view_candidates detections to the annotated instance; never substitute another object of the same category. Use original RGB and supplied observations to establish layout, perspective, support and occlusion. Answer visually when these resolve the comparison; measurement is optional.

## 2. Choose comparable surfaces
Use original-image pixels, x right and y down, in view 0; account for display resizing and check bounds. Choose a visible body patch whose entire 5x5 neighborhood belongs to that surface, away from box strokes, gaps, silhouettes, occluders and background. Box centers can hit empty space; finite samples do not certify object membership. If no faithful patch exists, reason visually.

Choose the reference once: use identical a_xy for both targets and a comparable main-body convention for both b_xy points. A sampled part of an extended object is not its hidden center or nearest extent. Do not compare a center-like patch with a favorable edge. Repairing the reference requires recomputing BOTH pairs.

## 3. Measure the relation
When measurement is useful, batch two ctx.tools.pair_point_distance calls with the same view and anchor. Accept status == 'ok'; compare finite, nonnegative data.distance values and retain each result_id. The smaller value supports the closer target under the chosen convention.

The distance uses local median predicted camera points from 5x5 neighborhoods: a visible predicted-surface proxy in per-view relative units, not object-center distance, complete minimum-surface distance or meter-calibrated ground truth. Camera-depth differences and pixel separation cannot replace Euclidean distance. Use ctx.tools.point_depth only for a necessary local diagnostic.

## 4. Check only consequential ambiguity
For questionable grounding or a close comparison, make a small, bounded sensitivity check. Choose alternative interior patches for surface reasons before seeing values. Keep the anchor and convention fixed; changing the anchor requires both pairs again. An ordering change is proxy instability. Do not sweep boxes, pick extrema, invent a centroid or switch to nearest-surface distance to obtain an answer.

## 5. Reconcile and finish
Check each sampled surface against RGB layout. If valid numbers conflict with strong visual evidence or unstable patches matter, YieldObservations their successful result IDs for model-level review of identity, surface membership and proxy adequacy. Do not discard a result because it opposes a preferred option. Reassess accumulated observations, then select the current option.

Emit Python def solve(ctx) using ctx.tools. Finish early with ReturnAnswer(AnswerPayload(value=selected_letter, unit='option', basis='mixed')) when measurements contribute, or basis='visual_estimate' for visual reasoning. Yield and Return terminate immediately. Variables vanish between rounds; use delivered observations, with no ReadObservation.

Use at most five programs plus one final JSON, six rounds total. At finalization emit only the required answer JSON with the selected option letter. Tool failure can mark the host untrusted and immediately end in run_error; never plan try-and-recover fallback. Use no fixed coordinates, tuned thresholds or category-specific answer defaults.
