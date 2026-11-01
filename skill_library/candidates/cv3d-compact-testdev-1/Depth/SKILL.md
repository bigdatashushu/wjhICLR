# CV-Bench3D Depth

## Ground the Targets
Read the question and options. Bind each named target to its original colored box and supplied option letter; color and mention order do not fix the letter. Compare those instances, not another object with the same category. Form a visual hypothesis from occlusion and scene layout. Use apparent size and image height in context, accounting for pose, elevation, camera tilt and hidden parts.

Use the original RGB pixel grid, x right and y down, with the supplied image dimensions. Convert any preview coordinates explicitly and verify them against the original box and recognizable features. A box is not an object mask. Candidate boxes are non-exhaustive localization hints: match them visually to the annotated instance before using them, even when their category is confirmed.

## Choose Useful Measurements
Answer early from RGB when the relationship is clear or usable surface evidence is unavailable; measurement is not compulsory. Otherwise choose a small, purposeful set of point_depth samples in the same view. Check bounds, known availability and surface support before calling. Use an available view summary only to resolve an availability question, not as target evidence. A failed tool may mark the host untrusted and immediately end with run_error; catching it does not enable recovery.

Pick separated, representative visible interiors of both targets. Check the full 5x5 neighborhood around the rounded pixel, not just its center. Keep it on the target's actual visible surface, away from annotation borders, silhouettes, gaps, supporting surfaces and foreground occluders. For thin or tiny targets, a pure neighborhood may not exist. Do not replace a target with nearby background. If safer target interiors still give implausible predictions, assess the depth proxy rather than mechanically forcing a numeric choice.

## Interpret and Compare
Generate Python def solve(ctx), using ctx.tools and the host's documented API. For successful point_depth responses, status is 'ok'; read data.depth, n_valid and pixel_xy, and retain the response result_id. Check the returned location and numerical support. Even n_valid covering the whole patch certifies predicted support, not target identity or accurate geometry.

Smaller predicted camera-depth means nearer sampled surfaces within one view. It is not metric distance or ground truth. Compare modest, visually justified sample groups with robust summaries such as medians and their spread. Keep distinct parts of an extended object interpretable; sparse samples do not establish its mask mean or 3D centroid. Do not change the comparison criterion after seeing numbers, discard points solely for disagreeing, or seek an extreme minimum.

Let usable ordering select the corresponding option. When groups overlap, ordering depends on a questionable point, or measurements conflict with RGB layout, YieldObservations with the relevant existing successful result_ids so the model can inspect the accumulated observations against RGB. Recheck coordinates and surface membership first, then proxy reliability. Resample only newly justified interiors when useful; otherwise decide visually without claiming rejected measurements support the answer.

## Finish
YieldObservations and ReturnAnswer terminate immediately. Python variables disappear between solve rounds; redefine needed state from the host-delivered observations. The total budget is six rounds: at most five programs and one final JSON, with early answers allowed. Return the mapped supplied letter through ReturnAnswer and AnswerPayload, unit='option', basis='mixed' for genuine visual and tool support or 'visual_estimate' for an image-based choice. In finalization emit only the required answer JSON using accumulated evidence.
