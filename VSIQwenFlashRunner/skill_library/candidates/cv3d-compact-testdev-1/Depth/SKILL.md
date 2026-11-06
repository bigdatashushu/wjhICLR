# CV-Bench3D Depth

## Ground the Targets
Bind each named target to its original colored box and supplied option letter; color and mention order do not fix the letter. Compare those instances. Form a visual hypothesis from occlusion and scene layout. Interpret size and image height with pose, elevation, camera tilt and hidden parts.

Use the original RGB grid and dimensions, x right and y down. Convert preview coordinates explicitly; verify them against the original box and recognizable features. A box is not a mask. Candidate boxes are non-exhaustive hints: match them visually to the annotated instance, even when their category is confirmed.

## Choose Useful Measurements
Answer early from RGB when the relationship is clear or usable surface evidence is unavailable; measurement is optional. Otherwise choose a small set of point_depth samples in one view. Check bounds, known availability and surface support first. Use a view summary to resolve availability, not as target evidence. A failed tool may mark the host untrusted and immediately end with run_error; catching it does not enable recovery.

Pick separated, representative interiors of both targets. Check the full 5x5 neighborhood around the rounded pixel. Keep it on the actual visible surface, away from annotation borders, silhouettes, gaps, supporting surfaces and occluders. Thin or tiny targets may lack a pure neighborhood; nearby background is not a substitute. If safer interiors still give implausible predictions, assess the proxy rather than force a numeric choice.

## Interpret and Compare
Generate Python def solve(ctx), using ctx.tools and the documented API. Successful point_depth has status='ok'; read data.depth, data.n_valid and data.pixel_xy, and retain result_id. Check location and numerical support. A full n_valid certifies predicted support, not target identity or accurate geometry.

Smaller predicted camera-depth means nearer surfaces within one view, not metric distance or ground truth. Compare visually justified groups using medians and spread. Keep distinct parts of extended objects interpretable; sparse samples do not establish mask means or 3D centroids. Do not change criteria after seeing numbers, discard points solely for disagreeing, or seek an extreme minimum.

Let usable ordering select the corresponding option. For overlapping groups, questionable points or RGB conflicts, YieldObservations with relevant existing successful result_ids for model inspection against RGB. Recheck coordinates and surface membership, then proxy reliability. Resample newly justified interiors only when useful; otherwise decide visually without claiming rejected measurements support the answer.

## Finish
YieldObservations and ReturnAnswer terminate immediately. Python variables disappear between rounds; redefine state from host-delivered observations. Six rounds allow at most five programs and one final JSON, or an early answer. Return the mapped letter via ReturnAnswer and AnswerPayload, unit='option', basis='mixed' for genuine visual and tool support or 'visual_estimate' for image-based choice. Finalization emits only the required answer JSON from accumulated evidence.
