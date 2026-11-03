# CV3D Depth: grounded surface comparison

Identify the two objects named by the question's colored boxes and preserve the
official option order. Start with a visual depth ordering from occlusion,
perspective and scene layout. The host catalog binds each box_id to a color;
it does not assign the answer letter or validate the object's visible surface.

When measurement helps, choose visible surface points inside the safe interior
of each question box using point_depth_in_box. u,v range from 0 to 1 over
safe_xyxy. Avoid annotation strokes, holes, background and occluding objects.
A box center is only a candidate point. Measurements are predicted camera Z
with relative scale; compare only this view. A 5x5 patch's n_valid describes
numerical support, not semantic membership.

Yield new measurements with a brief note naming the objects, colors, point
choices and initial visual ordering. Review them next round with the sampling
overlay. Confirm which surface each white patch samples. Compare the measured
ordering with the visible scene. If they disagree, inspect the surface mapping
and sample another visible point when useful. Prefer a clear scene relation
when the samples lie on background or an occluder. Every new measurement needs
another model request before the host accepts an answer.

If required boxes are missing or ambiguous, use the original image. Do not
invent box IDs or switch to raw pixel tools.

## Final assessment
Choose the exact official option letter. Recheck color-to-object and
object-to-option mappings. When measurements exist, inspect the overlay and
explain their agreement or conflict with the visual ordering. Distinguish
finite depth support from surface membership and calibrated physical scale.
Resolve the comparison using the best visible and measured evidence.
