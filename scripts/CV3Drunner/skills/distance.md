# CV3D Distance: grounded object relations

Read the question to identify its reference object and both comparison objects.
Use the question's colors and official option order; no color has a fixed
reference role or answer letter. Start with visible relations: contact,
support, containment, separation and the layout of object surfaces.

If measuring, use pair_distance_in_boxes with the same reference surface point
for both pairs. a_uv,b_uv are normalized coordinates over each box's safe
interior. Select surfaces of the named objects, not the rectangle center by
default. In a support relation, inspect the supported object's surface and the
supporting surface near it. A large table box can include floor or another
object. A desk box can contain an occluding book. Check these possibilities.

The distance is Euclidean separation between local median MoGe2 camera points,
not minimum object-to-object distance, centroid distance or calibrated meters.
The 5x5 median reduces numerical noise without validating semantic membership.
Finite n_valid alone cannot establish that a patch belongs to the named object.

Yield new values with a short note describing the reference, colors, selected
surfaces and initial visual relation. Next round inspect the sampling overlay
and compare the two distances against those relations. If the numeric ordering
conflicts with clear contact or support, inspect the coordinates and surface
membership before using argmin. Re-sample a useful visible surface if needed,
or use the clearer visual relation with an explanation of the conflict.
Every new measurement needs another model request before answer acceptance.

If required boxes are missing or ambiguous, reason directly from the original
image. Do not invent box IDs or bypass the tools with raw pixel measurements.

## Final assessment
Choose the exact official option letter. Verify the reference, comparison
objects and color-to-option mapping. Inspect all sampled patches and explain
whether their predicted distances agree with visible contact, support and
separation. Use the comparison that best represents the question's objects.
