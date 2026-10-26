---
name: estimate-object-size-g1m0
description: Method for object_size_estimation in initial evidence state g1m0.
---

Initial geometry_3d=available and metric_scale=unavailable.

# Use shape evidence without fabricating metric size

## Method

Disambiguate the complete target and inspect its outline and principal axes across images. Available object_centroid results may help identify its reconstructed position, but a centroid does not measure object size. object_3d_extent requires metric authorization and is not available in this initial state. Determine the longest complete dimension visually and give an explicit visual estimate in the requested unit. Do not convert normalized geometry to centimeters or write a visual scale guess into the tool path.

## Task decision sheet

First define the complete target using surrounding context: an inset appliance, its visible panel, and the entire surrounding fixture can have different boundaries. Keep the intended object and any excluded surrounding structure distinct in a working sheet. Do not settle this boundary from a category's typical size alone.

For length, width and height, note supporting frozen slots, visible endpoints, omitted or occluded portions, and perspective shortening. Prefer a view or crop that retains the complete outline and useful context. A crop containing only a prominent part cannot establish the whole object's maximum dimension. Select the longest physical axis using these observations before estimating its value.

Use a reference size only with an explicit uncertain range and a plausible depth relationship. Keep category priors separate from observations of this instance. Compare any authorized extent result against the same complete-object scope; partial or rotated point sets retain their stated limitations. Convert the selected dimension once into the requested unit, and keep a visual estimate separate from a metric tool measurement.

## Candidate and failure checks

- M5 provides candidates and technical validity checks. Even a unique category match does not prove the question's referent. Inspect frames/crops and select explicit object IDs.
- list_objects exposes candidates; an empty list does not prove scene-wide absence. detect_objects can supplement frozen views when authorized, but new box-only records do not have 3D geometry or valid masks.
- object_visible_frames is independent of 3D capability and requires a readable, consistent mask. Check the original images when masks are unavailable; do not equate failed calls with absence.
- Failed calls, invalid payloads and invalidated result IDs cannot support the answer. Keep valid observations from feedback; do not repeat deterministically invalid calls unchanged.
- The method stays fixed for the whole episode. Use only the currently authorized tools, including after a shared capability is withdrawn; continue with supported visual reasoning when necessary.
- Match the question's object roles, scope, complete options and units. Visual estimates are not tool measurements. Submit the requested answer with its actual evidence basis and references.
