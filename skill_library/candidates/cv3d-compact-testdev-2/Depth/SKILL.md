# Compare camera-depth

## Locate the targets
Match each named instance and colored box in RGB to its option letter. Use occlusion and scene layout to understand the objects; image height and apparent size alone can mislead. A box includes empty space and may not follow the object's surface.

## Read the aligned depth image
Locate the same visible surfaces in the depth image, when supplied. Compare their interior depth patterns with the RGB layout. Ignore invalid pixels and annotation strokes. Similar or saturated shades can hide differences; visible depth is a prediction, not object identity.

## Measure useful uncertainty
When the images establish a clear relationship, select the corresponding option. Otherwise use point_depth on a few separated interiors of each object. Keep the whole sampling neighborhood on the visible surface, away from silhouettes, gaps, support surfaces and occluders. Do not replace a thin or hidden target with nearby background. Whole-view statistics and detections can guide availability or location, not establish a target's depth.

Compare representative groups using medians and spread. Keep the chosen part of an extended object interpretable; sparse surface samples are not object centroids or mask averages. Choose patches for visible surface support before seeing their values, rather than search for favorable extrema. If ambiguity remains, recheck target identity and sample a newly justified patch only if it can resolve the comparison.

## Final assessment
Identify the specified instances in RGB and the aligned depth image. Use the available measurements to compare representative visible surfaces: smaller camera-depth means nearer. If measured groups overlap or conflict with the images, assess target correspondence, surface membership and prediction stability using existing evidence. Weigh uncertain measurements without silently discarding inconvenient results. Map the final object comparison to the supplied option letter.
