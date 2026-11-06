# Compare distance from a reference object

## Locate the comparison
Identify the reference object and both candidate instances in RGB, then map candidates to the supplied option letters. Bind colors to this question rather than assume a fixed color-to-letter rule. Use layout, perspective, support and occlusion to understand their positions.

## Select comparable surfaces
Use the aligned depth image, when supplied, to identify depth layers, invalid regions and surface boundaries. Its grayscale represents camera-depth, not distance from the reference object; grayscale differences and pixel separation cannot replace a 3D distance comparison.

Choose a visible body patch for each object. Keep the whole sampling neighborhood away from box strokes, silhouettes, gaps, occluders and background. Candidate boxes may help localization but do not certify the sampled surface. If no faithful patch exists, judge the relation from the images rather than substitute an unrelated surface.

## Measure the relation
If the images already resolve the relation, choose the option. Otherwise use pair_point_distance for the reference-to-A and reference-to-B pairs, with identical reference coordinates and a comparable surface convention for both targets. Smaller Euclidean distance supports the closer target under that convention. A visible patch is not an object's hidden center or its complete nearest extent. Use point_depth only for a specific depth diagnostic.

## Check consequential ambiguity
For a close or visually inconsistent comparison, examine a small number of alternative interior patches chosen for surface reasons. Keep the reference fixed; changing it requires recomputing both pairs. An ordering change indicates sensitivity to the surface proxy. Reconcile this with both images and existing measurements, rather than sweep boxes, choose extrema or change the distance definition to obtain a preferred answer. Return the option corresponding to the final object comparison.

## Final assessment
Use both images and the collected evidence to compare the reference-to-target relations. Give comparable pairs with the same reference point priority; account for surface-proxy instability. Camera-depth or grayscale differences alone do not establish the required 3D distance ordering. Select the option matching the resulting comparison of the specified instances.
