"""Current neutral Tool interface descriptions; no task-specific recipes."""

TOOL_DOCS_VERSION = "tool-docs-v11.5-core-engineering-repair"


DESCRIPTIONS = {
    "list_objects": (
        "Return object records as list[dict] with obj_id, category_name, visible_frames, track_id, "
        "source_label, binding_provenance, det_conf, det_conf_source, mask_support_ratio, "
        "binding_flags, grounding_status, and "
        "duplicate_suspect. det_conf is detector confidence; mask_support_ratio is the fraction of "
        "frozen frames with a nonempty propagated mask. Never substitute one for the other. "
        "An empty filter returns all records. "
        "Category filtering matches complete categories after case, separator, number, and synonym "
        "normalization, optionally removing registered appearance prefixes. It does not use substring "
        "or word-intersection matching. obj_id does not encode category. Interfaces requiring a unique "
        "object raise an ambiguity error for categories with multiple instances. Records may include "
        "2D-only candidates without measurable 3D geometry. An empty result does not prove scene-wide "
        "absence. Inspect all candidates and binding_flags; do not select the first record by default "
        "or align records with detections by list order. A semantic_label_mask_conflict blocks geometry "
        "from that object but leaves frozen-image inspection available. "
        "The conflict requires high mask overlap and explicitly incompatible complete labels, "
        "with at least one question-targeted binding. Compatible and unknown overlap relations "
        "remain audit diagnostics; they do not establish semantic correctness or merge instances."),
    "count_objects": (
        "Count instances using the same category rules as list_objects. Group records sharing a "
        "nonempty track_id; each untracked record forms its own group. Merge each group's point clouds, "
        "then consolidate duplicate groups using bidirectional nearest-neighbor overlap. Retain groups "
        "without point clouds once. Return dict: count (final group count), n_records, n_distinct_tracks "
        "(groups before geometric merging, including untracked records), n_geometric_merges, "
        "n_without_track, duplicate_suspect, evidence_degraded, and instance_consolidation "
        "(method, min_overlap, error)."),
    "object_centroid": (
        "Return dict: centroid_normalized (3D centroid in world coordinates, normalized units), "
        "centroid_metric (3D coordinates in meters, or None without metric authorization), "
        "category_name, track_id, world_up_used (world-space unit up vector), and handedness_used "
        "(right/left). The last two fields are None when world-frame evidence is unavailable or "
        "orientation metadata is invalid; the centroid remains usable."),
    "object_3d_extent": (
        "Return dict: extent_normalized (per-axis lengths [dx,dy,dz] in normalized units), "
        "extent_metric (per-axis lengths [dx,dy,dz] in meters), n_valid_points, and degradation_flags. "
        "The point-cloud branch also returns extent_longest_metric. Without point clouds, a bbox "
        "fallback may be used with bbox_fallback_no_pointcloud. Raise an error if valid scale "
        "or measurable geometry is unavailable."),
    "plane_fit_room_size": (
        "Fit floor/wall planes to the scene point cloud. Return dict: room_diagonal_normalized, "
        "room_area_m2 (square meters), plane_inlier_ratio (ground inliers/all points), "
        "floor_fit_ratio (ground inliers/near-floor candidates), n_near_floor, fit_quality ([0,1], "
        "sqrt(floor_fit_ratio*ground_planarity)), fit_quality_formula_version, degradation_flags, "
        "measurement_source, and "
        "prepared_receipt_sha256. measurement_source=prepared_verified_cache means paired arms read "
        "the same identity-verified preparation result. plane_fit_low_quality remains an uncalibrated "
        "audit flag, not an automatic rejection rule. Raise an error if no finite area is available."),
    "robust_distance": (
        "Low-quantile nearest-neighbor distance from the reference point to the target object's "
        "point set. reference=camera/observer/agent/self/camera_center/camera0 uses the first-frame "
        "camera center; otherwise it uses an object centroid. target is an object. Return dict: "
        "distance_normalized, distance_metric (meters, or None without authorization), scale_version, "
        "quantile_q, voxel_size, conf_warp_version, n_valid_points, and degradation_flags. "
        "Distances may be None when there are too few points."),
    "camera_object_distance": (
        "Low-quantile distance from the first-frame camera center to an object's point set. "
        "Return dict: distance_normalized, distance_metric (meters), scale_version, quantile_q, "
        "voxel_size, conf_warp_version, n_valid_points, and degradation_flags. Requires metric "
        "authorization; distances may be None when there are too few points."),
    "surface_distance_between_objects": (
        "Low quantile of bidirectional nearest-neighbor distances between two distinct objects' "
        "point sets, used as a nearest-surface-distance proxy. Return dict: surface_distance_normalized, "
        "surface_distance_metric (meters, or None without authorization), equivalent aliases "
        "distance_normalized/distance_metric, n_nn_samples, n_valid_points, point_contamination_suspect, "
        "quantile_q, voxel_size, and degradation_flags. Distances may be None with too few points."),
    "relative_distance_rank": (
        "reference is an object ID or uniquely resolvable category. candidate_categories must include "
        "the categories of all question options, with at least two categories and no duplicates or "
        "synonymous categories. For each category, select the instance with the smallest surface-distance "
        "proxy to the reference, using a low quantile of bidirectional nearest-neighbor point-set distances. "
        "Return dict: reference (obj_id/category_name), ranking (ascending distance records with category, "
        "obj_id, distance_normalized, n_instances_in_category), status "
        "(ok/incomplete_candidates/uncertain_geometry/ambiguous_grounding), closest_category "
        "(populated only when status=ok), per_candidate (None for indeterminate categories), candidates "
        "(per-category status/n_instances/instances; each instance includes obj_id, track_id, "
        "distance_normalized, n_valid_points, degradation_flags, audit), requested_categories, "
        "categories_without_detection, categories_with_invalid_geometry, tied_categories, "
        "margin_normalized, contract_version, definition, quantile_q, and degradation_flags. Missing "
        "instance measurements, object/track reuse across roles, or tied minima make the result "
        "indeterminate. Do not take another minimum over incomplete per_candidate values. "
        "Close but unequal distances only have their margin recorded; the threshold is uncalibrated. "
        "Metric scale is not required."),
    "object_distance_m": (
        "Low quantile of bidirectional nearest-neighbor distances between two distinct objects' point "
        "sets, used as a nearest-surface-distance proxy. Return dict: distance_m (meters), equivalent "
        "alias distance_metric, distance_normalized, a/b (obj_id/category_name), definition, quantile_q, "
        "voxel_size, n_valid_points, and degradation_flags. Raise an error without a finite metric value."),
    "relative_direction_of": (
        "Determine the horizontal direction of the target_id centroid from the observer_id centroid "
        "while facing the facing_at_id centroid. Arguments accept object IDs or resolvable categories. "
        "difficulty=easy returns left/right; medium returns left/right/back (absolute angles >=135 degrees "
        "are back); hard returns front-left/front-right/back-left/back-right. Return dict: direction, "
        "difficulty, world_up_used, handedness_used. Raise an error for missing/invalid world conventions "
        "or degenerate horizontal vectors."),
    "object_visible_frames": (
        "Return the object record's visible_frames as list[int]. Values are frame slots (0..N-1) "
        "in the frozen FrameSet, not physical video frame indices. Sorting and deduplication are not "
        "guaranteed. An empty list means this record has no registered visible frames; it does not "
        "prove the object is absent from the entire video."),
    "connectivity_graph": (
        "Project the point cloud onto the ground plane defined by world_up and derive reachability "
        "from connected free-space regions in an occupancy grid. Return dict: nodes, edges, "
        "traversable_matrix, start_node, and grid diagnostics. The graph is a geometric approximation. "
        "Raise an error without a point cloud or world up vector."),
    "exists_in_scene": (
        "Resolve an object ID or category and return bool. False means the currently loaded object "
        "inventory has no matching record. Raise an error when object artifacts are unavailable."),
    "reproject": (
        "Project a world-space 3D point through camera pose and intrinsics to pixel coordinates [u,v] "
        "in the specified frame. frame_idx is a frame slot. Raise an error for a point behind the "
        "camera or on its imaging plane."),
    "euclidean_distance": (
        "Return the Euclidean distance between two finite 3D points as float, in the input coordinates' "
        "unit. No artifact or evidence dependencies."),
    "inspect_frames": (
        "Read/crop the frozen FrameSet. frame_ids is a list of frame slots (0..N-1), at most 32 per call. "
        "boxes is optional; otherwise provide one [x0,y0,x1,y1] pixel box per frame with a top-left origin "
        "and at least 8 pixels per cropped side. Return dict: images (each with image_id, frame_id, "
        "source_frame_index, box_xyxy, source_hw, sent_hw, scale, content_sha256, transform_version), "
        "n_images, frame_ids, note. Resize the longest side to at most 640 pixels; do not resample "
        "the video. After YieldObservations, the framework delivers derived images within the image "
        "budget. An image_id alone does not mean the image has been observed."),
    "detect_objects": (
        "Detect a nonempty list of categories on frozen frame slots frame_ids. Return dict: status "
        "(ok/fault/empty), detections (category_name, frame_id, bbox_xyxy in pixels, confidence, "
        "bbox_norm1000), n_detections, n_new_objects, new_object_ids, service_healthy, frame_ids, "
        "categories, error, detector_endpoint_hash, note. fault means service failure; empty means a "
        "healthy detection run with no detections. Partial frame failures may produce status=ok with "
        "a nonempty error. The framework adds new candidates to this question's inventory; they have "
        "only 2D boxes/visible frames, without 3D centroids, point clouds, or track identities. "
        "Detection dictionaries have no obj_id. new_object_ids contains only newly added IDs; "
        "it can be empty on repeated detection while earlier objects remain registered. Verify "
        "real identities with list_objects; do not infer a one-to-one mapping from list order. "
        "A partial result or empty list does not establish complete frame/category coverage."),
}
