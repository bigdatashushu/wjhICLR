"""Current neutral Tool interface descriptions; no task-specific recipes."""

TOOL_DOCS_VERSION = "tool-docs-v11.2"


DESCRIPTIONS = {
    "list_objects": (
        "返回对象记录 list[dict]，字段 obj_id、category_name、visible_frames、track_id、"
        "det_conf、grounding_status、duplicate_suspect。空过滤串返回全部；类别过滤使用"
        "大小写/分隔符/单复数/同义词归一化后的完整类别匹配，可移除登记的外观前缀。"
        "不作子串或词交集匹配。obj_id 不编码类别；需要唯一对象的接口遇到多实例类别会抛歧义错误。"),
    "count_objects": (
        "按与 list_objects 相同的类别规则统计实例。相同非空 track_id 先分组，"
        "无 track 的记录各自一组；组内点云合并后按双向近邻重合率合并重复组；"
        "无点云的组保留一次。返回 dict：count（最终组数）、n_records、"
        "n_distinct_tracks（几何合并前的组数，含无 track 记录）、n_geometric_merges、"
        "n_without_track、duplicate_suspect、evidence_degraded、"
        "instance_consolidation（method、min_overlap、error）。"),
    "object_centroid": (
        "返回 dict：centroid_normalized（世界系三维质心，归一化单位）、"
        "centroid_metric（三维米制坐标，无米制授权时 None）、category_name、track_id、"
        "world_up_used（世界系单位上向量）、handedness_used（right/left）。"
        "世界系证据不可用或方向元数据无效时，后两项均为 None；质心仍可用。"),
    "object_3d_extent": (
        "返回 dict：extent_normalized（归一化逐轴边长 [dx,dy,dz]）、"
        "extent_metric（逐轴米制边长 [dx,dy,dz]）、n_valid_points、degradation_flags。"
        "点云分支另有 extent_longest_metric；缺点云时可用 bbox 并标记 bbox_fallback_no_pointcloud。"
        "缺少有效尺度或可测几何时抛错。"),
    "plane_fit_room_size": (
        "根据场景点云拟合地面/墙面，返回 dict：room_diagonal_normalized、"
        "room_area_m2（平方米）、plane_inlier_ratio、fit_quality（[0,1]）、"
        "degradation_flags。缺少有限面积值时抛错。"),
    "robust_distance": (
        "reference 点到 target 对象点集的近邻低分位距离；reference 为 camera/observer/"
        "agent/self/camera_center/camera0 时取首帧相机中心，否则取对象质心；target 为对象。"
        "返回 dict：distance_normalized、distance_metric（米，无授权时 None）、"
        "scale_version、quantile_q、voxel_size、conf_warp_version、n_valid_points、"
        "degradation_flags。点不足时距离可为 None。"),
    "camera_object_distance": (
        "首帧相机中心到对象点集的低分位距离，返回 dict：distance_normalized、"
        "distance_metric（米）、scale_version、quantile_q、voxel_size、conf_warp_version、"
        "n_valid_points、degradation_flags。需米制授权；点不足时距离可为 None。"),
    "surface_distance_between_objects": (
        "两个不同对象点集的双向最近邻距离低分位，作为表面最近距离代理。返回 dict："
        "surface_distance_normalized、surface_distance_metric（米，无授权时 None）、"
        "同值别名 distance_normalized/distance_metric、n_nn_samples、n_valid_points、"
        "point_contamination_suspect、quantile_q、voxel_size、degradation_flags。"
        "点不足时距离可为 None。"),
    "relative_distance_rank": (
        "reference 为参照对象 id 或唯一可解析类别名；candidate_categories 包含题目全部"
        "选项的类别名，至少两类，不得重复或为同义类别。"
        "每类取与参照对象的表面距离代理最小的实例，代理为点集双向最近邻距离低分位。"
        "返回 dict：reference（obj_id/category_name）、ranking（按距离升序的记录列表，"
        "每项 category、obj_id、distance_normalized、n_instances_in_category）、"
        "status（ok/incomplete_candidates/uncertain_geometry/ambiguous_grounding）、"
        "closest_category（仅 status=ok 有值）、per_candidate（不可判定类别为 None）、"
        "candidates（逐类别 status/n_instances/instances；实例含 obj_id、track_id、"
        "distance_normalized、n_valid_points、degradation_flags、audit）、"
        "requested_categories、categories_without_detection、categories_with_invalid_geometry、"
        "tied_categories、margin_normalized、contract_version、definition、quantile_q、"
        "degradation_flags。缺实例测量、同一对象/track 跨角色复用或最小距离并列均不可判定，"
        "不得对不完整 per_candidate 另取最小值。接近但不相等的距离仅记录差值，阈值未标定。"
        "无需米制尺度。"),
    "object_distance_m": (
        "两个不同对象点集的双向最近邻距离低分位，作为表面最近距离代理。"
        "返回 dict：distance_m（米）、同值别名 distance_metric、distance_normalized、"
        "a/b（obj_id/category_name）、definition、quantile_q、voxel_size、n_valid_points、"
        "degradation_flags。缺少有限米制值时抛错。"),
    "relative_direction_of": (
        "在 observer_id 质心处面向 facing_at_id 质心，判断 target_id 质心的水平相对方向。"
        "参数接受对象 id 或可解析类别名。difficulty=easy 返回 left/right；"
        "medium 返回 left/right/back（绝对转角 ≥135° 为 back）；"
        "hard 返回 front-left/front-right/back-left/back-right。"
        "返回 dict：direction、difficulty、world_up_used、handedness_used。"
        "世界系约定缺失/非法或水平向量退化时抛错。"),
    "object_visible_frames": (
        "返回对象记录的 visible_frames，类型 list[int]；数值为冻结 FrameSet 的帧槽位"
        "（0..N-1），不是视频物理帧号。不保证已排序或去重。"
        "空列表表示该记录没有可见帧登记，不证明对象在整个视频中不存在。"),
    "connectivity_graph": (
        "将点云投影到由 world_up 确定的地面平面，按占据栅格自由空间连通域生成可达关系。"
        "返回 dict：nodes、edges、traversable_matrix、start_node 及栅格诊断信息；"
        "该图是几何近似。缺点云或世界上向量时抛错。"),
    "exists_in_scene": (
        "按对象 id 或类别名解析，返回 bool；False 表示当前已加载对象清单中无匹配记录。"
        "对象产物不可用时抛错。"),
    "reproject": (
        "世界系三维点经相机位姿与内参投影，返回指定帧的像素坐标 [u,v]；"
        "frame_idx 为帧槽位。点在相机后方或成像平面上时抛错。"),
    "euclidean_distance": (
        "返回两个有限三维点的欧氏距离 float，单位与输入坐标相同。无产物或证据依赖。"),
    "inspect_frames": (
        "读取/裁剪冻结 FrameSet，frame_ids 为帧槽位列表（0..N-1），单次最多32帧；"
        "boxes 可省略，否则逐帧对应 [x0,y0,x1,y1] 像素框（左上原点），裁后每边至少8像素。"
        "返回 dict：images（每项 image_id、frame_id、source_frame_index、box_xyxy、"
        "source_hw、sent_hw、scale、content_sha256、transform_version）、n_images、"
        "frame_ids、note。图像最长边缩到不超过640像素；不重采样视频。"
        "YieldObservations 让出后由框架按图像预算交付派生图，image_id 本身不表示已看过图。"),
    "detect_objects": (
        "在冻结帧槽位 frame_ids 上按非空 categories 类别名列表检测。返回 dict："
        "status（ok/fault/empty）、detections（category_name、frame_id、bbox_xyxy 像素框、"
        "confidence、bbox_norm1000）、n_detections、n_new_objects、new_object_ids、"
        "service_healthy、frame_ids、categories、error、detector_endpoint_hash、note。"
        "fault 为服务故障，empty 为健康空检出；有检出但部分帧失败时可为 ok 且 error 非空。"
        "框架把新候选加入本题清单；仅有二维框/可见帧，不提供三维质心、点云或 track 身份。"),
}
