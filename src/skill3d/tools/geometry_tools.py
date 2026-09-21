"""确定性几何 Tool 真实实现（v6 §8 矩阵 / §9 逐 Tool 定义）。

全部注册进全局 REGISTRY 并带 ToolSpec；统一签名 (handle: SceneHandle, **args)。
所有函数为确定性纯计算，不调任何 VLM / 外部服务。

v6 三条纪律：

1. **每个 Tool 显式声明 `requires_evidence`**（§17.4 硬约束 7）：无声明不得进注册表；
   可选 `tolerates_degraded` 声明"降级也能用但答案要带标记"的能力子集；
2. **执行期 fail-closed**：`REGISTRY.call_tool` 在调用实现前校验产物 + 证据，
   缺失抛 `ArtifactUnavailableError` / `ConfidenceGateError`（绝不静默返回假值）；
3. **单项失败只收回依赖该证据的工具**（§7.2）：例如 `metric_scale` 不可用时
   只隐藏米制 Tool，`list_objects` / `object_centroid` / `relative_direction_of`
   照常可用。

域值错误（负距离 / 点在相机后方 / 对象不存在 / 单位错）统一抛
`DomainValueError`，进 `ProgramExecutionTrace.error_code="domain_value"`。
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from skill3d.reconstruction_gate.world_frame import direction_of
from skill3d.schemas import ToolSpec
from skill3d.schemas.evidence import EvidenceProfile

from .contract import (
    EVIDENCE_METRIC_SCALE,
    AnswerAlreadyGiven,
    DomainValueError,
)
from .distance_primitives import (
    DistancePrimitiveParams,
    object_extent,
    robust_distance_between_pointsets,
    robust_distance_to_reference,
    room_size_from_planes,
)
from .registry import REGISTRY
from .scene_handle import SceneHandle

# 证据能力名（§7.1 词汇表；写成常量避免字符串散落）
EV_GEOMETRY = "geometry_3d"
EV_WORLD = "world_frame"
EV_METRIC = EVIDENCE_METRIC_SCALE
EV_DETECTION = "object_detection"
EV_GROUNDING = "object_grounding"
EV_TRACK = "track_consensus"
EV_TEMPORAL = "temporal"


def _as_point(p: list[float], name: str) -> np.ndarray:
    arr = np.asarray(p, dtype=np.float64)
    if arr.shape != (3,):
        raise DomainValueError(
            "argument_check", f"{name} 必须为 3 维点, 实际 shape={arr.shape}",
            args={"name": name})
    if not np.all(np.isfinite(arr)):
        raise DomainValueError(
            "argument_check", f"{name} 含 NaN/Inf: {p!r}", args={"name": name})
    return arr


def _metric_scale_or_fail(handle: SceneHandle, tool: str) -> float:
    """米制 Tool 的世界→米换算系数；缺失/非有限 → fail-closed。

    未融合成功时这些 Tool 由证据门隐藏、执行层再拦一次；这里是第三道保险：
    即使有人绕过前两道，也**不得**拿世界单位冒充米制。
    """
    v = handle.metric_scale
    if v is None or not np.isfinite(float(v)) or float(v) <= 0:
        raise DomainValueError(
            tool, f"缺少可用的 metric_scale（世界单位→米 的换算系数）：{v!r}；"
                  "尺度融合未成功时米制 Tool 不得以世界单位冒充米制")
    return float(v)


def _get_object(handle: SceneHandle, object_id: str, tool: str):
    """取对象；不存在 → DomainValueError（域值错误，不是契约缺失）。"""
    try:
        return handle.get_object(object_id)
    except KeyError as exc:
        raise DomainValueError(
            tool, f"对象不存在: {object_id}（objects 产物可用且已列出全部实例）",
            args={"object_id": object_id}) from exc


def _metric_scale_if_authorized(handle: SceneHandle) -> Optional[float]:
    """本题获得米制授权时才返回世界→米的换算系数，否则 None（§5.3/§13.2）。

    `robust_distance` / `surface_distance_between_objects` 这类**非米制 Tool**
    在米制授权缺席时（题型不是米制 / gate 未过 / scope≠metric_enabled）必须只给
    归一化值：一旦附上米制数值，M11 `check_unit_consistent` 会判"无米制授权却产出
    绝对单位数值"（§13.2 fail-closed），等于管道自己先违规。
    """
    scale = handle.metric_scale
    if scale is None or not np.isfinite(float(scale)) or float(scale) <= 0:
        return None
    if not handle.allowed_metric_tasks:      # 本题未被授权（gate ∧ 题型 ∧ scope）
        return None
    return float(scale)


def _resolve_reference_xyz(handle: SceneHandle, reference: str, tool: str) -> np.ndarray:
    """解析`robust_distance` 的参考点：观察点（相机/agent）或某个对象。

    §9.6 官方口径：rel_distance 的参考是**观察点**（相机/agent），不是另一个对象。
    """
    key = str(reference or "").strip().lower()
    if key in ("camera", "observer", "agent", "self", "camera_center", "camera0"):
        return handle.camera_center(0)
    return handle.object_centroid_world(_get_object(handle, reference, tool).obj_id)


def _degraded_property(distance_result, *, metric_ok: bool) -> list[str]:
    """把距离原语的降级信号 + 尺度可用性汇总成 degradation_flags（进 trace）。"""
    flags = list(getattr(distance_result, "degradation_flags", None) or [])
    if not metric_ok:
        flags.append("metric_scale_unavailable")
    return sorted(set(flags))


# --------------------------------------------------------------- 对象清单 ----

@REGISTRY.register(
    ToolSpec(
        name="list_objects",
        description=("列出场景中的对象（自描述）：每项含 obj_id / category_name / "
                     "visible_frames / track_id。按类别筛选传 category_filter"
                     "（子串匹配，如 list_objects('chair')）。"
                     "**obj_id 里不含类别名**，不要用 'chair' in obj_id 这类判断。"
                     "计数请改用 count_objects（它以 track 共识为准）"),
        args_schema_ref="category_filter:str=''",
        returns_schema_ref="list[dict]",
        cost_estimate_ms=1.0,
        source_default="real",
        requires_artifacts=["objects"],
        requires_evidence=[EV_DETECTION],
    )
)
def list_objects(handle: SceneHandle, category_filter: str = "") -> list[dict]:
    """返回 scene 级基础清单（**自描述**，§8/§9.1）。

    `exists_in_scene(name)==False` 当且仅当 objects 产物可用且真无此实例；
    检测器故障时 `object_detection=unavailable` → 本 Tool 被隐藏且执行期抛错，
    **不**静默返回 `[]`。
    """
    out = []
    for oid in handle.list_objects_by_name(category_filter):
        o = handle.get_object(oid)
        out.append({
            "obj_id": o.obj_id,
            "category_name": o.category_name,
            "visible_frames": [int(i) for i in (o.visible_frames or [])],
            "track_id": o.track_id,
            "det_conf": float(o.det_conf),
            "grounding_status": o.grounding_status,
            "duplicate_suspect": bool(o.duplicate_suspect),
        })
    return out


@REGISTRY.register(
    ToolSpec(
        name="count_objects",
        description=("按类别统计实例数（category_name 子串匹配，如 'chair'）。"
                     "**以 track 共识为准**（同一物体跨帧只算一次），返回 "
                     "{count, n_distinct_tracks, duplicate_suspect, evidence_degraded}。"
                     "计数题必须用它，不要用 exists_in_scene 累加（那是布尔值）"),
        args_schema_ref="category_name:str",
        returns_schema_ref="dict",
        cost_estimate_ms=2.0,
        source_default="real",
        requires_artifacts=["objects"],
        requires_evidence=[EV_DETECTION, EV_TRACK],
        # §9.2：track_consensus 降级仍可计数，但答案必须带 evidence_degraded 标记
        tolerates_degraded=[EV_TRACK],
    )
)
def count_objects(handle: SceneHandle, category_name: str) -> dict:
    """track 共识计数（§9.2）——**不数清单长度**。

    v5 实测：清单既有重复实例（12）又有漏绑实例（0），直接数长度会系统性错。
    这里按 `track_id` 去重得到 `n_distinct_tracks`；若同一 track 出现在多条
    记录里（3D 去重没合干净）→ `duplicate_suspect=True` 并降级提示。
    """
    oids = handle.list_objects_by_name(category_name)
    tracks: dict[str, int] = {}
    n_no_track = 0
    dup = False
    for oid in oids:
        o = handle.get_object(oid)
        tid = o.track_id
        if tid:
            tracks[str(tid)] = tracks.get(str(tid), 0) + 1
        else:
            n_no_track += 1
        if o.duplicate_suspect:
            dup = True
    n_distinct = len(tracks) + n_no_track
    dup = dup or any(v > 1 for v in tracks.values())
    profile = handle.evidence_profile
    evidence_degraded = bool(
        profile is not None and profile.state(EV_TRACK) == "degraded")
    return {
        "count": int(n_distinct),
        "n_distinct_tracks": int(n_distinct),
        "n_records": len(oids),
        "n_without_track": int(n_no_track),
        "duplicate_suspect": bool(dup),
        "evidence_degraded": evidence_degraded,
    }


# --------------------------------------------------------------- 单对象量 ----

@REGISTRY.register(
    ToolSpec(
        name="object_centroid",
        description=("对象质心（世界系，**归一化单位**）。尺度融合成功时另附米制"
                     "坐标 centroid_metric。相对几何题（比较远近/方位）可直接用它，"
                     "不必换算成米"),
        args_schema_ref="obj_id:str",
        returns_schema_ref="dict",
        cost_estimate_ms=1.0,
        source_default="real",
        requires_artifacts=["objects"],
        # §7.3 示例 B：metric_scale 失败时 object_centroid（相对坐标）仍必须可用
        requires_evidence=[EV_GEOMETRY, EV_DETECTION, EV_GROUNDING],
    )
)
def object_centroid(handle: SceneHandle, obj_id: str) -> dict:
    """对象质心（§9.3）：归一化单位必给，米制值仅尺度可用时附上。"""
    obj = _get_object(handle, obj_id, "object_centroid")
    c = np.asarray(obj.centroid_world, dtype=np.float64)
    if c.shape != (3,):
        raise DomainValueError("object_centroid",
                               f"对象 {obj_id} centroid 维度异常: {obj.centroid_world}",
                               args={"obj_id": obj_id})
    if not np.all(np.isfinite(c)):
        raise DomainValueError("object_centroid",
                               f"对象 {obj_id} centroid 含 NaN/Inf: {obj.centroid_world}",
                               args={"obj_id": obj_id})
    scale = handle.metric_scale
    metric = (None if scale is None or not np.isfinite(float(scale))
              else [float(x) * float(scale) for x in c])
    return {
        "centroid_normalized": [float(x) for x in c],
        "centroid_metric": metric,
        "category_name": obj.category_name,
        "track_id": obj.track_id,
    }


@REGISTRY.register(
    ToolSpec(
        name="object_3d_extent",
        description=("对象 3D 包围盒边长（世界系归一化单位）与**米制** extent。"
                     "尺寸题（object_size_estimation）用它；米制值 = 归一化 extent × "
                     "尺度，面积按平方。缺尺度即 fail-closed（不做近似）"),
        args_schema_ref="obj_id:str",
        returns_schema_ref="dict",
        cost_estimate_ms=5.0,
        source_default="real",
        requires_artifacts=["objects", "scale"],
        requires_evidence=[EV_GEOMETRY, EV_METRIC, EV_DETECTION, EV_GROUNDING],
    )
)
def object_3d_extent(handle: SceneHandle, obj_id: str) -> dict:
    """对象 3D extent（§9.4）：点云 extent 优先，无点云时退回 bbox。"""
    obj = _get_object(handle, obj_id, "object_3d_extent")
    pts = handle.object_points(obj.obj_id)
    if pts.shape[0] >= 3:
        res = object_extent(pts, metric_scale=_metric_scale_or_fail(handle, "object_3d_extent"))
        return {
            "extent_normalized": res["extent_normalized"],
            "extent_metric": res["extent_metric"],
            "n_valid_points": int(res["n_valid_points"]),
            "degradation_flags": list(res.get("degradation_flags") or []),
        }
    if len(obj.bbox) == 6:
        ext = np.asarray(obj.bbox[3:], dtype=np.float64) - np.asarray(obj.bbox[:3],
                                                                     dtype=np.float64)
        if np.any(ext < 0):
            raise DomainValueError("object_3d_extent",
                                   f"对象 {obj_id} bbox 上下界反序（负边长）",
                                   args={"obj_id": obj_id})
        s = _metric_scale_or_fail(handle, "object_3d_extent")
        return {
            "extent_normalized": [float(x) for x in ext],
            "extent_metric": [float(x) * s for x in ext],
            "n_valid_points": 0,
            "degradation_flags": ["bbox_fallback_no_pointcloud"],
        }
    raise DomainValueError("object_3d_extent",
                           f"对象 {obj_id} 既无点云也无 bbox，无法测量尺寸",
                           args={"obj_id": obj_id})


@REGISTRY.register(
    ToolSpec(
        name="plane_fit_room_size",
        description=("拟合地面/墙面 → 房间对角线（归一化）与**米制**面积。"
                     "房间尺寸题（room_size_estimation）用它。平面拟合质量不过门时"
                     "降级输出（fit_quality 明确给出）"),
        args_schema_ref="",
        returns_schema_ref="dict",
        cost_estimate_ms=50.0,
        source_default="real",
        requires_artifacts=["point_cloud", "scale"],
        requires_evidence=[EV_GEOMETRY, EV_METRIC],
    )
)
def plane_fit_room_size(handle: SceneHandle) -> dict:
    """平面拟合房间尺寸（§9.5）。"""
    pm = handle.get_point_map()
    if pm is None or pm.size == 0:
        raise DomainValueError("plane_fit_room_size",
                               "未注入世界系点图，无法拟合平面（重建产物缺失）")
    res = room_size_from_planes(np.asarray(pm, dtype=np.float64).reshape(-1, 3),
                               metric_scale=_metric_scale_or_fail(
                                   handle, "plane_fit_room_size"))
    return {
        "room_diagonal_normalized": res.get("room_diagonal_normalized"),
        "room_area_m2": res.get("room_area_m2"),
        "plane_inlier_ratio": res.get("plane_inlier_ratio"),
        "fit_quality": res.get("fit_quality"),
        "degradation_flags": list(res.get("degradation_flags") or []),
    }


# ----------------------------------------------------------------- 距离 ----

@REGISTRY.register(
    ToolSpec(
        name="robust_distance",
        description=("稳健距离：reference 到 target 的**低分位**距离。reference 可写 "
                     "'camera'/'observer'（观察点）或某个对象 id；target 是对象 id。"
                     "**相对距离题（object_rel_distance）的官方口径就是"
                     "「观察点→各候选对象」分别算再比较**，不是对象↔对象距离，"
                     "也不需要米制尺度（比较中尺度会约掉）。"
                     "返回 {distance_normalized, distance_metric, quantile_q, "
                     "voxel_size, n_valid_points, degradation_flags}"),
        args_schema_ref="reference:str, target:str",
        returns_schema_ref="dict",
        cost_estimate_ms=20.0,
        source_default="real",
        requires_artifacts=["point_cloud", "objects"],
        requires_evidence=[EV_GEOMETRY, EV_DETECTION, EV_GROUNDING],
    )
)
def robust_distance(handle: SceneHandle, reference: str, target: str) -> dict:
    """§9.6/§12.3 稳健低分位距离（rel_distance 官方口径，不需米制尺度）。"""
    ref_xyz = _resolve_reference_xyz(handle, reference, "robust_distance")
    tgt = _get_object(handle, target, "robust_distance")
    pts = handle.object_points(tgt.obj_id)
    res = robust_distance_to_reference(
        pts, ref_xyz,
        metric_scale=_metric_scale_if_authorized(handle),
        params=DistancePrimitiveParams(),
        point_conf=handle.object_point_conf(tgt.obj_id),
        conf_warp_monotonic=handle.conf_warp_monotonic,
        duplicate_suspect=bool(tgt.duplicate_suspect),
    )
    metric_ok = res.distance_metric is not None
    return {
        "distance_normalized": res.distance_normalized,
        "distance_metric": res.distance_metric,
        "scale_version": res.scale_version,
        "quantile_q": res.quantile_q,
        "voxel_size": res.voxel_size,
        "conf_warp_version": res.conf_warp_version,
        "n_valid_points": int(res.n_valid_points),
        "degradation_flags": _degraded_property(res, metric_ok=metric_ok),
    }


@REGISTRY.register(
    ToolSpec(
        name="camera_object_distance",
        description=("**米制**距离：相机（观察点）到某对象表面的稳健低分位距离（米）。"
                     "绝对距离题（object_abs_distance）用它。需要米制尺度；"
                     "尺度不可用时本 Tool 不会出现在可用列表里"),
        args_schema_ref="obj_id:str",
        returns_schema_ref="dict",
        cost_estimate_ms=20.0,
        source_default="real",
        requires_artifacts=["point_cloud", "objects", "scale"],
        requires_evidence=[EV_GEOMETRY, EV_METRIC, EV_DETECTION, EV_GROUNDING],
    )
)
def camera_object_distance(handle: SceneHandle, obj_id: str) -> dict:
    """§12.3 abs_distance：R = 相机中心，P = 对象 mask 内 3D 点，米制 = 归一化 × 尺度。"""
    obj = _get_object(handle, obj_id, "camera_object_distance")
    scale = _metric_scale_or_fail(handle, "camera_object_distance")
    res = robust_distance_to_reference(
        handle.object_points(obj.obj_id), handle.camera_center(0),
        metric_scale=scale,
        params=DistancePrimitiveParams(),
        point_conf=handle.object_point_conf(obj.obj_id),
        conf_warp_monotonic=handle.conf_warp_monotonic,
        duplicate_suspect=bool(obj.duplicate_suspect),
    )
    return {
        "distance_normalized": res.distance_normalized,
        "distance_metric": res.distance_metric,
        "scale_version": res.scale_version,
        "quantile_q": res.quantile_q,
        "voxel_size": res.voxel_size,
        "conf_warp_version": res.conf_warp_version,
        "n_valid_points": int(res.n_valid_points),
        "degradation_flags": _degraded_property(res, metric_ok=res.distance_metric is not None),
    }


@REGISTRY.register(
    ToolSpec(
        name="surface_distance_between_objects",
        description=("两个对象**表面之间**的距离（双向最近邻的低分位，归一化单位）。"
                     "**注意：它不用于 object_rel_distance 题型的官方口径作答**"
                     "（那个口径是「观察点→各对象」）。仅在需要对象间距的场景使用"),
        args_schema_ref="obj_a:str, obj_b:str",
        returns_schema_ref="dict",
        cost_estimate_ms=30.0,
        source_default="real",
        requires_artifacts=["point_cloud", "objects"],
        requires_evidence=[EV_GEOMETRY, EV_DETECTION, EV_GROUNDING],
    )
)
def surface_distance_between_objects(handle: SceneHandle, obj_a: str, obj_b: str) -> dict:
    """§9.7：双向 NN 距离集合的低分位（O((n_A+n_B)·log·max)，不做全笛卡尔积）。"""
    a = _get_object(handle, obj_a, "surface_distance_between_objects")
    b = _get_object(handle, obj_b, "surface_distance_between_objects")
    if a.obj_id == b.obj_id:
        raise DomainValueError("surface_distance_between_objects",
                               f"obj_a 与 obj_b 是同一对象: {obj_a}",
                               args={"obj_a": obj_a, "obj_b": obj_b})
    res = robust_distance_between_pointsets(
        handle.object_points(a.obj_id), handle.object_points(b.obj_id),
        metric_scale=_metric_scale_if_authorized(handle),
        params=DistancePrimitiveParams(),
        duplicate_suspect=bool(a.duplicate_suspect or b.duplicate_suspect),
    )
    return {
        "surface_distance_normalized": res.distance_normalized,
        "surface_distance_metric": res.distance_metric,
        "n_nn_samples": int(getattr(res, "n_nn_samples", 0) or 0),
        "n_valid_points": int(res.n_valid_points),
        "point_contamination_suspect": (
            "point_contamination_suspect" in (res.degradation_flags or [])),
        "quantile_q": res.quantile_q,
        "voxel_size": res.voxel_size,
        "degradation_flags": list(res.degradation_flags or []),
    }


# ----------------------------------------------------------------- 方向 ----

@REGISTRY.register(
    ToolSpec(
        name="relative_direction_of",
        description=("站在 observer 处、**面向 facing_at**，判断 target 在 "
                     "front/behind/left/right 哪个方向。三个参数都是**对象 id 或类别名**"
                     "（如 'obj_3' / 'whiteboard'），内部取各自质心："
                     "facing 方向 = facing_at 质心 − observer 质心。"
                     "方向题优先用它，不要自己拼朝向向量"),
        args_schema_ref="observer_id:str, facing_at_id:str, target_id:str",
        returns_schema_ref="dict",
        cost_estimate_ms=2.0,
        source_default="real",
        requires_artifacts=["objects"],
        requires_evidence=[EV_GEOMETRY, EV_WORLD, EV_DETECTION, EV_GROUNDING],
    )
)
def relative_direction_of(handle: SceneHandle, observer_id: str,
                          facing_at_id: str, target_id: str) -> dict:
    """按对象 id 解析参照系后的相对方位（§9.8）。

    判据 `right ⟺ (f×d)·u < 0`，`u = world_up`。**world_up/handedness 缺失或非法
    → fail-closed（抛错）**，不退回无符号启发式。
    """
    obs = handle.object_centroid_world(_get_object(handle, observer_id,
                                                   "relative_direction_of").obj_id)
    fac = handle.object_centroid_world(_get_object(handle, facing_at_id,
                                                   "relative_direction_of").obj_id)
    tgt = handle.object_centroid_world(_get_object(handle, target_id,
                                                   "relative_direction_of").obj_id)
    try:
        direction = direction_of(
            observer_xyz=obs, facing_xyz=fac, target_xyz=tgt,
            world_up=handle.world_up, handedness=handle.handedness)
    except ValueError as exc:
        # §9.8：世界系约定缺失/非法 → fail-closed（DomainValueError，不猜）
        raise DomainValueError("relative_direction_of", str(exc),
                               args={"observer_id": observer_id,
                                     "facing_at_id": facing_at_id,
                                     "target_id": target_id}) from exc
    return {
        "direction": direction,
        "world_up_used": [float(x) for x in handle.world_up],
        "handedness_used": str(handle.handedness),
    }


# --------------------------------------------------------------- 时序/路线 ----

@REGISTRY.register(
    ToolSpec(
        name="object_visible_frames",
        description=("某对象在统一 FrameSet 中**可见**的帧槽位序号（升序 0..31；"
                     "空列表 = 全程无有效掩码）。用于外观顺序题：对每个候选对象取 "
                     "min(可见帧) 再排序即可"),
        args_schema_ref="obj_id:str",
        returns_schema_ref="list[int]",
        cost_estimate_ms=1.0,
        source_default="real",
        requires_artifacts=["objects"],
        requires_evidence=[EV_TEMPORAL, EV_DETECTION, EV_GROUNDING],
    )
)
def object_visible_frames(handle: SceneHandle, obj_id: str) -> list[int]:
    obj = _get_object(handle, obj_id, "object_visible_frames")
    return [int(i) for i in (obj.visible_frames or [])]


@REGISTRY.register(
    ToolSpec(
        name="connectivity_graph",
        description=("构建场景连通性/可达图（路线规划题 route_planning 用它）："
                     "地面占据栅格 → 自由空间连通域 → 对象节点与可达边。"
                     "返回 {nodes, edges, traversable_matrix, start_node}"),
        args_schema_ref="",
        returns_schema_ref="dict",
        cost_estimate_ms=100.0,
        source_default="real",
        requires_artifacts=["point_cloud", "objects"],
        requires_evidence=[EV_GEOMETRY, EV_WORLD, EV_DETECTION],
    )
)
def connectivity_graph(handle: SceneHandle) -> dict:
    """场景连通性/可达图（§9.10）。

    实现：地面平面占据栅格 → 自由空间连通域（`scipy.ndimage.label`）→
    对象与起点按所在自由域判定可达。阈值全 `[TODO_CALIBRATE]`。

    句柄缺世界系约定/点图时抛 `KeyError` → 转成受控 `DomainValueError`
    （§9.12：域值错误必须归因到 `domain_value`，不得崩成"服务故障"）。
    """
    try:
        return handle.connectivity_graph()
    except KeyError as exc:
        raise DomainValueError("connectivity_graph", str(exc), args={}) from exc


# --------------------------------------------------------- 通用小工具 ----

@REGISTRY.register(
    ToolSpec(
        name="exists_in_scene",
        description="按 obj_id 或类别名（大小写不敏感）判断对象是否存在于场景",
        args_schema_ref="name:str",
        returns_schema_ref="bool",
        cost_estimate_ms=1.0,
        source_default="real",
        # objects 不可用时抛 ArtifactUnavailableError；
        # 返回 False 当且仅当 objects 可用且真无此实例
        requires_artifacts=["objects"],
        requires_evidence=[EV_DETECTION],
    )
)
def exists_in_scene(handle: SceneHandle, name: str) -> bool:
    """`exists_in_scene(name)==False` 当且仅当 objects 可用且真无此实例（§9.1）。"""
    if not handle.objects_materialized:
        from .contract import ArtifactUnavailableError

        raise ArtifactUnavailableError(
            tool="exists_in_scene", missing=["objects"], available=[],
            route=handle.scene_route, args={"name": name})
    return bool(handle.exists(name))


@REGISTRY.register(
    ToolSpec(
        name="reproject",
        description="世界系 3D 点经相机位姿与内参重投影到指定帧像素坐标 [u, v]；点在相机后方时抛错",
        args_schema_ref="p3d:list[float], frame_idx:int",
        returns_schema_ref="list",
        cost_estimate_ms=1.0,
        source_default="real",
        requires_artifacts=["poses", "intrinsics"],
        requires_evidence=[EV_GEOMETRY],
    )
)
def reproject(handle: SceneHandle, p3d: list[float], frame_idx: int) -> list[float]:
    p = _as_point(p3d, "p3d")
    c2w = handle.get_c2w(frame_idx)
    K = handle.get_intrinsics(frame_idx)
    p_h = np.concatenate([p, [1.0]])
    p_cam = np.linalg.inv(c2w) @ p_h
    z = p_cam[2]
    if z <= 0:
        raise DomainValueError("reproject",
                               f"点在相机后方或成像平面上 (z={z})，无法重投影",
                               args={"p3d": p3d, "frame_idx": frame_idx})
    uv = K @ p_cam[:3]
    return [float(uv[0] / z), float(uv[1] / z)]


@REGISTRY.register(
    ToolSpec(
        name="euclidean_distance",
        description="两个 3D 点（世界系）的欧氏距离，返回非负浮点数",
        args_schema_ref="point_a:list[float], point_b:list[float]",
        returns_schema_ref="float",
        cost_estimate_ms=1.0,
        source_default="real",
        requires_artifacts=[],   # 纯算术：只依赖入参
        requires_evidence=[],    # 无证据依赖：任何证据状态下都可用
    )
)
def euclidean_distance(handle: SceneHandle, point_a: list[float],
                       point_b: list[float]) -> float:
    a = _as_point(point_a, "point_a")
    b = _as_point(point_b, "point_b")
    d = float(np.linalg.norm(a - b))
    if not np.isfinite(d) or d < 0:
        raise DomainValueError("euclidean_distance", f"距离非有限或为负: {d}",
                               args={"point_a": point_a, "point_b": point_b})
    return d


__all__ = [
    "AnswerAlreadyGiven",
    "EvidenceProfile",
    "REGISTRY",
]
