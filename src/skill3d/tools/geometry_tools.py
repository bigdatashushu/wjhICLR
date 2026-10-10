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

**`tolerates_degraded` 的口径（2026-09-21 真实 GPU 实测修正）**：

除了 `metric_scale`（D3 硬契约：米制 Tool 只在 gate 通过 =
`metric_scale==available` 时可用），其余能力的 `degraded` 一律**容忍**，
工具照常暴露但答案带 `evidence_degraded:<能力>` 标记。理由是真实数据给的教训：
本机真实 VGGT corpus（scene 41069043）实测 `warp_inlier=0.668` / `cloud_overlap=0.551`
→ 主门**通过**，但因为"相邻帧旋转跳变 43.3°"这条**诊断告警**，`geometry_3d`
被判 `degraded`；若不容忍，整个 3D Tool 面（含方向题唯一可用的
`relative_direction_of`）会被一起隐藏 —— 这与 §6.2"主门告警未崩 → 仍走 full_3d"
和 §2.3"4 个 MCA 题型不受度量路线影响"直接冲突。

`degraded` 的语义是"能用但有告警"，不是"不能用"；`unavailable` 才是收回工具的信号。

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
    consolidate_instances,
    object_extent,
    robust_distance_between_pointsets,
    robust_distance_to_reference,
    room_size_from_planes,
)
from .registry import REGISTRY
from .category_match import matches
from .ranking_contract import RANKING_CONTRACT_VERSION, UNCERTAIN_DISTANCE_FLAGS
from .scene_handle import SceneHandle

# 实例整合的双向点云重合阈值（v7 §9.1）。
# 实测分离度（scene 7b6477cb95 的 12 条 monitor 记录）：同一实例 0.38–0.69，
# 不同实例 ≤0.14 → 取 0.30 落在间隙内，两侧各留 >2 倍余量。
INSTANCE_OVERLAP_THRESHOLD = 0.30

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


def _require_metric(value, tool: str, what: str):
    """米制 Tool 的**非空米制值**校验（v7 §10.1 fail-closed）。

    实测根因（2026-09-22 真实 smoke）：`object_distance_m` 在点云有效点不足时会让
    原语返回 `distance_metric=None`，工具于是**静默**回一个 `{'distance_m': None}`。
    模型照常取键、交给 `ReturnAnswer(None)` → 提交值归一成空串 → 评分解析失败 →
    等值 0 分，而且**不会**触发恢复/零工具兜底（因为在管线上看这次调用是"成功"的）。
    宁可让这次调用显式失败（`domain_value`），把 episode 推进到恢复或视觉估计路径。
    """
    if value is None or not np.isfinite(float(value)):
        raise DomainValueError(
            tool,
            f"{what} 无法给出有限数值（{value!r}）：通常是该对象的有效 3D 点不足或"
            "点云被判定为降级。请不要把它当作 0 —— 改用其他可用工具，"
            "或直接依据图片给出视觉估计并在答案里标明是估计。")
    return float(value)


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
    """解析参考点：观察点（相机/agent）或某个对象。

    **v7 §2.2 纠错**：`object_rel_distance` 的官方口径是
    「**题面参照对象** → 各候选对象」，候选类别有多个实例时取最近实例
    （依据 VSI-Bench 原论文附录 B.1）。观察点（相机）只是**辅助**参考，
    不得用它替代题面对象间距离。这里保留 `camera`/`observer` 是因为相机量在
    其他工具与审计里仍需要，但相对距离题请用 `relative_distance_rank`。
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
        description=("列出场景中的对象，返回一个 **dict 列表**，每项形如 "
                     "{'obj_id': 'obj_7', 'category_name': 'chair', "
                     "'visible_frames': [3,4,5], 'track_id': 'trk7', 'det_conf': 0.8, "
                     "'grounding_status': 'base_list', 'duplicate_suspect': False}。"
                     "用法照抄：`objs = list_objects('chair')` 后取 "
                     "`objs[0]['obj_id']`；**列表元素是 dict，不是字符串**"
                     "（写 `for o in objs: 'chair' in o` 会报错）。"
                     "**obj_id 里不含类别名**，按类别筛选必须传 category_filter"
                     "（规范类别与同义词匹配）。计数请改用 count_objects（它以 track 共识为准）"),
        args_schema_ref="category_filter:str=''",
        returns_schema_ref="list[dict]",
        cost_estimate_ms=1.0,
        source_default="real",
        requires_artifacts=["objects"],
        requires_evidence=[EV_DETECTION],
        tolerates_degraded=[EV_DETECTION],
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
        description=("按规范类别与同义词统计实例数（如 'chair'）。"
                     "**以实例共识为准**：先按 track 去重，再按点云几何重合把"
                     "「同一物体被重复检出」的记录合并（v7 §9.1 三值之二）。"
                     "返回一个 dict："
                     "{'count': 4, 'n_distinct_tracks': 4, 'n_records': 5, "
                     "'n_geometric_merges': 1, 'n_without_track': 0, "
                     "'duplicate_suspect': False, 'evidence_degraded': False}。"
                     "**答案要取 `result['count']`**，不要把整个 dict 交给 ReturnAnswer。"
                     "计数题必须用它，不要用 exists_in_scene 累加（那是布尔值）。"
                     "若 count 与你在图片里数出来的明显不符，用 YieldObservations "
                     "把明细取回来再判断"),
        args_schema_ref="category_name:str",
        returns_schema_ref="dict",
        cost_estimate_ms=40.0,
        source_default="real",
        requires_artifacts=["objects"],
        requires_evidence=[EV_DETECTION, EV_TRACK],
        # §9.2：track_consensus 降级仍可计数，但答案必须带 evidence_degraded 标记
        tolerates_degraded=[EV_DETECTION, EV_TRACK],
    )
)
def count_objects(handle: SceneHandle, category_name: str) -> dict:
    """实例共识计数（§9.2）——**不数清单长度，也不只数 track 数**。

    v5/v6 实测：清单既有重复实例（同一显示器被分裂成多条 track）又有漏绑实例。
    只按 `track_id` 去重会漏掉"支撑帧互不相交 → 时序判据不合并"的重复记录：
    scene `7b6477cb95` 问 "How many monitor(s)"（GT=5）时返回 12。

    v7 在 track 去重之上再加一道**几何重合**整合（与帧无关），
    阈值有实测分离度支撑（同实例 0.38–0.69 vs 不同实例 ≤0.14）。
    """
    oids = handle.list_objects_by_name(category_name)
    groups: dict[tuple[str, str], list[str]] = {}
    n_no_track = 0
    dup = False
    for oid in oids:
        o = handle.get_object(oid)
        tid = o.track_id
        if tid:
            key = ("track", str(tid))
        else:
            n_no_track += 1
            key = ("object", oid)
        groups.setdefault(key, []).append(oid)
        if o.duplicate_suspect:
            dup = True
    n_track_distinct = len(groups)

    # Track 身份先合并；几何阶段只合并这些组，不能重新拆开同一 track。
    point_sets: dict[str, np.ndarray] = {}
    merge_error: Optional[str] = None
    for index, members in enumerate(groups.values()):
        clouds = []
        for oid in members:
            try:
                pts = np.asarray(handle.object_points(oid), dtype=np.float64)
                if pts.ndim != 2 or pts.shape[1] != 3:
                    raise ValueError(f"{oid}: point cloud must have shape (N, 3)")
                pts = pts[np.all(np.isfinite(pts), axis=1)]
                if len(pts):
                    clouds.append(pts)
            except Exception as exc:  # noqa: BLE001 - 缺点云的组仍按 track 计一次
                merge_error = f"{type(exc).__name__}: {exc}"
        if clouds:
            point_sets[str(index)] = np.concatenate(clouds, axis=0)
    n_geometric_merges = 0
    n_final = n_track_distinct
    if len(point_sets) >= 2:
        try:
            cons = consolidate_instances(
                point_sets,
                min_overlap=INSTANCE_OVERLAP_THRESHOLD,
                eps=float(DistancePrimitiveParams().voxel_size))
            n_geometric_merges = int(cons["n_input"]) - int(cons["n_clusters"])
            # 包括没有点云的 track 组；合并数只计算几何阶段减少的组数。
            n_final = n_track_distinct - n_geometric_merges
        except Exception as exc:  # noqa: BLE001 - 整合失败退化为 track 计数（不报错）
            merge_error = f"{type(exc).__name__}: {exc}"
            n_final = n_track_distinct

    profile = handle.evidence_profile
    evidence_degraded = bool(
        profile is not None and profile.state(EV_TRACK) == "degraded")
    return {
        "count": int(n_final),
        "n_distinct_tracks": int(n_track_distinct),
        "n_records": len(oids),
        "n_geometric_merges": int(n_geometric_merges),
        "n_without_track": int(n_no_track),
        "duplicate_suspect": bool(dup),
        "evidence_degraded": evidence_degraded,
        "instance_consolidation": {
            "method": "track_then_bidirectional_nn_pointcloud_overlap_v11",
            "min_overlap": float(INSTANCE_OVERLAP_THRESHOLD),
            "error": merge_error,
        },
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
        tolerates_degraded=[EV_GEOMETRY, EV_DETECTION, EV_GROUNDING],
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
    scale = _metric_scale_if_authorized(handle)
    metric = (None if scale is None
              else [float(x) * float(scale) for x in c])
    # 质心本身不依赖 world_frame；只在证据与实际元数据都有效时附带方向约定。
    up, handedness = None, None
    profile = handle.evidence_profile
    if profile is not None and profile.state(EV_WORLD) in ("available", "degraded"):
        raw_up, raw_hand = handle.world_up, handle.handedness
        if raw_up is not None:
            arr = np.asarray(raw_up, dtype=np.float64)
            if (arr.shape == (3,) and np.all(np.isfinite(arr))
                    and np.linalg.norm(arr) > 1e-12 and raw_hand in ("right", "left")):
                up = [float(x) for x in arr / np.linalg.norm(arr)]
                handedness = raw_hand
    return {
        "centroid_normalized": [float(x) for x in c],
        "centroid_metric": metric,
        "category_name": obj.category_name,
        "track_id": obj.track_id,
        "world_up_used": up,
        "handedness_used": handedness,
    }


@REGISTRY.register(
    ToolSpec(
        name="object_3d_extent",
        description=("对象 3D 包围盒边长。返回 dict："
                     "{'extent_normalized': [dx,dy,dz], 'extent_metric': [dx,dy,dz], "
                     "'n_valid_points': N, 'degradation_flags': [...]}。"
                     "**两个 extent 都是逐轴列表（x/y/z 三个数），不是标量。**"
                     "**单位是米**；若题目问『最长边、以厘米计』，取 "
                     "`max(object_3d_extent(obj)['extent_metric']) * 100`，"
                     "把结果（一个数）交给 ReturnAnswer —— 不要直接把列表交出去。"
                     "缺尺度即 fail-closed（不做近似）"),
        args_schema_ref="obj_id:str",
        returns_schema_ref="dict",
        cost_estimate_ms=5.0,
        source_default="real",
        requires_artifacts=["objects", "scale"],
        requires_evidence=[EV_GEOMETRY, EV_METRIC, EV_DETECTION, EV_GROUNDING],
        # metric_scale 是唯一不容忍 degraded 的能力（D3 硬契约）；其余容忍
        tolerates_degraded=[EV_GEOMETRY, EV_DETECTION, EV_GROUNDING],
    )
)
def object_3d_extent(handle: SceneHandle, obj_id: str) -> dict:
    """对象 3D extent（§9.4）：点云 extent 优先，无点云时退回 bbox。"""
    obj = _get_object(handle, obj_id, "object_3d_extent")
    pts = handle.object_points(obj.obj_id)
    if pts.shape[0] >= 3:
        res = object_extent(pts, metric_scale=_metric_scale_or_fail(handle, "object_3d_extent"))
        # v7 §10.1：米制边长缺失（点云降级）→ 显式失败，不能让 None 变成空答案
        ext_m = res.get("extent_metric")
        if ext_m is None:
            raise DomainValueError(
                "object_3d_extent",
                f"对象 {obj_id}（{obj.category_name}）的有效 3D 点不足"
                f"（n_valid={res.get('n_valid_points')}），无法给出米制尺寸。"
                "请不要把缺失值当作 0 —— 改用其他工具或依据图片给出视觉估计。",
                args={"obj_id": obj_id})
        return {
            "extent_normalized": res["extent_normalized"],
            "extent_metric": ext_m,
            "extent_longest_metric": res.get("extent_longest_metric"),
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
        description=("拟合地面/墙面 → 房间尺寸。返回 dict："
                     "{'room_diagonal_normalized': d, 'room_area_m2': A, "
                     "'plane_inlier_ratio': r, 'fit_quality': q, 'degradation_flags': [...]}。"
                     "房间面积题一律取 **`result['room_area_m2']`**（单位已是平方米，"
                     "**不要**再换算）；注意键名是 `room_area_m2`，不是 area_metric / "
                     "room_size / area。`fit_quality` 是 [0,1] 的拟合质量审计值"
                     "（真实房间上实测偏低，仅供审计，不要拿它当 abstain 的硬门）"),
        args_schema_ref="",
        returns_schema_ref="dict",
        cost_estimate_ms=50.0,
        source_default="real",
        requires_artifacts=["point_cloud", "scale"],
        requires_evidence=[EV_GEOMETRY, EV_METRIC],
        tolerates_degraded=[EV_GEOMETRY],
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
    area = _require_metric(res.get("room_area_m2"), "plane_fit_room_size",
                           "房间面积（平方米）")
    return {
        "room_diagonal_normalized": res.get("room_diagonal_normalized"),
        "room_area_m2": area,
        "plane_inlier_ratio": res.get("plane_inlier_ratio"),
        "fit_quality": res.get("fit_quality"),
        "degradation_flags": list(res.get("degradation_flags") or []),
    }


# ----------------------------------------------------------------- 距离 ----

@REGISTRY.register(
    ToolSpec(
        name="robust_distance",
        description=("稳健距离：reference 到 target 的**低分位**（近邻）距离。"
                     "reference 可写 'camera'/'observer'（观察点）**或某个对象 id**；"
                     "target 是对象 id。返回键："
                     "{'distance_normalized', 'distance_metric'(米，无米制授权时为 None), "
                     "'quantile_q', 'voxel_size', 'n_valid_points', 'degradation_flags'}。"
                     "**相对距离题请改用 `relative_distance_rank`**（它一次比较多个候选，"
                     "且用的是官方口径「题面参照对象→候选」）"),
        args_schema_ref="reference:str, target:str",
        returns_schema_ref="dict",
        cost_estimate_ms=20.0,
        source_default="real",
        requires_artifacts=["point_cloud", "objects"],
        requires_evidence=[EV_GEOMETRY, EV_DETECTION, EV_GROUNDING],
        tolerates_degraded=[EV_GEOMETRY, EV_DETECTION, EV_GROUNDING],
    )
)
def robust_distance(handle: SceneHandle, reference: str, target: str) -> dict:
    """§9.6/§12.3 稳健低分位距离（不依赖米制尺度即可比较远近）。"""
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
        description=("**米制**距离：相机（观察点）到某对象表面的稳健低分位距离。"
                     "返回 dict：{'distance_normalized': d, 'distance_metric': D, ...}；"
                     "**`distance_metric` 的单位是米**，答案取 "
                     "`result['distance_metric']`（若题目要厘米则 ×100）。"
                     "绝对距离题（object_abs_distance）用它。需要米制尺度；"
                     "尺度不可用时本 Tool 不会出现在可用列表里"),
        args_schema_ref="obj_id:str",
        returns_schema_ref="dict",
        cost_estimate_ms=20.0,
        source_default="real",
        requires_artifacts=["point_cloud", "objects", "scale"],
        requires_evidence=[EV_GEOMETRY, EV_METRIC, EV_DETECTION, EV_GROUNDING],
        tolerates_degraded=[EV_GEOMETRY, EV_DETECTION, EV_GROUNDING],
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
        description=("两个对象**表面之间**的最近距离（双向最近邻的低分位，归一化单位）。"
                     "**绝对距离题（object_abs_distance）用 `object_distance_m`**；"
                     "相对距离题用 `relative_distance_rank`。"
                     "返回键：{'surface_distance_normalized', "
                     "'surface_distance_metric'（米；无米制授权时 None）, "
                     "'n_nn_samples', 'n_valid_points', 'quantile_q', 'voxel_size', "
                     "'degradation_flags'}。兼容别名 `distance_normalized` / "
                     "`distance_metric` 同时返回，两者是同一个数"),
        args_schema_ref="obj_a:str, obj_b:str",
        returns_schema_ref="dict",
        cost_estimate_ms=30.0,
        source_default="real",
        requires_artifacts=["point_cloud", "objects"],
        requires_evidence=[EV_GEOMETRY, EV_DETECTION, EV_GROUNDING],
        tolerates_degraded=[EV_GEOMETRY, EV_DETECTION, EV_GROUNDING],
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
        # v7 §10.1 兼容别名：模型在 v6 实测里稳定地按 `distance_metric` 取名取键
        # （object_abs_distance 的 16 道题里 6 道因此 KeyError 而作废），
        # 同一份数多给一个键比让模型猜键名更划算。
        "distance_normalized": res.distance_normalized,
        "distance_metric": res.distance_metric,
        "n_nn_samples": int(getattr(res, "n_nn_samples", 0) or 0),
        "n_valid_points": int(res.n_valid_points),
        "point_contamination_suspect": (
            "point_contamination_suspect" in (res.degradation_flags or [])),
        "quantile_q": res.quantile_q,
        "voxel_size": res.voxel_size,
        "degradation_flags": list(res.degradation_flags or []),
    }


# ------------------------------------------------- 相对距离题（v7 §2.2 口径）----

@REGISTRY.register(
    ToolSpec(
        name="relative_distance_rank",
        description=("**相对距离题（object_rel_distance）的官方口径实现**：给定题面"
                     "**参照对象** reference，以及若干**候选类别** candidates，"
                     "比较「参照对象 → 各候选」的最近距离，返回排序结果。"
                     "候选类别有多个实例时**取离参照对象最近的实例**（原论文附录 B.1）。"
                     "不需要米制尺度（比较中尺度约掉）。"
                     "参数：reference 是对象 id 或类别名；candidates 是**类别名列表**"
                     "（如 ['telephone','keyboard']）。"
                     "必须传入题目全部选项类别。返回 status、candidates 实例明细、"
                     "ranking、closest_category、per_candidate、margin_normalized。"
                     "仅完整、无身份冲突且最小值唯一时 status=ok 并提供 closest_category；"
                     "缺失、不可测或并列时不得从部分排序强行选择。"),
        args_schema_ref="reference:str, candidate_categories:list[str]",
        returns_schema_ref="dict",
        cost_estimate_ms=60.0,
        source_default="real",
        requires_artifacts=["point_cloud", "objects"],
        requires_evidence=[EV_GEOMETRY, EV_DETECTION, EV_GROUNDING],
        tolerates_degraded=[EV_GEOMETRY, EV_DETECTION, EV_GROUNDING],
    )
)
def relative_distance_rank(handle: SceneHandle, reference: str,
                           candidate_categories: list[str]) -> dict:
    """参照对象 → 候选类别（取最近实例）的距离排序（v7 §2.2）。

    题面形如「which of these objects (a, b, c, d) is the closest to the X?」：
    X 是参照对象，a/b/c/d 是候选类别。官方口径是「X 到各候选的最近距离」，
    **不是**相机到候选的距离 —— v6 曾把后者写成官方口径，导致该类题系统性答错。
    """
    if (not isinstance(candidate_categories, (list, tuple))
            or len(candidate_categories) < 2
            or not all(isinstance(c, str) and c.strip() for c in candidate_categories)):
        raise DomainValueError("relative_distance_rank",
                               "candidate_categories 必须包含至少两个非空类别名",
                               args={"candidate_categories": candidate_categories})
    categories = [c.strip() for c in candidate_categories]
    if any(matches(a, b) for i, a in enumerate(categories) for b in categories[i + 1:]):
        raise DomainValueError("relative_distance_rank", "候选类别重复或同义，无法唯一排名")
    ref = _get_object(handle, reference, "relative_distance_rank")
    ref_pts = handle.object_points(ref.obj_id)
    ranking: list[dict] = []
    per_candidate: dict[str, Optional[float]] = dict.fromkeys(categories)
    candidates: list[dict] = []
    missing: list[str] = []
    invalid: list[str] = []
    flags: set[str] = set()
    identities: dict[tuple[str, str], str] = {("object", ref.obj_id): "reference"}
    if ref.track_id:
        identities[("track", ref.track_id)] = "reference"
    ambiguous = False
    for cat in categories:
        oids = handle.list_objects_by_name(cat)
        detail = {"category": cat, "n_instances": len(oids), "instances": [],
                  "status": "missing" if not oids else "ok"}
        candidates.append(detail)
        if not oids:
            missing.append(cat)
            flags.add("category_missing:" + cat)
            continue
        for oid in oids:
            obj = handle.get_object(oid)
            keys = [("object", oid)] + ([("track", obj.track_id)] if obj.track_id else [])
            collision = any(k in identities and identities[k] != cat for k in keys)
            for key in keys:
                identities[key] = cat
            instance = {"obj_id": oid, "track_id": obj.track_id,
                        "grounding_status": obj.grounding_status,
                        "duplicate_suspect": bool(obj.duplicate_suspect),
                        "distance_normalized": None, "n_valid_points": 0,
                        "degradation_flags": [], "status": "ok"}
            detail["instances"].append(instance)
            if collision:
                ambiguous = True
                detail["status"] = instance["status"] = "ambiguous_grounding"
                instance["degradation_flags"] = ["identity_collision"]
                flags.add("identity_collision")
                continue
            res = robust_distance_between_pointsets(
                ref_pts, handle.object_points(oid),
                metric_scale=None,
                params=DistancePrimitiveParams(),
                duplicate_suspect=bool(ref.duplicate_suspect or obj.duplicate_suspect))
            d = res.distance_normalized
            primitive_flags = list(res.degradation_flags or [])
            instance.update(
                distance_normalized=float(d) if d is not None and np.isfinite(d) else None,
                n_valid_points=int(res.n_valid_points),
                n_nn_samples=int(getattr(res, "n_nn_samples", 0) or 0),
                degradation_flags=primitive_flags, audit=res.audit)
            flags.update(primitive_flags)
            if (d is None or not np.isfinite(d) or d < 0 or res.n_valid_points <= 0
                    or UNCERTAIN_DISTANCE_FLAGS.intersection(primitive_flags)):
                instance["status"] = "uncertain_geometry"
                if detail["status"] == "ok":
                    detail["status"] = "uncertain_geometry"
                if cat not in invalid:
                    invalid.append(cat)
        # 未测实例仍可能更近，不能把其余实例的最小值当成全类最小值。
        if detail["status"] != "ok":
            continue
        best = min(detail["instances"], key=lambda i: (i["distance_normalized"], i["obj_id"]))
        per_candidate[cat] = best["distance_normalized"]
        ranking.append({"category": cat, "obj_id": best["obj_id"],
                        "distance_normalized": best["distance_normalized"],
                        "n_instances_in_category": len(oids)})
    ranking.sort(key=lambda r: (r["distance_normalized"], r["category"]))
    tied = ([r["category"] for r in ranking
             if r["distance_normalized"] == ranking[0]["distance_normalized"]]
            if len(ranking) >= 2 else [])
    tied = tied if len(tied) > 1 else []
    status = ("ambiguous_grounding" if ambiguous else
              "incomplete_candidates" if missing else
              "uncertain_geometry" if invalid or tied else "ok")
    if tied:
        flags.add("tied_minimum")
    margin = (ranking[1]["distance_normalized"] - ranking[0]["distance_normalized"]
              if len(ranking) == len(categories) else None)
    return {
        "contract_version": RANKING_CONTRACT_VERSION,
        "status": status, "requested_categories": categories, "candidates": candidates,
        "reference": {"obj_id": ref.obj_id, "category_name": ref.category_name,
                      "track_id": ref.track_id, "duplicate_suspect": bool(ref.duplicate_suspect)},
        "ranking": ranking,
        "closest_category": ranking[0]["category"] if status == "ok" else None,
        "per_candidate": per_candidate,
        "categories_without_detection": missing,
        "categories_with_invalid_geometry": invalid,
        "tied_categories": tied, "margin_normalized": margin,
        "uncertainty_policy": "exact_ties_only; near_tie_threshold_not_calibrated",
        "definition": "reference_object_to_nearest_candidate_instance_v7",
        "quantile_q": DistancePrimitiveParams().quantile_q,
        "degradation_flags": sorted(flags),
    }


@REGISTRY.register(
    ToolSpec(
        name="object_distance_m",
        description=("**绝对距离题（object_abs_distance）用**：题面点名的两个对象之间"
                     "的最近距离，**单位米**。参数 a、b 是对象 id 或类别名。"
                     "返回 {'distance_m': D, 'distance_normalized': d, ...}；"
                     "`distance_m` 单位是米（题面要厘米时 ×100）。**不是**相机到"
                     "对象的距离。需要米制尺度；尺度不可用时本 Tool 不出现"),
        args_schema_ref="obj_a:str, obj_b:str",
        returns_schema_ref="dict",
        cost_estimate_ms=30.0,
        source_default="real",
        requires_artifacts=["point_cloud", "objects", "scale"],
        requires_evidence=[EV_GEOMETRY, EV_METRIC, EV_DETECTION, EV_GROUNDING],
        tolerates_degraded=[EV_GEOMETRY, EV_DETECTION, EV_GROUNDING],
    )
)
def object_distance_m(handle: SceneHandle, obj_a: str, obj_b: str) -> dict:
    """两对象最近距离（米制）——绝对距离题的官方口径（v7 §4/§9.2）。"""
    a = _get_object(handle, obj_a, "object_distance_m")
    b = _get_object(handle, obj_b, "object_distance_m")
    if a.obj_id == b.obj_id:
        raise DomainValueError("object_distance_m",
                               f"obj_a 与 obj_b 是同一对象: {obj_a}",
                               args={"obj_a": obj_a, "obj_b": obj_b})
    scale = _metric_scale_or_fail(handle, "object_distance_m")
    res = robust_distance_between_pointsets(
        handle.object_points(a.obj_id), handle.object_points(b.obj_id),
        metric_scale=scale, params=DistancePrimitiveParams(),
        duplicate_suspect=bool(a.duplicate_suspect or b.duplicate_suspect))
    d_m = _require_metric(res.distance_metric, "object_distance_m",
                          f"{a.category_name} 与 {b.category_name} 的最近距离（米）")
    return {
        "distance_m": d_m,
        "distance_normalized": res.distance_normalized,
        "distance_metric": res.distance_metric,
        "a": {"obj_id": a.obj_id, "category_name": a.category_name},
        "b": {"obj_id": b.obj_id, "category_name": b.category_name},
        "definition": "pointcloud_surface_nearest_quantile",
        "quantile_q": res.quantile_q,
        "voxel_size": res.voxel_size,
        "n_valid_points": int(res.n_valid_points),
        "degradation_flags": _degraded_property(
            res, metric_ok=res.distance_metric is not None),
    }


# ----------------------------------------------------------------- 方向 ----
@REGISTRY.register(
    ToolSpec(
        name="relative_direction_of",
        description=("站在 observer 处、**面向 facing_at**，判断 target 相对我的方位。"
                     "参数是**对象 id 或类别名**（如 'obj_3' / 'whiteboard'），内部取"
                     "各自质心：facing 方向 = facing_at 质心 − observer 质心。"
                     "`difficulty` **必须按题面难度传**（默认为 medium），因为三个难度的"
                     "**选项集合不同**："
                     "easy → 只返回 left/right；"
                     "medium → left/right/back（转身 ≥135° 才算 back）；"
                     "hard → front-left/front-right/back-left/back-right。"
                     "返回 dict {'direction': 'left', 'difficulty': 'medium', "
                     "'theta_deg': -92.1, 'world_up_used': [...], "
                     "'handedness_used': 'right'}；**答案取 `result['direction']`**，"
                     "再按题面选项文本映射到选项字母。方向题优先用它"),
        args_schema_ref="observer_id:str, facing_at_id:str, target_id:str, difficulty:str='medium'",
        returns_schema_ref="dict",
        cost_estimate_ms=2.0,
        source_default="real",
        requires_artifacts=["objects"],
        requires_evidence=[EV_GEOMETRY, EV_WORLD, EV_DETECTION, EV_GROUNDING],
        # §9.8：world_up 为 degraded 时仍给出方向（带 evidence_degraded 标记）——
        # 真实数据上 pose-only 的 up 一致性很难稳定 ≥0.9，不容忍会让方向题直接不可答
        tolerates_degraded=[EV_GEOMETRY, EV_WORLD, EV_DETECTION, EV_GROUNDING],
    )
)
def relative_direction_of(handle: SceneHandle, observer_id: str,
                          facing_at_id: str, target_id: str,
                          difficulty: str = "medium") -> dict:
    """按对象 id 解析参照系后的相对方位（§9.3/§9.8）。

    选项集合由 `difficulty` 决定（§9.3）：**不得**用一个简化输出覆盖三个模板 ——
    实测 medium 题面的选项是「left/right/back」，旧实现返回的 `"front"` 不在选项里，
    模型只能硬猜一个，是该类题系统性答错的主因之一。

    **world_up/handedness 缺失或非法 → fail-closed（抛错）**，不退回无符号启发式。
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
            world_up=handle.world_up, handedness=handle.handedness,
            difficulty=difficulty)
    except ValueError as exc:
        # §9.8：世界系约定缺失/非法 → fail-closed（DomainValueError，不猜）
        raise DomainValueError("relative_direction_of", str(exc),
                               args={"observer_id": observer_id,
                                     "facing_at_id": facing_at_id,
                                     "target_id": target_id,
                                     "difficulty": difficulty}) from exc
    return {
        "direction": direction,
        "difficulty": str(difficulty).strip().lower(),
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
        tolerates_degraded=[EV_DETECTION, EV_GROUNDING],
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
        tolerates_degraded=[EV_GEOMETRY, EV_WORLD, EV_DETECTION],
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
        tolerates_degraded=[EV_DETECTION],
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
        tolerates_degraded=[EV_GEOMETRY],
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
