"""确定性几何 Tool 真实实现（§4 M6，numpy/scipy）。

全部注册进全局 REGISTRY 并带 ToolSpec；统一签名 (handle: SceneHandle, **args)。
所有函数为确定性纯计算，不调任何 VLM / 外部服务。

**每个 Tool 显式声明 `requires_artifacts`**（硬约束 23）：取值域见
`tools.contract.KNOWN_ARTIFACTS`；执行期由 `REGISTRY.call_tool` 统一做
fail-closed 校验，产物缺失抛 `ArtifactUnavailableError`（绝不静默返回假值）。

域值错误（负距离 / 点在相机后方 / 退化向量 / 超包围盒 / 单位错）统一抛
`DomainValueError`，进 `ProgramExecutionTrace.error_code="domain_value"`。
"""

from __future__ import annotations

import numpy as np

from skill3d.schemas import ToolSpec

from .contract import DomainValueError
from .registry import REGISTRY
from .scene_handle import SceneHandle


def _as_point(p: list[float], name: str) -> np.ndarray:
    arr = np.asarray(p, dtype=np.float64)
    if arr.shape != (3,):
        raise DomainValueError(
            "argument_check", f"{name} 必须为 3 维点, 实际 shape={arr.shape}",
            args={"name": name})
    return arr


def _metric_scale_or_fail(handle: SceneHandle, tool: str) -> float:
    """米制 Tool 的世界→米换算系数（HC29）；缺失/非有限 → fail-closed。

    历史缺陷（2026-09-21 定位）：三个米制 Tool 直接返回**世界单位**数值，
    而 HC29 明确定义 `metric_scale` 为单位换算（`scale_ci_abs_m = metric_scale ×
    scale_ci_rel`，合成几何亦自述 depth 单位为米且 `metric_scale=1.0`）。
    未授权时这些 Tool 不可调用，故当前影响为零；但标定数据到位后若不换算，
    三个米制题型会系统性偏差 metric_scale 倍（本机实测 ~5.23，面积类 ~27 倍）。
    """
    v = handle.metric_scale
    if v is None or not np.isfinite(float(v)) or float(v) <= 0:
        raise DomainValueError(
            tool, f"缺少可用的 metric_scale（世界单位→米 的换算系数）：{v!r}；"
                  "未锚定/未标定时米制 Tool 不得以世界单位冒充米制（HC29/30）")
    return float(v)


def _up_axis(handle: SceneHandle) -> int:
    """地面平面判定：取场景包围盒 extent 最小的轴为竖直轴（室内场景启发式）。"""
    bmin, bmax = handle.scene_bbox()
    return int(np.argmin(bmax - bmin))


def _vertical_axis(handle: SceneHandle):
    """竖直轴（视觉兜底）：由包围盒最小 extent 猜；仅在无相机位姿时使用。

    该启发式**无符号**，无法区分上下 —— 正式判定一律优先用
    `SceneHandle.up_direction()`（相机位姿导出，含符号）。
    """
    bmin, bmax = handle.scene_bbox()
    return int(np.argmin(bmax - bmin))


def _get_object(handle: SceneHandle, object_id: str, tool: str):
    """取对象；不存在 → DomainValueError（域值错误，不是契约缺失）。"""
    try:
        return handle.get_object(object_id)
    except KeyError as exc:
        raise DomainValueError(tool, f"对象不存在: {object_id}（objects 产物可用）",
                               args={"object_id": object_id}) from exc


@REGISTRY.register(
    ToolSpec(
        name="euclidean_distance",
        description="计算两个 3D 点（世界系）的欧氏距离，返回非负浮点数",
        args_schema_ref="point_a:list[float], point_b:list[float]",
        returns_schema_ref="float",
        cost_estimate_ms=1.0,
        source_default="real",
        requires_artifacts=[],  # 纯算术：只依赖入参
    )
)
def euclidean_distance(handle: SceneHandle, point_a: list[float], point_b: list[float]) -> float:
    a = _as_point(point_a, "point_a")
    b = _as_point(point_b, "point_b")
    d = float(np.linalg.norm(a - b))
    if not np.isfinite(d) or d < 0:
        raise DomainValueError("euclidean_distance", f"距离非有限或为负: {d}",
                               args={"point_a": point_a, "point_b": point_b})
    return d


@REGISTRY.register(
    ToolSpec(
        name="relative_direction",
        description=("以 observer 为原点、facing 为前向，判定 target 在 "
                     "front/behind/left/right 哪个方向。**facing 是方向向量**"
                     "（世界系，只取地面平面分量），不是目标点；若已知朝向某物体，"
                     "请直接用 relative_direction_of（传对象 id 即可）。"),
        args_schema_ref="observer:list[float], facing:list[float], target:list[float]",
        returns_schema_ref="str",
        cost_estimate_ms=1.0,
        source_default="real",
        requires_artifacts=["objects"],  # 竖直轴由场景包围盒（对象 bbox）判定
    )
)
def relative_direction(
    handle: SceneHandle,
    observer: list[float],
    facing: list[float],
    target: list[float],
) -> str:
    """在**水平面**内判定 target 相对 observer/facing 的方位。

    2026-09-21 真实缺陷修复（两处，都会让方向系统性算错）：
    1. **竖直轴改为相机位姿导出**（`SceneHandle.up_direction`）。旧实现取包围盒
       最小 extent 轴当竖直轴，实测在长条形房间里选到了水平轴 z，于是把竖直方向
       混进了 left/right 判定；
    2. **左右判据改为与"上"方向一致的叉积**：right ⟺ (f × d)·u < 0
       （u = 世界上方向；`right = forward × up` 的右手定则）。旧实现用固定的
       `f_x·d_z − f_z·d_x`，等价于假定某个特定手性，与真实世界系符号无关。
    水平分量 = 相对 u 的垂面投影（对倾斜相机也成立）。
    """
    obs = _as_point(observer, "observer")
    fac = _as_point(facing, "facing")
    tgt = _as_point(target, "target")
    u = handle.up_direction()
    if u is None:
        # 无位姿产物（罕见）：退回旧的轴向启发式，左右按 x-z 平面约定
        up_ax = _vertical_axis(handle)
        ground = [i for i in range(3) if i != up_ax]
        f2 = fac[ground]
        d2 = (tgt - obs)[ground]
        f_norm, d_norm = np.linalg.norm(f2), np.linalg.norm(d2)
        if f_norm < 1e-9:
            raise DomainValueError("relative_direction", "facing 向量在地面平面上退化为零",
                                   args={"facing": facing})
        if d_norm < 1e-9:
            raise DomainValueError("relative_direction", "target 与 observer 重合，方向未定义",
                                   args={"observer": observer, "target": target})
        f2, d2 = f2 / f_norm, d2 / d_norm
        cos_theta = float(np.dot(f2, d2))
        if cos_theta >= 0.7071:  # TODO_CALIBRATE 前向半角阈值（45°）
            return "front"
        if cos_theta <= -0.7071:  # TODO_CALIBRATE
            return "behind"
        cross = float(f2[0] * d2[1] - f2[1] * d2[0])
        return "left" if cross > 0 else "right"
    f = fac - np.dot(fac, u) * u
    d = (tgt - obs) - np.dot(tgt - obs, u) * u
    f_norm, d_norm = float(np.linalg.norm(f)), float(np.linalg.norm(d))
    if f_norm < 1e-9:
        raise DomainValueError("relative_direction", "facing 向量在水平面上退化为零",
                               args={"facing": facing})
    if d_norm < 1e-9:
        raise DomainValueError("relative_direction", "target 与 observer 重合，方向未定义",
                               args={"observer": observer, "target": target})
    f, d = f / f_norm, d / d_norm
    cos_theta = float(np.dot(f, d))
    if cos_theta >= 0.7071:  # TODO_CALIBRATE 前向半角阈值（45°）
        return "front"
    if cos_theta <= -0.7071:  # TODO_CALIBRATE：题目对"back"的定义 = 需转身 ≥135°，与此一致
        return "behind"
    # right = forward × up（右手定则）→ (f × d)·u < 0 即 target 在右手侧
    triple = float(np.dot(np.cross(f, d), u))
    return "right" if triple < 0 else "left"


@REGISTRY.register(
    ToolSpec(
        name="object_size_longest_dim",
        description="对象 bbox 最长边长度（**米制**，世界系）",
        args_schema_ref="object_id:str",
        returns_schema_ref="float",
        cost_estimate_ms=1.0,
        source_default="real",
        # v4 HC33：尺寸 = 米制量，必须同时依赖 objects 与 scale；相对单位的"尺寸"
        # 不得作为答案返回（旧的"scale 未知时给相对单位"语义已在 v4 废止）
        requires_artifacts=["objects", "scale"],
        supported_metric_tasks=["object_size_estimation"],
    )
)
def object_size_longest_dim(handle: SceneHandle, object_id: str) -> float:
    obj = _get_object(handle, object_id, "object_size_longest_dim")
    if len(obj.bbox) != 6:
        raise DomainValueError("object_size_longest_dim",
                               f"对象 {object_id} bbox 维度异常: {obj.bbox}",
                               args={"object_id": object_id})
    extents = np.asarray(obj.bbox[3:], dtype=np.float64) - np.asarray(obj.bbox[:3], dtype=np.float64)
    if np.any(extents < 0):
        raise DomainValueError("object_size_longest_dim",
                               f"对象 {object_id} bbox 上下界反序（负边长）",
                               args={"object_id": object_id})
    # HC29：世界单位 → 米（未标定则 fail-closed，不拿世界单位冒充米制）
    return float(np.max(extents)) * _metric_scale_or_fail(handle, "object_size_longest_dim")


@REGISTRY.register(
    ToolSpec(
        name="object_distance_meters",
        description="两个对象质心之间的**米制**距离（米）；用于绝对距离题型",
        args_schema_ref="object_a:str, object_b:str",
        returns_schema_ref="float",
        cost_estimate_ms=1.0,
        source_default="real",
        # v4 HC33：绝对距离对尺度误差最敏感，仅在 `object_abs_distance` 被授权
        # （即 scale_confidence=high）时可调用
        requires_artifacts=["objects", "scale"],
        supported_metric_tasks=["object_abs_distance"],
    )
)
def object_distance_meters(handle: SceneHandle, object_a: str, object_b: str) -> float:
    a = np.asarray(_get_object(handle, object_a, "object_distance_meters").centroid_world,
                   dtype=np.float64)
    b = np.asarray(_get_object(handle, object_b, "object_distance_meters").centroid_world,
                   dtype=np.float64)
    if a.shape != (3,) or b.shape != (3,):
        raise DomainValueError("object_distance_meters",
                               f"质心维度异常: {a.shape} / {b.shape}",
                               args={"object_a": object_a, "object_b": object_b})
    d = float(np.linalg.norm(a - b))
    if not np.isfinite(d) or d < 0:
        raise DomainValueError("object_distance_meters", f"距离非有限或为负: {d}",
                               args={"object_a": object_a, "object_b": object_b})
    # HC29：世界单位 → 米（未标定则 fail-closed）
    return d * _metric_scale_or_fail(handle, "object_distance_meters")


@REGISTRY.register(
    ToolSpec(
        name="room_size_m2",
        description="房间地面面积估计（**平方米**）：场景包围盒在地面平面上的两轴乘积",
        args_schema_ref="",
        returns_schema_ref="float",
        cost_estimate_ms=5.0,
        source_default="real",
        # v4 HC33：面积是米制量 → 依赖 objects + scale，逐题授权 room_size_estimation
        requires_artifacts=["objects", "scale"],
        supported_metric_tasks=["room_size_estimation"],
    )
)
def room_size_m2(handle: SceneHandle) -> float:
    bmin, bmax = handle.scene_bbox()
    extents = bmax - bmin
    up = int(np.argmin(extents))
    ground = [extents[i] for i in range(3) if i != up]
    area = float(ground[0] * ground[1])
    if not np.isfinite(area) or area <= 0:
        raise DomainValueError("room_size_m2", f"地面面积非正/非有限: {area}")
    # HC29：面积按**平方**换算（世界单位² → m²）；未标定则 fail-closed
    scale = _metric_scale_or_fail(handle, "room_size_m2")
    return area * scale * scale


@REGISTRY.register(
    ToolSpec(
        name="reproject",
        description="世界系 3D 点经 c2w 与 K 重投影到指定帧像素坐标 [u, v]；点在相机后方时抛错",
        args_schema_ref="p3d:list[float], frame_idx:int",
        returns_schema_ref="list",
        cost_estimate_ms=1.0,
        source_default="real",
        requires_artifacts=["poses", "intrinsics"],
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
        name="exists_in_scene",
        description="按 instance_id 或 class_hint（大小写不敏感）判断对象是否存在于场景",
        args_schema_ref="name:str",
        returns_schema_ref="bool",
        cost_estimate_ms=1.0,
        source_default="real",
        # 硬约束 23：objects 不可用时抛 ArtifactUnavailableError；
        # 返回 False 当且仅当 objects 可用且真无此实例
        requires_artifacts=["objects"],
    )
)
def exists_in_scene(handle: SceneHandle, name: str) -> bool:
    return bool(handle.exists(name))


@REGISTRY.register(
    ToolSpec(
        name="list_objects",
        description=("列出场景中的对象 id（形如 obj_0/obj_1，**id 里不含类别名**）。"
                     "按类别筛选必须传 class_hint（子串匹配，如 list_objects('chair')）；"
                     "不要写 'chair' in obj_id 这类判断（id 只有 obj_数字）。"
                     "计数题请配合 count_objects 使用"),
        args_schema_ref="class_hint:str=''",
        returns_schema_ref="list[str]",
        cost_estimate_ms=1.0,
        source_default="real",
        requires_artifacts=["objects"],
    )
)
def list_objects(handle: SceneHandle, class_hint: str = "") -> list[str]:
    if not handle.objects_materialized:
        from skill3d.tools.contract import ArtifactUnavailableError

        raise ArtifactUnavailableError(
            tool="list_objects", missing=["objects"], available=[],
            route="", args={"class_hint": class_hint})
    return handle.list_objects_by_hint(class_hint)


@REGISTRY.register(
    ToolSpec(
        name="count_objects",
        description=("统计场景中某类对象的实例数（class_hint 子串匹配，如 "
                     "'computer tower'/'chair'）。计数题**必须**用它，不要用 "
                     "exists_in_scene 累加（那是布尔值）"),
        args_schema_ref="class_hint:str",
        returns_schema_ref="int",
        cost_estimate_ms=1.0,
        source_default="real",
        requires_artifacts=["objects"],
    )
)
def count_objects(handle: SceneHandle, class_hint: str) -> int:
    if not handle.objects_materialized:
        from skill3d.tools.contract import ArtifactUnavailableError

        raise ArtifactUnavailableError(
            tool="count_objects", missing=["objects"], available=[],
            route="", args={"class_hint": class_hint})
    return len(handle.list_objects_by_hint(class_hint))


@REGISTRY.register(
    ToolSpec(
        name="object_centroid",
        description="对象质心（世界系 3D 点）",
        args_schema_ref="object_id:str",
        returns_schema_ref="list",
        cost_estimate_ms=1.0,
        source_default="real",
        requires_artifacts=["objects"],
    )
)
def object_centroid(handle: SceneHandle, object_id: str) -> list[float]:
    obj = _get_object(handle, object_id, "object_centroid")
    c = np.asarray(obj.centroid_world, dtype=np.float64)
    if c.shape != (3,):
        raise DomainValueError("object_centroid",
                               f"对象 {object_id} centroid 维度异常: {obj.centroid_world}",
                               args={"object_id": object_id})
    return [float(x) for x in c]


@REGISTRY.register(
    ToolSpec(
        name="relative_direction_of",
        description=("站在 observer 处、**面向 facing_at**，判断 target 在 "
                     "front/behind/left/right 哪个方向。三个参数都是**对象 id 或类别名**"
                     "（如 'obj_3' / 'whiteboard'），内部取各自质心："
                     "facing 方向 = facing_at 质心 − observer 质心。"
                     "方向题（left/right/back 之类）优先用它，不要自己拼 facing 向量。"),
        args_schema_ref="observer:str, facing_at:str, target:str",
        returns_schema_ref="str",
        cost_estimate_ms=1.0,
        source_default="real",
        requires_artifacts=["objects"],
    )
)
def relative_direction_of(handle: SceneHandle, observer: str, facing_at: str,
                          target: str) -> str:
    """按对象 id 解析参照系后的相对方位（避免"把目标点当方向向量"的语义误用）。"""
    obs = np.asarray(_get_object(handle, observer, "relative_direction_of").centroid_world,
                     dtype=np.float64)
    fac = np.asarray(_get_object(handle, facing_at, "relative_direction_of").centroid_world,
                     dtype=np.float64)
    tgt = np.asarray(_get_object(handle, target, "relative_direction_of").centroid_world,
                     dtype=np.float64)
    if obs.shape != (3,) or fac.shape != (3,) or tgt.shape != (3,):
        raise DomainValueError("relative_direction_of",
                               f"质心维度异常: {obs.shape}/{fac.shape}/{tgt.shape}")
    return relative_direction(handle, obs.tolist(), (fac - obs).tolist(), tgt.tolist())


@REGISTRY.register(
    ToolSpec(
        name="object_visible_frames",
        description=("某对象在统一 FrameSet 中**可见**的帧槽位序号（升序 0..31；空列表 = "
                     "全程无有效掩码）。用于物体首次出现顺序这类逐帧可见性题型："
                     "对每个候选对象取 min(可见帧) 再排序即可。"),
        args_schema_ref="object_id:str",
        returns_schema_ref="list[int]",
        cost_estimate_ms=1.0,
        source_default="real",
        requires_artifacts=["objects"],
    )
)
def object_visible_frames(handle: SceneHandle, object_id: str) -> list[int]:
    obj = _get_object(handle, object_id, "object_visible_frames")
    return [int(i) for i in (getattr(obj, "visible_frames", None) or [])]
