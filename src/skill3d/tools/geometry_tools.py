"""确定性几何 Tool 真实实现（§4 M6，numpy/scipy）。

全部注册进全局 REGISTRY 并带 ToolSpec；统一签名 (handle: SceneHandle, **args)。
所有函数为确定性纯计算，不调任何 VLM / 外部服务。
"""

from __future__ import annotations

import numpy as np

from skill3d.schemas import ToolSpec

from .registry import REGISTRY
from .scene_handle import SceneHandle


def _as_point(p: list[float], name: str) -> np.ndarray:
    arr = np.asarray(p, dtype=np.float64)
    if arr.shape != (3,):
        raise ValueError(f"{name} 必须为 3 维点, 实际 shape={arr.shape}")
    return arr


def _up_axis(handle: SceneHandle) -> int:
    """地面平面判定：取场景包围盒 extent 最小的轴为竖直轴（室内场景启发式）。"""
    bmin, bmax = handle.scene_bbox()
    return int(np.argmin(bmax - bmin))


@REGISTRY.register(
    ToolSpec(
        name="euclidean_distance",
        description="计算两个 3D 点（世界系）的欧氏距离，返回非负浮点数",
        args_schema_ref="point_a:list[float], point_b:list[float]",
        returns_schema_ref="float",
        cost_estimate_ms=1.0,
        source_default="real",
    )
)
def euclidean_distance(handle: SceneHandle, point_a: list[float], point_b: list[float]) -> float:
    a = _as_point(point_a, "point_a")
    b = _as_point(point_b, "point_b")
    return float(np.linalg.norm(a - b))


@REGISTRY.register(
    ToolSpec(
        name="relative_direction",
        description="以 observer 为原点、facing 为前向，判定 target 在 front/behind/left/right 哪个方向",
        args_schema_ref="observer:list[float], facing:list[float], target:list[float]",
        returns_schema_ref="str",
        cost_estimate_ms=1.0,
        source_default="real",
    )
)
def relative_direction(
    handle: SceneHandle,
    observer: list[float],
    facing: list[float],
    target: list[float],
) -> str:
    obs = _as_point(observer, "observer")
    fac = _as_point(facing, "facing")
    tgt = _as_point(target, "target")
    up = _up_axis(handle)
    ground = [i for i in range(3) if i != up]
    f = fac[ground]
    f_norm = np.linalg.norm(f)
    if f_norm < 1e-9:
        raise ValueError("facing 向量在地面平面上退化为零")
    f = f / f_norm
    d = (tgt - obs)[ground]
    d_norm = np.linalg.norm(d)
    if d_norm < 1e-9:
        raise ValueError("target 与 observer 重合，方向未定义")
    d = d / d_norm
    cos_theta = float(np.dot(f, d))
    if cos_theta >= 0.7071:  # TODO_CALIBRATE 前向半角阈值（45°）
        return "front"
    if cos_theta <= -0.7071:  # TODO_CALIBRATE
        return "behind"
    cross = float(f[0] * d[1] - f[1] * d[0])
    return "left" if cross > 0 else "right"


@REGISTRY.register(
    ToolSpec(
        name="object_size_longest_dim",
        description="对象 bbox 最长边长度（世界系单位；metric scale 未知时为相对单位）",
        args_schema_ref="object_id:str",
        returns_schema_ref="float",
        cost_estimate_ms=1.0,
        source_default="real",
    )
)
def object_size_longest_dim(handle: SceneHandle, object_id: str) -> float:
    obj = handle.get_object(object_id)
    if len(obj.bbox) != 6:
        raise ValueError(f"对象 {object_id} bbox 维度异常")  # TODO bbox 格式约定
    extents = np.asarray(obj.bbox[3:], dtype=np.float64) - np.asarray(obj.bbox[:3], dtype=np.float64)
    return float(np.max(extents))


@REGISTRY.register(
    ToolSpec(
        name="room_size_m2",
        description="房间地面面积估计：场景包围盒在地面平面上的两轴乘积",
        args_schema_ref="",
        returns_schema_ref="float",
        cost_estimate_ms=5.0,
        source_default="real",
    )
)
def room_size_m2(handle: SceneHandle) -> float:
    bmin, bmax = handle.scene_bbox()
    extents = bmax - bmin
    up = int(np.argmin(extents))
    ground = [extents[i] for i in range(3) if i != up]
    return float(ground[0] * ground[1])


@REGISTRY.register(
    ToolSpec(
        name="reproject",
        description="世界系 3D 点经 c2w 与 K 重投影到指定帧像素坐标 [u, v]；点在相机后方时抛错",
        args_schema_ref="p3d:list[float], frame_idx:int",
        returns_schema_ref="list",
        cost_estimate_ms=1.0,
        source_default="real",
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
        raise ValueError(f"点在相机后方或成像平面上 (z={z})，无法重投影")
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
    )
)
def exists_in_scene(handle: SceneHandle, name: str) -> bool:
    return bool(handle.exists(name))


@REGISTRY.register(
    ToolSpec(
        name="object_centroid",
        description="对象质心（世界系 3D 点）",
        args_schema_ref="object_id:str",
        returns_schema_ref="list",
        cost_estimate_ms=1.0,
        source_default="real",
    )
)
def object_centroid(handle: SceneHandle, object_id: str) -> list[float]:
    obj = handle.get_object(object_id)
    c = np.asarray(obj.centroid_world, dtype=np.float64)
    if c.shape != (3,):
        raise ValueError(f"对象 {object_id} centroid 维度异常")
    return [float(x) for x in c]
