"""M18 5 类空间 Metamorphic Relation（MR）变换：

viewpoint_change / object_permutation / rigid_transform / unit_change / occlusion_dropframe。
对可计算部分做真实 MR 检查：单位变换 m↔cm 数值等比缩放；刚体平移下相对距离不变。
MR 容差 TODO_CALIBRATE。
"""

from __future__ import annotations

import uuid

import numpy as np

from skill3d.schemas import MetamorphicTransform

MR_TOLERANCE = 1e-6  # TODO_CALIBRATE：数值 MR 容差

_UNIT_SCALE = {"m": 1.0, "cm": 100.0, "mm": 1000.0}


def make_transform(kind: str, params: dict, expected_relation: str) -> MetamorphicTransform:
    return MetamorphicTransform(
        transform_id=f"mr-{uuid.uuid4().hex[:12]}",
        kind=kind,  # type: ignore[arg-type]
        params=params,
        expected_relation=expected_relation,
    )


# ---------- 各 MR 的可计算检查 ----------

def check_unit_change(value: float, from_unit: str, to_unit: str,
                      transformed_value: float, tol: float = MR_TOLERANCE) -> bool:
    """unit_change：数值答案随单位等比缩放（m↔cm：×100）。"""
    ratio = _UNIT_SCALE[to_unit] / _UNIT_SCALE[from_unit]
    return abs(transformed_value - value * ratio) <= tol * max(1.0, abs(value * ratio))


def apply_unit_change(value: float, from_unit: str, to_unit: str) -> float:
    """单位变换的真实实现：返回等比缩放后的期望值。"""
    return value * (_UNIT_SCALE[to_unit] / _UNIT_SCALE[from_unit])


def check_rigid_transform_invariance(points_a: np.ndarray, points_b: np.ndarray,
                                     translation=None, tol: float = MR_TOLERANCE) -> bool:
    """rigid_transform：刚体平移下相对距离不变（pairwise 距离矩阵不变）。"""
    pa = np.asarray(points_a, dtype=float)
    pb = np.asarray(points_b, dtype=float)
    if pa.shape != pb.shape:
        return False
    if translation is not None:
        pb = pb - np.asarray(translation, dtype=float)  # 撤销平移后应与原重合
    da = np.linalg.norm(pa[:, None, :] - pa[None, :, :], axis=-1)
    db = np.linalg.norm(pb[:, None, :] - pb[None, :, :], axis=-1)
    return bool(np.allclose(da, db, atol=tol))


def check_object_permutation_invariance(answer_a, answer_b) -> bool:
    """object_permutation：对象排列不改变集合类答案（如计数）。"""
    return answer_a == answer_b


def check_viewpoint_change(scene_answer_a, scene_answer_b) -> bool:
    """viewpoint_change：视点变化下相对方向/距离关系类答案不变（标量比较占位）。"""
    return scene_answer_a == scene_answer_b


def check_occlusion_dropframe(answer_full, answer_dropped,
                              allow_degraded: bool = True) -> bool:
    """occlusion_dropframe：缺帧后答案应稳定或显式降级拒答。"""
    if answer_dropped == "unanswerable":
        return allow_degraded  # 降级拒答是可接受行为
    return answer_full == answer_dropped


def run_metamorphic(kind: str, **kwargs) -> bool:
    """按 kind 分发 MR 检查。"""
    dispatch = {
        "unit_change": check_unit_change,
        "rigid_transform": check_rigid_transform_invariance,
        "object_permutation": check_object_permutation_invariance,
        "viewpoint_change": check_viewpoint_change,
        "occlusion_dropframe": check_occlusion_dropframe,
    }
    if kind not in dispatch:
        raise ValueError(f"未知 MR 类型: {kind}")
    return bool(dispatch[kind](**kwargs))
