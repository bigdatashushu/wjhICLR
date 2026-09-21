"""M18 5 类空间 Metamorphic Relation（MR）变换：

viewpoint_change / object_permutation / rigid_transform / unit_change / occlusion_dropframe。
对可计算部分做真实 MR 检查：单位变换 m↔cm 数值等比缩放；刚体平移下相对距离不变。
MR 容差 TODO_CALIBRATE。
"""

from __future__ import annotations

import uuid
from typing import Optional, Sequence

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


# ---------- 5 类 MR 的**变换侧**（变换 + 不变量配对，供 hypothesis 驱动）----------

def random_rotation(rng: np.random.Generator) -> np.ndarray:
    """均匀随机旋转（QR 分解法，行列式为 +1）。"""
    a = rng.normal(size=(3, 3))
    q, r = np.linalg.qr(a)
    q = q @ np.diag(np.sign(np.diag(r)))
    if np.linalg.det(q) < 0:
        q[:, 0] = -q[:, 0]
    return q


def apply_rigid_transform(points: np.ndarray, rotation: np.ndarray,
                          translation: np.ndarray) -> np.ndarray:
    """刚体变换：`R @ p + t`（MR：成对距离、边长、夹角不变）。"""
    p = np.asarray(points, dtype=float)
    return p @ np.asarray(rotation, dtype=float).T + np.asarray(translation, dtype=float)


def apply_viewpoint_change(c2w_list: np.ndarray, rotation: np.ndarray,
                           translation: np.ndarray) -> np.ndarray:
    """视角/世界系变换：对整条相机轨迹施加同一刚体变换 `T`。

    MR 期望：场景内几何关系（相对距离/方向、计数、尺寸）不变 —— 因为这是
    同一个物理场景换了一个世界坐标系，任何依赖绝对坐标的答案都应当不变。
    """
    c2w = np.asarray(c2w_list, dtype=float).copy()
    t = np.eye(4)
    t[:3, :3] = np.asarray(rotation, dtype=float)
    t[:3, 3] = np.asarray(translation, dtype=float)
    return np.stack([t @ m for m in c2w])


def permute_objects(objects: Sequence, order: Sequence[int]) -> list:
    """对象排列变换（MR：计数、类别集合、成对距离不变）。"""
    return [objects[i] for i in order]


def drop_frames(arrays: Sequence[np.ndarray], drop_idx: Sequence[int]) -> list[np.ndarray]:
    """遮挡/缺帧变换：按索引删除帧（MR：保留帧的几何量与计数不变，或显式降级拒答）。"""
    dropped = set(int(i) for i in drop_idx)
    return [a for i, a in enumerate(arrays) if i not in dropped]


def pairwise_distance_matrix(points: np.ndarray) -> np.ndarray:
    """成对欧氏距离矩阵（刚体/视角变换的 MR 观测量）。"""
    p = np.asarray(points, dtype=float)
    return np.linalg.norm(p[:, None, :] - p[None, :, :], axis=-1)


def invariant_holds(observed: float, expected: float,
                    rel_tol: float = 1e-6, abs_tol: float = 1e-9) -> bool:
    """MR 不变量判定的统一口径（相对+绝对容差；TODO_CALIBRATE）。"""
    return abs(float(observed) - float(expected)) <= max(abs_tol,
                                                         rel_tol * abs(float(expected)))


def find_violating_transform(program_fn, transform_fn, invariant_fn, params,
                             max_examples: int = 200) -> Optional[dict]:
    """在给定参数空间内搜索首个违反 MR 的变换（CEGIS 的 verify 步骤，确定性枚举）。

    `program_fn(params) -> 观测值`；`transform_fn(params) -> 被变换后的 params`；
    `invariant_fn(params, transformed_params) -> (观测值, 期望值)`。
    返回 `{"params": ..., "observed": ..., "expected": ...}` 或 None。
    搜索顺序为参数列表顺序（确定性，可复现）。
    """
    for p in params[:max_examples]:
        tp = transform_fn(p)
        observed, expected = invariant_fn(p, tp)
        if not invariant_holds(observed, expected):
            return {"params": p, "transformed": tp,
                    "observed": float(observed), "expected": float(expected)}
    return None
