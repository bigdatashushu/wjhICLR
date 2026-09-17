"""Geometry Verifier（§4 M11）：确定性几何校验，纯 numpy，不调任何 VLM（硬约束 1/13）。

校验项：no_negative_distance / inside_bbox / unit_consistent / reprojection 误差。
注：§5 schemas 未定义 GeometryVerifyResult，故本地定义（不改 schemas/）。
"""

from __future__ import annotations

import json
from typing import Optional

import numpy as np
from pydantic import BaseModel, ConfigDict

from skill3d.schemas import ProgramExecutionTrace
from skill3d.tools.scene_handle import SceneHandle

TH_REPROJ_PX = 2.0  # TODO_CALIBRATE 重投影误差阈值（像素）
BBOX_MARGIN = 0.05  # TODO_CALIBRATE 包围盒外扩比例


class GeometryVerifyResult(BaseModel):
    model_config = ConfigDict(extra="forbid", protected_namespaces=())

    passed: bool
    checks: dict[str, bool]
    violations: list[str]


def _iter_tool_values(trace: ProgramExecutionTrace, tool_name: str):
    for r in trace.results:
        if r.tool == tool_name and r.error is None:
            try:
                yield json.loads(r.value)
            except (json.JSONDecodeError, TypeError):
                continue


def check_no_negative_distance(trace: ProgramExecutionTrace) -> bool:
    """euclidean_distance / object_size_longest_dim / room_size_m2 结果必须非负。"""
    for name in ("euclidean_distance", "object_size_longest_dim", "room_size_m2"):
        for v in _iter_tool_values(trace, name):
            if isinstance(v, (int, float)) and v < 0:
                return False
    return True


def check_inside_bbox(
    trace: ProgramExecutionTrace, handle: SceneHandle, margin: float = BBOX_MARGIN
) -> bool:
    """reproject 调用的 3D 输入点必须落在场景包围盒内（外扩 margin）。"""
    try:
        bmin, bmax = handle.scene_bbox()
    except ValueError:
        return True  # 无对象场景无从判定，不否决
    extent = bmax - bmin
    lo, hi = bmin - margin * extent, bmax + margin * extent
    for r in trace.results:
        if r.tool == "reproject" and "p3d" in r.args:
            p = np.asarray(r.args["p3d"], dtype=np.float64)
            if p.shape != (3,) or np.any(p < lo) or np.any(p > hi):
                return False
    return True


def check_unit_consistent(trace: ProgramExecutionTrace, scale_known: bool) -> bool:
    """单位一致性：metric 尺度未知时不允许产出带绝对单位的数值答案（占位规则）。

    TODO_CALIBRATE: 更细粒度的单位推断规则待校准。
    """
    if scale_known:
        return True
    for v in _iter_tool_values(trace, "room_size_m2"):
        if isinstance(v, (int, float)) and v > 0:
            return False  # scale 未知却给出绝对面积 → 单位不一致
    return True


def check_reprojection(
    trace: ProgramExecutionTrace, handle: SceneHandle, threshold_px: float = TH_REPROJ_PX
) -> bool:
    """对 reproject 成功调用复核误差：重新计算并与记录值比较（确定性）。"""
    errs = []
    for r in trace.results:
        if r.tool == "reproject" and r.error is None:
            try:
                expected = json.loads(r.value)
                p = np.asarray(r.args["p3d"], dtype=np.float64)
                c2w = handle.get_c2w(int(r.args["frame_idx"]))
                K = handle.get_intrinsics(int(r.args["frame_idx"]))
                p_cam = np.linalg.inv(c2w) @ np.concatenate([p, [1.0]])
                uv = K @ p_cam[:3] / p_cam[2]
                errs.append(float(np.linalg.norm(uv[:2] - np.asarray(expected, dtype=np.float64))))
            except Exception:
                return False
    if not errs:
        return True  # 无重投影调用则该项通过
    return float(np.mean(errs)) < threshold_px


def geometry_verify(
    trace: ProgramExecutionTrace,
    handle: SceneHandle,
    answer: Optional[str] = None,
    threshold_px: float = TH_REPROJ_PX,
) -> GeometryVerifyResult:
    """§4 M11 伪代码的确定性实现（不调 VLM / GPT-6）。"""
    checks = {
        "no_negative_distance": check_no_negative_distance(trace),
        "inside_bbox": check_inside_bbox(trace, handle),
        "unit_consistent": check_unit_consistent(trace, handle.scale_known),
        "reprojection": check_reprojection(trace, handle, threshold_px),
    }
    violations = [k for k, v in checks.items() if not v]
    return GeometryVerifyResult(passed=not violations, checks=checks, violations=violations)
