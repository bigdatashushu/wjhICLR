"""Geometry Verifier（§4 M11）：确定性几何校验，纯 numpy，不调任何 VLM（硬约束 1/13）。

校验项：no_negative_distance / unit_consistent / world_frame / inside_bbox / reprojection。

v6 迁移（§5.2/§5.3/§13/§20）：

1. **`scale_known` 事实源已删除**（§20：多锚点 + conformal 校准池整体废止）。
   "能不能给绝对单位数值"改由 v6 证据事实判定，唯一事实源是
   `SceneHandle.metric_gate_passed` ∧ `SceneState.metric_task_authorized(question_type)`
   （= 题型是米制 ∧ `MetricEvidenceGateResult.gate_passed` ∧
   `question_tool_scope=metric_enabled`，§13.2）。任何"用别的代理量推断尺度已知"的写法
   都是实现错误（§10.4：G5 类代理值严禁冒充）。
2. **世界系检查 fail-closed**：方向 / 连通性类 Tool 与"世界系 3D 点的 reproject"都依赖
   `world_up` + `handedness` 契约（M3 落盘、M4 校验，§5.2）。契约缺失时这些检查
   **判失败**（`world_frame=False`），绝不"缺就跳过"——跳过等于把错误坐标当对答案放行。
3. **G5 重投影残差永久 `not_available`**（§10.4/§20）：本模块的 `reprojection` 项是对
   **自算值**的确定性复核（同一 c2w/K 重算并比较），它**不是** BA 重投影残差，也**不得**
   被当作 G5 上报。M4 主门另有其人（warp 内点率 + 点云重叠率，D11）。

注：§5 schemas 未定义 GeometryVerifyResult，故本地定义（不改 schemas/）。
"""

from __future__ import annotations

import json
from typing import Any, Iterable, Optional

import numpy as np
from pydantic import BaseModel, ConfigDict

from skill3d.schemas import ProgramExecutionTrace
from skill3d.tools.scene_handle import SceneHandle

TH_REPROJ_PX = 2.0  # TODO_CALIBRATE 重投影误差阈值（像素）
BBOX_MARGIN = 0.05  # TODO_CALIBRATE 包围盒外扩比例

# 依赖**世界系契约**（`world_up` + `handedness`）的 Tool（§9.8/§9.10）。
# 口径与 `ToolSpec.requires_evidence` 含 `world_frame` 一致：契约缺失时这些调用
# 产出的方向/可达结论不可信 → 对应的几何检查必须 fail-closed。
WORLD_FRAME_TOOLS: tuple[str, ...] = ("relative_direction_of", "connectivity_graph")

# 米制量键：Tool 输出里若出现这些**非空**键，说明该次调用产出了绝对单位数值
# （米）。它们只在米制证据被授权时才允许出现（§13.2 fail-closed）。
_METRIC_VALUE_KEYS: dict[str, str] = {
    "plane_fit_room_size": "room_area_m2",
    "robust_distance": "distance_metric",
    "camera_object_distance": "distance_metric",
    "surface_distance_between_objects": "surface_distance_metric",
    "object_3d_extent": "extent_metric",
    "object_centroid": "centroid_metric",
}

# 合法手性取值（与 `reconstruction_gate/world_frame.py` 同一词汇表）
KNOWN_HANDEDNESS: frozenset[str] = frozenset({"right", "left"})


class GeometryVerifyResult(BaseModel):
    model_config = ConfigDict(extra="forbid", protected_namespaces=())

    passed: bool
    checks: dict[str, bool]
    violations: list[str]
    # 只读诊断（进 trace，便于"M11 为什么这么判"的事后归因；不参与判定）
    notes: list[str] = []
    metric_evidence_authorized: bool = False
    world_frame_available: bool = False


def _iter_tool_values(trace: ProgramExecutionTrace, tool_name: str) -> Iterable[Any]:
    for r in trace.results:
        if r.tool == tool_name and r.error is None:
            try:
                yield json.loads(r.value)
            except (json.JSONDecodeError, TypeError):
                continue


def _finite_number(v: Any) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool) and bool(
        np.isfinite(float(v)))


def _world_frame_available(handle: SceneHandle) -> bool:
    """世界系契约是否可用（`world_up` 合法非零单位向量 ∧ `handedness` 已知取值）。"""
    if handle.world_up is None:
        return False
    hand = handle.handedness
    return hand is not None and str(hand).lower() in KNOWN_HANDEDNESS


def metric_evidence_authorized(handle: SceneHandle) -> bool:
    """§13.2 米制证据是否被授权（v6 唯一事实源，取代 v5 `scale_known`）。

    = `MetricEvidenceGateResult.gate_passed` ∧ `metric_task_authorized(本题型)`。
    任一不成立 → 本次 episode 无权产出/使用绝对单位数值（fail-closed）。
    """
    if not bool(getattr(handle, "metric_gate_passed", False)):
        return False
    state = getattr(handle, "_state", None)
    qt = getattr(handle, "question_type", "")
    return bool(state is not None and state.metric_task_authorized(qt))


def check_no_negative_distance(trace: ProgramExecutionTrace) -> bool:
    """距离 / 尺寸 / 面积类结果必须非负（确定性，与证据状态无关）。"""
    for name in ("euclidean_distance", "robust_distance", "camera_object_distance",
                 "surface_distance_between_objects", "object_3d_extent",
                 "plane_fit_room_size"):
        for v in _iter_tool_values(trace, name):
            if _finite_number(v) and float(v) < 0:
                return False
            if isinstance(v, dict):
                for key in ("distance_metric", "distance_normalized",
                            "surface_distance_metric", "room_area_m2",
                            "room_diagonal_normalized"):
                    val = v.get(key)
                    if _finite_number(val) and float(val) < 0:
                        return False
    return True


def check_unit_consistent(trace: ProgramExecutionTrace, handle: SceneHandle) -> bool:
    """米制证据一致性：无授权却产出绝对单位数值 → 失败（§13.2 fail-closed）。

    v6 判定只读两件事：本次调用是否真的产出米制量（Tool 输出里的米制键），以及
    `metric_evidence_authorized(handle)`。**不**再读任何"尺度已知/置信度"中间量
    （v5 `scale_known` 已随校准路线废止，§20）。
    """
    if metric_evidence_authorized(handle):
        return True
    for tool, key in _METRIC_VALUE_KEYS.items():
        for v in _iter_tool_values(trace, tool):
            if isinstance(v, dict) and v.get(key) is not None:
                return False   # 无米制授权却给出米制量 → 单位/证据不一致
    return True


def check_world_frame(trace: ProgramExecutionTrace, handle: SceneHandle) -> bool:
    """世界系依赖检查（§5.2/§9.8 fail-closed）。

    - 本次 episode 未调用任何世界系依赖 Tool → 无可判定项，通过；
    - 调用了 → 必须有合法 `world_up` **且**已知 `handedness`；缺失/非法 → 判失败
      （方向题在错误平面里算 left/right 的历史缺陷已验证过代价）。
    """
    used = [r.tool for r in trace.results if r.tool in WORLD_FRAME_TOOLS]
    if not used:
        return True
    return _world_frame_available(handle)


def check_inside_bbox(
    trace: ProgramExecutionTrace, handle: SceneHandle, margin: float = BBOX_MARGIN
) -> bool:
    """reproject 调用的 3D 输入点必须落在场景包围盒内（外扩 margin）。

    世界系依赖：`p3d` 与世界系包围盒必须处在同一（且已确认的）世界系里。故
    ① 存在 reproject 调用但世界系契约缺失 → fail-closed（判失败）；
    ② 无对象（`scene_bbox()` 抛 ValueError）→ 无从判定，不否决。
    """
    has_reproj = any(r.tool == "reproject" and "p3d" in r.args for r in trace.results)
    if has_reproj and not _world_frame_available(handle):
        return False
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


def check_reprojection(
    trace: ProgramExecutionTrace, handle: SceneHandle, threshold_px: float = TH_REPROJ_PX
) -> bool:
    """对 reproject 成功调用复核误差：重新计算并与记录值比较（确定性）。

    这是**自算值复核**（同 c2w/K 重算），不是 BA 重投影残差、不构成 G5（§10.4）。
    重算所需产物缺失（位姿/内参未注入）→ 无法证明 → fail-closed（判失败）。
    """
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
    """§4 M11 伪代码的确定性实现（不调 VLM / 离线治理模型；硬约束 13）。"""
    authorized = metric_evidence_authorized(handle)
    wf_ok = _world_frame_available(handle)
    checks = {
        "no_negative_distance": check_no_negative_distance(trace),
        "unit_consistent": check_unit_consistent(trace, handle),
        "world_frame": check_world_frame(trace, handle),
        "inside_bbox": check_inside_bbox(trace, handle),
        "reprojection": check_reprojection(trace, handle, threshold_px),
    }
    violations = [k for k, v in checks.items() if not v]
    notes = [
        f"metric_evidence_authorized={authorized}"
        f"（gate_passed={bool(getattr(handle, 'metric_gate_passed', False))} ∧ "
        f"metric_task_authorized({getattr(handle, 'question_type', '')!r})；"
        "v5 scale_known 已废止，§20）",
        f"world_frame_available={wf_ok}"
        f"（world_up={'set' if handle.world_up is not None else 'missing'}，"
        f"handedness={handle.handedness}）",
    ]
    del answer  # 语义项（T2）在 verifier/t1_t4_checks.py；本模块保持纯几何
    return GeometryVerifyResult(
        passed=not violations, checks=checks, violations=violations, notes=notes,
        metric_evidence_authorized=authorized, world_frame_available=wf_ok)


__all__ = [
    "BBOX_MARGIN",
    "GeometryVerifyResult",
    "KNOWN_HANDEDNESS",
    "TH_REPROJ_PX",
    "WORLD_FRAME_TOOLS",
    "check_inside_bbox",
    "check_no_negative_distance",
    "check_reprojection",
    "check_unit_consistent",
    "check_world_frame",
    "geometry_verify",
    "metric_evidence_authorized",
]
