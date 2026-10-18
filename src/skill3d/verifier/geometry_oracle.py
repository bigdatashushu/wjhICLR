"""Geometry Verifier（§4 M11）：确定性几何校验，纯 numpy，不调任何 VLM（硬约束 1/13）。

校验项：no_negative_distance / unit_consistent / world_frame / inside_bbox / reprojection。

v6 迁移（§5.2/§5.3/§13/§20）：

1. **`scale_known` 事实源已删除**（§20：多锚点 + conformal 校准池整体废止）。
   "能不能给绝对单位数值"改由 v6 证据事实判定，唯一事实源是
   `SceneHandle.metric_gate_passed` ∧ `SceneState.metric_task_authorized(question_type)`
   （= 题型是米制 ∧ `MetricEvidenceGateResult.gate_passed` ∧
   `question_tool_scope=metric_enabled`，§13.2）。任何"用别的代理量推断尺度已知"的写法
   都是实现错误（§10.4：G5 类代理值严禁冒充）。
2. **局部世界系检查**：方向 / 连通性 / 房间面积依赖 `world_up` + `handedness`。
   重投影只依赖位姿与内参，不依赖重力方向。
3. **G5 重投影残差永久 `not_available`**（§10.4/§20）：本模块的 `reprojection` 项是对
   **自算值**的确定性复核（同一 c2w/K 重算并比较），它**不是** BA 重投影残差，也**不得**
   被当作 G5 上报。M4 主门另有其人（warp 内点率 + 点云重叠率，D11）。

注：§5 schemas 未定义 GeometryVerifyResult，故本地定义（不改 schemas/）。
"""

from __future__ import annotations

import json
from typing import Any, Iterable, Literal, Optional

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from skill3d.schemas import ProgramExecutionTrace
from skill3d.schemas.tool import FailureCause
from skill3d.tools.ranking_contract import ranking_problems
from skill3d.tools.scene_handle import SceneHandle

TH_REPROJ_PX = 2.0  # TODO_CALIBRATE 重投影误差阈值（像素）
BBOX_MARGIN = 0.05  # TODO_CALIBRATE 包围盒外扩比例

# 依赖**世界系契约**（`world_up` + `handedness`）的 Tool（§9.8/§9.10）。
# 口径与 `ToolSpec.requires_local_facts` 含 `world_frame` 一致：契约缺失时这些调用
# 产出的方向/可达结论不可信 → 对应的几何检查必须 fail-closed。
WORLD_FRAME_TOOLS: tuple[str, ...] = (
    "relative_direction_of", "connectivity_graph", "plane_fit_room_size")

# 米制量键：Tool 输出里若出现这些**非空**键，说明该次调用产出了绝对单位数值
# （米）。它们只在米制证据被授权时才允许出现（§13.2 fail-closed）。
_METRIC_VALUE_KEYS: dict[str, str] = {
    "plane_fit_room_size": "room_area_m2",
    "robust_distance": "distance_metric",
    "camera_object_distance": "distance_metric",
    "surface_distance_between_objects": "surface_distance_metric",
    "object_3d_extent": "extent_metric",
    "object_centroid": "centroid_metric",
    "object_distance_m": "distance_m",
}

_OBJECT_GEOMETRY_ARGS: dict[str, tuple[str, ...]] = {
    "object_centroid": ("obj_id",),
    "object_3d_extent": ("obj_id",),
    "robust_distance": ("reference", "target"),
    "camera_object_distance": ("obj_id",),
    "surface_distance_between_objects": ("obj_a", "obj_b"),
    "relative_distance_rank": ("reference", "candidate_categories"),
    "object_distance_m": ("obj_a", "obj_b"),
    "relative_direction_of": ("observer_id", "facing_at_id", "target_id"),
    "count_objects": ("category_name",),
}

# 合法手性取值（与 `reconstruction_gate/world_frame.py` 同一词汇表）
KNOWN_HANDEDNESS: frozenset[str] = frozenset({"right", "left"})


class GeometryIssue(BaseModel):
    """可定位的失败事实；未确证共享前提失效时 premise 必须为空。"""

    model_config = ConfigDict(extra="forbid")
    check: str
    tool: str = ""
    result_id: str = ""
    reason: str
    # 计算声明错误不代表输入 ToolResult 本身错误；只有结果级几何失败才应撤销结果。
    invalidate_result: bool = True
    confirmed_shared_premise: Optional[Literal[
        "geometry_3d", "world_frame", "metric_scale", "object_detection"
    ]] = None
    failure_cause: Optional[FailureCause] = None


class GeometryVerifyResult(BaseModel):
    model_config = ConfigDict(extra="forbid", protected_namespaces=())

    run_id: str = ""
    qa_id: str = ""
    passed: bool
    checks: dict[str, bool]
    check_status: dict[str, Literal["pass", "fail", "not_applicable", "not_performed"]] = Field(
        default_factory=dict)
    check_details: dict[str, dict] = Field(default_factory=dict)
    violations: list[str]
    issues: list[GeometryIssue] = Field(default_factory=list)
    # 只读诊断（进 trace，便于"M11 为什么这么判"的事后归因；不参与判定）
    notes: list[str] = []
    metric_evidence_authorized: bool = False
    world_frame_available: bool = False
    derivation_replay: dict = Field(default_factory=dict)


def _iter_tool_values(trace: ProgramExecutionTrace, tool_name: str) -> Iterable[Any]:
    for r in trace.results:
        if r.tool == tool_name and r.error is None:
            try:
                yield json.loads(r.value)
            except (json.JSONDecodeError, TypeError):
                continue


def _finite_number(v: Any) -> bool:
    if not isinstance(v, (int, float)) or isinstance(v, bool):
        return False
    try:
        return bool(np.isfinite(float(v)))
    except OverflowError:
        return False


def _valid_json_result(value: str) -> bool:
    """Successful public ToolResults must carry finite, decodable JSON."""
    def finite(node: Any) -> bool:
        if isinstance(node, float):
            return bool(np.isfinite(node))
        if isinstance(node, list):
            return all(finite(item) for item in node)
        if isinstance(node, dict):
            return all(finite(item) for item in node.values())
        return True

    try:
        return finite(json.loads(value))
    except (ValueError, TypeError, RecursionError):
        return False


def _world_frame_available(handle: SceneHandle) -> bool:
    """世界系契约是否可用（`world_up` 合法非零单位向量 ∧ `handedness` 已知取值）。"""
    return handle.world_frame_usable


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
                 "plane_fit_room_size", "object_distance_m"):
        for v in _iter_tool_values(trace, name):
            if isinstance(v, dict):
                for key in ("distance_metric", "distance_normalized",
                            "surface_distance_metric", "surface_distance_normalized",
                            "distance_m", "room_area_m2", "room_diagonal_normalized",
                            "extent_metric", "extent_normalized", "extent_longest_metric"):
                    val = v.get(key)
                    values = val if isinstance(val, list) else [val]
                    if any(x is not None and (
                        not _finite_number(x) or float(x) < 0
                    ) for x in values):
                        return False
            elif not _finite_number(v) or float(v) < 0:
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


def _has_binding_conflict(obj) -> bool:
    return any(
        flag.get("code") == "semantic_label_mask_conflict"
        for flag in (getattr(obj, "binding_flags", None) or [])
    )


def _binding_unavailable(handle, obj) -> bool:
    return _has_binding_conflict(obj) or not handle.binding_usable(obj.obj_id)


def check_binding_integrity(trace: ProgramExecutionTrace, handle: SceneHandle) -> bool:
    """复核实际使用对象的技术有效性；不判断其是否为题目所指实例。"""
    if not hasattr(handle, "object_records"):
        return True
    conflict_ids = {
        obj.obj_id for obj in handle.object_records()
        if _binding_unavailable(handle, obj)
    }
    if not conflict_ids and not handle.require_question_binding:
        return True

    def argument_conflicts(value) -> bool:
        values = value if isinstance(value, (list, tuple)) else [value]
        for item in values:
            if not isinstance(item, str):
                continue
            if item in conflict_ids:
                return True
            if item.strip().lower() in (
                    "camera", "observer", "agent", "self", "camera_center", "camera0"):
                continue
            if any(oid in conflict_ids for oid in handle.list_objects_by_name(item)):
                return True
        return False

    def payload_conflicts(node) -> bool:
        if isinstance(node, dict):
            for key, value in node.items():
                if key in ("obj_id", "object_id") and value in conflict_ids:
                    return True
                if payload_conflicts(value):
                    return True
        elif isinstance(node, list):
            return any(payload_conflicts(value) for value in node)
        return False

    for result in trace.results:
        if (result.status != "ok" or result.error is not None
                or result.tool not in _OBJECT_GEOMETRY_ARGS):
            continue
        if (result.tool == "count_objects"
                and handle.require_question_binding
                and not handle.category_binding_complete(
                    str(result.args.get("category_name", "")))):
            return False
        if (result.tool == "relative_distance_rank" and handle.require_question_binding
                and any(not handle.category_binding_complete(str(category))
                        for category in result.args.get("candidate_categories", []))):
            return False
        if result.object_dependencies:
            if conflict_ids.intersection(result.object_dependencies):
                return False
            continue
        if any(argument_conflicts(result.args.get(name))
               for name in _OBJECT_GEOMETRY_ARGS[result.tool]
               if name in result.args):
            return False
        try:
            if payload_conflicts(json.loads(result.value)):
                return False
        except (json.JSONDecodeError, TypeError):
            continue
    return True


def check_inside_bbox(
    trace: ProgramExecutionTrace, handle: SceneHandle, margin: float = BBOX_MARGIN
) -> bool:
    """reproject 调用的 3D 输入点必须落在场景包围盒内（外扩 margin）。

    点和包围盒使用同一重建坐标系，无需知道重力方向。
    无对象（`scene_bbox()` 抛 ValueError）→ 无从判定，不否决。
    """
    try:
        bmin, bmax = handle.scene_bbox()
    except ValueError:
        return True  # 无对象场景无从判定，不否决
    extent = bmax - bmin
    lo, hi = bmin - margin * extent, bmax + margin * extent
    for r in trace.results:
        if r.tool == "reproject" and "p3d" in r.args:
            p = np.asarray(r.args["p3d"], dtype=np.float64)
            if p.shape != (3,) or not np.all(np.isfinite(p)) \
                    or np.any(p < lo) or np.any(p > hi):
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
    *,
    options: Optional[list[str]] = None,
) -> GeometryVerifyResult:
    """§4 M11 伪代码的确定性实现（不调 VLM / 离线治理模型；硬约束 13）。"""
    authorized = metric_evidence_authorized(handle)
    wf_ok = _world_frame_available(handle)
    checks = dict.fromkeys((
        "valid_json", "no_negative_distance", "unit_consistent", "world_frame",
        "binding_integrity", "inside_bbox", "reprojection", "complete_ranking"), True)
    details = {check: {"applicable_result_ids": [], "tools": [], "executed_result_ids": [],
                      "object_ids": [], "execution_count": 0, "not_performed_reasons": []}
               for check in checks}
    issues: list[GeometryIssue] = []
    reasons = {
        "valid_json": "A successful tool result is not valid finite JSON",
        "no_negative_distance": "Distance, size, or area results contain negative or invalid numbers",
        "unit_consistent": "This question has no metric authorization, but the results contain absolute-unit values",
        "world_frame": "The world up direction or handedness contract is missing",
        "binding_integrity": (
            "The result references an unavailable or inconsistent object binding"),
        "inside_bbox": "The reprojection input point is outside the scene bounding box",
        "reprojection": "The reprojection result cannot be reproduced using the same pose/intrinsics",
    }
    # 逐结果检查：既能精确撤销，也避免单个错误被多条正确重投影的均值掩盖。
    for result in trace.results:
        if result.status != "ok" or result.error is not None or result.invalidated_by:
            continue
        single = trace.model_copy(update={"results": [result]})
        is_reproject = result.tool == "reproject"
        applicable = {
            "valid_json": True,
            "no_negative_distance": result.tool in (
                "euclidean_distance", "robust_distance", "camera_object_distance",
                "surface_distance_between_objects", "object_3d_extent",
                "plane_fit_room_size", "object_distance_m"),
            "unit_consistent": result.tool in _METRIC_VALUE_KEYS,
            "world_frame": result.tool in WORLD_FRAME_TOOLS,
            "binding_integrity": result.tool in _OBJECT_GEOMETRY_ARGS,
            "inside_bbox": is_reproject,
            "reprojection": is_reproject,
            "complete_ranking": result.tool == "relative_distance_rank",
        }
        functions = {
            "valid_json": lambda: _valid_json_result(result.value),
            "no_negative_distance": lambda: check_no_negative_distance(single),
            "unit_consistent": lambda: check_unit_consistent(single, handle),
            "world_frame": lambda: check_world_frame(single, handle),
            "binding_integrity": lambda: check_binding_integrity(single, handle),
            "inside_bbox": lambda: check_inside_bbox(single, handle),
            "reprojection": lambda: check_reprojection(single, handle, threshold_px),
        }
        row = {}
        for check, applies in applicable.items():
            if not applies:
                continue
            detail = details[check]
            detail["applicable_result_ids"].append(result.result_id)
            detail["tools"].append(result.tool)
            detail["object_ids"].extend(result.object_dependencies)
            reason = ""
            if check == "binding_integrity" and not hasattr(handle, "object_records"):
                reason = "object_records_unavailable"
            if check == "inside_bbox":
                try:
                    handle.scene_bbox()
                except ValueError:
                    reason = "scene_bbox_unavailable"
            if check == "complete_ranking" and not row.get("valid_json"):
                reason = "rank_payload_not_decodable"
            if reason:
                detail["not_performed_reasons"].append({
                    "result_id": result.result_id, "reason": reason})
                continue
            if check != "complete_ranking":
                row[check] = functions[check]()
            detail["executed_result_ids"].append(result.result_id)
            detail["execution_count"] += 1
        if applicable["complete_ranking"] and row.get("valid_json"):
            problems = ranking_problems(json.loads(result.value), options)
            row["complete_ranking"] = not problems
            reasons["complete_ranking"] = "; ".join(problems)
        for check, passed in row.items():
            if passed:
                continue
            checks[check] = False
            # 未获本题米制授权不等于全局尺度失效；域值/复算错误也不猜根因。
            premise = "world_frame" if not wf_ok and check == "world_frame" else None
            conflicted = [
                obj for obj in handle.object_records()
                if _binding_unavailable(handle, obj)
                and obj.obj_id in result.object_dependencies
            ] if check == "binding_integrity" else []
            cause = FailureCause(
                code=check, scope="shared_evidence" if premise else
                "local_object" if conflicted else "local_result",
                shared_premises=[premise] if premise else [],
                object_ids=[obj.obj_id for obj in conflicted],
                result_ids=[result.result_id],
                audit_ids=sorted({
                    flag["audit_sha256"] for obj in conflicted
                    for flag in obj.binding_flags
                    if flag.get("code") == "semantic_label_mask_conflict"
                    and flag.get("audit_sha256")
                }), message=reasons[check])
            issues.append(GeometryIssue(
                check=check, tool=result.tool, result_id=result.result_id,
                reason=reasons[check], confirmed_shared_premise=premise,
                failure_cause=cause))
    violations = [k for k, v in checks.items() if not v]
    status = {}
    for check, detail in details.items():
        for key in ("tools", "object_ids"):
            detail[key] = sorted(set(detail[key]))
        if not detail["applicable_result_ids"]:
            status[check] = "not_applicable"
            detail["reason"] = "no_applicable_results"
        elif not detail["execution_count"]:
            status[check] = "not_performed"
        elif not checks[check]:
            status[check] = "fail"
        elif detail["not_performed_reasons"]:
            status[check] = "not_performed"
        else:
            status[check] = "pass"
    notes = [
        f"metric_evidence_authorized={authorized}"
        f" (gate_passed={bool(getattr(handle, 'metric_gate_passed', False))} and "
        f"metric_task_authorized({getattr(handle, 'question_type', '')!r}); "
        "v5 scale_known has been retired, §20)",
        f"world_frame_available={wf_ok}"
        f"（world_up={'set' if handle.world_up is not None else 'missing'}，"
        f"handedness={handle.handedness}）",
    ]
    del answer  # 答案语义由在线提交验收处理；本模块保持纯几何
    return GeometryVerifyResult(
        passed=not violations, checks=checks, check_status=status, check_details=details,
        violations=violations, issues=issues, notes=notes,
        metric_evidence_authorized=authorized, world_frame_available=wf_ok)


__all__ = [
    "BBOX_MARGIN",
    "GeometryVerifyResult",
    "GeometryIssue",
    "KNOWN_HANDEDNESS",
    "TH_REPROJ_PX",
    "WORLD_FRAME_TOOLS",
    "check_inside_bbox",
    "check_binding_integrity",
    "check_no_negative_distance",
    "check_reprojection",
    "check_unit_consistent",
    "check_world_frame",
    "geometry_verify",
    "metric_evidence_authorized",
]
