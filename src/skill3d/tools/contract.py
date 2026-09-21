"""M6 Tool 契约：route → 可用产物映射 + 执行期 fail-closed 异常族（§4 M6 / 硬约束 23）。

两条纪律（D-3 决策 (a)+(b) 同时落地）：

1. **静态裁剪**：`REGISTRY.docs(route=scene.route)` 只暴露当前 route 下可用的 Tool
   文档，prompt 头部显式写"当前 route=…，可用产物=…"，使模型只能编排可用的 Tool；
2. **执行期 fail-closed**：`call_tool` 在 `validate(args)` 之后、`fn` 调用之前检查
   `requires_artifacts ⊆ available_artifacts`；缺失则抛 `ArtifactUnavailableError`，
   **绝不静默返回 False/0**。`exists_in_scene(name) == False` 当且仅当 objects 产物
   可用且场景中真无此实例。

`ROUTE_ARTIFACTS` 是纯函数映射，`tools.registry`、`routing.skill_retriever`、
`reconstruction_gate.scene_state` 共用同一份判定（单一事实源）。
"""

from __future__ import annotations

from typing import Iterable, Optional, Sequence

# 产物词汇表（与 SceneHandle / ReconstructionArtifact 的可用产物口径一致）
ARTIFACT_FRAMES = "frames"
ARTIFACT_INTRINSICS = "intrinsics"
ARTIFACT_DEPTH = "depth"
ARTIFACT_POSES = "poses"
ARTIFACT_POINT_CLOUD = "point_cloud"
ARTIFACT_OBJECTS = "objects"
ARTIFACT_SCALE = "scale"

KNOWN_ARTIFACTS: frozenset[str] = frozenset({
    ARTIFACT_FRAMES, ARTIFACT_INTRINSICS, ARTIFACT_DEPTH, ARTIFACT_POSES,
    ARTIFACT_POINT_CLOUD, ARTIFACT_OBJECTS, ARTIFACT_SCALE,
})

# route → 可用重建产物集合（纯函数；route 单调：full_3d ⊇ fallback_2d_only）
#
# 注意（v4 HC33 / §3 M6）：这里是 route 给出的**上界**，`scale` 只是"尺度产物可能
# 可用"。逐题授权由 `route_artifacts_for_question()` 在这之上收窄：仅当
# `question_type ∈ allowed_metric_tasks` 时才把 `scale` 计入可用集。
ROUTE_ARTIFACTS: dict[str, frozenset[str]] = {
    "full_3d": frozenset({
        ARTIFACT_FRAMES, ARTIFACT_INTRINSICS, ARTIFACT_DEPTH, ARTIFACT_POSES,
        ARTIFACT_POINT_CLOUD, ARTIFACT_OBJECTS, ARTIFACT_SCALE,
    }),
    "fallback_2d_only": frozenset({ARTIFACT_FRAMES, ARTIFACT_INTRINSICS}),
    "unanswerable": frozenset(),
}


def available_artifacts_for(route: str) -> frozenset[str]:
    """route → 可用产物集合（未知 route 保守取空集：fail-closed）。"""
    return ROUTE_ARTIFACTS.get(str(route), frozenset())


def route_artifacts_for_question(
    route: str,
    allowed_metric_tasks: Optional[Iterable[str]] = None,
    question_type: str = "",
) -> set[str]:
    """逐题可用产物集（v4 HC33，§3 M6 明文）。

    `full_3d` 的基础产物是 `{frames,intrinsics,depth,poses,point_cloud,objects}`；
    **仅当 `ScaleAssessment` 对当前 `question_type` 授权时再加入 `scale`**。
    `low` 档 → `scale` 不加入 → 依赖 `scale` 的米制 Tool 在 `docs()` 里被裁掉、
    执行期抛 `ArtifactUnavailableError`；**非米制 3D 产物不受影响**（HC33：不得因
    尺度为 low 一并降级 `depth/poses/point_cloud/objects`）。
    """
    base = set(available_artifacts_for(route))
    if ARTIFACT_SCALE not in base:
        return base
    allowed = {str(t) for t in (allowed_metric_tasks or set())}
    if question_type and str(question_type) in allowed:
        return base
    return base - {ARTIFACT_SCALE}


def check_metric_task_contract(
    tool: str,
    supported_metric_tasks: Sequence[str],
    allowed_metric_tasks: Iterable[str],
    question_type: str,
    *,
    args: Optional[dict] = None,
) -> None:
    """米制 Tool 的逐题授权校验（v4 HC33 / §3 M6）。

    检查 `question_type ∈ allowed_metric_tasks ∩ supported_metric_tasks`，
    否则抛 `ConfidenceGateError`（**不是**静默返回相对单位数值）。
    非米制 Tool（`supported_metric_tasks` 为空）**不读取** `scale_confidence`，
    直接放行——这是 HC33"只收回米制工具"的实现点。
    """
    if not supported_metric_tasks:
        return
    qt = str(question_type or "")
    allowed = {str(t) for t in (allowed_metric_tasks or set())}
    if not qt:
        raise ConfidenceGateError(
            tool,
            f"Tool {tool} 依赖米制尺度，但当前 episode 尚未分类出题型"
            "（HC33：未知题型不授权米制 Tool）",
            args=args)
    if qt not in allowed:
        raise ConfidenceGateError(
            tool,
            f"Tool {tool} 依赖米制尺度，但题型 {qt} 不在本次尺度评估授权的集合内"
            f"（allowed_metric_tasks={sorted(allowed)}；HC33 逐题型授权）",
            args=args)
    if qt not in set(supported_metric_tasks):
        raise ConfidenceGateError(
            tool,
            f"Tool {tool} 不支持题型 {qt}（supported_metric_tasks="
            f"{sorted(supported_metric_tasks)}）",
            args=args)


def tool_allowed(requires_artifacts: Iterable[str], route: str) -> bool:
    """该 Tool 在给定 route 下是否允许暴露/执行（两处共用同一判定）。"""
    return set(requires_artifacts).issubset(available_artifacts_for(route))


def route_aware_available(saved: Optional[Iterable[str]], route: str) -> set[str]:
    """可用产物集合：以 route 映射为准，SceneState 显式声明仅作上界补充。

    取交集语义：`SceneState.available_artifacts` 若已按 route 填好则与映射一致；
    若调用方手工缩小（例如对象绑定失败后 objects 不可用），以更小者为准（fail-closed）。
    """
    base = set(available_artifacts_for(route))
    if saved is None:
        return base
    declared = set(saved)
    if not declared:
        return base
    return base & declared if declared <= base else declared & base


# --------------------------------------------------------------- 异常族 ----

class ToolContractError(Exception):
    """Tool 契约违规基类（§4 M6 字段 9 / D-3）：确定性异常，进 FailureTaxonomy。"""

    error_code = "tool_contract"

    def __init__(self, tool: str, message: str, *, args: Optional[dict] = None,
                 route: str = "", missing: Optional[Sequence[str]] = None,
                 available: Optional[Iterable[str]] = None) -> None:
        super().__init__(message)
        self.tool = tool
        # 注意：不要赋值 self.args —— 那会覆盖 BaseException.args（异常消息会丢失）
        self.call_args = dict(args or {})
        self.route = route
        self.missing = sorted(missing or [])
        self.available = sorted(available or [])

    def detail(self) -> dict:
        return {
            "error_code": self.error_code,
            "tool": self.tool,
            "route": self.route,
            "missing": list(self.missing),
            "available": list(self.available),
            "message": str(self),
        }



class ArtifactUnavailableError(ToolContractError):
    """所需重建产物不在 `SceneState.available_artifacts`（硬约束 23）。"""

    error_code = "tool_contract"

    def __init__(self, tool: str, missing: Sequence[str], route: str = "",
                 available: Optional[Iterable[str]] = None,
                 args: Optional[dict] = None) -> None:
        msg = (f"Tool {tool} 所需产物缺失: {sorted(missing)}；"
               f"route={route or 'unknown'}，可用产物={sorted(available or [])}"
               "（硬约束 23：fail-closed，不静默返回 False/0）")
        super().__init__(tool, msg, args=args, route=route, missing=sorted(missing),
                         available=available)


class ConfidenceGateError(ToolContractError):
    """局部质量门未过（如 G9 跟踪一致性、米制题型未授权等）。"""

    error_code = "confidence_gate"


class DomainValueError(ToolContractError):
    """域值错误（负距离 / 点在相机后方 / 超包围盒 / 单位错）。"""

    error_code = "domain_value"


def check_artifact_contract(
    tool: str,
    requires_artifacts: Sequence[str],
    available_artifacts: Iterable[str],
    route: str = "",
    args: Optional[dict] = None,
) -> None:
    """执行期 fail-closed 校验：缺产物直接抛 `ArtifactUnavailableError`。"""
    available = set(available_artifacts)
    missing = sorted(set(requires_artifacts) - available)
    if missing:
        raise ArtifactUnavailableError(tool, missing, route=route,
                                       available=available, args=args)
