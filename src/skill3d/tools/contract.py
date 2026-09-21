"""M6 Tool 契约：scene_route → 产物映射 + 证据驱动的 Tool 收窄 + fail-closed 异常族。

v6（D4/D6/D7）三条纪律同时落地：

1. **scene_route 只挂产物**（§5.3）：`ROUTE_ARTIFACTS[scene_route]` 是纯函数，
   只由 M4 质量决定，**不含 `scale`** —— `scale` 是 `question_tool_scope` 派生的
   条件集（米制题 ∧ gate_passed）；
2. **逐 Tool 证据收窄**（§7.2）：`docs()` 按每个 Tool 的
   `requires_evidence` / `tolerates_degraded` 与当前 `EvidenceProfile` 逐项匹配，
   **单项失败只收回依赖该项的 Tool**，不做全局降级；
3. **执行期 fail-closed**（硬约束 23）：`call_tool` 在 `fn` 之前校验产物与证据，
   缺失抛 `ArtifactUnavailableError` / `ConfidenceGateError`，
   **绝不静默返回 False/0**。

`ROUTE_ARTIFACTS` 是纯函数映射，`tools.registry`、`routing.skill_retriever`、
`reconstruction_gate.scene_state` 共用同一份判定（单一事实源）。
"""

from __future__ import annotations

from typing import Iterable, Mapping, Optional, Sequence

from skill3d.schemas.evidence import (
    CAPABILITIES,
    CapabilityState,
    EvidenceProfile,
    capability_at_least,
)

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

# scene_route → 可用重建产物集合（v6 §5.3：纯函数，**只挂 scene_route**）
#
# 与 v5 的关键差异：`scale` **不在** `full_3d` 里。v6 §5.3 定义
#   available_artifacts = ROUTE_ARTIFACTS[scene_route] ∪ ({scale} if 米制题 ∧ gate_passed)
# 把 `scale` 收进条件集，使"米制尺度不可用"不再与"几何不可用"混为一谈（D3/D4）。
ROUTE_ARTIFACTS: dict[str, frozenset[str]] = {
    "full_3d": frozenset({
        ARTIFACT_FRAMES, ARTIFACT_INTRINSICS, ARTIFACT_DEPTH, ARTIFACT_POSES,
        ARTIFACT_POINT_CLOUD, ARTIFACT_OBJECTS,
    }),
    "fallback_2d_only": frozenset({ARTIFACT_FRAMES, ARTIFACT_INTRINSICS}),
    "unanswerable": frozenset(),
}

# ---- question_tool_scope（§5.3 D4：与 scene_route 正交的第二字段）----
SCOPE_FULL_3D = "full_3d"
SCOPE_METRIC_ENABLED = "metric_enabled"
SCOPE_FALLBACK_2D_ONLY = "fallback_2d_only"

QUESTION_TOOL_SCOPES: tuple[str, ...] = (
    SCOPE_FULL_3D, SCOPE_METRIC_ENABLED, SCOPE_FALLBACK_2D_ONLY,
)

# 米制题型的**证据依赖名**：任何 requires_evidence 含此项的 Tool 都是米制 Tool。
# 用它替代 v5 的 `supported_metric_tasks` 作为"是否米制工具"的判据，
# 使 Tool 暴露规则只有一条（证据），不必再维护第二份题型白名单。
EVIDENCE_METRIC_SCALE = "metric_scale"


def available_artifacts_for(route: str, *,
                            question_type: str = "",
                            gate_passed: bool = False,
                            metric_question: bool = False) -> frozenset[str]:
    """scene_route → 可用产物集合（§5.3；未知 route 保守取空集：fail-closed）。

    `metric_question` 为真且 `gate_passed` 为真时才追加 `scale`（D3 条件集）。
    注意 `metric_question` 由调用方按**题型**判定；`gate_passed` 由
    `MetricEvidenceGate` 判定 —— 两者都不看 VLM 裁决。
    """
    base = ROUTE_ARTIFACTS.get(str(route), frozenset())
    if not base:
        return base
    if ARTIFACT_SCALE in ROUTE_ARTIFACTS.get(str(route), frozenset()):
        # 未来若把 scale 放回基础集，这里保持兼容（当前不会发生）
        return base
    if metric_question and gate_passed and route == "full_3d":
        return base | {ARTIFACT_SCALE}
    return base


def question_tool_scope_of(scene_route: str, *,
                           metric_question: bool = False,
                           gate_passed: bool = False) -> str:
    """scene_route × 题级条件 → `question_tool_scope`（§5.3 D4）。

    不变量（§5.3）：`docs(question_tool_scope) ⊆ docs(scene_route)` —— 逐题只收窄。
    故 `scene_route != full_3d` 时 scope 只能是 2D-only（收窄），
    绝不允许出现"scene_route=fallback_2d_only 但 scope=full_3d"的越权。
    """
    if str(scene_route) != "full_3d":
        return SCOPE_FALLBACK_2D_ONLY
    if metric_question and gate_passed:
        return SCOPE_METRIC_ENABLED
    return SCOPE_FULL_3D


def route_artifacts_for_question(
    route: str,
    allowed_metric_tasks: Optional[Iterable[str]] = None,
    question_type: str = "",
    *,
    gate_passed: bool = False,
) -> set[str]:
    """逐题可用产物集（v6 §5.3；保留 v5 签名以便调用点平滑迁移）。

    `allowed_metric_tasks` 是 v5 的逐题型授权集合；v6 用 `gate_passed` 表达同一件事
    （gate 通过 ⟺ `metric_scale=available`）。为兼容既有调用点，两者取**逻辑与**：
    只有同时满足"题型在授权集合内"与"gate 通过"才追加 `scale`。
    传空授权集合（v5 默认）而 `gate_passed=True` 时按 v6 口径判（gate 说了算）。
    """
    base = set(available_artifacts_for(route))
    if route != "full_3d":
        return base
    allowed = {str(t) for t in (allowed_metric_tasks or set())}
    metric_q = bool(question_type) and (str(question_type) in allowed if allowed else False)
    if gate_passed and metric_q:
        return base | {ARTIFACT_SCALE}
    return base


def tool_allowed(requires_artifacts: Iterable[str], route: str) -> bool:
    """该 Tool 在给定 route 下是否允许暴露/执行（两件判定的产物维度）。"""
    return set(requires_artifacts).issubset(
        available_artifacts_for(route, metric_question=True, gate_passed=True))


# ------------------------------------------------- 证据驱动的 Tool 可见性 ----

def evidence_visible(spec, profile: Optional[EvidenceProfile]) -> bool:
    """§7.2 逐 Tool 判定：`requires_evidence` × `tolerates_degraded`。

    - 任一 required = `unavailable` → 不可见；
    - 任一 required = `degraded` 且不在 `tolerates_degraded` → 不可见；
    - 其余情况可见（`degraded` 且在容忍列表 → 可见但答案带 `evidence_degraded`）。

    `profile=None`（无证据画像，例如 mock_light 早期路径）→ 按"全部 available"
    处理**不成立**：此时只允许**无证据依赖**的 Tool 通过（fail-closed）。
    """
    required = list(getattr(spec, "requires_evidence", None) or [])
    if not required:
        return True
    if profile is None:
        return False
    tolerates = set(getattr(spec, "tolerates_degraded", None) or [])
    for cap in required:
        if cap not in CAPABILITIES:
            return False   # 未知能力名 = 声明错误 → 不暴露（fail-closed）
        state = profile.state(cap)
        if state == "unavailable":
            return False
        if state == "degraded" and cap not in tolerates:
            return False
    return True


def degraded_evidence_flags(spec, profile: Optional[EvidenceProfile]) -> list[str]:
    """该 Tool 在当前画像下会带上的降级标记（空 = 全 available）。"""
    if profile is None:
        return ["evidence_unavailable"]
    required = list(getattr(spec, "requires_evidence", None) or [])
    tolerates = set(getattr(spec, "tolerates_degraded", None) or [])
    out = []
    for cap in required:
        if cap not in CAPABILITIES:
            continue
        if profile.state(cap) == "degraded" and cap in tolerates:
            out.append(f"evidence_degraded:{cap}")
    return out


def scope_allows(spec, scope: str) -> bool:
    """`question_tool_scope` 对某 Tool 的收窄判定（§5.3）。

    - `metric_enabled`：允许全部（`docs(metric_enabled) ⊇ docs(full_3d)` 仅限
      米制 Tool 的追加，不含"新增工具"）；
    - `full_3d`：米制 Tool 一律不可见（gate 未过）；
    - `fallback_2d_only`：只允许不依赖 `depth/poses/point_cloud/objects/scale` 的 Tool。
    """
    required = set(getattr(spec, "requires_evidence", None) or [])
    if scope == SCOPE_METRIC_ENABLED:
        return True
    if scope == SCOPE_FULL_3D:
        return EVIDENCE_METRIC_SCALE not in required
    if scope == SCOPE_FALLBACK_2D_ONLY:
        return EVIDENCE_METRIC_SCALE not in required and \
            set(spec.requires_artifacts).issubset(ROUTE_ARTIFACTS["fallback_2d_only"])
    return False


def route_aware_available(saved: Optional[Iterable[str]], route: str) -> set[str]:
    """可用产物集合：以 route 映射为准，SceneState 显式声明仅作上界补充。

    取交集语义：`SceneState.available_artifacts` 若已按 route 填好则与映射一致；
    若调用方手工缩小（例如对象绑定失败后 objects 不可用），以更小者为准（fail-closed）。
    """
    base = set(available_artifacts_for(route, metric_question=True, gate_passed=True))
    if saved is None:
        return base
    declared = set(saved)
    if not declared:
        return base
    return declared & base


def check_metric_task_contract(
    tool: str,
    supported_metric_tasks: Sequence[str],
    allowed_metric_tasks: Iterable[str],
    question_type: str,
    *,
    args: Optional[dict] = None,
) -> None:
    """米制 Tool 的逐题授权校验（v5 HC33 语义，v6 由证据门主导）。

    v6 的主判据是 `EvidenceProfile.metric_scale == available`（= gate 通过），
    由 `docs()` 裁剪与 `check_evidence_contract` 的执行期二次校验承担。
    本函数保留为**题型维度**的补充校验：声明了 `supported_metric_tasks` 的 Tool
    仍要求 `question_type` 在 `allowed_metric_tasks ∩ supported_metric_tasks` 内。

    仍然 fail-closed：不满足抛 `ConfidenceGateError`，**不**静默返回相对单位数值。
    """
    if not supported_metric_tasks:
        return
    qt = str(question_type or "")
    allowed = {str(t) for t in (allowed_metric_tasks or set())}
    if not allowed:
        # v6：逐题型授权集合为空时，授权判定完全交给证据门（gate）——
        # 若证据门已通过（metric_scale=available），此处不再二次否决。
        return
    if not qt:
        raise ConfidenceGateError(
            tool,
            f"Tool {tool} 依赖米制尺度，但当前 episode 尚未分类出题型"
            "（未知题型不授权米制 Tool）",
            args=args)
    if qt not in allowed:
        raise ConfidenceGateError(
            tool,
            f"Tool {tool} 依赖米制尺度，但题型 {qt} 不在本次尺度评估授权的集合内"
            f"（allowed_metric_tasks={sorted(allowed)}）",
            args=args)
    if qt not in set(supported_metric_tasks):
        raise ConfidenceGateError(
            tool,
            f"Tool {tool} 不支持题型 {qt}（supported_metric_tasks="
            f"{sorted(supported_metric_tasks)}）",
            args=args)


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


def check_evidence_contract(
    tool: str,
    spec,
    profile: Optional[EvidenceProfile],
    *,
    gate_passed: Optional[bool] = None,
    route: str = "",
    args: Optional[dict] = None,
) -> None:
    """执行期**二次校验** `requires_evidence`（§13.2 双重 fail-closed）。

    与 `docs()` 的静态裁剪同源：静态层藏起来，执行层也要拦住 ——
    否则模型凭记忆写出被隐藏的 Tool 名就能绕过证据门。
    """
    if not evidence_visible(spec, profile):
        missing = [c for c in (getattr(spec, "requires_evidence", None) or [])
                   if profile is None or not capability_at_least(
                       profile.state(c) if c in CAPABILITIES else "unavailable",
                       "degraded")]
        raise ArtifactUnavailableError(
            tool, missing or ["(证据不足)"], route=route,
            available=[], args=args)
    # 米制 Tool 的专项二次校验（§13.2：执行层也校验 gate_passed）
    required = list(getattr(spec, "requires_evidence", None) or [])
    if EVIDENCE_METRIC_SCALE in required and gate_passed is not None and not gate_passed:
        raise ConfidenceGateError(
            tool,
            f"Tool {tool} 依赖米制尺度，但 MetricEvidenceGate 未通过"
            "（§13.2 执行期二次校验）",
            args=args)


# --------------------------------------------------------------- 异常族 ----

class ToolContractError(Exception):
    """Tool 契约违规基类（§9.12）：确定性异常，进 FailureTaxonomy。"""

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
    """所需重建产物/证据不在可用集（硬约束 23、§13.2）。"""

    error_code = "tool_contract"

    def __init__(self, tool: str, missing: Sequence[str], route: str = "",
                 available: Optional[Iterable[str]] = None,
                 args: Optional[dict] = None) -> None:
        msg = (f"Tool {tool} 所需产物/证据缺失: {sorted(missing)}；"
               f"route={route or 'unknown'}，可用产物={sorted(available or [])}"
               "（硬约束 23：fail-closed，不静默返回 False/0）")
        super().__init__(tool, msg, args=args, route=route, missing=sorted(missing),
                         available=available)


class ConfidenceGateError(ToolContractError):
    """局部质量门未过（MetricEvidenceGate 未过 / 米制题型未授权）。"""

    error_code = "confidence_gate"


class DomainValueError(ToolContractError):
    """域值错误（负距离 / 点在相机后方 / 超包围盒 / 单位错）。"""

    error_code = "domain_value"


class AnswerAlreadyGiven(ToolContractError):
    """`ReturnAnswer` 之后再调 Tool（§15.1 运行层保护）。

    v6 D7：`ReturnAnswer` 保留"记录/反作弊"语义（不中止执行），但一旦它被调用，
    后续任何 Tool 调用都必须抛本异常 —— 而不是让程序继续跑到 `IndexError`
    崩成"服务故障"。

    背景 [已实测]：v5 模型自然写出 `if not ids: ReturnAnswer("abstain")` 然后继续
    `object_centroid(ids[0])` → IndexError → `violation_runtime`，内测方向题全栽在这里。
    """

    error_code = "answer_already_given"


_ERROR_CODE_BY_CLASS: Mapping[str, str] = {
    "tool_contract": "tool_contract",
    "confidence_gate": "confidence_gate",
    "domain_value": "domain_value",
    "answer_already_given": "answer_already_given",
}
