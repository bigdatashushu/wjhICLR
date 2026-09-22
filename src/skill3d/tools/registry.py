"""Tool Registry（§4 M6 / v6 §7.2、§17.4）：预封装稳定函数注册表（硬约束 14）。

Tool 是预先封装好的稳定 Python 函数，不是每题生成；
Coding Agent（Qwen3-VL-8B）只生成编排这些 Tool 的 episode-specific program。

v6 三条纪律：

- **注册准入**（§17.4 硬约束 7）：`ToolSpec` 的 `requires_evidence` **无默认值**，
  不声明就构造不出来；`tolerates_degraded ⊆ requires_evidence`，声明的能力名必须在
  `schemas.evidence.CAPABILITIES` 词汇表内；
- **静态裁剪** `docs(question_tool_scope, evidence_profile)`（§7.2/§5.3）：
  逐 Tool 按证据匹配 + 按 scope 收窄，只暴露当前真正可执行的 Tool 文档；
- **执行期 fail-closed**（硬约束 23 / §13.2）：`call_tool` 在调用实现前校验
  产物与证据，缺失抛 `ArtifactUnavailableError`、米制门未过抛 `ConfidenceGateError`，
  绝不静默返回假值。
"""

from __future__ import annotations

import hashlib
import inspect
import json
import time
from typing import Any, Callable, Iterable, Optional

from skill3d.schemas import ToolResult, ToolSpec
from skill3d.schemas.evidence import CAPABILITIES, EvidenceProfile
from skill3d.schemas.reconstruction import METRIC_TASK_TYPES
from skill3d.schemas.tool import ToolSource
from skill3d.tools.contract import (
    EVIDENCE_METRIC_SCALE,
    KNOWN_ARTIFACTS,
    SCOPE_FALLBACK_2D_ONLY,
    SCOPE_FULL_3D,
    SCOPE_METRIC_ENABLED,
    ToolContractError,
    check_artifact_contract,
    check_evidence_contract,
    check_metric_task_contract,
    degraded_evidence_flags,
    evidence_visible,
    scope_allows,
    tool_allowed,
)
from skill3d.tools.mock_switch import MockSwitch, request_key
from skill3d.tools.scene_handle import SceneHandle

TOOL_FACE_VERSION: str = "tool-face-v6"


class ToolNotFoundError(KeyError):
    pass


class ToolArgValidationError(ValueError):
    pass


def _error_code_of(exc: BaseException) -> Optional[str]:
    """异常 → ProgramExecutionTrace.error_code 归因（非契约异常返回 None）。"""
    if isinstance(exc, ToolContractError):
        return exc.error_code
    return None


class _RegisteredTool:
    def __init__(self, spec: ToolSpec, fn: Callable[..., Any]) -> None:
        self.spec = spec
        self.fn = fn


class ToolRegistry:
    """ToolSpec 注册表：统一签名 (handle: SceneHandle, **args) -> Any。"""

    def __init__(self) -> None:
        self._tools: dict[str, _RegisteredTool] = {}

    # ---- 注册 ----
    def register(self, spec: ToolSpec) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
        def deco(fn: Callable[..., Any]) -> Callable[..., Any]:
            if spec.name in self._tools:
                raise ValueError(f"Tool 重复注册: {spec.name}")
            sig = inspect.signature(fn)
            params = list(sig.parameters.values())
            if not params or params[0].name not in ("handle", "scene"):
                raise ValueError(f"Tool {spec.name} 首参必须为 handle（SceneHandle）")
            unknown = set(spec.requires_artifacts) - set(KNOWN_ARTIFACTS)
            if unknown:
                raise ValueError(
                    f"Tool {spec.name} 声明了未知产物 {sorted(unknown)}；"
                    f"词汇表见 tools.contract.KNOWN_ARTIFACTS")
            # §17.4 硬约束 7：证据声明必须显式（空列表合法 = 无证据依赖）
            if spec.requires_evidence is None:
                raise ValueError(
                    f"Tool {spec.name} 未声明 requires_evidence"
                    "（v6 §17.4：无证据声明的 Tool 不得进入注册表）")
            unknown_ev = set(spec.requires_evidence) - set(CAPABILITIES)
            if unknown_ev:
                raise ValueError(
                    f"Tool {spec.name} 声明了未知证据能力 {sorted(unknown_ev)}；"
                    f"词汇表见 schemas.evidence.CAPABILITIES")
            bad_tol = set(spec.tolerates_degraded or []) - set(spec.requires_evidence)
            if bad_tol:
                raise ValueError(
                    f"Tool {spec.name} 的 tolerates_degraded 含未在 requires_evidence "
                    f"中声明的能力 {sorted(bad_tol)}（§7.2：容忍列表必须是依赖的子集）")
            unknown_tasks = set(spec.supported_metric_tasks) - set(METRIC_TASK_TYPES)
            if unknown_tasks:
                raise ValueError(
                    f"Tool {spec.name} 声明了未知米制题型 {sorted(unknown_tasks)}；"
                    f"词汇表见 schemas.reconstruction.METRIC_TASK_TYPES")
            if spec.supported_metric_tasks and "scale" not in spec.requires_artifacts:
                raise ValueError(
                    f"Tool {spec.name} 声明了 supported_metric_tasks "
                    f"{spec.supported_metric_tasks}，但 requires_artifacts 未包含 'scale'"
                    "（米制 Tool 必须 fail-closed 依赖尺度产物）")
            if (EVIDENCE_METRIC_SCALE in spec.requires_evidence
                    and "scale" not in spec.requires_artifacts):
                raise ValueError(
                    f"Tool {spec.name} 的 requires_evidence 含 {EVIDENCE_METRIC_SCALE}，"
                    "但 requires_artifacts 未包含 'scale'"
                    "（证据依赖与产物依赖必须一致，否则会出现"
                    "'证据说可用、产物读不到'的静默错答）")
            self._tools[spec.name] = _RegisteredTool(spec, fn)
            return fn

        return deco

    # ---- 查询 ----
    def names(self) -> list[str]:
        return sorted(self._tools.keys())

    def get(self, name: str) -> _RegisteredTool:
        if name not in self._tools:
            raise ToolNotFoundError(f"未注册 Tool: {name}")
        return self._tools[name]

    def __contains__(self, name: str) -> bool:
        return name in self._tools

    def spec(self, name: str) -> ToolSpec:
        return self.get(name).spec

    def requires_artifacts(self, name: str) -> list[str]:
        return list(self.get(name).spec.requires_artifacts)

    def requires_evidence(self, name: str) -> list[str]:
        return list(self.get(name).spec.requires_evidence or [])

    def supported_metric_tasks(self, name: str) -> list[str]:
        return list(self.get(name).spec.supported_metric_tasks)

    def is_metric_tool(self, name: str) -> bool:
        return EVIDENCE_METRIC_SCALE in set(self.get(name).spec.requires_evidence or [])

    def names_for_scope(self, scope: Optional[str] = None,
                        available: Optional[Iterable[str]] = None,
                        *,
                        evidence_profile: Optional[EvidenceProfile] = None,
                        gate_passed: Optional[bool] = None,
                        allowed_metric_tasks: Optional[Iterable[str]] = None,
                        question_type: str = "") -> list[str]:
        """当前 scope/证据下**可暴露**的 Tool 名（`docs()` 的过滤依据）。

        过滤顺序（任一不满足即隐藏）：

        1. `scope_allows`（§5.3）：`metric_enabled` 才允许米制 Tool；
           `fallback_2d_only` 只允许不依赖深度/位姿/点云/对象的 Tool；
        2. 证据匹配（§7.2）：逐 Tool 按 `requires_evidence` / `tolerates_degraded`；
        3. 实际装载产物（硬约束 23）：`available` 给定时按事实收窄 ——
           scope 是**声明**，句柄的 available_artifacts 是**事实**；不一致时以事实为准，
           避免 prompt 里列出必然抛 `ArtifactUnavailableError` 的 Tool。
        """
        names = self.names()
        if scope is not None:
            names = [n for n in names if scope_allows(self._tools[n].spec, scope)]
        if evidence_profile is not None:
            names = [n for n in names
                     if evidence_visible(self._tools[n].spec, evidence_profile)]
        if available is not None:
            avail = set(available)
            names = [n for n in names
                     if set(self._tools[n].spec.requires_artifacts).issubset(avail)]
        return names

    def docs(self, scope: Optional[str] = None,
             available: Optional[Iterable[str]] = None,
             *,
             evidence_profile: Optional[EvidenceProfile] = None,
             gate_passed: Optional[bool] = None,
             allowed_metric_tasks: Optional[Iterable[str]] = None,
             question_type: str = "") -> str:
        """只暴露 Tool 文档（名称/描述/参数签名 + 所需产物 + 所需证据）。

        `scope` = `SceneState.question_tool_scope`（§5.3）。关于 scope 单调：
        `docs(full_3d) ⊆ docs(metric_enabled)` 仅在"多出米制 Tool"的意义上成立，
        不新增任何非米制工具。
        """
        lines = []
        for name in self.names_for_scope(
                scope, available, evidence_profile=evidence_profile,
                gate_passed=gate_passed,
                allowed_metric_tasks=allowed_metric_tasks, question_type=question_type):
            t = self._tools[name]
            sig = inspect.signature(t.fn)
            params = [str(p) for p in list(sig.parameters.values())[1:]]
            needs = ", ".join(t.spec.requires_artifacts) or "无"
            ev = ", ".join(t.spec.requires_evidence or []) or "无"
            lines.append(f"- {name}({', '.join(params)}): {t.spec.description}"
                         f" [requires_artifacts: {needs}；requires_evidence: {ev}]")
        return "\n".join(lines)

    def docs_header(self, scope: str, available: Iterable[str],
                    *, question_type: str = "",
                    gate_passed: Optional[bool] = None,
                    evidence_profile: Optional[EvidenceProfile] = None,
                    gate_missing: Optional[Iterable[str]] = None) -> str:
        """prompt 头部：显式写"当前 scope=…、可用产物=…、米制门状态=…"（§5.3/§13.3）。

        与 scene 摘要**同源**（都从 `scene_route` + `question_tool_scope` 派生），
        杜绝"头部 fallback、摘要 full_3d"自相矛盾（§5.3 不变量）。

        **2026-09-21 真实实测修正（重要）**：此前的实现**无条件**打印
        "米制证据门未通过 + 缺失子条件"，连 `object_rel_direction` /
        `object_rel_distance` 这类**根本不需要米制尺度**的题也照打 —— 而 gate 的
        `missing_subconditions` 里必然含 `question_type_is_metric`（"本题不是米制题"，
        这是**设计如此**，不是缺陷）。模型读到"证据门未通过 + 缺失子条件"就直接
        abstain：实测 4 个 rel_direction 题全部 abstain、4 个 rel_distance 题 3 个
        abstain，而这两类题的相对几何在比较中会把尺度 s 约掉，**完全不需要米制**。

        修正：只对**米制题型**报米制门状态；非米制题明确写"本题不需要米制尺度"，
        不让模型把"gate 对本题不适用"误读成"证据不足"。
        """
        avail = ", ".join(sorted(available)) or "（无）"
        metric_q = str(question_type or "") in set(METRIC_TASK_TYPES)
        if not metric_q:
            metric_line = (
                f"本题题型={question_type or '未分类'}**不需要米制尺度** ——"
                "相对距离/方向、路线、外观顺序都在全局尺度 s 下自动约掉 s，"
                "直接用世界系（归一化单位）的 Tool 作答即可，不要因为"
                "「米制证据门」的状态而 abstain（那道门只约束米制数值题）。"
            )
        elif gate_passed:
            metric_line = (
                f"米制证据门已通过（本题题型={question_type}），"
                "允许调用上面列出的米制 Tool。")
        else:
            metric_line = (
                f"**米制证据门未通过**（本题题型={question_type}）"
                + (f"，缺失子条件={sorted(gate_missing)}" if gate_missing else "")
                + "；依赖米制尺度的 Tool 已被收回，"
                "严禁用世界单位/相对单位冒充米制数值。"
                "若本题必须给米制量，请在 program 里直接调用 "
                'ReturnAnswer("abstain")。')
        ev_line = ""
        if evidence_profile is not None:
            ev_line = ("证据画像=" + ", ".join(
                f"{c}:{evidence_profile.state(c)}" for c in CAPABILITIES) + "。\n")
        return (f"当前 question_tool_scope={scope}；可用重建产物={avail}。\n"
                f"{ev_line}{metric_line}\n"
                f"只允许调用下面列出的 Tool；它们所需的产物与证据都已就绪。"
                f"若某个量在当前条件下无法获得，请直接 ReturnAnswer(\"abstain\")，"
                f"不要猜测数值。")

    # ---- 参数校验 ----
    @staticmethod
    def _validate_args(spec_name: str, fn: Callable[..., Any], args: dict) -> None:
        sig = inspect.signature(fn)
        params = list(sig.parameters.values())[1:]  # 跳过 handle
        required = {
            p.name for p in params if p.default is inspect.Parameter.empty
        }
        allowed = {p.name for p in params}
        missing = required - set(args)
        unknown = set(args) - allowed
        if missing:
            raise ToolArgValidationError(f"Tool {spec_name} 缺参数: {sorted(missing)}")
        if unknown:
            raise ToolArgValidationError(f"Tool {spec_name} 未知参数: {sorted(unknown)}")

    # ---- 统一调用（§4 M6 伪代码）----
    def call_tool(
        self,
        name: str,
        args: dict,
        scene: SceneHandle,
        mode: ToolSource = "real",
        mock_switch: Optional[MockSwitch] = None,
    ) -> ToolResult:
        """validate(args) → **契约校验（fail-closed）** → mock 解析 → 执行 → ToolResult。

        v6 双重校验（§13.2）：先按 `question_tool_scope` 收窄，再按 `EvidenceProfile`
        逐项匹配；两者都过才执行。契约违规（产物/证据缺失、米制门未过、域值错误）
        写入 `error_code`，由 M10 kernel 归因到 `tool_contract`。
        """
        entry = self.get(name)
        self._validate_args(name, entry.fn, args)

        scope = scene.question_tool_scope
        available = scene.available_artifacts
        profile = scene.evidence_profile

        # ① scope 收窄（§5.3）：scope 之外的工具即便被模型写出也不得执行
        if not scope_allows(entry.spec, scope):
            # 归因（§9.12）：scope 违规时把"缺哪些产物"一并写进异常，
            # 让 M10 的 `tool_contract` 桶能给出可回放的缺失清单（而非一句"越权"）。
            missing = sorted(set(entry.spec.requires_artifacts) - set(available))
            raise ToolContractError(
                name,
                f"Tool {name} 不在当前 question_tool_scope={scope} 的允许集合内"
                "（§5.3：逐题只收窄；米制 Tool 需 metric_enabled）",
                args=args, route=scene.scene_route,
                missing=missing, available=available)

        # ② 米制 Tool 的逐题授权（v5 HC33 语义的题型维度补充）
        check_metric_task_contract(
            name, entry.spec.supported_metric_tasks,
            scene.allowed_metric_tasks, scene.question_type, args=args)

        # ③ 证据契约（§7.2/§13.2 执行期二次校验）
        check_evidence_contract(
            name, entry.spec, profile,
            gate_passed=scene.metric_gate_passed,
            route=scene.scene_route, args=args)

        # ④ 产物契约：缺产物直接抛 ArtifactUnavailableError（不静默返回假值）
        check_artifact_contract(name, entry.spec.requires_artifacts, available,
                               route=scene.scene_route, args=args)

        switch = mock_switch or MockSwitch()
        fn, source = switch.resolve(
            name, entry.fn, entry.spec.returns_schema_ref, args, scene, mode
        )
        # T4 防污染：准入阶段（mode=real）出现 mock_* source 必须抛错
        MockSwitch.assert_admission_clean(source, mode)

        digest = hashlib.sha256(
            f"{request_key(name, args)}:{scene.state_digest()}".encode()
        ).hexdigest()

        t0 = time.perf_counter()
        error: Optional[str] = None
        error_code: Optional[str] = None
        missing: list[str] = []
        try:
            value = fn(scene, **args)
        except ToolContractError as exc:  # 确定性契约异常 → 进失败归因（§4 M6 字段 9）
            value, error, error_code = None, f"{type(exc).__name__}: {exc}", exc.error_code
            missing = list(exc.missing)
            available = list(exc.available or available)
        except Exception as exc:  # noqa: BLE001 - 实现内部错误
            value, error = None, f"{type(exc).__name__}: {exc}"
            error_code = _error_code_of(exc)
        latency_ms = (time.perf_counter() - t0) * 1000.0

        degraded = degraded_evidence_flags(entry.spec, profile)
        return ToolResult(
            result_id=digest[:16],
            source_tool=name,
            tool=name,
            args=args,
            value=json.dumps(value, default=str),
            payload=(value if isinstance(value, dict) else None),
            status=("failed" if error is not None else "ok"),
            evidence_version=getattr(profile, "profile_version", "") or "",
            degraded_evidence=degraded,
            source=source,
            request_digest=digest,
            latency_ms=latency_ms,
            error=error,
            error_code=error_code,  # type: ignore[arg-type]
            missing_artifacts=missing,
            available_artifacts=sorted(set(available)),
        )


# 全局注册表（geometry_tools 等模块在 import 时注册进来）
REGISTRY = ToolRegistry()


def call_tool(
    name: str,
    args: dict,
    scene: SceneHandle,
    mode: ToolSource = "real",
    mock_switch: Optional[MockSwitch] = None,
) -> ToolResult:
    """§4 M6 伪代码的模块级入口，委托全局 REGISTRY。"""
    return REGISTRY.call_tool(name, args, scene, mode=mode, mock_switch=mock_switch)


__all__ = [
    "REGISTRY",
    "TOOL_FACE_VERSION",
    "ToolArgValidationError",
    "ToolNotFoundError",
    "ToolRegistry",
    "call_tool",
    "SCOPE_FALLBACK_2D_ONLY",
    "SCOPE_FULL_3D",
    "SCOPE_METRIC_ENABLED",
    "tool_allowed",
]
