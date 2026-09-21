"""Tool Registry（§4 M6）：预封装稳定函数注册表（硬约束 14）。

Tool 是预先封装好的稳定 Python 函数，不是每题生成；
Coding Agent（Qwen3-VL-8B）只生成编排这些 Tool 的 episode-specific program。

两条纪律（D-3）：
- 静态裁剪 `docs(route=...)`：只暴露当前 route 下产物齐备的 Tool 文档；
- 执行期 fail-closed：`call_tool` 在调用实现前校验 `requires_artifacts`，
  缺失抛 `ArtifactUnavailableError`（见 `tools/contract.py`），绝不静默返回假值。
"""

from __future__ import annotations

import hashlib
import inspect
import json
import time
from typing import Any, Callable, Iterable, Optional

from skill3d.schemas import ToolResult, ToolSpec
from skill3d.schemas.reconstruction import METRIC_TASK_TYPES
from skill3d.schemas.tool import ToolSource

from .contract import (
    KNOWN_ARTIFACTS,
    ToolContractError,
    available_artifacts_for,
    check_artifact_contract,
    check_metric_task_contract,
    route_artifacts_for_question,
    tool_allowed,
)
from .mock_switch import MockSwitch, request_key
from .scene_handle import SceneHandle


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
            unknown_tasks = set(spec.supported_metric_tasks) - set(METRIC_TASK_TYPES)
            if unknown_tasks:
                raise ValueError(
                    f"Tool {spec.name} 声明了未知米制题型 {sorted(unknown_tasks)}；"
                    f"词汇表见 schemas.reconstruction.METRIC_TASK_TYPES")
            if spec.supported_metric_tasks and "scale" not in spec.requires_artifacts:
                # 米制 Tool 必须显式依赖 scale 产物，否则会在尺度不可用时
                # 静默给出"相对单位当米制"的假值（硬约束 23 / HC33）
                raise ValueError(
                    f"Tool {spec.name} 声明了 supported_metric_tasks "
                    f"{spec.supported_metric_tasks}，但 requires_artifacts 未包含 'scale'"
                    "（米制 Tool 必须 fail-closed 依赖尺度产物）")
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

    def supported_metric_tasks(self, name: str) -> list[str]:
        return list(self.get(name).spec.supported_metric_tasks)

    def names_for_route(self, route: Optional[str] = None,
                        available: Optional[Iterable[str]] = None,
                        *,
                        allowed_metric_tasks: Optional[Iterable[str]] = None,
                        question_type: str = "") -> list[str]:
        """当前 route/题型下**可暴露**的 Tool 名（`docs(route)` 的过滤依据）。

        `available` 给定时再按"实际装载的产物"收窄（硬约束 23 的执行期判据同源）：
        route 是**声明**，句柄的 available_artifacts 是**事实**；两者不一致时（例如
        route=full_3d 但 M5 未产出 objects）以事实为准，避免 prompt 里列出必然
        抛 `ArtifactUnavailableError` 的 Tool。

        v4（HC33）：`allowed_metric_tasks` / `question_type` 给定时，进一步裁掉
        「依赖米制尺度但当前题型未授权」的 Tool —— 与执行期 `check_metric_task_contract`
        同一判定，保证 prompt 里列出的 Tool 都能真正执行。
        """
        if route is None and available is None and not allowed_metric_tasks \
                and not question_type:
            return self.names()
        names = self.names()
        if route is not None:
            names = [n for n in names
                     if tool_allowed(self._tools[n].spec.requires_artifacts, route)]
        if available is not None:
            avail = set(available)
            names = [n for n in names
                     if set(self._tools[n].spec.requires_artifacts).issubset(avail)]
        if allowed_metric_tasks is not None or question_type:
            allowed = {str(t) for t in (allowed_metric_tasks or set())}
            qt = str(question_type or "")
            metric_tasks_of = self._tools
            names = [
                n for n in names
                if (not metric_tasks_of[n].spec.supported_metric_tasks)
                or (bool(qt) and qt in allowed
                    and qt in set(metric_tasks_of[n].spec.supported_metric_tasks))
            ]
        return names

    def docs(self, route: Optional[str] = None,
             available: Optional[Iterable[str]] = None,
             *,
             allowed_metric_tasks: Optional[Iterable[str]] = None,
             question_type: str = "") -> str:
        """只暴露 Tool 文档（名称/描述/参数签名 + 所需产物），绝不暴露答案（§4 M8）。

        `route` 给定时按 `tools.contract.ROUTE_ARTIFACTS` 静态裁剪（D-3a）；
        关于 route 单调：`docs(full_3d) ⊇ docs(fallback_2d_only)`。
        `available` 给定时再按实际装载产物收窄（与执行期 fail-closed 判据同源）。
        `allowed_metric_tasks` / `question_type` 给定时按 HC33 逐题授权收窄。
        """
        lines = []
        for name in self.names_for_route(route, available,
                                         allowed_metric_tasks=allowed_metric_tasks,
                                         question_type=question_type):
            t = self._tools[name]
            sig = inspect.signature(t.fn)
            params = [str(p) for p in list(sig.parameters.values())[1:]]
            needs = ", ".join(t.spec.requires_artifacts) or "无"
            metric = ("；米制题型=" + ", ".join(t.spec.supported_metric_tasks)
                      if t.spec.supported_metric_tasks else "")
            lines.append(f"- {name}({', '.join(params)}): {t.spec.description}"
                         f" [requires_artifacts: {needs}{metric}]")
        return "\n".join(lines)

    def docs_header(self, route: str, available: Iterable[str],
                    *, allowed_metric_tasks: Optional[Iterable[str]] = None,
                    question_type: str = "") -> str:
        """prompt 头部：显式写"当前 route=…，可用产物=…，允许的米制题型=…"（§3 M6）。"""
        avail = ", ".join(sorted(available)) or "（无）"
        allowed = sorted({str(t) for t in (allowed_metric_tasks or set())})
        metric_line = (
            f"允许的米制题型={', '.join(allowed)}（当前题型={question_type or '未分类'}）"
            if allowed else
            f"允许的米制题型=（无）（当前题型={question_type or '未分类'}）"
            "；依赖米制尺度的 Tool 已被收回，请勿用相对单位冒充米制")
        return (f"当前 route={route}；可用重建产物={avail}。\n"
                f"{metric_line}。\n"
                f"只允许调用下面列出的 Tool；它们所需的产物都已就绪。"
                f"若某个量在当前 route 下无法获得，请在 program 里直接 abstain，"
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

        ToolResult 强制带 source 与 sha256 request_digest；
        契约违规（产物缺失/局部质量门未过/域值错误）写入 `error_code`，
        由 M10 kernel 归因到 `tool_contract`（D-3，硬约束 23）。
        """
        entry = self.get(name)
        self._validate_args(name, entry.fn, args)

        route = scene.route
        available = scene.available_artifacts

        # v4 HC33：米制 Tool 的**逐题授权**先于产物可用性判定 —— 授权失败属于
        # "局部质量门未过"（ConfidenceGateError，§3 M6 明文），不是"产物缺失"。
        # 授权通过后再按硬约束 23 校验产物；两者都是 ToolContractError 家族，
        # kernel 统一归因到 `tool_contract` 桶。
        check_metric_task_contract(
            name, entry.spec.supported_metric_tasks,
            scene.allowed_metric_tasks, scene.question_type, args=args)

        # 执行期 fail-closed：缺产物直接抛 ArtifactUnavailableError（不静默返回假值）
        check_artifact_contract(name, entry.spec.requires_artifacts, available,
                               route=route, args=args)

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
        except Exception as exc:  # noqa: BLE001 - 实现内部错误（含未归族的 ValueError）
            value, error = None, f"{type(exc).__name__}: {exc}"
            error_code = _error_code_of(exc)
        latency_ms = (time.perf_counter() - t0) * 1000.0

        return ToolResult(
            tool=name,
            args=args,
            value=json.dumps(value, default=str),
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
