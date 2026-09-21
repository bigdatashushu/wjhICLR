"""持久 kernel（§4 M10）：code-as-action，跨 cell 变量存活。

MVP 为 in-process RestrictedNamespaceKernel（exec 于受控 namespace）；
Docker 隔离由 docker_manager 负责（AST 检查不能替代容器隔离，§4 M9 字段 12）。

注入保留名：frames / scene / tools / show / ReturnAnswer。
cell 超时与错误捕获；两级兜底（no-tool CoT → 正则抽取）留钩子。
"""

from __future__ import annotations

import contextlib
import inspect
import io
import signal
import types
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

from skill3d.schemas import ToolCall, ToolResult
from skill3d.tools.contract import (
    ArtifactUnavailableError,
    ConfidenceGateError,
    DomainValueError,
    ToolContractError,
)
from skill3d.tools.registry import ToolRegistry
from skill3d.tools.scene_handle import SceneHandle

DEFAULT_CELL_TIMEOUT_S = 30  # TODO_CALIBRATE


class _AnswerSlot:
    """ReturnAnswer 保留名实现：记录答案。"""

    def __init__(self) -> None:
        self.answer: Optional[str] = None

    def __call__(self, value: Any) -> None:
        self.answer = str(value)


class _Timeout:
    """SIGALRM cell 超时（仅主线程可用；非主线程静默退化为无超时）。"""

    def __init__(self, seconds: int) -> None:
        self.seconds = seconds
        self._active = False

    def __enter__(self):
        try:
            signal.signal(signal.SIGALRM, self._raise)
            signal.alarm(self.seconds)
            self._active = True
        except (ValueError, AttributeError):
            self._active = False  # 非主线程或无 SIGALRM 平台
        return self

    def __exit__(self, *exc):
        if self._active:
            signal.alarm(0)
        return False

    @staticmethod
    def _raise(signum, frame):
        raise TimeoutError("cell 执行超时")


@dataclass
class CellResult:
    stdout_tail: str
    error: Optional[str] = None
    # timeout / oom / violation_syntax / violation_runtime / violation_policy / tool_contract
    error_code: Optional[str] = None
    answer: Optional[str] = None
    new_vars: list[str] = field(default_factory=list)
    # D-3：命中契约违规后，ReturnAnswer 的答案不得采纳（上层据此 abstain）
    answer_untrusted: bool = False
    contract_violations: list[dict] = field(default_factory=list)


class RestrictedNamespaceKernel:
    """in-process 持久 kernel：跨 cell 变量存活。"""

    def __init__(
        self,
        tool_registry: ToolRegistry,
        scene: SceneHandle,
        frames: Optional[list] = None,
        mode: str = "real",
        cell_timeout_s: int = DEFAULT_CELL_TIMEOUT_S,
        mock_switch=None,
        # 两级兜底钩子（termination node：no-tool CoT → 正则抽取），由上层注入
        fallback_cot_fn: Optional[Callable[[], str]] = None,
        fallback_regex_fn: Optional[Callable[[str], Optional[str]]] = None,
        # tool call_id 生成器：重放确定性模式由上层注入计数器（§4 M17 同 seed 字节级一致）
        call_id_factory: Optional[Callable[[], str]] = None,
    ) -> None:
        self._registry = tool_registry
        self._scene = scene
        self._mode = mode
        self._mock_switch = mock_switch
        self._call_id_factory = call_id_factory or (lambda: uuid.uuid4().hex[:12])
        self.cell_timeout_s = cell_timeout_s
        self.answer_slot = _AnswerSlot()
        self.show_log: list[Any] = []
        self.fallback_cot_fn = fallback_cot_fn
        self.fallback_regex_fn = fallback_regex_fn
        # M10 产物：ProgramExecutionTrace 的 calls/results（§5.4）
        self.tool_calls: list[ToolCall] = []
        self.tool_results: list[ToolResult] = []
        # D-3：契约违规记录；一旦命中，ReturnAnswer 的答案不得采纳
        self.contract_violations: list[ToolResult] = []
        self.answer_untrusted: bool = False

        # tools 命名空间：tools.<name>(**args) → 经 REGISTRY.call_tool
        tools_ns = types.SimpleNamespace()
        for name in tool_registry.names():
            tools_ns.__dict__[name] = self._make_tool_fn(name)

        self._ns: dict[str, Any] = {
            "frames": frames or [],
            "scene": scene,
            "tools": tools_ns,
            "show": self._show,
            "ReturnAnswer": self.answer_slot,
        }
        # 同步注入 Tool 顶层名（program 可直接写 euclidean_distance(...)）
        for name in tool_registry.names():
            self._ns[name] = tools_ns.__dict__[name]

    # ---- 注入 ----
    def inject(self, **kwargs: Any) -> None:
        for k, v in kwargs.items():
            if k in ("show", "ReturnAnswer", "tools", "scene", "frames"):
                raise ValueError(f"禁止覆盖保留名: {k}")
            self._ns[k] = v

    def _record_failed_call(self, name: str, call_args: dict, exc: ToolContractError,
                            result: Optional[ToolResult] = None) -> None:
        """契约违规也要进 calls/results（§5.4 审计可回放），再抛给 run_cell 归因。"""
        if result is None:
            result = ToolResult(
                tool=name, args=call_args, value="null", source=self._mode,  # type: ignore[arg-type]
                request_digest="", latency_ms=0.0,
                error=f"{type(exc).__name__}: {exc}", error_code=exc.error_code,  # type: ignore[arg-type]
                missing_artifacts=list(exc.missing),
                available_artifacts=list(exc.available))
            self.tool_calls.append(
                ToolCall(tool=name, args=call_args, call_id=self._call_id_factory()))
            self.tool_results.append(result)
        self.contract_violations.append(result)
        self.answer_untrusted = True

    def _make_tool_fn(self, name: str) -> Callable[..., Any]:
        """Tool 包装：位置/关键字传参均可（program 里 `euclidean_distance(a, b)` 与
        `euclidean_distance(point_a=a, point_b=b)` 等价）。"""
        sig = inspect.signature(self._registry.get(name).fn)

        def _fn(*args, **kwargs):
            bound = sig.bind(self._scene, *args, **kwargs)
            call_args = {k: v for k, v in bound.arguments.items()
                         if k not in ("handle", "scene")}
            try:
                result = self._registry.call_tool(
                    name, call_args, self._scene, mode=self._mode,
                    mock_switch=self._mock_switch,
                )
            except ToolContractError as exc:
                # 执行期 fail-closed：产物缺失在调用实现前就抛（硬约束 23）
                self._record_failed_call(name, call_args, exc)
                raise
            # 记录到 ProgramExecutionTrace（§5.4）：调用与结果成对入库
            self.tool_calls.append(
                ToolCall(tool=name, args=call_args, call_id=self._call_id_factory())
            )
            self.tool_results.append(result)
            if result.error_code == "tool_contract":
                # D-3/硬约束 23：产物缺失等契约违规 → 确定性异常，绝不静默返回假值
                self._record_failed_call(name, call_args, ArtifactUnavailableError(
                    name, result.missing_artifacts or ["(未标注)"],
                    route=self._scene.route,
                    available=result.available_artifacts,
                    args=call_args), result=result)
                raise ArtifactUnavailableError(
                    name, result.missing_artifacts or ["(未标注)"],
                    route=self._scene.route,
                    available=result.available_artifacts,
                    args=call_args)
            if result.error_code == "confidence_gate":
                exc = ConfidenceGateError(
                    name, result.error or "局部质量门未过", route=self._scene.route,
                    available=result.available_artifacts, args=call_args)
                self._record_failed_call(name, call_args, exc, result=result)
                raise exc
            if result.error_code == "domain_value":
                exc = DomainValueError(
                    name, result.error or "域值错误", route=self._scene.route,
                    available=result.available_artifacts, args=call_args)
                self._record_failed_call(name, call_args, exc, result=result)
                raise exc
                self.answer_untrusted = True
                raise DomainValueError(
                    name, result.error or "域值错误", route=self._scene.route,
                    available=result.available_artifacts, args=call_args)
            if result.error is not None:
                raise RuntimeError(f"Tool {name} 执行失败: {result.error}")
            import json

            return json.loads(result.value)

        return _fn

    def _show(self, obj: Any) -> None:
        """show() 视觉反馈通道：MVP 仅记录，不渲染。"""
        self.show_log.append(obj)

    # ---- 执行 ----
    def run_cell(self, code: str) -> CellResult:
        """执行一个 cell；捕获 stdout / 错误；答案经 ReturnAnswer 记录。

        `ToolContractError` 家族（tool_contract / confidence_gate / domain_value）
        由调用方归因：不静默吞掉，也不当作普通运行时错误（§4 M10，硬约束 23）。
        """
        buf = io.StringIO()
        before = set(self._ns.keys())
        error: Optional[str] = None
        error_code: Optional[str] = None
        try:
            with contextlib.redirect_stdout(buf), _Timeout(self.cell_timeout_s):
                exec(compile(code, "<episode_program>", "exec"), self._ns)
        except TimeoutError:
            error, error_code = "cell 执行超时", "timeout"
        except SyntaxError:
            error, error_code = "语法错误", "violation_syntax"
        except (ArtifactUnavailableError, ConfidenceGateError, DomainValueError) as exc:
            # 契约违规一律记 tool_contract 桶（§5 FailureTaxonomy 已有该桶）
            error = f"{type(exc).__name__}: {exc}"
            error_code = "tool_contract"
            self.answer_untrusted = True
        except Exception as exc:
            error, error_code = f"{type(exc).__name__}: {exc}", "violation_runtime"
        new_vars = sorted(set(self._ns.keys()) - before)
        return CellResult(
            stdout_tail=buf.getvalue()[-4096:],
            error=error,
            error_code=error_code,
            answer=self.answer_slot.answer,
            new_vars=new_vars,
            answer_untrusted=self.answer_untrusted,
            contract_violations=[r.model_dump() for r in self.contract_violations],
        )

    # ---- 状态重置（回灌重执行前必须重注入，§4 M6 字段 9）----
    def reset_user_namespace(self) -> None:
        """清空用户命名空间并重注入保留名（保留 Tool/帧/答案槽）。

        回灌重生成后重执行前必须调用：per-episode 状态不得跨次执行泄漏
        （SpatialClaw §E.3 先例）。
        """
        tools_ns = self._ns["tools"]
        keep = ("frames", "scene", "tools", "show", "ReturnAnswer")
        preserved = {k: self._ns[k] for k in keep if k in self._ns}
        self._ns.clear()
        self._ns.update(preserved)
        for name in self._registry.names():
            self._ns[name] = tools_ns.__dict__[name]
        self.answer_slot.answer = None
        self.tool_calls = []
        self.tool_results = []
        self.contract_violations = []
        self.answer_untrusted = False

    def run_program(self, program_source: str) -> CellResult:
        return self.run_cell(program_source)

    # ---- 两级兜底（no-tool CoT → 正则抽取）----
    def run_with_fallback(self, program_source: str) -> CellResult:
        result = self.run_cell(program_source)
        if result.answer is not None:
            return result
        # 第一级：no-tool CoT 钩子
        if self.fallback_cot_fn is not None:
            cot_text = self.fallback_cot_fn()
            # 第二级：正则抽取钩子
            if self.fallback_regex_fn is not None:
                extracted = self.fallback_regex_fn(cot_text)
                if extracted is not None:
                    result.answer = extracted
        return result
