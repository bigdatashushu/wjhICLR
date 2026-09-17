"""持久 kernel（§4 M10）：code-as-action，跨 cell 变量存活。

MVP 为 in-process RestrictedNamespaceKernel（exec 于受控 namespace）；
Docker 隔离由 docker_manager 负责（AST 检查不能替代容器隔离，§4 M9 字段 12）。

注入保留名：frames / scene / tools / show / ReturnAnswer。
cell 超时与错误捕获；两级兜底（no-tool CoT → 正则抽取）留钩子。
"""

from __future__ import annotations

import contextlib
import io
import signal
import types
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

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
    error_code: Optional[str] = None  # timeout / violation_syntax / violation_runtime / violation_policy
    answer: Optional[str] = None
    new_vars: list[str] = field(default_factory=list)


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
    ) -> None:
        self._registry = tool_registry
        self._scene = scene
        self._mode = mode
        self._mock_switch = mock_switch
        self.cell_timeout_s = cell_timeout_s
        self.answer_slot = _AnswerSlot()
        self.show_log: list[Any] = []
        self.fallback_cot_fn = fallback_cot_fn
        self.fallback_regex_fn = fallback_regex_fn

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

    def _make_tool_fn(self, name: str) -> Callable[..., Any]:
        def _fn(**args):
            result = self._registry.call_tool(
                name, args, self._scene, mode=self._mode, mock_switch=self._mock_switch
            )
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
        """执行一个 cell；捕获 stdout / 错误；答案经 ReturnAnswer 记录。"""
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
        except Exception as exc:
            error, error_code = f"{type(exc).__name__}: {exc}", "violation_runtime"
        new_vars = sorted(set(self._ns.keys()) - before)
        return CellResult(
            stdout_tail=buf.getvalue()[-4096:],
            error=error,
            error_code=error_code,
            answer=self.answer_slot.answer,
            new_vars=new_vars,
        )

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
