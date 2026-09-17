"""Tool Registry（§4 M6）：预封装稳定函数注册表（硬约束 14）。

Tool 是预先封装好的稳定 Python 函数，不是每题生成；
Coding Agent（Qwen3-VL-8B）只生成编排这些 Tool 的 episode-specific program。
"""

from __future__ import annotations

import hashlib
import inspect
import json
import time
from typing import Any, Callable, Optional

from skill3d.schemas import ToolResult, ToolSpec
from skill3d.schemas.tool import ToolSource

from .mock_switch import MockSwitch, request_key
from .scene_handle import SceneHandle


class ToolNotFoundError(KeyError):
    pass


class ToolArgValidationError(ValueError):
    pass


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

    def docs(self) -> str:
        """只暴露 Tool 文档（名称/描述/参数签名），绝不暴露答案（§4 M8）。"""
        lines = []
        for name in self.names():
            t = self._tools[name]
            sig = inspect.signature(t.fn)
            params = [str(p) for p in list(sig.parameters.values())[1:]]
            lines.append(f"- {name}({', '.join(params)}): {t.spec.description}")
        return "\n".join(lines)

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
        """校验 args → mock_switch 解析 → 经 SceneHandle 执行 → ToolResult。

        ToolResult 强制带 source 与 sha256 request_digest。
        """
        entry = self.get(name)
        self._validate_args(name, entry.fn, args)

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
        try:
            value = fn(scene, **args)
        except Exception as exc:  # Tool 抛确定性异常 → 进失败归因（§4 M6 字段 9）
            value, error = None, f"{type(exc).__name__}: {exc}"
        latency_ms = (time.perf_counter() - t0) * 1000.0

        return ToolResult(
            tool=name,
            args=args,
            value=json.dumps(value, default=str),
            source=source,
            request_digest=digest,
            latency_ms=latency_ms,
            error=error,
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
