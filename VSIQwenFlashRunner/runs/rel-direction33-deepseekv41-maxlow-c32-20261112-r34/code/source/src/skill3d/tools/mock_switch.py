"""Tool 三档 Mock 切换（§4 M6 / §9.2）。

- real:           真实 SceneState 计算
- mock_interface: 仅按返回 Schema 造占位值（接口联调用）
- mock_replay:    固定输入回放历史 receipt；state_digest 不匹配则降级 real 并记录
- mock_light:     合成轻量 SceneState 真算（句柄由调用方注入）

准入阶段（mode=real）出现任何 mock_* source 必须抛 MockContaminationError（T4 防污染）。
"""

from __future__ import annotations

import hashlib
import json
import logging
from typing import Any, Callable, Optional

from skill3d.schemas.tool import ToolSource

from .scene_handle import SceneHandle

logger = logging.getLogger(__name__)


class MockContaminationError(RuntimeError):
    """准入阶段（real 模式）出现 mock_* source，属硬污染，必须拒绝。"""


# mock_interface 按 returns_schema_ref 造占位值
_MOCK_INTERFACE_DEFAULTS: dict[str, Any] = {
    "float": 0.0,
    "int": 0,
    "bool": False,
    "str": "mock",
    "list": [],
    "dict": {},
}


def request_key(tool: str, args: dict) -> str:
    """请求级稳定 key（mock_replay 检索用）。"""
    return hashlib.sha256(
        json.dumps({"tool": tool, "args": args}, sort_keys=True, default=str).encode()
    ).hexdigest()


class MockSwitch:
    """按 mode 解析 (fn, source)。"""

    def __init__(self, light_handle: Optional[SceneHandle] = None) -> None:
        # mock_replay 回放表：request_key -> {"state_digest": ..., "value": ...}
        self._replay_store: dict[str, dict[str, Any]] = {}
        # 合成轻量场景（mock_light 用）
        self._light_handle = light_handle
        # 降级记录（state_digest 不匹配 / replay miss）
        self.degradation_log: list[dict[str, Any]] = []

    # ---- replay 回放表管理 ----
    def load_replay(self, tool: str, args: dict, state_digest: str, value: Any) -> None:
        self._replay_store[request_key(tool, args)] = {
            "state_digest": state_digest,
            "value": value,
        }

    # ---- 核心解析 ----
    def resolve(
        self,
        tool: str,
        real_fn: Callable[..., Any],
        returns_schema_ref: str,
        args: dict,
        handle: Optional[SceneHandle],
        mode: ToolSource,
    ) -> tuple[Callable[..., Any], ToolSource]:
        """返回 (可调用, source)。mode=real 恒返回真实实现。"""
        if mode == "real":
            return real_fn, "real"

        if mode == "mock_interface":
            default = _MOCK_INTERFACE_DEFAULTS.get(returns_schema_ref, None)
            return (lambda h, **kw: default), "mock_interface"

        if mode == "mock_light":
            if self._light_handle is not None:
                light = self._light_handle
                return (lambda h, **kw: real_fn(light, **kw)), "mock_light"
            # 无轻量场景时降级为接口 mock 并记录
            self.degradation_log.append(
                {"tool": tool, "reason": "mock_light_no_handle", "fallback": "mock_interface"}
            )
            default = _MOCK_INTERFACE_DEFAULTS.get(returns_schema_ref, None)
            return (lambda h, **kw: default), "mock_interface"

        if mode == "mock_replay":
            key = request_key(tool, args)
            rec = self._replay_store.get(key)
            current_digest = handle.state_digest() if handle is not None else None
            if rec is None:
                self.degradation_log.append(
                    {"tool": tool, "reason": "replay_miss", "fallback": "real"}
                )
                return real_fn, "real"
            if rec["state_digest"] != current_digest:
                # state_digest 不匹配 → 判过期，降级 real 并记录（§4 M6 字段 9）
                self.degradation_log.append(
                    {
                        "tool": tool,
                        "reason": "state_digest_mismatch",
                        "expected": rec["state_digest"],
                        "actual": current_digest,
                        "fallback": "real",
                    }
                )
                return real_fn, "real"
            return (lambda h, **kw: rec["value"]), "mock_replay"

        raise ValueError(f"未知 mock 模式: {mode}")

    # ---- T4 防污染硬规则 ----
    @staticmethod
    def assert_admission_clean(source: ToolSource, mode: ToolSource) -> None:
        """准入阶段（mode=real）出现 mock_* source 必须抛错。"""
        if mode == "real" and source != "real":
            raise MockContaminationError(
                f"准入阶段 mode=real 但 Tool source={source}（mock 污染，T4 拒绝）"
            )
