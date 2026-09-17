"""§5.3 工具 Schema。"""

from typing import Literal, Optional

from . import Spec

ToolSource = Literal["real", "mock_interface", "mock_replay", "mock_light"]


class ToolSpec(Spec):
    name: str
    description: str
    args_schema_ref: str
    returns_schema_ref: str
    cost_estimate_ms: float
    source_default: ToolSource


class ToolCall(Spec):
    tool: str
    args: dict
    call_id: str


class ToolResult(Spec):
    tool: str
    args: dict
    value: str
    source: ToolSource
    request_digest: str
    latency_ms: float
    error: Optional[str]
