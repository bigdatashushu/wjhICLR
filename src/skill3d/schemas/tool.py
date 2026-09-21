"""§5.3 工具 Schema。"""

from typing import Literal, Optional

from . import Spec

ToolSource = Literal["real", "mock_interface", "mock_replay", "mock_light"]

# Tool 执行期失败归因（§4 M6 字段 9 / §5.4）：与 ProgramExecutionTrace.error_code 同族
ToolErrorCode = Literal["tool_contract", "confidence_gate", "domain_value"]


class ToolSpec(Spec):
    """Tool 元数据（§4 M6）。

    `requires_artifacts` 必须在注册处**显式写出**（不写就没有默认值 → 构造直接报错），
    取值域与 `tools.contract.ROUTE_ARTIFACTS` 一致：`frames / intrinsics / depth /
    poses / point_cloud / objects / scale`。

    v4（硬约束 33）：**依赖米制尺度的 Tool** 还必须声明 `supported_metric_tasks`
    ——它支持哪些题型（取值域见 `schemas.reconstruction.METRIC_TASK_TYPES`）。
    执行期由 wrapper 检查 `question_type ∈ allowed_metric_tasks ∩ supported_metric_tasks`，
    否则抛 `ConfidenceGateError`。不依赖米制尺度的 Tool 显式声明空列表（默认值），
    且**不得**读取 `scale_confidence`。
    """

    name: str
    description: str
    args_schema_ref: str
    returns_schema_ref: str
    cost_estimate_ms: float
    source_default: ToolSource
    requires_artifacts: list[str]  # 显式声明，禁止默认（硬约束 23）
    # v4 HC33：米制 Tool 显式列出支持题型；非米制 Tool 保持空（默认）
    supported_metric_tasks: list[str] = []


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
    # 失败归因（成功为 None）：tool_contract / confidence_gate / domain_value
    error_code: Optional[ToolErrorCode] = None
    # tool_contract 归因细节（缺失产物 / 可用产物 / route），供回灌与审计
    missing_artifacts: list[str] = []
    available_artifacts: list[str] = []
