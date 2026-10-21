"""§5.3 工具 Schema。"""

from typing import Literal, Optional

from pydantic import model_validator

from . import Spec

ToolSource = Literal["real", "mock_interface", "mock_replay", "mock_light"]

# Tool 执行期失败归因（§4 M6 字段 9 / §5.4 / v6 §9.12）：与 ProgramExecutionTrace.error_code 同族
ToolErrorCode = Literal["tool_contract", "confidence_gate", "domain_value",
                        "answer_already_given"]


SharedPremise = Literal[
    "geometry_3d", "world_frame", "metric_scale", "object_detection",
]


class FailureCause(Spec):
    """失败的事实范围；工具依赖本身不能证明共享证据已失效。"""

    code: str
    scope: Literal[
        "local_object", "local_result", "local_call", "shared_evidence", "unknown",
    ]
    object_ids: list[str] = []
    result_ids: list[str] = []
    shared_premises: list[SharedPremise] = []
    audit_ids: list[str] = []
    message: str = ""

    @model_validator(mode="after")
    def _scope_consistency(self):
        if self.shared_premises and self.scope != "shared_evidence":
            raise ValueError("Only shared_evidence failures may invalidate shared premises")
        if self.scope == "shared_evidence" and not self.shared_premises:
            raise ValueError("A shared_evidence failure must identify a shared premise")
        return self


class ToolSpec(Spec):
    """Tool 元数据（§4 M6 / v6 §8、§17.4）。

    `requires_artifacts` 必须在注册处**显式写出**（不写就没有默认值 → 构造直接报错），
    取值域与 `tools.contract.ROUTE_ARTIFACTS` 一致：`frames / intrinsics / depth /
    poses / point_cloud / objects / scale`。

    v6（D6）新增两列，**都必须在注册处显式写出**（§17.4 硬约束 7：无
    `requires_evidence` 声明的 Tool 不得进入注册表）：

    - `requires_evidence`：本 Tool 依赖哪些证据能力（词汇表见
      `schemas.evidence.CAPABILITIES`）。任一项 `unavailable` → 该 Tool 从
      `docs(question_tool_scope)` 隐藏、执行期抛 `ArtifactUnavailableError`；
    - `requires_local_facts`：工具局部事实（如世界坐标约定），不进入公共画像。

    §7.2 的关键语义是**逐 Tool 收窄**：某能力失败只隐藏依赖它的 Tool，
    不连带隐藏其他 Tool。因此本声明是"单项失败只收回依赖该证据的工具"的
    唯一实现点。

    v4 HC33 遗留列 `supported_metric_tasks`：v6 用 `requires_evidence` 含
    `metric_scale` 表达"米制 Tool"，该列保留作题型级收窄的第二维。
    """

    name: str
    description: str
    args_schema_ref: str
    returns_schema_ref: str
    cost_estimate_ms: float
    source_default: ToolSource
    requires_artifacts: list[str]  # 显式声明，禁止默认（硬约束 23）
    # v6 D6：证据依赖（显式声明，禁止默认 —— 缺声明直接构造报错）
    requires_evidence: list[str]
    requires_local_facts: list[Literal["world_frame"]] = []
    # v4 HC33：米制 Tool 显式列出支持题型；非米制 Tool 保持空（默认）
    supported_metric_tasks: list[str] = []


class ToolCall(Spec):
    tool: str
    args: dict
    call_id: str


class ToolResult(Spec):
    """Tool 调用结果（v6 §5.7 + v5 审计字段）。

    v6 状态语义（§14.1）：

    - `status="failed"` 的结果**不进入** validated observations；
    - 若共享前提失效，依赖该前提的旧结果被**级联撤销**：从 validated observations
      移除并标记 `invalidated_by`（§14.1/§6.4）；
    - `evidence_version` 记录产出该结果时的 EvidenceProfile 版本，
      使"这分数是在哪个证据状态下算出来的"可审计。
    """

    # v6 §5.7 规范字段
    result_id: str = ""
    source_tool: str = ""
    status: Literal["ok", "failed"] = "ok"
    evidence_version: str = ""
    payload: Optional[dict] = None
    # Empty in legacy/internal records; model kernels stamp their fixed boundary.
    # payload/value remain RAW data, never the model response envelope.
    model_response_schema_version: str = ""
    # 级联撤销时填入导致失效的 result_id 列表（§14.1）
    invalidated_by: Optional[list[str]] = None
    # v9 §6.4：本次调用的统一授权收据（含被拒绝的调用 —— 同样要有凭据）
    authorization: dict = {}
    # 答案来自"降级证据"时的显式标记（§7.2：degraded 但容忍 → 暴露且带标记）
    degraded_evidence: list[str] = []
    failure_cause: Optional[FailureCause] = None
    failure_guidance: dict = {}  # Descriptive scope/actions; never changes authorization or invalidation
    # 名称解析后的实际几何输入；观察工具不声明对象几何依赖。
    object_dependencies: list[str] = []
    # 实参实际解析到的 ID（包括局部校验失败），不声称语义选择正确。
    resolved_object_ids: list[str] = []
    # 显式派生关系，用于对结果的传递撤销。
    result_dependencies: list[str] = []

    # v5 审计字段（保留：receipts/回放/失败归因需要）
    tool: str = ""
    args: dict = {}
    value: str = "null"
    source: ToolSource = "real"
    request_digest: str = ""
    latency_ms: float = 0.0
    error: Optional[str] = None
    # 失败归因（成功为 None）：tool_contract / confidence_gate / domain_value /
    # answer_already_given
    error_code: Optional[ToolErrorCode] = None
    # tool_contract 归因细节（缺失产物 / 可用产物 / route），供回灌与审计
    missing_artifacts: list[str] = []
    available_artifacts: list[str] = []
