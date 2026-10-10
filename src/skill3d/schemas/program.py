"""§5.4 程序 Schema。"""

from typing import Literal, Optional

from . import Spec
from .tool import ToolCall, ToolResult


class EpisodeProgram(Spec):
    program_id: str
    program_source: str
    skill_semver_used: list[str]
    intended_answer_slot: str


class ASTCheckResult(Spec):
    ok: bool
    violations: list[str]
    allowed_tool_calls: list[str]


class ProgramExecutionTrace(Spec):
    run_id: str = ""
    qa_id: str = ""
    program_id: str
    calls: list[ToolCall]
    results: list[ToolResult]
    stdout_tail: str
    error_code: Optional[
        Literal["timeout", "oom", "disk_full", "violation_syntax",
                "violation_runtime", "violation_policy", "tool_contract"]
    ]  # tool_contract：Tool 所需产物缺失/局部质量门未过/域值错误（D-3，硬约束 23）
    steps: int
    wallclock_s: float
    # D-3：一旦 `ReturnAnswer` 依赖的某次 Tool 抛 ToolContractError，最终答案不得采纳
    answer_untrusted: bool = False
