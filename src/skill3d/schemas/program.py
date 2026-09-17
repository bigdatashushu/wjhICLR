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
    program_id: str
    calls: list[ToolCall]
    results: list[ToolResult]
    stdout_tail: str
    error_code: Optional[
        Literal["timeout", "oom", "disk_full", "violation_syntax",
                "violation_runtime", "violation_policy"]
    ]
    steps: int
    wallclock_s: float
