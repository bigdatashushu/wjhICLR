"""T1–T4 链校验（§4 M11 / §9.2）。

- T1 Schema 校验（pydantic validate）
- T2 语义校验（占位规则）
- T3 receipt 哈希链完整性
- T4 source 防污染：source=real 声称但 receipt 标 mock_* → 拒（硬约束 13）
"""

from __future__ import annotations

from typing import Optional, Sequence

from pydantic import ValidationError

from skill3d.schemas import ProgramExecutionTrace
from skill3d.sandbox.receipt import SandboxReceipt, verify_chain


def t1_schema_check(trace_data: dict) -> tuple[bool, Optional[str]]:
    """T1：ProgramExecutionTrace 的 pydantic 校验。"""
    try:
        ProgramExecutionTrace.model_validate(trace_data)
        return True, None
    except ValidationError as exc:
        return False, str(exc)


def t2_semantic_check(trace: ProgramExecutionTrace, answer: Optional[str]) -> tuple[bool, Optional[str]]:
    """T2：语义占位规则。

    TODO_CALIBRATE: 占位实现——仅要求有非空答案且无 Tool 执行错误；
    更完整的语义一致性规则待集成时校准。
    """
    if answer is None or str(answer).strip() == "":
        return False, "答案为空"
    for r in trace.results:
        if r.error is not None:
            return False, f"Tool {r.tool} 执行错误: {r.error}"
    return True, None


def t3_receipt_chain_check(receipts: Sequence[SandboxReceipt]) -> tuple[bool, Optional[str]]:
    """T3：receipt 哈希链完整性。"""
    if not receipts:
        return False, "receipt 链为空"
    if not verify_chain(list(receipts)):
        return False, "receipt 哈希链校验失败（疑似篡改）"
    return True, None


def t4_source_check(
    trace: ProgramExecutionTrace,
    receipts: Sequence[SandboxReceipt],
    expected_mode: str = "real",
) -> tuple[bool, Optional[str]]:
    """T4 防污染：准入阶段（expected_mode=real）任何 mock_* source 拒绝；
    ToolResult 声称 real 但对应 receipt 标 mock_* 同样拒绝。
    """
    mock_receipt_events = [r for r in receipts if r.source.startswith("mock")]
    for r in trace.results:
        if expected_mode == "real" and r.source != "real":
            return False, f"Tool {r.tool} source={r.source}，准入阶段要求 real（T4 拒绝）"
        if r.source == "real" and mock_receipt_events:
            # 声称 real 的调用存在 mock_* receipt 记录 → 拒
            for rec in mock_receipt_events:
                if rec.event == f"tool_call:{r.tool}":
                    return False, (
                        f"Tool {r.tool} 声称 source=real 但 receipt {rec.receipt_id} 标 "
                        f"{rec.source}（T4 拒绝）"
                    )
    return True, None
