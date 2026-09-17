"""§6.1 在线推理 FSM（硬约束 1：在线链绝对无 GPT-6）。

本模块属于在线推理链（M1–M13 状态编排），**不得 import governance / gpt6 相关模块**；
任何状态、任何 prompt 不得出现 GPT-6 调用/verify/重试裁决。
纯 Python 轻量状态机实现（不依赖 transitions 库）：状态枚举 + 转移表 + guard 函数。

状态链：INGEST→INPUT_GATE→RECONSTRUCT→QUALITY_GATE→CLASSIFY_TASK→RETRIEVE_SKILL
→SYNTHESIZE_PROGRAM→STATIC_CHECK→SANDBOX_EXECUTE→GEOMETRY_VERIFY→BENCHMARK_EVAL
→ANSWER→LOG_TRACE；失败转移见 §6.3。
"""

from __future__ import annotations

from enum import Enum

# 有限次重试上限（TODO_CALIBRATE：起始参考值）
MAX_REGEN_ATTEMPTS = 3
MAX_KERNEL_RESTARTS = 2


class OnlineState(str, Enum):
    INGEST = "INGEST"
    INPUT_GATE = "INPUT_GATE"
    RECONSTRUCT = "RECONSTRUCT"
    QUALITY_GATE = "QUALITY_GATE"
    CLASSIFY_TASK = "CLASSIFY_TASK"
    RETRIEVE_SKILL = "RETRIEVE_SKILL"
    SYNTHESIZE_PROGRAM = "SYNTHESIZE_PROGRAM"
    STATIC_CHECK = "STATIC_CHECK"
    SANDBOX_EXECUTE = "SANDBOX_EXECUTE"
    GEOMETRY_VERIFY = "GEOMETRY_VERIFY"
    BENCHMARK_EVAL = "BENCHMARK_EVAL"
    ANSWER = "ANSWER"
    LOG_TRACE = "LOG_TRACE"  # 终态


TERMINAL = {OnlineState.LOG_TRACE}


class OnlineFSM:
    """在线推理状态机。ctx 携带各阶段产出与门禁判定。"""

    def __init__(self, max_regen: int = MAX_REGEN_ATTEMPTS,
                 max_kernel_restarts: int = MAX_KERNEL_RESTARTS) -> None:
        self.state = OnlineState.INGEST
        self.max_regen = max_regen
        self.max_kernel_restarts = max_kernel_restarts
        self.regen_count = 0
        self.kernel_restart_count = 0
        self.answer_flags: list[str] = []  # 如 geometry_rejected / unanswerable

    @property
    def terminated(self) -> bool:
        return self.state in TERMINAL

    def step(self, event: str, ctx: dict | None = None) -> OnlineState:
        """按 §6.3 转移表推进一步。event 见各状态分支；ctx 为判定输入。"""
        ctx = ctx or {}
        s = self.state

        if s is OnlineState.INGEST:
            # guard: split ok（final_test 永不进在线链）
            if event == "episode_ready" and ctx.get("split_ok", True):
                self.state = OnlineState.INPUT_GATE
            else:
                self._to_unanswerable()

        elif s is OnlineState.INPUT_GATE:
            if event != "gate_done":
                raise ValueError(f"{s} 不支持事件 {event}")
            action = ctx.get("action", "proceed")
            if action == "unanswerable":
                self._to_unanswerable()
            elif action == "fallback_2d_only":
                self.answer_flags.append("fallback_2d_only")
                self.state = OnlineState.QUALITY_GATE  # 受限 Tool 集，跳过重建
            else:
                self.state = OnlineState.RECONSTRUCT

        elif s is OnlineState.RECONSTRUCT:
            if event == "done":
                self.state = OnlineState.QUALITY_GATE
            else:
                self._to_unanswerable()  # 重建不可恢复

        elif s is OnlineState.QUALITY_GATE:
            if event != "gate_done":
                raise ValueError(f"{s} 不支持事件 {event}")
            action = ctx.get("action", "proceed")
            if action == "unanswerable":
                self._to_unanswerable()
            else:
                if action == "fallback_2d_only":
                    self.answer_flags.append("fallback_2d_only")
                self.state = OnlineState.CLASSIFY_TASK

        elif s is OnlineState.CLASSIFY_TASK:
            if event == "done":
                self.state = OnlineState.RETRIEVE_SKILL

        elif s is OnlineState.RETRIEVE_SKILL:
            # guard: Skill 前置满足（硬过滤）；不满足则用空模板继续
            if event == "done":
                self.state = OnlineState.SYNTHESIZE_PROGRAM

        elif s is OnlineState.SYNTHESIZE_PROGRAM:
            if event == "done":
                self.state = OnlineState.STATIC_CHECK

        elif s is OnlineState.STATIC_CHECK:
            if event == "pass":
                self.state = OnlineState.SANDBOX_EXECUTE
            elif event == "fail":
                self.regen_count += 1
                if self.regen_count < self.max_regen:
                    self.state = OnlineState.SYNTHESIZE_PROGRAM  # 有限次重生成
                else:
                    self._to_unanswerable()

        elif s is OnlineState.SANDBOX_EXECUTE:
            if event == "ok":
                self.state = OnlineState.GEOMETRY_VERIFY
            elif event == "error":
                self.kernel_restart_count += 1
                if self.kernel_restart_count < self.max_kernel_restarts:
                    pass  # 重注入/重启 kernel：停留本状态重试
                else:
                    self.answer_flags.append("no_tool_fallback")
                    self._to_unanswerable()  # 两级兜底后仍失败

        elif s is OnlineState.GEOMETRY_VERIFY:
            if event == "pass":
                self.state = OnlineState.BENCHMARK_EVAL
            elif event == "reject":
                self.answer_flags.append("geometry_rejected")
                self.state = OnlineState.BENCHMARK_EVAL

        elif s is OnlineState.BENCHMARK_EVAL:
            if event == "done":
                self.state = OnlineState.ANSWER

        elif s is OnlineState.ANSWER:
            if event == "done":
                self.state = OnlineState.LOG_TRACE  # 任何结局都写 trace

        return self.state

    def _to_unanswerable(self) -> None:
        self.answer_flags.append("unanswerable")
        self.state = OnlineState.ANSWER
