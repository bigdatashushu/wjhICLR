"""§6.1 在线推理 FSM（硬约束 1：在线链绝对无 GPT-6）。

本模块属于在线推理链（M1–M13 状态编排），**不得 import governance / gpt6 相关模块**；
任何状态、任何 prompt 不得出现 GPT-6 调用/verify/重试裁决。
纯 Python 轻量状态机实现（不依赖 transitions 库）：状态枚举 + 转移表 + guard 函数。

状态链（v4 新增 SCALE_ESTIMATE / SCALE_CALIBRATE，§5.1）：
INGEST→INPUT_GATE→RECONSTRUCT→SCALE_ESTIMATE(多锚点融合)→SCALE_CALIBRATE(冻结校准器,
逐题授权)→QUALITY_GATE(写回 artifact)→CLASSIFY_TASK→RETRIEVE_SKILL
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
    # v4 §5.1：尺度估计与校准是**独立状态**，位于重建之后、质量门禁之前。
    # 两者都不改 route（route 只由质量决定）；它们只产出 ScaleAssessment
    # （scale/ci_rel/confidence/allowed_metric_tasks），供 M7 逐题授权消费。
    SCALE_ESTIMATE = "SCALE_ESTIMATE"
    SCALE_CALIBRATE = "SCALE_CALIBRATE"
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
                self.state = OnlineState.SCALE_ESTIMATE
            elif event == "skip":
                # 复用既有 artifact / mock_light：尺度评估已随 artifact 落盘，跳过两态
                self.state = OnlineState.QUALITY_GATE
            else:
                self._to_unanswerable()  # 重建不可恢复

        elif s is OnlineState.SCALE_ESTIMATE:
            # v4 §5.1：多锚点鲁棒融合（地平面/相机高 + 门/桌/椅等标准物体）。
            # 失败（无有效锚点/融合非有限）**不**中止 episode：落 low 档并继续。
            if event != "done":
                raise ValueError(f"{s} 不支持事件 {event}")
            self.state = OnlineState.SCALE_CALIBRATE

        elif s is OnlineState.SCALE_CALIBRATE:
            # v4 §5.1：冻结 conformal 校准器 + 逐题型授权。
            # 校准器缺失 / 口径校验失败 / 锚点冲突 / 经验覆盖不足 → confidence=low 且
            # allowed_metric_tasks=∅；**不得**改变质量合格场景的非尺度 full_3d route。
            if event != "done":
                raise ValueError(f"{s} 不支持事件 {event}")
            if ctx.get("metric_tasks_withdrawn"):
                self.answer_flags.append("scale_low_metric_tasks_withdrawn")
            if ctx.get("scale_conflict"):
                self.answer_flags.append("scale_conflict")
            self.state = OnlineState.QUALITY_GATE

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
            # guard（v4 HC33 逐题型米制授权；G8 拒答门已随指标删除）：题型识别后立即生效
            # ctx: question_ok（False = 该题拒答）/ reject_flag / downgrade_2d_only
            if event != "done":
                raise ValueError(f"{s} 不支持事件 {event}")
            if not ctx.get("question_ok", True):
                self.answer_flags.append(str(ctx.get("reject_flag", "question_gate_reject")))
                self._to_unanswerable()
            else:
                if ctx.get("downgrade_2d_only"):
                    self.answer_flags.append("fallback_2d_only")
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
            elif event == "contract_recover":
                # D-3：回灌一次 / 裁剪 prompt 重生成一次 → 重执行（不消耗重启预算）
                self.answer_flags.append("tool_contract_recovered")
                pass
            elif event == "contract_fail":
                # D-3（v3 §6.1 新边）：两次机会用尽 → **显式 abstain**，主榜按错计
                self.answer_flags.append("tool_contract")
                self._to_unanswerable()
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
