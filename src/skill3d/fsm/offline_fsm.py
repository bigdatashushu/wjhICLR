"""§6.2 离线演进 FSM。

CLUSTER_TRACES → GPT6_SYNTHESIZE → LEAKAGE_CHECK
→ OPTIMIZATION_LOOP{ SYNTHESIZE → STATIC_CHECK → TEST(L1→L2→L3) → DIAGNOSE → REVISE }
→ PROMOTE / REJECT / QUARANTINE。

红线：L3 outer 只跑一次，失败即 REJECT，绝不基于 outer 失败再修订（硬约束 10）。
纯 Python 实现（不依赖 transitions 库）。
"""

from __future__ import annotations

from enum import Enum

# TODO_CALIBRATE：停止条件起始参考值
MAX_REVISIONS = 10


class OfflineState(str, Enum):
    CLUSTER_TRACES = "CLUSTER_TRACES"
    GPT6_SYNTHESIZE = "GPT6_SYNTHESIZE"
    LEAKAGE_CHECK = "LEAKAGE_CHECK"
    LOOP_SYNTHESIZE = "LOOP_SYNTHESIZE"
    LOOP_STATIC_CHECK = "LOOP_STATIC_CHECK"
    TEST_L1 = "TEST_L1"
    TEST_L2 = "TEST_L2"
    TEST_L3 = "TEST_L3"
    DIAGNOSE = "DIAGNOSE"
    REVISE = "REVISE"
    PROMOTE = "PROMOTE"      # 终态
    REJECT = "REJECT"        # 终态
    QUARANTINE = "QUARANTINE"  # 终态


TERMINAL = {OfflineState.PROMOTE, OfflineState.REJECT, OfflineState.QUARANTINE}


class OfflineFSM:
    def __init__(self, max_revisions: int = MAX_REVISIONS) -> None:
        self.state = OfflineState.CLUSTER_TRACES
        self.max_revisions = max_revisions
        self.revision_count = 0
        self.outer_attempted = False  # 硬约束 10：outer 只跑一次

    @property
    def terminated(self) -> bool:
        return self.state in TERMINAL

    def step(self, event: str, ctx: dict | None = None) -> OfflineState:
        ctx = ctx or {}
        s = self.state

        if s is OfflineState.CLUSTER_TRACES:
            # guard：同题型 ≥ N_min 跨 scene（TODO_CALIBRATE）
            if event == "clustered" and ctx.get("cross_scene_ok", False):
                self.state = OfflineState.GPT6_SYNTHESIZE
            else:
                self.state = OfflineState.REJECT  # 样本不足不归纳

        elif s is OfflineState.GPT6_SYNTHESIZE:
            if event == "candidate_ready":
                self.state = OfflineState.LEAKAGE_CHECK
            elif event == "gpt6_unavailable":
                self.state = OfflineState.QUARANTINE  # GPT-6 不可用：暂停（M16 降级）

        elif s is OfflineState.LEAKAGE_CHECK:
            # 硬门：含答案/ID 即拒（硬约束 13/19）
            self.state = OfflineState.LOOP_SYNTHESIZE if event == "pass" \
                else OfflineState.REJECT

        elif s is OfflineState.LOOP_SYNTHESIZE:
            if event == "done":
                self.state = OfflineState.LOOP_STATIC_CHECK

        elif s is OfflineState.LOOP_STATIC_CHECK:
            if event == "pass":
                self.state = OfflineState.TEST_L1
            else:
                self.state = OfflineState.DIAGNOSE  # 失败收反馈进诊断

        elif s is OfflineState.TEST_L1:
            self.state = OfflineState.TEST_L2 if event == "pass" else OfflineState.DIAGNOSE

        elif s is OfflineState.TEST_L2:
            self.state = OfflineState.TEST_L3 if event == "pass" else OfflineState.DIAGNOSE

        elif s is OfflineState.TEST_L3:
            # 硬约束 10：outer 只跑一次；失败即 REJECT，不修订
            self.outer_attempted = True
            self.state = OfflineState.PROMOTE if event == "pass" else OfflineState.REJECT

        elif s is OfflineState.DIAGNOSE:
            if event == "bundle_ready":
                self.state = OfflineState.REVISE

        elif s is OfflineState.REVISE:
            # guard：不可原地改 + 修订次数上限
            if event == "patched" and self.revision_count < self.max_revisions:
                self.revision_count += 1
                self.state = OfflineState.LOOP_SYNTHESIZE
            else:
                self.state = OfflineState.REJECT

        return self.state
