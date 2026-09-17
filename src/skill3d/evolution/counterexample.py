"""M18 最小反例挖掘：失败 trace 逐帧/逐 Tool 裁剪（shrink 框架）。

MVP：自动 shrink 未接入 hypothesis（未安装），提供贪心 delta-debugging 框架；
默认 shrunk_by="human" 占位（M18 异常降级）。
"""

from __future__ import annotations

import uuid
from typing import Callable, Sequence

from skill3d.schemas import CounterexampleBundle, CounterexampleCase, PairedOutcome


def shrink_trace(steps: Sequence, still_fails: Callable[[list], bool]) -> list:
    """贪心 shrink：逐步删除元素，若仍触发失败则保留删除，直到最小。

    steps：trace 帧/Tool 调用序列；still_fails：子序列是否仍复现失败。
    """
    current = list(steps)
    i = 0
    while i < len(current):
        trial = current[:i] + current[i + 1:]
        if trial and still_fails(trial):
            current = trial  # 删除成功，继续在同一位置尝试
        else:
            i += 1
    return current


def mine_counterexamples(outcome: PairedOutcome,
                         failed_trace_refs: list[str],
                         source_revision_id: str,
                         regression_set_ref: str = "",
                         shrunk_by: str = "human") -> CounterexampleBundle:
    """从失败分支挖反例包。MVP shrunk_by="human" 占位（自动 shrink 为增强版）。"""
    cases = [
        CounterexampleCase(
            counterexample_id=f"ce-{uuid.uuid4().hex[:12]}",
            branch_id=outcome.arm_b_branch_id,
            minimal_episode_ref=ref,
            trigger_trace_ref=ref,
            originating_test="paired_ab",
            metamorphic_transform=None,
            shrunk_by=shrunk_by,  # type: ignore[arg-type]
        )
        for ref in failed_trace_refs
    ]
    summary = (
        f"paired {outcome.pair_id}: n={outcome.n_episodes}, delta={outcome.delta:.4f}, "
        f"ci95=[{outcome.ci95_lo:.4f},{outcome.ci95_hi:.4f}], "
        f"failed={len(failed_trace_refs)}"
    )  # gpt6_visible_summary：只含失败类型/统计摘要，不含答案（硬约束 19）
    from datetime import datetime, timezone
    return CounterexampleBundle(
        bundle_id=f"bundle-{uuid.uuid4().hex[:12]}",
        source_revision_id=source_revision_id,
        failed_episode_refs=failed_trace_refs,
        minimal_counterexamples=cases,
        metamorphic_transforms=[],
        regression_set_ref=regression_set_ref,
        gpt6_visible_summary=summary,
        generated_at=datetime.now(timezone.utc).isoformat(),
    )
