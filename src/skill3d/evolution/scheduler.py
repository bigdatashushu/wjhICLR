"""M17 实验单元调度：与 scheduling/gpu_scheduler 配合。

硬约束：paired A/B 同 episode 固定同 gpu_rank（硬约束 18 配套：隔离因果增益）。
"""

from __future__ import annotations

from skill3d.scheduling.gpu_scheduler import GPUScheduler


def schedule_paired_units(paired_unit_ids: list[str],
                          scheduler: GPUScheduler) -> dict[str, int]:
    """为 paired 实验单元分配 gpu_rank：同一 paired_unit_id 的 A/B 两臂同卡。

    返回 {paired_unit_id: gpu_rank}。分配按调度器轮转，但 pair 内固定。
    """
    assignment: dict[str, int] = {}
    for uid in paired_unit_ids:
        assignment[uid] = scheduler.assign(uid)
    return assignment
