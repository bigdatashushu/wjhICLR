"""M20 多 GPU 几何任务逻辑分配。

- 阶段串行由实验驱动器负责，本模块只分配 GPU；
- 多卡重建任务由 dp_partition 均匀切分；
- 卡故障：剔除该卡后继续；
- paired A/B 同 episode 固定同 gpu_rank（硬约束 18 配套）。
"""

from __future__ import annotations

from skill3d.schemas import GPUJob

DEFAULT_WORLD_SIZE = 8  # 8×4090


class GPUScheduler:
    """本机多进程 DP 调度器（逻辑分配，不直接管理进程）。"""

    def __init__(self, world_size: int = DEFAULT_WORLD_SIZE) -> None:
        if world_size < 1:
            raise ValueError("world_size 必须 ≥ 1")
        self.active_gpus: list[int] = list(range(world_size))
        self._rr = 0  # round-robin 指针
        self._paired_pin: dict[str, int] = {}  # paired_unit_id -> gpu_rank

    # ---------- 卡故障 ----------
    def mark_gpu_failed(self, gpu_rank: int) -> None:
        """剔除故障卡，剩余卡继续 DP（8→7）。"""
        if gpu_rank in self.active_gpus:
            self.active_gpus.remove(gpu_rank)
        if not self.active_gpus:
            raise RuntimeError("无可用 GPU")
        self._rr %= len(self.active_gpus)

    # ---------- 分配 ----------
    def assign(self, paired_unit_id: str, role: str = "reconstruct") -> GPUJob:
        """分配实验单元：同一 paired_unit_id 永远固定同一 gpu_rank。"""
        if paired_unit_id in self._paired_pin:
            rank = self._paired_pin[paired_unit_id]
            if rank in self.active_gpus:
                return GPUJob(gpu_rank=rank, role=role, paired_unit_id=paired_unit_id)  # type: ignore[arg-type]
            # 原卡故障：重绑并固定
        rank = self.active_gpus[self._rr % len(self.active_gpus)]
        self._rr += 1
        self._paired_pin[paired_unit_id] = rank
        return GPUJob(gpu_rank=rank, role=role, paired_unit_id=paired_unit_id)  # type: ignore[arg-type]

    # ---------- DP 切分 ----------
    def dp_partition(self, units: list[str]) -> list[list[str]]:
        """把实验单元均匀切到各活跃卡（DP×world_size）。"""
        buckets: list[list[str]] = [[] for _ in self.active_gpus]
        for i, u in enumerate(units):
            buckets[i % len(self.active_gpus)].append(u)
        return buckets
