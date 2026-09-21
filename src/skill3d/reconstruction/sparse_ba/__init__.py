"""`reconstruction/sparse_ba`：`vggt_sparse_ba` 限定 PoC（§3 M3 / §10.1 / HC36）。

**这是唯一允许的新 BA 候选，且只做一次有止损条件的 PoC。**

- 前端：`SuperPoint 或 ALIKED + LightGlue + RANSAC + 多帧 track 合并`；
  后端：PyCOLMAP 三角化 + `bundle_adjustment`；
- **不得称为"官方 VGGSfM BA"**（那是 HC35 已否决的路线，代码在
  `reconstruction/legacy_vggsfm_ba/`）；
- 默认关闭：`sparse_ba_enabled()` 读环境变量/配置，默认 `False`；
- 任一 §10.1 硬门失败 → `rejected`，关闭生产接线，正式主线继续用 `vggt`；
- BA **不能恢复度量尺度**（gauge 自由度），尺度仍由多锚点 + 独立 conformal 校准决定。

模块分工（§2.1）：

| 模块 | 职责 | 本版状态 |
|---|---|---|
| `pair_graph.py` | 预注册 pair graph（不读问题/答案，覆盖全部 32 帧，哈希稳定） | 已实现（纯算术，可测） |
| `tracks.py` | 多帧 track 合并 + 不变量（同帧单观测/无循环/有限坐标/最小长度） | 已实现（纯逻辑，可测） |
| `receipts.py` | `SparseBAReceipt` 落盘/读取 + L0/L1/L2 门槛判定 | 已实现 |
| `features.py` / `matching.py` | SuperPoint/ALIKED + LightGlue 逐 pair 流式匹配 | `[待实码核验]` GPU 侧，L1 阶段实现 |
| `pycolmap_backend.py` | 三角化 + BA + 残差读回 | `[待实码核验]` L1 阶段实现 |
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Optional

from skill3d.schemas.sparse_ba import SparseBAReceipt  # noqa: F401  (对外统一出口)

# ---- 预注册门槛（全部 TODO_CALIBRATE；必须在看结果前冻结，禁止事后放宽）----
MIN_TRACK_LENGTH: int = 3            # TODO_CALIBRATE: track 最小观测数
MAX_PEAK_GPU_GIB: float = 20.0       # TODO_CALIBRATE: L1 单卡峰值上限
MIN_HEADROOM_GIB: float = 2.0        # TODO_CALIBRATE: 运行余量
MAX_SKIP_RATE: float = 0.30          # TODO_CALIBRATE: L2 预注册 skip_rate 上限
L2_MIN_EPISODES: int = 20

ENV_FLAG = "SKILL3D_SPARSE_BA"


def sparse_ba_enabled() -> bool:
    """生产开关：默认 **False**（HC36：未过 PoC 前不得接线）。"""
    return str(os.environ.get(ENV_FLAG, "0")).strip().lower() in {"1", "true", "yes", "on"}


@dataclass
class BAOutcome:
    """feed-forward / sparse BA 的统一结果载体（生产侧只看到这一层）。"""

    applied: bool = False
    recon_method: str = "vggt"
    g5_reproj_err_median: Optional[float] = None
    g5_reproj_err_p95: Optional[float] = None
    reprojection_status: str = "not_available"
    receipt_ref: Optional[str] = None
    reason: str = ""
    notes: list[str] = field(default_factory=list)

    @classmethod
    def feed_forward(cls, *, reason: str = "") -> "BAOutcome":
        """正式主线（无真 BA）：G5 恒 None + `not_available`（HC37）。"""
        return cls(applied=False, recon_method="vggt",
                   reprojection_status="not_available", reason=reason)

    def summary(self) -> str:
        if self.applied:
            return (f"sparse BA 生效 method={self.recon_method} "
                    f"g5_median={self.g5_reproj_err_median}")
        return (f"BA 未生效（{self.reason or '未启用'}）→ feed-forward；"
                "G5 记 None（HC37：禁止代理值；官方 VGGSfM BA 已按 HC35 退出）")


__all__ = [
    "BAOutcome",
    "MAX_PEAK_GPU_GIB",
    "MAX_SKIP_RATE",
    "MIN_HEADROOM_GIB",
    "MIN_TRACK_LENGTH",
    "L2_MIN_EPISODES",
    "SparseBAReceipt",
    "sparse_ba_enabled",
]
