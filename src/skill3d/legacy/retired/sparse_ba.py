"""§4.5 SparseBAReceipt：`vggt_sparse_ba` 限定 PoC 的独立回执。

**硬约束 36/37 的可审计落点**：

- `recon_method` 只能写 `vggt_sparse_ba`；官方 `VGGSfM tracker + PyCOLMAP BA`
  的失败回执与轻量路径**不共享**名称/receipt/artifact；
- 未通过 §10.1 全部门槛即 `rejected`，生产接线关闭，正式主线继续用 `vggt`；
- 失败时 `initial_cost/final_cost/g5_*` 必须为 None，不得用代理值填充。
"""

from typing import Literal, Optional

from . import Spec


class SparseBAReceipt(Spec):
    """一次 `vggt_sparse_ba` 尝试的完整回执（成功/失败/止损都要写）。"""

    status: Literal["not_run", "passed", "failed", "rejected"]
    frontend: Literal["superpoint_lightglue", "aliked_lightglue"]
    pair_graph_hash: str
    n_pairs: int
    n_matches: int
    n_inliers: int
    n_tracks: int
    peak_gpu_gib: float
    initial_cost: Optional[float] = None
    final_cost: Optional[float] = None
    # 未通过/未运行时必须写明原因（例如 oom / crash / excessive_skip）
    skip_reason: Optional[str] = None
    evidence_refs: list[str] = []

    # ---- §10.1 L2 报告字段（预注册门槛；未跑到该级一律留 None）----
    oom_rate: Optional[float] = None
    skip_rate: Optional[float] = None
    success_rate: Optional[float] = None
    wallclock_s: Optional[float] = None
    # 止损固定理由码：rejected_on_24g_oom / rejected_on_l1_gate / rejected_on_l2_gate
    rejected_reason: Optional[str] = None
