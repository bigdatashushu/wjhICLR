"""已废止机制的只读归档（v6 §20）。

本目录保存 harness3D **v5 及更早版本**中被 v6 正式废止的机制源码，用途只有两个：

1. **失败复现**：能回查"当时为什么会得出这个结论"（例如官方 VGGSfM BA 在 24 GiB
   上 OOM、`vggt_sparse_ba` L1 三角化 0 点）；
2. **迁移审计**：旧数字/旧口径的出处。

纪律（与 `skill3d.legacy.readers` 一致，违反即实现错误）：

- 本目录的模块**不参与**在线链、不参与路由、不参与统计、不产生任何可准入数字；
- 任何运行时代码**不得** import 本目录（在线链尤其严禁）；
- 这里的代码不保证可运行（依赖可能已随 v6 删除），**保持原样**是它的价值所在；
- 需要"用旧口径算一遍"时，必须在新代码里显式重写并在 trace 里标 `legacy_*`，
  不得直接把本目录的模块接回主链。

## 逐项废止依据（v6 §20 表）

| 归档模块 | 废止原因 | v6 替代物 |
|---|---|---|
| `scale_calibration.py` | conformal 校准池需要非重叠 GT 位姿标定数据（数据 blocker） | `reconstruction/metric_fusion.py`（零样本度量深度跨帧融合）[待实验] |
| `scale_assessment.py` / `scale_units.py` / `metric_scale.py` / `scale_poc.py` | 多锚点 + log-scale 融合路线整体替换 | 同上 |
| `scale_report.py` | 校准池诊断报告随尺度路线撤销 | `metric_fusion.write_per_frame_receipt`（32 个 s_k、median、MAD、离群帧） |
| `ba.py` / `legacy_vggsfm_ba/` | 官方 VGGSfM BA 已被 24 GiB OOM 否决，生产 hard-disable | 无；VGGT feed-forward 是唯一主线 |
| `sparse_ba/` / `sparse_ba.py` | `vggt_sparse_ba` L1 实测 `rejected_on_l1_gate`，按止损纪律关闭 | 无；G5 永久 `not_available` |
| `colmap_baseline.py` / `dust32_mast3r_fallback.py` | §5.2 把 `recon_method` 收窄为 `Literal["vggt"]` 后，COLMAP / DUSt3R-MASt3R 对照基线随之废止 | 无；VGGT feed-forward 是唯一重建路线 |
"""
