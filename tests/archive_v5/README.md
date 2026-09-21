# tests/archive_v5 —— 已废止机制的 v5 回归测试只读归档（v6 §20）

本目录保存 harness3D **v5 及更早版本**针对后来被 v6 正式废止的机制写下的回归测试与
golden 夹具。它们的作用只有两个，与 `src/skill3d/legacy/retired/` 完全一致：

1. **失败复现**：能回查"当时为什么这么判定"（例如 conformal 校准池的隔离硬门、
   `vggt_sparse_ba` 的 L0 合同、官方 VGGSfM BA 的 OOM 证据链）；
2. **迁移审计**：旧口径/旧数字的出处（`scale_ci_rel` 的分数口径、v5 golden 的
   指标分母、v4 与 v5 golden 不可比的判据）。

## 纪律（违反即实现错误）

- **刻意不收集**：`pyproject.toml` 的 `[tool.pytest.ini_options] norecursedirs` 已显式
  排除 `tests/archive_v5`。`pytest`（含 CI）不得把这些文件当回归测试跑，**不是**
  "暂时跳过"，而是它们测试的机制在 v6 已经不存在。
- **不参与任何判定**：本目录不得被运行时代码 import，不得进统计、不得进主表、
  不得作为"能力已具备"的证据（HC34）。
- **不保证可运行**：这些测试 import 的模块已移入 `skill3d/legacy/retired/`，
  依赖可能已随 v6 删除；**保持原样**是它们的价值所在，不做适配、不做修补。
- 需要"用旧口径再算一遍"时，必须在新代码里显式重写并在 trace 里标 `legacy_*`，
  不得把本目录的测试接回主链。

## 逐项对照（v6 §20 表）

| 归档文件/目录 | 覆盖的废止机制 | v6 替代物 |
|---|---|---|
| `test_scale_units_v4.py` | `scale_units.py`：`scale_ci_rel` 分数口径的 CI 单位层 | `reconstruction/metric_fusion.py`（零样本度量深度跨帧融合）[待实验] |
| `test_metric_scale.py` | `metric_scale.py`：多锚点（相机高/门/桌/椅）+ log-scale 融合 | 同上 |
| `test_scale_assessment_v4.py` | `scale_assessment.py`：置信度派生 / 多锚点鲁棒融合 / 逐题型授权 | 同上 |
| `test_scale_v5.py` | `scale_assessment.py`（v5 口径）："有点估计但无区间 = 未标定" | 同上 |
| `test_scale_calibration_v4.py` | `scale_calibration.py`：conformal 校准器 + 标定集隔离硬门 | 同上（v6 不再需要标定数据） |
| `test_scale_calibration_v51.py` | v5.1 标定池数据契约（scannet / scannetpp / arkitscenes 互斥） | 同上（该契约随"放弃校准"整体撤销） |
| `test_scale_poc_and_report_v4.py` | `scale_poc.py` / `scale_report.py`：PoC L0–L3 阶梯 + 尺度报告 | `metric_fusion.write_per_frame_receipt`（32 个 s_k、median、MAD、离群帧） |
| `test_ba_route_v4.py` | `ba.py` / `legacy_vggsfm_ba/`：官方 VGGSfM tracker + PyCOLMAP BA | **无**；VGGT feed-forward 是唯一主线（24 GiB OOM 已否决） |
| `test_ba_colmap.py` | `colmap_baseline.py`：COLMAP 对照重建与 G-13/G-15 BA 残差回填 | **无**；`recon_method` 受控枚举只有 `vggt`（§5.2） |
| `test_sparse_ba_l0.py` | `sparse_ba/` + `schemas/sparse_ba.py`：`vggt_sparse_ba` L0 合同 | **无**；G5 永久 `not_available`（§20 已按止损纪律关闭） |
| `test_metric_task_authorization_v4.py` | 旧逐题型授权（`question_gate`）在整条在线链上的行为 | `reconstruction_gate/evidence_profile.py`（EvidenceProfile + MetricEvidenceGate）+ `scene_state.scope_scene_to_question` |
| `golden_v5/`（原 `tests/golden/v5/`） | v5 golden 夹具与 `make_golden.py`（v5 指标分母：无 G8、G5=None） | 无直接替代；v6 口径由 `reconstruction_gate/m4_main_gate.py` + `schemas/trace.py` 的 v6 golden/trace 重新生成 |
| `golden_archive_v4/`（原 `tests/golden/archive_v4/`） | v4 golden 夹具（分母含已退役 G8，`incomparable_with_v5=true`） | **无**；只读历史证据，永不与新 golden 混比（HC39） |

> 归档判定依据见 `src/skill3d/legacy/retired/__init__.py` 与《系统架构 v6》§20；
> 对应源码的只读归档在 `src/skill3d/legacy/retired/`。

## 临时查看（不是"跑回归"）

若要复现历史结论，请先明确这是**迁移审计**而不是回归：把需要的文件复制到临时目录、
按当时的口径自行准备依赖，读结论即可；**不得**把本目录加回 `norecursedirs` 的排除名单之外，
也不得让 CI 收集它们。

## 追加归档（v6 迁移第二轮）

- `test_question_gate.py`：测 v5 的 `QuestionGateDecision` / `question_gate`。
  v6 用 `scope_scene_to_question(...) -> (SceneState, QuestionScopeDecision)` 取代
  （§5.3 D4：`scene_route` × `question_tool_scope` 解耦，逐题只收窄不新增）。
- `test_v3_pipeline_contracts.py`：测 v5 的 `legacy_vggsfm_ba.route`（正方形 pad 的
  坐标映射）。v6 §20 废止 BA 路线后不存在 pad 形态，坐标层只剩纯缩放。
- `test_v5_schema_and_legacy.py`：测 v5 Schema 5.0 的字段门（`scale_ci`/`g8_*`/
  `SparseBAReceipt`/`route_from_quality`）。v6 的对应门禁在 `tests/unit/test_v6_schema.py`
  （Schema 6.0：世界系契约 + 度量融合字段 + G5 固定 None + legacy 字段拒绝）。

## 追加归档（v6 迁移第三轮：质量门写回 / 米制尺度门）

均取自 `tests/unit/test_v3_gap_fixes.py`（v5 正文逐字保留）：

- `test_quality_writeback_v5.py`（原 `test_quality_gate_enriches_g7_g9_from_m5`）：
  测 v5 的"M5 统计 → quality 增量补写 + 原子落盘"（方案 X 的 P2 路径）。
  v6 替代物：`quality_metrics.compute_quality(..., dynamic_masks=…, track_ious=…)`
  在 M4 一次算清 G7/G9（§10.2 诊断项只产告警，无跨模块回写，
  `overall_from_metrics` / `compute_g1_g11` 已删除）；回归入口
  `tests/unit/test_quality_metrics.py::test_diagnostics_are_nan_without_data_and_measured_with_data`
  与 `tests/unit/test_v3_gap_fixes.py::test_frozen_artifact_is_never_rewritten_by_quality_gate`。
- `test_scale_gate_v5.py`（原 `test_scale_artifact_hidden_when_scale_unusable`）：
  测 v5 的 `scale_known` 中间量 + `allowed_metric_tasks` 逐题型授权对 `scale` 可用性的
  影响。v6 替代物：`EvidenceProfile.metric_scale`（三值）由 `MetricEvidenceGate` 的
  6 项子条件决定（§13，D3），`scale` 只在"米制题 ∧ gate_passed ∧
  scope=metric_enabled"时进 `available_artifacts`（§5.3，D4）；回归入口
  `tests/unit/test_v3_gap_fixes.py::test_scale_artifact_requires_metric_gate_and_metric_question`
  与 `tests/unit/test_v6_schema.py`（scope 收窄 + docs 裁剪）。
