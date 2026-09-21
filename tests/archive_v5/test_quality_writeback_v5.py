"""v5 `quality_gate` 的 M5 统计回写（P2）单测 —— 只读归档。

原位置：`tests/unit/test_v3_gap_fixes.py::test_quality_gate_enriches_g7_g9_from_m5`
（v5 期间正文，逐字保留，**不做适配、不做修补**）。

**v6 废止原因**（《系统架构 v6》§10.2 / §20）：v5 把 G7/G9 当作"可由 M5 统计增量补写"
的质量项 —— 落盘的 quality 缺 G7/G9，P2 拿到 M5 统计后补写并原子落盘。v6 把质量收成
**M4 一次算清的单一事实源**：G7/G9 是诊断项，`compute_quality(..., dynamic_masks=…,
track_ious=…)` 在计算当次就实算（没数据记 NaN），不再有跨模块回写，也不再有
`overall_from_metrics` / `compute_g1_g11` 这类 v5 聚合助手。

**v6 替代物（回归入口）**：
- `tests/unit/test_quality_metrics.py::test_diagnostics_are_nan_without_data_and_measured_with_data`
  （G7/G9 输入即测、无数据记 NaN）；
- `tests/unit/test_v3_gap_fixes.py::test_frozen_artifact_is_never_rewritten_by_quality_gate`
  （frozen artifact 只读：已算过的质量直接复用，不再二次写回）。

配套归档：`tests/archive_v5/test_scale_gate_v5.py`（同类原因：v5 的 `scale_known` +
`allowed_metric_tasks` 授权）。详见 `tests/archive_v5/README.md`。
"""

from __future__ import annotations

import numpy as np
import pytest

from skill3d.reconstruction_gate import quality_metrics as qm
from skill3d.reconstruction_gate.scene_state import quality_gate
from skill3d.schemas import ConfidenceMap, ReconstructionArtifact, SceneState  # noqa: F401


def _artifact(tmp_path) -> ReconstructionArtifact:
    """route=full_3d 的 artifact；位姿/内参数组不落位（reproject 会 fail-closed）。"""
    c2w = np.tile(np.eye(4), (32, 1, 1))
    k = np.tile(np.eye(3), (32, 1, 1))
    p = tmp_path / "a.npy"
    np.save(p, c2w)
    kn = tmp_path / "k.npy"
    np.save(kn, k)
    return ReconstructionArtifact(
        artifact_id="art-gap", artifact_version="v1", scene_name="scene-x",
        recon_method="vggt", frame_ids=list(range(32)),
        source_frame_indices=list(range(32)),
        timestamps=[float(i) for i in range(32)], frame_set_hash="h32",
        c2w_list=str(p), intrinsics=str(kn), depth_maps="", point_map="",
        point_conf="", track_list=None, metric_scale=None, scale_known=False,
        confidence=ConfidenceMap(per_point_confidence="", coverage_count_per_frame=""))


def _quality(**over) -> qm.QualityMetrics:
    base = dict(g1_blur_ok=1.0, g2_brightness=1.0, g3_motion_blur=1.0,
                g4_frame_count=32, g5_reproj_err_median=float("nan"),
                g5_reproj_err_p95=float("nan"), g6_depth_var_coeff=0.5,
                g7_dynamic_ratio=float("nan"), g9_tracker_consistency=float("nan"), g10_baseline_quality=0.3,
                g11_scale_ci=float("nan"), overall_quality=0.9)
    base.update(over)
    return qm.QualityMetrics(**base)


def test_quality_gate_enriches_g7_g9_from_m5(tmp_path):
    """方案 X 落盘的 quality 缺 G7/G9 → P2 拿到 M5 统计后补写并原子落盘。

    （G8 已按附录 A 删除，不再参与补写。）
    """
    art = _artifact(tmp_path).model_copy(
        update={"quality_status": "computed", "quality": _quality()})
    path = tmp_path / "art.json"
    path.write_text(art.model_dump_json(indent=2), encoding="utf-8")

    masks = np.zeros((32, 8, 8), dtype=bool)
    masks[:8] = True                                        # 动态占比 1/4
    scene = quality_gate(art, dynamic_masks=masks,
                         track_ious=[0.9, 0.8], artifact_path=str(path))

    q = scene.quality
    assert q.g7_dynamic_ratio == pytest.approx(0.25)
    assert q.g9_tracker_consistency == pytest.approx(0.85)
    # 补写后的 overall 必须重算（不再是"只有 G1-G4/G6/G10"的那份）
    assert q.overall_quality == pytest.approx(qm.overall_from_metrics(q))
    # 原子写回：落盘文件里也能读到补写后的值（单一事实源，不是内存幻觉）
    import json

    on_disk = json.loads(path.read_text(encoding="utf-8"))
    assert on_disk["quality"]["g7_dynamic_ratio"] == pytest.approx(0.25)
