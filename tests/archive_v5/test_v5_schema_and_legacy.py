"""v5 Schema 5.0 + legacy 隔离负向测试（§14.1-2 / HC37/38/39）。

本文件是 v5 完成定义（§14.3）的硬门禁测试：以下情形**必须** fail-closed，
且不得通过删除测试、放宽断言或 `--allow-mock` 获得假绿：

1. G5 三态：`not_available/failed` ⇒ 字段必须 None；`computed` ⇒ 必须有限；
   非 computed 却给有限值（代理值冒充重投影残差）→ hard fail；
2. G8 已退役字段（`g8_bbox_coverage_min` / `CoverageMap` / `coverage_ok`）→ 拒绝；
3. 旧尺度字段（`scale_ci` / `relative_ci`）→ 拒绝（不再"读进来降级"）；
4. 版本不符（缺 `schema_version`、旧 `quality_metric_version`）→ hard fail，
   且信息里给出"重跑 v5 pipeline"的可执行建议；
5. `LegacyArtifact` 不得进入运行时/统计（`eligible_for_*` 恒 False）；
6. `coverage_gate_status` 只能是 `not_defined`（不得写 passed/True）。
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from skill3d.legacy.readers import (
    CURRENT_QUALITY_METRIC_VERSION,
    CURRENT_SCHEMA_VERSION,
    audit_tree,
    detect_deprecated_fields,
    legacy_report,
    load_artifact_v5,
    read_legacy_artifact,
)
from skill3d.reconstruction_gate import quality_metrics as qm
from skill3d.reconstruction_gate.scene_state import route_from_quality
from skill3d.schemas import (
    LegacyArtifact,
    LegacyArtifactError,
    QualityMetrics,
    ReconstructionArtifact,
    SceneState,
    SparseBAReceipt,
)


def _art(**kw) -> ReconstructionArtifact:
    base = dict(
        artifact_id="a", artifact_version="v", scene_name="s", recon_method="vggt",
        c2w_list="", intrinsics="", depth_maps="", point_map="", point_conf="",
        track_list=None, metric_scale=None, scale_known=False,
        confidence={"per_point_confidence": "", "coverage_count_per_frame": ""})
    base.update(kw)
    return ReconstructionArtifact(**base)


def _q(**kw) -> QualityMetrics:
    base = dict(g1_blur_ok=1.0, g2_brightness=1.0, g3_motion_blur=0.0,
                g4_frame_count=32, g6_depth_var_coeff=0.1, g7_dynamic_ratio=0.0,
                g9_tracker_consistency=0.9, g10_baseline_quality=0.5,
                g11_scale_ci=0.05, overall_quality=0.9)
    base.update(kw)
    return QualityMetrics(**base)


# ------------------------------------------------------- 1. 版本字段与默认态 ----

def test_artifact_carries_v5_version_fields():
    art = _art()
    assert art.schema_version == CURRENT_SCHEMA_VERSION == "5.0"
    assert art.quality_metric_version == CURRENT_QUALITY_METRIC_VERSION
    assert art.reprojection_status == "not_available"
    assert art.g5_reproj_err_median is None and art.g5_reproj_err_p95 is None
    assert art.sparse_ba_receipt_ref is None


def test_recon_method_vocabulary_is_v5():
    """HC35/36：正式主线 `vggt`；BA 候选只能叫 `vggt_sparse_ba`（旧 `vggt_ba` 作废）。"""
    assert _art(recon_method="vggt").recon_method == "vggt"
    assert _art(recon_method="vggt_sparse_ba").recon_method == "vggt_sparse_ba"
    with pytest.raises(Exception):
        _art(recon_method="vggt_ba")


# --------------------------------------------------------------- 2. G5 三态 ----

def test_g5_not_available_forbids_proxy_values():
    """HC37：无真 BA 时给 G5 有限值 = 代理值冒充 → hard fail。"""
    with pytest.raises(Exception) as ei:
        _art(reprojection_status="not_available",
             g5_reproj_err_median=0.5, g5_reproj_err_p95=1.0)
    assert "代理值" in str(ei.value) or "G5" in str(ei.value)


def test_g5_failed_also_forbids_values():
    with pytest.raises(Exception):
        _art(reprojection_status="failed", g5_reproj_err_median=0.1,
             g5_reproj_err_p95=0.2)


def test_g5_computed_requires_both_scalars():
    with pytest.raises(Exception) as ei:
        _art(reprojection_status="computed", g5_reproj_err_median=0.5)
    assert "g5_reproj_err_p95" in str(ei.value)
    ok = _art(reprojection_status="computed", g5_reproj_err_median=0.5,
              g5_reproj_err_p95=1.0)
    assert ok.reprojection_status == "computed"


def test_g5_none_roundtrip_is_stable():
    """None 与 NaN 等价；序列化往返后仍是 None（不得变成 0 分）。"""
    art = _art()
    again = ReconstructionArtifact.model_validate_json(art.model_dump_json())
    assert again.g5_reproj_err_median is None and again.g5_reproj_err_p95 is None


# --------------------------------------------------- 3. G8/旧尺度字段拒绝 ----

@pytest.mark.parametrize("field", ["g8_bbox_coverage_min", "bbox_coverage_ratio"])
def test_g8_fields_rejected(field):
    with pytest.raises(Exception):
        _q(**{field: 0.9})


@pytest.mark.parametrize("field", ["scale_ci", "relative_ci"])
def test_legacy_scale_fields_rejected_on_artifact(field):
    with pytest.raises(Exception) as ei:
        _art(**{field: 0.02})
    assert "legacy" in str(ei.value)


def test_coverage_map_is_gone():
    """v5 Schema 不得再声明 CoverageMap（§4.1 不变量）。"""
    import skill3d.schemas as schemas

    assert not hasattr(schemas, "CoverageMap")
    assert not hasattr(schemas.reconstruction, "CoverageMap")


def test_coverage_gate_status_only_not_defined():
    sc = SceneState(artifact_ref="a", route="full_3d", frame="world", scale_known=False,
                    objects=[], summary="s")
    assert sc.coverage_gate_status == "not_defined"
    with pytest.raises(Exception):
        SceneState(artifact_ref="a", route="full_3d", frame="world", scale_known=False,
                   objects=[], summary="s", coverage_gate_status="passed")


def test_g8_absent_from_aggregation_not_filled():
    """G8 不在活动集合里，也不得被"补默认高分"顶替（附录 A 聚合不变量）。"""
    assert "g8_bbox_coverage_min" in qm.RETIRED_METRICS
    assert "g8_bbox_coverage_min" not in qm.ALWAYS_ACTIVE_METRICS
    assert "g5_reproj_err_median" in qm.CONDITIONAL_METRICS
    q_nog5 = _q()
    q_with_g5 = _q(g5_reproj_err_median=0.1, g5_reproj_err_p95=0.2)
    # G5 缺失时不得因为"少了两项 1.0"而虚高：分母真的少了两项
    assert qm.overall_from_metrics(q_nog5) == qm.overall_from_metrics(q_with_g5) == 1.0
    q_bad_g5 = _q(g5_reproj_err_median=99.0, g5_reproj_err_p95=99.0)
    assert qm.overall_from_metrics(q_bad_g5) < 1.0   # computed 时确实参与


# ------------------------------------------- 4. legacy readers / 版本 hard fail ----

def test_detect_deprecated_fields_recurses():
    raw = {"schema_version": "5.0", "quality": {"g8_bbox_coverage_min": 0.9},
           "nested": [{"relative_ci": 0.1}]}
    found = detect_deprecated_fields(raw)
    assert {"g8_bbox_coverage_min", "relative_ci"} <= found


def test_load_artifact_v5_rejects_legacy_json(tmp_path):
    """旧 artifact（v4 真实样本形态）→ hard fail 且给出重跑建议。"""
    legacy = {
        "artifact_id": "vggt-41069043", "artifact_version": "x", "scene_name": "41069043",
        "recon_method": "vggt", "metric_scale": 2.386, "scale_known": True,
        "scale_ci": 3.41, "scale_confidence": "low",
        "quality": {"g1_blur_ok": None, "g8_bbox_coverage_min": 0.9},
    }
    p = tmp_path / "old_artifact.json"
    p.write_text(json.dumps(legacy), encoding="utf-8")
    with pytest.raises(LegacyArtifactError) as ei:
        load_artifact_v5(p)
    msg = str(ei.value)
    assert "重跑 v5 pipeline" in msg and "scale_ci" in msg


def test_load_artifact_v5_rejects_wrong_quality_metric_version(tmp_path):
    art = _art(quality_status="computed", quality=_q())
    payload = json.loads(art.model_dump_json())
    payload["quality_metric_version"] = "v4-legacy"
    p = tmp_path / "art.json"
    p.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(LegacyArtifactError) as ei:
        load_artifact_v5(p)
    assert "quality_metric_version" in str(ei.value)


def test_load_artifact_v5_accepts_current_schema(tmp_path):
    art = _art(quality_status="computed", quality=_q())
    p = tmp_path / "art.json"
    p.write_text(art.model_dump_json(), encoding="utf-8")
    back = load_artifact_v5(p)
    assert back.schema_version == "5.0" and back.artifact_id == art.artifact_id


def test_legacy_artifact_is_never_runtime_or_statistics_eligible(tmp_path):
    p = tmp_path / "old.json"
    p.write_text(json.dumps({"metric_scale": 1.0, "scale_ci": 0.1}), encoding="utf-8")
    leg = read_legacy_artifact(p)
    assert isinstance(leg, LegacyArtifact)
    assert leg.eligible_for_runtime is False
    assert leg.eligible_for_statistics is False
    # Literal[False]：构造 True 直接被 Schema 拒绝
    with pytest.raises(Exception):
        LegacyArtifact(source_path=str(p), raw_fields={}, deprecated_fields=set(),
                       warnings=[], eligible_for_runtime=True)


def test_audit_tree_and_report(tmp_path):
    (tmp_path / "a_artifact.json").write_text(json.dumps({"scale_ci": 1.0}), encoding="utf-8")
    v5 = _art()
    (tmp_path / "b_artifact.json").write_text(v5.model_dump_json(), encoding="utf-8")
    items = audit_tree(tmp_path)
    assert len(items) == 1 and items[0].source_path.endswith("a_artifact.json")
    rep = legacy_report(items)
    assert rep["incomparable_with_v5"] is True and rep["n_legacy_artifacts"] == 1


# ------------------------------------------------- 5. SparseBAReceipt（§4.5）----

def test_sparse_ba_receipt_shape_and_stop_loss():
    r = SparseBAReceipt(status="rejected", frontend="superpoint_lightglue",
                        pair_graph_hash="h", n_pairs=0, n_matches=0, n_inliers=0,
                        n_tracks=0, peak_gpu_gib=20.8, skip_reason="oom",
                        rejected_reason="rejected_on_24g_oom")
    assert r.initial_cost is None and r.final_cost is None
    assert r.rejected_reason == "rejected_on_24g_oom"


# --------------------------------------------- 6. route 仍是 fail-closed（HC22）----

def test_route_never_full_3d_without_computed_quality():
    assert route_from_quality(None) == "fallback_2d_only"
    assert route_from_quality(_q(), quality_status="not_computed") == "fallback_2d_only"
    assert route_from_quality(_q(), quality_status="failed") == "fallback_2d_only"
    assert route_from_quality(_q(overall_quality=float("nan")),
                              quality_status="computed") == "fallback_2d_only"
    assert route_from_quality(_q(g4_frame_count=0)) == "unanswerable"
    assert route_from_quality(_q(), quality_status="computed") == "full_3d"


def test_quality_metrics_nan_roundtrip_keeps_semantics():
    """非 G5 指标仍以 NaN 表达"算不出"（ser_json_inf_nan="constants"），
    G5 用 None；两者不可互换。"""
    q = _q(g1_blur_ok=float("nan"))
    back = QualityMetrics.model_validate_json(q.model_dump_json())
    assert np.isnan(back.g1_blur_ok)
    assert back.g5_reproj_err_median is None
