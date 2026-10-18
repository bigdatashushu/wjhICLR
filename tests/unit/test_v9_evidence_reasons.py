"""两维证据原因码与 M5 诊断隔离；未运行、生产失败、空检出保留原始事实。"""

from __future__ import annotations

import numpy as np
import pytest

from skill3d.reconstruction_gate.evidence_profile import (
    M5EvidenceSummary,
    build_evidence_profile,
)
from skill3d.schemas import ConfidenceMap, QualityMetrics, ReconstructionArtifact
from skill3d.schemas.evidence import UNAVAILABLE_REASON_CODES

_nan = float("nan")


def _artifact(*, quality_status="computed", main_passed=True, fusion="success",
              metric_scale=1.5):
    return ReconstructionArtifact(
        artifact_id="a", artifact_version="v1", scene_name="s",
        recon_method="vggt", c2w_list="", intrinsics="", depth_maps="",
        point_map="", point_conf="", depth_conf="", track_list=None,
        metric_scale=metric_scale, scale_self_consistency=0.05,
        scale_fusion_status=fusion, metric_model="moge2",
        world_up=[0.0, 1.0, 0.0], handedness="right",
        world_frame_status="available",
        quality_status=quality_status,
        # schema 要求：quality 有值 ⟺ quality_status == "computed"
        quality=(QualityMetrics(
            warp_inlier_ratio=0.9, warp_photometric_inlier_ratio=0.9,
            cloud_overlap_ratio=0.9, main_gate_passed=main_passed,
            g1_blur_ok=_nan, g2_brightness=_nan, g3_motion_blur=_nan,
            g4_frame_count=32, g6_depth_var_coeff=_nan, g7_dynamic_ratio=_nan,
            g9_tracker_consistency=_nan, g10_baseline_quality=_nan,
            overall_quality=0.9)
                 if quality_status == "computed" else None),
        confidence=ConfidenceMap(per_point_confidence="",
                                 coverage_count_per_frame=""))


# --------------------------------------------- 未运行 vs 运行失败 vs 空检出 ----

def test_never_run_is_distinguishable_from_zero_detection():
    signatures = []
    facts = []
    for summary in (M5EvidenceSummary(),
                    M5EvidenceSummary(m5_ran=True, detection_fault=True),
                    M5EvidenceSummary(m5_ran=True, n_objects=0)):
        profile = build_evidence_profile(
            artifact=_artifact(), scene_route="full_3d", question_type="object_counting",
            gate=None, m5=summary)
        signatures.append(profile.as_signature())
        d = profile.producer_subvalues["candidate_diagnostics"]
        facts.append((d["m5_ran"], d["detection_fault"], d["n_objects"]))
        assert "object_detection" not in profile.state_reasons
    assert signatures[0] == signatures[1] == signatures[2]
    assert len(set(facts)) == 3


def test_metric_scale_not_run_and_failed_are_distinguished():
    never = build_evidence_profile(
        artifact=_artifact(fusion="not_run", metric_scale=None),
        scene_route="full_3d", question_type="object_counting", gate=None)
    assert never.state_reasons.get("metric_scale") == "not_run"
    failed = build_evidence_profile(
        artifact=_artifact(fusion="failed", metric_scale=None),
        scene_route="full_3d", question_type="object_counting", gate=None)
    assert failed.state_reasons.get("metric_scale") == "producer_failed"


def test_uncomputed_quality_registers_geometry_not_run():
    profile = build_evidence_profile(
        artifact=_artifact(quality_status="not_computed", main_passed=False),
        scene_route="fallback_2d_only", question_type="object_counting", gate=None)
    assert profile.geometry_3d == "unavailable"
    assert profile.state_reasons.get("geometry_3d") == "not_run"


# ------------------- 运行成功但质量门未过（第五值，用户 2026-09-27 裁定扩展）----

def test_main_gate_failure_is_registered_as_quality_gate_not_passed():
    """质量真算了、主门未过 → 必须与"没跑"和"跑失败"分开记（§6.1 第五值）。

    决策依据：§6.4 的四类没有这一格，此前只能留空 —— 审计读到"没记原因"，
    而事实是"跑了但不达标"。用户 2026-09-27 裁定扩展词表。
    """
    profile = build_evidence_profile(
        artifact=_artifact(main_passed=False),
        scene_route="fallback_2d_only", question_type="object_counting", gate=None)
    assert profile.geometry_3d == "unavailable"
    assert profile.state_reasons.get("geometry_3d") == "quality_gate_not_passed"
    # 与"质量未计算"必须是不同的码（不得塌成 not_run）
    uncomputed = build_evidence_profile(
        artifact=_artifact(quality_status="not_computed", main_passed=False),
        scene_route="fallback_2d_only", question_type="object_counting", gate=None)
    assert (uncomputed.state_reasons["geometry_3d"]
            != profile.state_reasons["geometry_3d"])


def test_fusion_success_but_gate_not_passed_registers_the_fifth_code():
    """融合成功（生产者正常）但门未全过 → `quality_gate_not_passed`，不是 producer_failed。"""
    from skill3d.schemas.evidence import GATE_SUBCONDITIONS, GATE_VERSION
    from skill3d.schemas import MetricEvidenceGateResult

    gate = MetricEvidenceGateResult(
        gate_passed=False, gate_version=GATE_VERSION,
        sub_results={name: name != "scale_self_consistency_ok"
                     for name in GATE_SUBCONDITIONS},
        values={"valid_frame_ratio": 1.0}, missing_subconditions=["scale_self_consistency_ok"])
    # 自洽值本身确实超阈值。
    art = _artifact(fusion="success").model_copy(
        update={"scale_self_consistency": 0.9})
    profile = build_evidence_profile(
        artifact=art, scene_route="full_3d",
        question_type="object_abs_distance", gate=gate)
    assert profile.metric_scale == "unavailable"
    assert profile.state_reasons.get("metric_scale") == "quality_gate_not_passed"


def test_world_frame_failure_only_changes_local_diagnostics():
    art = _artifact()
    art = art.model_copy(update={"world_frame_status": "degraded", "world_up": None})
    profile = build_evidence_profile(artifact=art, scene_route="full_3d",
                                     question_type="object_counting", gate=None)
    assert profile.geometry_3d == "available"
    assert "world_frame" not in profile.state_reasons
    assert profile.producer_subvalues["world_frame_diagnostics"]["world_up"] is None


@pytest.mark.parametrize("summary", [
    M5EvidenceSummary(m5_ran=True, n_objects=2),
    M5EvidenceSummary(m5_ran=True, n_tracks=5, track_fragmentation_ratio=.99),
    M5EvidenceSummary(m5_ran=True, n_objects=0),
])
def test_sparse_detection_and_track_stats_do_not_change_public_reasons(summary):
    profile = build_evidence_profile(
        artifact=_artifact(), scene_route="full_3d", question_type="object_counting",
        gate=None, m5=summary)
    assert profile.geometry_3d == "available"
    assert set(profile.state_reasons) <= {"geometry_3d", "metric_scale"}
    assert profile.producer_subvalues["candidate_diagnostics"]["n_objects"] == summary.n_objects


def test_unknown_world_frame_status_stays_unregistered():
    """`unavailable` 分不清"从未估计"与"估计失败" → 留空，不猜一个码填上。"""
    art = _artifact()
    art = art.model_copy(update={"world_frame_status": "unavailable", "world_up": None,
                                 "handedness": None})
    profile = build_evidence_profile(artifact=art, scene_route="full_3d",
                                     question_type="object_counting", gate=None)
    assert profile.geometry_3d == "available"
    assert "world_frame" not in profile.state_reasons


# --------------------------------------------- 原因码只在词表内 ----

def test_every_registered_reason_is_in_the_documented_vocabulary():
    """§6.4 的词表由 validator 强制；生产者不得写入词表外的自由字符串。"""
    for m5 in (M5EvidenceSummary(),
               M5EvidenceSummary(m5_ran=True, detection_fault=True),
               M5EvidenceSummary(m5_ran=True, n_objects=3)):
        profile = build_evidence_profile(
            artifact=_artifact(), scene_route="full_3d",
            question_type="object_counting", gate=None, m5=m5)
        for cap, reason in profile.state_reasons.items():
            assert reason in UNAVAILABLE_REASON_CODES, (cap, reason)


def test_bogus_reason_is_rejected_by_the_profile():
    from skill3d.schemas.evidence import EvidenceProfile

    base = dict(geometry_3d="unavailable", metric_scale="unavailable")
    with pytest.raises(Exception):
        EvidenceProfile(**base, state_reasons={"geometry_3d": "because_i_said_so"})


# --------------------------------------------- 级联撤销 → invalidated ----

def test_cascade_revocation_marks_invalidated():
    """§6.4：`invalidated` 必须与"没跑""跑失败"区分开。"""
    from skill3d.online.recovery import downgrade_profile
    from skill3d.schemas.evidence import EvidenceProfile

    profile = EvidenceProfile(geometry_3d="available", metric_scale="available")
    downgraded, changed = downgrade_profile(profile, "geometry_3d")
    assert changed, "撤销应产生能力变化"
    assert all(downgraded.state_reasons[c] == "invalidated" for c in changed), \
        downgraded.state_reasons
    for cap, reason in downgraded.state_reasons.items():
        assert reason in UNAVAILABLE_REASON_CODES, (cap, reason)
