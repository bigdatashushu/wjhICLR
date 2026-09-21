"""v4 HC32 标定隔离 + conformal 校准器单测。

对应《系统架构4.md》§0.2 HC32 / §10.2「标定数据隔离」：
- ARKitScenes 标定集必须按原始 scene/video ID 排除 VSI-Bench 的 150 个 scene，
  交集非空即 **hard fail**，拒绝生成 `scale_calibration_id`；
- 在线只加载**冻结**校准器，绝不读取 GT 位姿/深度/逐场景尺度；
- 校准器口径版本不符 / 载入失败 → 一律不可用（fail-closed）。
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from skill3d.reconstruction.scale_calibration import (
    CALIBRATOR_SCHEMA_VERSION,
    CalibrationSplitError,
    CalibrationUnavailable,
    ConformalCalibrator,
    build_scene_id_audit,
    calibrator_available,
    conformal_quantile,
    fit_conformal_calibrator,
    load_calibrator,
    save_calibrator,
    vsibench_arkitscenes_scene_ids,
    write_audit_manifest,
)
from skill3d.reconstruction.scale_units import CI_UNIT_VERSION

META = "data/vsi_bench_meta/test.jsonl"


# ------------------------------------------------------------ scene ID 层 ----

def test_vsibench_arkitscenes_scene_ids_are_150():
    """排除清单来自 VSI-Bench 官方 meta（可审计的原始 scene ID，非匿名文件列表）。"""
    if not Path(META).is_file():
        pytest.skip("VSI-Bench meta 未就位")
    ids = vsibench_arkitscenes_scene_ids(META)
    assert len(ids) == 150
    assert all(str(i).isdigit() for i in ids[:5])


def test_scene_id_audit_intersection_is_hard_fail():
    """HC32：交集非空即 hard fail（不 warnings、不静默过滤）。"""
    excluded = ["41069025", "41069043", "41069048"]
    with pytest.raises(CalibrationSplitError):
        build_scene_id_audit(["cal_1", "41069025"], excluded_scene_ids=excluded)
    # 不相交则通过，并给出两个可留档的哈希
    audit = build_scene_id_audit(["cal_1", "cal_2"], excluded_scene_ids=excluded)
    assert not audit.intersection()
    assert len(audit.calibration_split_hash) == 64
    assert len(audit.excluded_vsibench_scene_hash) == 64


def test_scene_id_audit_requires_exclusion_source():
    """不给排除清单也不给 meta → 无法证明隔离 → 拒绝标定（不得默认放行）。"""
    with pytest.raises(CalibrationSplitError):
        build_scene_id_audit(["cal_1"])


def test_conformal_set_must_be_disjoint_from_calibration_and_eval():
    """HC32：标定集、conformal 校准集、评测集**两两**不相交。

    - conformal ∩ excluded 非空 → hard fail；
    - conformal ∩ calibration 非空 → hard fail（否则覆盖自评，无独立留出意义）。
    """
    excluded = ["41069025", "41069043"]
    with pytest.raises(CalibrationSplitError):
        build_scene_id_audit(["cal_1"], excluded_scene_ids=excluded,
                             conformal_scene_ids=["41069025"])
    with pytest.raises(CalibrationSplitError):
        build_scene_id_audit(["cal_1"], excluded_scene_ids=excluded,
                             conformal_scene_ids=["cal_1"])
    ok = build_scene_id_audit(["cal_1"], excluded_scene_ids=excluded,
                              conformal_scene_ids=["conf_1"])
    assert ok.intersection() == frozenset()
    assert ok.conformal_split_hash and ok.conformal_split_hash != ok.calibration_split_hash
    detail = ok.overlap_detail()
    assert set(detail) == {"calibration_vs_excluded", "conformal_vs_excluded",
                           "calibration_vs_conformal"}
    assert all(v == [] for v in detail.values())


def test_audit_manifest_carries_all_three_hashes(tmp_path):
    audit = build_scene_id_audit([f"cal_{i}" for i in range(5)],
                                 excluded_scene_ids=["41069025"],
                                 conformal_scene_ids=[f"conf_{i}" for i in range(3)])
    out = write_audit_manifest(audit, tmp_path / "audit.json")
    data = json.loads(Path(out).read_text(encoding="utf-8"))
    assert data["n_conformal_scenes"] == 3
    assert data["conformal_split_hash"]
    assert data["overlap_detail"]["conformal_vs_excluded"] == []
    # 未单独留出时显式记 0（不假装有独立校准集）
    plain = build_scene_id_audit(["cal_1"], excluded_scene_ids=["41069025"])
    assert plain.as_manifest()["n_conformal_scenes"] == 0
    assert plain.conformal_split_hash


def test_audit_manifest_roundtrip_and_persistence(tmp_path):
    audit = build_scene_id_audit([f"cal_{i}" for i in range(5)],
                                 excluded_scene_ids=["41069025"])
    out = write_audit_manifest(audit, tmp_path / "audit.json")
    data = json.loads(Path(out).read_text(encoding="utf-8"))
    assert data["intersection_is_empty"] is True
    assert data["n_excluded_vsibench_scenes"] == 1
    assert data["calibration_scene_ids"] == sorted(data["calibration_scene_ids"])


def test_tampered_calibrator_with_intersection_is_rejected(tmp_path):
    """被篡改（把评测 scene 混进标定集）的校准器在反序列化时也必须被拒。"""
    audit = build_scene_id_audit([f"cal_{i}" for i in range(10)],
                                 excluded_scene_ids=["41069025"])
    cal = fit_conformal_calibrator(_records(10), confidence_level=0.90,
                                   split_audit=audit)
    payload = json.loads(cal.to_json())
    payload["split_audit"]["calibration_scene_ids"] = ["41069025"]
    with pytest.raises(CalibrationSplitError):
        ConformalCalibrator.from_dict(payload)


# ------------------------------------------------------------- conformal ----

def _records(n: int, *, sigma: float = 0.06, seed: int = 0,
             rel_ci_base: float = 0.02) -> list[dict]:
    """合成标定记录：`rel_ci` 与真实误差正相关（可算 Spearman ρ>0）。"""
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n):
        err = float(np.exp(rng.normal(0.0, sigma)))
        out.append({"scale_pred": err, "scale_true": 1.0,
                    "rel_ci": rel_ci_base + 0.5 * abs(np.log(err)),
                    "plane_identity_ok": True, "anchor_fired": True})
    return out


def test_conformal_quantile_is_finite_sample_correct():
    scores = np.arange(1, 11, dtype=float)          # n=10
    # level=0.9 → ceil(11*0.9)/10 = 10/10 → 第 10 个（最大）
    assert conformal_quantile(scores, 0.90) == pytest.approx(10.0)
    # level=0.5 → ceil(5.5)/10 = 6/10 → 第 6 个
    assert conformal_quantile(scores, 0.50) == pytest.approx(6.0)
    with pytest.raises(CalibrationUnavailable):
        conformal_quantile([], 0.90)


def test_fit_calibrator_monotone_width_and_coverage():
    audit = build_scene_id_audit([f"cal_{i}" for i in range(80)],
                                 excluded_scene_ids=["41069025"])
    cal = fit_conformal_calibrator(_records(80), confidence_level=0.90,
                                   split_audit=audit, holdout_records=_records(40, seed=7))
    assert cal.quantile > 0
    assert cal.coveral_rel_halfwidth == pytest.approx(np.sinh(cal.quantile), rel=1e-9)
    assert 0.0 <= cal.empirical_coverage <= 1.0
    # 更高 nominal → 更宽的区间（单调性）
    cal95 = fit_conformal_calibrator(_records(80), confidence_level=0.95,
                                     split_audit=audit)
    assert cal95.coveral_rel_halfwidth >= cal.coveral_rel_halfwidth


def test_conformal_apply_only_widens_never_narrows():
    """校准**只加宽不收窄**：解析 CI 更宽说明场景本身更不确定，不得被外部统计抹掉。"""
    audit = build_scene_id_audit([f"cal_{i}" for i in range(30)],
                                 excluded_scene_ids=["41069025"])
    cal = fit_conformal_calibrator(_records(30), confidence_level=0.90, split_audit=audit)
    s, wide = cal.apply(2.0, 0.90)                 # 解析比校准宽
    assert (s, wide) == (2.0, 0.90)
    s2, narrow = cal.apply(2.0, 0.001)             # 解析比校准窄 → 用校准值
    assert narrow == pytest.approx(cal.coveral_rel_halfwidth)
    # scale 非法 → 返回 (None, None)，不返回假数值
    assert cal.apply(None, 0.1) == (None, None)


def test_calibrator_rejects_version_mismatch():
    audit = build_scene_id_audit(["cal_1"], excluded_scene_ids=["41069025"])
    cal = fit_conformal_calibrator(_records(5), confidence_level=0.90, split_audit=audit)
    payload = json.loads(cal.to_json())
    payload["schema_version"] = "scale-conformal-v0"
    with pytest.raises(CalibrationUnavailable):
        ConformalCalibrator.from_dict(payload)
    payload = json.loads(cal.to_json())
    payload["unit_version"] = "v3.0-percent"
    with pytest.raises(CalibrationUnavailable):
        ConformalCalibrator.from_dict(payload)


def test_save_and_load_frozen_calibrator(tmp_path):
    audit = build_scene_id_audit([f"cal_{i}" for i in range(30)],
                                 excluded_scene_ids=["41069025"])
    cal = fit_conformal_calibrator(_records(30), confidence_level=0.90,
                                   split_audit=audit, calibration_id="frozen-001")
    p = save_calibrator(cal, tmp_path / "cal.json")
    back = load_calibrator(p)
    assert back.calibration_id == "frozen-001"
    assert back.quantile == pytest.approx(cal.quantile)
    assert back.schema_version == CALIBRATOR_SCHEMA_VERSION
    assert back.unit_version == CI_UNIT_VERSION
    assert back.split_audit is not None

    # 探针：不可用路径返回 False（不抛异常）
    assert calibrator_available("/nope/cal.json") is False
    assert calibrator_available(p) is True
    with pytest.raises(CalibrationUnavailable):
        load_calibrator(None)


def test_calibrator_requires_min_episodes_for_upgrade():
    """§10.2：标定 episode 数不足时不得支撑 medium（n<30 不具升级资格）。"""
    from skill3d.reconstruction.metric_scale import SCALE_CALIB_MIN_EPISODES

    audit = build_scene_id_audit([f"cal_{i}" for i in range(10)],
                                 excluded_scene_ids=["41069025"])
    cal = fit_conformal_calibrator(_records(10), confidence_level=0.90, split_audit=audit)
    assert cal.n_calibration < SCALE_CALIB_MIN_EPISODES
