"""v5.1 标定池口径修订单测：**按被评测数据集选择同源非重叠标定池**。

用户 2026-09-20 拍板：评测口径限定 scannet + scannetpp 时，标定池也用这两个数据集
的非重叠场景（同源优先），不再绑定 ARKitScenes；硬门（三向 scene 级互斥）不变。

覆盖面：
1. 三个数据集的排除集规模与官方一致（88 / 50 / 150），且集合两两不同；
2. 隔离 hard fail：标定集/留出集与被评测集相交 → 抛错，拒绝生成 calibration_id；
3. 数量校验：排除清单不完整（数量不符）→ hard fail（防止隔离形同虚设）；
4. 同源标记：`dataset_match` 正确反映"标定数据集 ∈ 被评测数据集"；
5. 多校准器：按数据集从目录选文件、找不到退回 fallback 并标注非同源。
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from skill3d.reconstruction.scale_calibration import (
    CALIBRATOR_SCHEMA_VERSION,
    CalibrationSplitError,
    CalibrationUnavailable,
    ConformalCalibrator,
    build_scene_id_audit,
    calibrator_path_for,
    load_calibrator,
    load_calibrator_for,
    save_calibrator,
    vsibench_arkitscenes_scene_ids,
    vsibench_scene_ids_by_dataset,
)

META = Path("data/vsi_bench_meta/test.jsonl")


def _meta_or_skip() -> Path:
    if not META.is_file():
        pytest.skip("VSI-Bench meta 不在本机（TODO_USER_INPUT）")
    return META


def test_three_datasets_have_official_sizes():
    meta = _meta_or_skip()
    sizes = {ds: len(vsibench_scene_ids_by_dataset(meta, ds))
             for ds in ("scannet", "scannetpp", "arkitscenes")}
    assert sizes == {"scannet": 88, "scannetpp": 50, "arkitscenes": 150}
    ids = {ds: set(vsibench_scene_ids_by_dataset(meta, ds)) for ds in sizes}
    assert not (ids["scannet"] & ids["scannetpp"])
    assert not (ids["scannet"] & ids["arkitscenes"])


def test_arkitscenes_helper_kept_for_backward_compat():
    meta = _meta_or_skip()
    assert vsibench_arkitscenes_scene_ids(meta) == \
        vsibench_scene_ids_by_dataset(meta, "arkitscenes")


def test_unknown_dataset_is_rejected():
    with pytest.raises(CalibrationSplitError):
        vsibench_scene_ids_by_dataset(_meta_or_skip(), "not_a_dataset")


def test_same_source_calibration_pool_passes_and_marks_match():
    """同源标定（scannetpp 标定 + scannet/scannetpp 评测）→ 通过且 dataset_match=True。"""
    meta = _meta_or_skip()
    audit = build_scene_id_audit(
        calibration_scene_ids=["cal_001", "cal_002"],
        meta_path=meta,
        dataset="scannetpp",
        evaluation_datasets=["scannet", "scannetpp"],
        conformal_scene_ids=["cal_hold_001"],
    )
    m = audit.as_manifest()
    assert m["dataset"] == "scannetpp"
    assert m["evaluation_datasets"] == ["scannet", "scannetpp"]
    assert m["dataset_match"] is True
    # 排除集是 scannet + scannetpp 的并集
    assert m["n_excluded_vsibench_scenes"] == 88 + 50


def test_offsource_calibration_is_allowed_but_flagged():
    """非同源标定（arkitscenes 标定 + scannet/scannetpp 评测）→ 允许但标记 False。"""
    meta = _meta_or_skip()
    audit = build_scene_id_audit(
        calibration_scene_ids=["cal_001"], meta_path=meta,
        dataset="arkitscenes", evaluation_datasets=["scannet", "scannetpp"])
    assert audit.as_manifest()["dataset_match"] is False


def test_overlap_with_evaluated_scenes_is_hard_fail():
    meta = _meta_or_skip()
    bad = vsibench_scene_ids_by_dataset(meta, "scannetpp")[0]
    with pytest.raises(CalibrationSplitError):
        build_scene_id_audit([bad], meta_path=meta, dataset="scannetpp",
                             evaluation_datasets=["scannetpp"])


def test_conformal_holdout_must_be_disjoint_too():
    meta = _meta_or_skip()
    with pytest.raises(CalibrationSplitError):
        build_scene_id_audit(["a", "b"], meta_path=meta, dataset="scannetpp",
                             evaluation_datasets=["scannetpp"],
                             conformal_scene_ids=["b"])


def test_incomplete_exclusion_list_is_hard_fail():
    """排除清单数量不符官方（漏读/截断）→ hard fail，不允许"隔离形同虚设"。"""
    meta = _meta_or_skip()
    with pytest.raises(CalibrationSplitError):
        build_scene_id_audit(["x"], excluded_scene_ids=["only_one"],
                             dataset="scannetpp",
                             expected_excluded_count=50)


def _fake_calibrator(dataset: str, cal_id: str) -> ConformalCalibrator:
    return ConformalCalibrator(
        calibration_id=cal_id, confidence_level=0.90, quantile=0.2,
        n_calibration=30, empirical_coverage=0.91, median_rel_error=0.12,
        spearman_rho=0.6, calibration_dataset=dataset,
        evaluation_datasets=("scannet", "scannetpp"))


def test_multi_calibrator_selection_by_dataset(tmp_path):
    root = tmp_path / "cal"
    root.mkdir()
    save_calibrator(_fake_calibrator("scannetpp", "cal-scannetpp"),
                    calibrator_path_for("scannetpp", root))
    save_calibrator(_fake_calibrator("scannet", "cal-scannet"),
                    calibrator_path_for("scannet", root))
    cal, notes = load_calibrator_for(["scannetpp", "scannet"], root)
    assert cal is not None and cal.calibration_id == "cal-scannetpp"
    assert cal.dataset_match is True and cal.calibration_dataset == "scannetpp"
    assert any("scannetpp" in n for n in notes)
    # 只要求 scannet → 取到 scannet
    cal2, _ = load_calibrator_for(["scannet"], root)
    assert cal2 is not None and cal2.calibration_id == "cal-scannet"
    # 目录里没有 arkitscenes → 退回 fallback，并标注非同源
    fb = tmp_path / "fallback.json"
    save_calibrator(_fake_calibrator("arkitscenes", "cal-arkit"), fb)
    cal3, notes3 = load_calibrator_for(["arkitscenes"], root, fallback=fb)
    assert cal3 is not None and cal3.calibration_id == "cal-arkit"
    assert cal3.dataset_match is False
    assert any("非同源" in n for n in notes3)


def test_no_calibrator_returns_none_not_error(tmp_path):
    cal, notes = load_calibrator_for(["scannetpp"], tmp_path / "empty")
    assert cal is None and notes


def test_calibrator_roundtrip_keeps_dataset_metadata(tmp_path):
    p = tmp_path / "c.json"
    save_calibrator(_fake_calibrator("scannetpp", "cal-x"), p)
    data = json.loads(p.read_text(encoding="utf-8"))
    assert data["schema_version"] == CALIBRATOR_SCHEMA_VERSION
    assert data["calibration_dataset"] == "scannetpp"
    assert data["dataset_match"] is True
    back = load_calibrator(p)
    assert back.calibration_dataset == "scannetpp" and back.dataset_match is True


def test_calibrator_version_mismatch_is_unavailable(tmp_path):
    p = tmp_path / "c.json"
    save_calibrator(_fake_calibrator("scannet", "cal-y"), p)
    data = json.loads(p.read_text(encoding="utf-8"))
    data["schema_version"] = "scale-conformal-v0"
    p.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(CalibrationUnavailable):
        load_calibrator(p)
