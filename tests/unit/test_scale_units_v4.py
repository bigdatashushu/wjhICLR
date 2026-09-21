"""v4 HC29 尺度 CI 口径单测（L0 口径层）。

对应《系统架构4.md》§0.2 HC29 / §4.1 v4 说明 / §10.2 L0：
`scale_ci_rel` 是**分数口径的半宽**，`scale_ci_abs_m = metric_scale × scale_ci_rel`；
百分数、全宽、NaN/Inf、历史不一致样例**全部 fail-closed**。
"""

from __future__ import annotations

import numpy as np
import pytest

from skill3d.reconstruction.scale_units import (
    CI_UNIT_VERSION,
    CiUnitError,
    LegacyCiMigration,
    assert_ci_consistent,
    check_ci_rel,
    ci_abs_m,
    migrate_legacy_ci,
)
from skill3d.schemas import ReconstructionArtifact


def _art(**kw) -> ReconstructionArtifact:
    base = dict(
        artifact_id="a", artifact_version="v", scene_name="s", recon_method="vggt",
        c2w_list="", intrinsics="", depth_maps="", point_map="", point_conf="",
        track_list=None, metric_scale=None, scale_known=False,
        confidence={"per_point_confidence": "", "coverage_count_per_frame": ""})
    base.update(kw)
    return ReconstructionArtifact(**base)


# ------------------------------------------------------------ 换算与自洽 ----

def test_ci_abs_is_scale_times_rel():
    assert ci_abs_m(2.386, 0.137) == pytest.approx(2.386 * 0.137)
    # 任一缺失/非正 → None（不猜、不用 0 冒充）
    assert ci_abs_m(None, 0.1) is None
    assert ci_abs_m(2.0, None) is None
    assert ci_abs_m(0.0, 0.1) is None
    assert ci_abs_m(-1.0, 0.1) is None
    assert ci_abs_m(2.0, float("nan")) is None


def test_assert_ci_consistent_accepts_consistent_and_rejects_others():
    assert_ci_consistent(2.0, 0.1, 0.2)                 # 自洽
    assert_ci_consistent(None, None, None)              # 未锚定：允许
    with pytest.raises(CiUnitError):
        assert_ci_consistent(2.386, 0.5, 32.81)         # §10.2 实测样例：不自洽
    with pytest.raises(CiUnitError):
        assert_ci_consistent(2.0, 0.1, None)            # 部分缺失
    with pytest.raises(CiUnitError):
        assert_ci_consistent(2.0, float("nan"), 0.2)


# --------------------------------------------------------------- 归一化 ----

def test_check_ci_rel_legal_values():
    chk = check_ci_rel(0.137, confidence_level=0.90)
    assert chk.ok and chk.value == pytest.approx(0.137) and chk.reason_code == "ok"
    assert check_ci_rel(0.0).ok                          # 0 是合法下界
    assert check_ci_rel(1.0, confidence_level=0.90).ok   # 100% 半宽合法（虽不可用）


@pytest.mark.parametrize("bad,reason", [
    (None, "absent"),
    (float("nan"), "nan_or_inf"),
    (float("inf"), "nan_or_inf"),
    (-0.1, "negative"),
    (21.875, "percent_like"),        # §10.2 实测的历史 relative_ci
    (500.0, "percent_like"),
])
def test_check_ci_rel_rejects_bad_values(bad, reason):
    chk = check_ci_rel(bad, confidence_level=0.90)
    assert not chk.ok and chk.value is None and chk.reason_code == reason


def test_check_ci_rel_rejects_full_width_misuse():
    """全宽误传会**低估一半**不确定性 → 必须拒绝（HC29）。"""
    chk = check_ci_rel(0.10, total_width=0.20)
    assert not chk.ok and chk.reason_code == "full_width_like"
    # 不是恰好 2 倍时不误伤
    assert check_ci_rel(0.10, total_width=0.31).ok


def test_ci_unit_version_is_recorded():
    """口径版本必须可查（校准器与 artifact 都按它做版本对齐）。"""
    assert CI_UNIT_VERSION.startswith("v4")


# ----------------------------------------------------- 历史字段只读迁移 ----

def test_migrate_legacy_relative_ci_is_never_admission_eligible():
    mig = migrate_legacy_ci(legacy_relative_ci=21.875, scale=2.386,
                            confidence_level=0.90)
    assert isinstance(mig, LegacyCiMigration)
    assert mig.confidence_cap == "low"
    assert not mig.usable_for_admission          # 无论数值是否可解析
    assert mig.reason_code == "legacy_ci_inconsistent"


def test_migrate_legacy_scale_ci_marks_unknown_source():
    """旧 `scale_ci=3.41`（旧公式，约低估 9.6 倍）→ 只作参考，不可准入。"""
    mig = migrate_legacy_ci(legacy_scale_ci=3.41, scale=2.386)
    assert mig.confidence_cap == "low" and not mig.usable_for_admission
    assert mig.reason_code in ("legacy_ci_unknown_source", "legacy_ci_inconsistent")
    if mig.scale_ci_rel is not None:
        assert mig.scale_ci_rel == pytest.approx(3.41 / 2.386, rel=1e-6)


def test_migrate_legacy_without_any_input_is_absent():
    mig = migrate_legacy_ci()
    assert mig.reason_code == "absent" and not mig.usable_for_admission


# ------------------------------------------------- schema 层 fail-closed ----

def test_artifact_inconsistent_ci_downgrades_to_low():
    """不自洽 artifact（HC29）→ 降 low + 清空米制授权 + 写明原因，不抛异常。

    v5 语义修正：降级事实写在 `scale_method` 的原因串里，**不**占用
    `scale_conflict`（HC31：该字段专属"锚点间尺度比超冲突阈值"）。
    """
    art = _art(metric_scale=2.386, scale_known=True, scale_ci_rel=0.5,
               scale_ci_abs_m=32.81, scale_confidence="high",
               allowed_metric_tasks={"object_abs_distance"})
    assert art.scale_confidence == "low"
    assert art.allowed_metric_tasks == set()
    assert art.scale_conflict is False          # 不是锚点冲突，不得误报
    assert "HC29" in art.scale_method


def test_artifact_legacy_only_ci_is_hard_rejected():
    """v5 HC39：旧 `scale_ci` 字段**根本不允许进入当前 Schema**（不是降级）。

    v4 的"读进来降为 low"会让历史数字继续在系统里流动；v5 要求版本隔离：
    含旧字段的 JSON 必须走 `skill3d.legacy.readers` 只读审计，并从原始帧重跑 pipeline。
    """
    import pytest as _pytest

    with _pytest.raises(Exception) as ei:
        _art(metric_scale=2.0, scale_known=True, scale_ci=3.41,
             scale_confidence="high", allowed_metric_tasks={"object_abs_distance"})
    assert "legacy 字段" in str(ei.value) and "scale_ci" in str(ei.value)


def test_artifact_consistent_ci_survives_roundtrip():
    art = _art(metric_scale=2.0, scale_known=True, scale_ci_rel=0.1,
               scale_ci_abs_m=0.2, scale_confidence_level=0.90,
               scale_confidence="medium", scale_calibration_id="cal-1",
               allowed_metric_tasks={"object_size_estimation"})
    assert art.scale_confidence == "medium"
    back = ReconstructionArtifact.model_validate_json(art.model_dump_json())
    assert back.scale_ci_rel == pytest.approx(0.1)
    assert back.scale_ci_abs_m == pytest.approx(0.2)
    assert back.allowed_metric_tasks == {"object_size_estimation"}


def test_uncalibrated_medium_is_downgraded_at_schema_level():
    """HC30 数据层兜底：没有冻结校准器就不可能出现 medium/high。

    这条规则堵住"任何绕过 `assess_scale` 直接构造 artifact"的路径。
    """
    art = _art(metric_scale=2.0, scale_known=True, scale_ci_rel=0.1,
               scale_ci_abs_m=0.2, scale_confidence="medium",
               allowed_metric_tasks={"object_size_estimation"})
    assert art.scale_confidence == "low" and art.allowed_metric_tasks == set()
    assert "HC29/30" in art.scale_method


def test_metric_task_overreach_is_downgraded_at_schema_level():
    """HC33 数据层兜底：medium 不得授权 `object_abs_distance`（仅 high）。"""
    over = _art(metric_scale=2.0, scale_known=True, scale_ci_rel=0.1,
                scale_ci_abs_m=0.2, scale_confidence="medium",
                scale_calibration_id="cal-1",
                allowed_metric_tasks={"object_abs_distance"})
    assert over.scale_confidence == "low" and over.allowed_metric_tasks == set()
    assert "越权授权" in over.scale_method
    # high 但 CI 非有限 → 也不得分授权（HF33：medium/high 必须携带区间）
    no_ci = _art(metric_scale=2.0, scale_known=True, scale_confidence="high",
                 scale_calibration_id="cal-1",
                 allowed_metric_tasks={"object_abs_distance"})
    assert no_ci.scale_confidence == "low" and not no_ci.allowed_metric_tasks


def test_artifact_unknown_metric_task_rejected():
    with pytest.raises(Exception):
        _art(allowed_metric_tasks={"relative_direction"})


def test_nan_ci_is_serializable_and_fail_closed():
    """G-34：NaN 必须能落盘；NaN 的 CI 一律不得被读成"不确定性很小"。"""
    art = _art(metric_scale=2.0, scale_known=True, scale_ci_rel=float("nan"),
               scale_ci_abs_m=float("nan"), scale_confidence="medium",
               allowed_metric_tasks={"object_size_estimation"})
    payload = art.model_dump_json()
    assert "NaN" in payload
    back = ReconstructionArtifact.model_validate_json(payload)
    assert back.scale_confidence == "low" and not back.allowed_metric_tasks
    assert not np.isfinite(back.scale_ci_rel)
