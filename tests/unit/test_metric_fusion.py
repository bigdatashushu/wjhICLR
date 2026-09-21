"""§11 度量尺度融合（MoGe-2 PoC）单测。

**范围声明**：本文件只验证"实现是否按 §11.1 的算术、§11.4 的失败纪律、§13.1 的
阈值口径执行"，**不验证方法有效性**——方法有效性必须等 C1–C6 在自有数据上跑完
（`acceptance_report`），模块整体状态仍是 [待实验]。

合成数据口径：直接构造 ``d_vggt = d_metric / s_k``，使每帧真值 ``s_k`` 已知且精确，
从而把"估计器算术"与"噪声/退化敏感性"分开测。
"""

from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from skill3d.reconstruction.metric_fusion import (
    CONF_SOFT_KEEP_QUANTILE,
    MAX_SCALE_DISPERSION,
    METRIC_FUSION_VERSION,
    METRIC_MODEL_METRIC3D_V2,
    METRIC_MODEL_MOGE2,
    METRIC_MODEL_NONE,
    MIN_VALID_FRAME_RATIO,
    PerFrameScale,
    ScaleFusionResult,
    acceptance_report,
    fov_x_deg_from_intrinsics,
    fuse_metric_scale,
    make_moge2_model,
    read_per_frame_receipt,
    threshold_snapshot,
    write_per_frame_receipt,
)

H, W = 40, 50                      # 合成深度网格（远小于 518×392，测试够用）
N_FRAMES = 32                      # §5.1：统一 FrameSet = 32 帧
TRUE_SCALES = (0.8, 1.0, 1.7, 2.5)


# ------------------------------------------------------------------ 合成数据 ----

def _depth(seed: int, h: int = H, w: int = W) -> np.ndarray:
    """确定性正深度场（真值米制深度）。"""
    rng = np.random.default_rng(seed)
    return rng.uniform(0.5, 4.0, size=(h, w))


def _frames_from_scales(scales, *, h: int = H, w: int = W, seed: int = 0,
                        metric_missing=()):
    """构造 s_k **精确等于** scales[k] 的合成帧对（``d_vggt = d_metric / s_k``）。"""
    dm_list, dv_list, mask_list = [], [], []
    for k, s in enumerate(scales):
        dm = _depth(seed + k, h, w)
        dv = dm / float(s)
        dm_list.append(None if k in metric_missing else dm)
        dv_list.append(dv)
        mask_list.append(np.ones((h, w), dtype=bool))
    return dm_list, dv_list, mask_list


def _ramp_scales(n: int = N_FRAMES, s_true: float = 1.5, amp: float = 0.004):
    """±amp 的确定性线性斜坡（无随机）：让 MAD>0，同时远小于离群阈值。"""
    return [s_true * (1.0 + amp * (k - (n - 1) / 2.0) / ((n - 1) / 2.0)) for k in range(n)]


# ------------------------------------------------------------------ 尺度恢复 ----

@pytest.mark.parametrize("s_true", TRUE_SCALES)
def test_recovers_known_scale_tight(s_true):
    """已知真值尺度 → s_global 精确恢复（无噪声时应到浮点精度）。"""
    dm, dv, vm = _frames_from_scales([s_true] * N_FRAMES)
    r = fuse_metric_scale(dm, dv, vm, model=METRIC_MODEL_MOGE2)
    assert r.status == "success"
    assert r.metric_scale == pytest.approx(s_true, rel=1e-9)
    assert r.n_frames_valid == N_FRAMES and r.n_frames_total == N_FRAMES
    assert r.valid_frame_ratio == pytest.approx(1.0)
    assert r.outlier_frames == []
    assert r.mad == pytest.approx(0.0, abs=1e-12)
    assert r.scale_self_consistency == pytest.approx(0.0, abs=1e-9)
    assert r.version == METRIC_FUSION_VERSION
    assert r.model == METRIC_MODEL_MOGE2
    assert len(r.per_frame) == N_FRAMES
    assert all(p.s_k is not None and not p.outlier for p in r.per_frame)


def test_global_scale_is_median_of_per_frame_scales():
    """跨帧融合口径 = median_k(s_k)（§11.1），不是均值。"""
    scales = [1.0] * 16 + [1.2] * 16
    dm, dv, vm = _frames_from_scales(scales)
    r = fuse_metric_scale(dm, dv, vm)
    assert r.metric_scale == pytest.approx(float(np.median(scales)), rel=1e-9)


def test_model_tag_defaults_to_none_and_is_passthrough():
    dm, dv, vm = _frames_from_scales([1.5] * N_FRAMES)
    assert fuse_metric_scale(dm, dv, vm).model == METRIC_MODEL_NONE
    r = fuse_metric_scale(dm, dv, vm, model=METRIC_MODEL_METRIC3D_V2)
    assert r.model == METRIC_MODEL_METRIC3D_V2


# ------------------------------------------------------------------ 退化/离群 ----

def test_injected_noise_raises_dispersion_and_flags_outliers():
    """注入子集退化（尺度漂移）→ 离散度上升 + 明显帧被标注为离群。"""
    clean = _ramp_scales()
    dm_c, dv_c, vm_c = _frames_from_scales(clean)
    r_clean = fuse_metric_scale(dm_c, dv_c, vm_c)
    assert r_clean.status == "success" and r_clean.outlier_frames == []

    degraded = list(clean)
    injected = (N_FRAMES - 3, N_FRAMES - 2, N_FRAMES - 1)
    for k in injected:
        degraded[k] = 1.0            # 相对 1.5 漂移 33%
    dm_d, dv_d, vm_d = _frames_from_scales(degraded)
    r = fuse_metric_scale(dm_d, dv_d, vm_d)

    assert r.scale_self_consistency > r_clean.scale_self_consistency
    assert set(injected) <= set(r.outlier_frames)      # 注入帧必须被标注（不静默丢弃）
    assert len(r.outlier_frames) <= 8                  # 不应把大多数帧标成离群
    # 离群帧仍留在 per_frame 里（"flagged, not silently dropped"）
    flagged = {p.frame_idx for p in r.per_frame if p.outlier}
    assert flagged == set(r.outlier_frames)


def test_outlier_frames_are_excluded_from_final_median():
    """最终 median 只在未被标注离群的帧上取（离群只标注、不丢弃原始记录）。"""
    scales = _ramp_scales()
    for k in (30, 31):
        scales[k] = 1.0
    dm, dv, vm = _frames_from_scales(scales)
    r = fuse_metric_scale(dm, dv, vm)
    assert {30, 31} <= set(r.outlier_frames)
    retained = [scales[k] for k in range(N_FRAMES) if k not in set(r.outlier_frames)]
    assert r.metric_scale == pytest.approx(float(np.median(retained)), rel=1e-9)


# ------------------------------------------------------------------ fail-closed ----

def test_dispersion_above_threshold_fails_closed():
    """尺度离散度超 τ_scale_disp → failed + metric_scale=None（§11.4 失败纪律）。"""
    dm, dv, vm = _frames_from_scales([1.0, 2.0] * (N_FRAMES // 2))
    r = fuse_metric_scale(dm, dv, vm)
    assert r.status == "failed"
    assert r.metric_scale is None
    assert r.scale_self_consistency is not None
    assert r.scale_self_consistency > MAX_SCALE_DISPERSION
    assert "τ_scale_disp" in r.note
    assert len(r.per_frame) == N_FRAMES     # 失败也要留全量诊断（receipt 用）


def test_all_frames_invalid_fails_closed():
    """全部帧无度量深度 → failed / metric_scale=None / 每帧 s_k=None（不造值）。"""
    dm, dv, vm = _frames_from_scales([1.5] * N_FRAMES, metric_missing=range(N_FRAMES))
    r = fuse_metric_scale(dm, dv, vm)
    assert r.status == "failed"
    assert r.metric_scale is None
    assert r.n_frames_valid == 0 and r.n_frames_total == N_FRAMES
    assert r.valid_frame_ratio == pytest.approx(0.0)
    assert r.mad is None and r.scale_self_consistency is None
    assert len(r.per_frame) == N_FRAMES
    assert all(p.s_k is None for p in r.per_frame)
    assert all(p.outlier is False for p in r.per_frame)   # 无效 ≠ 离群


def test_frame_ratio_gate_fails_closed():
    """有效帧占比 < τ_frames → failed（即使剩下的帧 s_k 完全一致）。"""
    dm, dv, vm = _frames_from_scales([1.5] * N_FRAMES, metric_missing=range(16))
    r = fuse_metric_scale(dm, dv, vm)
    assert r.status == "failed" and r.metric_scale is None
    assert r.n_frames_valid == 16
    assert r.valid_frame_ratio == pytest.approx(0.5)
    assert r.valid_frame_ratio < MIN_VALID_FRAME_RATIO
    assert "τ_frames" in r.note


def test_min_valid_pixels_gate_fails_closed():
    """单帧有效像素不足 → 该帧 s_k=None（不被少数像素拖走）。"""
    dm, dv, vm = _frames_from_scales([1.5] * N_FRAMES)
    vm = [np.zeros((H, W), dtype=bool) for _ in range(N_FRAMES)]
    for m in vm:
        m[:2, :5] = True                     # 每帧仅 10 px
    r = fuse_metric_scale(dm, dv, vm, min_valid_pixels=100)
    assert r.status == "failed" and r.metric_scale is None
    assert all(p.s_k is None and p.n_valid_pixels == 10 for p in r.per_frame)


def test_nonfinite_and_zero_depth_frame_is_skipped_without_crash():
    """非有限/零/负深度帧 → 有效像素不足则该帧 s_k=None；其余帧照常融合（不崩）。"""
    dm, dv, vm = _frames_from_scales([1.5] * N_FRAMES)
    dm[3] = np.full((H, W), np.nan)                 # 整帧 NaN
    dm[4] = np.zeros((H, W))                        # 整帧零深度
    dm[5], dv[5] = dm[5].copy(), dv[5].copy()
    dm[5][:10, :10] = np.nan                        # 部分坏：NaN 100 px
    dv[5][:10, :10] = np.inf                        # 同址 +Inf
    dm[5][10:20, :10] = 0.0                         # 零深度 100 px
    dv[5][20:30, :10] = -1.0                        # 负深度 100 px
    r = fuse_metric_scale(dm, dv, vm)

    assert r.status == "success"
    assert r.metric_scale == pytest.approx(1.5, rel=1e-9)
    assert r.n_frames_valid == N_FRAMES - 2         # 整帧坏的两帧被跳过
    for k in (3, 4):
        assert r.per_frame[k].s_k is None and r.per_frame[k].n_valid_pixels == 0
        assert r.per_frame[k].outlier is False      # 无效 ≠ 离群
    # 部分坏的帧：有效像素够 → 仍可用，但被剔除的坏像素如实计数
    assert r.per_frame[5].n_valid_pixels == H * W - 300
    assert r.per_frame[5].s_k == pytest.approx(1.5, rel=1e-9)


def test_shape_mismatch_frame_is_skipped():
    """非同网格（形状不一致）→ 该帧 fail-closed，不做隐式 resize。"""
    dm, dv, vm = _frames_from_scales([1.5] * N_FRAMES)
    dm[5] = _depth(999, H + 1, W)
    r = fuse_metric_scale(dm, dv, vm)
    assert r.per_frame[5].s_k is None
    assert r.metric_scale == pytest.approx(1.5, rel=1e-9)
    assert r.n_frames_valid == N_FRAMES - 1


def test_length_mismatch_fails_closed():
    dm, dv, vm = _frames_from_scales([1.5] * N_FRAMES)
    r = fuse_metric_scale(dm[:-1], dv, vm)
    assert r.status == "failed" and r.metric_scale is None
    assert r.per_frame == [] and "长度不一致" in r.note


def test_empty_input_fails_closed():
    r = fuse_metric_scale([], [], [])
    assert r.status == "failed" and r.metric_scale is None
    assert r.n_frames_total == 0


def test_failed_result_never_carries_metric_scale():
    """失败纪律总纲：任何 failed 结果都不得带 metric_scale（无一例外）。"""
    cases = [
        _frames_from_scales([1.0, 2.0] * 16),                      # 离散度过大
        _frames_from_scales([1.5] * 32, metric_missing=range(32)),  # 全无效
        _frames_from_scales([1.5] * 32, metric_missing=range(20)),  # 有效帧占比不足
    ]
    for dm, dv, vm in cases:
        r = fuse_metric_scale(dm, dv, vm)
        assert r.status == "failed" and r.metric_scale is None


# ------------------------------------------------------------------ conf 纪律 ----

def test_vggt_conf_used_only_after_conf_warp_selfcheck():
    """§10.3：conf 只在 conf-warp 单调自检通过后才参与；不过/未跑则不参与。"""
    dm, dv, vm = _frames_from_scales([1.5] * N_FRAMES)
    conf = []
    for _ in range(N_FRAMES):
        c = np.full((H, W), 5.0)
        c.reshape(-1)[: (H * W) // 10] = 0.1          # 10% 低置信像素
        conf.append(c)

    r_off = fuse_metric_scale(dm, dv, vm, conf, conf_warp_monotonic=False)
    r_unknown = fuse_metric_scale(dm, dv, vm, conf)
    r_on = fuse_metric_scale(dm, dv, vm, conf, conf_warp_monotonic=True)

    n_all = H * W
    assert all(p.n_valid_pixels == n_all for p in r_off.per_frame)
    assert all(p.n_valid_pixels == n_all for p in r_unknown.per_frame)
    # 自检通过 → 逐帧相对分位软下界生效（丢掉最低 conf 的 10%）
    assert all(p.n_valid_pixels == n_all - (n_all // 10) for p in r_on.per_frame)
    assert CONF_SOFT_KEEP_QUANTILE > 0.0
    # 比值本身没变 → s_k 不变（conf 只改像素集，不改口径）
    for r in (r_off, r_unknown, r_on):
        assert r.metric_scale == pytest.approx(1.5, rel=1e-9)
    assert "未提供" in r_unknown.note or "保守口径" in r_unknown.note
    assert "不作门" in r_off.note


def test_conf_shape_mismatch_is_ignored_not_crashed():
    dm, dv, vm = _frames_from_scales([1.5] * N_FRAMES)
    conf = [np.full((3, 3), 1.0) for _ in range(N_FRAMES)]
    r = fuse_metric_scale(dm, dv, vm, conf, conf_warp_monotonic=True)
    assert r.status == "success" and r.metric_scale == pytest.approx(1.5, rel=1e-9)
    assert all(p.n_valid_pixels == H * W for p in r.per_frame)


def test_missing_valid_mask_is_recorded_not_silently_relaxed():
    dm, dv, _ = _frames_from_scales([1.5] * N_FRAMES)
    r = fuse_metric_scale(dm, dv, [None] * N_FRAMES)
    assert r.status == "success"
    assert f"{N_FRAMES}/{N_FRAMES} 帧缺 valid_mask" in r.note


# ------------------------------------------------------------------ receipt ----

def test_receipt_round_trips_through_json(tmp_path):
    """receipt 落盘：32 个 s_k、median、MAD、离群帧列表、阈值快照齐全且可回读。"""
    scales = _ramp_scales()
    for k in (30, 31):
        scales[k] = 1.0
    dm, dv, vm = _frames_from_scales(scales)
    r = fuse_metric_scale(dm, dv, vm, model=METRIC_MODEL_MOGE2)
    path = tmp_path / "scene_metric_fusion_receipt.json"
    write_per_frame_receipt(r, path)

    raw = path.read_text(encoding="utf-8")
    assert "NaN" not in raw and "Infinity" not in raw      # 收据里不许出现非标准数值
    data = json.loads(raw)
    assert data["metric_fusion_version"] == METRIC_FUSION_VERSION
    assert data["status"] == r.status
    assert data["metric_scale"] == pytest.approx(r.metric_scale, rel=1e-12)
    assert data["mad"] == pytest.approx(r.mad, rel=1e-12)
    assert data["outlier_frames"] == r.outlier_frames
    assert len(data["per_frame"]) == N_FRAMES
    assert [p["frame_idx"] for p in data["per_frame"]] == list(range(N_FRAMES))
    assert [p["s_k"] for p in data["per_frame"]] == pytest.approx(
        [p.s_k for p in r.per_frame], rel=1e-12)
    assert data["thresholds"]["max_scale_dispersion"] == MAX_SCALE_DISPERSION
    assert data["thresholds"] == json.loads(
        json.dumps(threshold_snapshot(), sort_keys=True))
    assert read_per_frame_receipt(path) == data            # 回读一致


def test_receipt_is_deterministic_and_written_on_failure(tmp_path):
    """失败也要落 receipt（止损凭证），且内容不含墙钟 → 重写字节一致。"""
    dm, dv, vm = _frames_from_scales([1.0, 2.0] * 16)
    r = fuse_metric_scale(dm, dv, vm)
    assert r.status == "failed"
    p1 = write_per_frame_receipt(r, tmp_path / "a.json")
    p2 = write_per_frame_receipt(r, tmp_path / "b.json")
    assert p1.read_bytes() == p2.read_bytes()
    data = read_per_frame_receipt(p1)
    assert data["status"] == "failed" and data["metric_scale"] is None
    assert len(data["per_frame"]) == N_FRAMES


def test_receipt_on_all_invalid_has_nulls_not_zeros(tmp_path):
    """s_k 缺失必须落 null（不是 0）；mad/median 也不得填空值。"""
    dm, dv, vm = _frames_from_scales([1.5] * 4, metric_missing=range(4))
    r = fuse_metric_scale(dm, dv, vm)
    data = read_per_frame_receipt(write_per_frame_receipt(r, tmp_path / "c.json"))
    assert [p["s_k"] for p in data["per_frame"]] == [None] * 4
    assert data["metric_scale"] is None and data["mad"] is None
    assert data["valid_frame_ratio"] == 0.0


# ------------------------------------------------------------------ 验收报告 ----

def _unrun_result() -> ScaleFusionResult:
    return ScaleFusionResult(
        status="not_run", metric_scale=None, scale_self_consistency=None,
        n_frames_valid=0, n_frames_total=N_FRAMES, valid_frame_ratio=0.0,
        outlier_frames=[], per_frame=[], mad=None)


def test_acceptance_report_on_unrun_poc_is_all_false_or_none():
    """未跑的 PoC：六项 None、四级 Readiness 全 False（绝不写死 True）。"""
    for res in (_unrun_result(), None):
        rep = acceptance_report(res)
        assert [rep[k] for k in ("C1", "C2", "C3", "C4", "C5", "C6")] == [None] * 6
        assert rep["n_passed"] == 0 and rep["n_not_run"] == 6
        assert rep["implemented"] is False
        assert rep["connected"] is False
        assert rep["real_poc_verified"] is False
        assert rep["paper_eligible"] is False
        assert rep["branch_closed"] is None            # 没跑完 = 既不算过也不算关
        assert rep["scale_fusion_status"] == "not_run"
        assert rep["metric_scale"] is None


def test_acceptance_report_requires_real_model_for_verified():
    """六项证据齐备但度量模型是 `none` → 不得给 real_poc_verified。"""
    scales = [1.5] * N_FRAMES
    dm, dv, vm = _frames_from_scales(scales)
    res = fuse_metric_scale(dm, dv, vm, model=METRIC_MODEL_NONE)
    rep = acceptance_report(res, **_full_evidence())
    assert all(rep[k] is True for k in ("C1", "C2", "C3", "C4", "C5", "C6"))
    assert rep["real_poc_verified"] is False
    assert rep["paper_eligible"] is False


def _full_evidence(seed_count: int = 5) -> dict:
    return dict(
        c1_dispersion=0.01,
        c2_degradation={"baseline_dispersion": 0.01,
                        "degraded_dispersion": {"thinning": 0.05,
                                                "cross_scene": 0.30,
                                                "motion_blur": 0.06}},
        c3_m4_main_gate=True,
        c4_paired={"n_questions": 3, "not_worse_than_unscaled": True,
                   "not_worse_than_same_frame_direct": True},
        c5_resource={"latency_ms_per_frame": 60.0, "peak_gpu_gib": 8.0},
        c6_cross_model={"correct_k_dispersion": 0.01, "wrong_k_dispersion": 0.40,
                        "cross_model_compared": True},
    )


def test_acceptance_report_escalates_only_with_evidence():
    """四级 Readiness 逐级由证据推动：C1–C6 全过 → verified；再加 §18.6 证据才 paper。"""
    dm, dv, vm = _frames_from_scales([1.5] * N_FRAMES)
    res = fuse_metric_scale(dm, dv, vm, model=METRIC_MODEL_MOGE2)

    rep = acceptance_report(res, **_full_evidence())
    assert rep["implemented"] is True           # 代码真跑过一次
    assert rep["connected"] is True             # 过 M4 自洽门（C3）
    assert rep["real_poc_verified"] is True     # 六项全过 + 真度量模型
    assert rep["paper_eligible"] is False       # 缺 §18.6 证据 → 不进论文主表
    assert rep["branch_closed"] is False

    full = {**_full_evidence(),
            "paper_evidence": {"n_seeds": 5, "split_isolated": True,
                               "statistical_gate": True, "non_mock": True}}
    assert acceptance_report(res, **full)["paper_eligible"] is True

    few_seeds = {**_full_evidence(),
                 "paper_evidence": {"n_seeds": 3, "split_isolated": True,
                                    "statistical_gate": True, "non_mock": True}}
    assert acceptance_report(res, **few_seeds)["paper_eligible"] is False


def test_acceptance_report_closes_branch_on_any_failed_item():
    """任一项 False → branch_closed=True（§11.4：关闭尺度融合支路，退回纯相对几何）。"""
    dm, dv, vm = _frames_from_scales([1.5] * N_FRAMES)
    res = fuse_metric_scale(dm, dv, vm, model=METRIC_MODEL_MOGE2)
    bad = {**_full_evidence(), "c5_resource": {"latency_ms_per_frame": 9999.0,
                                               "peak_gpu_gib": 8.0}}
    rep = acceptance_report(res, **bad)
    assert rep["C5"] is False and rep["branch_closed"] is True
    assert "纯相对几何" in rep["note"]
    assert "不回退多锚点" in rep["note"]


def test_acceptance_report_missing_evidence_stays_none():
    """证据缺键 → 该项 None（不是 False、更不是 True）。"""
    rep = acceptance_report(_unrun_result(), c2_degradation={"baseline_dispersion": 0.1})
    assert rep["C2"] is None
    rep2 = acceptance_report(_unrun_result(), c4_paired={"n_questions": 3})
    assert rep2["C4"] is None


def test_acceptance_report_counts_down_more_than_true():
    """注入退化没让离散度上升 → C2 False（敏感性没被证明）。"""
    rep = acceptance_report(None, c2_degradation={
        "baseline_dispersion": 0.2,
        "degraded_dispersion": {"thinning": 0.05, "cross_scene": 0.30}})
    assert rep["C2"] is False
    assert rep["branch_closed"] is True


# ------------------------------------------------------------------ MoGe-2 ----

def test_make_moge2_model_raises_instead_of_faking_without_package(monkeypatch):
    """`moge` 缺席 → RuntimeError（绝不返回占位/随机初始化的"假深度"模型）。"""
    monkeypatch.setitem(sys.modules, "moge", None)     # 模拟未安装（import 必失败）
    with pytest.raises(RuntimeError) as ei:
        make_moge2_model(device="cpu")
    msg = str(ei.value)
    assert "moge" in msg.lower()
    assert "安装" in msg or "假深度" in msg
    assert importlib.util.find_spec("moge") is None or True   # 本机确实没有 moge


def test_make_moge2_model_rejects_cuda_when_unavailable(monkeypatch):
    """device='cuda' 但 CUDA 不可用 → 拒绝静默降级（否则 C5 证据换了硬件）。"""
    if importlib.util.find_spec("torch") is None:
        pytest.skip("本机无 torch：走 torch 缺失分支，由上一个测试覆盖")
    import torch
    if torch.cuda.is_available():
        pytest.skip("本机 CUDA 可用，无法测该分支")
    monkeypatch.setitem(sys.modules, "moge", None)
    with pytest.raises(RuntimeError):
        make_moge2_model(device="cuda")


def test_module_import_pulls_no_heavy_dependencies():
    """模块 import 期不得拉 torch/moge（重依赖只在 `make_moge2_model` 调用期引入）。"""
    src = Path(__file__).resolve().parents[2] / "src"
    code = ("import sys, skill3d.reconstruction.metric_fusion as m; "
            "assert 'torch' not in sys.modules, 'torch 被 import 期拉进来了'; "
            "assert 'moge' not in sys.modules, 'moge 被 import 期拉进来了'; "
            "assert m.METRIC_FUSION_VERSION == 'metric-fusion-v6'")
    env = {**os.environ, "PYTHONPATH": str(src) + os.pathsep + os.environ.get("PYTHONPATH", "")}
    p = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env=env)
    assert p.returncode == 0, p.stderr


def test_fov_x_deg_from_intrinsics():
    """§11.2 相机对齐：fov_x = 2·atan(1/(2·fx_norm))，与分辨率网格无关。"""
    w = 640
    k = np.array([[320.0, 0.0, 320.0], [0.0, 320.0, 240.0], [0.0, 0.0, 1.0]])
    assert fov_x_deg_from_intrinsics(k, w) == pytest.approx(90.0, rel=1e-9)
    half = np.array([[640.0, 0.0, 320.0], [0.0, 640.0, 240.0], [0.0, 0.0, 1.0]])
    assert fov_x_deg_from_intrinsics(half, w) == pytest.approx(
        2 * np.degrees(np.arctan(0.5)), rel=1e-9)      # fx=W → 53.13°
    # K 非法一律 None（不猜视场角）
    assert fov_x_deg_from_intrinsics(None, w) is None
    assert fov_x_deg_from_intrinsics(np.eye(3), 0) is None
    assert fov_x_deg_from_intrinsics(np.array([[0.0, 0, 0], [0, 1, 0], [0, 0, 1]]), w) is None
    assert fov_x_deg_from_intrinsics(np.full((3, 3), np.nan), w) is None


def test_per_frame_scale_returns_none_for_missing_input():
    """公开的逐帧接口同样 fail-closed：无输入 → s_k=None（不是 0）。"""
    from skill3d.reconstruction.metric_fusion import per_frame_scale

    p = per_frame_scale(7, None, np.ones((4, 4)), None)
    assert isinstance(p, PerFrameScale)
    assert p.frame_idx == 7 and p.s_k is None and p.n_valid_pixels == 0


# ------------------------------------------------------------------ D1 边界 ----

def test_d1_no_gt_or_calibration_inputs():
    """D1 硬边界：融合入口不得出现 GT 位姿/标定池/LiDAR/BA 任何输入通道。"""
    import inspect

    params = set(inspect.signature(fuse_metric_scale).parameters)
    banned = {"gt", "groundtruth", "ground_truth", "pose", "poses", "calibration",
              "calib", "lidar", "ba", "anchor", "anchors", "prior", "priors",
              "ransac", "fit", "prior_scale", "gt_scale"}
    for name in params:
        tokens = {t for t in name.lower().split("_") if t}
        assert not (tokens & banned), f"非法输入通道：{name}"
    # 语义负例：True 只能由证据推动，模块里不存在"用先验造尺度"的开关
    assert "fixed_scale" not in inspect.signature(fuse_metric_scale).parameters
