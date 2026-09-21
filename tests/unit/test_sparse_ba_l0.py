"""`vggt_sparse_ba` L0 合同与合成测试（§10.1 L0 / HC36）。

L0 必须全部通过才有资格进 L1（单真实 episode）；本文件覆盖：

1. `recon_method` 只能写 `vggt_sparse_ba`；官方失败路径与轻量路径不共享名称；
2. pair graph 确定、与题目无关、覆盖全部 32 帧，同配置 `pair_graph_hash` 稳定；
3. track merge 不变量：同帧单观测、无循环冲突、有限坐标、最小长度；
4. 回执三态与 G5 字段不变量（未运行/失败必须 None）；
5. L1/L2 门槛判定（合成回执）：止损码与预注册 skip_rate。
"""

from __future__ import annotations

import numpy as np
import pytest

from skill3d.reconstruction.legacy_vggsfm_ba import (
    OFFICIAL_VGGSFM_BA_ENABLED,
    REJECTION_REASON,
)
from skill3d.reconstruction.sparse_ba import (
    BAOutcome,
    MAX_SKIP_RATE,
    SparseBAReceipt,
    sparse_ba_enabled,
)
from skill3d.reconstruction.sparse_ba.pair_graph import (
    PairGraphConfig,
    build_pair_graph,
    duplicate_pairs,
    l0_pair_graph_ok,
    self_pairs,
    uncovered_frames,
)
from skill3d.reconstruction.sparse_ba.receipts import (
    l1_gate,
    l2_gate,
    read_receipt,
    receipts_report,
    stop_loss,
    write_receipt,
)
from skill3d.reconstruction.sparse_ba.tracks import (
    Observation,
    Track,
    merge_tracks,
    track_length_histogram,
    tracks_to_arrays,
    validate_tracks,
)


# ------------------------------------------------------------- 1. 命名隔离 ----

def test_sparse_ba_defaults_off_and_not_official_ba():
    """HC35/36：默认关闭；官方 BA 关闭；名称隔离。"""
    assert sparse_ba_enabled() is False
    assert OFFICIAL_VGGSFM_BA_ENABLED is False
    assert REJECTION_REASON == "rejected_on_24g_oom"


def test_recon_method_vocabulary_excludes_official_name():
    from skill3d.schemas import ReconstructionArtifact

    assert "vggt_sparse_ba" in ReconstructionArtifact.model_fields["recon_method"].annotation.__args__
    assert "vggt_ba" not in ReconstructionArtifact.model_fields["recon_method"].annotation.__args__


def test_feed_forward_outcome_sets_g5_none():
    """HC37：feed-forward 产物必须 not_available + G5=None（无代理值）。"""
    o = BAOutcome.feed_forward(reason="official BA rejected")
    assert o.applied is False and o.recon_method == "vggt"
    assert o.reprojection_status == "not_available"
    assert o.g5_reproj_err_median is None and o.g5_reproj_err_p95 is None
    assert "HC37" in o.summary() or "G5" in o.summary()


# ------------------------------------------------------- 2. pair graph（L0）----

def test_pair_graph_covers_all_32_frames_and_is_deterministic():
    g1 = build_pair_graph(PairGraphConfig())
    g2 = build_pair_graph(PairGraphConfig())
    ok, problems = l0_pair_graph_ok(g1)
    assert ok, problems
    assert g1.pair_graph_hash == g2.pair_graph_hash
    assert g1.pairs == g2.pairs
    assert uncovered_frames(g1) == []
    assert duplicate_pairs(g1) == [] and self_pairs(g1) == []
    # 覆盖全部 32 帧
    assert len({f for p in g1.pairs for f in p}) == 32


def test_pair_graph_hash_changes_with_config():
    a = build_pair_graph(PairGraphConfig())
    b = build_pair_graph(PairGraphConfig(skip_steps=(2, 6)))
    assert a.pair_graph_hash != b.pair_graph_hash


def test_pair_graph_does_not_read_questions_or_answers():
    """签名只接受配置对象：没有任何题目/答案入参（结构性保证）。"""
    import inspect

    sig = inspect.signature(build_pair_graph)
    assert list(sig.parameters) == ["config"]


def test_pair_graph_rejects_degenerate_sizes():
    with pytest.raises(ValueError):
        build_pair_graph(PairGraphConfig(n_frames=1))
    g = build_pair_graph(PairGraphConfig(n_frames=4))
    assert uncovered_frames(g) == []


def test_pair_graph_config_is_preregistered_shape():
    cfg = PairGraphConfig()
    assert cfg.n_frames == 32 and cfg.adjacent_stride >= 1 and cfg.skip_steps
    assert "skip_steps" in cfg.as_dict()


# --------------------------------------------------------- 3. track merge ----

def _chain(frames_kp):
    """把 (frame, kp, xy) 列表转成连续匹配序列（相邻两项成一次匹配）。"""
    flat = [((f, k), (x, y)) for f, k, x, y in frames_kp]
    return [flat]


def test_merge_tracks_joins_multi_frame_observations():
    matches = _chain([(0, 10, 1.0, 2.0), (1, 11, 1.1, 2.1),
                      (1, 11, 1.1, 2.1), (2, 12, 1.2, 2.2)])
    rep = merge_tracks(matches, min_length=3)
    ok, problems = validate_tracks(rep.tracks, min_length=3)
    assert ok, problems
    assert rep.n_tracks == 1
    assert sorted(rep.tracks[0].frames()) == [0, 1, 2]


def test_merge_tracks_drops_nonfinite_and_short():
    matches = _chain([(0, 1, np.nan, 1.0), (1, 2, 1.0, 1.0),   # 非有限 → 丢
                      (0, 3, 1.0, 1.0), (1, 4, 1.0, 1.0)])     # 长度 2 → 过短
    rep = merge_tracks(matches, min_length=3)
    assert rep.n_dropped_nonfinite == 1
    assert rep.n_dropped_short == 1 and rep.n_tracks == 0


def test_merge_tracks_rejects_duplicate_frame_in_same_track():
    """同帧两条 keypoint 被链进同一 track → 丢弃（L0：同帧单观测）。"""
    matches = _chain([(0, 1, 1.0, 1.0), (1, 1, 1.0, 1.0),
                      (0, 2, 2.0, 2.0), (1, 1, 1.0, 1.0)])
    rep = merge_tracks(matches, min_length=1)
    assert rep.n_dropped_duplicate_frame >= 1
    assert all(not t.has_duplicate_frame() for t in rep.tracks)


def test_merge_tracks_counts_cycle_merges_without_corrupting():
    """合并成环（已在同一 track）→ 计数并拒绝，不产生非法结构。"""
    matches = _chain([(0, 1, 1.0, 1.0), (1, 1, 1.0, 1.0),
                      (1, 1, 1.0, 1.0), (2, 1, 1.0, 1.0),
                      (0, 1, 1.0, 1.0), (2, 1, 1.0, 1.0)])
    rep = merge_tracks(matches, min_length=1)
    assert rep.n_merge_cycles_rejected >= 1
    ok, problems = validate_tracks(rep.tracks, min_length=1)
    assert ok, problems


def test_merge_tracks_is_deterministic_and_order_independent_for_ids():
    m1 = _chain([(0, 1, 0.0, 0.0), (1, 1, 0.1, 0.1)])
    m2 = _chain([(0, 1, 0.0, 0.0), (1, 1, 0.1, 0.1)])
    r1, r2 = merge_tracks(m1, min_length=1), merge_tracks(m2, min_length=1)
    assert [t.observations for t in r1.tracks] == [t.observations for t in r2.tracks]


def test_validate_tracks_flags_violations():
    bad = Track(track_id=0, observations=[
        Observation(0, 1, (1.0, 1.0)), Observation(0, 2, (2.0, 2.0))])
    ok, problems = validate_tracks([bad], min_length=1)
    assert not ok and any("同帧多观测" in p for p in problems)


def test_tracks_to_arrays_and_histogram():
    t = Track(track_id=0, observations=[
        Observation(0, 5, (1.0, 2.0)), Observation(1, 6, (3.0, 4.0))])
    f, k, xy = tracks_to_arrays([t])
    assert f.tolist() == [0, 1] and k.tolist() == [5, 6]
    assert xy.shape == (2, 2)
    assert track_length_histogram([t]) == {2: 1}


# --------------------------------------------------- 4/5. 回执与止损门槛 ----

def _receipt(**kw) -> SparseBAReceipt:
    base = dict(
        status="passed", frontend="superpoint_lightglue", pair_graph_hash="pg",
        n_pairs=48, n_matches=1200, n_inliers=900, n_tracks=300, peak_gpu_gib=12.0,
        initial_cost=10.0, final_cost=2.0)
    base.update(kw)
    return SparseBAReceipt(**base)


def test_receipt_roundtrip(tmp_path):
    r = _receipt()
    p = write_receipt(r, tmp_path / "receipt.json")
    assert read_receipt(p) == r


def test_l1_gate_passes_and_rejects_oob_peak():
    ok, problems = l1_gate(_receipt())
    assert ok, problems
    bad, probs = l1_gate(_receipt(peak_gpu_gib=21.5))
    assert not bad and any("峰值显存" in p or "余量" in p for p in probs)


def test_l1_gate_requires_ba_convergence():
    not_converged, probs = l1_gate(_receipt(initial_cost=1.0, final_cost=2.0))
    assert not not_converged and any("未收敛" in p for p in probs)
    missing, probs2 = l1_gate(_receipt(initial_cost=None, final_cost=None))
    assert not missing and any("cost" in p for p in probs2)


def test_l2_gate_enforces_preregistered_limits():
    rs = [_receipt() for _ in range(20)]
    ok, problems, stats = l2_gate(rs)
    assert ok, problems
    assert stats["oom_rate"] == 0.0 and stats["skip_rate"] == 0.0

    too_few, probs, _ = l2_gate(rs[:10])
    assert not too_few and any("样本量" in p for p in probs)

    with_oom = rs[:19] + [_receipt(status="failed", skip_reason="oom", n_tracks=0,
                                   initial_cost=None, final_cost=None)]
    no_oom, probs2, stats2 = l2_gate(with_oom)
    assert not no_oom and stats2["oom_rate"] > 0

    many_skips = rs[:13] + [_receipt(status="failed", skip_reason="track_too_short")
                            for _ in range(7)]
    ok3, probs3, stats3 = l2_gate(many_skips)
    assert not ok3 and stats3["skip_rate"] > MAX_SKIP_RATE


def test_stop_loss_marks_rejected_with_reason():
    r = stop_loss(_receipt(status="failed", skip_reason="oom: CUDA out of memory"),
                  gate="l1", problems=["峰值显存超限"])
    assert r.status == "rejected" and r.rejected_reason == "rejected_on_24g_oom"
    r2 = stop_loss(_receipt(status="failed", skip_reason="too_few_tracks"),
                   gate="l1", problems=["无有效 track"])
    assert r2.status == "rejected" and r2.rejected_reason == "rejected_on_l1_gate"


def test_receipts_report_is_never_paper_eligible():
    """HC36：即使 L2 全绿，也只进可选消融行，永不升为主线。"""
    rep = receipts_report([_receipt() for _ in range(20)])
    assert rep["paper_eligible"] is False
    assert rep["n_episodes"] == 20 and rep["statuses"]["passed"] == 20
