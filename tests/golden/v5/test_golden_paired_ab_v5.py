"""v5 golden 层：冻结 A/B artifact + 幂等重放 + paired 统计稳定性（HC39）。

§4 M17/§7.1 G-02 验收：
- A/B 两臂复用**同一** frozen ReconstructionArtifact（硬约束 18）；
- 同 seed 重放 trace **字节级一致**；
- paired bootstrap CI + Wilcoxon 可复现（重复调用逐位一致）；
- 夹具本身有内容哈希与版本三元组，被改动/版本不符即失败。

**v5 变更（HC37/38/39）**：
- 夹具不含 G8 字段，G5 为 `None` 且 `reprojection_status="not_available"`
  （无真 BA），`overall_quality` 分母不含 G5；
- 混用旧 golden（`tests/golden/archive_v4/`）必须 **hard fail**，
  不得自动重算或静默比较。

夹具由 `make_golden.py` 生成；若需重新生成，请在同一 commit 下运行并提交全部产物。
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from skill3d.evaluation.golden_v5 import (
    GOLDEN_QUALITY_METRIC_VERSION,
    GOLDEN_SCHEMA_VERSION,
    GOLDEN_VERSION,
    GoldenVersionError,
    assert_not_legacy_golden,
    load_golden_stats,
)
from skill3d.evolution.branch import derive_branches
from skill3d.evolution.paired_score import score_paired
from skill3d.evolution.snapshot import build_snapshot
from skill3d.online.runner import OnlineRunConfig, run_episode
from skill3d.schemas import ReconstructionArtifact
from skill3d.skills.paired_ab import (
    PairedArtifactMismatchError,
    assert_same_reconstruction_artifact,
    run_paired_ab,
)

DATA = Path(__file__).resolve().parent / "data"
ARTIFACT_JSON = DATA / "frozen_artifact.json"
STATS_JSON = DATA / "golden_stats.json"
ARCHIVE_DIR = Path(__file__).resolve().parents[1] / "archive_v4"

pytestmark = pytest.mark.skipif(
    not ARTIFACT_JSON.is_file(), reason="golden 夹具未生成（跑 make_golden.py）")


# ------------------------------------------------------------------ 夹具 ----

def _load_artifact() -> ReconstructionArtifact:
    art = ReconstructionArtifact.model_validate_json(
        ARTIFACT_JSON.read_text(encoding="utf-8"))
    return art.model_copy(update={
        "c2w_list": str(DATA / art.c2w_list),
        "intrinsics": str(DATA / art.intrinsics),
        "depth_maps": str(DATA / art.depth_maps),
        "point_map": str(DATA / art.point_map),
        "point_conf": str(DATA / art.point_conf),
    })


def _golden() -> dict:
    return json.loads(STATS_JSON.read_text(encoding="utf-8"))


def test_frozen_artifact_content_hash_matches_golden():
    """夹具未被篡改：文件字节 sha256 与 golden 记录一致。"""
    payload = ARTIFACT_JSON.read_bytes()
    assert hashlib.sha256(payload).hexdigest() == _golden()["artifact_content_sha256"]
    assert _load_artifact().artifact_version == _golden()["artifact_version"]


# ------------------------------------------------ v5 版本纪律（HC37/38/39）----

def test_golden_carries_v5_version_triple_and_matches_module_constants():
    g = load_golden_stats(STATS_JSON)          # 版本/环境不符会在此 hard fail
    assert g["golden_version"] == GOLDEN_VERSION
    assert g["schema_version"] == GOLDEN_SCHEMA_VERSION == "5.0"
    assert g["quality_metric_version"] == GOLDEN_QUALITY_METRIC_VERSION


def test_golden_has_no_g8_and_g5_is_none_not_aggregated():
    """HC37/38：G5=None（不聚合、不补分）；G8 字段不存在。"""
    art = _load_artifact()
    assert art.reprojection_status == "not_available"
    assert art.g5_reproj_err_median is None and art.g5_reproj_err_p95 is None
    assert not hasattr(art.quality, "g8_bbox_coverage_min")
    # overall_quality 等于活动指标聚合器对该夹具的返回值（分母不含 G5）
    from skill3d.reconstruction_gate.quality_metrics import overall_from_metrics

    assert art.quality.overall_quality == pytest.approx(
        overall_from_metrics(art.quality))


def test_mixing_legacy_golden_hard_fails():
    """HC39：旧 golden（archive_v4）与 v5 不可比 → hard fail，不自动重算。"""
    legacy_stats = ARCHIVE_DIR / "data" / "golden_stats.json"
    assert legacy_stats.is_file(), "旧 golden 必须归档保留（只读）"
    with pytest.raises(GoldenVersionError):
        load_golden_stats(legacy_stats)
    with pytest.raises(GoldenVersionError):
        assert_not_legacy_golden(legacy_stats)
    import json as _json

    archive = _json.loads((ARCHIVE_DIR / "ARCHIVE.json").read_text(encoding="utf-8"))
    assert archive["incomparable_with_v5"] is True
    assert "g8_bbox_coverage_min" in archive["deprecated_fields_present"]


def test_wrong_golden_version_is_rejected(tmp_path):
    """版本三元组任一不符 → hard fail（防止把 v5 数字套到别的口径上）。"""
    g = dict(_golden())
    g["golden_version"] = "v4-golden"
    p = tmp_path / "golden_stats.json"
    p.write_text(json.dumps(g), encoding="utf-8")
    with pytest.raises(GoldenVersionError):
        load_golden_stats(p)
    g2 = dict(_golden(), quality_metric_version="v4-legacy")
    p2 = tmp_path / "golden_stats2.json"
    p2.write_text(json.dumps(g2), encoding="utf-8")
    with pytest.raises(GoldenVersionError):
        load_golden_stats(p2)
    g3 = dict(_golden(), env_versions={"scipy": "0.0.0", "numpy": "0.0.0"})
    p3 = tmp_path / "golden_stats3.json"
    p3.write_text(json.dumps(g3), encoding="utf-8")
    with pytest.raises(GoldenVersionError):
        load_golden_stats(p3)


def test_frozen_artifact_arrays_load_and_are_consistent():
    """冻结产物自洽：c2w 为 SE(3)、深度为正、点云与深度反投影一致。"""
    art = _load_artifact()
    c2w = np.load(art.c2w_list)
    depth = np.load(art.depth_maps)
    pmap = np.load(art.point_map)
    k = np.load(art.intrinsics)
    assert c2w.shape[1:] == (4, 4) and depth.shape == pmap.shape[:3]
    for m in c2w:
        assert np.isclose(np.linalg.det(m[:3, :3]), 1.0, atol=1e-9)
    assert np.all(depth > 0)
    # 反投影一致性：把点云再投回像素应与像素坐标一致（半像素内）
    w2c = np.linalg.inv(c2w[0])
    cam = pmap[0] @ w2c[:3, :3].T + w2c[:3, 3]
    uv = (k[0] @ (cam / cam[..., 2:3]).reshape(-1, 3).T).T.reshape(cam.shape)[..., :2]
    vv, uu = np.meshgrid(np.arange(depth.shape[1]), np.arange(depth.shape[2]),
                         indexing="ij")
    assert np.abs(uv[..., 0] - uu).max() <= 0.5 + 1e-9
    assert np.abs(uv[..., 1] - vv).max() <= 0.5 + 1e-9
    assert art.scale_known and art.scale_confidence == "high"


# ------------------------------------------------------- 幂等重放（字节级）----

def _artifact_in(tmp_path) -> str:
    """把夹具 artifact 的数组 ref 改写为绝对路径并落到 tmp（夹具保持可移植）。"""
    art = _load_artifact()
    out = tmp_path / "artifact.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(art.model_dump_json(), encoding="utf-8")
    return str(out)


def _textured_pixels(frame_shape, n_frames: int, seed: int = 0):
    """确定性有纹理帧（均匀灰帧的 Laplacian 方差为 0，会被 M2 判整体不合格）。"""
    rng = np.random.default_rng(seed)
    return [rng.integers(0, 255, (*frame_shape, 3), dtype=np.uint8)
            for _ in range(n_frames)]


def _run_frozen(tmp_path, *, baseline: str, skills=None, seed: int = 0):
    """在冻结 artifact 上跑一条 episode（确定性地复用同一 artifact）。"""
    from skill3d.schemas import InputFrame, VSIBenchEpisode

    art = _load_artifact()
    depth = np.load(art.depth_maps)
    frames = [InputFrame(frame_idx=i, timestamp=float(i), width=depth.shape[2],
                         height=depth.shape[1], blur_var=200.0,
                         overexposed_ratio=0.0, underexposed_ratio=0.0,
                         quality_ok=True) for i in range(depth.shape[0])]
    episode = VSIBenchEpisode(
        qa_id="golden-qa-1", scene_name=art.scene_name, dataset="scannet",
        # object_counting：stub program 为 `len(scene.list_objects())`，
        # 在冻结 artifact（无 SAM2 对象）上也能确定性执行 → 适合做重放基线
        question_type="object_counting",
        question="How many table(s) are in this room?",
        options=None, ground_truth="0", frames=frames, split="inner_validation")
    pixels = _textured_pixels(depth.shape[1:], depth.shape[0], seed=0)
    cfg = OnlineRunConfig(mode="mock_light", baseline=baseline, seed=seed,
                          deterministic_replay=True,
                          reuse_artifact=_artifact_in(tmp_path),
                          skills=list(skills or []), trace_dir=str(tmp_path / "traces"),
                          memory_dir="")
    return run_episode(episode, pixels, cfg, geometry=None)


def test_deterministic_replay_is_byte_identical_across_runs(tmp_path):
    """同 seed 重放：两次运行的 trace 字节级一致（§4 M17 验收）。"""
    a1 = _run_frozen(tmp_path / "r1", baseline="C1_tools_program")
    a2 = _run_frozen(tmp_path / "r2", baseline="C1_tools_program")
    assert a1.states == a2.states
    assert a1.final_state == a2.final_state
    assert json.dumps(a1.episode_trace.model_dump(), sort_keys=True) == \
        json.dumps(a2.episode_trace.model_dump(), sort_keys=True)
    assert a1.receipts_ok and a2.receipts_ok
    # receipt 哈希链逐条一致（重放确定性）
    assert [r.receipt_hash for r in a1.receipts] == [r.receipt_hash for r in a2.receipts]


def test_frozen_artifact_is_reused_not_rebuilt(tmp_path):
    """A/B 同源：reuse_artifact 路径下 M3 不重建（硬约束 18）。"""
    out = _run_frozen(tmp_path, baseline="C1_tools_program")
    assert any("复用既有 artifact" in n for n in out.notes)
    assert out.scene_route is not None


# ------------------------------------------------------- paired A/B 编排 ----

def test_paired_ab_requires_same_artifact_and_snapshot(tmp_path):
    """硬约束 18：A/B 必须同 snapshot、同 reconstruction_artifact_ref。"""
    art = _load_artifact()
    snap = build_snapshot(
        reconstruction_artifact_ref=art.artifact_id,
        tool_registry_digest="tools-v1", memory_snapshot_ref="mem-1",
        skill_registry_snapshot_ref="skill-1", prompt_version="v1",
        code_commit="deadbeef", split_pointer="inner_validation",
        episode_set=["golden-qa-1"], seed=0, base_model_fingerprint="qwen3vl-8b")
    branches = derive_branches(snap, roles=("baseline", "candidate"))
    assert_same_reconstruction_artifact(branches[0], branches[1], snap)

    called: list[str] = []

    def _run(branch, refs):
        called.append(branch.role)
        return {"branch_id": branch.branch_id, "n": len(refs)}

    res = run_paired_ab(branches[0], branches[1], snap, ["e1", "e2"], _run)
    assert called == ["baseline", "candidate"]
    assert res["reconstruction_artifact_ref"] == art.artifact_id
    assert res["n_episodes"] == 2

    # 不同 snapshot 的分支 → 断言失败
    other = snap.model_copy(update={"snapshot_id": "snap-other"})
    bad = branches[1].model_copy(update={"snapshot_id": "snap-other"})
    with pytest.raises(PairedArtifactMismatchError):
        assert_same_reconstruction_artifact(branches[0], bad, other)


# ------------------------------------------------- paired 统计可复现性 ----

def _env_matches_golden() -> bool:
    """golden 统计数值由 scipy/numpy 版本决定；环境不符则跳过（§16.4 冻结环境）。"""
    try:
        load_golden_stats(STATS_JSON)
        return True
    except GoldenVersionError:
        return False


@pytest.mark.skipif(not _env_matches_golden(),
                    reason="golden 统计数值与冻结环境（scipy/numpy 版本）绑定")
def test_paired_score_matches_frozen_golden_values():
    """paired bootstrap CI + Wilcoxon 与冻结基线一致（统计可复现）。"""
    g = _golden()
    st = score_paired(g["scores_a"], g["scores_b"], g["task_types"], seed=g["seed"])
    assert st["delta"] == pytest.approx(g["delta"])
    assert st["ci95_lo"] == pytest.approx(g["ci95_lo"])
    assert st["ci95_hi"] == pytest.approx(g["ci95_hi"])
    assert st["wilcoxon_p"] == pytest.approx(g["wilcoxon_p"])
    assert st["slice_no_regression"] == g["slice_no_regression"]


def test_paired_score_repeated_calls_are_identical():
    """重复调用逐位一致（bootstrap 由 seed 固定，无隐藏随机源）。"""
    g = _golden()
    a = [score_paired(g["scores_a"], g["scores_b"], g["task_types"], seed=7)
         for _ in range(3)]
    assert a[0] == a[1] == a[2]
    b = score_paired(g["scores_a"], g["scores_b"], g["task_types"], seed=8)
    assert b["ci95_lo"] <= b["ci95_hi"]
    assert b["delta"] == pytest.approx(a[0]["delta"])   # 点估计与 seed 无关


def test_golden_c1_and_c0_baselines_differ_in_answer_path(tmp_path):
    """golden 双档对照：C0 直答不经沙箱，C1 走 program（§16.1）。"""
    c0 = _run_frozen(tmp_path / "c0", baseline="C0_direct_vlm")
    c1 = _run_frozen(tmp_path / "c1", baseline="C1_tools_program")
    assert c0.program is not None and c0.program.program_source == ""
    assert c1.program is not None and c1.program.program_source != ""
    assert c1.program_trace is not None and c1.program_trace.steps > 0
