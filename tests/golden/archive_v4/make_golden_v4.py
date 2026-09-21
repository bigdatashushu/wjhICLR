"""G-02 golden 夹具生成器：冻结 A/B 用的 ReconstructionArtifact 与统计基线。

一次性运行（结果入库，供 `tests/golden/evolution/` 断言）：

```bash
python tests/golden/evolution/make_golden.py
```

产出 `tests/golden/evolution/data/`：
- `frozen_artifact.json` + `frozen_{c2w,intrinsics,depth,point_map,point_conf}.npy`
  —— 冻结的重建产物（A/B 两臂**共用**同一份，硬约束 18）；
- `golden_stats.json` —— 冻结的 paired 统计（bootstrap CI / Wilcoxon p）与
  `golden_hash`（artifact 内容哈希，防止夹具被悄悄改动）。
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[3]
if str(_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_ROOT / "src"))

DATA = Path(__file__).resolve().parent / "data"

# 冻结的合成场景参数（确定性；改这里等于改夹具，需重新生成 golden）
N_FRAMES = 32   # 与 §4 M1 的 32 帧均匀采样一致（4 帧会被 M2 的 G4 判整体不合格）
H, W = 24, 32
SCENE = "golden-scene-001"


def _c2w() -> np.ndarray:
    out = np.zeros((N_FRAMES, 4, 4))
    for i in range(N_FRAMES):
        out[i, :3, :3] = np.eye(3)
        out[i, :3, 3] = [0.1 * i, 1.5, 0.0]
        out[i, 3, 3] = 1.0
    return out


def _intrinsics() -> np.ndarray:
    k = np.array([[4.0, 0.0, W / 2 - 0.5], [0.0, 4.0, H / 2 - 0.5], [0.0, 0.0, 1.0]])
    return np.broadcast_to(k, (N_FRAMES, 3, 3)).copy()


def _depth() -> np.ndarray:
    return np.full((N_FRAMES, H, W), 3.0)


def _point_map(depth: np.ndarray, c2w: np.ndarray, k: np.ndarray) -> np.ndarray:
    t = depth.shape[0]
    out = np.zeros((t, *depth.shape[1:], 3))
    for i in range(t):
        k_inv = np.linalg.inv(k[i])
        for v in range(H):
            for u in range(W):
                cam = k_inv @ np.array([u, v, 1.0]) * depth[i, v, u]
                out[i, v, u] = c2w[i][:3, :3] @ cam + c2w[i][:3, 3]
    return out


def build(seed: int = 0):
    from skill3d.schemas import ConfidenceMap, QualityMetrics, ReconstructionArtifact
    from skill3d.schemas.reconstruction import ScaleAnchorEvidence

    c2w, k, depth = _c2w(), _intrinsics(), _depth()
    pmap = _point_map(depth, c2w, k)
    pconf = np.full((N_FRAMES, H, W), 0.9)

    DATA.mkdir(parents=True, exist_ok=True)

    def _save(name: str, arr: np.ndarray) -> str:
        p = DATA / f"frozen_{name}.npy"
        np.save(p, arr)
        return str(p.name)

    refs = {"c2w_list": _save("c2w", c2w), "intrinsics": _save("intrinsics", k),
            "depth_maps": _save("depth", depth), "point_map": _save("point_map", pmap),
            "point_conf": _save("point_conf", pconf)}
    nan = float("nan")
    from skill3d.adapters.frame_set import frame_set_hash as _fsh

    ids = list(range(N_FRAMES))
    art = ReconstructionArtifact(
        artifact_id="golden-artifact", artifact_version="",
        scene_name=SCENE, recon_method="vggt",
        # 帧集身份（硬约束 21）：冻结夹具也走同一 FrameSet 契约
        frame_ids=ids, source_frame_indices=ids,
        timestamps=[float(i) for i in ids], frame_set_hash=_fsh(ids),
        c2w_list=refs["c2w_list"], intrinsics=refs["intrinsics"],
        depth_maps=refs["depth_maps"], point_map=refs["point_map"],
        point_conf=refs["point_conf"], track_list=None,
        metric_scale=1.0, scale_known=True,
        quality_status="computed",          # 硬约束 22：质量已算 → 允许 full_3d
        g5_reproj_err_median=0.5, g5_reproj_err_p95=1.0,
        scale_source="known_object_prior",
        quality=QualityMetrics(
            g1_blur_ok=1.0, g2_brightness=1.0, g3_motion_blur=0.0,
            g4_frame_count=N_FRAMES, g5_reproj_err_median=0.5, g5_reproj_err_p95=1.0,
            g6_depth_var_coeff=0.0, g7_dynamic_ratio=0.0, g8_bbox_coverage_min=0.9,
            g9_tracker_consistency=0.95, g10_baseline_quality=0.5,
            g11_scale_ci=0.02, overall_quality=0.9),
        confidence=ConfidenceMap(per_point_confidence=refs["point_conf"],
                                 coverage_count_per_frame=""),
        scale_ci=0.02, scale_confidence="high",
        scale_method="golden/frozen",
        # v4（HC29/31/33）：CI 口径必须自洽（abs = scale × rel），逐题型授权显式落盘。
        # 夹具同样走 v4 契约，不靠旧字段走捷径（否则会被 HC29 fail-closed 降为 low）。
        scale_ci_rel=0.02, scale_ci_abs_m=0.02, scale_confidence_level=0.90,
        scale_calibration_id="golden-frozen-calibrator",
        allowed_metric_tasks={"object_abs_distance", "object_size_estimation",
                              "room_size_estimation"},
        scale_anchor_fired=[
            ScaleAnchorEvidence(
                anchor_type="camera_height_floor", scale_estimate=1.0, ci_rel=0.02,
                residual=0.0, accepted=True, reason_code="ok",
                anchor_name="ground_plane_camera_height", measured=1.5, prior_m=1.5,
                weight=1.0, note="golden 冻结夹具锚点（合成，非实测）"),
            ScaleAnchorEvidence(
                anchor_type="table", scale_estimate=1.0, ci_rel=0.01, residual=0.001,
                accepted=True, reason_code="ok", anchor_name="table:top_height",
                measured=0.75, prior_m=0.75, weight=1.0,
                note="golden 冻结夹具锚点（合成，非实测）"),
        ],
    )
    payload = art.model_dump_json()
    version = hashlib.sha256(payload.encode()).hexdigest()[:16]
    art = art.model_copy(update={"artifact_version": version})
    payload = art.model_dump_json()
    (DATA / "frozen_artifact.json").write_text(payload, encoding="utf-8")
    return art, hashlib.sha256(payload.encode()).hexdigest()


def _env_versions() -> dict:
    """记录影响统计数值的库版本（scipy/numpy）。"""
    out = {}
    for mod in ("scipy", "numpy"):
        try:
            import importlib

            out[mod] = str(getattr(importlib.import_module(mod), "__version__", "?"))
        except Exception:  # noqa: BLE001
            out[mod] = "absent"
    return out


def golden_stats(seed: int = 0) -> dict:
    """冻结 paired 统计基线（bootstrap CI + Wilcoxon），用于稳定性回归。"""
    from skill3d.evolution.paired_score import score_paired

    scores_a = [1.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0]
    scores_b = [1.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 1.0]
    tasks = ["object_counting"] * 4 + ["room_size_estimation"] * 4
    st = score_paired(scores_a, scores_b, tasks, seed=seed)
    return {"scores_a": scores_a, "scores_b": scores_b, "task_types": tasks,
            "seed": seed, "delta": st["delta"], "ci95_lo": st["ci95_lo"],
            "ci95_hi": st["ci95_hi"], "wilcoxon_p": st["wilcoxon_p"],
            # §7：多重比较校正 + 效应量也纳入冻结基线
            "wilcoxon_p_bonferroni": st["wilcoxon_p_bonferroni"],
            "n_comparisons": st["n_comparisons"],
            "cliffs_delta": st["cliffs_delta"], "cohens_d": st["cohens_d"],
            "degenerate": st["degenerate"],
            "slice_no_regression": st["slice_no_regression"]}


def main() -> int:
    art, content_hash = build()
    stats = golden_stats()
    (DATA / "golden_stats.json").write_text(
        json.dumps({**stats, "artifact_content_sha256": content_hash,
                    "artifact_version": art.artifact_version,
                    # golden 数值由统计库版本决定 → 与冻结环境绑定（§16.4）
                    "env_versions": _env_versions()},
                   ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"golden 夹具已生成: {DATA}")
    print(f"  artifact_version={art.artifact_version} content_sha256={content_hash[:16]}")
    print(f"  stats: delta={stats['delta']:+.4f} "
          f"CI=[{stats['ci95_lo']:+.4f},{stats['ci95_hi']:+.4f}] "
          f"p={stats['wilcoxon_p']} cliff={stats['cliffs_delta']:+.3f} "
          f"d={stats['cohens_d']:+.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
