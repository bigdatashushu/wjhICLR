"""v5 golden 夹具生成器（HC39 / §14.1-5）。

与 v4 生成器的口径差异（**必须**重生成的原因）：

1. **G8 不再存在**：`g8_bbox_coverage_min` 从夹具中删除（HC38）；
2. **G5 是条件项**：正式 `vggt` 主线无真 BA → `reprojection_status="not_available"`
   且 `g5_reproj_err_*` 为 `None`；`overall_quality` 的分母**不含** G5
   （HC37：禁止补默认分维持旧总分）；
3. 夹具显式落 `schema_version` / `quality_metric_version` / `golden_version`。

```bash
python tests/golden/v5/make_golden.py     # 一次性生成，产物入库
```

产出 `tests/golden/v5/data/`：
- `frozen_artifact.json` + `frozen_{c2w,intrinsics,depth,point_map,point_conf}.npy`
  —— 冻结重建产物（A/B 两臂共用同一份，硬约束 18）；
- `golden_stats.json` —— 冻结 paired 统计（bootstrap CI/Wilcoxon/效应量）+ 版本三元组
  + `env_versions`；
- `GOLDEN.json` —— golden 版本清单（golden_version / commit / pip hash）。

**声明**：本夹具是**合成**数据，只用于验证 A/B 机制与统计管道的确定性，
**不得**作为任何真实精度结论的证据（HC24：mock 不进论文）。
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
N_FRAMES = 32   # 与 §4 M1 的 32 帧均匀采样一致（4 帧会被 M4 的 G4 判整体不合格）
H, W = 24, 32
SCENE = "golden-scene-001-v5"


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
    from skill3d.reconstruction_gate.quality_metrics import (
        ALWAYS_ACTIVE_METRICS,
        overall_from_metrics,
    )
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
    from skill3d.adapters.frame_set import frame_set_hash as _fsh

    ids = list(range(N_FRAMES))
    # v5：G5 为 None（无真 BA），非 G5 指标全部有值 → overall 分母 = 活动指标全集
    quality = QualityMetrics(
        g1_blur_ok=1.0, g2_brightness=1.0, g3_motion_blur=0.0,
        g4_frame_count=N_FRAMES, g6_depth_var_coeff=0.0, g7_dynamic_ratio=0.0,
        g9_tracker_consistency=0.95, g10_baseline_quality=0.5, g11_scale_ci=0.02,
        overall_quality=0.0)
    quality = quality.model_copy(
        update={"overall_quality": overall_from_metrics(quality)})
    assert quality.g5_reproj_err_median is None and quality.g5_reproj_err_p95 is None

    art = ReconstructionArtifact(
        artifact_id="golden-artifact-v5", artifact_version="",
        scene_name=SCENE, recon_method="vggt",
        schema_version="5.0", quality_metric_version="v5-no-g8-g5-optional",
        # 帧集身份（硬约束 21）：冻结夹具也走同一 FrameSet 契约
        frame_ids=ids, source_frame_indices=ids,
        timestamps=[float(i) for i in ids], frame_set_hash=_fsh(ids),
        c2w_list=refs["c2w_list"], intrinsics=refs["intrinsics"],
        depth_maps=refs["depth_maps"], point_map=refs["point_map"],
        point_conf=refs["point_conf"], track_list=None,
        metric_scale=1.0, scale_known=True,
        quality_status="computed",          # 硬约束 22：质量已算 → 允许 full_3d
        # v5 HC37：无真 BA → not_available + G5=None（不得补 0.5/1.0 维持旧总分）
        reprojection_status="not_available",
        g5_reproj_err_median=None, g5_reproj_err_p95=None,
        scale_source="known_object_prior",
        quality=quality,
        confidence=ConfidenceMap(per_point_confidence=refs["point_conf"],
                                 coverage_count_per_frame=""),
        scale_confidence="high",
        scale_method="golden/frozen-v5",
        # v5（HC29/31/33）：CI 口径必须自洽（abs = scale × rel），逐题型授权显式落盘
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
    # HC38：G8 不得出现在**字段名**里（版本串 "v5-no-g8-g5-optional" 含 "g8" 字样，是
    # 退役标记本身，不算字段，故按 JSON 键名判定）
    def _all_keys(obj) -> set[str]:
        if isinstance(obj, dict):
            out = set(obj)
            for v in obj.values():
                out |= _all_keys(v)
            return out
        if isinstance(obj, list):
            out: set[str] = set()
            for v in obj:
                out |= _all_keys(v)
            return out
        return set()

    keys = _all_keys(json.loads(payload))
    assert not {k for k in keys if k.startswith("g8")}, \
        f"v5 golden 不得含 G8 字段（HC38）：{sorted(k for k in keys if k.startswith('g8'))}"
    assert set(ALWAYS_ACTIVE_METRICS) <= set(json.loads(payload)["quality"]), \
        "v5 golden 必须覆盖全部活动指标"
    return art, hashlib.sha256(payload.encode()).hexdigest()


def golden_stats(seed: int = 0) -> dict:
    """冻结 paired 统计基线（bootstrap CI + Wilcoxon + 效应量），用于稳定性回归。"""
    from skill3d.evolution.paired_score import score_paired

    scores_a = [1.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0]
    scores_b = [1.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 1.0]
    tasks = ["object_counting"] * 4 + ["room_size_estimation"] * 4
    st = score_paired(scores_a, scores_b, tasks, seed=seed)
    return {"scores_a": scores_a, "scores_b": scores_b, "task_types": tasks,
            "seed": seed, "delta": st["delta"], "ci95_lo": st["ci95_lo"],
            "ci95_hi": st["ci95_hi"], "wilcoxon_p": st["wilcoxon_p"],
            "wilcoxon_p_bonferroni": st["wilcoxon_p_bonferroni"],
            "n_comparisons": st["n_comparisons"],
            "cliffs_delta": st["cliffs_delta"], "cohens_d": st["cohens_d"],
            "degenerate": st["degenerate"],
            "slice_no_regression": st["slice_no_regression"]}


def main() -> int:
    from skill3d.evaluation.golden_v5 import (
        GOLDEN_QUALITY_METRIC_VERSION,
        GOLDEN_SCHEMA_VERSION,
        GOLDEN_VERSION,
        write_golden_manifest,
    )

    art, content_hash = build()
    stats = golden_stats()
    meta = {
        **stats,
        "golden_version": GOLDEN_VERSION,
        "schema_version": GOLDEN_SCHEMA_VERSION,
        "quality_metric_version": GOLDEN_QUALITY_METRIC_VERSION,
        "artifact_content_sha256": content_hash,
        "artifact_version": art.artifact_version,
        # golden 数值由统计库版本决定 → 与冻结环境绑定（E-1）
        "env_versions": {"scipy": _ver("scipy"), "numpy": _ver("numpy")},
        "reprojection_status": art.reprojection_status,
        "g5_reproj_err_median": art.g5_reproj_err_median,
        "notes": ("v5 夹具：合成数据，只验证 A/B 机制与统计确定性，不得作为精度结论"
                  "（HC24）；G5=None 不入 overall_quality 分母（HC37）"),
    }
    (DATA / "golden_stats.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    write_golden_manifest(DATA / "GOLDEN.json", artifact_version=art.artifact_version,
                          artifact_content_sha256=content_hash)
    print(f"v5 golden 夹具已生成: {DATA}")
    print(f"  golden_version={GOLDEN_VERSION} artifact_version={art.artifact_version} "
          f"content_sha256={content_hash[:16]}")
    print(f"  overall_quality={art.quality.overall_quality:.4f} "
          f"reprojection_status={art.reprojection_status} g5=None")
    print(f"  stats: delta={stats['delta']:+.4f} "
          f"CI=[{stats['ci95_lo']:+.4f},{stats['ci95_hi']:+.4f}] "
          f"p={stats['wilcoxon_p']}")
    return 0


def _ver(mod: str) -> str:
    try:
        import importlib

        return str(getattr(importlib.import_module(mod), "__version__", "?"))
    except Exception:  # noqa: BLE001
        return "absent"


if __name__ == "__main__":
    raise SystemExit(main())
