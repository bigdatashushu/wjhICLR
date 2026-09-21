"""§10.2 D-2 尺度 PoC 阶梯（L0–L3）：可运行、可复现的通过/否决判定。

规格来源：《系统架构4.md》§10.2「PoC 阶梯与验收」+ HC29–33。四级阶梯：

| 级别 | 内容 | 通过标准 | 资格 |
|---|---|---|---|
| **L0** | 口径单测：合法尺度与 CI 断言 `abs_m = scale × rel`；百分数/全宽/NaN/历史不一致全部 fail-closed | 全部用例 fail-closed 正确 | 无（只验证口径） |
| **L1** | 合成多锚点：注入单个离群锚点，鲁棒融合不被拉偏；多数锚点冲突时必须 low | 融合偏差在容差内；冲突必 low | 无 |
| **L2** | 标定规模消融：非重叠场景 `N∈{10,30,100}`，报误差/ρ/coverage/区间宽/冲突率/锚点触发率 | 报告完整；`N=10` 仅诊断 | **N=10 不具备升级资格** |
| **L3** | 真实 held-out：三个米制题型分别报 MRA/refusal/coverage | 满足预注册门槛 | 满足者才可加入 `allowed_metric_tasks` |

**纪律**：L0/L1 是纯合成、无外部依赖，随时可跑；L2/L3 需要 ARKitScenes 非重叠
场景的真实 GT 位姿（`TODO_USER_INPUT`），缺数据时**明确报 UNVERIFIED 并退出码 1**，
不得用合成数据冒充标定结果（§10.5 反模式）。

运行：
```bash
python -m skill3d.reconstruction.scale_poc            # 跑 L0 + L1（离线可跑）
python -m skill3d.reconstruction.scale_poc --l2-records cal.jsonl --meta data/vsi_bench_meta/test.jsonl
```
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Sequence

import numpy as np

from .metric_scale import ScaleAnchor, fuse_scale_anchors_robust
from .scale_assessment import assess_scale, grade_confidence_v4
from .scale_calibration import (
    CalibrationUnavailable,
    ConformalCalibrator,
    build_scene_id_audit,
    fit_conformal_calibrator,
    vsibench_arkitscenes_scene_ids,
)
from .scale_units import check_ci_rel, ci_abs_m, migrate_legacy_ci

# L1 验收容差（TODO_CALIBRATE）：融合尺度与真值的相对偏差上限
L1_SCALE_TOL = 0.15
# L2 规模档（§10.2）
L2_SIZES = (10, 30, 100)
# N=10 仅诊断，不具备升级资格（§10.2 明文）
L2_MIN_UPGRADE_N = 30


@dataclass
class PocResult:
    """单级 PoC 结果（可落盘为 receipt）。"""

    level: str
    name: str
    passed: bool
    verified: bool = True          # False = 缺数据/未跑（UNVERIFIED，不得当通过）
    metrics: dict = field(default_factory=dict)
    failures: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {"level": self.level, "name": self.name, "passed": self.passed,
                "verified": self.verified, "metrics": self.metrics,
                "failures": self.failures, "notes": self.notes,
                "status": ("PASS" if self.passed else
                           ("UNVERIFIED" if not self.verified else "FAIL"))}


# ---------------------------------------------------------------- L0 口径 ----

def poc_l0(confidence_level: float = 0.90) -> PocResult:
    """L0：CI 口径单测（HC29）。合法值自洽通过；四类异常全部 fail-closed。"""
    failures: list[str] = []
    metrics: dict = {}

    # ① 合法值：abs = scale × rel
    scale, rel = 2.386, 0.137
    abs_m = ci_abs_m(scale, rel)
    if abs_m is None or abs(abs_m - scale * rel) > 1e-12:
        failures.append(f"合法值未满足 abs = scale × rel（得到 {abs_m}）")
    metrics["legal_abs_m"] = abs_m

    # ② 百分数误传（21.875 表示 2187.5%）
    chk = check_ci_rel(21.875, confidence_level=confidence_level)
    if chk.ok:
        failures.append("百分数量级未被拒（21.875）")
    metrics["percent_like_reason"] = chk.reason_code

    # ③ 全宽误传（total_width = 2 × 半宽）
    chk = check_ci_rel(0.10, total_width=0.20)
    if chk.ok:
        failures.append("全宽误传未被拒（total_width ≈ 2×halfwidth）")
    metrics["full_width_reason"] = chk.reason_code

    # ④ NaN / Inf / 负值
    for bad in (float("nan"), float("inf"), -0.1):
        chk = check_ci_rel(bad, confidence_level=confidence_level)
        if chk.ok:
            failures.append(f"非法值未被拒: {bad}")
    metrics["nonfinite_negative_rejected"] = True

    # ⑤ 历史字段不一致（§10.2 实测样例）→ 固定 low、不可准入
    mig = migrate_legacy_ci(legacy_relative_ci=21.875, scale=2.386,
                            legacy_scale_ci=3.41, confidence_level=confidence_level)
    if mig.usable_for_admission or mig.confidence_cap != "low":
        failures.append("历史 CI 迁移结果被允许用于准入（必须固定 low）")
    metrics["legacy_reason"] = mig.reason_code

    # ⑥ schema 层：不自洽 artifact 必须被 fail-closed 降为 low + 清空授权
    from skill3d.schemas.reconstruction import ReconstructionArtifact
    art = ReconstructionArtifact(
        artifact_id="l0", artifact_version="v", scene_name="s", recon_method="vggt",
        c2w_list="", intrinsics="", depth_maps="", point_map="", point_conf="",
        track_list=None, metric_scale=2.386, scale_known=True, scale_ci_rel=0.5,
        scale_ci_abs_m=32.81, scale_confidence="high",
        allowed_metric_tasks={"object_abs_distance"},
        confidence={"per_point_confidence": "", "coverage_count_per_frame": ""})
    if art.scale_confidence != "low" or art.allowed_metric_tasks:
        failures.append("schema 未对不自洽 CI fail-closed（HC29）")
    metrics["schema_downgraded_to"] = art.scale_confidence

    return PocResult("L0", "口径单测（HC29）", passed=not failures,
                     metrics=metrics, failures=failures,
                     notes=["纯离线、无外部依赖；每项异常都必须 fail-closed"])


# ------------------------------------------------------------ L1 合成多锚点 ----

def _anchor(name: str, ratio: float, weight: float = 1.0,
            kind: str = "object_prior") -> ScaleAnchor:
    return ScaleAnchor(kind=kind, name=name, measured=1.0, prior_m=ratio,
                       ratio=ratio, weight=weight, rel_sigma=0.03)


def poc_l1() -> PocResult:
    """L1：合成多锚点鲁棒性（HC31）—— 单离群不拉偏；多数冲突必须 low。"""
    failures: list[str] = []
    metrics: dict = {}

    # ① 单个离群锚点（8×）：多数派 2.0 → 融合不得被拉偏
    anchors = [_anchor("door:extent_up", 2.0),
               _anchor("table:top_height", 2.0, weight=200.0),
               _anchor("ground_plane_camera_height", 2.0,
                       kind="ground_plane_camera_height"),
               _anchor("chair:extent_up", 8.0, weight=5.0)]
    f = fuse_scale_anchors_robust(anchors)
    dev = abs(f.scale - 2.0) / 2.0 if f.scale else float("inf")
    metrics["single_outlier_scale"] = f.scale
    metrics["single_outlier_dev"] = dev
    if dev > L1_SCALE_TOL:
        failures.append(f"单个离群锚点把融合拉偏（{f.scale} vs 2.0，偏差 {dev:.3f}）")
    if f.conflict:
        failures.append("单个小权重离群锚点被误报为冲突（应只标 outlier）")
    metrics["single_outlier_reasons"] = dict(f.reason_codes)

    # ② 少数派权重极大（先验很窄）也不能赢过多数派
    f2 = fuse_scale_anchors_robust([_anchor("door:extent_up", 2.0, weight=188.0),
                                    _anchor("ground_plane_camera_height", 2.0,
                                            weight=1.0,
                                            kind="ground_plane_camera_height"),
                                    _anchor("table:top_height", 3.4, weight=225.0)])
    metrics["heavy_minority_scale"] = f2.scale
    if f2.scale is None or abs(f2.scale - 2.0) / 2.0 > L1_SCALE_TOL:
        failures.append(f"大权重少数派赢过多数派（scale={f2.scale}）")

    # ③ 多数冲突（1:1 两组）→ conflict=True 且置信度必须 low
    f3 = fuse_scale_anchors_robust([_anchor("door:extent_up", 2.0, weight=100.0),
                                    _anchor("table:top_height", 4.0, weight=100.0)])
    metrics["conflict_ratio"] = f3.conflict_ratio
    if not f3.conflict:
        failures.append("多数锚点冲突未被检出")
    conf, reasons = grade_confidence_v4(
        n_accepted=max(f3.n_accepted, 3), ci_rel=0.02, conflict=f3.conflict,
        has_plane=True, has_object_anchor=True, calibration=None)
    if conf != "low":
        failures.append(f"冲突场景未落 low（得到 {conf}）")
    metrics["conflict_confidence"] = conf

    # ④ 无数据/非法锚点 → 融合失败（不是"尺度=1.0"这类默认值）
    f4 = fuse_scale_anchors_robust([])
    if f4.ok:
        failures.append("空锚点集未判为融合失败")
    metrics["empty_ok"] = f4.ok

    return PocResult("L1", "合成多锚点鲁棒融合（HC31）", passed=not failures,
                     metrics=metrics, failures=failures,
                     notes=["注入单个离群锚点不得拉偏；多数冲突必须 low（§10.2）"])


# ------------------------------------------------------- L2 标定规模消融 ----

def _load_jsonl(path: str | Path) -> list[dict]:
    rows: list[dict] = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def poc_l2(records: Sequence[dict], *, meta_path: str = "",
           confidence_level: float = 0.90,
           holdout_records: Optional[Sequence[dict]] = None) -> PocResult:
    """L2：标定规模消融 `N∈{10,30,100}`（§10.2）。

    每条 record：`{"scene_id", "scale_pred", "scale_true", "rel_ci",
    "plane_identity_ok", "anchor_fired"}`。`scene_id` 必须能证明与 VSI-Bench 的
    150 个 ARKitScenes scene 不相交（HC32）→ 交集非空直接 fail。
    """
    failures: list[str] = []
    metrics: dict = {"n_records": len(records)}

    scene_ids = [str(r.get("scene_id", "")) for r in records if r.get("scene_id")]
    excluded: list[str] = []
    if meta_path:
        try:
            excluded = vsibench_arkitscenes_scene_ids(meta_path)
        except Exception as exc:  # noqa: BLE001 - 无法证明隔离 → 不得继续
            return PocResult("L2", "标定规模消融", passed=False, verified=False,
                             failures=[f"无法读取 VSI-Bench 排除清单: {exc}"],
                             metrics=metrics)
    if not excluded:
        return PocResult(
            "L2", "标定规模消融", passed=False, verified=False, metrics=metrics,
            failures=["未提供 VSI-Bench meta（--meta）：无法证明标定集与评测集不相交"
                      "（HC32 要求可审计的 ID 映射，禁止匿名文件列表）"],
            notes=["TODO_USER_INPUT：ARKitScenes 非重叠场景 GT 位姿 + VSI-Bench meta"])

    try:
        audit = build_scene_id_audit(scene_ids or [f"cal_{i}" for i in range(len(records))],
                                     excluded_scene_ids=excluded, source="L2")
    except Exception as exc:  # noqa: BLE001 - CalibrationSplitError → hard fail
        return PocResult("L2", "标定规模消融", passed=False, metrics=metrics,
                         failures=[f"标定集隔离 hard fail: {exc}"])

    metrics["n_excluded_vsibench_scenes"] = len(excluded)
    metrics["calibration_split_hash"] = audit.calibration_split_hash[:16]

    per_size: dict[str, dict] = {}
    for n in L2_SIZES:
        subset = list(records)[:n]
        if len(subset) < n:
            per_size[str(n)] = {"available": False, "n": len(subset),
                                "upgrade_eligible": False}
            continue
        try:
            cal = fit_conformal_calibrator(
                subset, confidence_level=confidence_level, split_audit=audit,
                holdout_records=(list(holdout_records)[:n] if holdout_records else None),
                fitted_from=f"L2:N={n}")
        except CalibrationUnavailable as exc:
            per_size[str(n)] = {"available": False, "error": str(exc),
                                "upgrade_eligible": False}
            continue
        per_size[str(n)] = {
            "available": True,
            "n_calibration": cal.n_calibration,
            "median_rel_error": cal.median_rel_error,
            "spearman_rho": cal.spearman_rho,
            "empirical_coverage": cal.empirical_coverage,
            "nominal_coverage": cal.confidence_level,
            "coverage_within_tolerance": cal.coverage_within_tolerance,
            "ci_halfwidth_rel": cal.coveral_rel_halfwidth,
            "anchor_fire_rate": cal.anchor_fire_rate,
            "conflict_rate": cal.conflict_rate,
            "upgrade_eligible": bool(n >= L2_MIN_UPGRADE_N
                                     and cal.coverage_within_tolerance),
        }
    metrics["per_size"] = per_size

    if not any(v.get("available") for v in per_size.values()):
        return PocResult("L2", "标定规模消融", passed=False, verified=False,
                         metrics=metrics,
                         failures=["无可用的标定规模档（样本不足或分数不可算）"],
                         notes=["TODO_USER_INPUT：需 ≥100 个非重叠 ARKitScenes 场景的 "
                                "GT 位姿才能完成 N∈{10,30,100} 消融"])

    # N=10 仅诊断（不得据此升级）；N≥30 且覆盖达标才算有升级资格
    if per_size.get("10", {}).get("upgrade_eligible"):
        failures.append("N=10 被标为具备升级资格（§10.2 明文仅诊断）")
    metrics["upgrade_eligible_sizes"] = sorted(
        int(k) for k, v in per_size.items() if v.get("upgrade_eligible"))
    return PocResult("L2", "标定规模消融（§10.2）", passed=not failures,
                     metrics=metrics, failures=failures,
                     notes=["N=10 仅诊断；升级资格需 N≥30 且经验覆盖在容差内"])


# ------------------------------------------------------- L3 真实 held-out ----

def poc_l3(per_task_outcomes: Optional[dict] = None,
           *, baseline_2d: Optional[dict] = None,
           min_mra_delta: float = 0.0) -> PocResult:
    """L3：真实 held-out 上按题型报 MRA/refusal/coverage（§10.2）。

    `per_task_outcomes`：`{task: {"mra", "refusal_rate", "coverage", "n"}}`
    —— 需在真实 held-out 上跑出（`evaluation/scale_report.py` 产出）。
    只有满足预注册门槛（MRA 不低于 2D-only baseline、refusal 可接受）的题型才可
    加入 `allowed_metric_tasks`。
    """
    if not per_task_outcomes:
        return PocResult(
            "L3", "真实 held-out 题型验收", passed=False, verified=False,
            failures=["未提供真实 held-out 的逐题型结果"],
            notes=["TODO_USER_INPUT：需 ARKitScenes 标定完成 + 真实 GPU 端到端跑三个"
                   "米制题型；在此之前 scale_recovery 不得记 real_poc_verified"])
    failures: list[str] = []
    metrics: dict = {}
    for task, m in per_task_outcomes.items():
        entry = dict(m)
        base = (baseline_2d or {}).get(task)
        if base is not None:
            entry["baseline_2d_mra"] = base
            if m.get("mra") is not None and m["mra"] < base + min_mra_delta:
                entry["eligible"] = False
                failures.append(f"{task}: MRA {m['mra']:.3f} 低于 2D-only baseline "
                                f"{base:.3f}（§10.2：不得加入 allowed_metric_tasks）")
            else:
                entry["eligible"] = True
        else:
            entry["eligible"] = None
            failures.append(f"{task}: 缺 2D-only baseline，无法判定是否达标")
        metrics[task] = entry
    return PocResult("L3", "真实 held-out 题型验收", passed=not failures,
                     metrics=metrics, failures=failures)


# ------------------------------------------------------------------ CLI ----

def run_ladder(*, l2_records: str = "", meta_path: str = "",
               l3_outcomes: str = "", confidence_level: float = 0.90,
               out: str = "") -> int:
    results = [poc_l0(confidence_level), poc_l1()]
    if l2_records:
        try:
            records = _load_jsonl(l2_records)
        except Exception as exc:  # noqa: BLE001
            results.append(PocResult("L2", "标定规模消融", passed=False, verified=False,
                                     failures=[f"读取 l2 records 失败: {exc}"]))
        else:
            results.append(poc_l2(records, meta_path=meta_path,
                                  confidence_level=confidence_level))
    else:
        results.append(PocResult("L2", "标定规模消融", passed=False, verified=False,
                                 failures=["未提供 --l2-records"],
                                 notes=["TODO_USER_INPUT：ARKitScenes GT 位姿"]))
    if l3_outcomes:
        with Path(l3_outcomes).open("r", encoding="utf-8") as f:
            results.append(poc_l3(json.load(f)))
    else:
        results.append(PocResult("L3", "真实 held-out 题型验收", passed=False,
                                 verified=False, failures=["未提供 --l3-outcomes"],
                                 notes=["TODO_USER_INPUT：真实 GPU 端到端结果"]))

    print("=" * 78)
    print("§10.2 D-2 尺度 PoC 阶梯")
    print("=" * 78)
    for r in results:
        d = r.as_dict()
        print(f"[{d['status']:10s}] {d['level']} {d['name']}")
        for k, v in r.metrics.items():
            print(f"    {k} = {v}")
        for f_ in r.failures:
            print(f"    [fail] {f_}")
        for n in r.notes:
            print(f"    [note] {n}")
    verified = [r for r in results if r.verified and r.passed]
    unverified = [r for r in results if not r.verified]
    print("-" * 78)
    print(f"通过(已核验)={[r.level for r in verified]}  "
          f"未核验={[r.level for r in unverified]}")
    print("纪律：只通过 L0/L1 不得声称尺度能力已恢复；L2/L3 未核验前 "
          "real_poc_verified=false、paper_eligible=false（HC34）")
    if out:
        Path(out).parent.mkdir(parents=True, exist_ok=True)
        Path(out).write_text(json.dumps([r.as_dict() for r in results],
                                        ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"receipt: {out}")
    return 0 if all(r.verified and r.passed for r in results) else 1


def main(argv: Optional[Sequence[str]] = None) -> int:
    p = argparse.ArgumentParser(description="§10.2 尺度 PoC 阶梯（L0–L3）")
    p.add_argument("--l2-records", default="", help="标定记录 jsonl（含 scene_id/scale_true）")
    p.add_argument("--meta", default="data/vsi_bench_meta/test.jsonl",
                   help="VSI-Bench meta（HC32 排除清单来源）")
    p.add_argument("--l3-outcomes", default="", help="逐题型真实结果 json")
    p.add_argument("--confidence-level", type=float, default=0.90)
    p.add_argument("--out", default="", help="PoC receipt 落盘路径")
    args = p.parse_args(list(argv) if argv is not None else None)
    return run_ladder(l2_records=args.l2_records, meta_path=args.meta,
                      l3_outcomes=args.l3_outcomes,
                      confidence_level=args.confidence_level, out=args.out)


if __name__ == "__main__":  # pragma: no cover - CLI 入口
    raise SystemExit(main())
