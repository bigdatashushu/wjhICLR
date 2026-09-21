"""尺度经验的 **conformal 校准器 + 标定集隔离审计**。

**v5.1 口径修订（用户 2026-09-20 拍板，覆盖 v5 HC32 的"指定 ARKitScenes"表述）**：

标定池不再绑定单一数据集，而是"**按被评测数据集选择同源非重叠标定池**"：

- 硬门不变：标定集 / conformal 留出集 / 被评测场景集，三者按原始 scene ID
  **两两互斥**，交集非空即 hard fail，拒绝生成 `scale_calibration_id`；
- 被排除集合按数据集取（`scannet` / `scannetpp` / `arkitscenes`），从 VSI-Bench meta
  读出该数据集实际被评测的全部 scene ID；
- **同源优先**：标定数据集与被评测数据集一致时记 `dataset_match=True`（覆盖保证的
  转移论证最直接）；不一致时允许但记 `False` 并要求在论文里显式论证转移性；
- 支持**多校准器并存**：每个数据集一个冻结校准器（`<dataset>.json`），
  在线按被评测数据集选用；将来扩到全量 288 scene 时把三个数据集的排除集并起来即可。

本模块负责三件事，全部 fail-closed：

1. **标定集隔离**：按原始 scene/video ID 做集合差（见上）；审计清单持久化
   （`SceneIdAudit`），`calibration_split_hash` / `excluded_vsibench_scene_hash`
   与 `dataset` / `dataset_match` 进 RunManifest（§7）。
2. **冻结校准器**：split-conformal，非一致性分数 = log 尺度误差
   `|log s_pred − log s_true|`；分位 `q̂ = quantile(ceil((n+1)·level)/n)` 给出有限样本
   覆盖保证。`apply()` 返回的 `scale_ci_rel = sinh(q̂)`（= 相对半宽），并在解析 CI
   更宽时取较大者（校准**只会加宽、不会收窄**不确定性）。
3. **覆盖证据**：名义 coverage 与经验 coverage 都要记录，供 medium/high 升级门槛
   判定（§10.2）。

**在线纪律**：测试时只加载冻结的 `scale_calibration_id`，**绝不读取** GT 位姿、
GT 深度或逐场景尺度；本模块不提供任何 oracle 附录路径（§10.2 / §10.5）。

数据依赖：ARKitScenes 非评测场景的 GT 位姿属 `TODO_USER_INPUT`；缺数据时
`fit_conformal_calibrator` 无从构造，`load_calibrator` 一律返回不可用 →
`scale_confidence` 恒 `low`（这正是当前实况，不得误报为已标定）。
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Optional, Sequence

import numpy as np

from .scale_units import CI_UNIT_VERSION

# VSI-Bench 官方规模（§1.2 已核验）：ARKitScenes 150 / ScanNet 88 / ScanNet++ 50
VSI_BENCH_ARKIT_SCENES = 150
VSI_BENCH_SCANNET_SCENES = 88
VSI_BENCH_SCANNETPP_SCENES = 50
_ARKIT_DATASET_NAMES = ("arkitscenes", "arkit_scenes", "arkitscene")
# 数据集规范名 → meta 里可能的取值别名（大小写不敏感）
DATASET_ALIASES: dict[str, tuple[str, ...]] = {
    "arkitscenes": _ARKIT_DATASET_NAMES,
    "scannet": ("scannet",),
    "scannetpp": ("scannetpp", "scannet_pp", "scannetplusplus"),
}
EXPECTED_VSIBENCH_SCENE_COUNTS: dict[str, int] = {
    "arkitscenes": VSI_BENCH_ARKIT_SCENES,
    "scannet": VSI_BENCH_SCANNET_SCENES,
    "scannetpp": VSI_BENCH_SCANNETPP_SCENES,
}

# 经验覆盖容差（§10.2：名义 90% 的 empirical coverage 须在预设容差内接近 90%）
COVERAGE_TOLERANCE = 0.10          # TODO_CALIBRATE
# 冻结校准器文件 schema 版本（版本不符 → 视为不可用，不静默兼容）
CALIBRATOR_SCHEMA_VERSION = "scale-conformal-v1"
# 非一致性分数的下限（避免全零分数给出 0 宽度区间）
MIN_NONCONFORMITY = 1e-3           # TODO_CALIBRATE


class CalibrationSplitError(RuntimeError):
    """标定集与评测集相交（HC32：hard fail，拒绝生成 scale_calibration_id）。"""


class CalibrationUnavailable(RuntimeError):
    """冻结校准器缺失 / 版本不符 / 未标定（调用方一律降级为 low）。"""


# ------------------------------------------------------------ 标定集隔离审计 ----

@dataclass(frozen=True)
class SceneIdAudit:
    """标定集 / conformal 校准集 vs VSI-Bench 评测 scene 的 ID 级隔离审计（HC32）。

    三个集合两两必须不相交：

    1. `calibration_scene_ids`：拟合锚点先验/解析不确定性的标定池；
    2. `conformal_scene_ids`：拟合 conformal 分位数的留出集（与 1 不相交才有
       独立覆盖保证；为空表示"未单独留出"，经验覆盖只能自评——已在 note 里标注）；
    3. `excluded_vsibench_scene_ids`：VSI-Bench 实际使用的 150 个 ARKitScenes scene。

    任一交集非空 → `assert_disjoint` 抛 `CalibrationSplitError`（hard fail，
    拒绝生成 `scale_calibration_id`）。
    """

    calibration_scene_ids: tuple[str, ...]
    excluded_vsibench_scene_ids: tuple[str, ...]
    conformal_scene_ids: tuple[str, ...] = ()
    source: str = ""
    dataset: str = "arkitscenes"
    # v5.1：被评测数据集集合（用于"同源优先"判定与论文口径说明）
    evaluation_datasets: tuple[str, ...] = ()

    @property
    def dataset_match(self) -> bool:
        """标定数据集是否与被评测数据集同源（同源 → 覆盖转移论证最直接）。"""
        ev = {str(d).strip().lower() for d in self.evaluation_datasets}
        return (not ev) or (str(self.dataset).strip().lower() in ev)

    def intersection(self) -> frozenset[str]:
        """任一跨集合交集（calibration/conformal × excluded，以及两者之间）。"""
        cal = frozenset(self.calibration_scene_ids)
        conf = frozenset(self.conformal_scene_ids)
        excl = frozenset(self.excluded_vsibench_scene_ids)
        return (cal & excl) | (conf & excl) | (cal & conf)

    def overlap_detail(self) -> dict[str, list[str]]:
        """交集的来源明细（审计用：谁和谁撞了）。"""
        cal = frozenset(self.calibration_scene_ids)
        conf = frozenset(self.conformal_scene_ids)
        excl = frozenset(self.excluded_vsibench_scene_ids)
        return {
            "calibration_vs_excluded": sorted(cal & excl),
            "conformal_vs_excluded": sorted(conf & excl),
            "calibration_vs_conformal": sorted(cal & conf),
        }

    def assert_disjoint(self) -> None:
        """任一交集非空 → hard fail（硬约束 32：存在交集即 hard fail）。"""
        detail = self.overlap_detail()
        bad = sorted({s for ids in detail.values() for s in ids})
        if bad:
            who = "；".join(f"{k}={v[:5]}" for k, v in detail.items() if v)
            raise CalibrationSplitError(
                f"标定/校准集与 VSI-Bench 评测 scene 相交（{len(bad)} 个）：{who}"
                "——按硬约束 32 拒绝生成 scale_calibration_id（hard fail）")

    @property
    def calibration_split_hash(self) -> str:
        return _hash_ids(self.calibration_scene_ids)

    @property
    def conformal_split_hash(self) -> str:
        return _hash_ids(self.conformal_scene_ids)

    @property
    def excluded_vsibench_scene_hash(self) -> str:
        return _hash_ids(self.excluded_vsibench_scene_ids)

    def as_manifest(self) -> dict:
        return {
            "schema_version": CALIBRATOR_SCHEMA_VERSION,
            "dataset": self.dataset,
            "evaluation_datasets": list(self.evaluation_datasets),
            "dataset_match": self.dataset_match,
            "source": self.source,
            "n_calibration_scenes": len(self.calibration_scene_ids),
            "n_conformal_scenes": len(self.conformal_scene_ids),
            "n_excluded_vsibench_scenes": len(self.excluded_vsibench_scene_ids),
            "calibration_split_hash": self.calibration_split_hash,
            "conformal_split_hash": self.conformal_split_hash,
            "excluded_vsibench_scene_hash": self.excluded_vsibench_scene_hash,
            "intersection": sorted(self.intersection()),
            "intersection_is_empty": not self.intersection(),
            "overlap_detail": self.overlap_detail(),
            "calibration_scene_ids": list(self.calibration_scene_ids),
            "conformal_scene_ids": list(self.conformal_scene_ids),
            "excluded_vsibench_scene_ids": list(self.excluded_vsibench_scene_ids),
        }


def _hash_ids(ids: Iterable[str]) -> str:
    payload = json.dumps(sorted({str(i) for i in ids}), sort_keys=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def vsibench_scene_ids_by_dataset(meta_path: str | Path, dataset: str) -> list[str]:
    """从 VSI-Bench meta jsonl 读出**指定数据集**被评测的全部 scene ID。

    这是"排除清单"的唯一真实来源：要求按原始 scene/video ID 做集合差，而不是
    "匿名文件列表"（§13.1 要求可审计的 ID 映射）。`dataset` 取 `scannet` /
    `scannetpp` / `arkitscenes`（大小写不敏感，含别名）。
    """
    key = str(dataset).strip().lower()
    aliases = DATASET_ALIASES.get(key)
    if aliases is None:
        raise CalibrationSplitError(
            f"未知数据集 {dataset!r}；已注册：{sorted(DATASET_ALIASES)}")
    p = Path(meta_path)
    if not p.is_file():
        raise CalibrationSplitError(
            f"VSI-Bench meta 不存在: {p}——无法证明标定集与评测集不相交（hard fail）")
    ids: set[str] = set()
    with p.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if str(row.get("dataset", "")).strip().lower() not in aliases:
                continue
            sid = str(row.get("scene_name") or row.get("video_id") or "").strip()
            if sid:
                ids.add(sid)
    if not ids:
        raise CalibrationSplitError(
            f"{p} 中未找到 {key} 的 scene（dataset 取值 {aliases}）"
            "——排除清单不完整，拒绝标定")
    return sorted(ids)


def vsibench_arkitscenes_scene_ids(meta_path: str | Path) -> list[str]:
    """（保留）ARKitScenes 排除清单；等价于 `vsibench_scene_ids_by_dataset(..., "arkitscenes")`。"""
    return vsibench_scene_ids_by_dataset(meta_path, "arkitscenes")


def build_scene_id_audit(
    calibration_scene_ids: Iterable[str],
    *,
    excluded_scene_ids: Optional[Iterable[str]] = None,
    meta_path: Optional[str | Path] = None,
    conformal_scene_ids: Optional[Iterable[str]] = None,
    source: str = "",
    dataset: str = "arkitscenes",
    evaluation_datasets: Optional[Iterable[str]] = None,
    expected_excluded_count: Optional[int] = None,
    check_expected_count: bool = True,
) -> SceneIdAudit:
    """构造并校验标定集隔离审计（任一交集非空 → `CalibrationSplitError`）。

    v5.1：`dataset` 指定标定池来源数据集；`excluded_scene_ids` 缺省时按
    `evaluation_datasets`（缺省 = `{dataset}`）从 meta 读出被评测场景集合并取并集。
    `check_expected_count=True` 时校验各数据集规模与官方一致（避免"排除清单不完整"
    导致隔离形同虚设——这类错误会让 hard fail 失效）。

    `conformal_scene_ids` 为拟合 conformal 分位数的留出集：与标定池、与评测集
    都必须不相交。
    """
    ev_datasets = sorted({str(d).strip().lower()
                          for d in (evaluation_datasets or [dataset]) if str(d).strip()})
    if excluded_scene_ids is None:
        if meta_path is None:
            raise CalibrationSplitError(
                "必须提供 excluded_scene_ids 或 meta_path 之一，"
                "否则无法证明标定集与被评测 scene 不相交")
        excluded: set[str] = set()
        for ds in ev_datasets:
            ids = vsibench_scene_ids_by_dataset(meta_path, ds)
            if check_expected_count:
                want = EXPECTED_VSIBENCH_SCENE_COUNTS.get(ds)
                if want is not None and len(ids) != want:
                    raise CalibrationSplitError(
                        f"{ds} 排除清单数量 {len(ids)} ≠ 官方 {want}"
                        "——排除清单不完整，隔离审计不可信（hard fail）")
            excluded |= set(ids)
        excluded_list = sorted(excluded)
    else:
        excluded_list = sorted({str(i) for i in excluded_scene_ids})
        if expected_excluded_count is not None \
                and len(excluded_list) != int(expected_excluded_count):
            raise CalibrationSplitError(
                f"排除清单数量 {len(excluded_list)} ≠ 期望 {expected_excluded_count}")
    audit = SceneIdAudit(
        calibration_scene_ids=tuple(sorted({str(i) for i in calibration_scene_ids})),
        excluded_vsibench_scene_ids=tuple(excluded_list),
        conformal_scene_ids=tuple(sorted({str(i) for i in (conformal_scene_ids or [])})),
        source=source,
        dataset=str(dataset).strip().lower(),
        evaluation_datasets=tuple(ev_datasets),
    )
    audit.assert_disjoint()
    return audit


def write_audit_manifest(audit: SceneIdAudit, path: str | Path) -> Path:
    """持久化审计清单（§7：标定清单与被排除清单都要留档）。"""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(out.suffix + ".tmp")
    tmp.write_text(json.dumps(audit.as_manifest(), ensure_ascii=False, indent=2),
                   encoding="utf-8")
    tmp.replace(out)
    return out


# ------------------------------------------------------------ conformal 校准 ----

def conformal_quantile(scores: Sequence[float], level: float) -> float:
    """split-conformal 分位数 `ceil((n+1)·level)/n`（有限样本覆盖保证）。

    `level=0.90, n=30` → `ceil(27.9)/30 = 28/30` 分位。分数为空 → 抛
    `CalibrationUnavailable`（不得返回 0 宽度）。
    """
    s = np.asarray([x for x in scores if np.isfinite(x)], dtype=np.float64)
    if s.size == 0:
        raise CalibrationUnavailable("非一致性分数为空，无法校准")
    n = int(s.size)
    k = int(math.ceil((n + 1) * float(level)))
    k = min(max(k, 1), n)
    return float(np.sort(s)[k - 1])


@dataclass(frozen=True)
class ConformalCalibrator:
    """冻结的尺度 conformal 校准器（在线只读加载，绝不重拟合）。"""

    calibration_id: str
    confidence_level: float
    quantile: float                       # log 尺度非一致性分数的 conformal 分位
    n_calibration: int
    empirical_coverage: float
    median_rel_error: float
    spearman_rho: float
    anchor_fire_rate: float = float("nan")
    conflict_rate: float = float("nan")
    # v5.1：标定池来源数据集与被评测数据集（多校准器并按数据集选用的依据）
    calibration_dataset: str = ""
    evaluation_datasets: tuple[str, ...] = ()
    split_audit: Optional[SceneIdAudit] = None
    unit_version: str = CI_UNIT_VERSION
    schema_version: str = CALIBRATOR_SCHEMA_VERSION
    method: str = "split_conformal_log_scale"
    fitted_from: str = ""
    notes: list[str] = field(default_factory=list)

    # ---- 应用 ----
    @property
    def coveral_rel_halfwidth(self) -> float:
        """conformal 分位对应的**相对半宽**（log 对称区间 → sinh 形式）。

        `|log s_pred − log s_true| <= q̂` ⟺ `s_pred/s_true ∈ [e^-q̂, e^q̂]`
        → 相对半宽 = `(e^q̂ − e^-q̂)/2 = sinh(q̂)`。
        """
        return float(math.sinh(max(float(self.quantile), MIN_NONCONFORMITY)))

    def apply(self, scale: Optional[float], ci_rel_analytic: Optional[float]
              ) -> tuple[Optional[float], Optional[float]]:
        """把解析 CI 校准为**有经验覆盖保证**的相对半宽。

        校准**只会加宽、不会收窄**：取 `max(解析 CI, sinh(q̂))`。理由：解析传播
        （`(σ_s/s)² ≈ (σ_h/h)² + (σ_anchor/h_anchor)²`）不具经验覆盖保证；若解析值
        更宽说明该场景本身更不确定（如锚点冲突），收窄它等于用外部统计抹掉
        场景内证据。
        """
        if scale is None or not np.isfinite(float(scale)) or float(scale) <= 0:
            return None, None
        cal = self.coveral_rel_halfwidth
        analytic = float(ci_rel_analytic) if (ci_rel_analytic is not None
                                              and np.isfinite(ci_rel_analytic)) else 0.0
        rel = float(max(cal, analytic))
        return float(scale), float(rel)

    @property
    def coverage_within_tolerance(self) -> bool:
        """经验覆盖是否在名义值的容差内（§10.2 medium/high 的必要条件）。"""
        return (np.isfinite(self.empirical_coverage)
                and abs(self.empirical_coverage - self.confidence_level)
                <= COVERAGE_TOLERANCE)

    # ---- 序列化（冻结产物：在线只读）----
    @property
    def dataset_match(self) -> bool:
        """标定数据集是否与被评测数据集同源（论文口径需要显式记录）。"""
        ev = {str(d).strip().lower() for d in self.evaluation_datasets}
        return (not ev) or (str(self.calibration_dataset).strip().lower() in ev)

    def to_json(self) -> str:
        payload = {
            "schema_version": self.schema_version,
            "calibration_id": self.calibration_id,
            "confidence_level": self.confidence_level,
            "quantile": self.quantile,
            "n_calibration": self.n_calibration,
            "empirical_coverage": self.empirical_coverage,
            "median_rel_error": self.median_rel_error,
            "spearman_rho": self.spearman_rho,
            "anchor_fire_rate": self.anchor_fire_rate,
            "conflict_rate": self.conflict_rate,
            # v5.1：标定数据集与被评测数据集（多校准器按数据集选用；论文口径说明）
            "calibration_dataset": self.calibration_dataset,
            "evaluation_datasets": list(self.evaluation_datasets),
            "dataset_match": self.dataset_match,
            "unit_version": self.unit_version,
            "method": self.method,
            "fitted_from": self.fitted_from,
            "notes": list(self.notes),
            "split_audit": self.split_audit.as_manifest() if self.split_audit else None,
        }
        return json.dumps(payload, ensure_ascii=False, indent=2)

    @classmethod
    def from_dict(cls, data: dict) -> "ConformalCalibrator":
        ver = str(data.get("schema_version", ""))
        if ver != CALIBRATOR_SCHEMA_VERSION:
            raise CalibrationUnavailable(
                f"校准器 schema 版本不符: {ver!r} != {CALIBRATOR_SCHEMA_VERSION!r}"
                "（不静默兼容，一律视为不可用）")
        unit = str(data.get("unit_version", ""))
        if unit != CI_UNIT_VERSION:
            raise CalibrationUnavailable(
                f"校准器 CI 口径版本不符: {unit!r} != {CI_UNIT_VERSION!r}"
                "（HC29：禁止把不同口径的 CI 混用）")
        audit = None
        raw_audit = data.get("split_audit")
        if raw_audit:
            audit = SceneIdAudit(
                calibration_scene_ids=tuple(raw_audit.get("calibration_scene_ids") or ()),
                excluded_vsibench_scene_ids=tuple(
                    raw_audit.get("excluded_vsibench_scene_ids") or ()),
                conformal_scene_ids=tuple(raw_audit.get("conformal_scene_ids") or ()),
                source=str(raw_audit.get("source", "")),
                dataset=str(raw_audit.get("dataset", "arkitscenes")),
                evaluation_datasets=tuple(raw_audit.get("evaluation_datasets") or ()),
            )
            # 反序列化后仍然强制隔离（防止被篡改的校准器绕过 HC32）
            audit.assert_disjoint()
        return cls(
            calibration_id=str(data["calibration_id"]),
            confidence_level=float(data["confidence_level"]),
            quantile=float(data["quantile"]),
            n_calibration=int(data.get("n_calibration", 0)),
            empirical_coverage=float(data.get("empirical_coverage", float("nan"))),
            median_rel_error=float(data.get("median_rel_error", float("nan"))),
            spearman_rho=float(data.get("spearman_rho", float("nan"))),
            anchor_fire_rate=float(data.get("anchor_fire_rate", float("nan"))),
            conflict_rate=float(data.get("conflict_rate", float("nan"))),
            calibration_dataset=str(data.get("calibration_dataset", "")),
            evaluation_datasets=tuple(data.get("evaluation_datasets") or ()),
            split_audit=audit,
            unit_version=unit,
            schema_version=ver,
            method=str(data.get("method", "split_conformal_log_scale")),
            fitted_from=str(data.get("fitted_from", "")),
            notes=list(data.get("notes") or []),
        )


def fit_conformal_calibrator(
    records: Sequence[dict],
    *,
    confidence_level: float,
    split_audit: SceneIdAudit,
    calibration_id: str = "",
    holdout_records: Optional[Sequence[dict]] = None,
    fitted_from: str = "",
    calibration_dataset: str = "",
    evaluation_datasets: Optional[Sequence[str]] = None,
) -> ConformalCalibrator:
    """从标定记录拟合冻结校准器（§10.2；**标定构建期**调用，不在线调用）。

    每条 record：`{"scale_pred": float, "scale_true": float, "rel_ci": float,
    "plane_identity_ok": bool|None, "anchor_fired": bool|None}`。

    非一致性分数 = `|log(scale_pred / scale_true)|`（对尺度这类乘性量取 log 才有
    对称误差语义）。`scale_true` 由**GT 位姿 pairwise 平移幅值比**给出
    （`_pairwise_scale_error`），不用深度 Umeyama 对齐——那会吸收尺度、自证循环。

    经验覆盖：给了 `holdout_records` 就在留出集上算（诚实）；否则用标定集自评
    并在 notes 里注明"in-sample，乐观偏置"。
    """
    split_audit.assert_disjoint()   # HC32：先过隔离门，再谈校准
    scores: list[float] = []
    rel_errs: list[float] = []
    cis: list[float] = []
    plane_bad = 0
    fired = 0
    for rec in records:
        err = scale_error_ratio(rec.get("scale_pred"), rec.get("scale_true"))
        if err is None:
            continue
        rel_errs.append(abs(err - 1.0))
        cis.append(float(rec.get("rel_ci") or 0.0))
        scores.append(max(abs(math.log(err)), MIN_NONCONFORMITY))
        if rec.get("plane_identity_ok") is False:
            plane_bad += 1
        if rec.get("anchor_fired"):
            fired += 1
    if not scores:
        raise CalibrationUnavailable("标定记录无可用的尺度误差样本（scale_pred/scale_true 缺失）")

    q = conformal_quantile(scores, confidence_level)
    notes: list[str] = []
    if holdout_records:
        hold_scores = []
        for rec in holdout_records:
            err = scale_error_ratio(rec.get("scale_pred"), rec.get("scale_true"))
            if err is not None:
                hold_scores.append(max(abs(math.log(err)), MIN_NONCONFORMITY))
        if hold_scores:
            empirical = float(np.mean(np.asarray(hold_scores) <= q))
        else:
            empirical = float("nan")
            notes.append("留出集无有效样本 → 经验覆盖不可算")
    else:
        empirical = float(np.mean(np.asarray(scores) <= q))
        notes.append("经验覆盖为 in-sample 自评（乐观偏置）；升级判定应用留出集")
    if not split_audit.conformal_scene_ids:
        notes.append("未单独留出 conformal 场景集（HC32 建议标定池与校准集互斥）；"
                     "经验覆盖的独立性弱于留出集口径")

    rho = _spearman(cis, rel_errs) if len(scores) >= 3 else float("nan")
    n_total = len(records)
    cal_id = calibration_id or f"scale-conf-{confidence_level:.2f}-n{len(scores)}-" \
                               f"{split_audit.calibration_split_hash[:12]}"
    return ConformalCalibrator(
        calibration_id=cal_id,
        confidence_level=float(confidence_level),
        quantile=float(q),
        n_calibration=len(scores),
        empirical_coverage=empirical,
        median_rel_error=float(np.median(rel_errs)),
        spearman_rho=float(rho),
        anchor_fire_rate=(fired / n_total) if n_total else float("nan"),
        conflict_rate=(plane_bad / n_total) if n_total else float("nan"),
        split_audit=split_audit,
        calibration_dataset=str(calibration_dataset or split_audit.dataset),
        evaluation_datasets=tuple(evaluation_datasets or split_audit.evaluation_datasets),
        fitted_from=fitted_from,
        notes=notes,
    )


def scale_error_ratio(scale_pred, scale_true) -> Optional[float]:
    """`scale_pred / scale_true`（两个标量或两条 c2w 序列均可，返回 None = 不可算）。"""
    if scale_pred is None or scale_true is None:
        return None
    if isinstance(scale_pred, (int, float)) and isinstance(scale_true, (int, float)):
        try:
            p, t = float(scale_pred), float(scale_true)
        except (TypeError, ValueError):
            return None
        if not (np.isfinite(p) and np.isfinite(t)) or p <= 0 or t <= 0:
            return None
        return float(p / t)
    # 序列形态：退回 GT 位姿 pairwise 平移幅值比（尺度无关参照）
    from .metric_scale import _pairwise_scale_error

    return _pairwise_scale_error(scale_pred, scale_true)


def _spearman(x: Sequence[float], y: Sequence[float]) -> float:
    from .metric_scale import _spearman as _sp

    return _sp(x, y)


def save_calibrator(cal: ConformalCalibrator, path: str | Path) -> Path:
    """落盘冻结校准器（原子写；在线只读加载）。"""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(out.suffix + ".tmp")
    tmp.write_text(cal.to_json(), encoding="utf-8")
    tmp.replace(out)
    return out


def load_calibrator(path: Optional[str | Path]) -> ConformalCalibrator:
    """在线加载冻结校准器；任何问题（缺失/损坏/版本不符/隔离门不过）→ 抛异常。

    调用方必须捕获并降级为 `scale_confidence="low"` + `allowed_metric_tasks=∅`
    ——**不允许**"加载失败就当作没约束"（HC30 fail-closed）。
    """
    if path in (None, ""):
        raise CalibrationUnavailable("未配置 scale_calibration_path（TODO_USER_INPUT）")
    p = Path(path)
    if not p.is_file():
        raise CalibrationUnavailable(f"冻结校准器不存在: {p}")
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001 - 损坏文件一律不可用
        raise CalibrationUnavailable(f"冻结校准器无法解析: {p}: {exc}") from exc
    return ConformalCalibrator.from_dict(data)


# ---------------------------------------------------- 多校准器（v5.1）----

def calibrator_path_for(dataset: str, root: str | Path) -> Path:
    """多校准器目录约定：`<root>/<dataset>.json`（每个被评测数据集一份）。"""
    return Path(root) / f"{str(dataset).strip().lower()}.json"


def load_calibrator_for(datasets: Sequence[str], root: str | Path,
                        fallback: Optional[str | Path] = None
                        ) -> tuple[Optional[ConformalCalibrator], list[str]]:
    """按被评测数据集加载校准器（多个数据集 → 用第一个可加载的，并记录来源）。

    返回 `(calibrator, notes)`；全部不可用 → `(None, notes)`（调用方降级 low）。
    纪律：**同源优先**——先找与数据集同名的文件；找不到才退回 `fallback`
    （例如全局 `calibrator.json`），并在 notes 里写明"非同源，需要转移性论证"。
    """
    notes: list[str] = []
    root_p = Path(root)
    for ds in datasets:
        p = calibrator_path_for(ds, root_p)
        try:
            cal = load_calibrator(p)
        except CalibrationUnavailable as exc:
            notes.append(f"{ds}: {exc}")
            continue
        notes.append(f"{ds}: 已加载 {p}（calibration_id={cal.calibration_id}，"
                     f"dataset={cal.calibration_dataset or '未标注'}，"
                     f"dataset_match={cal.dataset_match}）")
        return cal, notes
    if fallback not in (None, ""):
        try:
            cal = load_calibrator(fallback)
            notes.append(f"退回 fallback={fallback}（非同源 → 论文需论证覆盖转移）")
            return cal, notes
        except CalibrationUnavailable as exc:
            notes.append(f"fallback 不可用: {exc}")
    return None, notes


def calibrator_available(path: Optional[str | Path]) -> bool:
    """校准器是否可用（不抛异常的探针；用于报告与 gate 判定）。"""
    try:
        load_calibrator(path)
        return True
    except CalibrationUnavailable:
        return False
