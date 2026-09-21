"""HC29 尺度不确定性的**统一口径**单一事实源（v4）。

本模块只做一件事：把"尺度的相对置信区间"钉死为一种可校验的口径，任何模块
（M3 锚定 / M4 质量 / M7 逐题授权 / §7 报告 / RunManifest）都必须经它换算。

口径（硬约束 29）：

- `scale_ci_rel` 是**无量纲分数**，取值域 `[0, +∞)`，表示指定
  `confidence_level`（默认研究目标 0.90，`[TODO_CALIBRATE]`）下相对置信区间
  **半宽**；不是百分数、不是全宽、不是标准差、不是 1σ；
- `scale_ci_abs_m = metric_scale * scale_ci_rel`（单位 m）；
- 当 `metric_scale` 与 `scale_ci_rel` 均有限时，两者必须自洽，否则 fail-closed
  为 `low` 并写 `reason_code`（§4.1 v4 尺度 Schema 不变量）。

**禁止**：把百分数（21.875 表示 2187.5%）、全宽（= 2×半宽）、标准差（= 半宽/1.96）
或不同置信水平的数值混写进同一字段。检测到这种混写一律 fail-closed，绝不允许
"数值看起来更小 → 置信度更高"这种反模式（§10.5 反模式清单）。

历史字段（`relative_ci` / `scale_ci`）**只允许经带版本的迁移器读取**，且迁移结果
一律 `confidence_cap="low"`，不得用于准入（§4.1 v4 说明、§10.2）。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional, Sequence

import numpy as np

# 口径版本（写进 artifact/迁移记录，供审计确认"这批数值是哪套口径")
CI_UNIT_VERSION = "v4.0-fraction-halfwidth"
# 迁移器版本（历史字段 → v4 口径；结果只读、不可用于准入）
LEGACY_CI_MIGRATOR_VERSION = "legacy-ci-migrator-v1"
# `scale_ci_rel` 的合法上界：超过此值不可能是"分数口径的半宽"（多为百分数/绝对值误传）
CI_REL_SANITY_MAX = 100.0
# 自洽校验容差（相对）
CI_CONSISTENCY_RTOL = 1e-6


CiReasonCode = Literal[
    "ok",
    "absent",
    "nan_or_inf",
    "negative",
    "percent_like",
    "full_width_like",
    "scale_missing_or_nonpositive",
    "abs_m_inconsistent",
    "legacy_ci_inconsistent",
    "legacy_ci_unknown_source",
]


class CiUnitError(ValueError):
    """CI 口径违规（构造非法 / 不自洽）——调用方一律 fail-closed 为 low。"""


@dataclass(frozen=True)
class CiRelCheck:
    """一次 `scale_ci_rel` 校验的结果（**不抛异常**，由调用方决定降级）。"""

    value: Optional[float]      # 归一化后的分数（口径异常时为 None）
    ok: bool
    reason_code: CiReasonCode
    detail: str = ""

    def as_note(self) -> str:
        return f"scale_ci_rel={self.value} ok={self.ok} reason={self.reason_code} {self.detail}".strip()


def ci_abs_m(scale: Optional[float], ci_rel: Optional[float]) -> Optional[float]:
    """`scale_ci_abs_m = metric_scale * scale_ci_rel`（硬约束 29 的唯一换算）。

    任一入参非有限/非正（scale）时返回 None（不猜、不用 0 冒充）。
    """
    if scale is None or ci_rel is None:
        return None
    try:
        s, r = float(scale), float(ci_rel)
    except (TypeError, ValueError):
        return None
    if not (np.isfinite(s) and np.isfinite(r)) or s <= 0 or r < 0:
        return None
    return float(s * r)


def check_ci_rel(
    value: Optional[float],
    *,
    confidence_level: Optional[float] = None,
    total_width: Optional[float] = None,
) -> CiRelCheck:
    """校验并归一化一个候选 `scale_ci_rel`（fail-closed，不抛异常）。

    判定顺序：
    1. `None` → `absent`（未提供，不等于"不确定性为 0"）；
    2. 非有限 → `nan_or_inf`；负数 → `negative`；
    3. `> CI_REL_SANITY_MAX` → `percent_like`（百分数/绝对值误传）；
    4. `1 < value <= CI_REL_SANITY_MAX` 且 `confidence_level` 给定时 → `percent_like`
       （分数口径下 90% 区间的半宽超过 100% 已无意义，这种量级只可能来自百分数）；
    5. 给了 `total_width` 且 `total_width ≈ 2 * value`（容差内）→ `full_width_like`
       （我们只接受半宽，全宽会**低估**一半不确定性 → 必须拒绝）。
    """
    if value is None:
        return CiRelCheck(None, False, "absent", "未提供尺度相对 CI")
    try:
        v = float(value)
    except (TypeError, ValueError):
        return CiRelCheck(None, False, "nan_or_inf", f"无法解析为浮点: {value!r}")
    if not np.isfinite(v):
        return CiRelCheck(None, False, "nan_or_inf", f"非有限值: {value!r}")
    if v < 0:
        return CiRelCheck(None, False, "negative", f"CI 半宽不得为负: {v}")
    if v > CI_REL_SANITY_MAX:
        return CiRelCheck(None, False, "percent_like",
                          f"{v} > {CI_REL_SANITY_MAX}：不是分数口径的半宽")
    if v > 1.0 and confidence_level is not None:
        return CiRelCheck(None, False, "percent_like",
                          f"{v} 在 confidence_level={confidence_level} 下超过 100%，"
                          "疑似百分数误传（分数口径应为 0–1）")
    if total_width is not None:
        try:
            w = float(total_width)
            if np.isfinite(w) and w > 0 and abs(w - 2.0 * v) <= 1e-6 * max(w, 1.0):
                return CiRelCheck(None, False, "full_width_like",
                                  f"total_width={w} ≈ 2×{v}：传进来的是全宽而非半宽")
        except (TypeError, ValueError):
            pass
    return CiRelCheck(v, True, "ok")


def ci_consistency_status(
    scale: Optional[float],
    ci_rel: Optional[float],
    ci_abs: Optional[float],
) -> str:
    """CI 三元组的写入状态（HC29/30；写回 artifact 前的分类）。

    - `"absent"`：三者全空 → 连尺度都没锚定过，正常；
    - `"uncalibrated"`：有 `scale` 但两个 CI 都没有 → **未标定/无区间**，
      这是 v5 的常态（缺冻结校准器），不是矛盾：调用方必须降级为
      `scale_confidence="low"` + 清空 `allowed_metric_tasks`（HC30），
      **不得**因此让重建失败；
    - `"claim"`：三者都给全 → 必须过 `assert_ci_consistent` 的自洽断言。
    """
    if scale is None and ci_rel is None and ci_abs is None:
        return "absent"
    if ci_rel is None and ci_abs is None:
        return "uncalibrated"
    return "claim"


def assert_ci_consistent(
    scale: Optional[float],
    ci_rel: Optional[float],
    ci_abs: Optional[float],
    *,
    rtol: float = CI_CONSISTENCY_RTOL,
) -> None:
    """断言 `ci_abs ≈ scale * ci_rel`（§4.1 Schema 不变量）。

    - 三者全 None（未锚定）与"有 scale 但无 CI"（未标定）**都通过**：
      后者由调用方降级为 low（HC30），不是矛盾；
    - **部分提供**（只给 ci_rel 或只给 ci_abs）→ 抛 `CiUnitError`（真矛盾）；
    - 数值不自洽 / 非有限 → 抛 `CiUnitError`。
    """
    if ci_consistency_status(scale, ci_rel, ci_abs) in ("absent", "uncalibrated"):
        return
    if scale is None or ci_rel is None or ci_abs is None:
        raise CiUnitError(
            f"scale/ci_rel/ci_abs 必须同时提供或同时为空，收到 "
            f"scale={scale}, ci_rel={ci_rel}, ci_abs={ci_abs}")
    s, r, a = float(scale), float(ci_rel), float(ci_abs)
    if not (np.isfinite(s) and np.isfinite(r) and np.isfinite(a)):
        raise CiUnitError(f"scale/ci_rel/ci_abs 存在非有限值: {scale}, {ci_rel}, {ci_abs}")
    expected = ci_abs_m(s, r)
    if expected is None:
        raise CiUnitError(f"ci_abs 无法由 scale={s} × ci_rel={r} 导出")
    if abs(a - expected) > rtol * max(abs(expected), 1e-12):
        raise CiUnitError(
            f"ci_abs_m={a} 与 metric_scale×scale_ci_rel={expected} 不自洽"
            f"（容差 rtol={rtol}）——按硬约束 29 必须 fail-closed")


# ------------------------------------------------------- 历史字段迁移（只读）----

@dataclass(frozen=True)
class LegacyCiMigration:
    """历史 `relative_ci` / `scale_ci` 的只读迁移结果。

    `confidence_cap` 恒为 `"low"`：历史字段**不得用于准入**（§10.2 明文）。
    """

    scale_ci_rel: Optional[float]
    ci_abs_m: Optional[float]
    confidence_cap: Literal["low"] = "low"
    reason_code: CiReasonCode = "legacy_ci_inconsistent"
    source: str = ""
    detail: str = ""

    @property
    def usable_for_admission(self) -> bool:
        """历史迁移结果永远不可用于准入（不因数值"看起来合理"而放开）。"""
        return False


def migrate_legacy_ci(
    *,
    legacy_relative_ci: Optional[float] = None,
    legacy_scale_ci: Optional[float] = None,
    scale: Optional[float] = None,
    confidence_level: Optional[float] = None,
) -> LegacyCiMigration:
    """把历史 CI 字段迁移到 v4 口径（**只读**，结果不可用于准入）。

    历史实况（§10.2 实测）：`metric_scale=2.386, relative_ci=21.875,
    scale_ci_abs≈32.81m` 三者不能由同一公式互推 → 来源不明的历史 CI 一律
    标 `low` 并写 `reason_code="legacy_ci_inconsistent"`。

    迁移只在**能自证**时给出数值（例如 `legacy_scale_ci / scale` 有限且为正），
    但仍固定 `confidence_cap="low"`。
    """
    if legacy_relative_ci is not None:
        chk = check_ci_rel(legacy_relative_ci, confidence_level=confidence_level)
        if not chk.ok:
            return LegacyCiMigration(
                None, None, source=f"{LEGACY_CI_MIGRATOR_VERSION}:relative_ci",
                reason_code="legacy_ci_inconsistent",
                detail=f"历史 relative_ci 口径异常（{chk.reason_code}）：{chk.detail}")
        # 即便数值合法，历史来源无法核实 → 仍固定 low，只留一个"参考量"
        return LegacyCiMigration(
            chk.value, ci_abs_m(scale, chk.value),
            source=f"{LEGACY_CI_MIGRATOR_VERSION}:relative_ci",
            reason_code="legacy_ci_unknown_source",
            detail="历史 relative_ci 可解析但来源/置信水平不可考 → 仅作参考，不可准入")
    if legacy_scale_ci is not None and scale is not None:
        try:
            s, a = float(scale), float(legacy_scale_ci)
        except (TypeError, ValueError):
            return LegacyCiMigration(None, None,
                                     source=f"{LEGACY_CI_MIGRATOR_VERSION}:scale_ci",
                                     detail="历史 scale_ci / scale 无法解析")
        if np.isfinite(s) and s > 0 and np.isfinite(a) and a >= 0:
            rel = float(a / s)
            chk = check_ci_rel(rel, confidence_level=confidence_level)
            if chk.ok:
                return LegacyCiMigration(
                    chk.value, float(a),
                    source=f"{LEGACY_CI_MIGRATOR_VERSION}:scale_ci",
                    reason_code="legacy_ci_unknown_source",
                    detail=("历史 scale_ci 为绝对半宽（旧公式，约低估 9.6 倍，见 §10.5）"
                            "→ 只作参考，不可准入"))
        return LegacyCiMigration(None, None,
                                 source=f"{LEGACY_CI_MIGRATOR_VERSION}:scale_ci",
                                 detail="历史 scale_ci 与 scale 不自洽")
    return LegacyCiMigration(None, None, source=LEGACY_CI_MIGRATOR_VERSION,
                             reason_code="absent", detail="无历史 CI 字段可迁移")


# ------------------------------------------------- HC30/HC33 逐题型授权 ----
# 放在本模块（而非 scale_assessment）是为了让 **Schema 校验器**也能引用同一份策略，
# 避免"Schema 一套口径、评估器另一套口径"的双事实源；本模块不 import schemas，
# 因此不会形成循环依赖。

ConfidenceTier = Literal["high", "medium", "low"]

# 逐题型初始授权策略（§3 M7 v4，全部 `[TODO_CALIBRATE]`）：
# - object_abs_distance 仅 high（绝对距离对尺度误差最敏感）；
# - object_size_estimation / room_size_estimation medium/high 可授权
#   （medium 还必须携带区间，且 M7 再叠加 G9 等质量条件）。
TASK_MIN_CONFIDENCE: dict[str, ConfidenceTier] = {
    "object_abs_distance": "high",
    "object_size_estimation": "medium",
    "room_size_estimation": "medium",
}
_CONF_RANK: dict[str, int] = {"low": 0, "medium": 1, "high": 2}


def confidence_at_least(conf: str, need: str) -> bool:
    """置信档比较（未知档一律视为最低，fail-closed）。"""
    return _CONF_RANK.get(str(conf), 0) >= _CONF_RANK.get(str(need), 0)


def pre_authorized_metric_tasks(confidence: str,
                                ci_rel: Optional[float],
                                metric_tasks: Sequence[str] = (),
                                ) -> frozenset[str]:
    """按置信档导出**预授权**题型集合（HC33；M7 再叠加 G9 等质量条件）。

    - `object_abs_distance`：仅 `high`；
    - `object_size_estimation` / `room_size_estimation`：`medium/high`，且
      **必须携带有限区间**（`ci_rel` 有限）——"medium 必须携带区间"是 v4 明文；
    - `low` → 空集（只收回米制工具，非米制 3D 工具不受影响）。
    """
    tasks = tuple(metric_tasks) if metric_tasks else (
        "object_abs_distance", "object_size_estimation", "room_size_estimation")
    if str(confidence) == "low":
        return frozenset()
    if ci_rel is None:
        return frozenset()
    try:
        r = float(ci_rel)
    except (TypeError, ValueError):
        return frozenset()
    if not np.isfinite(r):
        return frozenset()
    return frozenset(
        t for t in tasks if confidence_at_least(str(confidence),
                                                TASK_MIN_CONFIDENCE.get(t, "high")))
