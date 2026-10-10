"""`legacy/readers`：读取 v5 之前产物的**唯一入口**（§2.1 / §4.7 / HC39）。

职责边界：

1. `read_legacy_artifact(path)` —— 把旧 JSON 解析成 `LegacyArtifact`（只读审计）。
   只保存原值 + 告警：**不猜单位、不把百分数转分数、不用互相矛盾的字段推导新值**。
2. `load_artifact_v5(path)` —— 当前运行时/路由/统计/主表加载器的**唯一** artifact
   入口。`schema_version == "5.0"` 才返回 `ReconstructionArtifact`；否则 hard fail
   （`LegacyArtifactError`），并指出需要重跑 v5 pipeline。
3. `detect_deprecated_fields(raw)` —— 识别旧尺度字段（`scale_ci` / `relative_ci`）、
   已退役 G8（`g8_bbox_coverage_min`）、`CoverageMap` 内部字段，以及"没有
   `quality_metric_version` 的旧 `overall_quality`"。

注意：本模块**不**把 `LegacyArtifact` 升格成 `ReconstructionArtifact`（§4.7 明令禁止
原地 cast）。要重新获得可准入产物，必须从原始帧重跑 v5 pipeline。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable, Optional, Union

from skill3d.schemas.legacy import (
    LegacyArtifact,
    LegacyArtifactError,
    LegacyEpisodeTrace,
    LegacyEpisodeTraceError,
)
from skill3d.schemas.reconstruction import (
    LEGACY_ONLY_FIELDS,
    ReconstructionArtifact,
)
from skill3d.schemas.trace import (
    EPISODE_TRACE_SCHEMA_VERSION,
    LEGACY_EPISODE_TRACE_SCHEMA_VERSIONS,
    EpisodeTrace,
)

CURRENT_SCHEMA_VERSION = "5.0"
CURRENT_QUALITY_METRIC_VERSION = "v5-no-g8-g5-optional"
# v6 当前 Schema（运行时**必须**能加载自己的产物；见 `load_artifact_v5` 注释）
CURRENT_V6_SCHEMA_VERSION = "6.0"
CURRENT_V6_QUALITY_METRIC_VERSION = "v6-warp-overlap-no-g5"

# 旧尺度字段（HC29/§10.2）：口径（百分数/分数、全宽/半宽、1σ/kσ）无法自证
LEGACY_SCALE_FIELDS: tuple[str, ...] = (
    "scale_ci", "relative_ci", "scale_ci_pct", "relative_ci_pct",
)
# 已退役 G8（HC38）与旧覆盖度载体
RETIRED_G8_FIELDS: tuple[str, ...] = (
    "g8_bbox_coverage_min", "bbox_coverage_ratio", "coverage_ok", "coverage_map",
)


def detect_deprecated_fields(raw: Any) -> set[str]:
    """递归识别旧字段（只看键名，不做任何值语义推断）。"""
    found: set[str] = set()
    interesting = set(LEGACY_ONLY_FIELDS) | set(LEGACY_SCALE_FIELDS) | set(RETIRED_G8_FIELDS)

    def walk(node: Any) -> None:
        if isinstance(node, dict):
            for k, v in node.items():
                if k in interesting:
                    found.add(str(k))
                walk(v)
        elif isinstance(node, list):
            for item in node:
                walk(item)

    walk(raw)
    return found


def read_legacy_artifact(path: Union[str, Path]) -> LegacyArtifact:
    """把旧产物解析为 `LegacyArtifact`（只读；绝不返回 v5 artifact）。"""
    p = Path(path)
    text = p.read_text(encoding="utf-8")
    raw = json.loads(text)
    if not isinstance(raw, dict):
        raise LegacyArtifactError(source_path=str(p), deprecated_fields=set(),
                                  detail="顶层不是 JSON object")
    deprecated = detect_deprecated_fields(raw)
    warnings: list[str] = []
    detected = raw.get("schema_version")
    if detected != CURRENT_SCHEMA_VERSION:
        warnings.append(
            f"schema_version={detected!r} ≠ {CURRENT_SCHEMA_VERSION!r} → "
            "与 v5 不可混写（HC39）")
    if "quality_metric_version" not in raw and "quality" in raw:
        warnings.append("quality 缺少 quality_metric_version → 旧 overall_quality 口径未知")
        deprecated.add("overall_quality(无 quality_metric_version)")
    if (raw.get("allowed_metric_tasks") is not None
            and raw.get("scale_calibration_id") in (None, "")):
        warnings.append("allowed_metric_tasks 存在但无 scale_calibration_id（未标定授权）")
    return LegacyArtifact(
        source_path=str(p),
        detected_schema_version=detected if isinstance(detected, str) else None,
        raw_fields=raw,
        deprecated_fields=deprecated,
        warnings=warnings,
    )


def load_artifact_v5(path: Union[str, Path]) -> ReconstructionArtifact:
    """当前 artifact 的唯一加载入口（fail-closed）。

    接受 **当前 Schema**（`schema_version == "6.0"` 且
    `quality_metric_version == "v6-warp-overlap-no-g5"`）；v5 及更早的 JSON
    一律 hard fail（含旧尺度/G8/BA 字段、缺版本字段、版本不符），错误信息给出
    可执行建议：用 `skill3d.legacy.readers` 只读审计，并重跑 v6 pipeline。

    > 命名说明：函数名保留 v5 时代的调用点签名（`runner` 的两处加载入口都读它），
    > 但**判据是当前 Schema** —— v6 迁移后运行时必须能加载自己的产物，
    > 否则 reuse/frozen artifact 路径（硬约束 18 的 A/B 同源）整条不可用。
    """
    p = Path(path)
    raw = json.loads(p.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise LegacyArtifactError(source_path=str(p), deprecated_fields=set(),
                                  detail="顶层不是 JSON object")
    detected = raw.get("schema_version")
    if detected not in (CURRENT_SCHEMA_VERSION, CURRENT_V6_SCHEMA_VERSION):
        raise LegacyArtifactError(
            source_path=str(p),
            deprecated_fields=detect_deprecated_fields(raw),
            detail=(f"schema_version={detected!r}（需要 "
                    f"{CURRENT_V6_SCHEMA_VERSION!r} 或 {CURRENT_SCHEMA_VERSION!r}）"))
    qmv = raw.get("quality_metric_version")
    if detected == CURRENT_V6_SCHEMA_VERSION:
        if qmv != CURRENT_V6_QUALITY_METRIC_VERSION:
            raise LegacyArtifactError(
                source_path=str(p), deprecated_fields=detect_deprecated_fields(raw),
                detail=(f"quality_metric_version={qmv!r}"
                        f"（{CURRENT_V6_SCHEMA_VERSION!r} 需要 "
                        f"{CURRENT_V6_QUALITY_METRIC_VERSION!r}）"))
    elif qmv != CURRENT_QUALITY_METRIC_VERSION:
        raise LegacyArtifactError(
            source_path=str(p), deprecated_fields=detect_deprecated_fields(raw),
            detail=(f"quality_metric_version={qmv!r}"
                    f"（需要 {CURRENT_QUALITY_METRIC_VERSION!r}）"))
    # 到这里才允许进 Schema（其中的 legacy/G5 校验器会继续 fail-closed）
    return ReconstructionArtifact.model_validate(raw)


def assert_runtime_eligible(obj: Any) -> Any:
    """运行时/统计/主表入口的统一护栏：`LegacyArtifact` 一律 hard fail。"""
    if isinstance(obj, LegacyArtifact):
        raise LegacyArtifactError(
            source_path=obj.source_path,
            deprecated_fields=set(obj.deprecated_fields),
            detail="legacy 载体 eligible_for_runtime=False / eligible_for_statistics=False")
    if isinstance(obj, LegacyEpisodeTrace):
        raise LegacyEpisodeTraceError(
            source_path=obj.source_path,
            detected=obj.detected_schema_version,
            detail="legacy episode trace eligible_for_runtime=False（v9 §17.2）")
    return obj


def _load_trace_payload(raw: Any, source_path: str) -> dict:
    if isinstance(raw, (str, Path)):
        p = Path(raw)
        source_path = source_path or str(p)
        raw = json.loads(p.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise LegacyEpisodeTraceError(
            source_path=source_path, detected=None, detail="顶层不是 JSON object")
    return raw


def read_legacy_episode_trace(raw: Any, *,
                              source_path: str = "") -> LegacyEpisodeTrace:
    """把旧 Schema 的 episode trace 解析为只读审计载体（不猜、不升级）。"""
    data = _load_trace_payload(raw, source_path)
    detected = data.get("schema_version")
    warnings: list[str] = []
    if detected is None:
        warnings.append("缺 schema_version → 无法判定口径（不按当前 Schema 解释）")
    elif str(detected) != str(EPISODE_TRACE_SCHEMA_VERSION):
        warnings.append(
            f"schema_version={detected!r} ≠ {EPISODE_TRACE_SCHEMA_VERSION!r} → "
            "该记录不含当前 Schema 的全部字段；缺字段不等于当时事实不存在，"
            "不得混入当前统计")
    current_only = ("round_trigger", "finalization_used", "eval_visual_fallback")
    missing = [f for f in current_only if f not in data]
    if missing:
        warnings.append(f"缺少当前字段 {missing} → 当时未记录这些事实")
    return LegacyEpisodeTrace(
        source_path=source_path,
        detected_schema_version=(str(detected) if detected is not None else None),
        raw_fields=data,
        warnings=warnings,
    )


def read_episode_trace(raw: Any, *, source_path: str = "") -> EpisodeTrace:
    """当前 episode trace 的**唯一**加载入口（fail-closed，v9 §17.2）。

    只有声明当前 Schema 身份（`EPISODE_TRACE_SCHEMA_VERSION`）的记录才被解析为
    `EpisodeTrace`。旧版本（`LEGACY_EPISODE_TRACE_SCHEMA_VERSIONS`）一律 hard fail
    并指向 `read_legacy_episode_trace` —— 直接 `model_validate` 旧 JSON 会让
    缺失字段被默认值静默补齐，把"当时没记"读成"当时不存在"。
    """
    data = _load_trace_payload(raw, source_path)
    detected = data.get("schema_version")
    if str(detected) != str(EPISODE_TRACE_SCHEMA_VERSION):
        hint = ("这是历史记录：请用 read_legacy_episode_trace 做只读审计"
                if str(detected) in LEGACY_EPISODE_TRACE_SCHEMA_VERSIONS
                else "版本不受支持")
        raise LegacyEpisodeTraceError(
            source_path=source_path, detected=(str(detected) if detected is not None else None),
            detail=f"{hint}（当前需要 {EPISODE_TRACE_SCHEMA_VERSION!r}）")
    return EpisodeTrace.model_validate(data)


def iter_artifact_files(root: Union[str, Path]) -> Iterable[Path]:
    """列出目录下的候选 artifact JSON（审计/迁移脚本用）。"""
    base = Path(root)
    if not base.exists():
        return []
    return sorted(base.rglob("*artifact*.json"))


def audit_tree(root: Union[str, Path]) -> list[LegacyArtifact]:
    """批量审计：目录下所有**非 v5** artifact 的只读清单。"""
    out: list[LegacyArtifact] = []
    for f in iter_artifact_files(root):
        try:
            raw = json.loads(Path(f).read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001 - 非 JSON 文件跳过（审计不因此中断）
            continue
        if isinstance(raw, dict) and raw.get("schema_version") == CURRENT_SCHEMA_VERSION:
            continue
        out.append(read_legacy_artifact(f))
    return out


def legacy_report(arts: Iterable[LegacyArtifact]) -> dict[str, Optional[object]]:
    """汇总审计结果（可写入归档清单）。"""
    items = list(arts)
    fields: set[str] = set()
    for a in items:
        fields |= set(a.deprecated_fields)
    return {
        "n_legacy_artifacts": len(items),
        "deprecated_fields": sorted(fields),
        "incomparable_with_v5": True,
        "sources": [a.source_path for a in items],
    }
