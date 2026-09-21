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

from skill3d.schemas.legacy import LegacyArtifact, LegacyArtifactError
from skill3d.schemas.reconstruction import (
    LEGACY_ONLY_FIELDS,
    ReconstructionArtifact,
)

CURRENT_SCHEMA_VERSION = "5.0"
CURRENT_QUALITY_METRIC_VERSION = "v5-no-g8-g5-optional"

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

    只有 `schema_version == "5.0"` 的 JSON 才会被反序列化为 v5 artifact；
    其余一律 hard fail（含旧尺度/G8 字段、缺版本字段、非 v5 版本），
    错误信息给出可执行建议：用 `skill3d.legacy.readers` 审计并重跑 v5 pipeline。
    """
    p = Path(path)
    raw = json.loads(p.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise LegacyArtifactError(source_path=str(p), deprecated_fields=set(),
                                  detail="顶层不是 JSON object")
    detected = raw.get("schema_version")
    if detected != CURRENT_SCHEMA_VERSION:
        raise LegacyArtifactError(
            source_path=str(p),
            deprecated_fields=detect_deprecated_fields(raw),
            detail=f"schema_version={detected!r}（需要 {CURRENT_SCHEMA_VERSION!r}）")
    qmv = raw.get("quality_metric_version")
    if qmv != CURRENT_QUALITY_METRIC_VERSION:
        raise LegacyArtifactError(
            source_path=str(p), deprecated_fields=detect_deprecated_fields(raw),
            detail=(f"quality_metric_version={qmv!r}"
                    f"（需要 {CURRENT_QUALITY_METRIC_VERSION!r}）"))
    # 到这里才允许进 v5 Schema（其中的 legacy/G5 校验器会继续 fail-closed）
    return ReconstructionArtifact.model_validate(raw)


def assert_runtime_eligible(obj: Any) -> Any:
    """运行时/统计/主表入口的统一护栏：`LegacyArtifact` 一律 hard fail。"""
    if isinstance(obj, LegacyArtifact):
        raise LegacyArtifactError(
            source_path=obj.source_path,
            deprecated_fields=set(obj.deprecated_fields),
            detail="legacy 载体 eligible_for_runtime=False / eligible_for_statistics=False")
    return obj


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
