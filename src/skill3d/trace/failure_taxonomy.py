"""FailureTaxonomy 归因（§4 M13 / §5.7）：按 error_code / 几何校验结果归到 8 类。"""

from __future__ import annotations

from typing import Optional, Sequence

from skill3d.schemas import FailureTaxonomy

Category = str  # FailureTaxonomy.categories 的 Literal 值


def attribute_failure(
    episode_id: str,
    error_code: Optional[str] = None,
    geometry_violations: Optional[Sequence[str]] = None,
    answer: Optional[str] = None,
    scale_known: bool = True,
    task_requires_scale: bool = False,
    note: str = "",
) -> FailureTaxonomy:
    """确定性归因规则（可调多个类别）。"""
    cats: list[str] = []

    # error_code 归因
    if error_code in ("violation_syntax",):
        cats.append("program_syntax")
    if error_code in ("violation_policy", "violation_runtime"):
        cats.append("tool_contract")
    if error_code in ("timeout", "oom", "disk_full"):
        cats.append("tool_contract")

    # 几何校验归因
    for v in geometry_violations or []:
        if v == "reprojection":
            cats.append("coordinate")
        elif v in ("no_negative_distance", "inside_bbox"):
            cats.append("reconstruction")
        elif v == "unit_consistent":
            cats.append("scale_unknown")
        else:
            cats.append("verifier_reject")

    # 尺度未知但题目要求绝对尺度
    if task_requires_scale and not scale_known and "scale_unknown" not in cats:
        cats.append("scale_unknown")

    # 无答案
    if answer is None or str(answer).strip() == "":
        cats.append("evaluator_noanswer")

    # 去重并保持序
    seen, uniq = set(), []
    for c in cats:
        if c not in seen:
            seen.add(c)
            uniq.append(c)
    if not uniq:
        uniq = ["perception"] if error_code is None and not geometry_violations else ["verifier_reject"]

    return FailureTaxonomy(episode_id=episode_id, categories=uniq, note=note)
