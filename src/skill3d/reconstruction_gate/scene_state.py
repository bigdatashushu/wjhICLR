"""M4 SceneState 构造 + 质量门禁 + 逐题 scope 派生（v6 §5.3/§6.2/§6.3）。

在线模块，硬约束 1：**严禁任何离线强模型相关依赖**（在线链无离线模型）。
重建先于 Skill 路由（硬约束 16）；Tool 只经 SceneState/SceneHandle 访问产物（硬约束 17）。

v6 的两条路由纪律（D4）：

- `scene_route` **只由 M4 质量决定**，在同一 scene 的所有 episode 间稳定；
- `question_tool_scope` 逐题派生，**只收窄不新增**；
- `scene_route` **不因逐题 metric_scale 失败而改变** —— 米制失败只把 scope 从
  `metric_enabled` 收窄到 `full_3d`，绝不把整 episode 降成 2D-only。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Sequence

import numpy as np

from skill3d.routing.task_classifier import canonical_task
from skill3d.schemas.evidence import EvidenceProfile, MetricEvidenceGateResult
from skill3d.schemas.reconstruction import (
    METRIC_TASK_TYPES,
    QualityMetrics,
    ReconstructionArtifact,
    SceneState,
)
from skill3d.tools.contract import (
    SCOPE_FALLBACK_2D_ONLY,
    SCOPE_FULL_3D,
    SCOPE_METRIC_ENABLED,
    available_artifacts_for,
    question_tool_scope_of,
)

from .evidence_profile import (
    M5EvidenceSummary,
    build_evidence_profile,
    evaluate_metric_gate,
)

# 整题门控阈值（TODO_CALIBRATE；低于则 scene_route=fallback_2d_only）
TH_OVERALL_QUALITY: float = 0.5   # TODO_CALIBRATE


def scene_route_from_quality(
    q: Optional[QualityMetrics],
    *,
    quality_status: str = "computed",
) -> str:
    """§6.2 分流规则（单一事实源，**fail-closed**，硬约束 22）。

    `quality_status != "computed"` / `quality is None` / `overall_quality` 为
    NaN 或非有限值 → **不得**停在 `full_3d`，必须落到 `fallback_2d_only`
    或 `unanswerable`。

    v6 主门（§10.1）取代 v5 的 G5 依赖：**主门通过** 且 输入质量达阈 → `full_3d`；
    主门未过 → `fallback_2d_only`。**多指标不得单挑**：主门本身已是 warp ∧ 重叠。
    """
    if q is None or quality_status != "computed":
        return "fallback_2d_only"
    if int(q.g4_frame_count) < 1:
        return "unanswerable"           # G4：无可用帧
    overall = float(q.overall_quality)
    if not np.isfinite(overall):
        return "fallback_2d_only"
    if not bool(q.main_gate_passed):
        return "fallback_2d_only"
    if overall < TH_OVERALL_QUALITY:
        return "fallback_2d_only"
    return "full_3d"


@dataclass
class QuestionScopeDecision:
    """逐题 scope 派生结果（v6 D4；替代 v5 的 QuestionGateDecision）。"""

    scene_route: str
    question_tool_scope: str
    gate: Optional[MetricEvidenceGateResult]
    allowed: bool = True
    flags: list[str] = field(default_factory=list)
    reasons: list[str] = field(default_factory=list)

    def note(self) -> str:
        if self.gate is None:
            return (f"逐题 scope 派生（scene_route={self.scene_route}，"
                    f"scope={self.question_tool_scope}，无米制门）")
        return (f"逐题 scope={self.question_tool_scope}"
                f"（scene_route={self.scene_route}，"
                f"metric_gate_passed={self.gate.gate_passed}，"
                f"missing={self.gate.missing_subconditions}）")


def _valid_frame_ratio(art: ReconstructionArtifact) -> float:
    """有效帧占比（§13.1 子条件 3 的输入）；读不到 receipt → 0.0（fail-closed）。

    receipt 由 `reconstruction.metric_fusion.write_per_frame_receipt` 落盘，
    含 `valid_frame_ratio`。读不到 = 无法证明融合覆盖足够 → 不授权米制能力。
    """
    ref = getattr(art, "per_frame_scale_ref", None)
    if not ref:
        return 0.0
    try:
        from skill3d.reconstruction.metric_fusion import read_per_frame_receipt

        rec = read_per_frame_receipt(ref)
    except Exception:  # noqa: BLE001 - 缺文件/坏 JSON → 视为无证据
        return 0.0
    try:
        return float(rec.get("valid_frame_ratio", 0.0))
    except (TypeError, ValueError):
        return 0.0


def make_metric_gate(
    art: ReconstructionArtifact,
    *,
    scene_route: str,
    question_type: str = "",
    inputs_finite: bool = True,
) -> MetricEvidenceGateResult:
    """从 artifact 的融合结果构造 §13.1 的 6 项子条件判定。"""
    q = art.quality
    main_ok = bool(getattr(q, "main_gate_passed", False)) if q is not None else False
    return evaluate_metric_gate(
        scene_route=scene_route,
        main_gate_passed=main_ok,
        fusion_status=str(getattr(art, "scale_fusion_status", "not_run")),
        scale_self_consistency=getattr(art, "scale_self_consistency", None),
        valid_frame_ratio=_valid_frame_ratio(art),
        metric_scale=getattr(art, "metric_scale", None),
        inputs_finite=inputs_finite,
        question_type=question_type,
    )


def build_scene_state(
    art: ReconstructionArtifact,
    *,
    m5: Optional[M5EvidenceSummary] = None,
    objects: Optional[Sequence[str]] = None,
    artifact_ref: str = "",
    question_type: str = "",
    metric_scale_override: Optional[str] = None,
) -> SceneState:
    """M4：artifact → SceneState（含 scene_route + 证据画像 + 米制门）。

    `question_type` 为空时按"未分类"处理：gate 的题型子条件不满足 → 米制 Tool
    一律收回（fail-closed：不知道题型就无法证明自己被授权）。
    """
    q = art.quality
    scene_route = scene_route_from_quality(
        q, quality_status=str(getattr(art, "quality_status", "not_computed")))
    gate = make_metric_gate(art, scene_route=scene_route, question_type=question_type)
    profile = build_evidence_profile(
        artifact=art, scene_route=scene_route, question_type=question_type,
        gate=gate, m5=m5,
        metric_scale_override=metric_scale_override)  # type: ignore[arg-type]

    metric_q = str(question_type) in METRIC_TASK_TYPES
    scope = question_tool_scope_of(
        scene_route, metric_question=metric_q, gate_passed=gate.gate_passed)
    avail = available_artifacts_for(scene_route, question_type=question_type,
                                    gate_passed=gate.gate_passed,
                                    metric_question=metric_q)

    summary = _scene_summary(art, scene_route=scene_route, scope=scope,
                             profile=profile, gate=gate, n_objects=len(objects or []))
    return SceneState(
        artifact_ref=artifact_ref or art.artifact_id,
        artifact=art,
        scene_route=scene_route,                       # type: ignore[arg-type]
        question_tool_scope=scope,                     # type: ignore[arg-type]
        available_artifacts=set(avail),
        evidence_profile=profile,
        metric_evidence_gate_result=gate,
        answer_source="abstain",
        objects=sorted(objects or []),
        summary=summary,
        question_type=str(question_type or ""),
        quality=q,
    )


def scope_scene_to_question(
    scene: SceneState,
    question_type: str,
    *,
    inputs_finite: bool = True,
    m5: Optional[M5EvidenceSummary] = None,
    metric_gate_override: Optional[MetricEvidenceGateResult] = None,
    evidence_profile_override: Optional[EvidenceProfile] = None,
) -> tuple[SceneState, QuestionScopeDecision]:
    """逐题派生 `question_tool_scope`（§5.3/§6.3）。

    **只收窄不新增**；`scene_route` 保持不变（§6.2：metric 失败不改 scene_route）。
    返回 `(新 SceneState, 决策)`。
    """
    try:
        task = canonical_task(question_type)
    except Exception:  # noqa: BLE001 - 未知题型：不额外门控，M7 会报错
        return scene, QuestionScopeDecision(
            scene_route=scene.scene_route,
            question_tool_scope=scene.question_tool_scope,
            gate=scene.metric_evidence_gate_result,
            allowed=scene.scene_route != "unanswerable",
            flags=["unknown_question_type"])

    if scene.scene_route == "unanswerable":
        return scene, QuestionScopeDecision(
            scene_route="unanswerable", question_tool_scope=SCOPE_FALLBACK_2D_ONLY,
            gate=scene.metric_evidence_gate_result, allowed=False,
            flags=["scene_unanswerable"],
            reasons=["场景门控已判 unanswerable（§6.2）"])

    art = getattr(scene, "artifact", None)
    gate = metric_gate_override or (
        make_metric_gate(art, scene_route=scene.scene_route,
                         question_type=task, inputs_finite=inputs_finite)
        if art is not None else scene.metric_evidence_gate_result)

    metric_q = task in METRIC_TASK_TYPES
    scope = question_tool_scope_of(
        scene.scene_route, metric_question=metric_q,
        gate_passed=bool(gate is not None and gate.gate_passed))
    avail = available_artifacts_for(
        scene.scene_route, question_type=task,
        gate_passed=bool(gate is not None and gate.gate_passed),
        metric_question=metric_q)

    profile = evidence_profile_override or (
        build_evidence_profile(
            artifact=art, scene_route=scene.scene_route, question_type=task,
            gate=gate, m5=m5 or M5EvidenceSummary()) if art is not None
        else scene.evidence_profile)

    flags: list[str] = []
    reasons: list[str] = []
    if metric_q and scope != SCOPE_METRIC_ENABLED:
        flags.append("metric_tool_withheld")
        reasons.append(
            f"题型 {task} 需要米制尺度，但米制证据门未通过"
            f"（missing={gate.missing_subconditions if gate else 'no_gate'}）"
            "→ 该题收回米制 Tool；scene_route 不变，非米制 3D Tool 保留（§6.2/D3）")

    new_scene = scene.model_copy(update={
        "question_tool_scope": scope,
        "available_artifacts": set(avail),
        "evidence_profile": profile,
        "metric_evidence_gate_result": gate,
        "question_type": task,
    })
    if new_scene.summary:
        new_scene = new_scene.model_copy(update={
            "summary": _scene_summary(
                art, scene_route=new_scene.scene_route, scope=scope,
                profile=profile, gate=gate,
                n_objects=len(new_scene.objects)) if art is not None
            else new_scene.summary})
    return new_scene, QuestionScopeDecision(
        scene_route=new_scene.scene_route, question_tool_scope=scope,
        gate=gate, allowed=True, flags=flags, reasons=reasons)


def _scene_summary(art, *, scene_route: str, scope: str,
                   profile: Optional[EvidenceProfile],
                   gate: Optional[MetricEvidenceGateResult],
                   n_objects: int) -> str:
    """场景摘要：**与 prompt 头部同源**（都从 scene_route + scope 派生，§5.3）。

    历史缺陷（v5 实测）：摘要写 `route=full_3d` 而 prompt 头部写 fallback，
    自相矛盾会让模型在错误的前提下编排 Tool。这里两者都由同一对字段生成。
    """
    q = getattr(art, "quality", None)
    if q is None:
        return (f"scene={getattr(art, 'scene_name', '')} "
                f"scene_route={scene_route} question_tool_scope={scope} "
                f"quality_status={getattr(art, 'quality_status', 'not_computed')} "
                "（质量未计算 → 无几何证据）")

    ev = ("；".join(f"{c}={profile.state(c)}" for c in
                    ("geometry_3d", "world_frame", "metric_scale",
                     "object_detection", "track_consensus", "object_grounding"))
          if profile is not None else "无证据画像")
    wf = getattr(art, "world_frame_status", "unavailable")
    return (
        f"scene={art.scene_name} method={art.recon_method} "
        f"scene_route={scene_route} question_tool_scope={scope} "
        f"main_gate_passed={bool(q.main_gate_passed)} "
        f"warp_inlier={float(q.warp_inlier_ratio):.3f} "
        f"cloud_overlap={float(q.cloud_overlap_ratio):.3f} "
        f"overall_quality={float(q.overall_quality):.3f} "
        f"world_frame={wf} metric_scale_status={getattr(art, 'scale_fusion_status', 'not_run')} "
        f"metric_gate={'passed' if (gate and gate.gate_passed) else 'not_passed'} "
        f"n_objects={n_objects} evidence=[{ev}] "
        f"frame_set_hash={str(art.frame_set_hash)[:12]}"
    )


def quality_gate(
    art: ReconstructionArtifact,
    *,
    m5: Optional[M5EvidenceSummary] = None,
    objects: Optional[Sequence[str]] = None,
    artifact_path: Optional[str] = None,
    persist_quality: bool = True,
    question_type: str = "",
) -> SceneState:
    """M4 入口（保留 v5 名称以便平滑迁移）：artifact → SceneState。

    与 v5 的差异：不再有 G7/G9 增量补写（诊断项不影响主门）与 `scale_is_usable`
    中间量；质量一旦 `computed` 即复用（硬约束 22 单一事实源）。
    """
    from .quality_metrics import compute_and_store_quality

    if str(getattr(art, "quality_status", "not_computed")) != "computed":
        art = compute_and_store_quality(
            art, artifact_path=(artifact_path if persist_quality else None))
    return build_scene_state(art, m5=m5, objects=objects,
                             artifact_ref=artifact_path or art.artifact_id,
                             question_type=question_type)


__all__ = [
    "TH_OVERALL_QUALITY",
    "QuestionScopeDecision",
    "build_scene_state",
    "make_metric_gate",
    "quality_gate",
    "scene_route_from_quality",
    "scope_scene_to_question",
]
