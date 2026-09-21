"""M4 SceneState 构造与质量门禁分流（§4 M4 伪代码）。

在线模块，严禁任何 GPT-6 相关依赖（硬约束 1）。
重建先于 Skill 路由（硬约束 16）；Tool 只经 SceneState 句柄访问产物（硬约束 17）。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Optional, Sequence

import numpy as np

from skill3d.reconstruction_gate import quality_metrics as qm
from skill3d.tools.contract import available_artifacts_for
from skill3d.reconstruction_gate.confidence_map import (
    coverage_ratio,
    fuse_confidence,
    per_object_coverage,
)
from skill3d.routing.task_classifier import (
    MEASUREMENT_TASKS,
    canonical_task,
)
from skill3d.schemas.reconstruction import (
    ConfidenceMap,
    QualityMetrics,
    ReconstructionArtifact,
    SceneState,
)

# 整体质量阈值（TODO_CALIBRATE，§4 M4 字段 7：低于则 route=fallback_2d_only）
TH_OVERALL_QUALITY: float = 0.5   # TODO_CALIBRATE
# v5 HC29：相对 CI 半宽（分数口径）的可用上限，与 metric_scale.MAX_USABLE_REL_CI 同源。
# 旧绝对字段 TH_SCALE_CI（m）已按 HC39 移除：口径不可互推，禁止再作回退。
TH_SCALE_CI_REL: float = 0.40     # TODO_CALIBRATE
# G-11 验收：只有 high/medium 档才允许 measurement 题按 metric 作答
ACCEPTED_SCALE_CONFIDENCE: tuple[str, ...] = ("high", "medium")


def route_from_quality(q: Optional[QualityMetrics], *,
                       quality_status: str = "computed") -> str:
    """§4 M4 分流规则（单一事实源，**fail-closed**，硬约束 22）。

    `quality_status != "computed"` / `quality is None` / `overall_quality` 为
    NaN 或非有限值 → **不得**停在 `full_3d`，必须落到 `fallback_2d_only`
    或 `unanswerable`（旧伪代码 `if q.overall_quality < TH` 在 NaN 下恒为 False，
    会把坏场景误判成 full_3d，这里堵死该路径）。

    **route 只由质量决定（Appendix A route 总纲）**：尺度不参与 route 判定。
    尺度只走两条**逐题**通道：M7 硬过滤（`metric_scale_required` →
    `scale_confidence ∈ {medium, high}`）与 `question_gate` 的 G-11 measurement
    降级。把"尺度不可用"提到全局 route 上会把 MCA 题（rel_direction /
    route_planning / appearance_order，都不需要 metric 尺度）也一起砍成 2D-only
    —— 未标定前 scale 恒 low，等于整条链退化成 2D（且违反 M6"存疑时判 full_3d、
    误删工具的代价是整题不可答"）。
    """
    if q is None or quality_status != "computed":
        return "fallback_2d_only"   # 质量未计算 → 保守受限 Tool 集
    if q.g4_frame_count < 1:
        return "unanswerable"       # G4：无可用帧
    overall = float(q.overall_quality)
    if not np.isfinite(overall):
        return "fallback_2d_only"   # NaN/Inf 一律不得停 full_3d
    if overall < TH_OVERALL_QUALITY:
        return "fallback_2d_only"
    return "full_3d"


def scale_is_usable(scale_known: bool, scale_confidence: Optional[str],
                    *, allowed_metric_tasks: Optional[Iterable[str]] = None,
                    scale_ci_rel: Optional[float] = None) -> bool:
    """G-11/v5 尺度可用性（单一事实源）。

    v5 口径（HC29/30/33）：
    - `allowed_metric_tasks` 给定时，**它说了算**：至少授权一个米制题型才算"尺度可用"
      （这是逐题型授权的落地口径；空集 = 所有米制 Tool/Skill 都该被收回）；
    - 否则退回 `scale_known` × CI 门 × 置信档，`None` 一律视为**不可用**
      ——"没有置信档"不等于"置信度没问题"（fail-closed，§3 M7）；
    - 只用 `scale_ci_rel`（分数口径）比较；旧绝对字段 `scale_ci` 已按 HC39 移除，
      **不得**再作为回退读入。
    """
    if not scale_known:
        return False
    if allowed_metric_tasks is not None:
        return bool({str(t) for t in allowed_metric_tasks})
    if scale_ci_rel is not None and np.isfinite(scale_ci_rel):
        if scale_ci_rel > TH_SCALE_CI_REL:
            return False
    if scale_confidence not in ACCEPTED_SCALE_CONFIDENCE:
        return False
    return True


@dataclass
class QuestionGateDecision:
    """逐题门控结果（v4 HC33 逐题型米制授权；附录 A G8 已删除）。"""

    route: str                       # full_3d | fallback_2d_only | unanswerable
    allowed: bool                    # False = 该题拒答（unanswerable）
    flags: list[str] = field(default_factory=list)
    reasons: list[str] = field(default_factory=list)
    # v4 HC33：本题的米制题型授权集合（可能被 medium 附加条件进一步收窄）
    allowed_metric_tasks: set[str] = field(default_factory=set)

    def note(self) -> str:
        if not self.flags:
            return (f"逐题门控通过（route={self.route}；"
                    f"授权米制题型={sorted(self.allowed_metric_tasks)}）")
        return f"逐题门控 route={self.route} flags={self.flags} reasons={self.reasons}"


def metric_tasks_after_quality_gates(
    task: str,
    pre_authorized: Iterable[str],
    *,
    confidence: str = "low",
    g9_tracker_consistency: Optional[float] = None,
    min_track_iou: float = qm.TH_G9_TRACK_IOU,
    plane_quality_ok: Optional[bool] = None,
) -> set[str]:
    """在尺度预授权之上叠加**题型相关质量条件**（§3 M7 v5 逐题型授权策略）。

    - `object_size_estimation`：`medium` 档**必须 G9 达标**（跟踪一致性 ≥ 独立标定
      阈值）。原附加条件含 G8 包围盒覆盖，但 **G8 已永久退役**（HC38），该条件不存在，
      且**不得**用任何指标无审计地替代；
    - `room_size_estimation`：`medium` 档必须**平面身份/拟合质量达标**。该判据的三态
      输入 `plane_quality_ok` 必须显式给定（`TODO_CALIBRATE` 阈值尚未标定）：
      `None`（证据缺失）与 `False` 一律视为未达标 → **fail-closed 不授权**。
      v5 **不设置** `geometric_coverage` 门，`coverage_gate_status` 只写 `not_defined`；
    - `high` 档不叠加本节条件 —— §10.2 已用**更严格且预先冻结**的误差/CI/coverage/
      最小样本门槛在标定期把关，再叠一遍会与"high 门槛更严"的预注册口径冲突；
    - 其余题型不需要米制尺度 → 返回空集（不因尺度为 low 失去非尺度 3D Tool）。
    """
    allowed = {str(t) for t in pre_authorized}
    if task not in allowed:
        return set()
    if str(confidence) == "high":
        return {task}
    if task == "object_size_estimation":
        g9_ok = (g9_tracker_consistency is not None
                 and np.isfinite(g9_tracker_consistency)
                 and g9_tracker_consistency >= min_track_iou)
        if not g9_ok:
            return set()
    if task == "room_size_estimation" and plane_quality_ok is not True:
        return set()
    return {task}


def question_gate(
    scene: SceneState,
    question_type: str,
    *,
    g9_tracker_consistency: Optional[float] = None,
    plane_quality_ok: Optional[bool] = None,
) -> QuestionGateDecision:
    """逐题门控（v5 HC33 逐题型米制授权 + HC38 无 coverage 门）。

    **G8 已永久退役**（HC38）：不得存在 `coverage_ok=True` 常量、不得用其他指标
    替代 `geometric_coverage` 门。v5 的 trace 只记录 `coverage_gate_status="not_defined"`，
    报告也不得写 "coverage passed"。

    v5 HC33 逐题型授权（**本函数不改变 route**，§3"尺度能力门控"已确认设计：
    "`scale_confidence=low` 时只移除 `scale` artifact 与米制工具，不得改变原本合格的
    `full_3d` route"）：

    - `object_counting` 与 4 个 MCA 题型**不需要**米制尺度 → 尺度为 low 也**不**
      过滤其非尺度 3D Tool；
    - `object_abs_distance` / `object_size_estimation` / `room_size_estimation`
      依据 `scene.allowed_metric_tasks` 逐题门控：未授权 → `allowed_metric_tasks` 记空集，
      米制 Tool 由**逐题产物收窄**（`route_artifacts_for_question` 去掉 `scale`）+
      `docs()` 裁剪 + 执行期 `ConfidenceGateError` 三重 fail-closed 收回。

    历史缺陷（2026-09-21 实测修出）：本函数此前在这三个题型上把 route 改成
    `fallback_2d_only`，而 `fallback_2d_only` 的产物集只有 `{frames, intrinsics}` →
    连带砍掉 `depth/poses/point_cloud/objects`，`object_centroid`/`list_objects`/
    `reproject` 等**非米制 3D Tool 全部消失**（outer_holdout 12/32 题的 prompt 里
    只剩一个 `euclidean_distance`），prompt 头部的 route 还与场景摘要自相矛盾
    （摘要写 `route=full_3d`）。这既违反 §3 的已确认设计，也让"程序路径"在这些题上
    直接退化为不可答。
    """
    try:
        task = canonical_task(question_type)
    except Exception:  # noqa: BLE001 - 未知题型不额外门控，交由 M7 报错
        return QuestionGateDecision(route=scene.route, allowed=scene.route != "unanswerable")

    flags: list[str] = []
    reasons: list[str] = []
    route = scene.route

    if scene.route == "unanswerable":
        return QuestionGateDecision(route="unanswerable", allowed=False,
                                    flags=["scene_unanswerable"],
                                    reasons=["场景门控已判 unanswerable（§4 M4）"])

    pre_authorized = {str(t) for t in (getattr(scene, "allowed_metric_tasks", None) or set())}
    allowed_tasks = metric_tasks_after_quality_gates(
        task, pre_authorized,
        confidence=str(getattr(scene, "scale_confidence", "low") or "low"),
        g9_tracker_consistency=g9_tracker_consistency,
        plane_quality_ok=plane_quality_ok,
    )

    if task in MEASUREMENT_TASKS and task not in allowed_tasks:
        flags.append("v5_metric_task_not_authorized")
        if pre_authorized:
            reasons.append(
                f"题型 {task} 虽在尺度预授权内，但未过附加质量条件"
                f"（pre_authorized={sorted(pre_authorized)}，"
                f"g9={g9_tracker_consistency}，plane_quality_ok={plane_quality_ok}）"
                "→ 该题收回米制 Tool（HC33；route 不变，非米制 3D Tool 保留）")
        else:
            reasons.append(
                f"题型 {task} 需要米制尺度，但本次尺度评估未授权任何米制题型"
                f"（confidence={scene.scale_confidence}，ci_rel={scene.scale_ci_rel}，"
                f"conflict={scene.scale_conflict}）"
                "→ 该题收回米制 Tool（HC33；route 不变，其余题型与非米制 3D Tool 不受影响）")

    return QuestionGateDecision(route=route, allowed=True, flags=flags, reasons=reasons,
                                allowed_metric_tasks=allowed_tasks)


def quality_gate(
    art: ReconstructionArtifact,
    *,
    frames: Optional[Sequence[np.ndarray]] = None,
    depth_maps: Optional[np.ndarray] = None,
    reproj_errors: Optional[np.ndarray] = None,
    dynamic_masks: Optional[np.ndarray] = None,
    track_ious: Optional[Sequence[float]] = None,
    c2w_list: Optional[np.ndarray] = None,
    object_point_indices: Optional[dict[str, np.ndarray]] = None,
    object_ids: Optional[Sequence[str]] = None,
    artifact_ref: str = "",
    artifact_path: Optional[str] = None,
    persist_quality: bool = True,
) -> SceneState:
    """按 §4 M4 伪代码：算 G1-G11 → **写回 artifact** → 融置信度 → SceneState → 分流。

    G-18 数据源接线：`reproj_errors`（G5，来自 BA 残差）、`track_ious`（G9，
    来自 M5 SAM2）、`dynamic_masks`（G7，来自 M5 刚性残差）
    由调用方传入；缺失时为 NaN，不伪造。

    硬约束 22（质量单一事实源）：
    - 入参 artifact 若已 `quality_status=="computed"` → 直接复用（P1 方案 X，
      P2 零重算）；
    - 否则按方案 Y 实算并把结果**原子写回** `artifact_path`（给定路径时），
      返回的 SceneState 上是写回后的新 artifact（不可变语义）；
    - route 判定一律 fail-closed（NaN/未计算 → 不得 full_3d）。

    M5 增量补写（D-5）：P1（方案 X）跑质量时 M5 还没跑，G7/G9 必然 NaN。
    P2 拿到 M5 统计后就地补进同一份 `quality` 并原子写回，否则这三项永久为空。
    `persist_quality=False` 用于 paired A/B 的 frozen artifact 复用（硬约束 18：
    两臂必须读同一 artifact，不得在 A/B 之间改动它）——此时只补进内存副本。

    口径说明：artifact 的 quality 是**场景级**（P1 一次写定），而 M5 的 prompt 依赖
    具体问题，故同一 scene 的不同问题在 P2 得到的 G7/G9 可能略有差异。补写采用
    "先到先得"（首个 episode 的 M5 统计落盘，后续复用），理由是：A/B 两臂按同一顺序
    跑同一批 episode → 两侧看到的 quality 完全一致（硬约束 18/21 的可比性优先）；
    逐题的米制授权门控走 `question_gate(...)`，用的是**该题**的新鲜 G9 统计，
    不受落盘值影响。
    """
    # G11 尺度 CI / 置信档由尺度锚定流程写入 artifact（G-11）
    # v5 HC29：G11 只用**分数口径**的相对 CI 半宽；旧绝对字段 scale_ci 已按 HC39
    # 从 Schema 移除，不允许再作为兼容回退读入。
    scale_ci_rel = art.scale_ci_rel
    allowed_metric_tasks = set(getattr(art, "allowed_metric_tasks", None) or set())
    scale_known = scale_is_usable(
        art.scale_known, art.scale_confidence,
        allowed_metric_tasks=allowed_metric_tasks, scale_ci_rel=scale_ci_rel)

    if qm.quality_is_computed(art):
        # 方案 X/Y 已落盘的实算值：零重算（P2 只读 artifact.quality）
        q = art.quality
        assert q is not None  # quality_is_computed 已保证
        if qm.needs_m5_enrichment(q) and (
                dynamic_masks is not None or track_ious):
            q = qm.enrich_with_m5(q, dynamic_masks=dynamic_masks,
                                  track_ious=track_ious)
            art = qm.apply_quality(art, q, status="computed",
                                   artifact_path=artifact_path if persist_quality else None)
    else:
        q = qm.compute_g1_g11(
            art,
            frames=frames,
            depth_maps=depth_maps,
            reproj_errors=reproj_errors,
            dynamic_masks=dynamic_masks,
            track_ious=track_ious,
            c2w_list=c2w_list,
            # v5：G11 用相对 CI 半宽（分数口径，HC29）
            scale_ci_rel=scale_ci_rel,
        )
        # 写回（不可变 + 可选落盘）：P2 只写内存即方案 Z，禁止
        art = qm.apply_quality(art, q, status="computed",
                               artifact_path=artifact_path if persist_quality else None)

    # 置信度融合（point_conf × coverage × reproj_err），数据缺失时退化为全 1
    coverage_overall = float("nan")
    try:
        point_conf = np.load(art.point_conf) if art.point_conf else None
    except Exception:
        point_conf = None  # ref 解析失败 → 覆盖率留 NaN（不伪造）
    if point_conf is not None:
        cov_count = (
            np.load(art.confidence.coverage_count_per_frame)
            if art.confidence.coverage_count_per_frame
            else np.full_like(point_conf, qm.MIN_FRAMES if hasattr(qm, "MIN_FRAMES") else 32)
        )
        fused = fuse_confidence(point_conf, cov_count, reproj_errors)
        coverage_overall = coverage_ratio(fused)
        if object_point_indices:
            per_object_coverage(fused, object_point_indices)

    # 分流（fail-closed：quality 未计算 / NaN 一律不得停 full_3d，硬约束 22）
    route = route_from_quality(q, quality_status=art.quality_status)

    summary = (
        f"scene={art.scene_name} method={art.recon_method} route={route} "
        f"overall_quality={float(q.overall_quality):.3f} "
        f"quality_status={art.quality_status} scale_known={scale_known} "
        f"scale_conf={art.scale_confidence} ci_rel={scale_ci_rel} "
        f"metric_tasks={sorted(allowed_metric_tasks)} "
        f"frame_set_hash={art.frame_set_hash[:12]} "
        f"coverage={coverage_overall if np.isfinite(coverage_overall) else 'NA'}"
    )

    return SceneState(
        artifact_ref=artifact_ref or art.artifact_id,
        route=route,  # type: ignore[arg-type]
        frame="world",
        scale_known=scale_known,
        objects=sorted(object_ids or []),   # §4.3：M5 绑定到的对象 id（无则空）
        summary=summary,
        scale_confidence=art.scale_confidence,
        scale_conflict=bool(getattr(art, "scale_conflict", False)),
        scale_ci_rel=scale_ci_rel,
        # v4 HC33：逐题型授权从 artifact 派生（M7 再叠加 G9 等质量条件）
        allowed_metric_tasks=allowed_metric_tasks,
        # v5 HC38：本版不设置 geometric_coverage 门（字段恒为 not_defined）
        coverage_gate_status="not_defined",
        available_artifacts=set(available_artifacts_for(route)),
        artifact=art,        # 运行时活句柄（只读引用，不序列化）
        quality=q,           # = artifact.quality 的只读引用（单一事实源）
    )
