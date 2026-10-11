"""M10 partial_tool_recovery（v6 §6.4/§14，D7）。

核心规则（逐条实现，不做变通）：

1. **保留未受污染的成功结果并回灌**：答案产生前某 Tool 失败时，不丢弃此前已成功
   执行的 Tool 结果；把成功结果当带来源/可信状态的先验（validated observations）
   回灌给模型继续恢复；
2. **失败不得伪装成功**：每个结果带 `{result_id, source_tool, status, evidence_version}`；
   `status="failed"` 的结果**不进** validated observations；
3. **保留范围**：仅当前 episode、当前 EvidenceProfile 版本下已成功、且**未被失败
   根因污染**的结果；
4. **共享前提失效时按依赖级联撤销**：若失败揭示共享前提无效（坐标/尺度/重建证据
   失效 → EvidenceProfile 对应能力降为 `unavailable`/`degraded`），级联撤销所有
   依赖该前提的旧结果，从 validated observations 移除并标 `invalidated_by`；
5. **重置命名空间后注入**：恢复时重置用户命名空间（避免引用已失效旧变量），
   重新注入 validated observations 摘要 + 失败信息（失败 Tool、根因、已撤销的
   result_id 列表）；
6. **有限次数**：恢复重试 ≤ N（`[TODO_CALIBRATE]`），超限切 `direct_vlm_routed` 或
   `abstain`；
7. **最终答案关联 result_ids**：trace 记录 `used_result_ids`、`recovery_count`、
   `partial_tool_recovery`、级联撤销列表。

性质（§14.3）：纯工程契约，与证据状态无关 —— 规则对所有 Tool / 所有
EvidenceProfile 状态统一适用；级联撤销会**回写** EvidenceProfile，但规则本身
不依赖特定能力值。
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Optional

from skill3d.schemas.evidence import CAPABILITY_ORDER, EvidenceProfile
from skill3d.schemas.reconstruction import SceneState
from skill3d.tools.contract import EVIDENCE_METRIC_SCALE

# 恢复重试上限（TODO_CALIBRATE：起始参考值，必须先标定再当结论）
MAX_RECOVERY_ATTEMPTS: int = 2   # TODO_CALIBRATE

# 失败根因 → 失效的共享前提（能力名）。None = 局部失败，不动共享证据。
#
# 判据来源：
# - `ArtifactUnavailableError` 的 missing 里若含深度/位姿/点云/对象 → 几何前提失效；
# - `ConfidenceGateError` 在米制 Tool 上 → 米制尺度前提失效；
# - 域值错误（负距离/点在相机后方）是**局部**的，不级联（§14.1 明文）。
_PREMISE_BY_ARTIFACT: dict[str, str] = {
    "depth": "geometry_3d",
    "poses": "geometry_3d",
    "point_cloud": "geometry_3d",
    "scale": EVIDENCE_METRIC_SCALE,
    "objects": "object_detection",
}

# 哪些能力失效会连带撤销依赖它的结果（撤销是**传递**的：几何没了，米制也没了）
_DEPENDENTS: dict[str, tuple[str, ...]] = {
    "geometry_3d": ("geometry_3d", "world_frame", EVIDENCE_METRIC_SCALE),
    "world_frame": ("world_frame",),
    "object_detection": ("object_detection", "object_grounding", "track_consensus"),
    EVIDENCE_METRIC_SCALE: (EVIDENCE_METRIC_SCALE,),
}

# 状态和可判定性先于测量，测量先于审计；不是题型解法。
_COMMON_FIELDS = (
    "status", "error", "service_healthy", "degradation_flags", "evidence_degraded",
    "duplicate_suspect", "grounding_status", "unit", "units",
    "obj_id", "category", "category_name", "track_id",
)
_SUMMARY_FIELDS = {
    "relative_distance_rank": (
        "status", "closest_category", "categories_without_detection",
        "categories_with_invalid_geometry", "tied_categories", "degradation_flags",
        "requested_categories", "reference", "ranking", "per_candidate",
        "margin_normalized", "candidates",
    ),
    "detect_objects": (
        "status", "service_healthy", "error", "n_detections", "n_new_objects",
        "new_object_ids", "categories", "frame_ids", "detections",
    ),
    "count_objects": ("count", "duplicate_suspect", "evidence_degraded",
                      "n_records", "n_distinct_tracks", "n_geometric_merges"),
    "object_centroid": ("centroid_normalized", "centroid_metric",
                        "category_name", "track_id", "world_up_used", "handedness_used"),
    "object_3d_extent": ("degradation_flags", "extent_longest_metric", "extent_metric",
                         "extent_normalized", "n_valid_points"),
    "plane_fit_room_size": ("degradation_flags", "room_area_m2", "fit_quality",
                            "plane_inlier_ratio", "room_diagonal_normalized"),
    "object_distance_m": ("degradation_flags", "distance_m", "distance_normalized", "a", "b"),
    "relative_direction_of": ("direction", "difficulty", "world_up_used", "handedness_used"),
    "inspect_frames": ("n_images", "frame_ids", "images", "note"),
}
_AUDIT_FIELDS = frozenset({"audit", "detector_endpoint_hash", "content_sha256"})


def _json(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, default=str)


def _payload_of(result):
    payload = getattr(result, "payload", None)
    if payload is not None:
        return payload
    raw = getattr(result, "value", None)
    try:
        return json.loads(raw) if isinstance(raw, str) else raw
    except (ValueError, TypeError):
        return raw


def summarize_payload(payload: object, *, tool: str = "", max_chars: int = 1200) -> str:
    """按完整字段/列表项压缩，所有省略明示；不截断 JSON 或把省略误作空值。"""
    omitted: list[str] = []

    def ordered(value, path=""):
        if isinstance(value, dict):
            keys = list(dict.fromkeys(
                (*(_SUMMARY_FIELDS.get(tool, ()) if not path else ()),
                 *_COMMON_FIELDS, *value)))
            out = {}
            for key in keys:
                if key not in value:
                    continue
                child = f"{path}.{key}" if path else key
                if key in _AUDIT_FIELDS:
                    omitted.append(child)
                else:
                    out[key] = ordered(value[key], child)
            return out
        if isinstance(value, (list, tuple)):
            return [ordered(item, f"{path}[{i}]") for i, item in enumerate(value)]
        return value

    def fit(value, budget, path="payload"):
        if len(_json(value)) <= budget:
            return value
        if isinstance(value, dict):
            out = {}
            for key, item in value.items():
                # 标量不截断；容器可以保留其完整前缀，但必须报告省略。
                room = budget - len(_json(out)) - len(_json(key)) - 4
                child = f"{path}.{key}"
                if room < 4:
                    omitted.append(child)
                    continue
                if isinstance(item, (dict, list)):
                    reduced = fit(item, room, child)
                    if reduced is None and item is not None:
                        continue
                    out[key] = reduced
                elif len(_json(item)) <= room:
                    out[key] = item
                else:
                    omitted.append(child)
            if out:
                return out
        elif isinstance(value, list):
            out = []
            for index, item in enumerate(value):
                if len(_json(out + [item])) > budget:
                    omitted.append(f"{path}[{index}:]（{len(value) - index} 项）")
                    break
                out.append(item)
            if out:
                return out
        omitted.append(path)
        return None

    clean = ordered(payload)
    reduced = fit(clean, max_chars)
    text = _json(reduced) if reduced is not None or payload is None else "（内容超出摘要预算）"
    if omitted:
        shown = list(dict.fromkeys(omitted))
        text += "\n  摘要省略：" + "、".join(shown[:8])
        if len(shown) > 8:
            text += f" 等共 {len(shown)} 处"
        text += "；省略不表示空值，不能据此假定候选或记录完整。"
    return text


@dataclass
class ValidatedObservation:
    """一条可信的既有观测（回灌给模型的先验）。"""

    result_id: str
    source_tool: str
    payload: object
    evidence_version: str = ""
    args: dict = field(default_factory=dict)
    degraded_evidence: list[str] = field(default_factory=list)

    def summary(self, max_chars: int = 1200) -> str:
        text = summarize_payload(self.payload, tool=self.source_tool, max_chars=max_chars)
        flags = (f" degraded_evidence={_json(self.degraded_evidence)}"
                 if self.degraded_evidence else "")
        return (f"- {self.source_tool}({_json(self.args)}) [result_id={self.result_id}]"
                f" ToolResult.status=ok evidence_version={self.evidence_version!r}{flags}"
                f"\n  payload={text}")


def observation_feedback(observations: list[ValidatedObservation]) -> str:
    """仅合并同版本、同参数、同完整内容的观察；账本与真实 ID 不变。"""
    groups: dict[str, list[ValidatedObservation]] = {}
    for obs in observations:
        key = json.dumps(
            [obs.source_tool, obs.evidence_version, obs.args, obs.payload,
             obs.degraded_evidence], ensure_ascii=False, sort_keys=True, default=str)
        groups.setdefault(key, []).append(obs)
    lines = ["仍然可信的既有观测（未撤销的成功调用；不等于业务结论可判定）："]
    if not groups:
        lines.append("当前没有任何可信的既有观测；可继续获取观察或依据原图判断。")
    for group in groups.values():
        lines.append(group[0].summary())
        aliases = list(dict.fromkeys(o.result_id for o in group[1:]
                                    if o.result_id != group[0].result_id))
        if aliases:
            lines.append(f"  相同观察另有 result_ids={_json(aliases)}；内容仅显示一次。")
    lines.append("payload.status、缺失项、歧义与降级标记仍须检查；"
                 "未检出不证明不存在，None/不可测不等于零。")
    return "\n".join(lines)


def failed_call_feedback(results) -> str:
    """只展示本轮真实失败调用，不从生成代码猜测参数。"""
    lines = []
    for result in results:
        if result.status == "ok" and result.error is None:
            continue
        lines.append(
            f"失败调用：{result.source_tool or result.tool}({_json(result.args)})"
            f" [result_id={result.result_id}] error_code={result.error_code}"
            f"；原因={result.error}")
    return "\n".join(lines)


@dataclass
class RecoveryPlan:
    """一次恢复动作的完整描述（进 trace，可审计）。"""

    attempt: int
    failed_tool: str
    failure_kind: str                      # tool_contract / confidence_gate / domain_value
    premise: Optional[str]                 # 失效的共享前提（None = 局部失败）
    validated: list[ValidatedObservation] = field(default_factory=list)
    invalidated_result_ids: list[str] = field(default_factory=list)
    invalidated_tools: list[str] = field(default_factory=list)
    feedback: str = ""
    downgraded_capabilities: dict[str, str] = field(default_factory=dict)
    exhausted: bool = False


def premise_of_failure(*, error_code: str, tool: str,
                       requires_evidence: tuple[str, ...] = (),
                       missing_artifacts: tuple[str, ...] = (),
                       unmet_evidence: tuple[str, ...] = ()) -> Optional[str]:
    """从失败事实推断失效的共享前提；`None` = 局部失败，不做级联（§14.1）。

    `unmet_evidence`：证据门给出的**逐项未达标能力**（`contract.evidence_unmet`
    的输出，可能带 `"(degraded 且不容忍)"` 之类的后缀）。**必须优先于产物判据**：
    实测踩过的坑是"证据缺口"被当成"缺产物"从而一律退成 `geometry_3d`，
    级联撤销就把 geometry/world_frame/metric_scale 全部无差别降级了。
    """
    # 证据缺口优先：取**具体**那个能力作前提（去掉括号后缀）
    for item in unmet_evidence:
        cap = str(item).split("(")[0].strip()
        if cap in _DEPENDENTS:
            return cap
    if error_code == "confidence_gate":
        if EVIDENCE_METRIC_SCALE in requires_evidence:
            return EVIDENCE_METRIC_SCALE
        # 其他局部质量门：不视为共享前提失效
        return None
    if error_code == "tool_contract":
        for art in missing_artifacts:
            p = _PREMISE_BY_ARTIFACT.get(str(art))
            if p is not None:
                return p
        return "geometry_3d"      # 缺产物但没标注具体产物 → 保守按几何前提失效
    # domain_value / answer_already_given / 其他：局部失败
    return None


def collect_validated(kernel, *, evidence_version: str = "") -> list[ValidatedObservation]:
    """从 kernel 的 calls/results 里取 `status=ok` 且未被撤销的结果（§14.1）。"""
    out: list[ValidatedObservation] = []
    for r in getattr(kernel, "tool_results", []) or []:
        if str(getattr(r, "status", "ok")) != "ok":
            continue                                  # 失败结果不进 validated
        if getattr(r, "error", None) is not None:
            continue
        if getattr(r, "invalidated_by", None):
            continue                                  # 已被级联撤销
        out.append(ValidatedObservation(
            result_id=str(getattr(r, "result_id", "") or getattr(r, "request_digest", "")),
            source_tool=str(getattr(r, "source_tool", "") or getattr(r, "tool", "")),
            payload=_payload_of(r),
            evidence_version=str(getattr(r, "evidence_version", "") or evidence_version),
            args=dict(getattr(r, "args", None) or {}),
            degraded_evidence=list(getattr(r, "degraded_evidence", None) or [])))
    return out


def cascade_invalidate(kernel, premise: str, *,
                       registry) -> tuple[list[str], list[str]]:
    """按依赖级联撤销（§14.1 第 4 条）。

    撤销范围 = 使用的能力落在 `_DEPENDENTS[premise]` 里的所有既有成功结果。
    返回 `(invalidated_result_ids, invalidated_tool_names)`；同时把被撤销结果的
    `invalidated_by` 写上（从 validated observations 移除的依据）。
    """
    affected = set(_DEPENDENTS.get(str(premise), (str(premise),)))
    ids: list[str] = []
    tools: list[str] = []
    for r in getattr(kernel, "tool_results", []) or []:
        if r.status != "ok" or r.invalidated_by:
            continue
        name = str(getattr(r, "source_tool", "") or getattr(r, "tool", ""))
        try:
            req = set(registry.requires_evidence(name))
        except Exception:  # noqa: BLE001 - 未注册名（回放/历史 trace）→ 不撤销
            continue
        if req & affected:
            rid = str(getattr(r, "result_id", "") or getattr(r, "request_digest", ""))
            if rid and rid not in ids:
                ids.append(rid)
            if name and name not in tools:
                tools.append(name)
            # 标记撤销原因（不可变 Schema：用 model_copy 写回列表元素）
            try:
                idx = kernel.tool_results.index(r)
                kernel.tool_results[idx] = r.model_copy(
                    update={"invalidated_by": sorted(set(
                        (getattr(r, "invalidated_by", None) or []) + [premise]))})
            except (ValueError, AttributeError):  # pragma: no cover - 防御
                pass
    return ids, tools


def invalidate_results(kernel, result_ids: list[str], *, reason: str) -> list[str]:
    """撤销已定位的局部结果，不推测它们与其他调用的共享依赖。"""
    wanted = set(result_ids)
    invalidated = []
    for index, result in enumerate(kernel.tool_results):
        if result.result_id in wanted and result.status == "ok" and not result.invalidated_by:
            kernel.tool_results[index] = result.model_copy(
                update={"invalidated_by": [reason]})
            invalidated.append(result.result_id)
    return invalidated


def downgrade_profile(profile: Optional[EvidenceProfile],
                      premise: str, *, confirmed_invalid: bool = False
                      ) -> tuple[Optional[EvidenceProfile], dict[str, str]]:
    """降级失效前提；M11 确证无效时直接收回对应能力。"""
    if profile is None:
        return None, {}
    changed: dict[str, str] = {}
    update: dict = {}
    reasons = dict(getattr(profile, "state_reasons", None) or {})
    for cap in _DEPENDENTS.get(str(premise), (str(premise),)):
        cur = profile.state(cap)
        if confirmed_invalid and cur != "unavailable":
            update[cap] = "unavailable"
            changed[cap] = f"{cur}→unavailable"
        elif cur == "available":
            update[cap] = "degraded"
            changed[cap] = "available→degraded"
        elif cur == "degraded":
            update[cap] = "unavailable"
            changed[cap] = "degraded→unavailable"
        else:
            continue
        # v9 §6.1：被级联撤销的能力记 `invalidated` —— 与"没跑"和"跑失败"区分开
        reasons[cap] = "invalidated"
    if update:
        update["state_reasons"] = reasons
    return (profile.model_copy(update=update) if update else profile), changed


def build_feedback(plan: RecoveryPlan, *, question_type: str,
                   scope: str, require_answer: bool = False,
                   include_observations: bool = True) -> str:
    """恢复用的回灌文本（§14.1 第 5 条）：失败信息 + validated observations 摘要。"""
    lines = [
        f"失败 Tool：{plan.failed_tool}（{plan.failure_kind}）",
    ]
    if plan.premise:
        lines.append(
            f"根因：共享前提 `{plan.premise}` 失效 → 已级联撤销依赖它的既有结果"
            f"（撤销 {len(plan.invalidated_result_ids)} 条："
            f"{plan.invalidated_result_ids[:6]}）；相应证据能力已降级 "
            f"{plan.downgraded_capabilities}")
        lines.append("**不要**再使用上面被撤销的数值。")
    else:
        lines.append("根因：该 Tool 的局部错误（域值/参数），共享证据仍然有效。")
    if include_observations:
        lines.append(observation_feedback(plan.validated))
    ending = (
        "请使用仍有效的观察或原图给出最佳答案；视觉估计如实标记为 visual_estimate，"
        "不要提交 abstain。" if require_answer else
        "若确实拿不到答案，直接 `ReturnAnswer(\"abstain\")`（主榜按错计，优于编造）。")
    lines.append(f"当前 question_tool_scope={scope}，本题题型={question_type}。")
    if include_observations:
        lines.append("只允许调用 prompt 里列出的 Tool；" + ending)
        lines.append("注意：`ReturnAnswer` 之后**不得**再调用任何 Tool（会抛 "
                     "AnswerAlreadyGiven）。")
    return "\n".join(lines)


def recovery_exhausted(attempt: int, *, max_attempts: int = MAX_RECOVERY_ATTEMPTS) -> bool:
    """恢复次数是否已用尽（超限 → direct_vlm_routed / abstain，§6.4）。"""
    return int(attempt) > int(max_attempts)


def worst_capability(states: list[str]) -> str:
    """取最差状态（诊断用；与 `schemas.evidence.worst_state` 同口径）。"""
    if not states:
        return "unavailable"
    return min((str(x) for x in states),
               key=lambda x: CAPABILITY_ORDER.get(x, 0))


__all__ = [
    "MAX_RECOVERY_ATTEMPTS",
    "RecoveryPlan",
    "ValidatedObservation",
    "build_feedback",
    "cascade_invalidate",
    "collect_validated",
    "downgrade_profile",
    "failed_call_feedback",
    "invalidate_results",
    "observation_feedback",
    "premise_of_failure",
    "recovery_exhausted",
    "summarize_payload",
]
