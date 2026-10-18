"""M10 partial_tool_recovery（v6 §6.4/§14，D7）。

核心规则（逐条实现，不做变通）：

1. **保留未受污染的成功结果并回灌**：答案产生前某 Tool 失败时，不丢弃此前已成功
   执行的 Tool 结果；把成功结果当带来源/可信状态的先验（validated observations）
   回灌给模型继续恢复；
2. **失败不得伪装成功**：每个结果带 `{result_id, source_tool, status, evidence_version}`；
   `status="failed"` 的结果**不进** validated observations；
3. **保留范围**：仅当前 episode、当前 EvidenceProfile 版本下已成功、且**未被失败
   根因污染**的结果；
4. **共享前提失效时按依赖级联撤销**：若失败揭示共享前提无效，公共几何／尺度
   置为 unavailable，局部坐标约定或候选产物按对应依赖收回；级联撤销所有
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

import hashlib
import json
from dataclasses import dataclass, field
from typing import Optional

from skill3d.schemas.evidence import CAPABILITIES, EvidenceProfile
from skill3d.schemas.reconstruction import SceneState
from skill3d.schemas.tool import FailureCause
from skill3d.tools.contract import EVIDENCE_METRIC_SCALE

# 恢复重试上限（TODO_CALIBRATE：起始参考值，必须先标定再当结论）
MAX_RECOVERY_ATTEMPTS: int = 2   # TODO_CALIBRATE

# 失败根因 → 失效的共享前提（能力名）。None = 局部失败，不动共享证据。
#
# 优先使用结构化事实；旧记录仅兼容明确缺失项，绝不从工具依赖猜根因。
_PREMISE_BY_ARTIFACT: dict[str, str] = {
    "depth": "geometry_3d",
    "poses": "geometry_3d",
    "point_cloud": "geometry_3d",
    "scale": EVIDENCE_METRIC_SCALE,
    "objects": "object_detection",
}

# 哪些能力失效会连带撤销依赖它的结果（撤销是**传递**的：几何没了，米制也没了）
_DEPENDENTS: dict[str, tuple[str, ...]] = {
    "geometry_3d": ("geometry_3d", EVIDENCE_METRIC_SCALE),
    "world_frame": ("world_frame",),
    "object_detection": ("objects",),
    EVIDENCE_METRIC_SCALE: (EVIDENCE_METRIC_SCALE,),
}

# 状态和可判定性先于测量，测量先于审计；不是题型解法。
_COMMON_FIELDS = (
    "status", "error", "service_healthy", "degradation_flags", "diagnostic_warnings",
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
    "count_objects": ("count", "duplicate_suspect", "diagnostic_warnings",
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


def _list_objects_summary(payload: list, max_chars: int) -> str:
    """先装候选身份，再按列补质量；审计正文只通过完整 payload 摘要标识。"""
    digest = hashlib.sha256(json.dumps(
        payload, sort_keys=True, ensure_ascii=False, default=str).encode()).hexdigest()
    rows = []
    priority = (
        "binding_flags", "grounding_status", "track_id", "det_conf", "det_conf_source",
        "mask_support_ratio", "duplicate_suspect",
    )
    audit_omitted = any(o.get("binding_flags") or o.get("binding_provenance")
                        or o.get("audit") for o in payload if isinstance(o, dict))

    def encode():
        omitted_fields = sum(
            len(o) - len(row) for o, row in zip(payload, rows) if isinstance(o, dict))
        body = {
            "objects": rows, "total": len(payload), "shown": len(rows),
            "omitted": len(payload) - len(rows), "fields_omitted": omitted_fields,
            "coverage_complete": len(rows) == len(payload),
            "audit_omitted": audit_omitted, "payload_sha256": digest,
        }
        return json.dumps(body, ensure_ascii=False, separators=(",", ":"), default=str)

    for obj in payload:
        if not isinstance(obj, dict):
            break
        identity = {k: obj[k] for k in ("obj_id", "category_name", "category") if k in obj}
        rows.append(identity)
        if len(encode()) > max_chars:
            rows.pop()
            break
    if not rows and payload:
        return json.dumps({"omitted": True, "payload_sha256": digest}, separators=(",", ":"))
    for key in priority:
        for source, row in zip(payload, rows):
            if key not in source:
                continue
            value = source[key]
            if key == "binding_flags":
                value = [{k: flag[k] for k in ("code", "audit_id", "audit_sha256") if k in flag}
                         for flag in value if isinstance(flag, dict)]
            row[key] = value
            if len(encode()) > max_chars:
                row.pop(key)
    return encode()


def summarize_payload(payload: object, *, tool: str = "", max_chars: int = 1200) -> str:
    """按完整字段/列表项压缩，所有省略明示；不截断 JSON 或把省略误作空值。"""
    if max_chars < 128:
        raise ValueError("summary max_chars must be at least 128")
    if tool == "list_objects" and isinstance(payload, list):
        text = _list_objects_summary(payload, max_chars)
        if len(text) <= max_chars:
            return text
        return '{"omitted":true,"coverage_complete":false}'
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
                    omitted.append(f"{path}[{index}:] ({len(value) - index} items)")
                    break
                out.append(item)
            if out:
                return out
        omitted.append(path)
        return None

    clean = ordered(payload)
    original_omitted = list(omitted)
    if not omitted and len(_json(clean)) <= max_chars:
        return _json(clean)
    # 固定预留省略说明预算，整个返回文本有界。路径只作例示，数量保留完整。
    reserve = 110
    reduced = fit(clean, max_chars - reserve)
    text = _json(reduced)
    shown = list(dict.fromkeys(omitted))
    example = ", ".join(shown[:2])[:40]
    note = (f"\nSummary omissions: {example}; total={len(shown)}; "
            "coverage_complete=false.")
    if len(text) + len(note) > max_chars:
        omitted[:] = original_omitted
        reduced = fit(clean, max_chars - len(note))
        text = _json(reduced)
    if len(text) + len(note) <= max_chars:
        return text + note
    return '{"omitted":true,"coverage_complete":false}'


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
    lines = ["Still-valid observations (successful, non-invalidated calls; task conclusions may remain indeterminate):"]
    if not groups:
        lines.append("No valid prior observations are available; obtain further observations or reason from original images.")
    for group in groups.values():
        lines.append(group[0].summary())
        aliases = list(dict.fromkeys(o.result_id for o in group[1:]
                                    if o.result_id != group[0].result_id))
        if aliases:
            lines.append(f"  Identical observations also have result_ids={_json(aliases)}; content is shown once.")
    lines.append("Check payload.status, missing items, ambiguity, and degradation flags. "
                 "No detection does not prove absence; None/unmeasurable does not mean zero.")
    return "\n".join(lines)


def failed_call_feedback(results) -> str:
    """只展示本轮真实失败调用，不从生成代码猜测参数。"""
    lines = []
    for result in results:
        if result.status == "ok" and result.error is None:
            continue
        cause = (f"; failure_cause={_json(result.failure_cause.model_dump(mode='json'))}"
                 if result.failure_cause else "")
        lines.append(
            f"Failed call: {result.source_tool or result.tool}({_json(result.args)})"
            f" [result_id={result.result_id}] error_code={result.error_code}"
            f"; reason={result.error}{cause}")
    return "\n".join(lines)


def evidence_state_fingerprint(scene, handle, kernel, *, excluded_ids=()) -> str:
    """M4: hash facts and authorization, excluding receipt/round bookkeeping."""
    excluded = set(excluded_ids)
    observations = set()
    for obs in collect_validated(kernel):
        if obs.result_id not in excluded:
            observations.add(json.dumps(
                [obs.source_tool, obs.evidence_version, obs.args,
                 obs.payload, obs.degraded_evidence],
                sort_keys=True, ensure_ascii=False, default=str))
    ledger = getattr(handle, "_ledger", None)
    images = [
        [image.image_id, image.content_sha256, image.observed()]
        for image in (ledger.all_images() if ledger is not None else [])
    ]
    value = {
        "observations": sorted(observations),
        "scene": scene.model_dump(mode="json") if scene is not None else None,
        "handle": handle.state_digest() if handle is not None else None,
        "objects": [obj.model_dump(mode="json") for obj in handle.object_records()]
        if handle is not None else [],
        "images": images,
    }
    return hashlib.sha256(json.dumps(
        value, sort_keys=True, ensure_ascii=False, default=str).encode()).hexdigest()


def deterministic_cell_failure(cell, results) -> bool:
    """Transient service/network failures must retain the original retry path."""
    transient = ("timeout", "connection", "service", "network", "unavailable", "fault", "offline")
    for result in results:
        payload = _payload_of(result)
        if isinstance(payload, dict) and (
                payload.get("service_healthy") is False or payload.get("status") == "fault"):
            return False
        if isinstance(payload, dict) and any(
                word in str(payload.get("error", "")).lower() for word in transient):
            return False
        error = str(getattr(result, "error", "") or "").lower()
        if any(word in error for word in transient):
            return False
    if cell.error_code == "timeout":
        return False
    return cell.error_type in {
        "KeyError", "AttributeError", "TypeError", "IndexError",
        "UnboundLocalError", "NameError",
        "ToolContractError", "DomainValueError", "ArtifactUnavailableError",
        "ConfidenceGateError",
    }


def field_repair_instructions(cell, results) -> str:
    """Derive a specific field repair from the real error and real ToolResult."""
    if cell.error_type != "KeyError" or cell.error_message.strip("'\"") != "obj_id":
        return ""
    lines = []
    for result in results:
        if (result.source_tool or result.tool) != "detect_objects":
            continue
        payload = _payload_of(result)
        if result.status != "ok" or not isinstance(payload, dict):
            continue
        detections = payload.get("detections")
        if not isinstance(detections, list) or not detections:
            continue
        if all(isinstance(item, dict) and "obj_id" not in item for item in detections):
            lines.append(
                f"Required field repair: detect_objects result_id={result.result_id} "
                "detections have no obj_id. "
                f"This result reports new_object_ids={_json(payload.get('new_object_ids'))}. "
                "Use the real new_object_ids and list_objects records to verify object identity. "
                "An empty new_object_ids list does not mean previously registered objects disappeared. "
                "Do not infer a detection-to-object mapping from list order. "
                "New 2D candidates have no 3D geometry.")
    return "\n".join(lines)


@dataclass
class RecoveryPlan:
    """一次恢复动作的完整描述（进 trace，可审计）。"""

    attempt: int
    failed_tool: str
    failure_kind: str                      # tool_contract / confidence_gate / domain_value
    premise: Optional[str]                 # 失效的共享前提（None = 局部失败）
    failure_cause: Optional[FailureCause] = None
    validated: list[ValidatedObservation] = field(default_factory=list)
    invalidated_result_ids: list[str] = field(default_factory=list)
    invalidated_tools: list[str] = field(default_factory=list)
    feedback: str = ""
    downgraded_capabilities: dict[str, str] = field(default_factory=dict)
    exhausted: bool = False


def premise_of_failure(*, error_code: str, tool: str,
                       requires_evidence: tuple[str, ...] = (),
                       missing_artifacts: tuple[str, ...] = (),
                       unmet_evidence: tuple[str, ...] = (),
                       failure_cause: Optional[FailureCause] = None) -> Optional[str]:
    """从失败事实推断失效的共享前提；`None` = 局部失败，不做级联（§14.1）。

    `unmet_evidence`：授权检查给出的**逐项未达标能力**（可能带
    `"(unavailable)"` 之类的后缀）。**必须优先于产物判据**：
    实测踩过的坑是"证据缺口"被当成"缺产物"从而一律退成 `geometry_3d`，
    级联撤销就把 geometry/world_frame/metric_scale 全部无差别降级了。
    """
    if failure_cause is not None:
        return next(iter(failure_cause.shared_premises), None)
    # 兼容无结构化原因的旧记录，只依据明确的未满足项恢复。
    for item in unmet_evidence:
        cap = str(item).split("(")[0].strip()
        if cap in _DEPENDENTS:
            return cap
    if error_code == "tool_contract":
        for art in missing_artifacts:
            p = _PREMISE_BY_ARTIFACT.get(str(art))
            if p is not None:
                return p
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
            spec = registry.spec(name)
            req = (set(spec.requires_evidence) | set(spec.requires_local_facts)
                   | set(spec.requires_artifacts))
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
    """撤销已定位的结果及显式派生引用，不推测共享依赖。"""
    wanted = set(result_ids)
    while True:
        dependents = {
            r.result_id for r in kernel.tool_results
            if wanted.intersection(r.result_dependencies)
        }
        if dependents.issubset(wanted):
            break
        wanted.update(dependents)
    invalidated = []
    for index, result in enumerate(kernel.tool_results):
        if result.result_id in wanted and result.status == "ok" and not result.invalidated_by:
            kernel.tool_results[index] = result.model_copy(
                update={"invalidated_by": [reason]})
            invalidated.append(result.result_id)
    return invalidated


def invalidate_objects(kernel, object_ids: list[str], *, reason: str) -> list[str]:
    """只撤销实际消费这些对象的几何结果及其显式派生结果。"""
    affected = set(object_ids)
    return invalidate_results(kernel, [
        r.result_id for r in kernel.tool_results
        if affected.intersection(r.object_dependencies)
    ], reason=reason)


def downgrade_profile(profile: Optional[EvidenceProfile],
                      premise: str, *, confirmed_invalid: bool = False,
                      excluded_capabilities=()
                      ) -> tuple[Optional[EvidenceProfile], dict[str, str]]:
    """失效即 unavailable；对象与坐标约定不写回公共画像。"""
    if profile is None:
        return None, {}
    changed: dict[str, str] = {}
    update: dict = {}
    reasons = dict(getattr(profile, "state_reasons", None) or {})
    for cap in _DEPENDENTS.get(str(premise), (str(premise),)):
        if cap not in CAPABILITIES or cap in excluded_capabilities:
            continue
        cur = profile.state(cap)
        if cur != "unavailable":
            update[cap] = "unavailable"
            changed[cap] = f"{cur}→unavailable"
        else:
            continue
        # v9 §6.1：被级联撤销的能力记 `invalidated` —— 与"没跑"和"跑失败"区分开
        reasons[cap] = "invalidated"
    if update:
        update["state_reasons"] = reasons
    return (EvidenceProfile.model_validate({**profile.model_dump(), **update})
            if update else profile), changed


def build_feedback(plan: RecoveryPlan, *, question_type: str,
                   scope: str, require_answer: bool = False,
                   include_observations: bool = True) -> str:
    """恢复用的回灌文本（§14.1 第 5 条）：失败信息 + validated observations 摘要。"""
    lines = [
        f"Failed tool: {plan.failed_tool} ({plan.failure_kind})",
    ]
    if plan.failure_cause:
        lines.append(f"Failure facts: {_json(plan.failure_cause.model_dump(mode='json'))}")
    if plan.premise:
        lines.append(
            f"Cause: shared premise `{plan.premise}` is invalid. Dependent results were "
            f"cascade-invalidated ({len(plan.invalidated_result_ids)} results: "
            f"{plan.invalidated_result_ids[:6]}); corresponding evidence capabilities were downgraded: "
            f"{plan.downgraded_capabilities}")
        lines.append("Do not reuse the invalidated values above.")
    else:
        failure_scope = plan.failure_cause.scope if plan.failure_cause else "unknown"
        lines.append(
            f"Cause: a local tool error or unlocated failure (scope={failure_scope}); "
            f"shared evidence remains valid. Invalidated result IDs: {plan.invalidated_result_ids}.")
    if include_observations:
        lines.append(observation_feedback(plan.validated))
    ending = (
        "Give the best answer using still-valid observations or original images. "
        "Honestly label a visual estimate as visual_estimate; do not submit abstain." if require_answer else
        "If no answer can be obtained, use `ReturnAnswer(\"abstain\")` "
        "(scored as incorrect on the main benchmark; preferable to fabrication).")
    lines.append(f"Current question_tool_scope={scope}; question type={question_type}.")
    if include_observations:
        lines.append("Call only tools listed in the prompt. " + ending)
        lines.append("Do not call any tool after `ReturnAnswer` (raises AnswerAlreadyGiven).")
    return "\n".join(lines)


def recovery_exhausted(attempt: int, *, max_attempts: int = MAX_RECOVERY_ATTEMPTS) -> bool:
    """恢复次数是否已用尽（超限 → direct_vlm_routed / abstain，§6.4）。"""
    return int(attempt) > int(max_attempts)


__all__ = [
    "MAX_RECOVERY_ATTEMPTS",
    "RecoveryPlan",
    "ValidatedObservation",
    "build_feedback",
    "cascade_invalidate",
    "collect_validated",
    "deterministic_cell_failure",
    "downgrade_profile",
    "evidence_state_fingerprint",
    "failed_call_feedback",
    "field_repair_instructions",
    "invalidate_results",
    "invalidate_objects",
    "observation_feedback",
    "premise_of_failure",
    "recovery_exhausted",
    "summarize_payload",
]
