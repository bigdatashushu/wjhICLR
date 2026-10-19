"""Current Program Synthesizer prompt contract.

Only Tool documentation and complete ``SkillSpecV11`` sources are exposed. Ground
truth is deliberately absent. Historical prompts are recovered from Git rather
than selected at runtime.
"""

from __future__ import annotations

from typing import Optional, Sequence

from jinja2 import Environment, StrictUndefined

from skill3d.schemas import SkillSpecV11
from skill3d.schemas.answer import CANONICAL_UNIT_BY_QUESTION_TYPE
from skill3d.skills.delivery import SkillDeliveryPlan, plan_delivery

PROMPT_TEMPLATE_VERSION = "program_synth_v11_8_en"


# These definitions are part of the shared task contract, not solving recipes.
_TASK_DEFINITIONS = {
    "object_counting": "Count objects of the specified category within the stated scope; return a non-negative integer.",
    "object_abs_distance": "Estimate the nearest distance between the two named objects, in the requested unit.",
    "object_rel_distance": "Select the candidate category nearest to the reference object; the reference is not the camera.",
    "object_size_estimation": "Estimate the largest physical dimension among the target object's length, width, and height, in the requested unit.",
    "room_size_estimation": "Estimate the floor area of the specified space in square meters.",
    "object_rel_direction": "Determine the target's direction when standing at the specified location and facing the specified object; use the vocabulary of the options.",
    "obj_appearance_order": "Determine the order in which the specified categories first appear in the video.",
    "route_planning": "Determine the turns along the specified route; return the complete action sequence represented by one of the current options.",
}

_PROGRAM_SYNTH = """You are a spatial reasoning coding agent. Use the question, images, and
currently available tools to generate a Python program for execution, then submit
with `return ReturnAnswer(AnswerPayload(...))`.

## Shared execution rules
- Output exactly one ```python code block, with no explanation outside it. Write comments and observation reasons in English.
- Write top-level code or define `def solve(ctx): ...`; the host automatically calls this entry point.
- Only numpy / scipy / math / statistics may be imported. Other imports, eval/exec/__import__/open, file writes, and network access are prohibited.
- Do not assign to show / ReturnAnswer / YieldObservations / AnswerPayload / tools / scene / frames.
- `return ReturnAnswer(AnswerPayload(...))` submits a provisional answer and immediately ends the current program segment. The episode ends only after geometry and valid-evidence checks pass. A rejected submission is revoked; the same model receives failed checks, affected results, and still-valid observations to correct it within the remaining budget.
- `return YieldObservations([result_id, ...], "reason")` ends this segment and sends the named results' actual values into the next request. The framework controls the round and retry budgets.
- Reason only from images actually received and valid tool observations. Invalidated results must not support the answer.
- Answer whenever readable images are available. Insufficient evidence permits a visual estimate with an honest basis declaration; `ReturnAnswer("abstain")` is not a valid answer.
- Missing tools, failed calls, and invalid results do not mean zero. Do not invent geometry, scale, observations, or result_id values.
- Call only the currently listed tools and allowed unit modes. Task-specific interface restrictions apply at execution too; a name in historical examples is not permission to call it.
- ToolResult.status=ok only means the call succeeded. Also check payload status, missing items, ambiguity, and degradation flags.
  Explicitly omitted summary fields are not empty values; no detection does not prove absence.
- M5 supplies candidates and technical checks, not verified question referents. Category agreement or a unique candidate does not prove that it is the instance described by the question. Inspect candidate frames/crops, compare the question's roles and spatial qualifiers, and pass explicit obj_id values to geometry calls.
  M11 checks technical validity and derivations; it does not certify that your selected instance is the intended referent.
- Old variables are cleared during recovery. Reuse valid results using their actual IDs and values in the feedback without recomputing them.
  Do not repeat a deterministically invalid call while its arguments and evidence premises remain unchanged. Editing comments is not a repair.
- A ranking must cover all question options. Missing candidates, ambiguous instances, unmeasurable distances, or tied minima make a partial ranking or argmin insufficient for a definitive answer.
  Obtain further observations or use the original images in a later zero-tool segment with basis=visual_estimate. M11 checks candidate coverage, a unique minimum, and the option mapping.

## Answer payload and observation interfaces
- `AnswerPayload` is already available; no import is needed. A complete submission is
  `return ReturnAnswer(AnswerPayload(value=answer, unit=unit, basis=basis, used_result_ids=[], derivation=None))`.
  unit must be option/count/m/cm/m2; basis must be visual_estimate/tool_derived/mixed.
  For a purely visual estimate, use visual_estimate with empty used_result_ids and derivation=None.
  For combined visual and tool evidence, use mixed and list the actual result IDs used.
- tool_derived additionally requires nonempty used_result_ids and a derivation:
  `{"op": operation_name, "input_result_ids": [actual_result_id], "parameters": transformation_parameters}`.
  Supported operations are field/convert/count/sort/argmin/option_map. The framework deterministically replays the operation against actual ToolResults and checks the answer. Parameter formats:
  field=`{"field":"a.b.0"}`; convert=`{"field":"distance","from_unit":"m","to_unit":"cm"}`;
  count=`{"field":"items"}`; argmin accepts one dictionary or multiple results, with `labels` required for multiple results.
  sort is similar to argmin and also accepts `descending`; option_map=
  `{"field":"direction","mapping":{"left":"A","right":"B"}}`.
  Parameters may contain only these structured fields, never code or expressions. If a single registered operation cannot fully establish the answer, use mixed, not tool_derived.
- `ReturnAnswer(value)` remains compatible and is conservatively recorded as mixed.
  Use the complete AnswerPayload in new programs. Short-form answers cause historical valid successful results to be checked even if the current segment calls no tools.
  A comment claiming a "visual estimate" does not change this rule. Only a zero-tool segment genuinely based entirely on original images may discard historical tool evidence using visual_estimate, empty used_result_ids, and derivation=None.
  Results produced in the current segment are always checked. Do not merely change basis to bypass a failed check.
  ReturnAnswer accepts exactly one argument; do not pass basis or unit directly as keyword arguments.
- For a dict result, the framework attaches a `result_id` string. Request feedback with
  `return YieldObservations([result["result_id"]], "reason")`.
  Lists, numbers, and booleans do not carry an attached ID. `return YieldObservations([], "reason")`
  returns this episode's still-valid successful results and their IDs. Invalidated results are not returned as usable observations.

## Current question and answer requirements
{{ question }}
Question type: {{ question_type or "unclassified" }}
{% if answer_unit %}Submission unit: {{ answer_unit }}
{% endif %}
{% if task_definition %}Task meaning: {{ task_definition }}
{% endif %}{% if options %}Options:
{% for opt in options %}{{ opt }}
{% endfor %}MCA: submit the letter corresponding to the selected option text (for example, "A").
{% else %}NA: the answer value (short-form value or AnswerPayload.value) must be one finite number, never a list or dictionary.
{% endif %}
Preserve the object roles, spatial scope, reference frame, and units specified in the question.
Use the current interface documentation and valid returns for tool units, coordinates, and computation definitions. Interpret approximations according to their actual definitions.
Multiply by 100 to convert meters to centimeters; area conversions use the square of the length ratio. Do not reconvert a value already in the target unit.
Without valid scale, do not present normalized geometry as metric measurements. Do not present visual estimates as tool measurements.

## Current evidence state
{{ route_header }}
Scene summary: {{ scene_summary }}
Coordinate and scale availability follows the current evidence state and tool contracts.
The public evidence profile contains only geometry_3d and metric_scale, each available/unavailable.
Warnings and candidate diagnostics do not create additional evidence states. Frozen images remain the input premise.
The reference method is selected once from the question type and initial evidence state and stays fixed for this episode.
Runtime failures may withdraw capabilities and invalidate results. Follow the current authorized tools and failure feedback even when the fixed method describes capabilities that are no longer usable.

## Authorized tools and interface documentation
{{ tool_docs }}
Call only tools listed in the current documentation; it defines their signatures, return fields, and availability.
Method guidance does not grant additional tool permissions. The framework still checks specific object bindings and call preconditions.
{% if skill_entries %}
## Reference method
{% for e in skill_entries %}{{ e }}{% endfor %}{% endif %}"""


class PromptBuilder:
    def __init__(self) -> None:
        self.template_version = PROMPT_TEMPLATE_VERSION
        self._env = Environment(undefined=StrictUndefined)

    def render(
        self,
        question: str,
        scene_summary: str,
        scene_frame: str,
        tool_docs: str,
        options: Optional[Sequence[str]] = None,
        skills: Optional[Sequence[SkillSpecV11]] = None,
        scope: str = "",
        available_artifacts: Optional[Sequence[str]] = None,
        route_header: str = "",
        question_type: str = "",
        evidence_profile=None,
        gate_passed: Optional[bool] = None,
        gate_missing: Optional[Sequence[str]] = None,
        *,
        skill_plan: Optional[SkillDeliveryPlan] = None,
        method_context_max_chars: Optional[int] = None,
    ) -> str:
        """渲染 prompt。签名中刻意不含 ground_truth（§4 M8 字段 7，防泄漏）。

        **v6 头部同源纪律**（§5.3）：`route_header` 由调用方从
        `scene_route` + `question_tool_scope` + `EvidenceProfile` 派生，且与
        `scene_summary` **同源** —— 杜绝"头部 fallback、摘要 full_3d"的自相矛盾
        （v5 实测过：12/32 题的 prompt 里只剩 1 个 Tool，头部却说 full_3d）。

        v6 不再有 `scale_known` 这种全局布尔：米制可用性由证据门逐题决定，
        头部必须写清"米制证据门是否通过 + 缺失哪些子条件"（§13.3）。

        v9（§13.5）：`skill_plan` 是**已算好**的交付计划（调用方持有它才能记交付
        记录）；不传时用 `skills` + `method_context_max_chars` 现算一个。渲染的条目
        就是计划里的条目 —— 于是"prompt 里的方法正文"与"trace 里的正文 hash"
        必然同源。
        """
        if not route_header and scope:
            from skill3d.tools import REGISTRY

            route_header = REGISTRY.docs_header(
                scope, available_artifacts or (),
                question_type=question_type, gate_passed=gate_passed,
                evidence_profile=evidence_profile, gate_missing=gate_missing)
        plan = skill_plan
        if plan is None and skills:
            plan = plan_delivery(skills, max_chars=method_context_max_chars)
        tpl = self._env.from_string(_PROGRAM_SYNTH)
        return tpl.render(
            question=question,
            scene_summary=scene_summary,
            scene_frame=scene_frame,
            tool_docs=tool_docs,
            route_header=route_header or "(Scope unspecified: use only tools independent of reconstruction artifacts.)",
            question_type=question_type,
            answer_unit=CANONICAL_UNIT_BY_QUESTION_TYPE.get(question_type, ""),
            task_definition=_TASK_DEFINITIONS.get(question_type, ""),
            options=list(options) if options else None,
            skill_entries=[e.text for e in (plan.entries if plan is not None else [])],
        )

    def render_messages(self, **kwargs) -> list[dict]:
        return [{"role": "user", "content": self.render(**kwargs)}]


def round_action_instructions(question_type: str, *, finalizing: bool = False) -> str:
    """反馈末尾的公共执行合同；不包含选项答案或 Skill 方法。"""
    unit = CANONICAL_UNIT_BY_QUESTION_TYPE.get(question_type)
    unit_expr = repr(unit) if unit else "unit"
    lines = ["Required action this round: output exactly one ```python code block. "
             "Repair executable statements; do not merely edit comments or repeat the previous program."]
    if finalizing:
        lines.append("This is the final answer request within the budget. "
                     "Only a zero-tool program is allowed; do not call YieldObservations.")
    else:
        lines.append(
            "Call only currently documented tools. Do not repeat a deterministically invalid call "
            "with unchanged arguments and evidence premises. Request new observations with "
            "`return YieldObservations([actual_result_id], 'reason')`; old variables no longer exist.")
    lines.append(
        "First define answer in code from the question and valid evidence "
        "(an option letter or a finite number for this question). "
        "Do not submit the variable names below as literal answers."
        + ("" if unit else " Define a valid unit according to the question."))
    lines.append(
        "For an answer genuinely based entirely on original images, submit in a zero-tool segment: "
        f"return ReturnAnswer(AnswerPayload(value=answer, unit={unit_expr}, "
        "basis='visual_estimate', used_result_ids=[], derivation=None))")
    lines.append(
        "For combined visual and valid tool evidence, put the actual result IDs used in used_ids, then submit: "
        f"return ReturnAnswer(AnswerPayload(value=answer, unit={unit_expr}, "
        "basis='mixed', used_result_ids=used_ids, derivation=None)). "
        "Pure tool derivations require tool_derived and a replayable derivation under the shared contract.")
    lines.append(
        "Invalidated results and rejected answers cannot serve as evidence. "
        "Do not bypass validation with short-form answers or by merely changing basis. "
        "Answer when images are available; do not submit abstain. Do not call tools after ReturnAnswer.")
    return "\n".join(lines)


def build_image_messages(text: str, frames, *, max_images: int = 32,
                         jpeg_quality: int = 80,
                         extra: Optional[Sequence] = None) -> list[dict]:
    """把文本 prompt + **同一 FrameSet 的 32 帧**（+ 可选派生图）装成多模态 messages。

    在线链必须让模型**看到**重建所依据的帧（否则退化成纯 Tool 查询，B-1）。
    硬约束 26：M8 必须多模态，不得纯文本生成程序、不得静默丢帧
    —— 帧数与 `frame_ids` 不匹配时抛错而不是悄悄少送。
    图像以 JPEG data URL 内联；`max_images` 与 §4 M1 的 32 帧采样对齐。
    实测：32 帧 @max_pixels=131072 ≈ 9.7k prompt tokens（单卡 FP8 1.4s）。

    v9 §9.4：`extra` 是本次要一并送达的**派生图**（`inspect_frames` 的裁剪结果），
    顺序在请求里排在原帧**之前**（与 runner 的 `ImageRound.image_ids` 一一对应）。
    超服务图像上限时抛错（不静默丢图）；被省略的图由账本记为 unobserved。
    """
    import base64

    import cv2
    import numpy as np

    content: list[dict] = [{"type": "text", "text": text}]
    arr = list(frames)
    if extra:
        # 派生图先入（对应 runner 已声明的布局顺序）
        arr = [np.asarray(img) for _, img in extra] + arr
    n = len(arr)
    if n > max_images:
        # 硬约束 26：不得静默丢帧 —— 帧数超过在线模型图像上限必须报错
        raise ValueError(
            f"M8 收到 {n} 帧（含派生图）但 max_images={max_images}"
            "（online_model.n_frames 上限）：禁止静默丢帧/丢图，"
            "请统一 FrameSet 帧数/布局或调大上限（硬约束 21/26；§9.4）")
    if n == 0:
        # 硬约束 26：M8 必须多模态（32 帧 + 文本），零帧等于退化成纯文本
        raise ValueError(
            "M8 收到 0 帧：硬约束 26 要求多模态输入，禁止退化为纯文本生成程序")
    for i in range(n):
        img = np.asarray(arr[i])
        try:
            ok, buf = cv2.imencode(".jpg", cv2.cvtColor(img, cv2.COLOR_RGB2BGR),
                                   [int(cv2.IMWRITE_JPEG_QUALITY), int(jpeg_quality)])
        except Exception as exc:  # noqa: BLE001 - 空帧/损坏/非法 dtype 都会在这里炸
            raise ValueError(
                f"M8 第 {i}/{n} 帧无法编码为 JPEG（{type(exc).__name__}: {exc}）："
                "禁止静默丢帧，请检查帧数组（dtype/shape/损坏）后重跑（硬约束 21/26）"
            ) from exc
        if not ok:
            # 硬约束 26：编码失败也不许静默跳过（少一帧就破坏了 FrameSet 对齐）
            raise ValueError(
                f"M8 第 {i}/{n} 帧 JPEG 编码失败：禁止静默丢帧，"
                "请检查帧数组（dtype/shape/损坏）后重跑（硬约束 21/26）")
        b64 = base64.b64encode(buf.tobytes()).decode("ascii")
        content.append({"type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{b64}"}})
    return [{"role": "user", "content": content}]
