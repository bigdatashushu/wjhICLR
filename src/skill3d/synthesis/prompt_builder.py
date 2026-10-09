"""Current Program Synthesizer prompt contract.

Only Tool documentation and complete ``SkillSpecV11`` sources are exposed. Ground
truth is deliberately absent. Historical prompts are recovered from Git rather
than selected at runtime.
"""

from __future__ import annotations

from typing import Optional, Sequence

from jinja2 import Environment, StrictUndefined

from skill3d.schemas import SkillSpecV11
from skill3d.skills.delivery import SkillDeliveryPlan, plan_delivery

PROMPT_TEMPLATE_VERSION = "program_synth_v11_2"


# These definitions are part of the shared task contract, not solving recipes.
_TASK_DEFINITIONS = {
    "object_counting": "统计题面指定类别与范围内的对象数量，答案为非负整数。",
    "object_abs_distance": "估计题面点名的两个对象之间的最近距离，单位按题目要求。",
    "object_rel_distance": "选择距离题面参照对象最近的候选类别；参照对象不是相机。",
    "object_size_estimation": "估计目标对象长、宽、高中的最大物理维度，单位按题目要求。",
    "room_size_estimation": "估计题面指定空间的地面面积，答案单位为平方米。",
    "object_rel_direction": "判断站在题面指定位置、面向指定对象时目标的方向，答案词表由选项决定。",
    "obj_appearance_order": "判断题面指定类别在给定视频中的首次出现顺序。",
    "route_planning": "判断沿题面给定路线行进时的转向，答案为当前选项中的完整动作序列。",
}

_PROGRAM_SYNTH = """你是一名空间推理 Coding Agent。根据题目、图像和当前可用工具，
生成并执行一段 Python program，最终通过 `return ReturnAnswer(答案)` 提交。

## 公共执行规则
- 只输出一个 ```python 代码块，代码块之外不要写解释。
- 可以顶层直接写代码，也可以定义 `def solve(ctx): ...`，host 会自动调用该入口。
- 只允许 import numpy / scipy / math / statistics；禁止其他 import，以及 eval/exec/__import__/open/文件写/网络。
- 禁止对 show / ReturnAnswer / YieldObservations / AnswerPayload / tools / scene / frames 赋值。
- `return ReturnAnswer(答案)` 提交暂存答案并立即结束当前程序片段；框架通过几何与有效证据验收后才结束本题。验收失败时撤销提交，反馈检查项、受影响结果及仍有效的观察，由同一模型在剩余预算内修正。
- `return YieldObservations([result_id, ...], "理由")` 结束本段程序，将点名结果的实际返回值送入下一轮请求；求解轮数和重试次数由框架预算控制。
- 只能依据实际收到的图像和有效工具观察推理；已失效结果不得继续支持答案。
- 有图必答：有可读图像时，证据不足可按视觉估计作答，并如实标记依据；`ReturnAnswer("abstain")` 不是合法答案。
- 工具缺失、失败或无效返回不等于零；不得编造几何、尺度、观察或 result_id。

## 答案载荷与观察接口
- `AnswerPayload` 已在运行环境中提供，无需 import。完整提交为
  `return ReturnAnswer(AnswerPayload(value=答案, unit=单位, basis=依据, used_result_ids=[], derivation=None))`。
  unit 取 option/count/m/cm/m2；basis 取 visual_estimate/tool_derived/mixed。
  纯视觉估计使用 visual_estimate，used_result_ids 为空且 derivation 为 None；
  结合视觉与工具使用 mixed，并列出实际使用的结果 ID。
- tool_derived 还需声明非空 used_result_ids 和 derivation：
  `{"op": 操作名, "input_result_ids": [实际结果ID], "parameters": {变换参数}}`。
  op 仅支持 field/convert/count/sort/argmin/option_map，框架会按真实 ToolResult
  确定性重放并核对答案。参数格式：
  field=`{"field":"a.b.0"}`；convert=`{"field":"distance_m","from_unit":"m","to_unit":"cm"}`；
  count=`{"field":"items"}`；argmin 可对一个字典或多个结果取最小值，多个结果需给
  `labels`；sort 与 argmin 类似，可给 `descending`；option_map=
  `{"field":"direction","mapping":{"left":"A","right":"B"}}`。
  参数只能是这些结构化字段，不能放代码或表达式。无法用一个登记操作完整证明答案时，
  使用 mixed，不得声明 tool_derived。
- 兼容 `ReturnAnswer(value)`，此时依据保守登记为 mixed。
  ReturnAnswer 只接受一个参数，不能直接传 basis、unit 等关键字。
- 工具返回 dict 时，框架附加 `result_id` 字符串；可用
  `return YieldObservations([result["result_id"]], "理由")` 请求回传。
  list/数值/bool 返回值不附加 ID；`return YieldObservations([], "理由")`
  回传本 episode 仍有效的成功工具结果及其 ID；已失效结果不会作为观察回传。

## 当前题目与答案要求
{{ question }}
本题题型: {{ question_type or "未分类" }}
{% if task_definition %}题义: {{ task_definition }}
{% endif %}{% if options %}选项:
{% for opt in options %}{{ opt }}
{% endfor %}MCA 题：提交与选项文本对应的选项字母（如 "A"）。
{% else %}NA 题：答案值（短写 value 或 AnswerPayload.value）必须为一个有限数值，不得为列表或字典。
{% endif %}
题面中的对象角色、空间范围、参照系和要求单位必须保持一致。
工具的单位、坐标与计算定义以当前接口文档和有效返回为准；近似结果应按其实际定义解释。
米转厘米乘以 100，面积转换按长度比例的平方进行；已为目标单位的值不再换算。
缺少有效尺度时不得将归一化几何量冒充米制测量；视觉估计不得冒充工具测量。

## 当前证据状态
{{ route_header }}
场景摘要: {{ scene_summary }}
坐标与尺度的可用性以当前证据状态和工具合同为准。

## 获准工具及接口文档
{{ tool_docs }}
只允许调用当前文档列出的工具；签名、返回字段与可用性以该文档为准。
方法指导不会新增工具权限；具体对象绑定与调用前置条件仍由框架检查。
{% if skill_entries %}
## 参考方法
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
            route_header=route_header or "（未声明 scope：仅使用不依赖重建产物的 Tool）",
            question_type=question_type,
            task_definition=_TASK_DEFINITIONS.get(question_type, ""),
            options=list(options) if options else None,
            skill_entries=[e.text for e in (plan.entries if plan is not None else [])],
        )

    def render_messages(self, **kwargs) -> list[dict]:
        return [{"role": "user", "content": self.render(**kwargs)}]


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
        # 硬约束 26：不得静默丢帧 —— 帧数超过 vLLM 图像上限是配置错误，必须报错
        raise ValueError(
            f"M8 收到 {n} 帧（含派生图）但 max_images={max_images}"
            "（vLLM --limit-mm-per-prompt image 上限）：禁止静默丢帧/丢图，"
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
