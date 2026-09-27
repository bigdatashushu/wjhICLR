"""Program Synthesizer prompt 构建（§4 M8）：Jinja2 版本化模板。

只暴露 Tool 文档与 SkillSpec 模板，绝不暴露 ground_truth（防泄漏）。
模板字符串内联于本模块，按模板名版本化。

v9 变更（§13.5/§13.6）：方法条目不再在模板里逐字段拼接，而是渲染
`skills.delivery.SkillDeliveryPlan` 里**已经是最终文本**的条目 —— 这样 prompt 里
出现的方法正文与 trace 里记的 `delivered_content_sha256` 出自同一个函数，
"已交付正文 hash"才真的能证明"模型收到的是这段文本"。
上下文放不下的条目由交付计划**整条**丢弃（§13.5：不得截断后仍称完整交付）。
"""

from __future__ import annotations

from typing import Optional, Sequence

from jinja2 import Environment, StrictUndefined

from skill3d.schemas import SkillSpec
from skill3d.skills.delivery import SkillDeliveryPlan, plan_delivery

# program_synth_v8（内联版本化模板）
#
# v7→v8 的唯一变更：方法条目走交付计划（§13.5 完整正文 + 上下文上限 + 交付 hash）。
# 其余口径（v7 的"取消拒答""两个控制接口都是终结操作""相对距离官方口径"）逐字保留。
TEMPLATE_VERSION = "program_synth_v8"

# program_synth_v7（内联版本化模板）
#
# v6→v7 的三处口径变更（对应《系统架构v7》D1/D2/D4）：
#   1. **取消拒答**：`ReturnAnswer("abstain")` 不再被接受为终态。有可读图片时，
#      证据不足只改变**求解方式**（换工具，或直接依据图片做视觉估计），不改变
#      "必须给出答案"这一条。v6 里 32% 的 episode 走 abstain → 全部零分。
#   2. **两个控制接口都是终结操作**：`return ReturnAnswer(...)` 立即结束；
#      `return YieldObservations([...], "理由")` 结束当前片段并把**实际结果**
#      回灌给同一个模型，由模型写下一段程序。这是 D4"根据执行反馈调整程序"的
#      实现方式 —— 成功结果里的歧义、需要看新图、方法不适用都可以让出。
#   3. **相对距离题的官方口径纠正**（v7 §2.2，依据原论文附录 B.1）：
#      比较**题面参照对象**到各候选对象的距离，不是"观察点→候选"。
#
# v5→v6 保留的纪律（不放松任何门）：只输出一个代码块、不自我辩论、
# 不存在隐藏全局变量、AST 白名单、单位换算。
_PROGRAM_SYNTH_V3 = """你是一名空间推理 Coding Agent。请为下面的题目生成一段 Python program，
通过编排已注册的 Tool 完成推理，**并以 `return ReturnAnswer(答案)` 结束**。

## 当前可用产物与证据状态（scope + EvidenceProfile 裁剪，只暴露真正可执行的 Tool）
{{ route_header }}

## 场景摘要
{{ scene_summary }}

坐标系: {{ scene_frame }}；本题题型: {{ question_type or "未分类" }}

## 可用 Tool（只允许调用以下函数）
{{ tool_docs }}

{% if skill_entries %}
## 参考 Skill 模板（题型级程序合成模板，非答案）
{% for e in skill_entries %}
{{ e }}
{% endfor %}
{% endif %}

## 题目
{{ question }}
{% if options %}
选项:
{% for opt in options %}
{{ opt }}
{% endfor %}
MCA 题：ReturnAnswer 只接受选项字母（如 "A"）。
{% else %}
NA 题：ReturnAnswer 接受数值。
{% endif %}

## 两个控制接口（都是终结操作，调用即结束本段程序）
1. `return ReturnAnswer(答案)` —— 提交答案，立即结束本题。
2. `return YieldObservations([result_id, ...], "理由")` —— **不提交答案**，
   结束本段程序，把你点名的 Tool 结果的**实际返回值**回灌给你，你再写下一段
   程序。适用于：结果有歧义需要判断、需要看某张裁剪图、某个方法不适用、
   或需要先拿到一批数再决定下一步。可以多次让出（受预算限制）。

**有图必答（硬要求）**：图片是可读的，因此**必须给出答案**。
不要输出 `abstain`、也不要因为工具缺失/证据不足而拒绝作答 ——
证据不足只改变**求解方式**：换一个还能用的工具，或者直接依据你在图片里
看到的内容给出**视觉估计**（`basis` 记 `visual_estimate`）。
`ReturnAnswer("abstain")` 不是合法答案，会被判为违反作答纪律。

## 输出纪律（必须遵守）
- 只输出**一个** ```python 代码块；代码块之外不要写解释、不要写推理过程。
- **MCA 题**：工具/中间结果是方向词或类别词时，必须**先与选项文本逐条对应**，
  再返回选项字母（例：工具返回 `behind`、选项 B 是 "back" → `ReturnAnswer("B")`）。
  **不要把选项字母硬编码**：先把所有候选都算出来，再按题面给的选项文本匹配。
- 题目里的参照系（站在哪、面向谁）必须**从题面读取**，不要自己假设。
- 计算完就立刻 `return ReturnAnswer(...)`，不要再调用任何 Tool。
- 不要在代码里写"如果不确定就……"的多轮自我辩论；不确定就给最佳估计。

## 题目口径（按题型对照，写错口径会系统性答错）
- **`object_rel_distance`（相对距离）**：题面问的是「哪个选项**离题面提到的参照
  对象**最近」。等价做法是：对每个候选类别取它**离参照对象最近的那个实例**，
  比较这些距离取最小。**不要**改成"相机→候选"距离，也不要改成对象质心距离
  以外的东西 —— 这是官方口径。
- **`object_abs_distance`（绝对距离）**：题面**点名的两个对象之间**的最近距离
  （米）。不是相机到对象的距离。
- **`object_size_estimation`（尺寸）**：对象**最长边**，题面常问厘米 → 米制值 ×100。
- **`room_size_estimation`（房间面积）**：平方米，不要再换算。
- **`object_rel_direction`（相对方向）**：用 `relative_direction_of(
  observer=站在哪, facing_at=面向谁, target=要判断的对象, difficulty=难度)`。
  **`difficulty` 必须按题面选项集合传**：题面只有 left/right → `easy`；
  只有 left/right/back → `medium`；四象限 → `hard`。
  传错会让工具返回一个**不在选项里**的方向词（例如 medium 题面没有 front），
  再映射到选项就一定错。**返回的方向词必须能与选项文本逐条对上**。
- **`obj_appearance_order`（外观顺序）**：按各类别**最早可见帧**排序；
  若某类别缺少检出，仍要结合选项集合与你在图片里看到的顺序给出最佳答案。
- **`object_counting`（计数）**：优先 `count_objects`；若它与你在图片里看到的
  明显不符，用 `YieldObservations` 取回实例明细后判断，**不要直接放弃**。

## 单位纪律（尺寸/距离/面积题必读）
- 所有米制 Tool 返回的都是**米 / 平方米**；题面常问**厘米**。
  题面问厘米时**必须 ×100**（问平方米时直接用，不要再乘）。
- 返回给 ReturnAnswer 的必须是**一个数**（或选项字母），不是列表/字典。
- **返回值键名以 Tool 文档为准**（例如 `surface_distance_between_objects` 返回
  `surface_distance_metric`，`count_objects` 返回 `count`）。不确定就先打印一次
  再取键，或让出观察。

## 约束
- 只允许 import numpy / scipy / math / statistics；禁止其他 import
- 禁止 eval/exec/__import__/open/文件写/网络
- 禁止对 show / ReturnAnswer / YieldObservations / tools / scene / frames 赋值
- 输出一个 ```python ... ``` 代码块。**两种写法都支持，任选一种**：
  1. 顶层直接写（推荐，最不容易出错）：最后用 `return ReturnAnswer(答案)`；
  2. 定义入口 `def solve(ctx): ...`：**host 会自动调用它**，你不需要自己调用。
  无论哪种写法，都必须真的执行到 `ReturnAnswer` / `YieldObservations`。
"""


class PromptBuilder:
    def __init__(self, template_version: str = TEMPLATE_VERSION) -> None:
        self.template_version = template_version
        self._env = Environment(undefined=StrictUndefined)

    def render(
        self,
        question: str,
        scene_summary: str,
        scene_frame: str,
        tool_docs: str,
        options: Optional[Sequence[str]] = None,
        skills: Optional[Sequence[SkillSpec]] = None,
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
        if self.template_version != TEMPLATE_VERSION:
            raise ValueError(f"未知模板版本: {self.template_version}")
        if not route_header and scope:
            from skill3d.tools import REGISTRY

            route_header = REGISTRY.docs_header(
                scope, available_artifacts or (),
                question_type=question_type, gate_passed=gate_passed,
                evidence_profile=evidence_profile, gate_missing=gate_missing)
        plan = skill_plan
        if plan is None and skills:
            plan = plan_delivery(skills, max_chars=method_context_max_chars)
        tpl = self._env.from_string(_PROGRAM_SYNTH_V3)
        return tpl.render(
            question=question,
            scene_summary=scene_summary,
            scene_frame=scene_frame,
            tool_docs=tool_docs,
            route_header=route_header or "（未声明 scope：仅使用不依赖重建产物的 Tool）",
            question_type=question_type,
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
