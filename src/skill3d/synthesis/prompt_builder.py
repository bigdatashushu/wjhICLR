"""Program Synthesizer prompt 构建（§4 M8）：Jinja2 版本化模板。

只暴露 Tool 文档与 SkillSpec 模板，绝不暴露 ground_truth（防泄漏）。
模板字符串内联于本模块，按模板名版本化。
"""

from __future__ import annotations

from typing import Optional, Sequence

from jinja2 import Environment, StrictUndefined

from skill3d.schemas import SkillSpec

TEMPLATE_VERSION = "program_synth_v2"

# program_synth_v2.j2（内联版本化模板）
#
# v1→v2（2026-09-21 真实缺陷修复）：outer_holdout 32 题里有 9 题（28%）M8 输出退化
# 成"反复自我辩论是否该 abstain"的重复段落，撑爆 max_tokens 后**代码块未闭合**，
# 旧解析器直接放弃 → episode 记 unavailable。根因是 v1 只写了"必须调用 ReturnAnswer"
# 与"无法获得就 abstain"，却没说 abstain 本身就是合法的最终动作，模型于是在
# "必须给数值"与"不能猜"之间循环。v2 明确三件事，**不放松任何门**：
#   1. `ReturnAnswer("abstain")` 是合法的最终答案（主榜按错计，不刷分）；
#   2. 只输出一个代码块，不写长篇解释、不自我辩论；
#   3. 不存在隐藏全局变量（`objects` 等），对象清单只能用 `list_objects()` 取。
_PROGRAM_SYNTH_V2 = """你是一名空间推理 Coding Agent。请为下面的题目生成一段 Python program，
通过编排已注册的 Tool 完成推理，最后一行必须调用 ReturnAnswer(答案)。

## 当前可用产物（route 裁剪，只暴露产物齐备的 Tool）
{{ route_header }}

## 场景摘要
{{ scene_summary }}

坐标系: {{ scene_frame }}；尺度已知: {{ scale_known }}

## 可用 Tool（只允许调用以下函数）
{{ tool_docs }}

{% if skills %}
## 参考 Skill 模板（题型级程序合成模板，非答案）
{% for s in skills %}
### {{ s.skill_id }}@{{ s.semver }} (task_type={{ s.task_type }})
{{ s.description }}
模板:
{{ s.call_graph_template }}
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

## 输出纪律（必须遵守）
- 只输出**一个** ```python 代码块；代码块之外不要写解释、不要写推理过程。
- 最后一行调用一次 `ReturnAnswer(...)`（且只调用一次）。
- **MCA 题**：工具/中间结果是方向词或类别词时，必须**先与选项文本逐条对应**，
  再返回选项字母（例：工具返回 `behind`、选项 B 是 "back" → `ReturnAnswer("B")`）。
  题目里的参照系（站在哪、面向谁）必须**从题面读取**，不要自己假设；
  题干说的面向某物体，用 `relative_direction_of(observer=…, facing_at=…, target=…)`。
- **`ReturnAnswer` 只记录答案，不会中止 program**：写了 `ReturnAnswer(...)` 之后
  代码仍会继续执行。所以不要用"中途 ReturnAnswer 提前返回"的写法（后面若继续用
  空列表/未定义变量会直接抛错）。请把答案先算进变量、**在最后一行**只调用一次
  `ReturnAnswer(变量)`；分支里改写变量，不要提前 return。
- 若当前 route/工具下确实拿不到该量：**直接** `ReturnAnswer("abstain")` 结束。
  `"abstain"` 是合法的最终答案（主榜按错计），**优于**编造数值；
  不要在 program 里反复讨论要不要 abstain —— 判定一次就直接作答。
- **不存在隐藏全局变量**：没有 `objects`/`scene`/`frames` 这类可直接读的变量。
  对象清单只能来自 `list_objects()`；某对象的质心用 `object_centroid(对象id)`。
  把 `list_objects()` 返回的对象 id 字符串列表当作唯一入口，不要索引不存在的字典。

## 约束
- 只允许 import numpy / scipy / math / statistics；禁止其他 import
- 禁止 eval/exec/__import__/open/文件写/网络
- 禁止对 show / ReturnAnswer / tools / scene / frames 赋值
- 用 ```python ... ``` 代码块输出 program
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
        scale_known: bool,
        tool_docs: str,
        options: Optional[Sequence[str]] = None,
        skills: Optional[Sequence[SkillSpec]] = None,
        route: str = "",
        available_artifacts: Optional[Sequence[str]] = None,
        route_header: str = "",
        allowed_metric_tasks: Optional[Sequence[str]] = None,
        question_type: str = "",
    ) -> str:
        """渲染 prompt。注意：签名中刻意不含 ground_truth（§4 M8 字段 7）。

        `route` / `available_artifacts` 会显式写进 prompt 头部（§4 M6 字段 7：
        "当前 route=…，可用产物=…"），使模型只能编排当前 route 下可用的 Tool，
        从源头避免"调用了产物缺失的 Tool"这类静默错答。

        v4 HC33：头部还要写"允许的米制题型=…"——米制 Tool 按题型授权，模型必须
        知道本题是否放行米制数值，否则会拿相对单位当米制作答。
        """
        if self.template_version != TEMPLATE_VERSION:
            raise ValueError(f"未知模板版本: {self.template_version}")
        if not route_header and route:
            from skill3d.tools import REGISTRY

            route_header = REGISTRY.docs_header(
                route, available_artifacts or (),
                allowed_metric_tasks=allowed_metric_tasks,
                question_type=question_type)
        tpl = self._env.from_string(_PROGRAM_SYNTH_V2)
        return tpl.render(
            question=question,
            scene_summary=scene_summary,
            scene_frame=scene_frame,
            scale_known=scale_known,
            tool_docs=tool_docs,
            route_header=route_header or "（未声明 route：仅使用不依赖重建产物的 Tool）",
            options=list(options) if options else None,
            skills=list(skills) if skills else [],
        )

    def render_messages(self, **kwargs) -> list[dict]:
        return [{"role": "user", "content": self.render(**kwargs)}]


def build_image_messages(text: str, frames, *, max_images: int = 32,
                         jpeg_quality: int = 80) -> list[dict]:
    """把文本 prompt + **同一 FrameSet 的 32 帧**装成多模态 messages（§4 M8）。

    在线链必须让模型**看到**重建所依据的帧（否则退化成纯 Tool 查询，B-1）。
    硬约束 26：M8 必须多模态，不得纯文本生成程序、不得静默丢帧
    —— 帧数与 `frame_ids` 不匹配时抛错而不是悄悄少送。
    图像以 JPEG data URL 内联；`max_images` 与 §4 M1 的 32 帧采样对齐。
    实测：32 帧 @max_pixels=131072 ≈ 9.7k prompt tokens（单卡 FP8 1.4s）。
    """
    import base64

    import cv2
    import numpy as np

    content: list[dict] = [{"type": "text", "text": text}]
    arr = list(frames)
    n = len(arr)
    if n > max_images:
        # 硬约束 26：不得静默丢帧 —— 帧数超过 vLLM 图像上限是配置错误，必须报错
        raise ValueError(
            f"M8 收到 {n} 帧但 max_images={max_images}（vLLM --limit-mm-per-prompt "
            "image 上限）：禁止静默丢帧，请统一 FrameSet 帧数或调大上限（硬约束 21/26）")
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
