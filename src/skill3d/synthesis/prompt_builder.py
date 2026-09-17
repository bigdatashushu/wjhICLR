"""Program Synthesizer prompt 构建（§4 M8）：Jinja2 版本化模板。

只暴露 Tool 文档与 SkillSpec 模板，绝不暴露 ground_truth（防泄漏）。
模板字符串内联于本模块，按模板名版本化。
"""

from __future__ import annotations

from typing import Optional, Sequence

from jinja2 import Environment, StrictUndefined

from skill3d.schemas import SkillSpec

TEMPLATE_VERSION = "program_synth_v1"

# program_synth_v1.j2（内联版本化模板）
_PROGRAM_SYNTH_V1 = """你是一名空间推理 Coding Agent。请为下面的题目生成一段 Python program，
通过编排已注册的 Tool 完成推理，最后一行必须调用 ReturnAnswer(答案)。

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
    ) -> str:
        """渲染 prompt。注意：签名中刻意不含 ground_truth（§4 M8 字段 7）。"""
        if self.template_version != TEMPLATE_VERSION:
            raise ValueError(f"未知模板版本: {self.template_version}")
        tpl = self._env.from_string(_PROGRAM_SYNTH_V1)
        return tpl.render(
            question=question,
            scene_summary=scene_summary,
            scene_frame=scene_frame,
            scale_known=scale_known,
            tool_docs=tool_docs,
            options=list(options) if options else None,
            skills=list(skills) if skills else [],
        )

    def render_messages(self, **kwargs) -> list[dict]:
        return [{"role": "user", "content": self.render(**kwargs)}]
