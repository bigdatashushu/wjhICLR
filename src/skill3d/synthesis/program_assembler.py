"""Program Assembler（§4 M8）：从模型输出解析 program 并组装 EpisodeProgram。

解析失败抛 SynthesisError，供上层有限次重生成（§4 M8 字段 9）。
"""

from __future__ import annotations

import ast
import hashlib
import re
from typing import Optional, Sequence

from skill3d.schemas import EpisodeProgram

_PY_BLOCK = re.compile(r"```python\s*\n(.*?)```", re.DOTALL)
_GENERIC_BLOCK = re.compile(r"```\s*\n(.*?)```", re.DOTALL)


class SynthesisError(RuntimeError):
    """program 无法解析；上层据此触发有限次重生成。"""


def extract_program_source(text: str) -> str:
    """提取 ```python 代码块；无显式代码块但整体可解析时回退全文。"""
    m = _PY_BLOCK.search(text)
    if m is None:
        m = _GENERIC_BLOCK.search(text)
    source = m.group(1) if m is not None else text.strip()
    if not source:
        raise SynthesisError("模型输出为空，无法提取 program")
    try:
        ast.parse(source)
    except SyntaxError as exc:
        raise SynthesisError(f"program 语法错误: {exc}") from exc
    return source


def assemble_program(
    model_output: str,
    skill_semver_used: Optional[Sequence[str]] = None,
    intended_answer_slot: str = "ReturnAnswer",
) -> EpisodeProgram:
    source = extract_program_source(model_output)
    program_id = hashlib.sha256(source.encode()).hexdigest()[:16]
    return EpisodeProgram(
        program_id=program_id,
        program_source=source,
        skill_semver_used=list(skill_semver_used or []),
        intended_answer_slot=intended_answer_slot,
    )
