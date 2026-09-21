"""Program Assembler（§4 M8）：从模型输出解析 program 并组装 EpisodeProgram。

解析失败抛 SynthesisError，供上层有限次重生成（§4 M8 字段 9）。

健壮性（2026-09-21 真实缺陷修复）：模型偶发在**代码块未闭合**时就因
`max_tokens` 截断或退化成重复自辩（实测 20 KB 全是重复段落的注释）而停止输出。
此前的实现只认"```python … ```"这种闭合块，遇到未闭合时直接回退整段文本 →
`ast.parse` 在第 1 行（```python 本身）即失败 → 整个 episode 记 `unavailable`
（outer_holdout 32 题里 9 题，占 28%）。
现在按"**最保守的可用后缀/前缀**"顺序回退，绝不臆造代码：

1. 闭合的 ```python 块；
2. 闭合的任意 ``` 块；
3. **未闭合的 ```python 围栏**：取围栏之后的全部内容，再逐行**从尾部**丢弃
   直到 `ast.parse` 通过（截断的 program 只保留可解析前缀）；
4. 整段文本（原有行为）。

回退只影响"能否拿到一段可执行 program"，**不放松任何门**：拿不到可解析代码
仍然抛 SynthesisError；解析出的 program 照旧过 M9 AST 白名单与 M10 沙箱。
"""

from __future__ import annotations

import ast
import hashlib
import re
from typing import Optional, Sequence

from skill3d.schemas import EpisodeProgram

_PY_BLOCK = re.compile(r"```python\s*\n(.*?)```", re.DOTALL)
_GENERIC_BLOCK = re.compile(r"```\s*\n(.*?)```", re.DOTALL)
# 代码围栏起始行（闭合与否都从这里开始识别）
_FENCE_OPEN = re.compile(r"```[ \t]*(?:python|py)?[ \t]*\r?\n")
# 截断恢复时最多从尾部丢弃的行数（防止把整段输出逐行削成残片）
_MAX_TRIM_LINES = 200


class SynthesisError(RuntimeError):
    """program 无法解析；上层据此触发有限次重生成。"""


def _parses(source: str) -> bool:
    try:
        ast.parse(source)
    except SyntaxError:
        return False
    return True


def _recover_truncated(source: str) -> Optional[str]:
    """未闭合代码块 → 从尾部逐行丢弃，返回**首个可解析前缀**（无则 None）。

    截断只发生在输出末尾，因此被丢弃的一定是尾部残行；保留下来的前缀是模型
    自己写出的完整代码。丢弃后若出现"只有注释/空行"的前缀也能解析，但那种
    program 没有 ReturnAnswer，执行期自然判无答案，不会伪造答案。
    """
    lines = source.splitlines()
    limit = min(len(lines), _MAX_TRIM_LINES)
    for drop in range(limit + 1):
        candidate = "\n".join(lines[: len(lines) - drop]).rstrip()
        if candidate and _parses(candidate):
            return candidate
    return None


def _open_fence_remainder(text: str) -> Optional[str]:
    """取**未闭合**围栏之后的剩余文本（含截断内容）；无未闭合围栏返回 None。"""
    last = None
    for m in _FENCE_OPEN.finditer(text):
        last = m
    if last is None:
        return None
    tail = text[last.end():]
    if "```" in tail:       # 后面还有闭合围栏 → 不是"未闭合"场景
        return None
    return tail


def extract_program_source(text: str) -> str:
    """提取 program 源码。

    顺序：闭合 ```python 块 → 闭合 ``` 块 → 未闭合围栏的**可解析前缀** →
    整段文本（需自洽可解析）。全部失败抛 SynthesisError。
    """
    raw = str(text or "")
    if not raw.strip():
        raise SynthesisError("模型输出为空，无法提取 program")

    m = _PY_BLOCK.search(raw)
    if m is None:
        m = _GENERIC_BLOCK.search(raw)
    if m is not None:
        source = m.group(1)
        if not source.strip():
            raise SynthesisError("代码块为空，无法提取 program")
        if not _parses(source):
            # 闭合块内仍有语法错误 → 试着丢掉截断残行（同未闭合路径的保守回退）
            recovered = _recover_truncated(source)
            if recovered is None:
                exc = _syntax_error(source)
                raise SynthesisError(f"program 语法错误: {exc}") from exc
            return recovered
        return source

    remainder = _open_fence_remainder(raw)
    if remainder is not None:
        recovered = _recover_truncated(remainder)
        if recovered is not None:
            return recovered
        exc = _syntax_error(remainder)
        raise SynthesisError(
            f"program 代码块未闭合且无可解析前缀（疑似 max_tokens 截断/输出退化）: {exc}"
        ) from exc

    # 无围栏：整段文本自洽时才接受（保持原有回退）
    stripped = raw.strip()
    if not _parses(stripped):
        exc = _syntax_error(stripped)
        raise SynthesisError(f"program 语法错误: {exc}") from exc
    return stripped


def _syntax_error(source: str) -> SyntaxError:
    try:
        ast.parse(source)
    except SyntaxError as exc:  # pragma: no cover - 调用点保证必然抛错
        return exc
    raise AssertionError("_syntax_error 只在源码无法解析时调用")  # pragma: no cover


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
