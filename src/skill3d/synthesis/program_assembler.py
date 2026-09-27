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

v6 §15.2/§19.3：**解析回退必须单列**为 `m8_parse_recovered`，与
`vllm_parse_error`（真的解析不出来）和 `vllm_service_error`（服务故障）区分开 ——
否则"回退救回来多少题"这件事在 trace 里看不见。

v6 §15.3：**退化输出触发重生成** —— 20 KB 重复段落 / 超长无意义输出 /
重复度超阈值（阈值 `[TODO_CALIBRATE]`）→ 不直接解析，先重生成一次；
重生成必须重过 M9 AST 检查。
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

# ---- 退化输出检测阈值（全部 TODO_CALIBRATE）----
TH_DEGENERATE_CHARS: int = 20000      # TODO_CALIBRATE: 输出长度上限（v5 实测单条 20311 字节）
TH_DEGENERATE_LINES: int = 200        # TODO_CALIBRATE: 行数上限（v5 实测 289 行）
TH_REPEAT_RATIO: float = 0.6          # TODO_CALIBRATE: 重复行占比上限
TH_MIN_DISTINCT_RATIO: float = 0.05   # TODO_CALIBRATE: 去重后行占比下限（几乎全重复）


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


def degenerate_reason(text: str) -> Optional[str]:
    """§15.3 退化输出检测：返回原因串（None = 未退化）。

    判据（**全部** `[TODO_CALIBRATE]`，起始参考值来自 v5 实测）：

    - 长度超 `TH_DEGENERATE_CHARS`（v5 实测单条 20311 字节、289 行全是注释）；
    - 行数超 `TH_DEGENERATE_LINES`；
    - 重复行占比 ≥ `TH_REPEAT_RATIO`（模型在"必须给数值"与"不能猜"之间自辩，
      反复输出同一段）；
    - 去重后行占比 < `TH_MIN_DISTINCT_RATIO`（几乎全是重复）。

    只**判定**不修改输出：调用方据此触发重生成（重生成必须重过 M9 AST）。
    """
    raw = str(text or "")
    if len(raw) > TH_DEGENERATE_CHARS:
        return f"输出长度 {len(raw)} 字符 > {TH_DEGENERATE_CHARS}（疑似退化重复）"
    lines = [ln.strip() for ln in raw.splitlines() if ln.strip()]
    if len(lines) > TH_DEGENERATE_LINES:
        return f"输出行数 {len(lines)} > {TH_DEGENERATE_LINES}（疑似退化重复）"
    if len(lines) >= 8:
        uniq = len(set(lines))
        if uniq / len(lines) < TH_MIN_DISTINCT_RATIO:
            return (f"去重后行占比 {uniq / len(lines):.3f} < "
                    f"{TH_MIN_DISTINCT_RATIO}（几乎全是重复行）")
        repeat_ratio = 1.0 - uniq / len(lines)
        if repeat_ratio >= TH_REPEAT_RATIO:
            return (f"重复行占比 {repeat_ratio:.3f} ≥ {TH_REPEAT_RATIO}"
                    "（模型自辩循环）")
    return None


def extract_program_source_ex(text: str) -> tuple[str, bool]:
    """同 `extract_program_source`，但额外返回**是否走了保守回退**
    （True → 上层把 `synthesis_source` 记为 `m8_parse_recovered`，§15.2）。"""
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
        if _parses(source):
            return source, False            # 干净路径：没有回退
        recovered = _recover_truncated(source)
        if recovered is None:
            exc = _syntax_error(source)
            raise SynthesisError(f"program 语法错误: {exc}") from exc
        return recovered, True

    remainder = _open_fence_remainder(raw)
    if remainder is not None:
        recovered = _recover_truncated(remainder)
        if recovered is not None:
            return recovered, True
        exc = _syntax_error(remainder)
        raise SynthesisError(
            f"program 代码块未闭合且无可解析前缀（疑似 max_tokens 截断/输出退化）: {exc}"
        ) from exc

    stripped = raw.strip()
    if not _parses(stripped):
        exc = _syntax_error(stripped)
        raise SynthesisError(f"program 语法错误: {exc}") from exc
    # §11.3（规范原文）："无围栏但完整程序**不计**解析恢复。"
    # 这里既没有截断，也没有丢弃内容，更没有补造答案 —— 只是没写围栏，属正常产出，
    # 因此返回 False（干净路径）。此前返回 True 会把这类 episode 记成
    # `m8_parse_recovered`，虚增回退率并低估 `vllm_ok`。
    return stripped, False


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
    source, _recovered = extract_program_source_ex(model_output)
    return _program_of(source, skill_semver_used, intended_answer_slot)


def assemble_program_ex(
    model_output: str,
    skill_semver_used: Optional[Sequence[str]] = None,
    intended_answer_slot: str = "ReturnAnswer",
) -> tuple[EpisodeProgram, bool]:
    """`assemble_program` + `recovered` 标记（v6 §19.3 `m8_parse_recovered`）。"""
    source, recovered = extract_program_source_ex(model_output)
    return _program_of(source, skill_semver_used, intended_answer_slot), recovered


def _program_of(source: str, skill_semver_used: Optional[Sequence[str]],
                intended_answer_slot: str) -> EpisodeProgram:
    program_id = hashlib.sha256(source.encode()).hexdigest()[:16]
    return EpisodeProgram(
        program_id=program_id,
        program_source=source,
        skill_semver_used=list(skill_semver_used or []),
        intended_answer_slot=intended_answer_slot,
    )
