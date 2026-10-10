"""v9 P5：与规范**直接冲突**的三处修正（§17.3 编码前核验清单）。

1. §11.3（规范 L429 原文）："无围栏但完整程序**不计**解析恢复" ——
   此前该路径返回回退标记，把正常产出记成 `m8_parse_recovered`；
2. §14.3：候选准入必须在**该题型完整 inner 面板**上判定，`outer_holdout` 是快照
   冻结后的独立验证证据 —— 此前准入消费的是 outer，用 holdout 选代会让它失效；
3. §14.3 与 G-31 简化阶段门的冲突：简化不能导致"没有准入证据"。
"""

from __future__ import annotations

from skill3d.synthesis.program_assembler import extract_program_source_ex

PROGRAM = "def solve(ctx):\n    return ReturnAnswer(4)\n"


# --------------------------------------------- §11.3 无围栏不计解析恢复 ----

def test_unfenced_complete_program_is_not_parse_recovery():
    """§429："无围栏但完整程序不计解析恢复"。"""
    source, recovered = extract_program_source_ex(PROGRAM)
    assert source.strip() == PROGRAM.strip()
    assert recovered is False, "无围栏的完整程序属正常产出，不得记 m8_parse_recovered"


def test_fenced_program_is_still_the_clean_path():
    _, recovered = extract_program_source_ex(f"```python\n{PROGRAM}```\n")
    assert recovered is False


def test_unclosed_fence_prefix_is_still_recovery():
    """截断后的可解析前缀**是**回退（§11.3 允许并标记）。"""
    _, recovered = extract_program_source_ex(f"```python\n{PROGRAM}")
    assert recovered is True


def test_truncated_body_recovery_is_still_flagged():
    truncated = "```python\ndef solve(ctx):\n    x = 1\n    y = 2\n"
    _, recovered = extract_program_source_ex(truncated)
    assert recovered is True


# --------------------------------------------- §14.3 准入消费 inner 面板 ----
