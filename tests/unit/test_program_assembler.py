"""M8 program 解析单测（`synthesis/program_assembler.py`）。

回归背景（2026-09-21）：outer_holdout 32 题里 9 题（28%）因模型输出**代码块未闭合**
（`max_tokens` 截断 / 退化重复段落）而被旧解析器整段放弃 → episode 记 `unavailable`。
本文件锁定新的回退顺序，并锁死"绝不臆造代码 / 拿不到可解析代码仍必须抛错"的纪律。
"""

from __future__ import annotations

import ast

import pytest

from skill3d.synthesis.program_assembler import (
    SynthesisError,
    assemble_program,
    extract_program_source,
)


def test_closed_python_block_is_extracted():
    text = '说明文字\n```python\nn = 3\nReturnAnswer(n)\n```\n结尾'
    src = extract_program_source(text)
    assert src.startswith("n = 3")
    assert "说明文字" not in src


def test_closed_generic_block_is_extracted_when_no_python_fence():
    text = '```\nReturnAnswer("A")\n```'
    assert extract_program_source(text).strip() == 'ReturnAnswer("A")'


def test_plain_text_that_is_valid_python_is_accepted():
    assert extract_program_source('ReturnAnswer("A")').strip() == 'ReturnAnswer("A")'


def test_truncated_unclosed_fence_keeps_parseable_prefix():
    """未闭合围栏（截断）→ 保留可解析前缀，而不是整段放弃。"""
    text = (
        "```python\n"
        "n = count_objects('chair')\n"
        "if n == 0:\n"
        "    ReturnAnswer"        # 截断在行中（悬空名字，解析合法）
    )
    src = extract_program_source(text)
    tree = ast.parse(src)
    assert "n = count_objects('chair')" in src
    # 只保留模型自己写出的内容：不得出现被补全成形的 ReturnAnswer(...) 调用
    calls = [n for n in ast.walk(tree)
             if isinstance(n, ast.Call) and getattr(n.func, "id", "") == "ReturnAnswer"]
    assert calls == []


def test_degenerate_repetition_loop_still_yields_a_program():
    """退化重复段落（实测 20 KB 注释循环）→ 不抛异常，保留最大可解析前缀。

    真实样本：模型在"必须给数值"与"不能猜"之间反复自辩，输出全是注释、
    最后被 max_tokens 截断。旧实现直接 SynthesisError → 整题 unavailable。
    """
    body = "# we must abstain\n# but the system requires a numerical answer\n" * 50
    text = "```python\n" + body + "# The only valid action is to abstain from"
    src = extract_program_source(text)
    ast.parse(src)
    assert len(src.splitlines()) > 50


def test_empty_output_raises():
    with pytest.raises(SynthesisError):
        extract_program_source("   \n  ")


def test_unclosed_fence_without_parseable_prefix_raises():
    """未闭合且无任何可解析前缀 → 仍必须抛错（不得臆造 program）。"""
    text = "```python\ndef broken(:\n    ???\n"
    with pytest.raises(SynthesisError):
        extract_program_source(text)


def test_closed_block_with_syntax_error_raises_when_unrecoverable():
    text = "```python\ndef broken(:\n```"
    with pytest.raises(SynthesisError):
        extract_program_source(text)


def test_prose_only_output_raises():
    with pytest.raises(SynthesisError):
        extract_program_source("我认为答案是 C，因为房间里有一张桌子。")


def test_assemble_program_id_is_content_hash_and_stable():
    p1 = assemble_program('```python\nReturnAnswer("A")\n```')
    p2 = assemble_program('```\nReturnAnswer("A")\n```')
    assert p1.program_id == p2.program_id
    assert p1.program_source == 'ReturnAnswer("A")\n'
    assert p1.intended_answer_slot == "ReturnAnswer"
    assert p2.skill_semver_used == []


def test_prompt_states_returnanswer_does_not_terminate():
    """prompt 必须写明 `ReturnAnswer` 不中止执行、且答后调 Tool 会抛 AnswerAlreadyGiven。

    回归背景（inner_validation 1501–1504 实测）：模型写
    `if not ids: ReturnAnswer("abstain")` 之后继续执行 `object_centroid(ids[0])`
    → IndexError → `violation_runtime`（不是干净的 abstain）。

    v6 §15.1：`ReturnAnswer` 保留"记录/反作弊"语义（**不中止**执行），但
    ① 静态层（AST）禁止答后再调 Tool；② 运行层抛受控 `AnswerAlreadyGiven`。
    prompt 必须把这两点讲清楚（模型据此改写"先算后答"的 program）。
    """
    from skill3d.synthesis.prompt_builder import TEMPLATE_VERSION, PromptBuilder
    from skill3d.tools import REGISTRY
    from skill3d.tools.contract import SCOPE_FULL_3D

    text = PromptBuilder().render(
        question="Q", scene_summary="s", scene_frame="world",
        tool_docs=REGISTRY.docs(SCOPE_FULL_3D),
        scope=SCOPE_FULL_3D, question_type="object_rel_direction",
        available_artifacts=["frames", "objects"])
    assert "不会中止" in text
    assert "最后一行" in text
    # §15.1 新指引：答后调 Tool 会被硬拦截，异常名必须写出来
    assert "AnswerAlreadyGiven" in text
    assert "ReturnAnswer` 之后调用任何 Tool" in text or "之后再调用工具" in text
    # 头部与摘要同源（§5.3）：scope 由调用方传入并透传
    assert f"question_tool_scope={SCOPE_FULL_3D}" in text
    assert TEMPLATE_VERSION


def test_prompt_header_carries_scope_evidence_and_metric_gate():
    """v6 prompt 头部：`scope` / `question_type` / `evidence_profile` / `gate_passed`
    / `gate_missing` 五件都必须写进头部（§5.3/§13.3）。

    `scale_known` / `route` 两个 v5 参数已废止（§20：米制可用性由证据门逐题决定）。
    """
    import inspect

    from skill3d.schemas.evidence import EvidenceProfile
    from skill3d.synthesis.prompt_builder import PromptBuilder
    from skill3d.tools import REGISTRY
    from skill3d.tools.contract import SCOPE_FULL_3D

    params = set(inspect.signature(PromptBuilder.render).parameters)
    assert {"scope", "question_type", "evidence_profile", "gate_passed",
            "gate_missing"} <= params
    assert "scale_known" not in params and "route" not in params

    profile = EvidenceProfile(
        geometry_3d="available", world_frame="degraded", metric_scale="unavailable",
        object_detection="available", track_consensus="degraded", temporal="available",
        image_2d="available", object_grounding="available")
    text = PromptBuilder().render(
        question="Q", scene_summary="s", scene_frame="world",
        tool_docs=REGISTRY.docs(SCOPE_FULL_3D, evidence_profile=profile),
        scope=SCOPE_FULL_3D, question_type="object_abs_distance",
        available_artifacts=["frames", "objects"], evidence_profile=profile,
        gate_passed=False, gate_missing=["scale_self_consistency_ok"])
    assert "question_tool_scope=full_3d" in text
    assert "米制证据门未通过" in text
    assert "scale_self_consistency_ok" in text          # 缺失子条件必须点名（§13.3）
    assert "metric_scale:unavailable" in text           # 证据画像逐项写清
    assert "abstain" in text                            # 拿不到就 abstain，不许编造
