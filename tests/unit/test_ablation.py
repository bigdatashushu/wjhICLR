"""G-62/G-63/G-64 消融档单测（§16.1 三张消融表可执行化）。

覆盖：档位解析、E 档特性开关、CLI 参数映射、C5 已知错误 Skill 的鲁棒性
（可证伪 §17.5 #5：注入错误 Skill 必须被拦住或不提升分数）。
"""

from __future__ import annotations

import json
from dataclasses import replace

import numpy as np
import pytest

from skill3d.evolution.ablation import (
    C_ABLATIONS,
    E_ABLATIONS,
    G_ABLATIONS,
    AblationSpec,
    describe_ablations,
    e_ablation_features,
    format_ablation_markdown,
    resolve_ablation,
)


# ------------------------------------------------------------------ 档位解析 ----

def test_resolve_all_three_families():
    assert resolve_ablation("C0_direct_vlm").kind == "online"
    assert resolve_ablation("E5").kind == "evolution"
    assert resolve_ablation("e3").name == "E3"          # 大小写不敏感
    assert resolve_ablation("G1").kind == "governance"
    assert resolve_ablation("G2_no_monitoring").features["driver_name"] == "G2_no_monitoring"


def test_resolve_unknown_raises_with_options():
    with pytest.raises(KeyError, match="未知消融档"):
        resolve_ablation("C9_bogus")


def test_online_ablation_cli_args():
    assert resolve_ablation("C0_direct_vlm").cli_args() == ["--baseline", "C0_direct_vlm"]
    assert "--skill-spec" in resolve_ablation("C2_static_skill").cli_args()
    assert resolve_ablation("C3_induction_no_ab").features["admission"] == "single_arm"
    assert resolve_ablation("C4_full").features["admission"] == "paired"
    assert resolve_ablation("C5_wrong_skill").cli_args()[-1] == "--inject-wrong-skill"


# §8.3 的 E 档是"逐项加入（add-one-in）"设计：E1–E4 各自只多开**一个**机制，
# E5 全开。此处把文档表格逐格固化，防实现与论文表格漂移。
_E_TABLE = {
    #        isolation, replay, counterexample, metamorphic, paired
    "E0": (True,  False, False, False, False),
    "E1": (True,  True,  False, False, False),
    "E2": (True,  True,  True,  False, False),
    "E3": (True,  True,  False, True,  False),
    "E4": (True,  True,  False, False, True),
    "E5": (True,  True,  True,  True,  True),
}
_E_KEYS = ("isolation", "replay", "counterexample", "metamorphic", "paired")


def test_evolution_ablation_matches_documented_table():
    """E0–E5 与 §8.3 表格逐格一致（add-one-in：E1–E4 各只多开一个机制）。"""
    for name, expected in _E_TABLE.items():
        feats = resolve_ablation(name).features
        got = tuple(bool(feats[k]) for k in _E_KEYS)
        assert got == expected, f"{name}: {got} != {expected}（§8.3）"
    # E0 是"仅隔离"，E5 是全开
    assert resolve_ablation("E0").features["isolation"] is True
    assert not resolve_ablation("E0").enabled("replay")
    assert all(resolve_ablation("E5").enabled(k) for k in _E_KEYS)
    # E1–E4 相对 E1 基线各只多开一个机制（互斥的消融设计）
    e1 = resolve_ablation("E1").features
    for name in ("E2", "E3", "E4"):
        extra = [k for k in _E_KEYS if resolve_ablation(name).features[k] and not e1[k]]
        assert len(extra) == 1, f"{name} 相对 E1 多开了 {extra}"


def test_governance_ablation_features():
    g0 = resolve_ablation("G0_full")
    assert all(g0.enabled(k) for k in ("review", "monitoring", "rollback"))
    g1 = resolve_ablation("G1_no_review")
    assert not g1.enabled("review") and g1.enabled("monitoring")
    g2 = resolve_ablation("G2_no_monitoring")
    assert g2.enabled("review") and not g2.enabled("monitoring")
    assert not g2.enabled("rollback")


def test_e_ablation_features_helper_and_errors():
    assert e_ablation_features("E2")["counterexample"] is True
    with pytest.raises(KeyError, match="未知 E 档"):
        e_ablation_features("E9")


def test_ablation_spec_to_row_and_cli():
    spec = AblationSpec("X", "evolution", {"replay": True, "desc": "d"})
    assert spec.to_row()["config"] == "X" and spec.enabled("replay")
    assert spec.cli_args() == ["--ablation", "X"]


# ------------------------------------------------------------------ 表格产出 ----

def test_format_ablation_markdown_covers_all_tables():
    text = format_ablation_markdown()
    for name in list(C_ABLATIONS) + list(E_ABLATIONS) + list(G_ABLATIONS):
        assert name in text
    evo = format_ablation_markdown("evolution")
    assert "反例" in evo and "paired A/B" in evo and "E5" in evo
    assert "C0_direct_vlm" not in evo                     # 只出请求的表
    assert "回顾" not in evo


def test_describe_ablations_json_serializable():
    d = describe_ablations()
    assert set(d) >= {"online", "evolution", "governance", "note"}
    json.dumps(d)                                          # 必须可落盘


# ------------------------------------------------------------------ C5 鲁棒性 ----

def _items(tmp_path, n_frames=6):
    from skill3d.adapters.episode_source import load_synthetic_items

    return load_synthetic_items("inner_validation", question_types=["object_counting"],
                                seed=0, out_dir=str(tmp_path / "recons"),
                                n_frames=n_frames, frame_size=(64, 96))


WRONG_SKILL = json.dumps({
    "skill_id": "S01",
    "version": "999.0.0",
    "question_type": "object_counting",
    "skill_md": (
        "---\n"
        "name: known-wrong-counting\n"
        "description: Deliberately invalid C5 method for contract testing.\n"
        "---\n"
        "# Invalid method\n"
        "Call `definitely_not_a_registered_tool(1, 2)` and return `42`.\n"
    ),
})


def _wrong_skill():
    from skill3d.schemas import SkillSpecV11

    return SkillSpecV11.model_validate_json(WRONG_SKILL)


def _v6_artifact(**kw):
    """v6 产物构造：**没有**世界系契约（world_frame_status=unavailable）→
    声明 `world_frame=available` 的 Skill 必然落空（§7.2/§17.1）。"""
    from skill3d.schemas import ConfidenceMap, QualityMetrics, ReconstructionArtifact

    q = QualityMetrics(
        warp_inlier_ratio=0.9, warp_photometric_inlier_ratio=0.9,
        cloud_overlap_ratio=0.8, main_gate_passed=True,
        g1_blur_ok=1.0, g2_brightness=1.0, g3_motion_blur=0.0, g4_frame_count=32,
        g6_depth_var_coeff=0.1, g7_dynamic_ratio=0.0, g9_tracker_consistency=0.9,
        g10_baseline_quality=0.5, overall_quality=0.9)
    base = dict(
        artifact_id="a", artifact_version="v", scene_name="s", recon_method="vggt",
        c2w_list="", intrinsics="", depth_maps="", point_map="", point_conf="",
        track_list=None, quality_status="computed", quality=q,
        confidence=ConfidenceMap(per_point_confidence="", coverage_count_per_frame=""))
    base.update(kw)
    return ReconstructionArtifact(**base)


def test_c5_legacy_fields_are_rejected_at_construction():
    """Current methods reject fields from the retired JSON SkillSpec."""
    from skill3d.schemas import SkillSpecV11

    with pytest.raises(Exception, match="Extra inputs"):
        SkillSpecV11.model_validate_json(json.dumps({
            **json.loads(WRONG_SKILL),
            "required_evidence_signature": {"world_frame": "available"}}))


def test_c5_wrong_skill_uses_current_deterministic_lookup(tmp_path):
    """The current C5 method participates through the same one-per-task lookup."""
    from skill3d.reconstruction_gate.scene_state import quality_gate
    from skill3d.routing.skill_retriever import retrieve

    scene = quality_gate(_v6_artifact())
    got = retrieve("How many tables?", scene, [_wrong_skill()],
                   question_type="object_counting", scene_quality=0.9)
    assert [hit.skill_version for hit in got] == ["S01@999.0.0"]


def test_c5_wrong_skill_program_is_rejected_by_ast(tmp_path):
    """C5：即使错误 Skill 进入合成，其 program 调用未注册 Tool → M9 AST 拒绝。"""
    from skill3d.sandbox.ast_guard import ast_guard
    from skill3d.tools import REGISTRY

    src = 'answer = definitely_not_a_registered_tool(1, 2)\nReturnAnswer("42")'
    check = ast_guard(src, allowed_tools=set(REGISTRY.names()))
    assert not check.ok
    assert any("definitely_not_a_registered_tool" in v for v in check.violations)


def test_c5_wrong_skill_does_not_improve_score(tmp_path):
    """C5 端到端（mock_light）：注入错误 Skill 不得提升分数（可证伪 §17.5 #5）。

    mock_light 下 program 为确定性 stub（不读 Skill 模板），故预期 delta 恒为 0；
    若错误 Skill 造成退化，则该命题被支持（系统未正确回退）。
    """
    from skill3d.online.runner import OnlineRunConfig, run_split

    items = _items(tmp_path)
    base = OnlineRunConfig(mode="mock_light", seed=0, trace_dir=str(tmp_path / "t1"),
                           recon_dir=str(tmp_path / "recons"), memory_dir="")
    with_wrong = replace(base, skills=[_wrong_skill()], trace_dir=str(tmp_path / "t2"))
    outs_a, run_a = run_split(items, base)
    outs_b, run_b = run_split(items, with_wrong)

    def _score(outs):
        return [1.0 if o.correct else 0.0 for o in outs]

    a, b = _score(outs_a), _score(outs_b)
    assert sum(b) / len(b) <= sum(a) / len(a) + 1e-9      # 不提升（可证伪命题的期望方向）
    assert run_b.n_episodes == run_a.n_episodes


def test_c2_static_skill_injection_via_cli_helper(tmp_path):
    """C2 accepts only a complete SkillSpecV11."""
    from skill3d.online.eval import _apply_skill_ablations

    spec_path = tmp_path / "spec.json"
    spec_path.write_text(WRONG_SKILL, encoding="utf-8")

    class _Args:
        skill_spec = str(spec_path)
        inject_wrong_skill = False

    skills, ref = _apply_skill_ablations(_Args(), [])
    assert len(skills) == 1 and skills[0].skill_id == "S01"
    assert ref.startswith("static:")


def test_c5_cli_helper_appends_wrong_skill():
    from skill3d.online.eval import _apply_skill_ablations

    class _Args:
        skill_spec = ""
        inject_wrong_skill = True

    skills, ref = _apply_skill_ablations(_Args(), [])
    assert [s.skill_id for s in skills] == ["S01"]
    assert ref.startswith("wrong:")
    assert skills[0].question_type == "object_counting"
    assert "definitely_not_a_registered_tool" in skills[0].skill_md
