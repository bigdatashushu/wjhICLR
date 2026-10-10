"""v11 complete-source delivery and public-contract identity in model requests."""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import pytest

from skill3d.adapters.episode_source import load_synthetic_items
from skill3d.online.runner import OnlineRunConfig, _synthesize, run_episode
from skill3d.schemas import EvidenceProfile, SceneState
from skill3d.schemas.evidence import CAPABILITIES
from skill3d.skills.delivery import (
    plan_delivery,
    render_skill_entry,
    skill_content_sha256,
)
from skill3d.skills.v11_library import load_skill_source_v11, validate_v11_snapshot
from skill3d.synthesis.prompt_builder import PROMPT_TEMPLATE_VERSION
from skill3d.trace.store import TraceStore
from skill3d.verifier.geometry_oracle import GeometryIssue, GeometryVerifyResult

from test_tool_contract_recovery import (  # noqa: E402 - 同目录测试夹具
    _FakeClient,
    _write_v6_artifact,
)

ROOT = Path(__file__).resolve().parents[2]
LIBRARY = ROOT / "skill_library"
S03_REF = "versions/S03/1.1.0/rank-object-distances/SKILL.md"
FRAME_SIZE = (120, 160)


class _RecordingClient(_FakeClient):
    def __init__(self, programs):
        super().__init__(programs)
        self.request_params = []

    def chat(self, messages, max_tokens=512, **kwargs):
        before = len(self.calls)
        text = super().chat(messages, max_tokens=max_tokens, **kwargs)
        if len(self.calls) > before:
            self.request_params.append({"max_tokens": max_tokens, **kwargs})
        return text


def _prompt_text(messages: list[dict]) -> str:
    parts = []
    for message in messages:
        content = message.get("content")
        if isinstance(content, str):
            parts.append(content)
        elif isinstance(content, list):
            parts.extend(
                part.get("text", "")
                for part in content
                if isinstance(part, dict) and part.get("type") == "text"
            )
    return "\n".join(parts)


def test_v11_complete_source_is_the_only_planned_model_text():
    source = load_skill_source_v11(
        LIBRARY,
        source_ref=S03_REF,
        skill_id="S03",
        version="1.1.0",
        question_type="object_rel_distance",
    )
    plan = plan_delivery([source.spec], max_chars=8000)
    plan.mark_delivered()

    assert plan.delivered_skill_versions == ["S03@1.1.0"]
    assert plan.delivered_content_sha256 == {
        "S03@1.1.0": skill_content_sha256(source.spec)}
    assert render_skill_entry(source.spec) == source.spec.skill_md
    assert len(plan.entries) == 1
    assert plan.entries[0].text == source.spec.skill_md
    assert plan.entries[0].chars == len(source.spec.skill_md)
    assert "categories_without_detection" in plan.entries[0].text
    assert "### S03@1.1.0 (task_type=" not in plan.entries[0].text


def test_v11_complete_source_reaches_runner_model_request_once():
    item = load_synthetic_items(
        "inner_validation",
        question_types=["object_rel_distance"],
        frame_size=FRAME_SIZE,
        seed=0,
    )[0]
    source = load_skill_source_v11(
        LIBRARY,
        source_ref=S03_REF,
        skill_id="S03",
        version="1.1.0",
        question_type="object_rel_distance",
    )
    client = _FakeClient(['ReturnAnswer("A")\n'])
    cfg = OnlineRunConfig(
        mode="real",
        vllm_endpoints=["http://fake"],
        skills=[source.spec],
        seed=0,
    )

    result = _synthesize(
        item.episode,
        scene=None,
        handle=None,
        skills=[source.spec],
        cfg=cfg,
        llm=client,
        pixels=list(item.pixels),
    )

    assert result.program is not None, result.note
    sent = _prompt_text(client.calls[0])
    assert sent.count(source.spec.skill_md) == 1
    assert "### S03@1.1.0 (task_type=" not in sent
    assert result.delivery.delivered_skill_versions == ["S03@1.1.0"]
    assert result.delivery.delivered_content_sha256 == {
        "S03@1.1.0": hashlib.sha256(
            source.spec.skill_md.encode("utf-8")
        ).hexdigest()
    }
    assert result.program.skill_semver_used == ["S03@1.1.0"]


def _v11_specs():
    snapshot = json.loads(
        (LIBRARY / "snapshots/snapshot_S0-v11-contract-repair.json").read_text())
    return validate_v11_snapshot(snapshot, library_root=LIBRARY)


@pytest.mark.parametrize("spec", _v11_specs(), ids=lambda spec: spec.skill_id)
@pytest.mark.parametrize("evidence_available", [True, False])
def test_skill_on_off_requests_only_differ_by_complete_method(spec, evidence_available):
    item = load_synthetic_items(
        "inner_validation", question_types=[spec.question_type],
        frame_size=FRAME_SIZE, limit=1, seed=0)[0]
    reply = 'ReturnAnswer("A")\n' if item.episode.options else 'ReturnAnswer(1)\n'
    profile = EvidenceProfile(**{
        key: "available" if evidence_available or key in ("image_2d", "temporal")
        else "unavailable"
        for key in CAPABILITIES})
    scene = SceneState(
        artifact_ref="test", question_type=spec.question_type, evidence_profile=profile,
        scene_route="full_3d" if evidence_available else "fallback_2d_only",
        question_tool_scope="metric_enabled" if evidence_available else "fallback_2d_only",
    )
    sent, results = [], []
    for skills in ([], [spec]):
        client = _RecordingClient([reply])
        cfg = OnlineRunConfig(mode="real", skills=skills, seed=7)
        result = _synthesize(
            item.episode, scene=scene, handle=None, skills=skills,
            cfg=cfg, llm=client, pixels=list(item.pixels))
        assert result.program is not None, result.note
        assert client.request_params == [{"max_tokens": cfg.max_tokens, "seed": 7}]
        assert f"本题题型: {spec.question_type}" in _prompt_text(client.calls[0])
        sent.append(client.calls[0])
        results.append(result)
    off_text, on_text = (_prompt_text(messages) for messages in sent)
    assert on_text == off_text + "\n## 参考方法\n" + spec.skill_md
    assert on_text.count(spec.skill_md) == 1
    # Images, roles and all other message parts must also be the same.
    assert sent[0][0]["role"] == sent[1][0]["role"]
    assert sent[0][0]["content"][1:] == sent[1][0]["content"][1:]
    assert results[0].delivery.delivered_skill_versions == []
    assert results[0].delivery.delivered_content_sha256 == {}
    assert results[0].program.skill_semver_used == []
    key = f"{spec.skill_id}@{spec.version}"
    assert results[1].delivery.delivered_content_sha256 == {key: spec.content_sha256}
    assert results[1].program.skill_semver_used == [key]
    assert "对象质心距离\n  以外的东西" not in on_text
    assert "优先 `count_objects`" not in off_text
    assert "theta_deg" not in off_text
    assert "AnswerPayload" in off_text
    assert ("- relative_direction_of(" in off_text) == evidence_available
    assert ("- object_visible_frames(" in off_text) == evidence_available
    assert "- inspect_frames(" in off_text


@pytest.fixture(scope="module")
def room_source_and_artifact(tmp_path_factory):
    item = load_synthetic_items(
        "inner_validation", question_types=["room_size_estimation"],
        frame_size=FRAME_SIZE, limit=1, seed=0)[0]
    spec = next(s for s in _v11_specs() if s.skill_id == "S05")
    artifact = _write_v6_artifact(
        tmp_path_factory.mktemp("v11-room"), with_poses=True)
    return item, spec, artifact


@pytest.mark.parametrize(("first_program", "round_limit", "expected_trigger"), [
    ("import os\nReturnAnswer(1)\n", 4, "ast_feedback"),
    ("euclidean_distance([0.0, 0.0], 1.0)\nReturnAnswer(1)\n", 4, "error_recovery"),
    ('img = inspect_frames([0])\nreturn YieldObservations([img], "查看图像")\n',
     4, "observation"),
    ('img = inspect_frames([0])\nreturn YieldObservations([img], "查看图像")\n',
     2, "finalize"),
])
def test_v11_contract_survives_solver_rounds_and_reaches_trace(
        room_source_and_artifact, tmp_path, first_program, round_limit, expected_trigger):
    item, spec, artifact = room_source_and_artifact
    final_response = "```python\nReturnAnswer(1)\n```\n"
    client = _RecordingClient([first_program, final_response])
    cfg = OnlineRunConfig(
        mode="real", reuse_artifact=artifact, skills=[spec], seed=7,
        trace_dir=str(tmp_path / "traces"),
        max_solver_rounds=round_limit, finalization_rounds=1,
    )
    out = run_episode(
        item.episode, item.pixels, cfg, llm=client, trace_store=TraceStore(cfg.trace_dir))
    assert out.final_state == "answer", out.notes
    assert len(client.calls) == 2, out.notes
    for messages in client.calls:
        # Prior programs and feedback can follow this first text part.
        text = messages[0]["content"][0]["text"]
        assert "## 公共执行规则" in text
        assert "## 题目口径" not in text
        assert text.count(spec.skill_md) == 1
    assert all(p["seed"] == 7 for p in client.request_params)
    if expected_trigger == "ast_feedback":
        assert "上一次生成被 AST 拒绝" in _prompt_text(client.calls[1])
    elif expected_trigger == "finalize":
        assert out.finalization_used
    else:
        assert any(r.get("trigger") == expected_trigger for r in out.rounds), out.rounds
    assert out.first_synthesis["template_version"] == PROMPT_TEMPLATE_VERSION
    assert out.first_synthesis["tool_docs_version"] == "tool-docs-v11.1"
    rows = (tmp_path / "traces/trace_record.jsonl").read_text().splitlines()
    assert len(rows) == 1
    assert json.loads(rows[0])["template_version"] == PROMPT_TEMPLATE_VERSION
    assert json.loads(rows[0])["tool_docs_version"] == "tool-docs-v11.1"
    program_row = json.loads(
        (tmp_path / "traces/episode_program.jsonl").read_text().splitlines()[0]
    )
    assert program_row["response_text"] == final_response
    assert program_row["response_texts"] == [first_program, final_response]
    assert program_row["program_source"] == "ReturnAnswer(1)\n"


def _verify_result(*, passed, issue=None):
    issues = [] if issue is None else [issue]
    return GeometryVerifyResult(
        passed=passed,
        checks={"no_negative_distance": passed},
        violations=[] if passed else [issue.check],
        issues=issues,
    )


def _run_m11_case(room_source_and_artifact, tmp_path, monkeypatch, programs, verifier,
                  *, max_rounds=3, retries=1):
    import skill3d.online.runner as runner

    item, spec, artifact = room_source_and_artifact
    item = copy.deepcopy(item)
    item.episode.ground_truth = "1"
    client = _RecordingClient(programs)
    monkeypatch.setattr(runner, "geometry_verify", verifier)
    cfg = OnlineRunConfig(
        mode="real", reuse_artifact=artifact, skills=[spec], seed=7,
        trace_dir=str(tmp_path / "traces"),
        max_solver_rounds=max_rounds, max_retries_per_operation=retries,
        finalization_rounds=1)
    out = run_episode(item.episode, item.pixels, cfg, llm=client)
    return out, client


def test_m11_rejects_first_answer_revokes_local_result_and_scores_correction(
        room_source_and_artifact, tmp_path, monkeypatch):
    calls = 0
    def verifier(trace, handle, answer):
        nonlocal calls
        calls += 1
        if calls == 1:
            result = trace.results[0]
            return _verify_result(
                passed=False,
                issue=GeometryIssue(
                    check="no_negative_distance", tool=result.tool,
                    result_id=result.result_id, reason="synthetic local violation"))
        return _verify_result(passed=True)

    out, client = _run_m11_case(
        room_source_and_artifact, tmp_path, monkeypatch,
        ["d = euclidean_distance([0, 0, 0], [1, 0, 0])\nReturnAnswer(1)\n",
         "ReturnAnswer(2)\n"],
        verifier)

    assert len(client.calls) == 2
    assert out.final_state == "answer" and out.answer == "2"
    # 被拒绝的首答恰好等于 GT；若旧路径仍采纳它会是 1.0。
    assert out.mra_value == pytest.approx(0.0)
    first = out.rounds[0]
    assert first["submission_accepted"] is False
    assert first["rejected_submission"]["answer"] == "1"
    assert first["invalidated_result_ids"]
    assert first["invalidated_result_ids"] == out.invalidated_result_ids
    second_text = _prompt_text(client.calls[1])
    assert "M11 拒绝上一轮提交" in second_text
    assert second_text.count("synthetic local violation") == 1
    assert first["invalidated_result_ids"][0] not in first["recovery_observation_ids"]


def test_m11_confirmed_shared_premise_cascades_and_visual_close_is_clean(
        room_source_and_artifact, tmp_path, monkeypatch):
    calls = 0
    def verifier(trace, handle, answer):
        nonlocal calls
        calls += 1
        if calls == 1:
            result = trace.results[0]
            return _verify_result(
                passed=False,
                issue=GeometryIssue(
                    check="inside_bbox", tool=result.tool, result_id=result.result_id,
                    reason="synthetic confirmed premise",
                    confirmed_shared_premise="geometry_3d"))
        assert trace.results == []
        return _verify_result(passed=True)

    out, _ = _run_m11_case(
        room_source_and_artifact, tmp_path, monkeypatch,
        ["area = plane_fit_room_size()\nReturnAnswer(1)\n",
         "ReturnAnswer(AnswerPayload(value=2, unit='m2', "
         "basis='visual_estimate', used_result_ids=[], derivation=None))\n"],
        verifier)

    assert out.final_state == "answer" and out.answer == "2"
    assert out.evidence_profile.state("geometry_3d") == "unavailable"
    assert out.rounds[0]["downgraded_capabilities"]["geometry_3d"].endswith(
        "→unavailable")
    assert out.used_result_ids == []
    assert out.answer_basis == "visual_estimate"


def test_m11_checks_valid_cross_round_results_after_yield(
        room_source_and_artifact, tmp_path, monkeypatch):
    seen = []
    def verifier(trace, handle, answer):
        seen.extend(r.result_id for r in trace.results)
        return _verify_result(passed=True)

    out, client = _run_m11_case(
        room_source_and_artifact, tmp_path, monkeypatch,
        ["d = euclidean_distance([0, 0, 0], [1, 0, 0])\n"
         "return YieldObservations([], 'observe')\n",
         "ReturnAnswer(1)\n"],
        verifier)

    assert len(client.calls) == 2
    assert out.final_state == "answer"
    assert seen and out.used_result_ids == seen
    assert out.rounds[1]["verification_result_ids"] == seen


def test_m11_replays_derivation_without_invalidating_a_valid_tool_result(
        room_source_and_artifact, tmp_path, monkeypatch):
    def verifier(trace, handle, answer):
        return _verify_result(passed=True)

    out, client = _run_m11_case(
        room_source_and_artifact,
        tmp_path,
        monkeypatch,
        [
            "r = plane_fit_room_size()\n"
            "ReturnAnswer(AnswerPayload("
            "value=999, unit='m2', basis='tool_derived', "
            "used_result_ids=[r['result_id']], "
            "derivation={'op':'field', "
            "'input_result_ids':[r['result_id']], "
            "'parameters':{'field':'room_area_m2'}}))\n",
            "ReturnAnswer(AnswerPayload("
            "value=2, unit='m2', basis='visual_estimate', "
            "used_result_ids=[], derivation=None))\n",
        ],
        verifier,
    )

    assert len(client.calls) == 2
    assert out.final_state == "answer" and out.answer == "2"
    first = out.rounds[0]
    replay = first["geometry_verify"]["derivation_replay"]
    assert replay["performed"] and not replay["passed"]
    assert {issue["code"] for issue in replay["issues"]} == {"value_mismatch"}
    assert first["invalidated_result_ids"] == []
    assert first["recovery_observation_ids"], "正确 ToolResult 应保留给修正轮"
    assert "derivation 重算值" in _prompt_text(client.calls[1])


def test_m11_accepts_a_matching_replayed_derivation(
        room_source_and_artifact, tmp_path, monkeypatch):
    def verifier(trace, handle, answer):
        return _verify_result(passed=True)

    program = (
        "r = plane_fit_room_size()\n"
        "ReturnAnswer(AnswerPayload("
        "value=r['room_area_m2'], unit='m2', basis='tool_derived', "
        "used_result_ids=[r['result_id']], "
        "derivation={'op':'field', "
        "'input_result_ids':[r['result_id']], "
        "'parameters':{'field':'room_area_m2'}}))\n"
    )
    out, client = _run_m11_case(
        room_source_and_artifact,
        tmp_path,
        monkeypatch,
        [program],
        verifier,
    )

    assert len(client.calls) == 1
    assert out.final_state == "answer"
    replay = out.rounds[0]["geometry_verify"]["derivation_replay"]
    assert replay["passed"] and replay["computed_unit"] == "m2"
    assert out.episode_trace.derivation_replay == replay


def test_m11_budget_exhaustion_clears_rejected_answer_and_keeps_denominator(
        room_source_and_artifact, tmp_path, monkeypatch):
    def verifier(trace, handle, answer):
        return _verify_result(
            passed=False,
            issue=GeometryIssue(
                check="unit_consistent", reason="synthetic unresolved violation"))

    out, client = _run_m11_case(
        room_source_and_artifact, tmp_path, monkeypatch,
        ["ReturnAnswer(1)\n", "ReturnAnswer(1)\n"],
        verifier, max_rounds=2, retries=1)

    assert len(client.calls) == 2
    assert out.agent_rounds == 2
    assert out.final_state == "run_error" and out.answer is None
    assert out.mra_value == 0.0
    assert all(r["submission_accepted"] is False for r in out.rounds)
    assert out.rounds[-1]["trigger"] == "finalize"
