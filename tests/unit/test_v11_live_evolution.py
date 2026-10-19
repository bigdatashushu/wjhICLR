"""Deterministic contracts for the production per-episode scheduler."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from skill3d.evolution.experience_v11 import V11CaseEligibility
from skill3d.evolution.live_v11 import (
    V11LiveEvolutionError,
    V11LiveEvolutionScheduler,
    V11LiveSchedulerConfig,
    prepare_validation_panels,
)
from skill3d.evolution.streaming_v11 import StreamingAttemptResult
from skill3d.online.runner import OnlineRunConfig
from skill3d.schemas import SkillSpecV11, V11ExperienceCase

ROOT = Path(__file__).resolve().parents[2]
SOURCE = (
    ROOT
    / "skill_library"
    / "versions"
    / "S03"
    / "1.2.0"
    / "rank-object-distances"
    / "SKILL.md"
).read_text(encoding="utf-8")


def _item(index: int, *, split: str, scene: str | None = None):
    return SimpleNamespace(
        episode=SimpleNamespace(
            qa_id=f"qa-{split}-{index:02d}",
            question_type="object_rel_distance",
            split=split,
            scene_name=scene or f"scene-{split}-{index:02d}",
        ),
        input_error=None,
        pixels=[],
    )


def _parent() -> SkillSpecV11:
    return SkillSpecV11(
        skill_id="S03",
        version="1.2.0",
        question_type="object_rel_distance",
        evidence_state="g1m0",
        skill_md=SOURCE,
    )


def _case(parent: SkillSpecV11, item, index: int) -> V11ExperienceCase:
    evidence = {
        "geometry_3d": "available",
        "metric_scale": "unavailable",
    }
    return V11ExperienceCase(
        case_id=f"live-case-{index:02d}",
        qa_id=item.episode.qa_id,
        scene_id=item.episode.scene_name,
        question_type=parent.question_type,
        evidence_state=parent.evidence_state,
        question_text=f"Which object is closest in case {index}?",
        options=["chair", "table", "lamp", "sofa"],
        source_split="induction",
        split="learning",
        label_access=True,
        outcome="failure",
        episode_status="answered",
        delivered_skill_version=f"{parent.skill_id}@{parent.version}",
        delivered_content_sha256=parent.content_sha256,
        delivered_skill_md=parent.skill_md,
        response_text="ReturnAnswer('A')",
        response_texts=["ReturnAnswer('A')"],
        program_source="ReturnAnswer('A')",
        initial_evidence_states=evidence,
        evidence_states=evidence,
        evidence_history=[{
            "trigger": "initial",
            "evidence_states": evidence,
        }],
        model_answer="A",
        reference_answer="B",
        score=0.0,
        failure_categories=["answer_incorrect"],
        error_direction="wrong_option",
        trace_ref=f"trace:{item.episode.qa_id}",
    )


def test_validation_panels_freeze_eight_exact_slot_pairs(tmp_path):
    items = [_item(index, split="inner_validation") for index in range(10)]
    items.append(
        _item(99, split="inner_validation", scene="learning-scene")
    )
    artifacts = {
        item.episode.qa_id: tmp_path / f"{item.episode.qa_id}.json"
        for item in items
    }

    def resolve(item, _path):
        return "g0m0" if item.episode.qa_id.endswith("09") else "g1m0"

    panels = prepare_validation_panels(
        items,
        artifacts,
        required_slots=[("object_rel_distance", "g1m0")],
        min_pairs=8,
        excluded_scene_ids=["learning-scene"],
        state_resolver=resolve,
    )
    panel = panels["object_rel_distance/g1m0"]
    assert len(panel.items) == 8
    assert len(set(panel.qa_ids)) == 8
    assert all(
        item.episode.scene_name != "learning-scene"
        for item in panel.items
    )


def test_validation_panel_shortage_fails_before_model_calls(tmp_path):
    items = [_item(index, split="inner_validation") for index in range(7)]
    artifacts = {
        item.episode.qa_id: tmp_path / f"{item.episode.qa_id}.json"
        for item in items
    }
    with pytest.raises(V11LiveEvolutionError, match="require at least 8"):
        prepare_validation_panels(
            items,
            artifacts,
            required_slots=[("object_rel_distance", "g1m0")],
            min_pairs=8,
            state_resolver=lambda _item, _path: "g1m0",
        )


def test_live_scheduler_runs_isolated_episodes_and_triggers_at_twelve(tmp_path):
    parent = _parent()
    items = [_item(index, split="induction") for index in range(1, 13)]
    artifacts = {
        item.episode.qa_id: tmp_path / f"{item.episode.qa_id}.json"
        for item in items
    }
    active_calls = []
    execution_calls = []
    attempt_calls = []

    def active():
        active_calls.append(True)
        return [parent], "snapshot-parent", "a" * 64

    def snapshot(snapshot_id):
        assert snapshot_id == "snapshot-parent"
        return [parent], "a" * 64

    def execute(item, _artifact, skills, snapshot_id, manifest_hash, trace_dir):
        execution_calls.append((
            item.episode.qa_id,
            f"{skills[0].skill_id}@{skills[0].version}",
            snapshot_id,
            manifest_hash,
        ))
        trace_dir.mkdir(parents=True)
        return f"online_run:run-{item.episode.qa_id}"

    def collect(_trace_dir, **kwargs):
        qa_id = kwargs["source_run_ref"].removeprefix(
            "online_run:run-"
        )
        item = next(
            candidate
            for candidate in items
            if candidate.episode.qa_id == qa_id
        )
        index = items.index(item) + 1
        return V11CaseEligibility(
            eligible=True,
            case=_case(parent, item, index),
        )

    def reject(request):
        attempt_calls.append(request)
        return StreamingAttemptResult(
            outcome="rejected",
            campaign_id="live-campaign-001",
            reason="candidate did not improve",
        )

    scheduler = V11LiveEvolutionScheduler(
        V11LiveSchedulerConfig(
            stream_id="live-test",
            root=tmp_path / "run",
            library_root=tmp_path / "library",
            batch_size=12,
            flush_tail=True,
        ),
        items=items,
        artifact_paths=artifacts,
        base_cfg=OnlineRunConfig(),
        attempt=reject,
        state_resolver=lambda _item, _path: "g1m0",
        active_provider=active,
        snapshot_provider=snapshot,
        episode_executor=execute,
        case_collector=collect,
    )
    summary = scheduler.run()
    assert len(active_calls) == 12
    assert len(execution_calls) == 12
    assert all(row[1:] == ("S03@1.2.0", "snapshot-parent", "a" * 64)
               for row in execution_calls)
    assert len(attempt_calls) == 1
    assert len(attempt_calls[0].cases) == 12
    assert summary["attempt_outcomes"] == ["rejected"]

    resumed = scheduler.run()
    assert resumed == summary
    assert len(active_calls) == 12
    assert len(execution_calls) == 12
