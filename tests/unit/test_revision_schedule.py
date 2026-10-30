import json
from dataclasses import asdict, FrozenInstanceError

import pytest

from skill3d.evolution.revision_schedule import (
    failure_patterns, normalized_method_sha256, scheduling_decision,
)
from skill3d.evolution.fitset_stream_v11 import FitSetEvolutionConfig
from test_fitset_stream_v11 import make_runner, no_change_revision_factory


def feedback(*, event="tool_argument_error", rounds=1, band="zero_score_answered"):
    return {"signatures": [{
        "n": rounds, "score_band": band,
        "process": {
            "execution": {"terminal_action": "answer", "answer_submitted": True,
                          "model_rounds": rounds},
            "error_chain": [{"event_type": event, "round_index": rounds}],
        },
    }]}


def test_novelty_ignores_frequency_and_round_but_detects_new_error():
    seen = set(failure_patterns(feedback()))
    kwargs = dict(mode="novel_failure_v1", previous_patterns=seen,
                  revision_requests_used=1, gate_candidates_used=1,
                  max_revision_requests=18, max_gate_candidates=6)
    assert scheduling_decision(feedback=feedback(rounds=9), **kwargs)["reason_code"] == (
        "no_new_failure_pattern")
    assert scheduling_decision(feedback=feedback(event="program_missing_action"), **kwargs)[
        "scheduled"]
    assert scheduling_decision(feedback=feedback(), **{
        **kwargs, "revision_requests_used": 18})["reason_code"] == "slot_revision_request_limit"
    assert scheduling_decision(feedback=feedback(), **{
        **kwargs, "gate_candidates_used": 6})["reason_code"] == "slot_candidate_limit"


def test_dedup_ignores_version_and_newlines_but_keeps_semantics():
    text = "---\nname: sample\ndescription: method\nversion: 1.0.0\n---\n# Method\nUse a table.\n"
    assert normalized_method_sha256(text) == normalized_method_sha256(
        text.replace("1.0.0", "2.0.0").replace("\n", "\r\n") + "\r\n")
    assert normalized_method_sha256(text) != normalized_method_sha256(
        text.replace("a table", "a different table"))
    assert normalized_method_sha256(text) != normalized_method_sha256(
        text.replace("description: method", "description: different"))


def test_repeated_failure_batch_no_change_does_not_block_resume(tmp_path):
    runner = make_runner(tmp_path, n=24)
    runner.evolution_config = FitSetEvolutionConfig()
    runner.revision_client_factory = no_change_revision_factory
    summary = runner.run(resume=False)
    assert summary["status"] == "complete"
    stream = json.loads(next((runner.root / "workers").glob("*/buffer/stream.json")).read_text())
    assert [row["outcome"] for row in stream["attempts"]] == ["no_change", "no_change"]
    schedules = [json.loads(path.read_text()) for path in
                 (runner.root / "workers").glob("*/campaigns/*/schedule.json")]
    assert sorted(row["reason_code"] for row in schedules) == [
        "no_new_failure_pattern", "scheduled"]
    assert len(list((runner.root / "workers").glob(
        "*/campaigns/*/revision/slots/*/*/request_intent_*.json"))) == 1
    assert not stream["slots"]["object_counting/g0m0"]["deferred_reason"]
    assert runner.run(resume=True)["status"] == "complete"


def test_same_parent_candidate_skips_second_gate(tmp_path):
    runner = make_runner(tmp_path, n=24, reference="1")
    runner.run(resume=False)
    stream = json.loads(next((runner.root / "workers").glob("*/buffer/stream.json")).read_text())
    assert [row["outcome"] for row in stream["attempts"]] == ["rejected", "duplicate"]
    assert len(list((runner.root / "workers").glob("*/campaigns/*/decision.json"))) == 1
    duplicate = json.loads(next((runner.root / "workers").glob(
        "*/campaigns/*/duplicate.json")).read_text())
    assert duplicate["gate_reexecuted"] is False


def test_slot_request_cap_prevents_additional_revision(tmp_path):
    runner = make_runner(tmp_path, n=24)
    runner.evolution_config = FitSetEvolutionConfig(
        scheduling_mode="batch_v1", max_revision_requests_per_slot=1)
    runner.revision_client_factory = no_change_revision_factory
    runner.run(resume=False)
    schedules = [json.loads(path.read_text()) for path in
                 (runner.root / "workers").glob("*/campaigns/*/schedule.json")]
    assert sorted(row["reason_code"] for row in schedules) == [
        "scheduled", "slot_revision_request_limit"]
    assert len(list((runner.root / "workers").glob(
        "*/campaigns/*/revision/slots/*/*/request_intent_*.json"))) == 1


def test_explicit_slot_caps_are_frozen_and_other_caps_remain_unchanged(tmp_path):
    caps = {"object_counting/g0m0": 1}
    config = FitSetEvolutionConfig(scheduling_mode="batch_v1", revision_request_caps=caps)
    caps["object_counting/g0m0"] = 99
    assert config.revision_request_cap("object_counting/g0m0") == 1
    assert config.revision_request_cap("object_size_estimation/g0m0") == 18
    assert FitSetEvolutionConfig(**asdict(config)) == config
    with pytest.raises(FrozenInstanceError):
        config.max_revision_requests_per_slot = 99
    runner = make_runner(tmp_path, n=24)
    runner.evolution_config = config
    runner.revision_client_factory = no_change_revision_factory
    runner.run(resume=False)
    schedules = [json.loads(path.read_text()) for path in
                 (runner.root / "workers").glob("*/campaigns/*/schedule.json")]
    assert sorted(row["reason_code"] for row in schedules) == [
        "scheduled", "slot_revision_request_limit"]


@pytest.mark.parametrize("caps", [
    {"unknown/g0m0": 1}, {"object_counting/g0m0": True},
    {"object_counting/g0m0": 0}, {"object_counting/g0m0": 1.5},
    [("object_counting/g0m0", 1), ("object_counting/g0m0", 2)],
])
def test_invalid_slot_caps_are_refused(caps):
    with pytest.raises(ValueError, match="unique valid slots"):
        FitSetEvolutionConfig(revision_request_caps=caps)
