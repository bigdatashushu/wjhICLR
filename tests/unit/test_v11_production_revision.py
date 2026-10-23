"""Backward-compatible production proposal and declared-section contracts."""

import importlib.util
from pathlib import Path

import pytest

from skill3d.schemas import V11CampaignDecision, V11RevisionProposal


@pytest.mark.parametrize("value", [None, "", "a candidate method"])
def test_no_change_explicitly_rejects_the_source_field(value):
    with pytest.raises(ValueError, match="must omit full_skill_md"):
        V11RevisionProposal(
            action="no_change", full_skill_md=value,
            modification_reason="No supported change.", source_run_ref="offline:test",
        )


def test_no_change_serialization_omits_source_and_round_trips():
    proposal = V11RevisionProposal(
        action="no_change", modification_reason="Keep the parent.", source_run_ref="offline:test",
    )
    assert "full_skill_md" not in proposal.model_dump(mode="json")
    assert V11RevisionProposal.model_validate_json(proposal.model_dump_json()) == proposal


def test_legacy_revision_still_requires_source_and_accepts_no_declarations():
    proposal = V11RevisionProposal(
        full_skill_md="complete legacy method", modification_reason="Legacy reason.",
        source_run_ref="offline:legacy",
    )
    assert proposal.action == "revise"
    assert proposal.changed_sections == []
    with pytest.raises(ValueError, match="requires nonempty"):
        V11RevisionProposal(modification_reason="Missing source.", source_run_ref="offline:test")


def test_declared_revision_metadata_round_trips_without_losing_provenance():
    proposal = V11RevisionProposal(
        full_skill_md="complete method", modification_reason="Clarify ambiguity.",
        source_run_ref="offline:receipt", change_hypothesis="Clarify the trigger.",
        changed_sections=["Method"], expected_process_effect="Check before ranking.",
        evidence_signature_refs=["evidence:one"],
    )
    restored = V11RevisionProposal.model_validate_json(proposal.model_dump_json())
    assert restored == proposal
    assert restored.change_hypothesis == "Clarify the trigger."
    assert restored.changed_sections == ["Method"]
    assert restored.evidence_signature_refs == ["evidence:one"]


@pytest.mark.parametrize("changes", [
    {"change_hypothesis": ""}, {"expected_process_effect": " "},
    {"changed_sections": [""]}, {"changed_sections": ["Method", "Method"]},
    {"evidence_signature_refs": [""]},
])
def test_provided_revision_declarations_are_nonempty_and_unique(changes):
    with pytest.raises(ValueError, match="declared revision"):
        V11RevisionProposal(
            full_skill_md="method", modification_reason="Reason.",
            source_run_ref="offline:test", **changes,
        )


@pytest.mark.parametrize("changes", [
    {"stage": "static_validation"}, {"candidate_id": "fake-candidate"},
    {"conditions": {"static_validation_passed": True}}, {"reasons": []},
])
def test_no_change_decision_cannot_claim_a_static_candidate(changes):
    fields = {
        "campaign_id": "no-change", "question_type": "object_counting",
        "evidence_state": "g0m0", "stage": "revision", "outcome": "no_change",
        "parent_snapshot_id": "snapshot-parent", "parent_skill_version": "S01@1.0.0",
        "conditions": {"revision_no_change": True}, "reasons": ["No justified change."],
        **changes,
    }
    with pytest.raises(ValueError, match="no_change requires"):
        V11CampaignDecision(**fields)


@pytest.fixture
def cli():
    root = Path(__file__).resolve().parents[2]
    spec = importlib.util.spec_from_file_location(
        "production_revision_cli_test", root / "scripts/run_streaming_evolution_v11.py",
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_production_cli_defaults_to_off_and_disables_transport_replays(cli):
    args = cli.build_parser().parse_args(["--stream-id", "test", "--run-root", "/tmp/test"])
    assert args.diagnosis_mode == "off"
    assert args.diagnosis_max_cases == 4
    assert args.diagnosis_max_input_chars == 64000
    assert args.diagnosis_max_images == 8
    assert args.diagnosis_max_output_tokens == 4096
    assert args.deepseek_max_retries == 0
    assert args.diagnosis_image_root == []


@pytest.mark.parametrize(("extra", "expected"), [
    (["--diagnosis-mode", "multimodal"], "explicitly injected image provider"),
    (["--deepseek-max-retries", "1"], "must be zero"),
    ([], "explicit --max-requests budget"),
])
def test_production_cli_fail_closed_before_loading_or_requesting(cli, monkeypatch, capsys, extra, expected):
    def forbidden(*_args, **_kwargs):
        raise AssertionError("configuration/model execution should not start")

    monkeypatch.setattr(cli, "load_config", forbidden)
    assert cli.main(["--stream-id", "test", "--run-root", "/tmp/test", *extra]) == 2
    assert expected in capsys.readouterr().err
