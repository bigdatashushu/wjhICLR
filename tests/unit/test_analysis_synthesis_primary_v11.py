"""Primary evolution exercises real planning, edits, gate and publication offline."""
from dataclasses import replace
import json
from pathlib import Path

import pytest

from skill3d.evolution.deployment_continuation_v11 import revision_intents_used
from skill3d.evolution.fitset_revision_v11 import (
    FitSetSlotReviser, RevisionOperationalError,
)
from skill3d.evolution.fitset_stream_v11 import freeze_fitset_stream
from skill3d.evolution.revision_edits_v11 import PROTOCOL, source_anchors
from skill3d.evolution.trajectory_analysis_v11 import ANALYSIS_PROTOCOL
from skill3d.schemas import V11ExperienceBundle
from test_fitset_stream_v11 import LIBRARY, full_evidence_factory, make_runner
from test_revision_edits_v11 import Client, parent


def responding_client(*, change=True, invalid_analysis=False):
    prompts = []

    def respond(messages):
        text = messages[-1]["content"]
        prompts.append(text)
        if "Offline learning-trajectory analysis." in text:
            if invalid_analysis:
                return json.dumps({"protocol": ANALYSIS_PROTOCOL, "records": []})
            refs = json.loads(text.split("Expected source references:\n")[1].splitlines()[0])
            batch_hash = text.split("batch_sha256: ")[1].splitlines()[0]
            return json.dumps({
                "protocol": ANALYSIS_PROTOCOL, "batch_sha256": batch_hash,
                "records": [{
                    "case_index": int(index),
                    "observed_facts": "The program issued a final answer.",
                    "failure_hypothesis": "The method may need a completeness check.",
                    "supported_skill_change": "Preserve supported observations while checking completeness.",
                    "uncertainty": "No perception failure is established.",
                } for index in refs],
            })
        assert "Source-linked learning analyses." in text
        # The final edit sees the complete analyses, not the raw QA evidence.
        assert "Complete planned learning evidence batch:" not in text
        assert "How many chairs are visible" not in text
        if not change:
            return json.dumps({
                "protocol": PROTOCOL, "base_sha256": parent().content_sha256,
                "action": "no_change", "modification_reason": "Keep the supported method.",
            })
        source = parent()
        anchor = source_anchors(source.skill_md)[-1]
        return json.dumps({
            "protocol": PROTOCOL, "base_sha256": source.content_sha256,
            "action": "revise", "modification_reason": "Check completeness.",
            "change_hypothesis": "A completeness check may improve the method.",
            "expected_process_effect": "Preserve supported observations.",
            "edits": [{
                "section_id": anchor["section_id"], "anchor_id": anchor["anchor_id"],
                "after": anchor["source"] + " Use FITREV checks.",
            }],
        })

    client = Client(respond)
    client.prompts = prompts
    return client


def primary_runner(tmp_path, client, *, n=13, request_cap=18):
    runner = make_runner(
        tmp_path, n=n, learning_input_mode="analysis_synthesis_v1",
        revision_client_factory=lambda: client)
    runner.evolution_config = replace(
        runner.evolution_config, max_revision_requests_per_slot=request_cap)
    return runner


def campaign_root(runner):
    return next((runner.root / "workers").glob("*/campaigns/*/request.json")).parent


def test_default_launcher_selects_primary_mode():
    from skill3d.infra import experiment_campaign, experiment_stages_v11

    root = Path(__file__).resolve().parents[2]
    profile = json.loads((root / "configs/experiments/prepared.json").read_text())
    assert experiment_campaign.validate_profile(profile)["runtime"]["learning_input_mode"] == (
        "analysis_synthesis_v1")
    manifest = {"profile": profile, "run_root": "/synthetic/run"}
    cli = experiment_stages_v11.fitset_cli()
    arguments = cli.build_parser().parse_args(
        experiment_stages_v11.system_argv(manifest, "learning", preflight=True))
    assert arguments.learning_input_mode == "analysis_synthesis_v1"
    assert arguments.gate_mode == "qa_disjoint"
    assert arguments.request_governor_root == "/synthetic/run/requests"


def test_primary_pipeline_promotes_at_twelve_and_next_uses_new_skill(tmp_path):
    client = responding_client()
    runner = primary_runner(tmp_path, client)
    assert runner.run(resume=False)["completed_variants"] == 13
    root = campaign_root(runner)
    final = json.loads((root / "revision/slots/object_counting/g0m0/final.json").read_text())
    assert final["status"] == "static_valid"
    assert final["all_batches_resolved"] and final["all_evidence_delivered"]
    assert final["analyst_requests"] == final["editor_requests"] == 1
    assert revision_intents_used(root.parent.parent, "object_counting/g0m0") == 2
    assert len(client.prompts) == 2
    rows = runner.completed()
    assert rows["fit-011@g0m0"]["initial_skill_selection"]["skill_version"].endswith("@1.0.0")
    assert rows["fit-012@g0m0"]["initial_skill_selection"]["skill_version"].endswith("@1.1.0")
    decision = json.loads((root / "decision.json").read_text())
    assert decision["conditions"] == {"137": True, "138": True}
    assert runner.run(resume=True)["completed_variants"] == 13
    assert len(client.prompts) == 2
    assert freeze_fitset_stream(runner.root, tmp_path / "frozen")["revised_slot_count"] == 1
    receipt_text = "\n".join(path.read_text() for path in (root / "revision").rglob("*.json"))
    assert "The method may need a completeness check." not in receipt_text


def test_primary_no_change_and_unresolved_analysis_skip_gate(tmp_path):
    for name, options, expected_calls, status in (
        ("keep", {"change": False}, 2, "no_change"),
        ("unresolved", {"invalid_analysis": True}, 2, "analysis_rejected"),
    ):
        client = responding_client(**options)
        runner = primary_runner(tmp_path / name, client, n=12)
        assert runner.run(resume=False)["completed_variants"] == 12
        root = campaign_root(runner)
        assert not (root / "decision.json").exists()
        assert len(client.prompts) == expected_calls
        final = json.loads((root / "revision/slots/object_counting/g0m0/final.json").read_text())
        assert final["status"] == status
        if status == "analysis_rejected":
            assert not (root / "revision/slots/object_counting/g0m0/synthesis").exists()


def test_complete_analysis_budget_is_reserved_before_any_request(tmp_path):
    client = responding_client()
    runner = primary_runner(tmp_path, client, n=12, request_cap=4)
    assert runner.run(resume=False)["completed_variants"] == 12
    root = campaign_root(runner)
    refusal = json.loads((root / "input_budget_refusal.json").read_text())
    assert refusal["budget_unit"] == "requests"
    assert refusal["needed"] == 5 and refusal["available"] == 4
    assert refusal["model_requests_initiated"] == 0
    assert client.prompts == []
    assert not list(root.rglob("request_intent_*.json"))
    assert not (root / "decision.json").exists()


@pytest.fixture
def direct_revision(tmp_path):
    runner = make_runner(
        tmp_path / "evidence", n=12, learning_input_mode="full_cases_v1",
        revision_client_factory=full_evidence_factory)
    runner.run(resume=False)
    original = campaign_root(runner)
    experience = V11ExperienceBundle.model_validate_json(
        (original / "experience_bundle.json").read_bytes())
    feedback = json.loads((original / "feedback.json").read_text())
    client = responding_client()
    reviser = FitSetSlotReviser(
        output_root=tmp_path / "revision", panel=runner.panel,
        panel_name=runner.panel_name, variants=[], accumulation_root=runner.root,
        label_metadata={}, library_root=LIBRARY, client=client,
        evidence_request_budget=15,
        method_context_max_chars=runner.base_cfg.retrieval_policy.method_context_max_chars)
    arguments = {
        "slot": ("object_counting", "g0m0"), "parent": parent(),
        "feedback": feedback, "forbidden_ids": set(), "experience": experience,
        "learning_input_mode": "analysis_synthesis_v1",
    }
    return reviser, client, arguments


def test_completed_edit_recovery_does_not_repeat_analysis(direct_revision):
    reviser, client, arguments = direct_revision
    candidate = reviser._revise_slot(**arguments)
    root = reviser.root / "slots/object_counting/g0m0"
    (root / "final.json").unlink()
    recovered = reviser._revise_slot(**arguments)
    assert recovered == candidate and len(client.prompts) == 2
    receipt = root / "analysis/analysis_final.json"
    receipt.write_text(receipt.read_text() + "\n")
    with pytest.raises(RevisionOperationalError, match="receipts changed"):
        reviser._revise_slot(**arguments)
    assert len(client.prompts) == 2


def test_interrupted_analysis_never_replays_a_paid_request(direct_revision):
    reviser, client, arguments = direct_revision
    original = client.respond
    client.respond = lambda _messages: (_ for _ in ()).throw(ConnectionError("synthetic failure"))
    with pytest.raises(RevisionOperationalError, match="service failed"):
        reviser._revise_slot(**arguments)
    client.respond = original
    with pytest.raises(RevisionOperationalError, match="not replayed"):
        reviser._revise_slot(**arguments)
    assert client.calls == 1


def test_multiple_batches_are_independent_and_all_feed_one_synthesis(direct_revision):
    from skill3d.evolution.revision_edits_v11 import RevisionBudget
    from skill3d.evolution.trajectory_analysis_v11 import (
        TrajectoryAnalyst, compact_analysis_feedback, prepare_analysis_batches,
    )

    reviser, client, arguments = direct_revision
    reviser.evidence_request_budget = 60
    for limit in (60000, 50000, 40000, 30000, 25000):
        reviser.input_max_bytes = limit
        analyst = TrajectoryAnalyst(
            output_root=reviser.root / "planning", client=client,
            budget=RevisionBudget(input_max_bytes=limit),
            method_context_max_chars=reviser.method_context_max_chars,
            expected_model="test-model")
        try:
            batches, _ = prepare_analysis_batches(
                reviser, analyst=analyst, parent=arguments["parent"],
                feedback=compact_analysis_feedback(arguments["feedback"]),
                experience=arguments["experience"])
        except RevisionOperationalError:
            continue
        if len(batches) > 1:
            break
    else:
        pytest.fail("Synthetic full cases did not exercise bounded multi-batch analysis")
    assert client.calls == 0
    candidate = reviser._revise_slot(**arguments)
    assert candidate and len(client.prompts) == len(batches) + 1
    assert all("The method may need a completeness check." not in prompt
               for prompt in client.prompts[:-1])
    analyses = json.loads(client.prompts[-1].split(
        "No verification/test feedback is present.\n")[1])
    assert len(analyses) == len(batches)
    assert {row["case_index"] for batch in analyses for row in batch["records"]} == set(range(12))


def test_primary_artifacts_use_instance_local_bounded_writers(direct_revision):
    from skill3d.evolution import fitset_revision_v11, revision_edits_v11, trajectory_analysis_v11
    from skill3d.storage.framework_state_io import FrameworkStateIO

    reviser, client, arguments = direct_revision
    original_writers = [module._write_create_once for module in (
        fitset_revision_v11, revision_edits_v11, trajectory_analysis_v11)]
    written = set()

    class Storage:
        adapt_analysis_reviser = FrameworkStateIO.adapt_analysis_reviser

        def validate_root(self, path):
            assert Path(path).is_relative_to(reviser.root)

        def write_json_once(self, path, value):
            self.validate_root(path)
            written.add(Path(path))
            fitset_revision_v11._write_create_once(path, value)

        def write_text_once(self, path, value):
            self.validate_root(path)
            written.add(Path(path))
            fitset_revision_v11._write_text_create_once(path, value)

    FrameworkStateIO.adapt_reviser(Storage(), reviser)
    assert reviser._revise_slot(**arguments)
    assert len(client.prompts) == 2
    assert written == {path for path in reviser.root.rglob("*") if path.is_file()}
    assert original_writers == [module._write_create_once for module in (
        fitset_revision_v11, revision_edits_v11, trajectory_analysis_v11)]


def test_orphan_request_without_plan_stops_before_any_new_call(direct_revision):
    reviser, client, arguments = direct_revision
    directory = reviser.root / "slots/object_counting/g0m0/analysis"
    directory.mkdir(parents=True)
    (directory / "request_intent_02.json").write_text("{}")
    with pytest.raises(RevisionOperationalError, match="without its frozen plan"):
        reviser._revise_slot(**arguments)
    assert client.calls == 0


def test_analysis_and_editor_share_the_solver_request_ledger(direct_revision, tmp_path):
    from skill3d.synthesis.request_governor import (
        GovernedClient, RequestGovernor, RequestGovernorConfig,
    )

    reviser, client, arguments = direct_revision
    governor = RequestGovernor(
        tmp_path / "requests", RequestGovernorConfig(max_requests=5, max_inflight=1))
    solver = GovernedClient(Client(lambda _messages: "{}"), governor, category="solver")
    solver.chat([{"role": "user", "content": "synthetic solver input"}], seed=137)
    reviser.client = GovernedClient(client, governor, category="reviser")
    candidate = reviser._revise_slot(**arguments)
    assert candidate and governor.summary()["completed_requests"] == 3
    assert reviser._revise_slot(**arguments) == candidate
    assert governor.summary()["completed_requests"] == 3
