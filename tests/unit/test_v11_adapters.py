"""CPU contracts for the production v11 campaign adapters."""

from __future__ import annotations

import json
import re
from pathlib import Path
from types import SimpleNamespace

import pytest

from skill3d.adapters.episode_source import load_synthetic_items
from skill3d.evaluation.skill_ablation_v11 import quality_contract
from skill3d.evolution import adapters_v11 as adapters
from skill3d.online.runner import OnlineRunConfig
from skill3d.schemas import SkillCandidateV11, SkillSpecV11
from skill3d.skills.delivery import check_service_limit

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


def _parent() -> SkillSpecV11:
    return SkillSpecV11(
        skill_id="S03",
        version="1.2.0",
        question_type="object_rel_distance",
        evidence_state="g1m0",
        skill_md=SOURCE,
    )


def _candidate() -> SkillCandidateV11:
    spec = SkillSpecV11(
        skill_id="S03",
        version="1.3.0",
        question_type="object_rel_distance",
        evidence_state="g1m0",
        skill_md=SOURCE.replace(
            "Focus on incorrect bindings",
            "Prioritize checking incorrect bindings",
            1,
        ),
    )
    return SkillCandidateV11(
        candidate_id="cand-s03-test",
        parent_snapshot_id="snapshot-parent",
        parent_skill_version="S03@1.2.0",
        full_skill_spec=spec,
        modification_reason="improve ambiguity handling",
        source_run_ref="offline:test",
    )


class _Meta:
    def __init__(self, text: str):
        self.text = text
        self.request_id = "request-1"
        self.truncated = False

    def manifest_fields(self):
        return {
            "request_id": self.request_id,
            "text_sha256": "f" * 64,
            "n_chars": len(self.text),
        }


class _OfflineClient:
    def __init__(self, text: str):
        self.text = text
        self.model = "fixture-offline-model"
        self.health_calls = 0
        self.calls = []

    def manifest_fields(self):
        return {
            "provider": "fixture", "model_id": self.model,
            "endpoint_hash": "e" * 64, "prompt_version": "fixture-prompt",
            "request_params": {"temperature": 0},
            "health_checks": self.health_calls, "statistics": {"calls": len(self.calls)},
        }

    def require_service(self):
        self.health_calls += 1

    def chat_with_meta(self, messages, **kwargs):
        self.calls.append((messages, kwargs))
        return _Meta(self.text)


class _Experience:
    parent_snapshot_id = "snapshot-parent"
    cases = []

    def model_copy(self, **_kwargs):
        return _Experience()

    def model_dump(self, **_kwargs):
        return {
            "source_split": "induction",
            "label_access": True,
            "cases": [{
                "case_id": "private-case",
                "question_text": "Which object is closest?",
                "reference_answer": "B",
            }],
        }


def test_offline_reviser_returns_complete_source_and_writes_metadata_only(tmp_path):
    candidate_md = _candidate().full_skill_spec.skill_md
    feedback = check_service_limit(_parent(), max_chars=1)
    assert feedback
    client = _OfflineClient(json.dumps({
        "full_skill_md": candidate_md,
        "modification_reason": "Use a stronger ambiguity check.",
    }))
    reviser = adapters.V11OfflineReviser(
        client,
        audit_root=tmp_path,
        require_health_check=True,
    )

    proposal = reviser(
        campaign_id="campaign-test",
        parent=_parent(),
        experience=_Experience(),
        attempt=1,
        static_feedback=feedback,
        idempotency_key="campaign-test:revise:1",
    )

    assert proposal.full_skill_md == candidate_md
    assert client.health_calls == 1
    assert client.calls[0][1]["seed"] == 1
    audit_path = tmp_path / "campaign-test" / "revision_01.json"
    audit = audit_path.read_text(encoding="utf-8")
    assert "prompt_sha256" in audit and "response_sha256" in audit
    assert "Which object is closest?" not in audit
    assert "reference_answer" not in audit
    assert str(audit_path) in proposal.source_run_ref
    assert "Current Tool interfaces" in client.calls[0][0][1]["content"]
    tool_text = client.calls[0][0][1]["content"].split("Current Tool interfaces", 1)[1].split(
        "Induction experience JSON", 1)[0]
    assert "- relative_distance_rank(" in tool_text
    assert "- object_surface_distance(" not in tool_text
    assert "- object_centroid(" not in tool_text
    assert "- reproject(" not in tool_text
    assert "- plane_fit_room_size(" not in tool_text
    assert "- camera_object_distance(" not in tool_text
    assert not re.search(r"[\u4e00-\u9fff]", json.dumps(client.calls[0][0], ensure_ascii=False))
    assert all(problem in client.calls[0][0][1]["content"] for problem in feedback)
    repeated = reviser(
        campaign_id="campaign-test", parent=_parent(), experience=_Experience(),
        attempt=1, static_feedback=feedback, idempotency_key="campaign-test:revise:1")
    assert repeated == proposal
    assert len(client.calls) == 1
    frozen = json.loads((tmp_path / "campaign-test/revision_input.json").read_bytes())
    identity = frozen["client_identity"]
    assert set(identity) == {
        "provider", "model_id", "endpoint_hash", "prompt_version", "request_params",
    }
    exchange = json.loads((tmp_path / "campaign-test/revision_01.exchange.json").read_bytes())
    assert exchange["request"]["client_identity"] == identity


def test_invalid_offline_response_is_recorded_and_not_requested_again(tmp_path):
    client = _OfflineClient("not JSON")
    reviser = adapters.V11OfflineReviser(client, audit_root=tmp_path)
    for _ in range(2):
        with pytest.raises(adapters.V11RevisionFormatError):
            reviser(
                campaign_id="bad", parent=_parent(), experience=_Experience(),
                attempt=1, static_feedback=[], idempotency_key="bad:revise:1")
    assert len(client.calls) == 1
    exchange = json.loads((tmp_path / "bad/revision_01.exchange.json").read_bytes())
    assert exchange["text"] == "not JSON"


def _revision_kwargs(**changes):
    return {
        "campaign_id": "revision-test", "parent": _parent(),
        "experience": _Experience(), "attempt": 1, "static_feedback": [],
        "idempotency_key": "revision-test:revise:1", **changes,
    }


def test_offline_no_change_preserves_reason_and_response_receipt(tmp_path):
    client = _OfflineClient(json.dumps({
        "action": "no_change", "modification_reason": "Evidence is insufficient.",
    }))
    reviser = adapters.V11OfflineReviser(client, audit_root=tmp_path)
    proposal = reviser(**_revision_kwargs())
    assert proposal.action == "no_change"
    assert proposal.full_skill_md is None
    assert proposal.modification_reason == "Evidence is insufficient."
    assert "response:" in proposal.source_run_ref
    assert reviser(**_revision_kwargs()) == proposal
    assert len(client.calls) == 1


@pytest.mark.parametrize("source", [None, "", "method source"])
def test_offline_no_change_rejects_any_explicit_source_field(tmp_path, source):
    client = _OfflineClient(json.dumps({
        "action": "no_change", "full_skill_md": source, "modification_reason": "Keep parent.",
    }))
    reviser = adapters.V11OfflineReviser(client, audit_root=tmp_path)
    with pytest.raises(adapters.V11RevisionFormatError, match="must omit full_skill_md"):
        reviser(**_revision_kwargs())
    assert len(client.calls) == 1


def test_offline_revision_preserves_declared_change_metadata(tmp_path):
    client = _OfflineClient(json.dumps({
        "action": "revise", "full_skill_md": _candidate().full_skill_spec.skill_md,
        "modification_reason": "Clarify the check.",
        "change_hypothesis": "An explicit trigger reduces omissions.",
        "changed_sections": ["Method"], "expected_process_effect": "Check ambiguity first.",
        "evidence_signature_refs": ["evidence:test"],
    }))
    proposal = adapters.V11OfflineReviser(client, audit_root=tmp_path)(**_revision_kwargs())
    assert proposal.change_hypothesis == "An explicit trigger reduces omissions."
    assert proposal.changed_sections == ["Method"]
    assert proposal.expected_process_effect == "Check ambiguity first."
    assert proposal.evidence_signature_refs == ["evidence:test"]


@pytest.mark.parametrize("response", [
    '{"full_skill_md": "incomplete',
    json.dumps({"full_skill_md": "prior complete candidate", "modification_reason": ""}),
    json.dumps({"action": "invalid", "full_skill_md": "prior complete candidate",
                "modification_reason": "A reason."}),
])
def test_format_failure_keeps_actual_candidate_or_malformed_response(tmp_path, response):
    client = _OfflineClient(response)
    reviser = adapters.V11OfflineReviser(client, audit_root=tmp_path)
    with pytest.raises(adapters.V11RevisionFormatError) as captured:
        reviser(**_revision_kwargs())
    expected = response if response.endswith("incomplete") else "prior complete candidate"
    assert captured.value.candidate_text == expected
    assert "response:" in captured.value.source_run_ref


def test_unknown_completion_never_replays_even_with_a_new_reviser(tmp_path):
    class InterruptedClient(_OfflineClient):
        def chat_with_meta(self, messages, **kwargs):
            self.calls.append((messages, kwargs))
            raise RuntimeError("connection lost after request submission")

    client = InterruptedClient("")
    first = adapters.V11OfflineReviser(client, audit_root=tmp_path)
    with pytest.raises(RuntimeError, match="connection lost"):
        first(**_revision_kwargs())
    assert (tmp_path / "revision-test/revision_01.intent.json").is_file()
    assert not (tmp_path / "revision-test/revision_01.exchange.json").exists()
    resumed = adapters.V11OfflineReviser(client, audit_root=tmp_path)
    with pytest.raises(adapters.V11AdapterError, match="unknown"):
        resumed(**_revision_kwargs())
    assert len(client.calls) == 1


@pytest.mark.parametrize("changed", ["evidence", "max_tokens", "model", "feedback", "repair"])
def test_frozen_revision_request_rejects_changed_resume_inputs(tmp_path, changed):
    client = _OfflineClient(json.dumps({
        "action": "no_change", "modification_reason": "Keep the parent.",
    }))
    reviser = adapters.V11OfflineReviser(client, audit_root=tmp_path)
    reviser(**_revision_kwargs())
    kwargs = _revision_kwargs()
    if changed == "evidence":
        class ChangedExperience(_Experience):
            def model_dump(self, **kwargs):
                value = super().model_dump(**kwargs)
                value["cases"][0]["question_text"] = "Different frozen evidence"
                return value

        kwargs["experience"] = ChangedExperience()
    elif changed == "max_tokens":
        reviser = adapters.V11OfflineReviser(client, audit_root=tmp_path, max_tokens=100)
    elif changed == "model":
        client.model = "different-offline-model"
    elif changed == "feedback":
        kwargs["static_feedback"] = ["changed feedback"]
    else:
        kwargs["repair_context"] = {"candidate_text": "changed previous candidate"}
    with pytest.raises(adapters.V11AdapterError, match="changed"):
        reviser(**kwargs)
    assert len(client.calls) == 1


def test_text_diagnosis_is_frozen_once_and_original_cases_reach_every_revision(tmp_path):
    class Diagnoser:
        config = {"mode": "text", "max_cases": 4, "max_input_chars": 64000,
                  "max_images": 8, "max_output_tokens": 4096}

        def __init__(self):
            self.calls = []

        def diagnose(self, parent, experience, image_provider=None, context=None):
            self.calls.append((parent, experience, image_provider, context))
            return {"facts": ["observed failure"], "hypotheses": ["uncertain cause"]}

    diagnosis = Diagnoser()
    client = _OfflineClient(json.dumps({
        "action": "no_change", "modification_reason": "Insufficient evidence.",
    }))
    reviser = adapters.V11OfflineReviser(client, audit_root=tmp_path, diagnoser=diagnosis)
    reviser(**_revision_kwargs())
    repair = {"candidate_text": "prior failed candidate", "candidate_sha256": "c" * 64}
    second_kwargs = _revision_kwargs(
        attempt=2, idempotency_key="revision-test:revise:2", repair_context=repair,
    )
    reviser(**second_kwargs)
    resumed = adapters.V11OfflineReviser(client, audit_root=tmp_path, diagnoser=diagnosis)
    resumed(**second_kwargs)
    assert len(diagnosis.calls) == 1
    assert diagnosis.calls[0][2] is None
    assert diagnosis.calls[0][3]["idempotency_key"] == "revision-test:diagnose"
    assert len(client.calls) == 2
    for messages, _ in client.calls:
        prompt = messages[1]["content"]
        assert "Which object is closest?" in prompt
        assert "reference_answer" in prompt
        assert "observed failure" in prompt
        assert "uncertain cause" in prompt
    assert "prior failed candidate" in client.calls[1][0][1]["content"]
    assert "c" * 64 in client.calls[1][0][1]["content"]
    off = adapters.V11OfflineReviser(client, audit_root=tmp_path)
    with pytest.raises(adapters.V11AdapterError, match="frozen revision input changed"):
        off(**_revision_kwargs())


def test_multimodal_diagnosis_requires_explicit_real_image_provider(tmp_path):
    diagnosis = SimpleNamespace(config={"mode": "multimodal"})
    with pytest.raises(adapters.V11AdapterError, match="explicit image provider"):
        adapters.V11OfflineReviser(_OfflineClient(""), audit_root=tmp_path, diagnoser=diagnosis)


@pytest.mark.parametrize("attempt", [0, 101])
def test_revision_fresh_request_guard_has_a_bounded_attempt_identity(tmp_path, attempt):
    client = _OfflineClient("")
    reviser = adapters.V11OfflineReviser(client, audit_root=tmp_path)
    with pytest.raises(adapters.V11AdapterError, match=r"\[1, 100\]"):
        reviser(**_revision_kwargs(attempt=attempt))
    assert not client.calls


def test_audit_identity_is_shared_stable_across_repairs_and_bound_to_request_seed(tmp_path):
    class AuditedClient(_OfflineClient):
        def audit_identity(self, *, seed):
            return {"provider": "audited", "model_id": self.model, "seed": seed}

    client = AuditedClient(json.dumps({
        "action": "no_change", "modification_reason": "Keep the method.",
    }))
    reviser = adapters.V11OfflineReviser(client, audit_root=tmp_path)
    reviser(**_revision_kwargs())
    second = _revision_kwargs(attempt=2, idempotency_key="revision-test:revise:2")
    reviser(**second)
    frozen = json.loads((tmp_path / "revision-test/revision_input.json").read_bytes())
    exchange = json.loads((tmp_path / "revision-test/revision_02.exchange.json").read_bytes())
    assert frozen["client_identity"] == {
        "provider": "audited", "model_id": client.model, "seed": 1,
    }
    assert exchange["request"]["client_identity"] == {
        "provider": "audited", "model_id": client.model, "seed": 2,
    }
    assert reviser(**second).action == "no_change"
    assert len(client.calls) == 2
    client.model = "changed-offline-model"
    with pytest.raises(adapters.V11AdapterError, match="changed"):
        reviser(**second)
    assert len(client.calls) == 2


def _pair_row(seed: int, parent: SkillSpecV11, candidate: SkillSpecV11) -> dict:
    def arm(label: str, spec: SkillSpecV11, score: float):
        key = f"{spec.skill_id}@{spec.version}"
        return {
            "qa_id": "qa-1",
            "input_error": False,
            "score": score,
            "runtime_error": False,
            "program_error": False,
            "untrusted_geometry_used": False,
            "legal_answer": True,
            "delivery_ok": True,
            "delivered_skill_versions": [key],
            "delivered_content_sha256": {key: spec.content_sha256},
            "requests": [{
                "solver": True,
                "seed": seed,
                "success": True,
                "arm": label,
            }],
        }

    return {
        "qa_id": "qa-1",
        "arms": {
            "B01": arm("B01", parent, 0.0),
            "B11": arm("B11", candidate, 1.0),
        },
    }


def test_paired_evaluator_rejects_cache_without_actual_panel(tmp_path):
    seed = 17
    parent = _parent()
    candidate = _candidate()
    cfg = OnlineRunConfig(
        mode="real",
        seed=seed,
        online_model="qwen3.8-flash",
    )
    model_identity = adapters.model_config_identity_v11(cfg)
    solver_identity = adapters.solver_config_identity_v11(cfg, seed=seed)
    contract = quality_contract()
    output = tmp_path / "campaign-test" / "e11"
    output.mkdir(parents=True)
    pair = _pair_row(seed, parent, candidate.full_skill_spec)
    (output / "manifest.json").write_text(json.dumps({
        "experiment_id": "evaluation-1",
        "qa_ids": ["qa-1"],
        "question_types": ["object_rel_distance"],
        "inputs": {"qa-1": {"sha256": "1" * 64}},
        "config_sha256": solver_identity["sha256"],
        "quality_contract": contract,
    }), encoding="utf-8")
    (output / "summary.json").write_text(json.dumps({
        "status": "completed",
        "n_pairs": 1,
        "formal_result_eligible": False,
    }), encoding="utf-8")
    (output / "paired_results.jsonl").write_text(
        json.dumps(pair) + "\n", encoding="utf-8")
    evaluator = adapters.V11PairedEvaluator(
        items=[],
        artifact_paths={},
        base_cfg=cfg,
        output_root=tmp_path,
        library_root=tmp_path / "library",
    )

    with pytest.raises(adapters.V11AdapterError, match="panel"):
        evaluator(
            campaign_id="campaign-test", parent=parent, candidate=candidate,
            experience=_Experience(), question_type=parent.question_type, seed=seed,
            model_id=cfg.online_model, model_config_sha256=model_identity["sha256"],
            quality_contract_sha256=contract["sha256"],
            solver_config_sha256=solver_identity["sha256"],
            idempotency_key="campaign-test:evaluate",
        )


def test_promotion_evaluator_rejects_outer_holdout_source(tmp_path):
    evaluator = adapters.V11PairedEvaluator(
        items=[],
        artifact_paths={},
        base_cfg=OnlineRunConfig(mode="real", seed=17),
        output_root=tmp_path,
        library_root=tmp_path / "library",
        purpose="promotion",
        source_split="outer_holdout",
    )
    with pytest.raises(
        adapters.V11AdapterError,
        match="promotion evaluation must use inner_validation",
    ):
        evaluator(
            campaign_id="campaign-test",
            parent=_parent(),
            candidate=_candidate(),
            experience=_Experience(),
            question_type="object_rel_distance",
            seed=17,
        )


def test_post_publish_verifier_uses_normal_active_loading(
    tmp_path,
    monkeypatch,
):
    candidate = _candidate()
    item = load_synthetic_items(
        "induction",
        question_types=["object_rel_distance"],
        limit=1,
        n_frames=2,
        frame_size=(32, 48),
    )[0]
    artifact = tmp_path / "artifact.json"
    artifact.write_text("{}", encoding="utf-8")
    seen = []

    monkeypatch.setattr(
        adapters,
        "load_active_skills",
        lambda _path: ([candidate.full_skill_spec], [], "snapshot-published"),
    )
    monkeypatch.setattr(
        adapters,
        "active_snapshot_provenance",
        lambda _path: ("snapshot-published", "a" * 64),
    )

    def fake_run_episode(_episode, _pixels, cfg, **_kwargs):
        seen.append(cfg)
        return SimpleNamespace(final_state="answer", rounds=[])

    monkeypatch.setattr(adapters.runner, "run_episode", fake_run_episode)
    monkeypatch.setattr(
        adapters,
        "build_v11_experience_bundle_from_trace_store",
        lambda *_args, **_kwargs: SimpleNamespace(
            cases=[SimpleNamespace(trace_ref="online_run:post:qa:one")],
            source_run_ref="online_run:post",
        ),
    )
    verifier = adapters.V11PostPublishVerifier(
        items=[item],
        artifact_paths={item.episode.qa_id: artifact},
        base_cfg=OnlineRunConfig(mode="real", seed=23),
        output_root=tmp_path / "runs",
        library_root=tmp_path / "library",
    )

    observation = verifier(
        campaign_id="campaign-test",
        candidate=candidate,
        snapshot_id="snapshot-published",
        question_type="object_rel_distance",
        idempotency_key="campaign-test:post-publish",
    )

    assert observation.delivered_skill_version == "S03@1.3.0"
    assert observation.learning_event_refs
    assert len(seen) == 1
    assert seen[0].evaluation_binding is None
    assert seen[0].active_snapshot_ref == "snapshot-published"
    assert seen[0].skills == [candidate.full_skill_spec]
    online = json.loads(
        (
            tmp_path
            / "runs"
            / "campaign-test"
            / "post_publish_learning"
            / "online_run.jsonl"
        ).read_text(encoding="utf-8")
    )
    assert online["split"] == "induction"
    assert online["label_access"] is False
