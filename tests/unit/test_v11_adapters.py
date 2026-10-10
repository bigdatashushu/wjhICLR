"""CPU contracts for the production v11 campaign adapters."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from skill3d.adapters.episode_source import load_synthetic_items
from skill3d.evaluation.skill_ablation_v11 import quality_contract
from skill3d.evolution import adapters_v11 as adapters
from skill3d.online.runner import OnlineRunConfig
from skill3d.schemas import SkillCandidateV11, SkillSpecV11

ROOT = Path(__file__).resolve().parents[2]
SOURCE = (
    ROOT
    / "skill_library"
    / "versions"
    / "S03"
    / "1.1.0"
    / "rank-object-distances"
    / "SKILL.md"
).read_text(encoding="utf-8")


def _parent() -> SkillSpecV11:
    return SkillSpecV11(
        skill_id="S03",
        version="1.1.0",
        question_type="object_rel_distance",
        skill_md=SOURCE,
    )


def _candidate() -> SkillCandidateV11:
    spec = SkillSpecV11(
        skill_id="S03",
        version="1.2.0",
        question_type="object_rel_distance",
        skill_md=SOURCE.replace(
            "重点核查会改变前两名的错误绑定",
            "优先核查会改变前两名的错误绑定",
            1,
        ),
    )
    return SkillCandidateV11(
        candidate_id="cand-s03-test",
        parent_snapshot_id="snapshot-parent",
        parent_skill_version="S03@1.1.0",
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
        self.health_calls = 0
        self.calls = []

    def require_service(self):
        self.health_calls += 1

    def chat_with_meta(self, messages, **kwargs):
        self.calls.append((messages, kwargs))
        return _Meta(self.text)


class _Experience:
    parent_snapshot_id = "snapshot-parent"
    cases = []

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
        static_feedback=[],
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
    repeated = reviser(
        campaign_id="campaign-test", parent=_parent(), experience=_Experience(),
        attempt=1, static_feedback=[], idempotency_key="campaign-test:revise:1")
    assert repeated == proposal
    assert len(client.calls) == 1


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
        vllm_endpoints=["http://127.0.0.1:8100"],
        vllm_model="qwen-test",
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
            model_id=cfg.vllm_model, model_config_sha256=model_identity["sha256"],
            quality_contract_sha256=contract["sha256"],
            solver_config_sha256=solver_identity["sha256"],
            idempotency_key="campaign-test:evaluate",
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

    assert observation.delivered_skill_version == "S03@1.2.0"
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
