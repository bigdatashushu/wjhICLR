"""Offline diagnosis preserves sources, budgets, image bytes and replay boundaries."""

import base64
import hashlib
from io import BytesIO
import json
from pathlib import Path
from types import SimpleNamespace

from PIL import Image
import pytest

from skill3d.evolution.diagnosis_v11 import (
    FrozenCaseImageProvider, OfflineDiagnosisConfig, OfflineDiagnosisError, V11CaseDiagnoser,
)
from skill3d.evolution.revision_input_v11 import build_repair_context, candidate_sample_copy_leaks
from skill3d.skills.registry import load_active_skills
from skill3d.synthesis.request_governor import GovernedClient, RequestGovernor, RequestGovernorConfig

ROOT = Path(__file__).resolve().parents[2]


def evidence():
    parent = next(s for s in load_active_skills(ROOT / "skill_library" / "snapshots")[0]
                  if s.slot_key == "object_counting/g0m0")
    cases = [{
        "case_id": f"case-{i}", "score": score, "question_text": "Count all chairs in this room.",
        "reference_answer": "2", "model_answer": "1",
        "delivered_content_sha256": parent.content_sha256,
        "rounds": [{"program_source": "ReturnAnswer('1')", "results": []}],
        "image_refs": [],
    } for i, score in enumerate([0, 1, 0.5])]
    value = {
        "source_split": "induction", "label_access": True, "campaign_id": "diagnostic-test",
        "parent_skill_version": f"{parent.skill_id}@{parent.version}",
        "parent_content_sha256": parent.content_sha256, "cases": cases,
    }
    return parent, value


class Client:
    def __init__(self, invalid=False, fail=False):
        self.calls = []
        self.invalid = invalid
        self.fail = fail
        self.last_usage = {"total_tokens": 7}

    def chat(self, messages, **kwargs):
        self.calls.append((messages, kwargs))
        if self.fail:
            raise RuntimeError("provider lost completion")
        content = messages[-1]["content"]
        text = content if isinstance(content, str) else content[0]["text"]
        source = json.loads(text.split("\n", 1)[1])
        return json.dumps({
            "observations": [{"statement": "The recorded answer is one.",
                              "evidence_refs": ["/cases/0/rounds/0" if not self.invalid else "/fake"]}],
            "hypotheses": [{
                "statement": "A completeness check may help.",
                "evidence_refs": ["/cases/0/rounds/0"],
                "parent_step_refs": [source["parent_step_map"][0]["ref"]],
                "confidence": "low", "alternatives": ["Visual evidence may be incomplete."],
            }], "unknowns": ["The cause of the wrong answer is unknown."],
        })


def test_source_linked_balanced_cases_and_resume_share_paid_budget(tmp_path):
    parent, bundle = evidence()
    raw = Client()
    governor = RequestGovernor(tmp_path / "budget", RequestGovernorConfig(1, 1))
    client = GovernedClient(raw, governor, category="diagnosis")
    diagnoser = V11CaseDiagnoser(client, tmp_path / "audit", OfflineDiagnosisConfig(mode="text"))
    first = diagnoser.diagnose(parent, bundle)
    assert first["selected_case_ids"] == ["case-1", "case-0", "case-2"]
    assert first["source_excerpts"]["cases"][0] == bundle["cases"][1]
    assert first == diagnoser.diagnose(parent, bundle)
    assert len(raw.calls) == 1 and governor.summary()["categories"] == {"diagnosis": 1}
    assert governor.summary()["usage"] == {"total_tokens": 7}
    changed = {**bundle, "cases": [*bundle["cases"], {**bundle["cases"][0], "case_id": "new"}]}
    with pytest.raises(Exception, match="immutable"):
        diagnoser.diagnose(parent, changed)
    assert len(raw.calls) == 1


def test_invalid_source_reference_is_not_reasked(tmp_path):
    parent, bundle = evidence()
    client = Client(invalid=True)
    diagnoser = V11CaseDiagnoser(client, tmp_path, OfflineDiagnosisConfig(mode="text"))
    for _ in range(2):
        with pytest.raises(OfflineDiagnosisError, match="nonexistent"):
            diagnoser.diagnose(parent, bundle)
    assert len(client.calls) == 1


def test_model_policy_change_cannot_reuse_old_completion_but_usage_can_change(tmp_path):
    class IdentifiedClient(Client):
        model = "offline-model-a"

        def manifest_fields(self):
            return {
                "provider": "fixture", "model_id": self.model,
                "endpoint_hash": "e" * 64, "request_params": {"max_retries": 0},
                "token_usage": {"total_tokens": len(self.calls) * 7},
            }

    parent, bundle = evidence()
    client = IdentifiedClient()
    diagnoser = V11CaseDiagnoser(client, tmp_path, OfflineDiagnosisConfig(mode="text"))
    first = diagnoser.diagnose(parent, bundle)
    assert diagnoser.diagnose(parent, bundle) == first
    client.model = "offline-model-b"
    with pytest.raises(Exception, match="immutable"):
        diagnoser.diagnose(parent, bundle)
    assert len(client.calls) == 1


def test_unknown_completion_is_not_replayed(tmp_path):
    parent, bundle = evidence()
    client = Client(fail=True)
    diagnoser = V11CaseDiagnoser(client, tmp_path, OfflineDiagnosisConfig(mode="text"))
    with pytest.raises(RuntimeError):
        diagnoser.diagnose(parent, bundle)
    with pytest.raises(OfflineDiagnosisError, match="completion unknown"):
        diagnoser.diagnose(parent, bundle)
    assert len(client.calls) == 1


def test_text_client_truncation_cannot_be_accepted_even_with_valid_json(tmp_path):
    parent, bundle = evidence()
    client = Client()
    client.last_generation = {"finish_reason": "length"}
    diagnoser = V11CaseDiagnoser(client, tmp_path, OfflineDiagnosisConfig(mode="text"))
    for _ in range(2):
        with pytest.raises(OfflineDiagnosisError, match="truncated"):
            diagnoser.diagnose(parent, bundle)
    assert len(client.calls) == 1


def test_no_truncated_case_or_request_when_budget_cannot_fit(tmp_path):
    parent, bundle = evidence()
    client = Client()
    diagnoser = V11CaseDiagnoser(
        client, tmp_path, OfflineDiagnosisConfig(mode="text", max_input_chars=1))
    with pytest.raises(OfflineDiagnosisError, match="no complete case"):
        diagnoser.diagnose(parent, bundle)
    assert not client.calls


def test_multimodal_supplies_frozen_pixels_and_hash_verified_limit(tmp_path):
    parent, bundle = evidence()
    buffer = BytesIO()
    Image.new("RGB", (4, 4), (23, 47, 71)).save(buffer, format="PNG")
    data = buffer.getvalue()
    path = tmp_path / "frame.png"
    path.write_bytes(data)
    ref = {"path": str(path), "file_sha256": hashlib.sha256(data).hexdigest()}
    for case in bundle["cases"]:
        case["image_refs"] = [ref, ref]
    client = Client()
    diagnoser = V11CaseDiagnoser(
        client, tmp_path / "audit", OfflineDiagnosisConfig(mode="multimodal", max_images=1))
    provider = FrozenCaseImageProvider([tmp_path])
    result = diagnoser.diagnose(parent, bundle, image_provider=provider)
    actual = [part for part in client.calls[0][0][-1]["content"] if part["type"] == "image_url"]
    assert len(actual) == len(result["supplied_images"]) == 1
    assert base64.b64decode(actual[0]["image_url"]["url"].split(",", 1)[1]) == data
    path.write_bytes(b"changed")
    with pytest.raises(OfflineDiagnosisError, match="hash mismatch"):
        diagnoser.diagnose(parent, bundle, image_provider=provider)
    assert len(client.calls) == 1


def test_multimodal_refuses_unresolved_refs_before_api(tmp_path):
    parent, bundle = evidence()
    bundle["cases"][0]["image_refs"] = [{"image_id": "frame-0"}]
    client = Client()
    diagnoser = V11CaseDiagnoser(client, tmp_path, OfflineDiagnosisConfig(mode="multimodal"))
    with pytest.raises(OfflineDiagnosisError, match="no verifiable image"):
        diagnoser.diagnose(parent, bundle, image_provider=lambda *_: {})
    assert not client.calls


def test_repair_redacts_multiline_copy_even_when_other_line_already_redacted():
    question = "Which of all the many objects stands closest to the wooden cabinet?"
    case = {"case_id": "qa-secret", "question_text": question}
    text = "Mention qa-secret.\nWhich of all the many objects\nstands closest to the wooden cabinet?\n"
    result = build_repair_context(text, [{"code": "leakage_detected"}], cases=[case])
    assert result["candidate_sha256"] == hashlib.sha256(text.encode()).hexdigest()
    assert result["redacted_lines"] == [1, 2, 3]
    assert not result["fully_delivered"] and not candidate_sample_copy_leaks(result["candidate_text"], [case])


@pytest.mark.parametrize("usage", [
    {"usage": {"total_tokens": 9, "unsafe": "private"}},
    {"total_tokens": 9, "prompt_tokens": 5, "completion_tokens": 4},
])
def test_meta_requests_are_governed_and_usage_is_counted(tmp_path, usage):
    class MetaClient:
        def chat_with_meta(self, *_args, **_kwargs):
            return SimpleNamespace(text="ok", **usage)

    governor = RequestGovernor(tmp_path, RequestGovernorConfig(1, 1))
    client = GovernedClient(MetaClient(), governor, category="reviser")
    assert client.chat_with_meta([]).text == "ok"
    assert governor.summary()["reserved_requests"] == 1
    assert governor.summary()["usage"]["total_tokens"] == 9
    assert "unsafe" not in governor.summary()["usage"]
