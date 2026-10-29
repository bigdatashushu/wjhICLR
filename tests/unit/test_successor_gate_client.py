"""Gate service and effective-wire failures must stop, with fixed error codes."""
import importlib.util
import json
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import pytest

from skill3d.synthesis.request_audit import describe_request

_path = Path(__file__).resolve().parents[2] / "scripts/run_successor_gate_v11.py"
_spec = importlib.util.spec_from_file_location("successor_gate_test", _path)
gate = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(gate)


class Client:
    transport_retries = 3

    def audit_identity(self, **kwargs):
        return {"model": "test-model", **kwargs}

    def build_payload(self, *args, **kwargs):
        return {"messages": args[0], "model": "test-model", **kwargs}

    def chat(self, *args, **kwargs):
        self.last_request_identity = describe_request(
            self.build_payload(*args, **kwargs),
            model_identity=self.audit_identity(seed=kwargs["seed"]),
            wire_encoding_known=True)
        self.last_response_identity = {"model": "test-model"}
        return "received"


def test_gate_checks_effective_wire_identity_and_has_zero_retries():
    client = Client()
    checked = gate.CheckedClient(client, "test-model", 137)
    assert checked.transport_retries == 0
    assert checked.chat([{"role": "user", "content": "public input"}], seed=137) == "received"
    with pytest.raises(gate.GateClientError, match="gate_request_identity"):
        checked.chat([], seed=138)


def test_gate_service_exception_is_replaced_by_content_free_error():
    class Failed(Client):
        def chat(self, *args, **kwargs):
            raise RuntimeError("provider text must not escape")

    checked = gate.CheckedClient(Failed(), "test-model", 137)
    with pytest.raises(gate.GateClientError) as error:
        checked.chat([], seed=137)
    assert str(error.value) == "gate_service_failure"


def test_gate_rejects_returned_model_change():
    class Changed(Client):
        def chat(self, *args, **kwargs):
            text = super().chat(*args, **kwargs)
            self.last_response_identity["model"] = "changed"
            return text

    checked = gate.CheckedClient(Changed(), "test-model", 137)
    with pytest.raises(gate.GateClientError, match="gate_response_or_wire_identity"):
        checked.chat([], seed=137)


def test_complete_zero_request_preflight_seals_workspace_binding(tmp_path, monkeypatch):
    @dataclass
    class Config:
        gate_seeds: tuple = (137, 138)
        gate_mode: str = "qa_disjoint"

    @dataclass
    class SolverConfig:
        seed: int = 137
        temperature: float = 0.0
        max_tokens: int = 256
        max_rounds: int = 4

    class Writer:
        def write_json_once(self, path, value):
            Path(path).write_text(json.dumps(value))

    runner = SimpleNamespace(base_cfg=SolverConfig(), framework_state_io=Writer(),
                             request_governor_config=SimpleNamespace(max_requests=40))

    class Arm:
        label_metadata = {}
        capture_offline_case_input = False

        def __init__(self, name):
            self.root = tmp_path / "campaigns/test/gate/137" / name

        def _input_episode_lease(self, variant):
            return nullcontext()

        def _prepare_one(self, variant, *, preflight=False):
            assert preflight
            return {"item": SimpleNamespace(episode=SimpleNamespace(ground_truth=None)),
                    "plan": {"frozen_public_input": "same"}}

    class Worker:
        config = Config()

        def _gate(self, request, parent, candidate, root):
            runner._run_gate_episodes(
                [(Arm(name), {"variant_id": "test-variant"}) for name in ("parent", "candidate")])

    spec = SimpleNamespace(slot_key="object_counting/g0m0", question_type="object_counting",
                           evidence_state="g0m0", content_sha256="0" * 64)
    candidate = SimpleNamespace(content_sha256="1" * 64)
    revision_ready = {"protected_inputs": {}, "source_identity": {}, "gate_request_cap": 512,
                      "model_identity": {"model": "test-model"}}
    monkeypatch.setattr(gate, "setup", lambda args: (
        SimpleNamespace(protected=lambda *args: {}), runner, Worker(),
        [({"campaign_id": "test-campaign"}, spec, candidate)], revision_ready))
    binding = {"cache_root": "test-cache", "workspace_bytes": 2_000_000_000}
    (tmp_path / "framework_io_binding.json").write_text(json.dumps(binding))
    revision = tmp_path / "revision"
    revision.mkdir()
    (revision / "revision_final.json").write_text("{}")
    args = SimpleNamespace(output=tmp_path, revision_root=revision, state_root=tmp_path, frozen_root=tmp_path)
    result = gate.preflight(args)
    receipt = json.loads((tmp_path / "preflight.json").read_text())
    assert result["model_requests"] == 0 and result["planned_episodes"] == 2
    assert receipt["framework_workspace_binding"] == binding
    assert receipt["request_cap"] == 40
    assert receipt["validation_labels_hidden_during_execution"]
    assert receipt["admission_policy_source_identity"] == gate.identity(gate.ADMISSION_POLICY_PATH)


def test_successor_native_scoring_uses_new_admission_and_preserves_other_checks():
    parent = {
        "mean_score": 0.5, "errors": 0, "program_errors": 1, "legal_rate": 1.0,
        "shared_input_hashes": ["same"], "returned_models": ["test-model"],
        "backend_fingerprints": [], "trusted": True, "delivered": True,
        "service_ok": True, "seed_observed": True,
    }
    candidate = {**parent, "mean_score": 0.75, "program_errors": 4}

    class Worker:
        def _gate(self):
            return {
                "passed": False, "conditions": {"137": False, "138": False},
                "admission_program_error_rule": "legacy_any_round_error_code",
                "arms": [{"seed": seed, "parent": parent, "candidate": candidate}
                         for seed in (137, 138)],
            }

    worker = Worker()
    gate.bind_gate_admission(worker)
    result = worker._gate()
    assert result["passed"] and result["conditions"] == {"137": True, "138": True}
    assert result["arms"][0]["candidate"]["program_errors"] == 4
    assert "admission_program_error_rule" not in result
    assert result["admission_rule_version"] == "fitset-gate-admission/2"
    candidate["errors"] = 1
    result = worker._gate()
    assert not result["passed"] and result["conditions"] == {"137": False, "138": False}
