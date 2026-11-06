"""Live wiring validation without credentials, service calls or simulator launches."""
from copy import deepcopy
from pathlib import Path
import sys

import pytest

import action_live as live
from action_adapters import FakeAdapter
from action_audit import seal_record
from action_bridge import LocalTransport
from action_contract import public_hash
from action_identity import ARMS, source_identity
from action_run import FakeClient
from evidence import CurrentEvidencePreparer, PerceptionBackends
from mutation_protocol import MutationSession
from protocol import ContractError, OperationalError, ROOT, canonical, sha256


def test_output_validation_happens_before_preflight(tmp_path, monkeypatch):
    called = []
    monkeypatch.setattr(live, "preflight", lambda *a: called.append(True))
    with pytest.raises(ContractError):
        live.live_run(ROOT / "config/action-live-4090.json", tmp_path / "outside.json")
    assert called == []


def test_existing_output_is_rejected(tmp_path, monkeypatch):
    monkeypatch.setattr(live, "ROOT", tmp_path)
    output = tmp_path / "run.json"
    output.write_text("preserve")
    with pytest.raises(ContractError):
        live.live_run(tmp_path / "unused.json", output)
    assert output.read_text() == "preserve"


def test_live_operational_stop_preserves_denominator_without_extra_requests(tmp_path, monkeypatch):
    from action_run import FakeDepth
    monkeypatch.setattr(live, "ROOT", tmp_path)
    config = {"integration_step_limit": 1, "require_identical_initial_observation": False}
    item = {"environment": "eb_navigation", "subset": "base", "index": 0,
            "episode_id": "eb_navigation:base:000", "source_sha256": public_hash({"task": 0})}
    preparer = CurrentEvidencePreparer(PerceptionBackends(depth_estimator=FakeDepth()))
    sources = source_identity()["files_sha256"]
    runtime = {
        "schema": "embodied-v2-runtime-check/1", "environment": "eb_navigation", "runtime_ready": True,
        "model_requests": 0, "simulator_launches": 0,
        "source_sha256": {name: sources["embodiedRunner/" + name] for name in
                          ("action_worker_v2.py", "action_adapters.py", "mutation_protocol.py", "protocol.py")},
    }
    receipt = seal_record({
        "schema": "embodied-live-preflight/1", "status": "ready", "selection": [item],
        "runtime": {"eb_navigation": runtime}, "live_config_sha256": public_hash(config),
        "evidence_config_sha256": preparer.config_sha256,
        "live_requests": 0, "official_simulator_launches": 0,
        "isolation": {"backend": "linux-bwrap-seccomp/1", "namespaces_verified": True,
                      "syscall_allowlist": True, "inherited_fds_closed": True},
    })
    monkeypatch.setattr(live, "load_config", lambda *a: config)
    monkeypatch.setattr(live, "preflight", lambda *a: (receipt, preparer))
    monkeypatch.setenv("DASHSCOPE_API_KEY", "unit-test-value")
    monkeypatch.setattr(live, "worker_command", lambda *a: [])
    monkeypatch.setattr(live, "NDJSONTransport", lambda *a, **k:
                        LocalTransport(MutationSession(FakeAdapter("eb_navigation"))))
    import action_identity
    real_identity = action_identity.configuration_identity
    def linux_identity(*a):
        result = real_identity(*a)
        result["runtime"]["platform"] = "Linux"
        return seal_record(result)
    monkeypatch.setattr(live, "configuration_identity", linux_identity)
    calls = []
    class BrokenClient(FakeClient):
        def chat(self, *a, **k):
            calls.append(True)
            raise OperationalError("private-service-token")
    monkeypatch.setattr(live, "make_qwen_client", lambda: BrokenClient("Direct"))
    output = tmp_path / "result.json"
    result = live.live_run(tmp_path / "unused.json", output)
    assert result["audit"]["status"] == "passed", result["audit"]
    assert calls == [True] and len(result["attempts"]) == 3
    assert result["live_requests"] == 1
    assert result["attempts"][0]["terminal_reason"] == "operational_failure"
    assert all(r["terminal_reason"] == "not_started_after_operational_stop"
               for r in result["attempts"][1:])
    assert b"private-service-token" not in canonical(result)
    assert b"unit-test-value" not in output.read_bytes()
    assert len(list((tmp_path / "result-episodes").glob("attempt-*.json"))) == 3


def test_config_rejects_nonloopback_before_auth(tmp_path, monkeypatch):
    config = {
        "schema": live.SCHEMA,
        "selection": [{"environment": "eb_navigation", "subset": "base", "index": 0}],
        "arms": list(ARMS),
        "integration_step_limit": 1,
        "require_identical_initial_observation": False,
        "navigation_python": sys.executable,
        "household_python": sys.executable,
        "render_gpu": 0,
        "perception_gpu": 1,
        "detector": {
            "endpoint": "http://127.0.0.1:9999",
            "token_env": "HARNESS3D_PERCEPTION_TOKEN",
            "model_sha256": "0" * 64,
            "backend_config_sha256": "0" * 64,
            "deployment_receipt": str(tmp_path / "unused-receipt.json"),
        },
        "reviewer": {"endpoint": "http://127.0.0.1:9998",
                     "expected_identity_sha256": "0" * 64},
        "sam2": {"checkpoint": str(tmp_path / "unused-sam2"),
                 "checkpoint_sha256": "0" * 64, "config": "synthetic"},
        "moge2": {"checkpoint": str(tmp_path / "unused-moge2"),
                  "checkpoint_sha256": "0" * 64},
    }
    config["detector"]["endpoint"] = "https://example.com"
    monkeypatch.setattr(live, "ROOT", tmp_path)
    path = tmp_path / "config.json"
    path.write_bytes(canonical(config))
    with pytest.raises(ContractError, match="loopback"):
        live.load_config(path)


def test_guard_diagnostics_are_finite_codes_without_program_text():
    from action_solver import program_guard_codes
    source = "from action_contract import ActionPayload\ndef solve(ctx):\n    x = ctx.observation"
    codes = program_guard_codes(source)
    assert codes == ["context_field", "forbidden_import", "reserved_assignment"]
    assert program_guard_codes(
        "def solve(ctx):\n    SubmitActions(ActionPayload(ctx.decision_id, ctx.observation_revision,"
        " ctx.snapshot_sha256, ctx.action_table_sha256, [0], []))") == []
