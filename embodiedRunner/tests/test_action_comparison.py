"""Exercise the new two-arm path and lossless public trajectory offline."""
import json
import pytest

import action_live as live
from action_adapters import FakeAdapter
from action_audit import seal_record
from action_bridge import LocalTransport
from action_contract import public_hash
from action_identity import source_identity
from action_run import FakeClient, FakeDepth
from action_trajectory import restore
from evidence import CurrentEvidencePreparer, PerceptionBackends
from mutation_protocol import MutationSession
from protocol import sha256


@pytest.mark.parametrize("environment", ["eb_navigation", "eb_manipulation"])
@pytest.mark.parametrize("arms", [["Direct", "Full"], ["Full"]])
def test_two_complete_arms_and_lossless_trajectory(tmp_path, monkeypatch, environment, arms):
    monkeypatch.setattr(live, "ROOT", tmp_path)
    config = {"arms": arms, "integration_step_limit": None,
              "require_identical_initial_observation": False}
    item = {"environment": environment, "subset": "base", "index": 0,
            "episode_id": environment + ":base:000", "source_sha256": public_hash({"task": 0})}
    preparer = CurrentEvidencePreparer(PerceptionBackends(depth_estimator=FakeDepth()))
    sources = source_identity()["files_sha256"]
    receipt = seal_record({
        "schema": "embodied-live-preflight/1", "status": "ready", "selection": [item],
        "runtime": {environment: {
            "environment": environment, "runtime_ready": True,
            "model_requests": 0, "simulator_launches": 0,
            "source_sha256": {name: sources["embodiedRunner/" + name] for name in
                              ("action_worker_v2.py", "action_adapters.py",
                               "mutation_protocol.py", "protocol.py")}}},
        "live_config_sha256": public_hash(config), "evidence_config_sha256": preparer.config_sha256,
        "live_requests": 0, "official_simulator_launches": 0,
        "isolation": {"backend": "linux-bwrap-seccomp/1", "namespaces_verified": True,
                      "syscall_allowlist": True, "inherited_fds_closed": True},
    })
    monkeypatch.setattr(live, "load_config", lambda *a: config)
    monkeypatch.setattr(live, "preflight", lambda *a: (receipt, preparer))
    monkeypatch.setenv("DASHSCOPE_API_KEY", "unit-test-secret-value")
    monkeypatch.setattr(live, "worker_command", lambda *a: [])
    monkeypatch.setattr(live, "NDJSONTransport", lambda *a, **k:
                        LocalTransport(MutationSession(FakeAdapter(environment))))
    real_identity = live.configuration_identity
    def linux_identity(*a):
        value = real_identity(*a)
        value["runtime"]["platform"] = "Linux"
        return seal_record(value)
    monkeypatch.setattr(live, "configuration_identity", linux_identity)
    class ReplayClient(FakeClient):
        offline_fake = False
        def chat(self, messages, **kwargs):
            self.arm = "Full" if messages[0]["role"] == "system" and (
                "program_source" in messages[0]["content"]) else "Direct"
            text = super().chat(messages, **kwargs)
            self.last_response_identity = {"model": "qwen3.8-flash"}
            return text
    monkeypatch.setattr(live, "make_qwen_client", lambda: ReplayClient("Direct"))
    result = live.live_run(tmp_path / "unused.json", tmp_path / "compare.json",
                           compare=arms != ["Full"])
    assert result["audit"]["status"] == "passed", result["audit"]
    assert [r["arm"] for r in result["attempts"]] == arms
    if arms == ["Full"]:
        assert result["audit"]["paired_comparison_eligible"] is False
    assert all(r["terminal_reason"] == "official_done" for r in result["attempts"])
    from audit_action_trajectory import audit
    assert audit(tmp_path / "compare.json")["status"] == "passed"
    bundle = json.loads((tmp_path / "compare-episodes/trajectories.json").read_text())
    for episode, report in zip(bundle["episodes"], result["attempts"]):
        directory = tmp_path / "compare-episodes" / episode["directory"]
        events = [restore(e, directory) for e in episode["trajectory"]["events"]]
        requests = [e for e in events if e["kind"] == "model_request"]
        responses = [e for e in events if e["kind"] == "model_response"]
        observations = [e for e in events if e["kind"] == "observation"]
        assert len(requests) == len(responses) == len(report["model_requests"])
        assert len(observations) == len(report["observations"])
        for request, response, receipt in zip(requests, responses, report["model_requests"]):
            assert public_hash(request["messages"]) == receipt["messages_sha256"]
            assert sha256(response["text"].encode()) == receipt["response_sha256"]
        for observation, receipt in zip(observations, report["observations"]):
            assert sha256(observation["png"]) == receipt["rgb_sha256"]
        if report["arm"] == "Full":
            assert any(e["kind"] == "prepared_evidence" for e in events)
            executions = [e for e in events if e["kind"] == "execution"]
            assert executions and executions[0]["receipt"]["receipt"]["execution_trace"][
                "tool_calls"][0]["tool"] == "point_depth"
            assert any(e["kind"] == "perception_response" for e in events)
        else:
            assert not any(e["kind"] in {"prepared_evidence", "execution"} for e in events)
        assert events[-1]["kind"] == "episode_result"
        assert b"unit-test-secret-value" not in (directory / "trajectory.json").read_bytes()
