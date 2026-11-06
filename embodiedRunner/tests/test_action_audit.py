"""Meaningful corruption/failure tests for frozen paired offline runs."""
from copy import deepcopy

import pytest

from action_adapters import FakeAdapter
from action_audit import audit_run, seal_record
from action_bridge import LocalTransport
from action_run import FakeClient, fake_run
from mutation_protocol import MutationSession
from protocol import OperationalError, canonical, sha256


@pytest.fixture(scope="module")
def completed():
    result = fake_run(environments=("eb_navigation",), run_id="p4-test")
    assert result["audit"]["status"] == "passed", result["audit"]
    return result


def reseal(result):
    for report in result["attempts"]:
        seal_record(report)
    seal_record(result)
    return result


def test_complete_fixed_pair_and_current_source_validation(completed):
    audit = audit_run(completed)
    assert audit["status"] == "passed" and audit["pairs"] == 1 and audit["attempts"] == 3
    assert all(m["selected"] == m["attempted"] == 1 for m in audit["metrics"].values())
    assert audit["source_reverified"] is True


@pytest.mark.parametrize("corruption", (
    "missing", "duplicate", "source", "model", "method", "delivery", "request",
    "revocation", "stale_observation", "transition", "unknown_ack", "direct_tools",
    "isolation", "execution_order", "budget", "usage", "reported_model",
))
def test_reject_semantic_corruption_after_rehash(completed, corruption):
    result = deepcopy(completed)
    direct, b01, b11 = result["attempts"]
    if corruption == "missing":
        result["attempts"].pop()
    elif corruption == "duplicate":
        result["attempts"].append(deepcopy(b11))
    elif corruption == "source":
        b01["source_sha256"] = sha256(b"other-source")
    elif corruption == "model":
        b01["model_requests"][0]["model_config_sha256"] = sha256(b"other-config")
    elif corruption == "method":
        b11["decisions"][0]["method_sha256"] = sha256(b"other-skill")
    elif corruption == "delivery":
        b11["model_requests"][0]["delivery"]["entries"][0]["chars"] -= 1
    elif corruption == "request":
        b11["model_requests"].pop()
    elif corruption == "revocation":
        b01["revoked_decisions"].pop()
    elif corruption == "stale_observation":
        b01["decisions"][1]["observation_revision"] = 0
    elif corruption == "transition":
        b01["transitions"][1]["previous_sha256"] = sha256(b"other-chain")
    elif corruption == "unknown_ack":
        b01["mutations"][1]["status"] = "unknown"
    elif corruption == "direct_tools":
        direct["budget"]["reserved"]["tools"] = 1
    elif corruption == "isolation":
        b01["executions"][0]["receipt"]["isolation_sealed"] = False
    elif corruption == "execution_order":
        b01["executions"].reverse()
    elif corruption == "budget":
        b01["budget"]["reserved"]["qwen"] += 1
    elif corruption == "usage":
        b01["model_requests"][0]["usage"]["total_tokens"] += 1
    elif corruption == "reported_model":
        b01["model_requests"][0]["reported_model_sha256"] = sha256(b"wrong-model")
    assert audit_run(reseal(result))["status"] == "rejected"


def test_raw_report_tampering_fails_hash(completed):
    result = deepcopy(completed)
    result["attempts"][0]["task_success"] = False
    assert audit_run(result)["errors"] == ["run_hash"]


def test_missing_or_changed_implementation_file_identity(completed):
    result = deepcopy(completed)
    source = result["manifest"]["implementation_source"]
    source["files_sha256"].pop(next(iter(source["files_sha256"])))
    from action_contract import public_hash
    source["sha256"] = public_hash(source["files_sha256"])
    seal_record(result["manifest"])
    for report in result["attempts"]:
        report["implementation_source_sha256"] = source["sha256"]
        report["paired_manifest_sha256"] = result["manifest"]["sha256"]
    assert audit_run(reseal(result))["errors"] == ["source_changed_after_freeze"]


def test_endpoint_failure_keeps_entire_denominator():
    class BrokenHandshake:
        def handshake(self):
            raise OperationalError("private-provider-url-token")
        def close(self):
            pass
    result = fake_run(environments=("eb_navigation",), transport_factory=lambda e: BrokenHandshake())
    assert result["status"] == "failed" and result["audit"]["status"] == "passed", result["audit"]
    assert len(result["attempts"]) == 3
    assert all(m["selected"] == m["attempted"] == m["aborted"] == 1
               and m["success_rate"] == 0 for m in result["audit"]["metrics"].values())
    assert b"private-provider" not in canonical(result)


def test_model_service_failure_is_once_and_fixed_denominator():
    class BrokenClient(FakeClient):
        def chat(self, *args, **kwargs):
            self.calls += 1
            raise RuntimeError("private-provider-url-token")
    result = fake_run(environments=("eb_navigation",), client_factory=BrokenClient)
    assert result["audit"]["status"] == "passed", result["audit"]
    for report in result["attempts"]:
        assert report["terminal_reason"] == "operational_failure"
        assert report["budget"]["reserved"]["qwen"] == 1
        assert len(report["model_requests"]) == 1
    assert b"private-provider" not in canonical(result)


def test_official_failed_step_invalid_counter_is_separate_from_format_mutations():
    class OfficialFailure(FakeAdapter):
        def step(self, action, first_in_plan):
            self.invalid_count += 1
            return super().step(action, first_in_plan)
    result = fake_run(environments=("eb_alfred",), transport_factory=lambda e:
                      LocalTransport(MutationSession(OfficialFailure(e, failed_steps=(1, 2)))))
    assert result["audit"]["status"] == "passed", result["audit"]
    for row in result["attempts"]:
        assert row["environment_steps"] == row["invalid_actions"] == 2
        assert row["budget"]["reserved"].get("invalid", 0) == 0
        assert [o["official_invalid_actions"] for o in row["observations"]] == [0, 1, 2]


def test_preparation_failure_keeps_snapshot_revocation_and_request_charge():
    from action_run import FakeDepth
    from evidence import CurrentEvidencePreparer, PerceptionBackends
    class BrokenDepth(FakeDepth):
        def __call__(self, rgb):
            raise RuntimeError("private-perception-url-token")
    result = fake_run(environments=("eb_navigation",), preparer_factory=lambda e:
                      CurrentEvidencePreparer(PerceptionBackends(depth_estimator=BrokenDepth())))
    assert result["audit"]["status"] == "passed", result["audit"]
    for report in result["attempts"][1:]:
        assert report["status"] == "aborted" and len(report["decisions"]) == 1
        assert report["decisions"][0]["evidence_status"] == "preparing"
        assert len(report["revoked_decisions"]) == 1
        assert not report["model_requests"] and report["budget"]["reserved"]["moge"] == 1
    assert b"private-perception" not in canonical(result)


def test_unknown_action_ack_not_retried_and_kept_as_failure():
    class DropACK(LocalTransport):
        def exchange(self, command, timeout_s):
            result = super().exchange(command, timeout_s)
            if command["op"] == "step":
                raise OperationalError("private-wire")
            return result
    result = fake_run(environments=("eb_navigation",), transport_factory=lambda e:
                      DropACK(MutationSession(FakeAdapter(e))))
    assert result["audit"]["status"] == "passed", result["audit"]
    assert all(r["terminal_reason"] == "action_outcome_unknown"
               and len(r["mutations"]) == 2 and r["environment_steps"] == 0
               and r["budget"]["reserved"]["steps"] == 1 for r in result["attempts"])


@pytest.mark.parametrize("fault", ("model", "usage"))
def test_real_service_deviation_stops_without_retry_and_audit_rejects(fault):
    class DeviatingClient(FakeClient):
        def chat(self, *args, **kwargs):
            result = super().chat(*args, **kwargs)
            if fault == "model":
                self.last_response_identity = {"model": "different-private-model"}
            else:
                self.last_usage = {"prompt_tokens": 8, "completion_tokens": 8, "total_tokens": 999}
            return result
    result = fake_run(environments=("eb_navigation",), client_factory=DeviatingClient)
    assert result["audit"]["status"] == "rejected"
    assert all(len(r["model_requests"]) == 1 and r["environment_steps"] == 0
               and r["status"] == "aborted" for r in result["attempts"])
    assert b"different-private-model" not in canonical(result)


def test_format_repair_retains_usage_and_complete_skill():
    class FirstBad(FakeClient):
        def chat(self, *args, **kwargs):
            result = super().chat(*args, **kwargs)
            return "{}" if self.calls == 1 else result
    result = fake_run(environments=("eb_navigation",), client_factory=FirstBad)
    assert result["audit"]["status"] == "passed", result["audit"]
    for report in result["attempts"]:
        assert report["budget"]["qwen_tokens_observed"] == 48
        assert report["budget"]["requests"][-3 if report["arm"] == "Direct" else -4]["usage"]
        assert report["model_requests"][0]["status"] == "format_error"


@pytest.mark.parametrize("value", ({}, {"schema": "wrong"}, {"schema": "embodied-offline-run/1"}))
def test_malformed_records_rejected_without_exception(value):
    assert audit_run(value)["status"] == "rejected"
