"""Identical analysis inputs, ordering, budget and fail-closed bounded dispatch."""
import json
import threading
import time
from pathlib import Path

import pytest

from skill3d.evolution import trajectory_analysis_v11 as analysis
from skill3d.evolution.fitset_revision_v11 import RevisionOperationalError
from skill3d.evolution.revision_edits_v11 import RevisionBudget, digest
from skill3d.storage.framework_state_io import FrameworkStateIO
from skill3d.synthesis.request_governor import (
    GovernedClient, RequestGovernor, RequestGovernorConfig,
)
from test_revision_edits_v11 import Client, parent


def response(messages):
    prompt = messages[-1]["content"]
    batch_hash = prompt.split("batch_sha256: ")[1].splitlines()[0]
    return json.dumps({
        "protocol": analysis.ANALYSIS_PROTOCOL, "batch_sha256": batch_hash,
        "records": [{"case_index": 0, **{name: "Supported process observation." for name in (
            "observed_facts", "failure_hypothesis", "supported_skill_change", "uncertainty")}}],
    })


def make_analyst(root, concurrency, factory):
    return analysis.TrajectoryAnalyst(
        output_root=root, client=Client(response), client_factory=factory,
        analysis_concurrency=concurrency,
        budget=RevisionBudget(max_requests=8, repair_reserve=0),
        method_context_max_chars=12000, expected_model="test-model")


def execute(analyst):
    return analyst.analyze(
        parent=parent(), feedback={}, batches=["first", "second", "third", "fourth"],
        cases=[], forbidden_ids=set())


def test_parallel_has_same_prompts_ordinals_replay_and_order_with_shared_cap(
        tmp_path, monkeypatch):
    monkeypatch.setattr(analysis, "batch_case_rounds", lambda batch: {0: {0}})
    prompts, completed, instances, counts = [], [], [], {}
    lock = threading.Lock()
    second_finished = threading.Event()
    governor = RequestGovernor(tmp_path / "requests", RequestGovernorConfig(
        max_requests=8, max_inflight=2, requests_per_second=0))

    def factory():
        def respond(messages):
            text = messages[-1]["content"]
            batch = text.split("Complete planned learning evidence batch:\n")[1].splitlines()[0]
            repair = "Host schema feedback:" in text
            with lock:
                prompts.append((batch, repair, digest(text)))
                counts[batch] = counts.get(batch, 0) + 1
                attempt = counts[batch]
            if batch == "first" and not repair:
                assert second_finished.wait(10)
            time.sleep(0.03)
            with lock:
                completed.append((batch, repair))
            if batch == "second" and not repair:
                second_finished.set()
            if batch == "first" and attempt == 1:
                return "{}"
            return response(messages)
        client = Client(respond)
        instances.append(client)
        return GovernedClient(client, governor, category="reviser", context={})

    parallel = make_analyst(tmp_path / "parallel", 4, factory)
    result = execute(parallel)
    assert completed.index(("second", False)) < completed.index(("first", False))
    assert completed[-1] == ("first", True)
    assert len(instances) == 5 and all(client.calls == 1 for client in instances)
    assert governor.summary()["peak_inflight"] == 2
    assert governor.summary()["completed_requests"] == parallel.requests == 5
    intents = [json.loads(p.read_text()) for p in sorted(parallel.root.glob("request_intent_*.json"))]
    assert [row["batch_index"] for row in intents] == [0, 1, 2, 3, 0]
    assert [row["ordinal"] for row in intents] == [1, 2, 3, 4, 5]

    serial_prompts, serial_counts = [], {}
    def serial_response(messages):
        text = messages[-1]["content"]
        batch = text.split("Complete planned learning evidence batch:\n")[1].splitlines()[0]
        serial_prompts.append((batch, "Host schema feedback:" in text, digest(text)))
        serial_counts[batch] = serial_counts.get(batch, 0) + 1
        return "{}" if batch == "first" and serial_counts[batch] == 1 else response(messages)
    serial = make_analyst(tmp_path / "serial", 1, None)
    serial.client = Client(serial_response)
    assert execute(serial) == result
    assert sorted(serial_prompts) == sorted(prompts)
    assert json.loads((serial.root / "analysis_final.json").read_text()) == json.loads(
        (parallel.root / "analysis_final.json").read_text())


def test_parallel_service_failure_stops_dispatch_and_drains_inflight(tmp_path, monkeypatch):
    monkeypatch.setattr(analysis, "batch_case_rounds", lambda batch: {0: {0}})
    peer_started, failure_seen = threading.Event(), threading.Event()
    calls, finished = [], []
    def factory():
        def respond(messages):
            batch = messages[-1]["content"].split(
                "Complete planned learning evidence batch:\n")[1].splitlines()[0]
            calls.append(batch)
            if batch == "first":
                assert peer_started.wait(10)
                failure_seen.set()
                raise RuntimeError("private provider failure")
            peer_started.set()
            assert failure_seen.wait(10)
            time.sleep(0.05)
            finished.append(batch)
            return response(messages)
        return Client(respond)
    analyst = make_analyst(tmp_path, 2, factory)
    with pytest.raises(RevisionOperationalError, match="service failed"):
        execute(analyst)
    assert set(calls) == {"first", "second"} and finished == ["second"]
    assert analyst.requests == 2
    assert not (tmp_path / "analysis_final.json").exists()
    assert len(list(tmp_path.glob("request_intent_*.json"))) == 2
    assert "private provider" not in "".join(p.read_text() for p in tmp_path.glob("*.json"))


def test_parallel_actors_use_local_storage_adapters_and_reject_client_reuse(tmp_path, monkeypatch):
    monkeypatch.setattr(analysis, "batch_case_rounds", lambda batch: {0: {0}})
    original_writer = analysis._write_create_once
    written = []
    class Storage:
        adapt_analysis_reviser = FrameworkStateIO.adapt_analysis_reviser
        def validate_root(self, path):
            assert Path(path).is_relative_to(tmp_path)
        def write_json_once(self, path, value):
            written.append(Path(path).name)
            original_writer(path, value)
        def write_text_once(self, path, value):
            raise AssertionError("analysis never writes text")
    analyst = make_analyst(tmp_path / "valid", 2, lambda: Client(response))
    Storage().adapt_analysis_reviser(analyst)
    assert len(execute(analyst)) == 4
    assert len([name for name in written if name.startswith("request_intent")]) == 4
    assert "analysis_final.json" in written
    assert analysis._write_create_once is original_writer
    reused = Client(response)
    bad = make_analyst(tmp_path / "bad", 2, lambda: reused)
    with pytest.raises(RevisionOperationalError, match="reused a mutable client"):
        execute(bad)
    assert reused.calls <= 1
    assert not (bad.root / "analysis_final.json").exists()


def test_parallel_budget_refusal_happens_before_any_request(tmp_path, monkeypatch):
    monkeypatch.setattr(analysis, "batch_case_rounds", lambda batch: {0: {0}})
    clients = []
    def factory():
        clients.append(Client(response))
        return clients[-1]
    analyst = make_analyst(tmp_path, 4, factory)
    analyst.budget = RevisionBudget(max_requests=7, repair_reserve=0)
    with pytest.raises(RevisionOperationalError, match="complete delivery and replay reserve"):
        execute(analyst)
    assert clients == [] and analyst.requests == 0
