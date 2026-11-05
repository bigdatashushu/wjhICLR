"""Offline regression for fixed-question host batches and complete accounting."""
from copy import deepcopy
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]

from Ominirunner import budget, replay
from Ominirunner.audit import verify_charges
from Ominirunner.common import Config, digest, read, sha, write
from Ominirunner.jev import Policy
from Ominirunner.reuse import request_cap
from Ominirunner.test_contracts import OfflineClient, offline_only, scene_data


@pytest.mark.parametrize("review_cap,expected_status", [(8, "answered"), (1, "service_error")])
def test_production_worker_batch_uses_intern_and_real_dual_governors(
        tmp_path, monkeypatch, scene_data, review_cap, expected_status):
    import types
    from Ominirunner import jev, worker
    from Ominirunner.audit import replay_contexts
    from Ominirunner.common import VERSION
    from Ominirunner.data import pair_identity
    from Ominirunner.protocol import messages
    from Ominirunner.test_jev import InternOpener, SAMPLE, ratio
    opener = InternOpener()
    policy = Policy(provider="intern", max_requests=review_cap,
                    service_identity_sha256=opener.identity_sha, yes_threshold=.5).public()
    rgb, frame, cache = scene_data[1:4]
    monkeypatch.setattr(worker, "load", lambda *a, **k: (rgb, frame, cache))
    monkeypatch.setattr(worker, "validate_source", lambda source: None)
    monkeypatch.setattr(worker, "sha", lambda path: worker.AUTH_SHA)
    monkeypatch.setitem(sys.modules, "transport_auth", types.SimpleNamespace(
        load_transport_token=lambda: pytest.fail("no real proxy transport")))
    original_client = jev.Client
    monkeypatch.setattr(jev, "Client", lambda policy, governor, **k:
                        original_client(policy, governor, opener=opener))
    clients = []
    def solver(**kwargs):
        client = OfflineClient([SAMPLE, ratio])
        clients.append(client)
        return client
    monkeypatch.setattr(worker, "BailianClient", solver)
    shared_root = tmp_path / "smallimage" / "requests"
    def accounting(base):
        path = shared_root / "state.json"
        rows = read(path)["requests"] if path.exists() else []
        return {"used": len(rows), "global_cap": 20000, "mmsi_remaining_cap": 0,
                "mindcube_remaining_cap": 0, "cv3d_remaining_policy_cap": 0,
                "shared_campaign_counts": {"synthetic-host-batch": len(rows)}}
    monkeypatch.setattr(budget, "audit", accounting)
    inputs = [{**scene_data[0], "sample_id": str(i),
               "question": "What is the height ratio of the chair and desk?", "task_type": "float"}
              for i in (1, 2)]
    pairs = [{"sample_id": item["sample_id"], "pair_identity": pair_identity(item),
              "initial_receipts": {"skill": messages(
                  item, rgb, frame, cache, arm="skill", config=Config(),
                  measurement_reviewer=policy)[1]}} for item in inputs]
    cap = request_cap(dict.fromkeys(("1", "2")), ("1", "2"), policy)
    source = {name: "synthetic" for name in (
        "scripts/Ominirunner/jev.py", "scripts/Ominirunner/jev_protocol.py",
        "scripts/Ominirunner/jev_trial.py", "scripts/Ominirunner/static_skill_host.txt",
        "scripts/Ominirunner/intern_review.py")}
    frozen = {"schema": VERSION, "sample_ids": ["1", "2"], "source": source,
              "config": Config().public(), "measurement_reviewer": policy,
              "campaign": "synthetic-host-batch", "reused_direct": {"1": {}, "2": {}},
              "max_requests": cap, "pairs": pairs, "gateway": "http://127.0.0.1:1",
              "auth_module_root": str(tmp_path), "shared_budget_root": str(shared_root),
              "local_budget_policy": {"max_requests": cap, "max_inflight": 2, "requests_per_second": 0},
              "shared_budget_policy": {"max_requests": 20000, "max_inflight": 4, "requests_per_second": 0}}
    results = []
    for item, pair in zip(inputs, pairs):
        directory = tmp_path / ("episode-" + item["sample_id"])
        directory.mkdir()
        job = {"schema": worker.SCHEMA, "item": item, "arm": "skill",
               "output_directory": str(directory), "budget_root": str(tmp_path / "requests"),
               "stop_file": str(tmp_path / "STOP.json"), "batch_frozen": frozen,
               "initial_receipt": pair["initial_receipts"]["skill"],
               "run_identity": digest(frozen), **{k: frozen[k] for k in (
                   "source", "config", "measurement_reviewer", "campaign", "gateway",
                   "auth_module_root", "shared_budget_root", "local_budget_policy", "shared_budget_policy")}}
        result = worker.execute(job)
        results.append(result)
        assert result["status"] == expected_status
        for index, payload in replay_contexts(item, result, rgb, frame, cache):
            assert payload == read(directory / f"{index:02d}.messages.json")
        if (tmp_path / "STOP.json").exists():
            break
    shared = read(shared_root / "state.json")["requests"]
    local = read(tmp_path / "requests/state.json")["requests"]
    projection = lambda rows: [{k: r[k] for k in ("category", "context", "status", "usage")} for r in rows]
    assert projection(shared) == projection(local)
    if review_cap == 8:
        assert [r["jev_request_count"] for r in results] == [2, 2]
        assert len(shared) == 8 and sum(c.count for c in clients) == 4
        expected = {}
        for result in results:
            expected[(result["sample_id"], "skill", "solver")] = (2, result["usage"])
            expected[(result["sample_id"], "skill", "gate")] = (2, result["jev_usage"])
        assert verify_charges(local, frozen, expected) == verify_charges(shared, frozen, expected)
    else:
        assert len(results) == 1 and results[0]["jev_request_count"] == 1
        assert len(shared) == 2 and clients[0].count == 1


def test_host_cap_includes_each_question_and_replacement_budget():
    ids = tuple(str(i) for i in range(20))
    p = Policy(provider="intern", yes_threshold=.5).public()
    assert request_cap(dict.fromkeys(ids), ids, p) == 280
    assert request_cap({}, ids, p) == 300
    assert request_cap(dict.fromkeys(ids), ids) == 120


def test_admission_preserves_cv_reservation_and_does_not_add_default_omni_cap():
    state = {"used": 18958, "global_cap": 20000, "mmsi_remaining_cap": 0,
             "mindcube_remaining_cap": 0, "cv3d_remaining_policy_cap": 3538,
             "omni_requested_cap": 140}
    with pytest.raises(ValueError, match="aggregate_reserved_budget_insufficient"):
        budget.admission(state, 280)
    assert budget.admission(state, 280, budget.UNLIMITED_POLICY)["admission_upper_bound"] == 22776
    with pytest.raises(ValueError, match="aggregate_budget_policy_invalid"):
        budget.admission(state, 280, {"global_limit": None})
    state["cv3d_remaining_policy_cap"] = 0
    assert budget.admission(state, 280)["admission_upper_bound"] == 19238
    state["used"] += 40
    assert budget.admission(state, 240)["admission_upper_bound"] == 19238


def test_cv_release_requires_explicit_receipt_and_unchanged_evidence(tmp_path):
    rows = [{"status": "completed", "request_id": str(i)} for i in range(2)]
    records = {budget.CV_LEDGER: rows}
    policy = {"max_requests": 10, "max_inflight": 4, "requests_per_second": 0}
    assert budget.completed_cv_release(tmp_path, records, policy) is None
    for name in budget.CV_COMPLETIONS:
        p = tmp_path / name
        p.parent.mkdir(parents=True, exist_ok=True)
        write(p, {"status": "complete"})
    receipt = {"schema": "omni-cv3d-completed-reservation-release/1", "status": "released",
               "authorization": "explicit_user", "active_budget_slots": 0,
               "operator_lock_free": True, "original_process_absent": True, "model_requests": 0,
               "reference_answers_returned": False, "policy": policy,
               "ledger_snapshot": {"count": 2, "requests_sha256": digest(rows)},
               "released_unused_requests": 8,
               "completion_files": {name: sha(tmp_path / name) for name in budget.CV_COMPLETIONS}}
    p = tmp_path / budget.CV_RELEASE_PATH
    p.parent.mkdir(parents=True, exist_ok=True)
    write(p, receipt)
    assert budget.completed_cv_release(tmp_path, records, policy)["released_unused_requests"] == 8
    receipt["authorization"] = "inferred"
    write(p, receipt)
    with pytest.raises(ValueError, match="cv_completed_reservation_release_invalid"):
        budget.completed_cv_release(tmp_path, records, policy)
    receipt["authorization"] = "explicit_user"
    write(p, receipt)
    rows.append({"status": "started"})
    with pytest.raises(ValueError, match="cv_completed_reservation_release_invalid"):
        budget.completed_cv_release(tmp_path, records, policy)
    rows.pop()
    write(tmp_path / budget.CV_COMPLETIONS[0], {"status": "changed"})
    with pytest.raises(ValueError, match="cv_completed_reservation_evidence_changed"):
        budget.completed_cv_release(tmp_path, records, policy)


def test_external_review_snapshot_is_immutable_and_gate_only(tmp_path):
    name = "smallimage/omni_external_review/check"
    path = tmp_path / name / "state.json"
    path.parent.mkdir(parents=True)
    write(path, {"policy": {}, "requests": [{"category": "gate", "status": "completed"}]})
    receipt = {"schema": "omni-external-review-accounting/1", "model_requests": 0,
               "ledgers": {name: {"count": 1, "file_sha256": sha(path)}}}
    write(tmp_path / budget.EXTERNAL_PATH, receipt)
    assert len(budget.external_ledgers(tmp_path)[0]) == 1
    state = read(path)
    state["requests"][0]["status"] = "started"
    write(path, state)
    with pytest.raises(ValueError, match="external_review_snapshot_changed"):
        budget.external_ledgers(tmp_path)


def test_solver_and_gate_mirrors_allow_interleaving_but_reject_missing_and_reassignment():
    frozen = {"campaign": "batch"}
    records = []
    expected = {}
    for sample, category, usage in (("1", "solver", {"total_tokens": 12}),
                                    ("1", "gate", {"input_tokens": 10, "output_tokens": 1}),
                                    ("2", "gate", {"input_tokens": 13, "output_tokens": 1})):
        records.append({"category": category, "status": "completed", "usage": usage,
                        "context": {"campaign_id": "batch", "run_identity": digest(frozen),
                                    "slot_key": sample, "variant_id": "skill"}})
        expected[(sample, "skill", category)] = (1, usage)
    assert verify_charges(records, frozen, expected) == verify_charges(records[::-1], frozen, expected)
    with pytest.raises(ValueError, match="per_episode_charge_mismatch"):
        verify_charges(records[:-1], frozen, expected)
    changed = deepcopy(records)
    changed[1]["context"]["slot_key"] = "2"
    with pytest.raises(ValueError, match="per_episode_charge_mismatch"):
        verify_charges(changed, frozen, expected)


def replay_fixture(tmp_path, monkeypatch):
    inputs = [{"sample_id": "229"}, {"sample_id": "243"}]
    monkeypatch.setattr(replay, "select", lambda root, ids: inputs)
    monkeypatch.setattr(replay, "pair_identity", lambda item: item["sample_id"])
    write(tmp_path / "inputs.json", inputs)
    frozen = {"config": Config().public(), "sample_ids": ["229", "243"], "source": {},
              "inputs_sha256": sha(tmp_path / "inputs.json"),
              "pairs": [{"sample_id": i["sample_id"], "pair_identity": i["sample_id"]} for i in inputs]}
    write(tmp_path / "frozen.json", frozen)
    write(tmp_path / "source_manifest.json", {})
    for name in ("preflight.json", "audit.json", "completed.json"):
        write(tmp_path / name, {"run_identity": digest(frozen), "status":
                               "passed" if name == "audit.json" else "complete", "episodes": 4})
    return inputs


def test_replay_keeps_question_order_and_never_reads_scores(tmp_path, monkeypatch):
    inputs = replay_fixture(tmp_path, monkeypatch)
    actual, receipt = replay.inspect(tmp_path, tmp_path)
    assert actual == inputs and receipt["sample_ids"] == ["229", "243"]
    assert "scores.json" not in receipt["files"]
    assert replay.validate(receipt, tmp_path) == inputs
    inputs.reverse()
    with pytest.raises(ValueError, match="replayed_public_inputs_changed"):
        replay.validate(receipt, tmp_path)


def test_replay_rejects_changed_prior_completion(tmp_path, monkeypatch):
    replay_fixture(tmp_path, monkeypatch)
    _, receipt = replay.inspect(tmp_path, tmp_path)
    write(tmp_path / "completed.json", {"status": "stopped", "run_identity": receipt["source_run_identity"]})
    with pytest.raises(ValueError, match="replayed_completed_contract_changed"):
        replay.validate(receipt, tmp_path)
