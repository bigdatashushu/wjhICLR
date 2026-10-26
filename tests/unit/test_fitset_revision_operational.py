"""Real revision/buffer/governor recovery with synthetic local fault injection."""

from dataclasses import dataclass, replace
import importlib.util
import json
import multiprocessing
import os
from pathlib import Path
import sys
import time
import traceback

import pytest

from skill3d.evaluation.fitset_v11 import FitSetError, _write_create_once
from skill3d.evolution.fitset_revision_v11 import FitSetSlotReviser
from skill3d.skills.registry import load_active_skills
from skill3d.synthesis.request_governor import (
    RequestBudgetExhausted, RequestGovernor, RequestGovernorConfig,
)
from test_fitset_stream_v11 import make_runner


SECRET = "SYNTHETIC-SECRET-NOT-A-CREDENTIAL"
SECRET_URL = f"https://synthetic.invalid/?key={SECRET}"


@dataclass
class FaultClient:
    calls_root: Path
    mode: str

    def chat(self, _messages, **_kwargs):
        _write_create_once(self.calls_root / f"{time.time_ns()}.json", {"called": True})
        if self.mode == "hard_stop":
            os._exit(29)
        if self.mode == "service_error":
            raise RuntimeError(SECRET) from ValueError(SECRET_URL)
        if self.mode == "invalid_method":
            return json.dumps({
                "action": "revise", "full_skill_md": "not a valid complete Skill",
                "modification_reason": "Synthetic invalid candidate.",
                "change_hypothesis": "Synthetic change.",
                "changed_sections": ["__body__"], "expected_process_effect": "Synthetic effect.",
                "evidence_signature_refs": [],
            })
        return json.dumps({"action": "no_change", "modification_reason": "Synthetic no change."})


@dataclass
class FaultFactory:
    calls_root: Path
    mode: str

    def __call__(self):
        return FaultClient(self.calls_root, self.mode)


def batch_runner(tmp_path, mode):
    runner = make_runner(
        tmp_path, n=12, revision_client_factory=FaultFactory(tmp_path / "calls", mode))
    runner.evolution_config = replace(
        runner.evolution_config, max_revision_attempts=1, max_revision_requests_per_slot=1)
    runner.request_governor_config = RequestGovernorConfig(max_requests=1, max_inflight=1)
    return runner


def complete_batch(runner):
    runner.run(resume=True)


def campaign(runner):
    roots = list((runner.root / "workers/00/campaigns").glob("*"))
    assert len(roots) == 1
    return roots[0]


def assert_pending(runner):
    root = campaign(runner)
    assert not (root / "result.json").exists()
    assert not (root / "decision.json").exists()
    assert not list(root.glob("revision/slots/*/*/final.json"))
    state = json.loads((runner.root / "workers/00/buffer/stream.json").read_text())
    slot = state["slots"]["object_counting/g0m0"]
    assert slot["new_since_attempt"] == 12
    assert slot["attempt_count"] == 0 and not slot["attempted_ever"]
    assert not slot["deferred_reason"]
    assert len(slot["case_refs"]) == 12 and state["attempts"] == []
    assert len(list(root.glob("revision/slots/*/*/request_intent_*.json"))) == 1
    assert len(list((runner.root / "workers/00/buffer/attempts").glob("*.json"))) == 0
    assert len(list(runner.root.parent.glob("calls/*.json"))) == 1


def assert_operational(exc):
    assert type(exc).__name__ == "RevisionOperationalError"
    assert not isinstance(exc, FitSetError)


def test_hard_stop_unknown_request_resume_preserves_batch_and_reserved_budget(tmp_path):
    runner = batch_runner(tmp_path, "hard_stop")
    assert runner.run(resume=False, max_variants=11)["completed_variants"] == 11
    child = multiprocessing.get_context("spawn").Process(target=complete_batch, args=(runner,))
    child.start()
    try:
        child.join(timeout=40)
        assert not child.is_alive(), "synthetic fault child did not finish"
        assert child.exitcode == 29
    finally:
        if child.is_alive():
            child.kill()
        child.join()
    assert_pending(runner)
    assert not list(campaign(runner).glob("revision/slots/*/*/attempt_*.json"))
    governor = RequestGovernor(runner.request_governor_root, runner.request_governor_config)
    assert governor.summary()["unknown_requests"] == 1
    before = {path: path.read_bytes()
              for root in (campaign(runner), runner.root / "workers/00/buffer",
                           runner.request_governor_root)
              for path in root.rglob("*.json")}
    for _ in range(2):
        with pytest.raises(RuntimeError) as caught:
            batch_runner(tmp_path, "hard_stop").run(resume=True)
        assert_operational(caught.value)
        assert_pending(runner)
    assert {path: path.read_bytes() for path in before} == before
    assert governor.summary()["reserved_requests"] == 1
    assert governor.summary()["unknown_requests"] == 1
    with pytest.raises(RequestBudgetExhausted):
        with governor.request("reviser", {}):
            pytest.fail("an unknown request must still occupy the only budget slot")


def test_fresh_and_recorded_service_failure_stop_without_rejected_terminal(tmp_path):
    runner = batch_runner(tmp_path, "service_error")
    with pytest.raises(RuntimeError) as caught:
        runner.run(resume=False)
    assert_operational(caught.value)
    assert_pending(runner)
    exchange = next(campaign(runner).glob("revision/slots/*/*/attempt_*.json"))
    body = json.loads(exchange.read_text())
    assert body["service_error"] is True and body["exception_type"] == "RuntimeError"
    assert SECRET not in exchange.read_text() and SECRET_URL not in exchange.read_text()
    before = exchange.read_bytes()
    for _ in range(2):
        with pytest.raises(RuntimeError) as caught:
            batch_runner(tmp_path, "service_error").run(resume=True)
        assert_operational(caught.value)
        assert_pending(runner)
    assert exchange.read_bytes() == before
    summary = RequestGovernor(
        runner.request_governor_root, runner.request_governor_config).summary()
    assert summary["reserved_requests"] == summary["error_requests"] == 1


@pytest.mark.parametrize("mode,outcome", [
    ("invalid_method", "rejected"), ("no_change", "no_change"),
])
def test_completed_method_decisions_remain_terminal_and_do_not_replay(tmp_path, mode, outcome):
    runner = batch_runner(tmp_path, mode)
    assert runner.run(resume=False)["status"] == "complete"
    root = campaign(runner)
    result = json.loads((root / "result.json").read_text())
    assert result["outcome"] == outcome
    state = json.loads((runner.root / "workers/00/buffer/stream.json").read_text())
    slot = state["slots"]["object_counting/g0m0"]
    assert slot["new_since_attempt"] == 0 and slot["attempt_count"] == 1
    assert len(slot["case_refs"]) == 12 and len(state["attempts"]) == 1
    assert batch_runner(tmp_path, mode).run(resume=True)["status"] == "complete"
    assert len(list(tmp_path.glob("calls/*.json"))) == 1


def test_provider_exception_cause_and_traceback_do_not_expose_synthetic_secret(tmp_path):
    reviser = FitSetSlotReviser(
        output_root=tmp_path / "revision", panel={}, panel_name="synthetic", variants=[],
        accumulation_root=tmp_path, label_metadata={}, library_root=tmp_path,
        client=FaultClient(tmp_path / "calls", "service_error"), max_attempts=1,
    )
    with pytest.raises(RuntimeError) as caught:
        reviser._request(slot=("object_counting", "g0m0"), attempt=1, prompt="synthetic")
    assert_operational(caught.value)
    assert caught.value.__cause__ is None and caught.value.__suppress_context__
    formatted = "".join(traceback.format_exception(caught.value))
    assert SECRET not in formatted and SECRET_URL not in formatted
    for path in reviser.root.rglob("*.json"):
        assert SECRET not in path.read_text() and SECRET_URL not in path.read_text()


@pytest.mark.parametrize("mode,reason", [
    ("fresh", "revision request failed"),
    ("recorded", "recorded revision service failure"),
    ("unknown", "revision request has no completion receipt; ambiguous request is not retried"),
])
def test_cli_exits_safely_for_operational_failure_without_provider_text(
        tmp_path, monkeypatch, capsys, mode, reason):
    script = Path(__file__).resolve().parents[2] / "scripts/run_fitset_v11.py"
    spec = importlib.util.spec_from_file_location("fitset_boundary_cli", script)
    cli = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, cli)
    spec.loader.exec_module(cli)
    reviser = FitSetSlotReviser(
        output_root=tmp_path / "revision", panel={}, panel_name="synthetic", variants=[],
        accumulation_root=tmp_path, label_metadata={}, library_root=tmp_path,
        client=FaultClient(tmp_path / "calls", "service_error"), max_attempts=1,
    )
    if mode == "recorded":
        with pytest.raises(RuntimeError):
            reviser._request(slot=("object_counting", "g0m0"), attempt=1, prompt="synthetic")
    elif mode == "unknown":
        _write_create_once(
            reviser.root / "slots/object_counting/g0m0/request_intent_01.json",
            {"request": {}, "started_at": "synthetic"},
        )

    def handle(_args):
        reviser._request(slot=("object_counting", "g0m0"), attempt=1, prompt="synthetic")
        pytest.fail("operational failure must stop the CLI handler")

    monkeypatch.setattr(cli, "_run_system", handle)
    code = cli.main([
        "run-system", "--panel", "synthetic-panel", "--panel-name", "ready_579",
        "--metadata", "synthetic-metadata", "--phase", "accumulation",
        "--output-dir", str(tmp_path / "unused"),
    ])
    output = capsys.readouterr()
    assert code == 2 and output.out == ""
    assert output.err == f"RevisionOperationalError: object_counting/g0m0: {reason}\n"
    assert SECRET not in output.err and SECRET_URL not in output.err
    assert "Traceback" not in output.err
    assert len(list(tmp_path.glob("calls/*.json"))) == int(mode != "unknown")
    for path in reviser.root.rglob("*.json"):
        assert SECRET not in path.read_text() and SECRET_URL not in path.read_text()


def test_completed_exchange_with_changed_request_is_operational_not_method_failure(tmp_path):
    reviser = FitSetSlotReviser(
        output_root=tmp_path / "revision", panel={}, panel_name="synthetic", variants=[],
        accumulation_root=tmp_path, label_metadata={}, library_root=tmp_path,
        client=FaultClient(tmp_path / "calls", "no_change"), max_attempts=1,
    )
    reviser._request(slot=("object_counting", "g0m0"), attempt=1, prompt="synthetic original")
    with pytest.raises(RuntimeError) as caught:
        reviser._request(slot=("object_counting", "g0m0"), attempt=1, prompt="synthetic changed")
    assert_operational(caught.value)
    assert len(list(tmp_path.glob("calls/*.json"))) == 1


def test_failed_service_receipt_write_has_a_safe_exception_chain(tmp_path, monkeypatch):
    from skill3d.evolution import fitset_revision_v11 as revision
    reviser = FitSetSlotReviser(
        output_root=tmp_path / "revision", panel={}, panel_name="synthetic", variants=[],
        accumulation_root=tmp_path, label_metadata={}, library_root=tmp_path,
        client=FaultClient(tmp_path / "calls", "service_error"), max_attempts=1,
    )
    original = revision._write_create_once

    def fail_exchange(path, body):
        if path.name == "attempt_01.json":
            raise OSError(28, SECRET_URL)
        return original(path, body)

    monkeypatch.setattr(revision, "_write_create_once", fail_exchange)
    with pytest.raises(RuntimeError) as caught:
        reviser._request(slot=("object_counting", "g0m0"), attempt=1, prompt="synthetic")
    assert_operational(caught.value)
    assert caught.value.__suppress_context__ and caught.value.__cause__ is None
    formatted = "".join(traceback.format_exception(caught.value))
    assert SECRET not in formatted and SECRET_URL not in formatted
    assert not list(reviser.root.glob("slots/*/*/attempt_*.json"))
    assert len(list(reviser.root.glob("slots/*/*/request_intent_*.json"))) == 1


def test_cli_receipt_write_failure_preserves_batch_budget_and_does_not_replay(
        tmp_path, monkeypatch, capsys):
    from skill3d.evolution import fitset_revision_v11 as revision
    script = Path(__file__).resolve().parents[2] / "scripts/run_fitset_v11.py"
    spec = importlib.util.spec_from_file_location("fitset_receipt_failure_cli", script)
    cli = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, cli)
    spec.loader.exec_module(cli)
    runner = batch_runner(tmp_path, "service_error")
    original = revision._write_create_once

    def fail_exchange(path, body):
        if path.name == "attempt_01.json":
            raise OSError(28, SECRET_URL)
        return original(path, body)

    monkeypatch.setattr(revision, "_write_create_once", fail_exchange)
    for resume, reason in [
        (False, "revision failure receipt unavailable"),
        (True, "revision request has no completion receipt; ambiguous request is not retried"),
    ]:
        monkeypatch.setattr(cli, "_run_system", lambda _args: runner.run(resume=resume))
        code = cli.main([
            "run-system", "--panel", "synthetic-panel", "--panel-name", "ready_579",
            "--metadata", "synthetic-metadata", "--phase", "accumulation",
            "--output-dir", str(tmp_path / "unused"),
        ])
        output = capsys.readouterr()
        assert code == 2 and output.out == ""
        assert output.err == f"RevisionOperationalError: object_counting/g0m0: {reason}\n"
        assert SECRET not in output.err and SECRET_URL not in output.err
        assert "Traceback" not in output.err
        assert_pending(runner)
        assert not list(campaign(runner).glob("revision/slots/*/*/attempt_*.json"))
        summary = RequestGovernor(
            runner.request_governor_root, runner.request_governor_config).summary()
        assert summary["reserved_requests"] == summary["error_requests"] == 1


@pytest.mark.parametrize("mode", ["invalid_method", "no_change"])
def test_method_validation_and_corrupt_frozen_revision_have_distinct_types(tmp_path, mode):
    library = Path(__file__).resolve().parents[2] / "skill_library"
    parent = next(skill for skill in load_active_skills(library / "snapshots")[0]
                  if skill.slot_key == "object_counting/g0m0")
    reviser = FitSetSlotReviser(
        output_root=tmp_path / "revision", panel={}, panel_name="synthetic", variants=[],
        accumulation_root=tmp_path, label_metadata={}, library_root=library,
        client=FaultClient(tmp_path / "calls", mode), max_attempts=1,
    )
    kwargs = {
        "slot": ("object_counting", "g0m0"), "parent": parent,
        "feedback": {"signatures": []}, "forbidden_ids": set(),
    }
    if mode == "invalid_method":
        with pytest.raises(FitSetError) as caught:
            reviser._revise_slot(**kwargs)
        assert type(caught.value) is FitSetError
    else:
        assert reviser._revise_slot(**kwargs) is None
        final = next(reviser.root.glob("slots/*/*/final.json"))
        receipt = json.loads(final.read_text())
        receipt["identity_sha256"] = "0" * 64
        final.write_text(json.dumps(receipt))
        with pytest.raises(RuntimeError) as caught:
            reviser._revise_slot(**kwargs)
        assert_operational(caught.value)
    assert len(list(tmp_path.glob("calls/*.json"))) == 1
