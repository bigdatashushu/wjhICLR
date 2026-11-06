"""Three fake environments run through the real host and independent sandbox."""
from copy import deepcopy
from types import SimpleNamespace

import pytest

from action_adapters import FakeAdapter
from action_bridge import ActionBridge, LocalTransport
from action_host import ActionHost, PreparedEvidence, Proposal
from budget import RunBudget
from mutation_protocol import MutationSession
from protocol import ContractError, OperationalError, sha256

SUBMIT = """def solve(ctx):
    SubmitActions(ActionPayload(ctx.decision_id, ctx.observation_revision,
        ctx.snapshot_sha256, ctx.action_table_sha256, [0, 1]))
"""


class FakeSolver:
    def __init__(self, arm="B01", scripted=()):
        self.arm, self.scripted, self.calls = arm, list(scripted), []

    def generate(self, snapshot, *, feedback, usage):
        assert set(snapshot.public_dict()["observation"]).isdisjoint({
            "official", "done", "task_success", "task_progress", "reward", "invalid_actions"})
        self.calls.append((snapshot, deepcopy(feedback)))
        if self.scripted:
            next_value = self.scripted.pop(0)
            if isinstance(next_value, Exception):
                raise next_value
            return Proposal(source=next_value) if self.arm != "Direct" else Proposal(actions=next_value)
        return Proposal(source=SUBMIT) if self.arm != "Direct" else Proposal(actions=(0, 1))


def setup(environment="eb_navigation", *, arm="B01", solver=None, preparer=None,
          transport_type=LocalTransport, **fake_options):
    adapter = FakeAdapter(environment, **fake_options)
    budget = RunBudget("run", [environment]).episode("episode", environment)
    bridge = ActionBridge(transport_type(MutationSession(adapter)), environment=environment,
                          episode_id="episode", budget=budget)
    host = ActionHost(bridge, solver or FakeSolver(arm), run_id="run", arm=arm,
                      evidence_preparer=preparer, method_sha256=sha256(b"method") if arm == "B11" else "")
    return host, adapter


def run(host):
    return host.run(subset="base", index=0, source_sha256=sha256(b"input"))


@pytest.mark.parametrize("environment", ("eb_navigation", "eb_alfred", "eb_habitat"))
@pytest.mark.parametrize("arm", ("Direct", "B01", "B11"))
def test_single_action_new_snapshot_three_environments_and_arms(environment, arm):
    host, adapter = setup(environment, arm=arm, failed_steps=(1,), stationary_rgb=True)
    report = run(host)
    assert report["status"] == "finished" and report["terminal_reason"] == "official_done", report
    assert adapter.step_count == 2 and adapter.closed
    assert report["environment_steps"] == 2 and report["mutation_revision"] == 3
    assert len(report["decisions"]) == 2 and len(report["revoked_decisions"]) == 2
    assert report["decisions"][0]["rgb_sha256"] == report["decisions"][1]["rgb_sha256"]
    assert report["decisions"][0]["decision_id"] != report["decisions"][1]["decision_id"]
    assert host.solver.calls[0][0].observation.observation_revision == 0
    assert host.solver.calls[1][0].observation.observation_revision == 1
    assert "Action failed" in host.solver.calls[1][0].observation.public_feedback
    assert report["budget"]["reserved"]["qwen"] == 2
    assert len(report["executions"]) == (0 if arm == "Direct" else 2)
    assert report["task_success"] is True


def test_fresh_process_drops_variables_during_repair_without_mutating_environment():
    source = "private_variable = 99\nraise ValueError('synthetic-private')\n" + SUBMIT
    second = ("stale = False\ntry:\n    private_variable\n    stale = True\n"
              "except Exception:\n    pass\nassert stale is False\n" + SUBMIT)
    solver = FakeSolver(scripted=(source, second))
    host, adapter = setup(solver=solver)
    report = run(host)
    assert report["terminal_reason"] == "official_done", report
    assert adapter.step_count == 2 and len(report["executions"]) == 3
    assert solver.calls[0][0].decision_id == solver.calls[1][0].decision_id
    assert solver.calls[1][1]["code"] == "violation_runtime"
    assert "synthetic-private" not in repr(report)


@pytest.mark.parametrize("environment", ("eb_alfred", "eb_habitat"))
def test_format_invalid_reuses_rgb_but_new_decision_and_revision(environment):
    solver = FakeSolver(scripted=(ContractError("model_output_format_error"),))
    host, adapter = setup(environment, solver=solver, stationary_rgb=True)
    report = run(host)
    assert report["terminal_reason"] == "official_done", report
    assert adapter.invalid_count == 1 and adapter.step_count == 2
    assert report["mutation_revision"] == 4 and report["observation_revision"] == 3
    assert solver.calls[0][0].decision_id != solver.calls[1][0].decision_id
    assert solver.calls[1][0].observation.environment_step == 0
    assert solver.calls[1][0].observation.mutation_revision == 2
    assert report["budget"]["reserved"]["invalid"] == 1


def test_empty_stop_does_not_self_declare_success():
    solver = FakeSolver(scripted=(SUBMIT.replace("[0, 1]", "[]"),))
    host, adapter = setup(solver=solver)
    report = run(host)
    assert report["terminal_reason"] == "empty_plan_stop" and not report["task_success"]
    assert adapter.step_count == 0 and report["environment_steps"] == 0


def test_model_service_failure_stops_once_and_still_counts_permit():
    solver = FakeSolver(scripted=(OperationalError("synthetic-private-url"),))
    host, adapter = setup(solver=solver)
    report = run(host)
    assert report["status"] == "aborted" and report["terminal_reason"] == "operational_failure"
    assert len(solver.calls) == 1 and report["budget"]["reserved"]["qwen"] == 1
    assert adapter.step_count == 0 and "synthetic-private" not in repr(report)


def test_execution_cap_stops_without_another_request():
    host, adapter = setup(solver=FakeSolver(scripted=("print('x' * 100000)\n" + SUBMIT,)))
    report = run(host)
    assert report["terminal_reason"] == "budget_exhausted:execution_output_limit", report
    assert len(host.solver.calls) == 1 and adapter.step_count == 0


@pytest.mark.parametrize("environment", ("eb_alfred", "eb_habitat"))
def test_direct_invalid_tail_rejected_before_first_action(environment):
    solver = FakeSolver("Direct", scripted=((0, True),))
    host, adapter = setup(environment, arm="Direct", solver=solver)
    report = run(host)
    assert report["terminal_reason"] == "official_done", report
    assert adapter.invalid_count == 1 and adapter.step_count == 2
    assert report["errors"][0]["code"] == "invalid_action_plan"


def test_lost_action_ack_stops_without_retry_or_reset():
    class DropACK(LocalTransport):
        def exchange(self, command, timeout_s):
            ack = super().exchange(command, timeout_s)
            if command["op"] == "step":
                raise OperationalError("synthetic-wire")
            return ack
    host, adapter = setup(transport_type=DropACK)
    report = run(host)
    assert report["status"] == "aborted" and report["terminal_reason"] == "action_outcome_unknown"
    assert adapter.step_count == 1 and adapter.open_count == 1
    assert report["environment_steps"] == 0 and report["budget"]["reserved"]["steps"] == 1
    assert len(host.solver.calls) == 1


def test_three_program_errors_stop_at_same_decision_cap_without_actions():
    host, adapter = setup(solver=FakeSolver(scripted=("raise ValueError()",) * 3))
    report = run(host)
    assert report["terminal_reason"] == "budget_exhausted:decision_synthesis", report
    assert adapter.step_count == 0 and len(report["executions"]) == 3
    assert len(host.solver.calls) == 3


def test_same_decision_yield_uses_bounded_reader_then_action_reprepares_geometry():
    import numpy as np
    from PIL import Image
    import io
    prepared_calls = []
    def prepare(snapshot, binding, budget):
        prepared_calls.append(snapshot.decision_id)
        rgb = np.asarray(Image.open(io.BytesIO(snapshot.observation.png)).convert("RGB"))
        geometry = SimpleNamespace(frames=[rgb], candidates=[], provenance={},
                                   geometries={0: {"depth": np.ones((500, 500), dtype=np.float32),
                                                   "valid_mask": np.ones((500, 500), dtype=bool)}})
        return PreparedEvidence(geometry, sha256(snapshot.decision_id.encode()),
                                ("current_rgb", "relative_depth"))
    class YieldSolver(FakeSolver):
        def generate(self, snapshot, *, feedback, usage):
            self.calls.append((snapshot, deepcopy(feedback)))
            if not feedback:
                return Proposal(source="r = point_depth(0,5,5)\nYieldObservations([r['result_id']], 'inspect')")
            rid = feedback["yielded_result_ids"][0]
            return Proposal(source=f"ReadObservation('{rid}', ['depth'])\n" + SUBMIT)
    solver = YieldSolver()
    host, adapter = setup(solver=solver, preparer=prepare)
    report = run(host)
    assert report["terminal_reason"] == "official_done", report
    assert adapter.step_count == 2 and len(prepared_calls) == 2
    assert len(solver.calls) == 4 and len(report["executions"]) == 4
    assert report["budget"]["reserved"]["tools"] == 2 and report["budget"]["reserved"]["reads"] == 4
    assert solver.calls[1][1]["observations"][0]["data"]["depth"] == 1
    assert solver.calls[0][0].decision_id == solver.calls[1][0].decision_id
    assert solver.calls[1][0].decision_id != solver.calls[2][0].decision_id
    assert len(report["revoked_decisions"]) == 2
