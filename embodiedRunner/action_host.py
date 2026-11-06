"""Host-owned single-action FSM. Solvers never receive a simulator/bridge handle."""
from __future__ import annotations

from dataclasses import dataclass
import base64
import sys

from action_adapters import validate_packet
from action_contract import (ActionPayload, EpisodeBinding, ExecutionEnvelope, plain,
                             public_observation, validate_action_payload)
from action_tools import ResultStore
from budget import BudgetExhausted
from protocol import ContractError, OperationalError, canonical, sha256
from sandbox_process import execute_isolated
from state import EpisodeState

FORMAT_ERRORS = frozenset({"model_output_format_error", "invalid_action_plan"})
PROGRAM_ERRORS = frozenset({
    "timeout", "oom", "output_limit", "tool_budget", "violation_syntax",
    "violation_runtime", "tool_contract", "evidence_rejected", "fallthrough"})
FATAL_EXECUTION = frozenset({
    "execution_worker_failure", "execution_result_cap", "worker_exit", "protocol_failure"})
EXECUTION_CAP_ERRORS = frozenset({"timeout", "oom", "output_limit", "tool_budget"})


@dataclass(frozen=True)
class Proposal:
    """Exactly one completion decoded/assembled by a task-local solver."""
    source: str | None = None
    actions: tuple | None = None


@dataclass(frozen=True)
class PreparedEvidence:
    prepared: object
    manifest_sha256: str
    capabilities: tuple


def execution_envelope(state):
    obs, snapshot = state.observation, state.snapshot
    return ExecutionEnvelope(obs.episode_id, state.session_id, obs.mutation_revision,
                             obs.environment_step, snapshot.decision_id, snapshot.snapshot_sha256)


class ActionHost:
    def __init__(self, bridge, solver, *, run_id, arm, split_role="development",
                 evidence_preparer=None, method_sha256="", python=sys.executable, trajectory=None):
        if arm not in {"Direct", "B01", "B11", "Full"} or bool(method_sha256) != (arm in {"B11", "Full"}):
            raise ContractError("host_arm_method_mismatch")
        if arm == "Direct" and evidence_preparer is not None:
            raise ContractError("direct_prepared_evidence_forbidden")
        self.bridge, self.solver, self.budget = bridge, solver, bridge.budget
        self.run_id, self.arm, self.split_role = run_id, arm, split_role
        self.evidence_preparer, self.method_sha256, self.python = evidence_preparer, method_sha256, python
        self.state, self.store = None, None
        self.decisions, self.executions, self.errors, self.observations = [], [], [], []
        self.integration_step_limit = None
        self.trajectory = trajectory

    def _observe(self, packet, kind, *, memory=None):
        png = validate_packet(packet)
        if self.state is None:
            binding = EpisodeBinding(
                self.run_id, self.bridge.episode_id, packet["official_task_id"],
                packet["environment"], packet["subset"], packet["input_sha256"], self.split_role)
            self.state = EpisodeState(binding, self.bridge.session_id, self.method_sha256, self.budget.sha256)
        else:
            binding = self.state.binding
        revision = 0 if self.state.observation is None else self.state.observation.observation_revision + 1
        if self.store:
            self.store.revoke()
            self.store = None
        previous_memory = memory
        if previous_memory is None and self.state.observation is not None:
            previous_memory = self.state.observation.public_memory
        if previous_memory is not None:
            previous_memory = plain(previous_memory)
            # Historical facts stay attributed to the observation that supplied
            # them; they do not become current object/geometry state.
            for record in previous_memory["records"]:
                record["confidence"] = "historical"
        feedback = packet["env_feedback"] + (
            "\nLast official action succeeded: " + str(packet["last_action_success"]).lower())
        observation = public_observation(
            binding, png=png, instruction=packet["instruction"], environment_step=packet["step"],
            mutation_revision=self.bridge.revision, observation_revision=revision,
            action_table=packet["actions"], public_feedback=feedback, public_memory=previous_memory,
            manipulation_input=packet.get("manipulation_input"),
            planner_png=base64.b64decode(packet["planner_image_base64"], validate=True)
            if "planner_image_base64" in packet else None)
        self.state.advance_observation(observation, mutation_kind=kind)
        self.state.invalid_actions = packet["official"]["invalid_actions"]
        self.observations.append({
            "observation_revision": revision, "mutation_kind": kind,
            "mutation_revision": observation.mutation_revision,
            "environment_step": observation.environment_step,
            "rgb_sha256": observation.rgb_sha256, "pixel_sha256": observation.pixel_sha256,
            "action_table_sha256": observation.action_table_sha256,
            "feedback_sha256": observation.feedback_sha256,
            "memory_sha256": observation.memory_sha256,
            "input_sha256": binding.input_sha256,
            "official_invalid_actions": packet["official"]["invalid_actions"],
        })
        if packet["environment"] == "eb_manipulation":
            self.observations[-1]["manipulation_input_sha256"] = packet["manipulation_input"]["sha256"]
            self.observations[-1]["planner_rgb_sha256"] = sha256(observation.planner_png)
        if self.trajectory:
            self.trajectory.emit("observation", mutation_kind=kind,
                                 observation=observation.public_dict(), png=png,
                                 **({"planner_png": observation.planner_png} if observation.planner_png else {}),
                                 receipt=self.observations[-1])

    def _finish(self, reason, *, aborted=False):
        if self.store:
            self.store.revoke()
        official = self.bridge.packet["official"] if self.bridge.packet else {
            "task_success": False, "task_progress": 0.0}
        if self.state and self.state.phase not in {"finished", "aborted"}:
            self.state.finish_episode(reason, aborted=aborted, task_success=official["task_success"],
                                      task_progress=official["task_progress"])
        return self.report(reason=reason, aborted=aborted)

    def report(self, *, reason="", aborted=False):
        state = self.state
        return {
            "schema": "embodied-action-attempt/1", "run_id": self.run_id,
            "episode_id": self.bridge.episode_id, "environment": self.bridge.environment,
            "arm": self.arm, "split_role": self.split_role,
            "status": state.phase if state else ("aborted" if aborted else "opening"),
            "terminal_reason": state.terminal_reason if state else reason,
            "task_success": state.task_success if state else False,
            "task_progress": state.task_progress if state else 0.0,
            "environment_steps": self.bridge.step_count,
            "mutation_revision": self.bridge.revision,
            "invalid_actions": state.invalid_actions if state else 0,
            "observation_revision": state.observation.observation_revision if state else -1,
            "method_sha256": self.method_sha256, "budget": self.budget.summary(),
            "transitions": list(state.ledger) if state else [],
            "mutations": list(self.bridge.receipts), "decisions": list(self.decisions),
            "executions": list(self.executions), "errors": list(self.errors),
            "observations": list(self.observations),
            "revoked_decisions": sorted(state.revoked_decisions) if state else [],
            "reference_answer_visible": False, "input_policy": "current_only",
            "execution_policy": "single_action", "automatic_retry": False,
        }

    def _new_decision(self):
        snapshot = self.state.prepare_snapshot()
        evidence_status, prepared = "preparing" if self.evidence_preparer else "not_used", None
        try:
            if self.evidence_preparer is not None:
                prepared = self.evidence_preparer(snapshot, self.state.binding, self.budget)
                if not isinstance(prepared, PreparedEvidence):
                    raise ContractError("invalid_evidence_preparer_result")
                self.state.extend_evidence(manifest_sha256=prepared.manifest_sha256,
                                           capabilities=prepared.capabilities)
                validate = getattr(self.evidence_preparer, "validate", None)
                if validate is not None:
                    validate(self.state.snapshot, prepared)
                evidence_status = "prepared"
                if self.trajectory:
                    from sandbox_wire import pack_prepared
                    self.trajectory.emit(
                        "prepared_evidence", decision_id=self.state.snapshot.decision_id,
                        manifest=validate(self.state.snapshot, prepared) if validate else None,
                        evidence=pack_prepared(self.state.snapshot, prepared.prepared))
        finally:
            self._record_decision(self.state.snapshot, evidence_status)
        self.store = ResultStore(self.state.snapshot)
        self.state.transition("synthesizing", "current_evidence_ready")
        return prepared

    def _record_decision(self, snapshot, evidence_status):
        self.decisions.append({
            "decision_id": snapshot.decision_id, "snapshot_sha256": snapshot.snapshot_sha256,
            "observation_revision": snapshot.observation.observation_revision,
            "mutation_revision": snapshot.observation.mutation_revision,
            "environment_step": snapshot.observation.environment_step,
            "rgb_sha256": snapshot.observation.rgb_sha256,
            "action_table_sha256": snapshot.observation.action_table_sha256,
            "method_sha256": snapshot.method_sha256,
            "evidence_manifest_sha256": snapshot.evidence_manifest_sha256,
            "capabilities": list(snapshot.capabilities),
            "evidence_revision": snapshot.evidence_revision,
            "evidence_config_sha256": getattr(self.evidence_preparer, "config_sha256", ""),
            "evidence_status": evidence_status,
        })
        if snapshot.observation.environment == "eb_manipulation":
            self.decisions[-1]["manipulation_input_sha256"] = snapshot.observation.manipulation_input["sha256"]
            self.decisions[-1]["planner_rgb_sha256"] = sha256(snapshot.observation.planner_png)
        if self.trajectory:
            self.trajectory.emit("decision", snapshot=snapshot.public_dict(),
                                 receipt=self.decisions[-1])

    def _format_failure(self, code):
        self.errors.append({"code": code, "decision_id": self.state.snapshot.decision_id,
                            "observation_revision": self.state.observation.observation_revision})
        if self.bridge.environment not in {"eb_navigation", "eb_manipulation"}:
            packet = self.bridge.invalid()
            self._observe(packet, "invalid")
            return True
        return False

    def run(self, *, subset, index, source_sha256):
        try:
            self._observe(self.bridge.open(subset=subset, index=index, source_sha256=source_sha256), "open")
            while True:
                self.budget.check_time()
                if self.bridge.packet["official"]["done"]:
                    return self._finish("official_done")
                if (self.integration_step_limit is not None
                        and self.bridge.step_count >= self.integration_step_limit):
                    return self._finish("integration_step_limit")
                if self.bridge.step_count >= self.budget.policy["steps"]:
                    return self._finish("budget_exhausted:steps")
                prepared = self._new_decision()
                feedback, records = {}, ()
                restart = False
                while not restart:
                    snapshot = self.state.snapshot
                    stage = "repair" if feedback else "synthesis"
                    try:
                        # The generator must make exactly one service request;
                        # bounded parser/assembler work belongs to this permit.
                        with self.budget.request(stage, decision_id=snapshot.decision_id) as permit:
                            proposal = self.solver.generate(snapshot, feedback=feedback, usage=permit)
                        if not isinstance(proposal, Proposal):
                            raise ContractError("model_output_format_error")
                        if self.arm == "Direct":
                            if proposal.source is not None or proposal.actions is None:
                                raise ContractError("model_output_format_error")
                            try:
                                payload = ActionPayload(
                                    snapshot.decision_id, snapshot.observation.observation_revision,
                                    snapshot.snapshot_sha256, snapshot.observation.action_table_sha256,
                                    proposal.actions)
                                validate_action_payload(
                                    payload, snapshot, execution_envelope(self.state),
                                    session_id=self.bridge.session_id, max_actions=self.budget.policy["steps"])
                            except ContractError:
                                raise ContractError("invalid_action_plan") from None
                            self.state.transition("validating", "direct_plan")
                        else:
                            if proposal.actions is not None or type(proposal.source) is not str:
                                raise ContractError("model_output_format_error")
                            segment = self.budget.reserve_segment(snapshot.decision_id)
                            self.state.transition("executing", "assembled_program")
                            result = execute_isolated(
                                proposal.source, snapshot, self.state.binding, execution_envelope(self.state),
                                prepared=None if prepared is None else prepared.prepared,
                                previous_results=records, limits=segment.limits,
                                max_actions=self.budget.policy["steps"], python=self.python,
                                capture_details=self.trajectory is not None)
                            segment.settle(result)
                            self.executions.append({
                                "decision_id": snapshot.decision_id,
                                "program_sha256": sha256(proposal.source.encode("utf-8")),
                                "error_code": result.error_code, "terminated": result.terminated,
                                "tool_calls": result.tool_calls, "reader_calls": result.reader_calls,
                                "receipt": result.receipt})
                            if self.trajectory:
                                self.trajectory.emit(
                                    "execution", decision_id=snapshot.decision_id,
                                    program_source=proposal.source,
                                    submission=result.submission.to_dict() if result.submission else None,
                                    results=result.results, reads=result.reads,
                                    yielded_result_ids=result.yielded_result_ids,
                                    receipt=self.executions[-1])
                            if result.error_code in FATAL_EXECUTION:
                                return self._finish("execution_operational_failure", aborted=True)
                            if result.error_code in EXECUTION_CAP_ERRORS:
                                return self._finish("budget_exhausted:execution_" + result.error_code)
                            records = result.results
                            self.store = ResultStore(snapshot, records)
                            if result.error_code is not None or result.terminated == "yield":
                                code = result.error_code or "yield_observations"
                                if code not in PROGRAM_ERRORS | {"yield_observations"}:
                                    raise ContractError("unexpected_execution_error")
                                feedback = {"code": code, "line": result.error_line,
                                            "yielded_result_ids": list(result.yielded_result_ids)[:32]}
                                if code == "violation_syntax":
                                    feedback["guard_codes"] = (
                                        getattr(self.solver, "receipts", [{}])[-1].get("program_guard_codes", []))
                                if result.terminated == "yield":
                                    pages = []
                                    for result_id in result.yielded_result_ids[:3]:
                                        self.budget.reserve_feedback_read(snapshot.decision_id)
                                        page = self.store.read(result_id, limit=4)
                                        candidate = {**feedback, "observations": pages + [page]}
                                        if len(canonical(candidate)) > 4000:
                                            break
                                        pages.append(page)
                                    feedback["observations"] = pages
                                if len(canonical(feedback)) > 4000:
                                    raise ContractError("repair_feedback_cap")
                                self.errors.append({"code": code, "decision_id": snapshot.decision_id})
                                self.state.transition("synthesizing", code)
                                continue
                            payload = result.submission
                            self.state.transition("validating", "program_submitted")
                        validate_action_payload(
                            payload, snapshot, execution_envelope(self.state),
                            session_id=self.bridge.session_id, max_actions=self.budget.policy["steps"],
                            actual_result_ids=tuple(self.store.records), authorize_result=self.store.authorize)
                        if not payload.actions:
                            return self._finish("empty_plan_stop")
                        memory = self.state.adopt_memory(payload)
                        self.state.transition("applying", "validated_single_action")
                        if self.trajectory:
                            if self.bridge.environment == "eb_manipulation":
                                self.trajectory.emit("action_submission", payload=payload.to_dict(),
                                                     action=list(payload.actions[0]))
                            else:
                                self.trajectory.emit(
                                    "action_submission", payload=payload.to_dict(),
                                    action_id=payload.actions[0],
                                    action_name=snapshot.observation.action_table[payload.actions[0]])
                        self._observe(self.bridge.step(payload.actions[0]), "step", memory=memory)
                        restart = True
                    except ContractError as exc:
                        code = str(exc)
                        if code not in FORMAT_ERRORS:
                            # Contract defects are not fed back as invented
                            # environment feedback or automatic model retries.
                            return self._finish("contract_failure", aborted=True)
                        if self.state.phase == "validating":
                            self.state.transition("synthesizing", code)
                        restart = self._format_failure(code)
                        if not restart:
                            feedback = {"code": code}
        except BudgetExhausted as exc:
            return self._finish("budget_exhausted:" + exc.dimension)
        except OperationalError as exc:
            reason = "action_outcome_unknown" if str(exc) == "action_outcome_unknown" else "operational_failure"
            return self._finish(reason, aborted=True)
        except ContractError:
            return self._finish("contract_failure", aborted=True)
        except Exception:
            return self._finish("host_operational_failure", aborted=True)
        finally:
            self.bridge.close()
