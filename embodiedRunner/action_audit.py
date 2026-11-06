"""Strict, read-only paired-run audit. Failed attempts remain in the denominator."""
from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

from action_contract import plain, public_hash
from action_identity import (ARMS, COMPARE_ARMS, ENVIRONMENTS, configuration_identity, protocol_identity,
                             source_identity)
from action_runtime.execution_contract import ActionExecutionContract
from action_solver import MODEL_CONFIG
from budget import STAGES, QWEN_STAGES, numeric_usage, policy_for
from methods import load_methods
from protocol import ContractError, canonical, json_object, sha256
from state import TRANSITIONS

ROOT = Path(__file__).resolve().parent
DIMENSIONS = ("steps", "invalid", "qwen", "models", "reviews", "dino", "sam",
              "moge", "tools", "reads", "segments")


def record_hash(value):
    return public_hash({k: v for k, v in value.items() if k not in {"sha256", "audit"}})


def seal_record(value):
    value["sha256"] = record_hash(value)
    return value


class Rejected(ValueError):
    pass


def require(condition, code):
    if not condition:
        raise Rejected(code)


def count(value):
    require(type(value) is int and value >= 0, "invalid_count")
    return value


def hashed(value):
    require(isinstance(value, str) and len(value) == 64
            and all(c in "0123456789abcdef" for c in value), "invalid_hash")
    return value


def policy_flags(row):
    require(row["automatic_retry"] is False and row["reference_answer_visible"] is False
            and row["input_policy"] == "current_only"
            and row["execution_policy"] == "single_action", "policy_deviation")


def audit_attempt(report, item, manifest):
    require(report["schema"] == "embodied-action-attempt/1", "attempt_schema")
    require(report["sha256"] == record_hash(report), "attempt_hash")
    arm, environment = report["arm"], item["environment"]
    require(report["environment"] == environment and report["episode_id"] == item["episode_id"]
            and report["source_sha256"] == item["source_sha256"]
            and report["run_id"] == manifest["run_id"] + ":" + arm, "attempt_identity")
    for key in ("paired_manifest_sha256", "implementation_source_sha256",
                "protocol_sha256", "prepared_config_sha256"):
        expected = {
            "paired_manifest_sha256": manifest["sha256"],
            "implementation_source_sha256": manifest["implementation_source"]["sha256"],
            "protocol_sha256": manifest["protocol"]["sha256"],
            "prepared_config_sha256": manifest["configuration"]["sha256"],
        }[key]
        require(report[key] == expected, "attempt_frozen_identity")
    policy_flags(report)
    require(report["offline_fake"] is manifest["offline_fake"], "attempt_mode")
    require(report["status"] in {"finished", "aborted"} and report["terminal_reason"],
            "incomplete_attempt")
    require(type(report["task_success"]) is bool and type(report["task_progress"]) in {int, float}
            and 0 <= report["task_progress"] <= 1, "official_result_shape")
    if report["status"] == "aborted":
        require(report["task_success"] is False, "aborted_success")
    config = manifest["configuration"]
    method = config["methods"][environment]
    method_hash = method["content_sha256"] if arm in {"B11", "Full"} else ""
    require(report["method_sha256"] == method_hash, "method_identity")
    budget = report["budget"]
    policy = plain(policy_for(environment))
    require(budget["episode_id"] == item["episode_id"]
            and budget["policy"] == policy
            and budget["policy_sha256"] == public_hash(policy)
            and budget["automatic_retry"] is False, "budget_policy")
    reserved = Counter({k: count(v) for k, v in budget["reserved"].items()})
    require(set(reserved) <= set(DIMENSIONS)
            and all(reserved[k] <= policy[k] for k in DIMENSIONS), "budget_cap")

    phase, previous, obs_revision = "opening", "", -1
    observed = []
    for ordinal, row in enumerate(report["transitions"]):
        require(row["ordinal"] == ordinal and type(row["ordinal"]) is int
                and row["from"] == phase and row["to"] in TRANSITIONS[phase]
                and row["episode_id"] == item["episode_id"]
                and row["previous_sha256"] == previous
                and row["sha256"] == record_hash(row), "transition_chain")
        if row["to"] == "observed":
            obs_revision += 1
            observed.append(row["reason"])
        require(row["observation_revision"] == obs_revision, "transition_observation")
        phase, previous = row["to"], row["sha256"]
    # An endpoint can fail before a host state exists; its empty history is explicit.
    if report["transitions"]:
        require(phase == report["status"]
                and report["transitions"][-1]["reason"] == report["terminal_reason"],
                "terminal_transition")
    else:
        require(report["status"] == "aborted" and not report["observations"]
                and not report["decisions"], "missing_transition_history")

    revision, step, table, session = 0, 0, "", None
    acked, mutations = [], report["mutations"]
    command_ids = set()
    for ordinal, row in enumerate(mutations):
        require(row["op"] == ("open" if ordinal == 0 else row["op"])
                and row["op"] in {"open", "step", "invalid"}
                and (ordinal == 0 or row["op"] != "open")
                and (environment != "eb_navigation" or row["op"] != "invalid"),
                "mutation_order")
        require(row["command_id"] not in command_ids, "duplicate_mutation")
        command_ids.add(row["command_id"])
        hashed(row["command_digest"])
        require(row["episode_id"] == item["episode_id"]
                and row["protocol_version"] == manifest["protocol"]["versions"][environment]
                and type(row["transport_sequence"]) is int and row["transport_sequence"] == ordinal + 1
                and row["expected_mutation_revision"] == revision
                and row["expected_environment_step"] == step
                and row["expected_action_table_sha256"] == table
                and row["automatic_retry"] is False, "mutation_identity")
        session = session or hashed(row["session_sha256"])
        require(row["session_sha256"] == session, "mutation_session")
        if row["status"] == "unknown":
            require(ordinal == len(mutations) - 1
                    and report["terminal_reason"] == "action_outcome_unknown",
                    "unknown_mutation_resumed")
            break
        require(row["status"] == "ack", "incomplete_mutation")
        revision += 1
        step += row["op"] == "step"
        require(row["mutation_revision"] == revision and row["environment_step"] == step,
                "mutation_revision")
        table = hashed(row["action_table_sha256"])
        hashed(row["packet_sha256"])
        hashed(row["image_sha256"])
        acked.append(row)
    require(report["mutation_revision"] == revision and report["environment_steps"] == step
            and reserved["steps"] == sum(r["op"] == "step" for r in mutations)
            and reserved["invalid"] == sum(r["op"] == "invalid" for r in mutations),
            "mutation_budget")
    require(sum(r["to"] == "applying" for r in report["transitions"])
            == sum(r["op"] == "step" for r in mutations), "action_transition_count")
    observations = report["observations"]
    require(len(observations) == len(acked) == len(observed)
            and report["observation_revision"] == len(observations) - 1,
            "observation_denominator")
    official_invalid = 0
    for ordinal, (obs, ack, kind) in enumerate(zip(observations, acked, observed)):
        require(obs["observation_revision"] == ordinal and obs["mutation_kind"] == kind == ack["op"]
                and obs["mutation_revision"] == ack["mutation_revision"]
                and obs["environment_step"] == ack["environment_step"]
                and obs["rgb_sha256"] == ack["image_sha256"]
                and obs["action_table_sha256"] == ack["action_table_sha256"],
                "observation_identity")
        for key in ("pixel_sha256", "feedback_sha256", "memory_sha256", "input_sha256"):
            hashed(obs[key])
        if ordinal:
            require(obs["input_sha256"] == observations[0]["input_sha256"], "changed_task")
        if kind == "invalid":
            require(obs["rgb_sha256"] == observations[ordinal - 1]["rgb_sha256"],
                    "invalid_changed_rgb")
        # Official environments may increment invalid actions for a normal
        # step that fails. Host format-invalid permits count separate mutations.
        new_invalid = count(obs.get("official_invalid_actions", official_invalid + (kind == "invalid")))
        require(new_invalid >= official_invalid and new_invalid <= policy["invalid"]
                and (kind != "invalid" or new_invalid == official_invalid + 1), "official_invalid_counter")
        official_invalid = new_invalid
    require(report["invalid_actions"] == official_invalid,
            "invalid_count")

    decisions = {}
    decision_observations = set()
    for row in report["decisions"]:
        decision_id = row["decision_id"]
        require(decision_id not in decisions, "duplicate_decision")
        index = count(row["observation_revision"])
        require(index < len(observations) and index not in decision_observations,
                "decision_observation")
        decision_observations.add(index)
        obs = observations[index]
        require(all(row[k] == obs[k] for k in (
            "observation_revision", "mutation_revision", "environment_step",
            "rgb_sha256", "action_table_sha256")), "stale_decision")
        if environment == "eb_manipulation":
            require(all(row[k] == obs[k] for k in
                        ("manipulation_input_sha256", "planner_rgb_sha256")),
                    "stale_manipulation_input")
            hashed(row["manipulation_input_sha256"])
            hashed(row["planner_rgb_sha256"])
        hashed(row["snapshot_sha256"])
        require(row["method_sha256"] == method_hash, "decision_method")
        if arm == "Direct":
            require(row["evidence_manifest_sha256"] == "" and row["evidence_revision"] == 0
                    and row["evidence_config_sha256"] == ""
                    and row["capabilities"] == ["current_rgb"], "direct_evidence")
        else:
            require(row["evidence_config_sha256"] == config["evidence_config_sha256"]
                    and "current_rgb" in row["capabilities"], "decision_evidence")
            if row["evidence_status"] == "preparing":
                require(report["status"] == "aborted" and row is report["decisions"][-1]
                        and not any(r["decision_id"] == decision_id for r in report["model_requests"]),
                        "failed_preparation_continued")
            else:
                hashed(row["evidence_manifest_sha256"])
                require(row["evidence_revision"] == 1 and row["evidence_status"] == "prepared",
                        "decision_evidence")
        decisions[decision_id] = row
    require(report["revoked_decisions"] == sorted(decisions), "decision_revocation")
    require(len(decisions) == sum(r["to"] == "preparing" for r in report["transitions"]),
            "decision_denominator")

    requests, solver_requests = budget["requests"], report["model_requests"]
    permits, request_counts, per_decision, tokens = set(), Counter(), Counter(), 0
    service_views = set()
    for row in requests:
        require(row["permit_id"] not in permits, "duplicate_permit")
        permits.add(row["permit_id"])
        stage = row["stage"]
        require(stage in STAGES and row["episode_id"] == item["episode_id"]
                and row["automatic_retry"] is False
                and row["status"] in {"completed", "error"}, "request_receipt")
        request_counts.update(STAGES[stage])
        usage = numeric_usage(row["usage"])
        expected_missing = (stage in QWEN_STAGES or stage == "category_review") and (
            "total_tokens" not in usage)
        require(row["usage_missing"] is expected_missing, "request_usage_missing")
        if stage in {"dino", "sam", "moge"}:
            key = (stage, hashed(row["view_sha256"]))
            require(key not in service_views, "repeated_view_service")
            service_views.add(key)
        if stage in QWEN_STAGES:
            tokens += usage.get("total_tokens", usage.get("prompt_tokens", 0)
                                + usage.get("completion_tokens", 0))
        if stage in {"synthesis", "repair"}:
            require(row["decision_id"] in decisions, "request_stale_decision")
            per_decision[row["decision_id"]] += 1
            require(per_decision[row["decision_id"]] <= policy["synthesis_per_decision"],
                    "decision_request_cap")
    require(all(reserved[k] == request_counts[k] for k in
                ("qwen", "models", "reviews", "dino", "sam", "moge"))
            and budget["qwen_tokens_observed"] == tokens, "request_budget")
    solver_permits = [r for r in requests if r["stage"] in {"synthesis", "repair"}]
    require(len(solver_permits) == len(solver_requests), "request_missing_receipt")
    last_observation = -1
    for receipt, permit in zip(solver_requests, solver_permits):
        decision = decisions[receipt["decision_id"]]
        require(receipt["stage"] == permit["stage"] and receipt["decision_id"] == permit["decision_id"]
                and receipt["arm"] == arm and receipt["environment"] == environment
                and receipt["model_config_sha256"] == manifest["model_config_sha256"]
                and receipt["contract_sha256"] == ActionExecutionContract(environment=environment).sha256
                and receipt["reference_answer_visible"] is False
                and type(receipt["input_image_count"]) is int and receipt["input_image_count"] == 1
                and all(receipt[k] == decision[k] for k in (
                    "snapshot_sha256", "observation_revision", "rgb_sha256",
                    "action_table_sha256", "method_sha256", "evidence_manifest_sha256")),
                "solver_request_identity")
        if environment == "eb_manipulation":
            require(all(receipt[k] == decision[k] for k in
                        ("manipulation_input_sha256", "planner_rgb_sha256")),
                    "solver_manipulation_input_identity")
        require(receipt["observation_revision"] >= last_observation, "request_revoked_decision")
        last_observation = receipt["observation_revision"]
        hashed(receipt["messages_sha256"])
        require(count(receipt["text_chars"]) <= 96000, "prompt_context_cap")
        status = receipt["status"]
        require(status in {"valid_program", "valid_plan", "format_error", "service_error"},
                "solver_service_or_usage_deviation")
        if status == "service_error":
            require(permit["status"] == "error" and receipt["delivery"]["channel"] == "request_failed"
                    and receipt is solver_requests[-1]
                    and report["status"] == "aborted", "service_error_resumed")
        else:
            require(receipt["delivery"]["channel"] == "model_request"
                    and receipt.get("usage", {}) == permit["usage"], "solver_usage")
            if status.startswith("valid_"):
                require(permit["status"] == "completed"
                        and status == ("valid_plan" if arm == "Direct" else "valid_program"),
                        "solver_status")
            if "reported_model_sha256" in receipt:
                expected = sha256(("offline-fake" if manifest["offline_fake"]
                                   else MODEL_CONFIG["model"]).encode())
                require(receipt["reported_model_sha256"] == expected, "reported_model_deviation")
        delivery = receipt["delivery"]
        expected_entries = ([{"skill_version": method["skill_version"],
                              "content_sha256": method_hash,
                              "chars": method["complete_body_chars"]}] if arm in {"B11", "Full"} else [])
        require(delivery["entries"] == expected_entries and delivery["dropped"] == []
                and delivery["used_chars"] == sum(e["chars"] for e in expected_entries)
                and delivery["max_chars"] == ActionExecutionContract().method_context_max_chars
                and delivery["channel_note"] == ""
                and delivery["delivered_skill_versions"] == (
                    [e["skill_version"] for e in expected_entries] if status != "service_error" else []),
                "method_delivery")
    executions = report["executions"]
    require(reserved["segments"] == len(executions), "execution_denominator")
    execution_events = [r for r in report["transitions"] if r["to"] == "executing"]
    require(len(execution_events) == len(executions), "execution_transition_count")
    if arm == "Direct":
        require(not executions and all(not reserved[k] for k in
                ("tools", "reads", "segments", "dino", "sam", "moge", "reviews")),
                "direct_tools_or_perception")
    actual_tools, actual_reads = 0, 0
    valid_programs = Counter((r["decision_id"], r["program_sha256"])
                             for r in solver_requests if r["status"] == "valid_program")
    for row, event in zip(executions, execution_events):
        require(row["decision_id"] in decisions, "execution_stale_decision")
        require(event["observation_revision"] == decisions[row["decision_id"]]["observation_revision"],
                "execution_observation_order")
        key = (row["decision_id"], row["program_sha256"])
        require(valid_programs[key] > 0, "execution_program_identity")
        valid_programs[key] -= 1
        actual_tools += count(row["tool_calls"])
        actual_reads += count(row["reader_calls"])
        receipt = row["receipt"]
        require(receipt["automatic_retry"] is False, "execution_retry")
        if receipt.get("isolation_sealed"):
            require(receipt["isolation_sealed"] is True
                    and receipt["program_sha256"] == row["program_sha256"]
                    and receipt["snapshot_sha256"] == decisions[row["decision_id"]]["snapshot_sha256"]
                    and len(receipt["tool_receipts"]) == row["tool_calls"],
                    "execution_receipt")
            for tool in receipt["tool_receipts"]:
                require(tool["episode_id"] == item["episode_id"]
                        and type(tool["allowed"]) is bool, "tool_authorization")
        else:
            require(row["error_code"] is not None and row is executions[-1],
                    "unsealed_execution_continued")
    require(actual_tools <= reserved["tools"] and actual_reads <= reserved["reads"],
            "execution_budget")
    for decision_id, row in budget["decisions"].items():
        # Preparation can stop before its snapshot is added to the decision report.
        require(decision_id in decisions or report["status"] == "aborted",
                "budget_stale_decision")
        require(set(row) <= {"synthesis", "qwen", "models", "reviews", "dino",
                             "sam", "moge", "tools", "reads", "segments"},
                "decision_budget_dimensions")
        for key in ("synthesis", "tools", "reads", "segments"):
            require(count(row.get(key, 0)) <= policy[key + "_per_decision"],
                    "decision_budget_cap")
    return reserved


def audit_run(result, *, verify_current_sources=True):
    """Reject malformed/deviating runs; retain all selected items in metrics."""
    try:
        require(result["schema"] in {"embodied-offline-run/1", "embodied-live-run/1"}, "run_schema")
        live = result["schema"] == "embodied-live-run/1"
        require(result["sha256"] == record_hash(result), "run_hash")
        manifest = result["manifest"]
        require(manifest["schema"] == "embodied-paired-run/1"
                and manifest["sha256"] == record_hash(manifest), "manifest_hash")
        policy_flags(manifest)
        arms = manifest["arms"]
        require(arms in (list(ARMS), list(COMPARE_ARMS)) and manifest["offline_fake"] is (not live),
                "run_arms_or_mode")
        if not live:
            require(result["live_requests"] == result["official_simulator_launches"]
                    == result["deployments"] == 0, "offline_external_call")
        source, protocol, config = (manifest["implementation_source"], manifest["protocol"],
                                    manifest["configuration"])
        require(source["sha256"] == public_hash(source["files_sha256"]), "source_freeze_hash")
        require(protocol == protocol_identity(), "protocol_freeze")
        current_config = configuration_identity(load_methods(), manifest["evidence_config_sha256"])
        if live:
            require(config["sha256"] == record_hash(config)
                    and config["runtime"]["platform"] == "Linux"
                    and len(config["runtime"]["python"]) == 3
                    and config["runtime"]["python"][:2] == [3, 11]
                    and isinstance(config["runtime"]["numpy"], str), "live_runtime_identity")
            # The Mac auditor verifies scientific/configuration pins while the
            # sealed preflight records the independently selected Linux runtime.
            current_config["runtime"] = config["runtime"]
            seal_record(current_config)
        require(config == current_config,
                "configuration_freeze")
        if verify_current_sources:
            require(source == source_identity(), "source_changed_after_freeze")
        require(manifest["model_config_sha256"] == public_hash(MODEL_CONFIG)
                and manifest["method_library_sha256"] == config["method_library_sha256"],
                "manifest_configuration")
        selection = manifest["selection"]
        require(type(selection) is list and selection
                and count(manifest["denominator_per_arm"]) == len(selection), "fixed_denominator")
        items = {}
        for item in selection:
            require(item["environment"] in ENVIRONMENTS and item["episode_id"] not in items
                    and isinstance(item["subset"], str) and item["subset"]
                    and type(item["index"]) is int and item["index"] >= 0,
                    "selection_identity")
            hashed(item["source_sha256"])
            items[item["episode_id"]] = item
        require(len({(i["environment"], i["subset"], i["index"]) for i in selection})
                == len(selection), "duplicate_selection_source")
        environments = {i["environment"] for i in selection}
        require(manifest["budget_sha256_by_environment"] == {
                    e: public_hash(policy_for(e)) for e in environments}
                and manifest["method_sha256_by_environment"] == {
                    e: config["methods"][e]["content_sha256"] for e in environments},
                "manifest_environment_identity")
        attempts = result["attempts"]
        if live:
            limit = manifest["integration_step_limit"]
            require(limit is None or type(limit) is int and 1 <= limit <= 20, "live_step_limit")
            require(type(manifest["require_identical_initial_observation"]) is bool, "live_pairing_policy")
            require(all(r["terminal_reason"] != "integration_step_limit" or
                        limit is not None and r["status"] == "finished" and
                        r["environment_steps"] == limit for r in attempts), "live_step_limit_receipt")
            preflight = manifest["live_preflight"]
            require(preflight["schema"] == "embodied-live-preflight/1"
                    and preflight["sha256"] == record_hash(preflight)
                    and preflight["status"] == "ready" and preflight["selection"] == selection
                    and preflight["live_config_sha256"] == manifest["live_config_sha256"]
                    and preflight["evidence_config_sha256"] == manifest["evidence_config_sha256"]
                    and preflight["live_requests"] == preflight["official_simulator_launches"] == 0,
                    "live_preflight_identity")
            require(preflight["isolation"] == {
                "backend": "linux-bwrap-seccomp/1", "namespaces_verified": True,
                "syscall_allowlist": True, "inherited_fds_closed": True}, "live_isolation_preflight")
            require(set(preflight["runtime"]) == environments, "live_runtime_denominator")
            for environment, row in preflight["runtime"].items():
                require(row["runtime_ready"] is True and row["environment"] == environment
                        and row["simulator_launches"] == row["model_requests"] == 0
                        and all(digest == source["files_sha256"]["embodiedRunner/" + name]
                                for name, digest in row["source_sha256"].items()),
                        "live_official_runtime")
            require([(r["arm"], r["episode_id"]) for r in attempts]
                    == [(a, i["episode_id"]) for i in selection for a in arms],
                    "live_attempt_order")
            stopped = False
            for row in attempts:
                if stopped:
                    require(row["terminal_reason"] == "not_started_after_operational_stop"
                            and row["official_open_attempted"] is False
                            and not row["mutations"] and not row["model_requests"]
                            and not row["budget"]["requests"], "live_operational_stop_resumed")
                stopped = stopped or row["status"] == "aborted"
            require(count(result["live_requests"]) == sum(sum(
                        r["budget"]["reserved"].get(k, 0) for k in
                        ("qwen", "reviews", "dino", "sam", "moge")) for r in attempts)
                    and count(result["official_simulator_launches"]) == sum(
                        bool(r["observations"]) for r in attempts)
                    and count(result["official_open_attempts"]) == sum(
                        r["official_open_attempted"] for r in attempts)
                    and result["deployments"] == 0, "live_external_accounting")
        expected = {(a, i) for a in arms for i in items}
        found, sums = set(), {a: Counter() for a in arms}
        for report in attempts:
            key = report["arm"], report["episode_id"]
            require(key in expected and key not in found, "duplicate_or_unselected_attempt")
            found.add(key)
            sums[key[0]].update(audit_attempt(report, items[key[1]], manifest))
        require(found == expected, "missing_paired_attempt")
        for arm in arms:
            budget = result["run_budgets"][arm]
            caps = Counter()
            for item in selection:
                caps.update({k: policy_for(item["environment"])[k] for k in DIMENSIONS})
            require(budget["run_id"] == manifest["run_id"] + ":" + arm
                    and budget["planned_episodes"] == len(selection)
                    and budget["caps"] == dict(caps)
                    and Counter(budget["reserved"]) == sums[arm]
                    and budget["automatic_retry"] is False and budget["max_inflight"] == 1,
                    "run_budget_aggregation")
        expected_status = ("passed" if all(r["status"] != "aborted" and (
                            manifest["integration_step_limit"] is None
                            or r["environment_steps"] >= manifest["integration_step_limit"]
                            or r["terminal_reason"] == "official_done") for r in attempts) else "failed"
                           ) if live else (
                           "passed" if all(r["terminal_reason"] == "official_done" for r in attempts) else "failed")
        require(result["status"] == expected_status, "run_status")
        initial_matches = {}
        for episode_id in items:
            first_observations = [r["observations"][0] for r in attempts
                                  if r["episode_id"] == episode_id and r["observations"]]
            if first_observations:
                # Public task/action/input identity remains exact in integration
                # smoke even when independent official renders vary.
                require(all(all(o[k] == first_observations[0][k] for k in
                                ("input_sha256", "action_table_sha256", "memory_sha256"))
                            for o in first_observations), "paired_initial_task_identity")
                if items[episode_id]["environment"] == "eb_manipulation":
                    require(all(o["manipulation_input_sha256"] ==
                                first_observations[0]["manipulation_input_sha256"]
                                for o in first_observations), "paired_manipulation_input")
                identical = all(o == first_observations[0] for o in first_observations)
                initial_matches[episode_id] = identical
                if not live or manifest["require_identical_initial_observation"]:
                    require(identical, "paired_initial_observation")
            else:
                initial_matches[episode_id] = False
        metrics = {}
        for arm in arms:
            rows = [r for r in attempts if r["arm"] == arm]
            successes = sum(r["task_success"] and r["status"] == "finished" for r in rows)
            metrics[arm] = {"selected": len(selection), "attempted": len(rows),
                            "started": sum(bool(r["observations"]) for r in rows),
                            "successes": successes, "success_rate": successes / len(selection),
                            "aborted": sum(r["status"] == "aborted" for r in rows),
                            "terminal_reasons": dict(Counter(r["terminal_reason"] for r in rows))}
        return {"schema": "embodied-paired-audit/1", "status": "passed", "errors": [],
                "pairs": len(selection), "attempts": len(attempts), "metrics": metrics,
                "run_sha256": result["sha256"], "live_requests": result["live_requests"],
                "source_reverified": verify_current_sources,
                "initial_observation_match": initial_matches,
                "paired_comparison_eligible": (
                    all(initial_matches.values())
                    and all(r["observations"] for r in attempts)
                    and (not live or manifest["integration_step_limit"] is None)),
                "validation_scope": ("live integration, protocol and accounting; task success reported separately"
                                     if live else "offline protocol and accounting; no benchmark quality claim")}
    except Rejected as error:
        code = str(error)
    except (KeyError, TypeError, ValueError, OSError, IndexError, AttributeError):
        code = "malformed_audit_record"
    return {"schema": "embodied-paired-audit/1", "status": "rejected", "errors": [code],
            "live_requests": 0}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True, type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    path = args.run.resolve()
    if not path.is_relative_to(ROOT) or not path.is_file() or path.stat().st_size > 32 * 1024**2:
        parser.error("run must be a bounded JSON file inside embodiedRunner")
    try:
        result = audit_run(json_object(path.read_bytes()))
    except ContractError:
        result = {"schema": "embodied-paired-audit/1", "status": "rejected",
                  "errors": ["invalid_run_json"], "live_requests": 0}
    if args.output:
        output = args.output.resolve()
        if not output.is_relative_to(ROOT) or not output.parent.is_dir() or output.exists():
            parser.error("output must be a new file in an existing embodiedRunner directory")
        with output.open("xb") as stream:
            stream.write(canonical(result) + b"\n")
    print(canonical(result).decode())
    if result["status"] != "passed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
