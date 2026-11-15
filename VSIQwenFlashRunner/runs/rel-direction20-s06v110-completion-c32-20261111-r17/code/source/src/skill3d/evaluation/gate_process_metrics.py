"""Descriptive gate metrics; these do not replace the legacy admission rule."""

from collections import Counter

from skill3d.evolution.process_feedback import extract_process_facts

PROCESS_METRICS_VERSION = "gate-process-metrics/1.0"


def episode_process_metrics(outcome: dict, responses: list[dict]) -> dict:
    facts = extract_process_facts(outcome)
    events = facts.error_chain
    usage = [row.get("usage") or {} for row in responses]
    return {
        "schema_version": PROCESS_METRICS_VERSION,
        "error_events": dict(sorted(Counter(event.event_type for event in events).items())),
        "recovered_errors": sum(event.recovered is True for event in events),
        "terminal_errors": sum(event.terminal for event in events),
        "no_progress_rounds": facts.execution.repeated_no_progress_rounds,
        "tool_calls": sum(facts.execution.tool_calls_by_name.values()),
        "request_count": len(responses),
        "request_errors": sum(row.get("status") == "error" for row in responses),
        "request_latency_s": sum(float(row.get("elapsed_s") or 0) for row in responses),
        "usage_measured_requests": sum(bool(row) for row in usage),
        "prompt_tokens": sum(int(row.get("prompt_tokens") or 0) for row in usage),
        "completion_tokens": sum(int(row.get("completion_tokens") or 0) for row in usage),
        "token_totals_complete": bool(responses) and all(
            "prompt_tokens" in row and "completion_tokens" in row for row in usage),
        "answer_untrusted": bool(outcome.get("answer_untrusted")),
    }


def aggregate_gate_process_metrics(rows: list[dict]) -> dict:
    events = Counter()
    for row in rows:
        events.update(row["error_events"])
    totals = {key: sum(row[key] for row in rows) for key in (
        "recovered_errors", "terminal_errors", "no_progress_rounds", "tool_calls",
        "request_count", "request_errors", "request_latency_s", "usage_measured_requests",
        "prompt_tokens", "completion_tokens",
    )}
    return {
        "schema_version": PROCESS_METRICS_VERSION, **totals,
        "error_events": dict(sorted(events.items())),
        "token_totals_complete": bool(rows) and all(row["token_totals_complete"] for row in rows),
    }
