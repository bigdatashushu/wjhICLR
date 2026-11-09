"""One answer session; native tool continuations, no generated code or revision."""
from __future__ import annotations

import hashlib
import json
import re
import time

from skill3d.small_image.datasets import normalize_answer
from skill3d.synthesis.request_audit import payload_bytes
from skill3d.synthesis.request_governor import RequestBudgetExhausted

from .client import ServiceError
from .common import PROTOCOL, digest, write_json
from .data import build_direct_messages, build_messages, render_depth, verified_prepared
from .tools import ToolDispatcher, tool_instructions, tool_schemas


def parse_final_answer(text, options):
    """Score explicit choices only. Reasoning and GT never enter this parser."""
    try:
        return normalize_answer(text, options)
    except ValueError:
        pass
    if not isinstance(text, str):
        return None
    value = text.strip()
    try:
        row = json.loads(value)
        if isinstance(row, dict) and isinstance(row.get("answer"), str):
            return normalize_answer(row["answer"], options)
    except (ValueError, TypeError):
        pass
    # A single explicit answer line may accompany explanatory prose.
    labels = set()
    for line in value.splitlines():
        line = line.strip().replace("**", "")
        match = re.fullmatch(
            r"(?i)(?:(?:the\s+)?(?:final\s+)?answer\s*(?:is\s*)?[:=]?\s*)?"
            r"\(?([AB])\)?(?:[.):]\s*(.*)|[.!]?)", line)
        if match:
            label, suffix = match.groups()
            if suffix and suffix.strip().casefold() != options["AB".index(label.upper())].casefold():
                continue
            labels.add(label.upper())
    return labels.pop() if len(labels) == 1 else None


def prepare_request(public_input, prepared_directory=None, *, arm="tools"):
    if arm == "direct":
        messages = build_direct_messages(public_input)
        return None, messages, [], None, {
            **public_input.identity(), "arm": arm, "tools_sha256": digest([]),
            "initial_messages_sha256": digest(messages),
        }
    if arm != "tools":
        raise ValueError("unknown experiment arm")
    prepared = verified_prepared(public_input, prepared_directory)
    geometry = prepared.geometries[0]
    png, display = render_depth(geometry["depth"], geometry["valid_mask"])
    messages = build_messages(
        public_input, png, width=display["width"], height=display["height"],
        tool_instructions=tool_instructions(public_input.problem.task_type))
    tools = tool_schemas(public_input.problem.task_type)
    identity = {
        **public_input.identity(), "artifact_sha256": prepared.artifact_sha256,
        "display": display, "tools_sha256": digest(tools),
        "initial_messages_sha256": digest(messages),
    }
    return prepared, messages, tools, png, identity


def solve(public_input, prepared_directory, *, client, governor, output_directory,
          stop_event=None, expected_identity=None, arm="tools", resume=False):
    """First non-tool response ends the episode, even if its answer is invalid."""
    from pathlib import Path

    directory = Path(output_directory)
    directory.mkdir(parents=True, exist_ok=True)
    prepared, messages, tools, png, identity = prepare_request(
        public_input, prepared_directory, arm=arm)
    if expected_identity is not None and identity != expected_identity:
        raise ValueError("prepared request changed after preflight")
    problem = public_input.problem
    restored = None
    if resume and (directory / "started.json").exists():
        from .replay import replay_prepared
        if (directory / "result.json").exists():
            raise ValueError("completed result must be skipped by scheduler")
        restored = replay_prepared(
            public_input, directory, prepared=prepared, messages=messages,
            tools=tools, png=png, identity=identity, arm=arm, client=client,
            expected_identity=expected_identity)
        messages = restored["messages"]
    else:
        write_json(directory / "started.json", {
            "protocol": PROTOCOL, "input_identity": identity,
        }, exclusive=True)
        if png is not None:
            (directory / "depth.png").write_bytes(png)
    result = {
        "protocol": PROTOCOL, "arm": arm, "sample_id": problem.sample_id,
        "source_split": problem.source_split, "task_type": problem.task_type,
        "input_identity": identity, "status": "in_progress", "answer": None,
        "request_count": 0, "attempted_tool_calls": 0, "succeeded_tool_calls": 0,
        "tool_seconds": 0.0, "usage": {}, "final_text": None,
        "error_code": None,
    }
    dispatcher = (restored["dispatcher"] if restored else
                  ToolDispatcher(prepared, problem) if arm == "tools" else None)
    seen_call_ids = restored["seen_call_ids"] if restored else set()
    if restored:
        for key in ("request_count", "attempted_tool_calls", "succeeded_tool_calls",
                    "tool_seconds", "usage"):
            result[key] = restored[key]
        result["continuation_verified_requests"] = restored["verified_requests"]
    started = time.monotonic()
    replay_final = restored["final_response"] if restored else None
    while True:
        if replay_final is not None:
            text = replay_final["message"].get("content")
            answer = parse_final_answer(text, problem.options)
            if replay_final["finish_reason"] in {"length", "content_filter"}:
                result.update(status="invalid_answer", final_text=text,
                              error_code="truncated_or_filtered_response")
            else:
                result.update(status="answered" if answer else "invalid_answer",
                              answer=answer, final_text=text)
            break
        if stop_event is not None and stop_event.is_set():
            result["status"] = "cancelled"
            break
        step = result["request_count"]
        try:
            with governor.request("solver", {
                "variant_id": arm, "slot_key": problem.sample_id,
            }) as receipt:
                result["request_count"] += 1
                intent = {
                    "messages_sha256": digest(messages), "tools_sha256": digest(tools),
                    "model_configuration": client.config.public_dict(),
                }
                # Include the actual native-tool payload hash, not only messages.
                if hasattr(client, "payload"):
                    intent["wire_payload_sha256"] = hashlib.sha256(
                        payload_bytes(client.payload(messages, tools))).hexdigest()
                write_json(directory / f"{step:04d}.intent.json", intent, exclusive=True)
                response = client.chat_tools(messages, tools)
                receipt["usage"] = response["usage"]
            write_json(directory / f"{step:04d}.response.json", response, exclusive=True)
            for key, value in response["usage"].items():
                result["usage"][key] = result["usage"].get(key, 0) + value
            message = response["message"]
            calls = message.get("tool_calls") or []
            if response["finish_reason"] in {"length", "content_filter"}:
                result.update(status="invalid_answer", final_text=message.get("content"),
                              error_code="truncated_or_filtered_response")
                break
            if not calls:
                text = message.get("content")
                answer = parse_final_answer(text, problem.options)
                result.update(status="answered" if answer else "invalid_answer",
                              answer=answer, final_text=text)
                break
            if arm == "direct":
                raise ServiceError("unexpected_direct_tool_calls")
            ids = [call["id"] for call in calls]
            if len(set(ids)) != len(ids) or set(ids) & seen_call_ids:
                raise ServiceError("duplicate_tool_call_id")
            seen_call_ids.update(ids)
            # Preserve assistant tool calls and their actual results in the same session.
            messages.append(message)
            traces = []
            for call in calls:
                result["attempted_tool_calls"] += 1
                observation, trace = dispatcher.call(call)
                traces.append(trace)
                result["succeeded_tool_calls"] += int(observation["status"] == "ok")
                result["tool_seconds"] += trace["elapsed_seconds"]
                messages.append({
                    "role": "tool", "tool_call_id": call["id"],
                    "content": json.dumps(observation, ensure_ascii=False, allow_nan=False),
                })
            write_json(directory / f"{step:04d}.tools.json", traces, exclusive=True)
        except ServiceError as exc:
            result.update(status="service_error", error_code=exc.code,
                          http_status=exc.http_status)
            if stop_event is not None:
                stop_event.set()
            break
        except RequestBudgetExhausted:
            result.update(status="budget_exhausted", error_code="global_request_budget")
            if stop_event is not None:
                stop_event.set()
            break
        except Exception as exc:
            # Do not persist exception text, which can include endpoints or headers.
            result.update(status="host_error", error_code=type(exc).__name__)
            if stop_event is not None:
                stop_event.set()
            break
    result["elapsed_seconds"] = round(time.monotonic() - started, 6)
    result["tool_seconds"] = round(result["tool_seconds"], 6)
    result["receipt_sha256"] = digest(result)
    write_json(directory / "result.json", result, exclusive=True)
    return result
