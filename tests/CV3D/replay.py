"""Reconstruct a saved native-tool session without issuing model requests."""
import hashlib
import json
from pathlib import Path

from skill3d.synthesis.request_audit import payload_bytes

from .common import digest, read_json
from .tools import ToolDispatcher


def restore_assistant(message):
    """Restore the original client field order, lost by sorted receipt storage."""
    clean = {"role": "assistant", "content": message.get("content")}
    if "reasoning_content" in message:
        clean["reasoning_content"] = message["reasoning_content"]
    if message.get("tool_calls"):
        clean["tool_calls"] = [{
            "id": call["id"], "type": "function",
            "function": {
                "name": call["function"]["name"],
                "arguments": call["function"]["arguments"],
            },
        } for call in message["tool_calls"]]
    assert clean == message
    return clean


def replay_session(public, directory, prepared_directory, *, arm, client, expected_identity):
    from .inference import prepare_request

    directory = Path(directory)
    prepared, messages, tools, png, identity = prepare_request(
        public, prepared_directory, arm=arm)
    return replay_prepared(
        public, directory, prepared=prepared, messages=messages, tools=tools, png=png,
        identity=identity, arm=arm, client=client, expected_identity=expected_identity)


def replay_prepared(public, directory, *, prepared, messages, tools, png,
                    identity, arm, client, expected_identity):
    directory = Path(directory)
    assert identity == expected_identity
    assert read_json(directory / "started.json")["input_identity"] == identity
    dispatcher = ToolDispatcher(prepared, public.problem) if arm == "tools" else None
    calls_seen = set()
    state = {
        "request_count": 0, "attempted_tool_calls": 0, "succeeded_tool_calls": 0,
        "tool_seconds": 0.0, "usage": {}, "final_response": None,
    }
    for ordinal, path in enumerate(sorted(directory.glob("*.intent.json"))):
        step = f"{ordinal:04d}"
        assert path.name == step + ".intent.json"
        intent = read_json(path)
        assert intent["model_configuration"] == client.config.public_dict()
        assert intent["messages_sha256"] == digest(messages)
        wire = payload_bytes(client.payload(messages, tools))
        assert hashlib.sha256(wire).hexdigest() == intent["wire_payload_sha256"]
        response = read_json(directory / (step + ".response.json"))
        assert response["model"] == client.config.model
        state["request_count"] += 1
        for key, value in response["usage"].items():
            state["usage"][key] = state["usage"].get(key, 0) + value
        message = restore_assistant(response["message"])
        calls = message.get("tool_calls") or []
        if not calls or response["finish_reason"] in {"length", "content_filter"}:
            state["final_response"] = response
            assert ordinal + 1 == len(list(directory.glob("*.intent.json")))
            break
        assert dispatcher is not None
        traces = read_json(directory / (step + ".tools.json"))
        assert len(traces) == len(calls)
        messages.append(message)
        for call, trace in zip(calls, traces):
            assert call["id"] == trace["tool_call_id"] and call["id"] not in calls_seen
            calls_seen.add(call["id"])
            # The numerical tools are deterministic. Replaying them restores the
            # exact original JSON string order, and verifies the saved value.
            observation, _ = dispatcher.call(call)
            assert observation == trace["response"]
            messages.append({
                "role": "tool", "tool_call_id": call["id"],
                "content": json.dumps(observation, ensure_ascii=False, allow_nan=False),
            })
            state["attempted_tool_calls"] += 1
            state["succeeded_tool_calls"] += int(observation["status"] == "ok")
            state["tool_seconds"] += trace["elapsed_seconds"]
    state.update(
        prepared=prepared, messages=messages, tools=tools, png=png,
        identity=identity, dispatcher=dispatcher, seen_call_ids=calls_seen,
        verified_requests=state["request_count"],
    )
    return state
