"""Read-only verification of full JSON traces; no model or program execution."""
from __future__ import annotations

import argparse
import base64
import hashlib
import json
from pathlib import Path
import sys

if not __package__:
    root = Path(__file__).resolve().parents[2]
    sys.path[:0] = [str(root / "src"), str(root / "scripts")]
    __package__ = "MindcubeRunner"

from .common import digest, read
from .program import parse_program
from .traces import SCHEMA, TraceError, privacy, response_fields, unwire


def verify(directory):
    directory = Path(directory)
    if directory.is_symlink():
        raise TraceError("trace_symlink")
    events, previous = [], None
    paths = sorted((directory / "events").glob("*.json"))
    for index, path in enumerate(paths):
        if path.is_symlink() or path.name != f"{index:06d}.json":
            raise TraceError("event_sequence_changed")
        event = read(path)
        if (event["sequence"] != index or event["previous_event_sha256"] != previous
                or event["event_sha256"] != digest(
                    {k: v for k, v in event.items() if k != "event_sha256"})):
            raise TraceError("event_hash_changed")
        privacy(event)
        events.append(event)
        previous = event["event_sha256"]
    if not events or events[0]["kind"] != "started":
        raise TraceError("trace_start_missing")
    snapshot = read(directory / "trace.json")
    if snapshot["schema"] != SCHEMA or snapshot["trace_sha256"] != digest(
            {k: v for k, v in snapshot.items() if k != "trace_sha256"}):
        raise TraceError("trace_hash_changed")
    # An interruption between event commit and snapshot commit is recoverable.
    if snapshot["events"] != events[:len(snapshot["events"])]:
        raise TraceError("snapshot_event_mismatch")
    messages, client, gateway, program, response = None, None, None, None, None
    image_hashes, terminal, admitted, sent = [], False, False, False
    for event in events:
        kind, data = event["kind"], event["data"]
        if terminal:
            raise TraceError("event_after_terminal")
        if kind == "prepared":
            messages = data["messages"]
            if digest(messages) != data["receipt"]["messages_sha256"]:
                raise TraceError("messages_hash_changed")
            for message in messages:
                for part in message["content"]:
                    if part["type"] == "image_url":
                        url = part["image_url"]["url"]
                        if not url.startswith("data:image/"):
                            raise TraceError("external_image_in_trace")
                        raw = base64.b64decode(url.split(",", 1)[1], validate=True)
                        image_hashes.append(hashlib.sha256(raw).hexdigest())
            if image_hashes != data["receipt"]["image_sha256"]:
                raise TraceError("image_order_or_bytes_changed")
        elif kind == "admitted":
            if messages is None:
                raise TraceError("admitted_before_prepared")
            admitted = True
        elif kind in {"client_payload", "gateway_envelope", "response_body"}:
            raw = unwire(data["body"])
            if kind == "client_payload":
                if not admitted:
                    raise TraceError("payload_before_admission")
                client = json.loads(raw)
                if client["messages"] != messages:
                    raise TraceError("wire_messages_changed")
            elif kind == "gateway_envelope":
                gateway = json.loads(raw)
                if gateway["payload"] != client:
                    raise TraceError("gateway_payload_changed")
            else:
                if not sent:
                    raise TraceError("response_before_send")
                if data["fields"] != response_fields(
                        raw, complete=data["complete"], http_status=data["http_status"]):
                    raise TraceError("response_fields_changed")
                response = data["fields"]
        elif kind == "request_sent":
            if gateway is None:
                raise TraceError("send_before_envelope")
            sent = True
        elif kind == "program":
            program = data["source"]
            if hashlib.sha256(program.encode()).hexdigest() != data["utf8_sha256"]:
                raise TraceError("program_hash_changed")
            if response and response["content"]["presence"] == "present":
                if parse_program(response["content"]["value"]) != program:
                    raise TraceError("program_response_mismatch")
        elif kind == "terminal":
            terminal = True
    if not terminal and not (directory / "orphan.json").exists():
        status = "incomplete"
    else:
        status = "terminal" if terminal else "interrupted"
    return {"trace": str(directory), "status": status, "events": len(events),
            "images": len(image_hashes), "wire_request_verified": client is not None,
            "response_observed": response is not None,
            "provider_body_observed": bool(response and response["provider_body_observed"]),
            "program_preserved": program is not None, "model_requests": 0,
            "programs_executed": 0, "last_event_sha256": previous}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path")
    args = parser.parse_args()
    path = Path(args.path)
    directories = [path] if (path / "events").is_dir() else sorted(
        path.glob("episodes/*/*/traces/*"))
    if not directories:
        raise TraceError("no_traces_found")
    print(json.dumps({"schema": "mindcube-trace-replay/1",
                      "traces": [verify(p) for p in directories]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
