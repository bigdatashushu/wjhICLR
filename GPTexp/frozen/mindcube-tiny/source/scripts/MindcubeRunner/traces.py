"""Durable, self-contained JSON records at the runner's observed boundaries."""
from __future__ import annotations

import base64
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import time
import uuid

from .common import digest, encoded, read

SCHEMA = "mindcube-full-json-trace/1"
POLICY = {"schema": SCHEMA, "images": "inline-original-base64",
          "wire_bytes": "base64", "events": "atomic-fsync-hash-chain",
          "provider_failure_boundary": "gateway-only-if-upstream-body-not-forwarded",
          "reference_answer_visible": False, "automatic_retry": False}
FORBIDDEN = {"gt_answer", "ground_truth", "reference_answer", "api_key",
             "transport_token", "password"}


class TraceError(RuntimeError):
    pass


def plain(value):
    """Serialize only explicit data types; never introspect arbitrary objects."""
    if value is None or type(value) in (str, bool, int):
        return value
    if type(value) is float:
        if math.isfinite(value):
            return value
        return {"nonfinite_float": str(value)}
    if isinstance(value, (list, tuple)):
        return [plain(v) for v in value]
    if type(value) is dict:
        if any(type(k) is not str for k in value):
            return {"mapping_entries": [[plain(k), plain(v)] for k, v in value.items()]}
        return {k: plain(v) for k, v in value.items()}
    # Numerical tool arguments may be NumPy scalars or arrays.
    import numpy as np
    if isinstance(value, np.generic):
        return plain(value.item())
    if isinstance(value, np.ndarray):
        return {"ndarray": plain(value.tolist()), "dtype": str(value.dtype),
                "shape": list(value.shape)}
    raise TraceError("unsupported_trace_data_type")


def privacy(value, secrets=(), *, check_keys=True):
    if isinstance(value, dict):
        for key, item in value.items():
            if check_keys and key.lower() in FORBIDDEN:
                raise TraceError("private_trace_field")
            privacy(item, secrets, check_keys=check_keys)
    elif isinstance(value, list):
        for item in value:
            privacy(item, secrets, check_keys=check_keys)
    elif isinstance(value, str):
        if any(secret and secret in value for secret in secrets):
            raise TraceError("credential_in_trace")


def atomic(path, value, *, exclusive=False):
    path = Path(path)
    raw = encoded(value) + b"\n"
    temporary = path.with_name("." + path.name + "." + uuid.uuid4().hex + ".tmp")
    fd = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(raw)
            stream.flush()
            os.fsync(stream.fileno())
        if exclusive:
            os.link(temporary, path)
            temporary.unlink()
        else:
            os.replace(temporary, path)
        directory_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        if temporary.exists():
            temporary.unlink()


def storage_check(directory):
    directory = Path(directory)
    if shutil.disk_usage(directory).free < 512 * 1024**2:
        raise TraceError("trace_storage_insufficient")
    probe = directory / (".trace-check-" + uuid.uuid4().hex + ".json")
    atomic(probe, {"schema": SCHEMA})
    probe.unlink()
    return dict(POLICY)


def wire(raw):
    return {"encoding": "base64", "bytes": len(raw),
            "sha256": hashlib.sha256(raw).hexdigest(),
            "base64": base64.b64encode(raw).decode("ascii")}


def unwire(value):
    raw = base64.b64decode(value["base64"], validate=True)
    if len(raw) != value["bytes"] or hashlib.sha256(raw).hexdigest() != value["sha256"]:
        raise TraceError("wire_hash_changed")
    return raw


def response_fields(raw, *, complete=True, http_status=200):
    unknown = {"presence": "not_observed_at_boundary"}
    try:
        body = json.loads(raw)
    except (ValueError, UnicodeError):
        body = None
    provider = (complete and http_status == 200 and isinstance(body, dict)
                and isinstance(body.get("choices"), list) and bool(body["choices"])
                and isinstance(body["choices"][0], dict)
                and isinstance(body["choices"][0].get("message"), dict))
    def field(mapping, key):
        if not provider:
            return dict(unknown)
        return ({"presence": "present", "value": mapping[key]} if key in mapping else
                {"presence": "absent_in_observed_provider_body"})
    choice = body["choices"][0] if provider else {}
    message = choice.get("message", {})
    reasoning = []
    def visit(value, path):
        if isinstance(value, dict):
            for key, item in value.items():
                current = [*path, key]
                if "reasoning" in key.lower() or key.lower() in {"thinking", "thoughts"}:
                    reasoning.append({"path": current, "presence": "present", "value": item})
                visit(item, current)
        elif isinstance(value, list):
            for index, item in enumerate(value):
                visit(item, [*path, index])
    if provider:
        visit(body, [])
    return {"provider_body_observed": bool(provider),
            "observation_boundary": "forwarded_provider_body" if provider else "gateway_only",
            "assistant_message": field(choice, "message"), "content": field(message, "content"),
            "reasoning_content": field(message, "reasoning_content"),
            "reasoning_fields": reasoning, "finish_reason": field(choice, "finish_reason"),
            **{key: field(body or {}, key) for key in
               ("id", "model", "created", "system_fingerprint", "usage")}}


class TraceWriter:
    def __init__(self, directory, identity, *, secrets=()):
        self.directory = Path(directory)
        self.directory.mkdir(mode=0o700, parents=True, exist_ok=False)
        (self.directory / "events").mkdir(mode=0o700)
        self.identity = plain(identity)
        privacy(self.identity, secrets)
        self.secrets = tuple(secrets)
        self.events, self.failed = [], False
        self.started = time.monotonic()
        self.event("started", {"identity": self.identity, "policy": POLICY})

    def event(self, kind, data):
        try:
            data = plain(data)
            privacy(data, self.secrets)
            event = {"sequence": len(self.events), "kind": kind, "data": data,
                     "utc_epoch": time.time(), "elapsed_seconds": time.monotonic() - self.started,
                     "previous_event_sha256": self.events[-1]["event_sha256"] if self.events else None}
            event["event_sha256"] = digest(event)
            atomic(self.directory / "events" / f"{len(self.events):06d}.json", event,
                   exclusive=True)
            self.events.append(event)
            snapshot = {"schema": SCHEMA, "identity": self.identity, "events": self.events,
                        "terminal": kind == "terminal", "reference_answer_visible": False}
            snapshot["trace_sha256"] = digest(snapshot)
            atomic(self.directory / "trace.json", snapshot)
        except BaseException:
            self.failed = True
            raise

    def body(self, kind, raw, **metadata):
        # Scan decoded bytes before base64 makes a secret invisible to the checker.
        privacy(raw.decode("utf-8", errors="replace"), self.secrets)
        data = {"body": wire(raw), **metadata}
        if kind == "response_body":
            data["fields"] = response_fields(raw, complete=metadata.get("complete", True),
                                            http_status=metadata.get("http_status", 200))
        self.event(kind, data)

    def finish(self, status, state, execution):
        self.event("terminal", {"status": status, "state_after": state,
                                "execution": execution})


def mark_orphans(directory, reason):
    """Parent records incomplete rounds without inventing a response or retry."""
    for path in sorted(Path(directory).glob("traces/*")):
        events = sorted((path / "events").glob("*.json"))
        if (events and read(events[-1])["kind"] != "terminal"
                and not (path / "orphan.json").exists()):
            atomic(path / "orphan.json", {"schema": SCHEMA, "status": "interrupted",
                   "reason": reason, "last_event_sha256": read(events[-1])["event_sha256"],
                   "automatic_retry": False}, exclusive=True)
