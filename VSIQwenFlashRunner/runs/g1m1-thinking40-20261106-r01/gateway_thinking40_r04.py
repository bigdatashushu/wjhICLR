#!/usr/bin/env python3
"""Isolated transport recovery; original model identity and request bytes survive."""

import argparse
import ast
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sqlite3
import subprocess
import sys
import threading
import time
import traceback

OPS = Path("/home/cvailab/wangjh/harness3d/precompute-20261022-r01/ops")
TASK = OPS / "vsi-thinking40-gateway-20261106-r04"
CONFIG = OPS.parent / "perception_config.json"
NATIVE_SHA = "bc527c76ebacc21b5e66549b679e8b61345ffc425869605986b47f51a0bad5cc"
PYTHON = "/home/cvailab/anaconda3/envs/skill3d-exp/bin/python"
PORT = 27451
UPSTREAM_WAIT = 1200
CLIENT_WAIT = 1260
ADDITIONAL_CAP = 152


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_new(path, value):
    with path.open("x") as stream:
        json.dump(value, stream, sort_keys=True, indent=2)
        stream.write("\n")


def identity(pid):
    root = Path("/proc") / str(pid)
    return {
        "pid": pid, "start_ticks": int((root / "stat").read_text().split(") ", 1)[1].split()[19]),
        "cmdline_sha256": sha(root / "cmdline"),
    }


def authorized_environment():
    # Execute only the already-inspected environment assignments, in memory.
    tree = ast.parse((OPS / "launch_perception_service.py").read_text())
    assignments = []
    for node in tree.body:
        if not isinstance(node, ast.Assign) or len(node.targets) != 1:
            continue
        target = node.targets[0]
        if (isinstance(target, ast.Subscript)
                and isinstance(target.value, ast.Attribute)
                and isinstance(target.value.value, ast.Name)
                and target.value.value.id == "os" and target.value.attr == "environ"):
            assignments.append(node)
    assert len(assignments) == 2
    required = {
        node.slice.value for assignment in assignments for node in ast.walk(assignment.value)
        if isinstance(node, ast.Subscript) and isinstance(node.value, ast.Attribute)
        and isinstance(node.value.value, ast.Name) and node.value.value.id == "os"
        and node.value.attr == "environ" and isinstance(node.slice, ast.Constant)
    }
    missing = required - set(os.environ)
    if missing:
        shared = Path("/proc/1959663")
        assert sha(shared / "cmdline") == "22974f31f512874c71705fe1738ff2638f61504374b39cccdf7e7a88241178db"
        assert identity(1959663)["start_ticks"] == 1293307966
        # Retain only explicitly referenced environment values from the verified owner.
        values = dict(entry.split(b"=", 1) for entry in
                      (shared / "environ").read_bytes().split(b"\0") if b"=" in entry)
        for name in missing:
            os.environ[name] = values[name.encode()].decode()
        del values
    module = ast.Module(body=assignments, type_ignores=[])
    exec(compile(module, "<authorized-environment>", "exec"), {"os": os, "Path": Path})


def native_module():
    path = OPS / "cluster_perception_gateway.py"
    assert sha(path) == NATIVE_SHA
    sys.path.insert(0, str(OPS.parent / "source_frozen/src"))
    spec = importlib.util.spec_from_file_location("thinking40_native_gateway", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class LongerWait:
    def __init__(self, delegate):
        self.delegate = delegate

    def __call__(self, request, timeout):
        return self.delegate(request, timeout=UPSTREAM_WAIT)


def wait_preflight(native):
    received = []
    marker = object()
    request = object()

    def fake(value, *, timeout):
        received.append((value, timeout))
        return marker

    assert LongerWait(fake)(request, 180) is marker
    assert received == [(request, 1200)]
    ordinary = native.BailianClient()
    changed = native.BailianClient(opener=LongerWait(fake))
    assert ordinary.audit_identity() == changed.audit_identity()
    return {"model_requests": 0, "same_request_object": True, "same_client_identity": True,
            "single_opener_invocation": True, "actual_upstream_timeout_s": 1200}


def make_gateway(native, config):
    class RecoveryGateway(native.PerceptionGateway):
        def __init__(self):
            def factory(**options):
                client = native.BailianClient(**options)
                client._opener = LongerWait(client._opener)
                return client

            super().__init__(config, TASK / "cache", max_concurrency=8, client_factory=factory)
            assert self._options["model"] == "qwen3.8-flash"
            assert self._options["temperature"] == 0
            assert self._options["max_output_tokens"] == 8192
            assert self._options["timeout_s"] == 180
            self.audit_lock = threading.RLock()
            audit_path = TASK / "cache/transport_audit.sqlite3"
            self.audit = sqlite3.connect(audit_path, check_same_thread=False, timeout=30)
            audit_path.chmod(0o600)
            self.audit.execute("PRAGMA synchronous=FULL")
            with self.audit:
                self.audit.execute(
                    "CREATE TABLE IF NOT EXISTS invocations ("
                    "request_id TEXT PRIMARY KEY, payload_sha256 TEXT, state TEXT, "
                    "started_at REAL, finished_at REAL, http_status INTEGER, "
                    "thinking INTEGER, seed INTEGER, max_tokens INTEGER, n_images INTEGER, "
                    "prompt_tokens INTEGER, completion_tokens INTEGER)")
                self.audit.execute(
                    "UPDATE invocations SET state='unknown' WHERE state='started'")

        def health(self):
            with self.audit_lock:
                counts = dict(self.audit.execute(
                    "SELECT state,COUNT(*) FROM invocations GROUP BY state").fetchall())
            return {
                **super().health(),
                "transport_recovery": {
                    "schema": "thinking40-transport-recovery/1",
                    "service_script_sha256": sha(Path(__file__)),
                    "native_gateway_sha256": NATIVE_SHA,
                    "configured_identity_timeout_s": 180,
                    "actual_upstream_timeout_s": UPSTREAM_WAIT,
                    "actual_client_timeout_s": CLIENT_WAIT,
                    "automatic_model_retry": False, "transport_retries": 0,
                    "maximum_additional_invocations": ADDITIONAL_CAP,
                    "invocation_counts": counts,
                },
            }

        def _validate(self, raw):
            request_id, payload, frozen_identity = super()._validate(raw)
            if (payload["enable_thinking"] is not True or payload.get("seed") != 137
                    or payload["temperature"] != 0 or payload["max_tokens"] != 8192):
                raise native.GatewayError(422, "thinking40_frozen_parameter_mismatch")
            return request_id, payload, frozen_identity

        def _invoke(self, request_id, payload, frozen_identity, payload_hash):
            from skill3d.synthesis.request_audit import describe_request
            request = describe_request(payload, model_identity=frozen_identity)
            with self.audit_lock:
                assert self.audit.execute("SELECT COUNT(*) FROM invocations").fetchone()[0] < ADDITIONAL_CAP
                with self.audit:
                    self.audit.execute(
                        "INSERT INTO invocations VALUES (?,?,?, ?,NULL,NULL,?,?,?, ?,NULL,NULL)",
                        (request_id, payload_hash, "started", time.time(), 1, 137,
                         payload["max_tokens"], len(request["images_in_order"])))
            status, reply = super()._invoke(request_id, payload, frozen_identity, payload_hash)
            usage = json.loads(reply).get("last_usage", {}) if status == 200 else {}
            numeric = lambda key: usage.get(key) if type(usage.get(key)) is int else None
            with self.audit_lock:
                with self.audit:
                    self.audit.execute(
                        "UPDATE invocations SET state=?, finished_at=?, http_status=?, "
                        "prompt_tokens=?, completion_tokens=? WHERE request_id=?",
                        ("completed" if status == 200 else "error", time.time(), status,
                         numeric("prompt_tokens"), numeric("completion_tokens"), request_id))
            return status, reply

    return RecoveryGateway()


def serve():
    authorized_environment()
    native = native_module()
    preflight = wait_preflight(native)
    config = json.loads(CONFIG.read_text())
    gateway = make_gateway(native, config)
    server = native.make_server(gateway, port=PORT, max_connections=128)
    write_new(TASK / "service_start.json", {
        **identity(os.getpid()), "started_at": time.time(),
        "gateway_port": PORT, "model_requests_at_start": 0,
        "transport_preflight": preflight, "health": gateway.health(),
        "public_config_sha256": sha(CONFIG),
    })
    try:
        server.serve_forever()
    finally:
        server.server_close()
        gateway.close()


def launch():
    if (TASK / "service_launch.json").exists():
        prior = json.loads((TASK / "service_launch.json").read_text())
        assert not (Path("/proc") / str(prior["pid"])).exists()
        assert not (TASK / "service_start.json").exists()
        assert not (TASK / "cache").exists()
        archive = TASK / "startup_history" / str(time.time_ns())
        archive.mkdir(mode=0o700, parents=True)
        for name in ("service_launch.json", "service.log"):
            (TASK / name).rename(archive / name)
    import socket
    with socket.socket() as sock:
        assert sock.connect_ex(("127.0.0.1", PORT)) != 0
    env = dict(os.environ)
    for key in list(env):
        if any(word in key.upper() for word in ("TOKEN", "API_KEY", "SECRET", "PASSWORD")):
            env.pop(key)
    env.update(CUDA_VISIBLE_DEVICES="", PYTHONUNBUFFERED="1")
    with (TASK / "service.log").open("xb") as log:
        process = subprocess.Popen(
            [PYTHON, str(Path(__file__).resolve()), "--action", "serve"], env=env,
            stdout=log, stderr=subprocess.STDOUT, start_new_session=True, stdin=subprocess.DEVNULL)
    receipt = {**identity(process.pid), "launched_at": time.time(),
               "service_script_sha256": sha(Path(__file__)), "port": PORT}
    write_new(TASK / "service_launch.json", receipt)
    print(json.dumps(receipt, sort_keys=True), flush=True)


if __name__ == "__main__":
    try:
        parser = argparse.ArgumentParser()
        parser.add_argument("--action", choices=("launch", "serve", "preflight"), required=True)
        action = parser.parse_args().action
        if action == "launch":
            launch()
        elif action == "serve":
            serve()
        else:
            print(json.dumps(wait_preflight(native_module()), sort_keys=True))
    except BaseException as error:
        print(json.dumps({
            "status": "stopped", "error_type": type(error).__name__,
            "stack": [{"file": Path(row.filename).name, "line": row.lineno, "function": row.name}
                      for row in traceback.extract_tb(error.__traceback__)],
        }), flush=True)
        raise SystemExit(1)
