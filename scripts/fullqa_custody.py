#!/usr/bin/env python3
"""Durable server custody around the unchanged fullQA worker and queue.

Delivery is request-idempotent. A lost transport retries the same lease. A dead
computing process is quarantined; it never implicitly becomes another model
attempt. Journals contain operational identity only, not the process environment.
"""
from __future__ import annotations

import contextlib
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
import urllib.error
import urllib.request

METHOD = "b89bf2353210174360941afe95f64665a7349b4f26bdafd7033412b90121d22e"
WORKER = "fc0f4c7390c8e33fef0f27af2e1d15203e576f44c65d43b71ae57c9e80e3d716"
QUEUE = "49a3fe57227bc3c836039445c5b54aadd939f09361d068271530f86afabb3b9b"
IDENTITY = "e4ead2f7b7d1b3259fea4053b44dffcb7b23fc07f0820cb810444a9845a287e6"
COLLECTOR = "4ff0458253f548d034ea99015e7bc9da57d13aa5a365aec96c88969295b7d0d0"
GIB = 1024 ** 3
ACTIVE = ("leased", "running", "publishing")


class Refused(RuntimeError):
    pass


def require(condition, code):
    if not condition:
        raise Refused(code)


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def canon(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=True, allow_nan=False).encode()


def read(path):
    return json.loads(Path(path).read_bytes())


def durable(path, value, *, immutable=False):
    """fsync file and directory; an immutable name accepts only identical bytes."""
    return durable_bytes(path, canon(value) + b"\n", immutable=immutable)


def durable_bytes(path, raw, *, immutable=False):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    temporary = path.parent / (path.name + "." + str(os.getpid()) + "." +
                               str(time.time_ns()) + ".part")
    fd = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(raw)
            stream.flush()
            os.fsync(stream.fileno())
        if immutable:
            try:
                os.link(temporary, path)
            except FileExistsError:
                require(path.read_bytes() == raw, "immutable_record_changed")
        else:
            os.replace(temporary, path)
        fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)
    finally:
        temporary.unlink(missing_ok=True)
    return digest(raw)


def process(pid):
    directory = Path("/proc") / str(pid)
    try:
        fields = (directory / "stat").read_text().rsplit(")", 1)[1].split()
        return {"pid": int(pid), "start_ticks": int(fields[19]),
                "cmdline_sha256": digest((directory / "cmdline").read_bytes()),
                "boot_id": Path("/proc/sys/kernel/random/boot_id").read_text().strip(),
                "state": fields[0]}
    except FileNotFoundError:
        return None


def live(identity):
    current = process(identity["pid"])
    return current is not None and current["state"] != "Z" and all(
        current[k] == identity[k] for k in
        ("pid", "start_ticks", "cmdline_sha256", "boot_id") if k in identity)


def load_pinned(name, path, sha256):
    raw = Path(path).read_bytes()
    require(digest(raw) == sha256, "source_pin_changed")
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    exec(compile(raw, str(path), "exec"), module.__dict__)
    return module


def pins(root):
    require(digest((root / "method_config.json").read_bytes()) == METHOD, "method_changed")
    require(digest((root / "source_frozen/scripts/cluster_worker.py").read_bytes()) == WORKER,
            "worker_changed")
    require(digest((root / "source_frozen/src/skill3d/cluster/queue.py").read_bytes()) == QUEUE,
            "queue_changed")


def http(endpoint, route, body, *, timeout=20):
    request = urllib.request.Request(endpoint.rstrip("/") + route, canon(body),
        headers={"Authorization": "Bearer " + os.environ["HARNESS3D_CLUSTER_TOKEN"],
                 "Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.load(response)


def ro_database(root):
    connection = sqlite3.connect((root / "state/jobs.sqlite3").as_uri() + "?mode=ro",
                                uri=True, timeout=10)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA query_only=ON")
    return connection


def admission(db, worker_id, now):
    """Run inside the native claim's BEGIN IMMEDIATE, before native _settle."""
    require(not db.execute(
        "SELECT 1 FROM jobs WHERE status IN ('leased','running','publishing') "
        "AND lease_until<=? LIMIT 1", (now + 120,)).fetchone(), "expired_or_expiring_custody")
    require(not db.execute(
        "SELECT 1 FROM jobs WHERE status IN ('leased','running','publishing') "
        "AND worker_id=? LIMIT 1", (worker_id,)).fetchone(), "worker_already_owned")
    require(not db.execute(
        "SELECT 1 FROM jobs WHERE status='queued' AND "
        "(attempts<>0 OR claims<>0 OR resume_phase<>'prefetch') LIMIT 1"
    ).fetchone(), "historical_retry_requires_review")


def native_claim(root, worker):
    """Use unmodified native claim, fencing any side-effecting settlement."""
    from skill3d.cluster.queue import SQLiteJobQueue, WorkerIdentity
    queue = object.__new__(SQLiteJobQueue)
    queue.db_path = (root / "state/jobs.sqlite3").resolve()
    queue._clock, queue._timeout = time.time, 10
    original = queue._write

    @contextlib.contextmanager
    def guarded():
        with original() as db:
            admission(db, worker["worker_id"], time.time())
            yield db

    queue._write = guarded
    return queue.claim(WorkerIdentity(**worker), lease_seconds=21600,
                       stages=["gpu_aggregate"])


def recovery_action(state, child_live, pending_exists, ack_exists):
    """The single place that decides whether another process can be launched."""
    if ack_exists:
        return "copy_ack"
    if child_live:
        return "observe"
    if state in {"delivered", "launching"}:
        return "launch"
    if pending_exists and state in {"running", "publishing", "starting"}:
        return "resume_publication"
    return "quarantine"


def recover_delivery(row, lease, stored, intent, worker):
    """Resolve a lost claim reply without issuing another non-idempotent claim."""
    if stored is not None and stored["lease"]["token"] == lease.token:
        require(stored["lease"]["spec"] == lease.spec.to_dict(), "current_delivery_fenced")
        return stored
    require(intent is not None and intent.get("phase") == "claiming"
            and intent["worker"] == worker
            and row["status"] == "leased" and row["attempts"] == 0
            and row["claims"] == 1 and row["updated_at"] >= intent["at"],
            "unowned_active_slot")
    return None


def run_child(config, delivery_path):
    root, home, cache = map(Path, (config["root"], config["home"], config["cache"]))
    pins(root)
    sys.path[:0] = [str(root / "source_frozen/src"), str(root / "source_frozen/scripts")]
    native = load_pinned("cluster_worker", root / "source_frozen/scripts/cluster_worker.py", WORKER)
    delivery = read(delivery_path)
    from skill3d.cluster.queue import Lease
    lease = Lease.from_dict(delivery["lease"])
    require(lease.spec.identity.identity_hash == IDENTITY, "delivery_identity_changed")
    require(lease.worker.to_dict() == config["worker"], "delivery_worker_changed")
    state_path = Path(delivery_path).with_suffix(".state.json")
    lock = threading.RLock()
    prior = read(state_path) if state_path.exists() else {}
    state = {**prior, "job_key": lease.job_key, "process": process(os.getpid()), "at": time.time()}

    def update(phase):
        with lock:
            state.update(phase=phase, at=time.time())
            durable(state_path, state)

    if prior.get("phase") in {"starting", "running", "publishing"}:
        # Only a retained checkpoint can be resumed after the model started.
        key = digest(lease.job_key.encode())
        pending = list((cache / "jobs" / key).glob("attempt-*/pending.json"))
        require(len(pending) == 1, "dead_compute_without_checkpoint")
        stored = read(pending[0])
        require(not stored.get("blocked"), "checkpoint_already_blocked")
        lease = Lease.from_dict(stored["lease"])
    else:
        update("delivered")

    class RetainedTransport(native.Client):
        def request(self, route, body, **kwargs):
            require(route not in {"/claim", "/fail", "/transfer-failed"}, "no_implicit_model_retry")
            if route == "/start":
                update("starting")
            while True:
                try:
                    result = super().request(route, body, attempts=1)
                    if route == "/start":
                        update("running")
                    elif route == "/finish":
                        update("publishing")
                    return result
                except native.CoordinatorRejected:
                    raise
                except native.TransportError:
                    time.sleep(5)

        def download(self, current, destination):
            while True:
                try:
                    return super().download(current, destination)
                except (native.TransportError, OSError, urllib.error.URLError):
                    time.sleep(5)

        def publish(self, current, archive, metadata):
            while True:
                if native.hash_file(archive) != (metadata["archive_sha256"], metadata["archive_bytes"]):
                    raise native.CoordinatorRejected("checkpoint_changed")
                try:
                    return super().publish(current, archive, metadata)
                except native.CoordinatorRejected:
                    raise
                except (native.TransportError, OSError, native.http.client.HTTPException):
                    time.sleep(5)

    try:
        if prior.get("phase") not in {"starting", "running", "publishing"}:
            # Fetch while an independent heartbeat retains this exact lease.
            # Decode failures remain quarantined and never return to claim().
            key = digest(lease.job_key.encode())
            staging = cache / "jobs" / key
            staging.mkdir(parents=True, exist_ok=True)
            client = RetainedTransport(config["coordinator"],
                                      os.environ["HARNESS3D_CLUSTER_TOKEN"])
            with native.Heartbeat(client, lease):
                archive = staging / "input.tar"
                if not archive.exists():
                    client.download(lease, archive)
                if not (staging / "input").exists():
                    temp = staging / "input.unpacking"
                    require(not temp.exists(), "partial_input_retained")
                    temp.mkdir()
                    native.safe_extract(archive, temp, 20 * GIB)
                    os.rename(temp, staging / "input")
        result = native.run_job(RetainedTransport(config["coordinator"],
            os.environ["HARNESS3D_CLUSTER_TOKEN"]), cache, lease, 52 * GIB)
        require(result.get("status") == "complete", "native_worker_did_not_complete")
        key = digest(lease.job_key.encode())
        raw = (cache / "acks" / (key + ".json")).read_bytes()
        durable_bytes(root / "worker_cache/acks" / (key + ".json"), raw, immutable=True)
        update("complete")
    except BaseException as exc:
        with lock:
            state.update(phase="quarantined", error_type=type(exc).__name__, at=time.time())
            durable(state_path, state)
        raise SystemExit(1)


def probe_source(source, arguments):
    result = subprocess.run([sys.executable, "-B", "-", *map(str, arguments)],
        input=source.encode(), stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, timeout=120)
    require(result.returncode == 0, "native_probe_rejected")
    return json.loads(result.stdout)
