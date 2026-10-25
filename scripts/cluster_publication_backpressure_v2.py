#!/usr/bin/env python3
"""Transport-only v2 around the unchanged frozen coordinator.

Eight bounded publication disk spools feed four FIFO slots. Input keepalives
cover ONLY the wait for final headers, not body stalls. Every input is still
hashed by the frozen handler; no digest cache or receipt-seeded hash bypass.
"""
from __future__ import annotations

import argparse
import base64
from collections import deque
import hashlib
from http.server import ThreadingHTTPServer
import json
import math
import os
from pathlib import Path
import shutil
import sys
import threading
import time
import uuid


PATHS = {
    "spool": ".publication_spool_v2",
    "events": "receipts/publication_fifo_events_v2.jsonl",
    "prepared": "receipts/publication_fifo_prepared_v2.json",
    "activation": "receipts/publication_fifo_activate_v2.json",
    "startup": "receipts/coordinator_fifo_startup_v2.json",
}


class FifoSlots:
    """Same bounded FIFO publication admission as v1."""

    def __init__(self, capacity=4):
        self.capacity = capacity
        self.active = 0
        self.waiters = deque()
        self.condition = threading.Condition()

    def acquire(self, timeout=None):
        ticket = object()
        with self.condition:
            self.waiters.append(ticket)
            while self.waiters[0] is not ticket or self.active >= self.capacity:
                self.condition.wait()
            self.waiters.popleft()
            self.active += 1
            self.condition.notify_all()
            return True

    def release(self):
        with self.condition:
            if self.active <= 0:
                raise ValueError("publication slot released without admission")
            self.active -= 1
            self.condition.notify_all()


class Evidence:
    def __init__(self, path):
        self.path = Path(path)
        self.lock = threading.Lock()

    def write(self, event, **fields):
        raw = json.dumps({"schema": "harness3d-publication-fifo-event/2",
                          "at": time.time(), "event": event, **fields},
                         sort_keys=True, allow_nan=False).encode() + b"\n"
        with self.lock:
            fd = os.open(self.path, os.O_WRONLY | os.O_APPEND | os.O_CREAT |
                         os.O_NOFOLLOW, 0o600)
            with os.fdopen(fd, "ab") as stream:
                stream.write(raw)
                stream.flush()
                os.fsync(stream.fileno())

    def observe(self, event, **fields):
        # Input instrumentation must not turn a successful response into 409.
        try:
            self.write(event, **fields)
        except OSError as exc:
            print("v2 input evidence unavailable: " + type(exc).__name__,
                  file=sys.stderr, flush=True)


class InterimResponse:
    """Serialize interim bytes against the FIRST final response emission."""

    def __init__(self, handler, seconds):
        self.handler = handler
        self.seconds = seconds
        self.lock = threading.RLock()
        self.stop = threading.Event()
        self.thread = None
        self.request_at = time.time()
        self.started = time.monotonic()
        self.fields = {"request_id": uuid.uuid4().hex}
        self.final_status = None
        self.final_response_at = None
        self.final_response_seconds = None
        self.headers_at = None
        self.headers_seconds = None
        self.first_body_at = None
        self.first_body_seconds = None
        self.body_bytes = 0
        self.interim_count = 0
        self.last_interim_seconds = None
        self.interim_error_type = None
        self.ack = None
        self.observations = []

    def start(self):
        self.thread = threading.Thread(target=self.run, daemon=True)
        self.thread.start()

    def observe(self, event, **fields):
        self.observations.append((event, {**self.timing(), **fields}))

    def run(self):
        while not self.stop.wait(self.seconds):
            with self.lock:
                if self.stop.is_set():
                    return
                try:
                    self.handler.wfile.write(b"HTTP/1.1 100 Continue\r\n\r\n")
                    self.handler.wfile.flush()
                    self.interim_count += 1
                    self.last_interim_seconds = time.monotonic() - self.started
                except OSError as exc:
                    self.interim_error_type = type(exc).__name__
                    return

    def final(self, status):
        self.stop.set()
        if self.final_status is None:
            self.final_status = status
            self.final_response_at = time.time()
            self.final_response_seconds = time.monotonic() - self.started

    def close(self):
        with self.lock:
            self.stop.set()
        if self.thread is not None:
            self.thread.join()

    def timing(self):
        return {
            **self.fields, "request_at": self.request_at,
            "final_status": self.final_status,
            "final_response_at": self.final_response_at,
            "pre_header_seconds": self.final_response_seconds,
            "headers_at": self.headers_at, "headers_seconds": self.headers_seconds,
            "first_body_at": self.first_body_at,
            "first_body_seconds": self.first_body_seconds,
            "body_bytes_written": self.body_bytes,
            "interim_count": self.interim_count,
            "last_interim_seconds": self.last_interim_seconds,
            "interim_error_type": self.interim_error_type,
        }


class InputWriter:
    """Observe writes without buffering, framing, modifying or hashing bytes."""

    def __init__(self, stream, state):
        self.stream = stream
        self.state = state

    def __getattr__(self, name):
        return getattr(self.stream, name)

    def write(self, data):
        state = self.state
        is_body = state.headers_at is not None
        if is_body and data and state.first_body_at is None:
            state.first_body_at = time.time()
            state.first_body_seconds = time.monotonic() - state.started
        result = self.stream.write(data)
        if is_body:
            state.body_bytes += len(data) if result is None else result
        return result


def adapt_handler(handler, coordinator, evidence, *, interim_seconds=15):
    """Subclass, never replace frozen route/receive/publish/queue validators."""
    if (isinstance(interim_seconds, bool) or not math.isfinite(interim_seconds)
            or interim_seconds <= 0):
        raise ValueError("interim interval must be positive and finite")
    spool_slots = threading.BoundedSemaphore(8)
    spool_root = coordinator.root / PATHS["spool"]
    spool_root.mkdir(mode=0o700, exist_ok=True)
    if any(spool_root.iterdir()):
        raise RuntimeError("previous v2 publication spool needs explicit review")
    coordinator.publish_slots = FifoSlots(4)

    class AdaptedHandler(handler):
        def send_response(self, status, message=None):
            state = (getattr(self, "_input_state", None) or
                     getattr(self, "_publication_state", None))
            if state is None:
                return super().send_response(status, message)
            with state.lock:
                # Stop before even the status line is buffered. The interim
                # writer uses this same lock, including its flush.
                state.final(status)
                return super().send_response(status, message)

        def send_header(self, keyword, value):
            result = super().send_header(keyword, value)
            state = getattr(self, "_input_state", None)
            if state is not None and state.headers_at is None:
                if keyword.lower() == "x-content-sha256":
                    state.fields["content_sha256"] = value
                elif keyword.lower() == "content-length":
                    state.fields["content_length"] = value
            return result

        def end_headers(self):
            result = super().end_headers()
            state = getattr(self, "_input_state", None)
            if state is not None and state.headers_at is None:
                state.headers_at = time.time()
                state.headers_seconds = time.monotonic() - state.started
                state.observe("input_headers_sent")
            return result

        def read_json(self):
            state = getattr(self, "_input_state", None)
            if state is not None:
                # The original do_POST calls this only AFTER authentication.
                state.start()
            body = super().read_json()
            if state is not None:
                lease = body.get("lease")
                if isinstance(lease, dict) and isinstance(lease.get("spec"), dict):
                    key = lease["spec"].get("job_key")
                    if isinstance(key, str) and key in coordinator.specs:
                        state.fields["job_key"] = key
                        attempt = lease.get("attempt")
                        if type(attempt) is int and 0 <= attempt <= coordinator.specs[key].max_attempts:
                            state.fields["attempt"] = attempt
                state.observe("input_request_parsed",
                              parsed_seconds=time.monotonic() - state.started)
            return body

        def do_POST(self):
            if self.path != "/input":
                return super().do_POST()
            state = InterimResponse(self, interim_seconds)
            self._input_state = state
            original_output = self.wfile
            self.wfile = InputWriter(original_output, state)
            state.observe("input_request_received")
            error_type = None
            try:
                return super().do_POST()
            except BaseException as exc:
                error_type = type(exc).__name__
                raise
            finally:
                state.close()
                self.wfile = original_output
                finished = time.monotonic() - state.started
                finished_at = time.time()
                # All timestamps were sampled in the actual phase; defer disk
                # I/O so timing evidence cannot stall input headers or body.
                for event, fields in state.observations:
                    evidence.observe(event, **fields)
                evidence.observe(
                    "input_request_finished", **state.timing(),
                    finished_at=finished_at, finished_seconds=finished,
                    post_header_seconds=(None if state.headers_seconds is None else
                                         finished - state.headers_seconds),
                    finish_phase=("pre_header" if state.headers_at is None else
                                  "body" if state.first_body_at is not None else "headers"),
                    error_type=error_type)
                self._input_state = None

        def reply(self, status, value):
            state = getattr(self, "_publication_state", None)
            if state is None:
                return super().reply(status, value)
            with state.lock:
                state.final(status)
                state.ack = {k: value.get(k) for k in
                             ("status", "job_key", "completed_key", "receipt_hash",
                              "fingerprint", "publication_root", "error") if k in value}
                return super().reply(status, value)

        def receive_archive(self):
            state = InterimResponse(self, interim_seconds)
            self._publication_state = state
            fields = state.fields
            temporary = spool_root / (fields["request_id"] + ".tar")
            original_input = self.rfile
            acquired = False
            state.start()
            try:
                size = int(self.headers.get("Content-Length", "0"))
                if not 0 < size <= 20 * 1024**3:
                    raise ValueError("archive size outside limit")
                if shutil.disk_usage(coordinator.root).free < 3 * size + coordinator.reserve:
                    raise ValueError("publication spool reserve would be crossed")
                wire_raw = base64.b64decode(self.headers["X-Lease-JSON"], validate=True)
                wire = json.loads(wire_raw)
                fields.update(lease_header_sha256=hashlib.sha256(wire_raw).hexdigest(),
                              job_key=wire.get("job_key"), attempt=wire.get("attempt"),
                              archive_bytes=size,
                              declared_archive_sha256=self.headers["X-Archive-SHA256"])
                spool_slots.acquire()
                acquired = True
                evidence.write("spool_started", **fields)
                digest = hashlib.sha256()
                remaining = size
                with temporary.open("xb") as output:
                    while remaining:
                        block = original_input.read(min(4 * 1024**2, remaining))
                        if not block:
                            raise ValueError("incomplete publication spool")
                        output.write(block)
                        digest.update(block)
                        remaining -= len(block)
                    output.flush()
                    os.fsync(output.fileno())
                if digest.hexdigest() != fields["declared_archive_sha256"]:
                    raise ValueError("publication spool digest mismatch")
                evidence.write("spool_sealed", **fields,
                               actual_archive_sha256=digest.hexdigest())
                with temporary.open("rb") as stream:
                    self.rfile = stream
                    super().receive_archive()
                evidence.write("publication_finished", **fields,
                               final_status=state.final_status, ack=state.ack,
                               interim_count=state.interim_count)
            except Exception as exc:
                evidence.write("transport_or_validation_error", **fields,
                               error_type=type(exc).__name__)
                raise
            finally:
                self.rfile = original_input
                state.close()
                temporary.unlink(missing_ok=True)
                if acquired:
                    spool_slots.release()

    return AdaptedHandler


def build_warm_coordinator(frozen, root, jobs, *, reserve_gib=160):
    """Original constructor, existing-only admission under one RO snapshot.

    Called only before any server threads exist. The temporary module binding
    is restored even on failure. All subsequent queue APIs are the frozen ones.
    """
    base_queue = frozen.SQLiteJobQueue
    queue_module = sys.modules[base_queue.__module__]
    database = Path(root).resolve() / "state/jobs.sqlite3"
    if not database.is_file():
        raise frozen.QueueError("warm admission requires an existing queue")
    started = time.monotonic()

    class ExistingQueue(base_queue):
        def __init__(self, db_path):
            self.db_path = queue_module._local_database_path(db_path)
            self._clock = time.time
            self._timeout = self._positive(30.0, "busy_timeout")

        def enqueue_many(self, specs):
            results = []
            with self._connect(readonly=True) as db:
                db.execute("BEGIN")
                if (db.execute("PRAGMA journal_mode").fetchone()[0].lower() != "wal" or
                        db.execute("PRAGMA user_version").fetchone()[0] != 1 or
                        db.execute("PRAGMA application_id").fetchone()[0] !=
                        int.from_bytes(b"S3DQ", "big")):
                    raise frozen.QueueError("warm admission requires the frozen WAL queue")
                for spec in specs:
                    if not isinstance(spec, frozen.JobSpec):
                        raise frozen.QueueError("enqueue requires JobSpec values")
                    spec.__post_init__()
                    body = frozen.canonical_json(spec.to_dict())
                    old = db.execute("SELECT * FROM jobs WHERE job_key=?",
                                     (spec.job_key,)).fetchone()
                    if old is None:
                        raise frozen.QueueError("warm admission cannot add a missing job")
                    self._verify_dependencies(db, old)
                    if old["spec_json"] != body:
                        raise queue_module.EnqueueConflict(
                            "job key already frozen with another specification")
                    results.append(queue_module.EnqueueResult(
                        spec.job_key, False, old["status"]))
                if db.total_changes != 0:
                    raise frozen.QueueError("warm admission unexpectedly mutated the queue")
                db.rollback()
            return results

    frozen.SQLiteJobQueue = ExistingQueue
    try:
        coordinator = frozen.Coordinator(Path(root), Path(jobs), reserve_gib=reserve_gib)
    finally:
        frozen.SQLiteJobQueue = base_queue
    coordinator.queue.enqueue_many = base_queue.enqueue_many.__get__(coordinator.queue, base_queue)
    coordinator.warm_admission = {
        "existing_jobs_verified": len(coordinator.specs), "readonly_snapshot": True,
        "queue_rows_changed": 0, "queue_events_added": 0, "write_transactions": 0,
        "seconds": time.monotonic() - started,
        "runtime_enqueue_restored": True,
    }
    return coordinator


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def validate_activation(command, root, *, method_sha256, adapter_sha256, warm_pid):
    """Consume the main operator's handover; never stop or restart workers."""
    require(command["status"] == "old_coordinator_exited", "old coordinator not retired")
    require(command["method_config_sha256"] == method_sha256, "activation method mismatch")
    require(command["adapter_sha256"] == adapter_sha256, "activation adapter mismatch")
    require(command["warm_pid"] == warm_pid, "activation warm PID mismatch")
    old_pid = command["old_pid"]
    require(type(old_pid) is int and old_pid > 0 and old_pid != warm_pid,
            "invalid old coordinator PID")
    require(Path("/proc/self/stat").is_file(), "activation requires Linux process evidence")
    require(not (Path("/proc") / str(old_pid)).exists(), "old coordinator still exists")
    for name in (".uploads", ".staging", ".publication_spool_v1", PATHS["spool"]):
        path = Path(root) / name
        require(not path.exists() or (path.is_dir() and not any(path.iterdir())),
                "handover requires empty " + name)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--expected-method-sha256", required=True)
    parser.add_argument("--port", type=int, default=27049)
    args = parser.parse_args()
    import cluster_coordinator as frozen

    root = args.root.resolve()
    method_raw = (root / "method_config.json").read_bytes()
    require(hashlib.sha256(method_raw).hexdigest() == args.expected_method_sha256,
            "method configuration differs from the frozen expected hash")
    method = json.loads(method_raw)
    for name, expected in method["code_sha256"].items():
        require(hashlib.sha256((root / "source_frozen" / name).read_bytes()).hexdigest() == expected,
                "frozen code hash mismatch: " + name)
    require(Path(frozen.__file__).resolve() == root / "source_frozen/scripts/cluster_coordinator.py",
            "coordinator import is not from source_frozen")
    require(Path(sys.modules[frozen.SQLiteJobQueue.__module__].__file__).resolve() ==
            root / "source_frozen/src/skill3d/cluster/queue.py",
            "queue import is not from source_frozen")
    for name in ("prepared", "activation", "startup", "events"):
        require(not (root / PATHS[name]).exists(), "v2 path already exists: " + PATHS[name])
    adapter_sha = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    jobs = root / "coordinator_jobs.json"
    jobs_sha = hashlib.sha256(jobs.read_bytes()).hexdigest()
    db_stat = (root / "state/jobs.sqlite3").stat()
    coordinator = build_warm_coordinator(frozen, root, jobs)
    prepared = root / PATHS["prepared"]
    frozen.atomic_json(prepared, {
        "schema": "harness3d-publication-fifo-prepared/2", "status": "prepared",
        "at": time.time(), "pid": os.getpid(),
        "method_config_sha256": args.expected_method_sha256,
        "adapter_sha256": adapter_sha, "jobs_sha256": jobs_sha,
        "code_files": len(method["code_sha256"]), "publication_capacity": 4,
        "spool_capacity": 8, "input_interim_seconds": 15,
        "input_keepalive_phase": "before_final_response_only", "input_digest_cache": False,
        "frozen_validators_delegated": True, "warm_admission": coordinator.warm_admission,
        "model_requests": 0, "label_access": False,
    })
    activation = root / PATHS["activation"]
    while not activation.exists():
        time.sleep(0.5)
    command = json.loads(activation.read_text())
    validate_activation(command, root, method_sha256=args.expected_method_sha256,
                        adapter_sha256=adapter_sha, warm_pid=os.getpid())
    require(hashlib.sha256(jobs.read_bytes()).hexdigest() == jobs_sha, "jobs changed after warm")
    current_stat = (root / "state/jobs.sqlite3").stat()
    require((db_stat.st_dev, db_stat.st_ino) == (current_stat.st_dev, current_stat.st_ino),
            "queue database replaced after warm")
    evidence = Evidence(root / PATHS["events"])

    class AdaptedServer(ThreadingHTTPServer):
        def __init__(self, address, handler):
            super().__init__(address, adapt_handler(handler, coordinator, evidence))

    original_atomic = frozen.atomic_json
    original_server = frozen.ThreadingHTTPServer

    def startup_json(path, value):
        if Path(path) == root / "receipts/coordinator_startup.json":
            path = root / PATHS["startup"]
            require(not path.exists(), "v2 startup receipt already exists")
            value = {**value, "schema": "harness3d-publication-fifo-coordinator-startup/2",
                     "method_config_sha256": args.expected_method_sha256,
                     "adapter_sha256": adapter_sha, "publication_capacity": 4,
                     "spool_capacity": 8, "input_interim_seconds": 15,
                     "original_startup_preserved": True}
        original_atomic(path, value)

    frozen.ThreadingHTTPServer = AdaptedServer
    frozen.atomic_json = startup_json
    try:
        frozen.serve(coordinator, "0.0.0.0", args.port)
    finally:
        frozen.ThreadingHTTPServer = original_server
        frozen.atomic_json = original_atomic


if __name__ == "__main__":
    main()
