#!/usr/bin/env python3
"""Single durable coordinator; HTTP clients never open the SQLite database."""
from __future__ import annotations

import argparse
import base64
from dataclasses import asdict
import hashlib
import hmac
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import tarfile
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from skill3d.cluster.identity import QueueError, canonical_json, digest_json
from skill3d.cluster.publication import hash_file, inspect_receipt, validate_artifacts
from skill3d.cluster.queue import JobSpec, Lease, SQLiteJobQueue, WorkerIdentity


def atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
    with temp.open("w") as stream:
        json.dump(value, stream, sort_keys=True, allow_nan=False)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temp, path)


def safe_extract(archive: Path, destination: Path, max_bytes: int) -> None:
    total = 0
    with tarfile.open(archive, "r:") as stream:
        seen = set()
        for member in stream:
            relative = PurePosixPath(member.name)
            if (relative.is_absolute() or ".." in relative.parts or
                    not relative.parts or member.name in seen or
                    not (member.isfile() or member.isdir())):
                raise QueueError("unsafe publication archive member")
            seen.add(member.name)
            target = destination.joinpath(*relative.parts)
            if not target.resolve().is_relative_to(destination.resolve()):
                raise QueueError("publication member escapes staging")
            if member.isdir():
                target.mkdir(parents=True, exist_ok=True)
                continue
            total += member.size
            if total > max_bytes:
                raise QueueError("publication exceeds allowed bytes")
            target.parent.mkdir(parents=True, exist_ok=True)
            source = stream.extractfile(member)
            if source is None:
                raise QueueError("cannot read publication member")
            with source, target.open("xb") as output:
                shutil.copyfileobj(source, output, length=4 * 1024 * 1024)
                output.flush()
                os.fsync(output.fileno())


class Coordinator:
    def __init__(self, root: Path, jobs: Path, *, reserve_gib: int = 160,
                 ready_inputs_only: bool = False):
        self.root = root.resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.queue = SQLiteJobQueue(self.root / "state/jobs.sqlite3")
        self.specs = {row["job_key"]: JobSpec.from_dict(row)
                      for row in json.loads(jobs.read_text())["jobs"]}
        self.reserve = reserve_gib * 1024 ** 3
        self.publish_slots = threading.BoundedSemaphore(4)
        self.job_locks = {key: threading.Lock() for key in self.specs}
        if ready_inputs_only:
            self._admitted = set()
            self.admit_ready_inputs()
            threading.Thread(target=self.watch_ready_inputs, daemon=True).start()
        else:
            self.queue.enqueue_many(list(self.specs.values()))

    def admit_ready_inputs(self):
        progress = self.root / "receipts/package_progress.json"
        if not progress.exists():
            return
        receipt = json.loads(progress.read_text())
        for row in receipt["scenes"].values():
            key = row["job_key"]
            if key in self._admitted:
                continue
            if key not in self.specs or receipt["identity_hash"] != self.specs[key].identity.identity_hash:
                raise QueueError("ready input has another frozen identity")
            path = self.input_path(key)
            if path.stat().st_size != row["archive_bytes"]:
                raise QueueError("ready input size changed")
            if hash_file(path) != (row["archive_sha256"], row["archive_bytes"]):
                raise QueueError("ready input digest changed")
            self.queue.enqueue_many([self.specs[key]])
            self._admitted.add(key)

    def watch_ready_inputs(self):
        while len(self._admitted) < len(self.specs):
            try:
                self.admit_ready_inputs()
            except Exception as exc:
                atomic_json(self.root / "receipts/input_admission_failure.json", {
                    "at": time.time(), "error_type": type(exc).__name__,
                    "detail": str(exc)[:1000], "admitted_jobs": len(self._admitted),
                })
                return
            time.sleep(2)

    def publish(self, lease: Lease, archive: Path, archive_sha: str, size: int) -> dict:
        expected = json.loads(lease.output_metadata_json)
        if expected.get("archive_sha256") != archive_sha or expected.get("archive_bytes") != size:
            raise QueueError("archive differs from checkpointed output")
        key = hashlib.sha256(lease.job_key.encode()).hexdigest()
        target = self.root / "results" / key / f"attempt-{lease.attempt}"
        with self.job_locks[lease.job_key]:
            temporary = self.root / ".staging" / uuid.uuid4().hex
            temporary.mkdir(parents=True)
            try:
                if not target.is_dir():
                    safe_extract(archive, temporary, size)
                    publication = inspect_receipt(
                        temporary / "publication.json",
                        expected_receipt_hash=expected["receipt_sha256"])
                    validate_artifacts(publication, temporary)
                    if lease.spec.stage == "gpu_aggregate":
                        from skill3d.cluster.scene_compute import validate_published_scene
                        scene = lease.job_key.split("/")[-2]
                        scope = json.loads(lease.spec.identity.manifest_json)["scenes"][scene]
                        validation = validate_published_scene(
                            temporary, expected_scene_id=scene,
                            expected_qa_ids=scope["qa_ids"],
                            expected_method_config=json.loads(lease.spec.identity.method_config_json))
                        if publication.receipt.validator != "harness3d-scene-qa-validator-v1":
                            raise QueueError("untrusted scene publication validator")
                    target.parent.mkdir(parents=True, exist_ok=True)
                    os.rename(temporary, target)
                    directory_fd = os.open(target.parent, os.O_RDONLY)
                    try:
                        os.fsync(directory_fd)
                    finally:
                        os.close(directory_fd)
                    if lease.spec.stage == "gpu_aggregate":
                        # Receipts pin their destination. Validate again at the
                        # final immutable path before recording durable success.
                        validation = validate_published_scene(
                            target, expected_scene_id=scene, expected_qa_ids=scope["qa_ids"],
                            expected_method_config=json.loads(lease.spec.identity.method_config_json))
                        atomic_json(self.root / "receipts/scene_validation" / (key + ".json"), validation)
                result = self.queue.complete(
                    lease, receipt_path=target / "publication.json",
                    expected_receipt_hash=expected["receipt_sha256"], artifact_root=target)
                return {**asdict(result), "publication_root": str(target.relative_to(self.root))}
            finally:
                shutil.rmtree(temporary, ignore_errors=True)

    def input_path(self, job_key: str) -> Path:
        if job_key not in self.specs:
            raise QueueError("unknown job")
        key = hashlib.sha256(job_key.encode()).hexdigest()
        path = self.root / "inputs/jobs" / (key + ".tar")
        if not path.is_file():
            raise QueueError("frozen job input is not ready")
        return path


def serve(coordinator: Coordinator, host: str, port: int) -> None:
    token = os.environ["HARNESS3D_CLUSTER_TOKEN"]

    class Handler(BaseHTTPRequestHandler):
        def setup(self):
            super().setup()
            self.connection.settimeout(45)

        def log_message(self, *args):
            pass

        def authenticated(self):
            if not hmac.compare_digest(self.headers.get("Authorization", ""), "Bearer " + token):
                self.reply(403, {"error": "unauthorized"})
                return False
            return True

        def reply(self, status, value):
            raw = json.dumps(value, sort_keys=True, allow_nan=False).encode()
            try:
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(raw)))
                self.end_headers()
                self.wfile.write(raw)
            except (BrokenPipeError, ConnectionResetError):
                pass

        def do_GET(self):
            if not self.authenticated():
                return
            if self.path == "/progress":
                return self.reply(200, coordinator.queue.progress_summary())
            return self.reply(404, {"error": "unknown route"})

        def read_json(self):
            size = int(self.headers.get("Content-Length", "0"))
            if not 0 < size <= 1024 * 1024:
                raise QueueError("request size outside limit")
            value = json.loads(self.rfile.read(size))
            if not isinstance(value, dict):
                raise QueueError("request must be an object")
            return value

        def do_POST(self):
            if not self.authenticated():
                return
            try:
                if self.path == "/publish":
                    self.receive_archive()
                    return
                body = self.read_json()
                if self.path == "/claim":
                    lease = coordinator.queue.claim(
                        WorkerIdentity(**body["worker"]), stages=body.get("stages"),
                        lease_seconds=body.get("lease_seconds", 600))
                    return self.reply(200, {"lease": None if lease is None else lease.to_dict()})
                lease = Lease.from_dict(body["lease"])
                if lease.job_key not in coordinator.specs:
                    raise QueueError("lease is outside frozen scope")
                if self.path == "/heartbeat":
                    value = coordinator.queue.heartbeat(lease, lease_seconds=600)
                elif self.path == "/start":
                    value = coordinator.queue.start_attempt(lease)
                elif self.path == "/finish":
                    value = coordinator.queue.finish_attempt(lease, output_metadata=body["metadata"])
                elif self.path == "/fail":
                    coordinator.queue.fail(lease, error_code=body["error_code"],
                                           detail=body["detail"], retry_delay=body.get("retry_delay", 5),
                                           retryable=body.get("retryable", True))
                    return self.reply(200, {"recorded": True})
                elif self.path == "/transfer-failed":
                    coordinator.queue.transfer_failed(
                        lease, phase=body["phase"], error_code=body["error_code"],
                        detail=body["detail"], retry_delay=body.get("retry_delay", 5))
                    return self.reply(200, {"recorded": True})
                elif self.path == "/input":
                    path = coordinator.input_path(lease.job_key)
                    digest, size = hash_file(path)
                    self.send_response(200)
                    self.send_header("Content-Length", str(size))
                    self.send_header("X-Content-SHA256", digest)
                    self.end_headers()
                    with path.open("rb") as stream:
                        shutil.copyfileobj(stream, self.wfile, length=4 * 1024 * 1024)
                    return
                else:
                    return self.reply(404, {"error": "unknown route"})
                self.reply(200, {"lease": value.to_dict()})
            except (QueueError, ValueError, KeyError, OSError) as exc:
                self.reply(409, {"error": type(exc).__name__})

        def receive_archive(self):
            if not coordinator.publish_slots.acquire(timeout=1):
                return self.reply(429, {"error": "publication slots busy"})
            archive = coordinator.root / ".uploads" / (uuid.uuid4().hex + ".tar")
            archive.parent.mkdir(parents=True, exist_ok=True)
            try:
                wire = json.loads(base64.b64decode(self.headers["X-Lease-JSON"], validate=True))
                if wire.get("schema") == "harness3d-lease-header/1":
                    job_key = wire["job_key"]
                    if job_key not in coordinator.specs:
                        raise QueueError("publication is outside frozen scope")
                    job = coordinator.queue.get_job(job_key)
                    lease = Lease(
                        coordinator.specs[job_key], WorkerIdentity(**wire["worker"]),
                        wire["token"], wire["expires_at"], wire["attempt"], "publish",
                        canonical_json(job["output_metadata"]))
                else:
                    lease = Lease.from_dict(wire)
                size = int(self.headers.get("Content-Length", "0"))
                if not 0 < size <= 20 * 1024 ** 3:
                    raise QueueError("archive size outside limit")
                if shutil.disk_usage(coordinator.root).free < 2 * size + coordinator.reserve:
                    raise QueueError("publication reserve would be crossed")
                digest = hashlib.sha256()
                remaining = size
                with archive.open("xb") as output:
                    while remaining:
                        block = self.rfile.read(min(4 * 1024 * 1024, remaining))
                        if not block:
                            raise QueueError("incomplete publication stream")
                        output.write(block)
                        digest.update(block)
                        remaining -= len(block)
                    output.flush()
                    os.fsync(output.fileno())
                actual = digest.hexdigest()
                if actual != self.headers["X-Archive-SHA256"]:
                    raise QueueError("publication archive digest mismatch")
                result = coordinator.publish(lease, archive, actual, size)
                self.reply(200, result)
            finally:
                archive.unlink(missing_ok=True)
                coordinator.publish_slots.release()

    server = ThreadingHTTPServer((host, port), Handler)
    stat = Path("/proc/self/stat")
    atomic_json(coordinator.root / "receipts/coordinator_startup.json", {
        "pid": os.getpid(), "host": host, "port": port, "started_at": time.time(),
        "start_ticks": stat.read_text().split()[21] if stat.is_file() else None,
        "jobs": len(coordinator.specs),
        "scope_sha256": digest_json(sorted(coordinator.specs)),
    })
    server.serve_forever(poll_interval=0.5)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--jobs", type=Path, required=True)
    parser.add_argument("--host", required=True)
    parser.add_argument("--port", type=int, default=27033)
    parser.add_argument("--reserve-gib", type=int, default=160)
    parser.add_argument("--ready-inputs-only", action="store_true")
    args = parser.parse_args()
    serve(Coordinator(args.root, args.jobs, reserve_gib=args.reserve_gib,
                      ready_inputs_only=args.ready_inputs_only), args.host, args.port)


if __name__ == "__main__":
    main()
