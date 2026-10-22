#!/usr/bin/env python3
"""GPU scene worker with durable output checkpoints and independent heartbeats."""
from __future__ import annotations

import argparse
import base64
import hashlib
import http.client
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import tarfile
import threading
import time
import urllib.error
import urllib.parse
import urllib.request

from skill3d.cluster import ArtifactDigest, Lease, PublicationReceipt, hash_file
from cluster_coordinator import atomic_json, safe_extract


class TransportError(RuntimeError):
    pass


class CoordinatorRejected(TransportError):
    pass


class Client:
    def __init__(self, endpoint: str, token: str):
        self.endpoint = endpoint.rstrip("/")
        self.token = token

    def request(self, route: str, body: dict, *, attempts: int = 4) -> dict:
        raw = json.dumps(body, sort_keys=True, allow_nan=False).encode()
        for attempt in range(attempts):
            req = urllib.request.Request(
                self.endpoint + route, raw,
                headers={"Authorization": "Bearer " + self.token,
                         "Content-Type": "application/json"})
            try:
                with urllib.request.urlopen(req, timeout=90) as response:
                    return json.load(response)
            except urllib.error.HTTPError as exc:
                if exc.code == 409:
                    raise CoordinatorRejected("coordinator rejected " + route) from exc
                if exc.code not in (429, 500, 502, 503, 504):
                    raise CoordinatorRejected("coordinator HTTP failure") from exc
            except (urllib.error.URLError, OSError, TimeoutError):
                pass
            if attempt + 1 < attempts:
                time.sleep(min(5, attempt + 1))
        raise TransportError("coordinator request unavailable")

    def download(self, lease: Lease, path: Path) -> None:
        req = urllib.request.Request(
            self.endpoint + "/input", json.dumps({"lease": lease.to_dict()}).encode(),
            headers={"Authorization": "Bearer " + self.token})
        temp = path.with_suffix(".part")
        try:
            with urllib.request.urlopen(req, timeout=90) as response, temp.open("wb") as output:
                expected = response.headers["X-Content-SHA256"]
                size = int(response.headers["Content-Length"])
                digest = hashlib.sha256()
                written = 0
                while block := response.read(4 * 1024 * 1024):
                    output.write(block)
                    digest.update(block)
                    written += len(block)
                output.flush()
                os.fsync(output.fileno())
            if written != size or digest.hexdigest() != expected:
                raise TransportError("input transport digest mismatch")
            os.replace(temp, path)
        finally:
            temp.unlink(missing_ok=True)

    def publish(self, lease: Lease, archive: Path, metadata: dict) -> dict:
        endpoint = urllib.parse.urlsplit(self.endpoint)
        cls = http.client.HTTPSConnection if endpoint.scheme == "https" else http.client.HTTPConnection
        connection = cls(endpoint.hostname, endpoint.port, timeout=120)
        try:
            connection.putrequest("POST", endpoint.path.rstrip("/") + "/publish")
            connection.putheader("Authorization", "Bearer " + self.token)
            connection.putheader("Content-Length", str(metadata["archive_bytes"]))
            connection.putheader("Content-Type", "application/x-tar")
            connection.putheader("X-Archive-SHA256", metadata["archive_sha256"])
            connection.putheader("X-Lease-JSON", base64.b64encode(
                json.dumps({
                    "schema": "harness3d-lease-header/1", "job_key": lease.job_key,
                    "worker": lease.worker.to_dict(), "token": lease.token,
                    "expires_at": lease.expires_at, "attempt": lease.attempt,
                }).encode()).decode())
            connection.endheaders()
            with archive.open("rb") as stream:
                while block := stream.read(4 * 1024 * 1024):
                    connection.send(block)
            response = connection.getresponse()
            raw = response.read(1024 * 1024)
            if response.status != 200:
                error = (TransportError if response.status in (429, 500, 502, 503, 504)
                         else CoordinatorRejected)
                raise error("publication rejected with HTTP " + str(response.status))
            result = json.loads(raw)
            if result.get("status") != "complete" or result.get("job_key") != lease.job_key:
                raise TransportError("invalid durable publication ACK")
            return result
        finally:
            connection.close()


class Heartbeat:
    interval = 25

    def __init__(self, client: Client, lease: Lease):
        self.client, self.lease = client, lease
        self.stop_event = threading.Event()
        self.failed = False
        self.lock = threading.Lock()
        self.pending = None
        self.thread = threading.Thread(target=self.run, daemon=True)

    def __enter__(self):
        self.thread.start()
        return self

    def run(self):
        while not self.stop_event.wait(self.interval):
            try:
                renewed = Lease.from_dict(self.client.request(
                    "/heartbeat", {"lease": self.lease.to_dict()})["lease"])
                with self.lock:
                    # A delayed heartbeat ACK must not overwrite a newer phase.
                    if (renewed.attempt >= self.lease.attempt and
                            (renewed.phase == "publish" or self.lease.phase != "publish")):
                        self.lease = renewed
                    if self.pending is not None:
                        path, metadata = self.pending
                        save_pending(path, self.lease, metadata)
            except TransportError:
                self.failed = True

    def retain(self, lease: Lease, path: Path, metadata: dict):
        with self.lock:
            self.lease = lease
            self.pending = (path, metadata)
            save_pending(path, lease, metadata)

    def clear_pending(self):
        with self.lock:
            self.pending = None

    def __exit__(self, *args):
        self.stop_event.set()
        self.thread.join(timeout=60)


def local_path(root: Path, locator: str) -> Path:
    relative = PurePosixPath(locator)
    if relative.is_absolute() or ".." in relative.parts or not relative.parts:
        raise ValueError("unsafe checkpoint locator")
    result = root.joinpath(*relative.parts)
    if not result.resolve().is_relative_to(root.resolve()):
        raise ValueError("checkpoint escapes worker root")
    return result


def checkpoint(root: Path, output: Path, lease: Lease) -> dict:
    artifacts = []
    for path in sorted(output.rglob("*")):
        if path.is_symlink():
            raise ValueError("output symlinks are forbidden")
        if path.is_file() and path.name != "publication.json":
            digest, size = hash_file(path)
            artifacts.append(ArtifactDigest(str(path.relative_to(output)), digest, size))
    receipt = PublicationReceipt(
        lease.job_key, lease.spec.completed_key, lease.spec.identity.identity_hash,
        lease.spec.stage, lease.attempt, "harness3d-scene-qa-validator-v1",
        tuple(artifacts))
    atomic_json(output / "publication.json", receipt.to_dict())
    archive = output.parent / "output.tar"
    temp = archive.with_suffix(".part")
    with tarfile.open(temp, "w:") as stream:
        for path in sorted(output.rglob("*")):
            if path.is_file():
                stream.add(path, arcname=str(path.relative_to(output)), recursive=False)
    with temp.open("rb") as stream:
        os.fsync(stream.fileno())
    os.replace(temp, archive)
    digest, size = hash_file(archive)
    metadata = {
        "output_locator": str(archive.relative_to(root)), "archive_sha256": digest,
        "archive_bytes": size, "receipt_sha256": hash_file(output / "publication.json")[0],
    }
    atomic_json(output.parent / "checkpoint.json", metadata)
    return metadata


def save_pending(path: Path, lease: Lease, metadata: dict) -> None:
    atomic_json(path, {
        "schema": "harness3d-pending-checkpoint/1",
        "lease": lease.to_dict(), "metadata": metadata,
    })
    directory_fd = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)


def block_pending(path: Path, detail: str) -> dict:
    pending = json.loads(path.read_text())
    pending["blocked"] = detail
    atomic_json(path, pending)
    return {"status": "checkpoint_blocked", "job_key": pending["lease"]["spec"]["job_key"],
            "output_locator": pending["metadata"]["output_locator"], "error": detail}


def finish_with_retry(client: Client, lease: Lease, metadata: dict) -> Lease:
    """Keep the running lease and its heartbeat until finish is acknowledged."""
    while True:
        try:
            return Lease.from_dict(client.request(
                "/finish", {"lease": lease.to_dict(), "metadata": metadata})["lease"])
        except CoordinatorRejected:
            raise
        except TransportError:
            time.sleep(5)


def publish_with_retry(client: Client, lease: Lease, archive: Path, metadata: dict,
                       *, attempts: int = 4) -> dict:
    for attempt in range(attempts):
        # Recheck outside the retry handler: corrupt retained output is not retried.
        if hash_file(archive) != (metadata["archive_sha256"], metadata["archive_bytes"]):
            raise TransportError("retained output checkpoint changed")
        try:
            return client.publish(lease, archive, metadata)
        except CoordinatorRejected:
            raise
        except (TransportError, OSError, ValueError, http.client.HTTPException):
            if attempt + 1 == attempts:
                raise
            time.sleep(min(5, attempt + 1))
    raise ValueError("publication attempts must be positive")


def recover_pending(client: Client, root: Path, worker: dict, reserve: int) -> dict | None:
    """Resume only this worker's stored lease, never invent a replacement lease."""
    for path in sorted((root / "jobs").glob("*/attempt-*/pending.json")):
        try:
            pending = json.loads(path.read_text())
        except FileNotFoundError:
            # Another worker may have cleaned its acknowledged output meanwhile.
            continue
        if pending.get("blocked") or pending["lease"]["worker"] != worker:
            continue
        lease = Lease.from_dict(pending["lease"])
        if pending.get("schema") != "harness3d-pending-checkpoint/1":
            return block_pending(path, "unknown pending checkpoint schema")
        if lease.phase != "publish":
            try:
                lease = Lease.from_dict(client.request(
                    "/heartbeat", {"lease": lease.to_dict()})["lease"])
            except CoordinatorRejected:
                return block_pending(path, "stored lease expired or fenced; retained output "
                                     "cannot be checkpointed without a live lease")
        # Accepted publication retries are idempotent even after lease expiry.
        return run_job(client, root, lease, reserve)
    return None


def run_job(client: Client, root: Path, lease: Lease, reserve: int) -> dict:
    key = hashlib.sha256(lease.job_key.encode()).hexdigest()
    staging = root / "jobs" / key
    staging.mkdir(parents=True, exist_ok=True)
    with Heartbeat(client, lease) as heartbeat:
        if lease.phase != "publish":
            pending_path = staging / f"attempt-{lease.attempt}" / "pending.json"
            if pending_path.is_file():
                pending = json.loads(pending_path.read_text())
                stored = Lease.from_dict(pending["lease"])
                if (stored.spec, stored.worker, stored.token, stored.attempt) != (
                        lease.spec, lease.worker, lease.token, lease.attempt):
                    return block_pending(pending_path, "pending checkpoint belongs to another lease")
                metadata = pending["metadata"]
                archive = local_path(root, metadata["output_locator"])
                if hash_file(archive) != (metadata["archive_sha256"], metadata["archive_bytes"]):
                    return block_pending(pending_path, "retained output checkpoint changed")
            else:
                archive = staging / "input.tar"
                inputs = staging / "input"
                try:
                    if not archive.is_file():
                        client.download(lease, archive)
                    if not inputs.is_dir():
                        temp = staging / "input.unpacking"
                        shutil.rmtree(temp, ignore_errors=True)
                        temp.mkdir()
                        safe_extract(archive, temp, archive.stat().st_size)
                        os.rename(temp, inputs)
                    if shutil.disk_usage(root).free < reserve + 4 * archive.stat().st_size:
                        raise TransportError("worker disk reserve would be crossed")
                except (TransportError, OSError, urllib.error.URLError, ValueError,
                        tarfile.TarError) as exc:
                    if isinstance(exc, tarfile.TarError):
                        archive.unlink(missing_ok=True)
                        shutil.rmtree(staging / "input.unpacking", ignore_errors=True)
                    client.request("/transfer-failed", {
                        "lease": lease.to_dict(), "phase": "prefetch",
                        "error_code": "input_transfer_failed", "detail": type(exc).__name__})
                    return {"status": "prefetch_retry", "job_key": lease.job_key}
                lease = Lease.from_dict(client.request("/start", {"lease": lease.to_dict()})["lease"])
                output = staging / f"attempt-{lease.attempt}" / "output"
                output.mkdir(parents=True, exist_ok=True)
                try:
                    from skill3d.cluster.scene_compute import compute_scene
                    from skill3d.cluster.perception_client import RemoteBailianClient

                    method = json.loads(lease.spec.identity.method_config_json)
                    result = compute_scene(
                        inputs, output, method,
                        client_factory=lambda *args, **kwargs: RemoteBailianClient(**kwargs))
                    atomic_json(output / "scene_result.json", result)
                    metadata = checkpoint(root, output, lease)
                except Exception as exc:
                    client.request("/fail", {
                        "lease": lease.to_dict(), "error_code": "scene_compute_failed",
                        "detail": type(exc).__name__, "retryable": True, "retry_delay": 15})
                    atomic_json(staging / f"attempt-{lease.attempt}" / "failure.json", {
                        "job_key": lease.job_key, "error": type(exc).__name__,
                        "message": str(exc)[:1500], "at": time.time()})
                    return {"status": "model_retry", "job_key": lease.job_key,
                            "error": type(exc).__name__}
                pending_path = output.parent / "pending.json"
            heartbeat.retain(lease, pending_path, metadata)
            try:
                lease = finish_with_retry(client, lease, metadata)
            except CoordinatorRejected as exc:
                heartbeat.clear_pending()
                return block_pending(pending_path, str(exc))
        metadata = json.loads(lease.output_metadata_json)
        pending_path = staging / f"attempt-{lease.attempt}" / "pending.json"
        heartbeat.retain(lease, pending_path, metadata)
        archive = local_path(root, metadata["output_locator"])
        try:
            ack = publish_with_retry(client, lease, archive, metadata)
        except CoordinatorRejected as exc:
            heartbeat.clear_pending()
            return block_pending(pending_path, str(exc))
        except (TransportError, OSError, ValueError, http.client.HTTPException) as exc:
            client.request("/transfer-failed", {
                "lease": lease.to_dict(), "phase": "return",
                "error_code": "output_transfer_failed", "detail": type(exc).__name__})
            heartbeat.clear_pending()
            pending_path.unlink(missing_ok=True)
            return {"status": "publish_retry", "job_key": lease.job_key}
        atomic_json(root / "acks" / (key + ".json"), ack)
        heartbeat.clear_pending()
        # All eight workers share this root. A reclaimed publishing lease reads
        # the original archive; deletion only follows the durable commit ACK.
        shutil.rmtree(staging)
        return {"status": "complete", "job_key": lease.job_key}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--endpoint", required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--worker-id", required=True)
    parser.add_argument("--host-id", default="3090")
    parser.add_argument("--gpu-id", required=True)
    parser.add_argument("--reserve-gib", type=int, default=100)
    parser.add_argument("--max-jobs", type=int, default=0)
    args = parser.parse_args()
    root = args.root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    client = Client(args.endpoint, os.environ["HARNESS3D_CLUSTER_TOKEN"])
    worker = {"worker_id": args.worker_id, "host": args.host_id, "gpu_id": args.gpu_id}
    done = 0
    while not args.max_jobs or done < args.max_jobs:
        try:
            result = recover_pending(client, root, worker, args.reserve_gib * 1024 ** 3)
            if result is not None:
                print(json.dumps(result), flush=True)
                if result["status"] == "complete":
                    done += 1
                continue
            # Claim is not request-idempotent; never retry an uncertain claim.
            payload = client.request("/claim", {
                "worker": worker, "lease_seconds": 600, "stages": ["gpu_aggregate"]},
                attempts=1)
            if payload["lease"] is None:
                print(json.dumps({"status": "idle", "worker_id": args.worker_id}), flush=True)
                if args.max_jobs:
                    return
                time.sleep(10)
                continue
            lease = Lease.from_dict(payload["lease"])
            print(json.dumps({"status": "claimed", "job_key": lease.job_key,
                              "phase": lease.phase, "worker_id": args.worker_id}), flush=True)
            result = run_job(client, root, lease, args.reserve_gib * 1024 ** 3)
            print(json.dumps(result), flush=True)
            if result["status"] == "complete":
                done += 1
        except TransportError as exc:
            print(json.dumps({"status": "transport_unavailable", "error": str(exc)}), flush=True)
            time.sleep(10)


if __name__ == "__main__":
    main()
