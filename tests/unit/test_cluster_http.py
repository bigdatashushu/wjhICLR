"""Real HTTP transport checks without GPU or model requests."""
import base64
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import tarfile
import time
import urllib.error
import urllib.request

import pytest

from skill3d.cluster.identity import FrozenIdentity
from skill3d.cluster.publication import ArtifactDigest, PublicationReceipt
from skill3d.cluster.queue import JobSpec

ROOT = Path(__file__).resolve().parents[2]


def unused_port():
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def request(endpoint, route, body=None, *, headers=None, raw=None):
    supplied = {"Authorization": "Bearer integration-test-only"}
    supplied.update(headers or {})
    data = raw if raw is not None else None if body is None else json.dumps(body).encode()
    req = urllib.request.Request(endpoint + route, data=data, headers=supplied)
    with urllib.request.urlopen(req, timeout=10) as response:
        return response.read()


def wait_ready(proc, endpoint, route):
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            raise AssertionError(proc.stderr.read().decode())
        try:
            request(endpoint, route)
            return
        except urllib.error.URLError:
            time.sleep(0.05)
    raise AssertionError("HTTP service did not start")


def stop(proc):
    proc.terminate()
    try:
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait(timeout=5)


def test_transfer_verified_resume_and_conflict(tmp_path):
    source = tmp_path / "source.bin"
    source.write_bytes(b"frozen exact payload" * 1000)
    row = {"source": str(source), "relative": "arrays/source.bin",
           "size": source.stat().st_size, "sha256": hashlib.sha256(source.read_bytes()).hexdigest()}
    manifest = tmp_path / "export.json"
    manifest.write_text(json.dumps({"files": [row]}))
    port = unused_port()
    endpoint = f"http://127.0.0.1:{port}"
    env = {**os.environ, "HARNESS3D_TRANSFER_TOKEN": "integration-test-only"}
    proc = subprocess.Popen([sys.executable, str(ROOT / "scripts/cluster_transfer.py"),
                             "serve", "--manifest", str(manifest), "--host", "127.0.0.1",
                             "--port", str(port), "--startup", str(tmp_path / "startup.json")],
                            env=env, stderr=subprocess.PIPE)
    spec = importlib.util.spec_from_file_location("cluster_transfer_test", ROOT / "scripts/cluster_transfer.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    try:
        wait_ready(proc, endpoint, "/manifest")
        with pytest.MonkeyPatch.context() as patch:
            patch.setenv("HARNESS3D_TRANSFER_TOKEN", "integration-test-only")
            args = (endpoint, tmp_path / "target", tmp_path / "receipt.json")
            first = module.fetch(*args, workers=2, reserve=0)
            second = module.fetch(*args, workers=2, reserve=0)
            assert first["status"] == second["status"] == "verified"
            assert second["verified_bytes"] == row["size"]
            target = tmp_path / "target/arrays/source.bin"
            target.write_bytes(b"tampered")
            failed = module.fetch(*args, workers=1, reserve=0)
            assert failed["status"] == "incomplete"
            assert target.read_bytes() == b"tampered"
    finally:
        stop(proc)


def test_coordinator_publish_ack_restart_and_digest_rejection(tmp_path):
    root = tmp_path / "coordinator"
    root.mkdir()
    identity = FrozenIdentity.freeze({"scene": "s", "qa_ids": ["q"]}, {"method": "frozen"})
    job = JobSpec("scene-s/reconstruction", "scene-s/reconstruction/done", identity, "reconstruction")
    jobs = tmp_path / "jobs.json"
    jobs.write_text(json.dumps({"jobs": [job.to_dict()]}))
    input_path = root / "inputs/jobs" / (hashlib.sha256(job.job_key.encode()).hexdigest() + ".tar")
    input_path.parent.mkdir(parents=True)
    input_path.write_bytes(b"frozen input")
    port = unused_port()
    endpoint = f"http://127.0.0.1:{port}"
    env = {**os.environ, "PYTHONPATH": str(ROOT / "src"),
           "HARNESS3D_CLUSTER_TOKEN": "integration-test-only"}
    command = [sys.executable, str(ROOT / "scripts/cluster_coordinator.py"),
               "--root", str(root), "--jobs", str(jobs), "--host", "127.0.0.1", "--port", str(port)]
    proc = subprocess.Popen(command, env=env, stderr=subprocess.PIPE)
    try:
        wait_ready(proc, endpoint, "/progress")
        claim = json.loads(request(endpoint, "/claim", {"worker": {
            "worker_id": "cpu-check", "host": "local", "gpu_id": "cpu"}}))["lease"]
        assert request(endpoint, "/input", {"lease": claim}) == b"frozen input"
        lease = json.loads(request(endpoint, "/start", {"lease": claim}))["lease"]
        output = tmp_path / "result"
        output.mkdir()
        (output / "result.json").write_text('{"status":"complete"}')
        data = (output / "result.json").read_bytes()
        receipt = PublicationReceipt(job.job_key, job.completed_key, identity.identity_hash,
                                     job.stage, lease["attempt"], "model-free-http-integration",
                                     (ArtifactDigest("result.json", hashlib.sha256(data).hexdigest(), len(data)),))
        (output / "publication.json").write_text(json.dumps(receipt.to_dict(), sort_keys=True))
        archive_path = tmp_path / "output.tar"
        with tarfile.open(archive_path, "w:") as stream:
            for path in sorted(output.iterdir()):
                stream.add(path, arcname=path.name)
        archive = archive_path.read_bytes()
        archive_sha = hashlib.sha256(archive).hexdigest()
        metadata = {"archive_sha256": archive_sha, "archive_bytes": len(archive),
                    "receipt_sha256": hashlib.sha256((output / "publication.json").read_bytes()).hexdigest()}
        lease = json.loads(request(endpoint, "/finish", {"lease": lease, "metadata": metadata}))["lease"]
        headers = {"X-Lease-JSON": base64.b64encode(json.dumps(lease).encode()).decode(),
                   "X-Archive-SHA256": "0" * 64}
        with pytest.raises(urllib.error.HTTPError) as error:
            request(endpoint, "/publish", headers=headers, raw=archive)
        assert error.value.code == 409
        # Tiny test publications need no production disk reserve.
        loader = importlib.util.spec_from_file_location("coordinator_test", ROOT / "scripts/cluster_coordinator.py")
        module = importlib.util.module_from_spec(loader)
        loader.loader.exec_module(module)
        stop(proc)
        script = "from cluster_coordinator import Coordinator,serve; from pathlib import Path; serve(Coordinator(Path(" + repr(str(root)) + "),Path(" + repr(str(jobs)) + "),reserve_gib=0),'127.0.0.1'," + str(port) + ")"
        env["PYTHONPATH"] = str(ROOT / "src") + os.pathsep + str(ROOT / "scripts")
        proc = subprocess.Popen([sys.executable, "-c", script], env=env, stderr=subprocess.PIPE)
        wait_ready(proc, endpoint, "/progress")
        headers["X-Archive-SHA256"] = archive_sha
        first = json.loads(request(endpoint, "/publish", headers=headers, raw=archive))
        repeated = json.loads(request(endpoint, "/publish", headers=headers, raw=archive))
        assert not first["duplicate"] and repeated["duplicate"]
        progress = json.loads(request(endpoint, "/progress"))
        assert progress["counts"]["complete"] == 1
        assert progress["model_attempts"] == 1
        assert json.loads(request(endpoint, "/claim", {"worker": {
            "worker_id": "another", "host": "local", "gpu_id": "cpu"}}))["lease"] is None
    finally:
        if proc.poll() is None:
            stop(proc)
