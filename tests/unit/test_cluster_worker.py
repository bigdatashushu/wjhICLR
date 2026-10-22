"""CPU-only worker recovery checks using real loopback HTTP and publication.

Most cases use the real non-GPU publication path to isolate worker behavior.
The GPU semantic publication path has its own regression below.
"""

from __future__ import annotations

import base64
import hashlib
from http.server import ThreadingHTTPServer
import importlib.util
import io
import json
from pathlib import Path
import shutil
import socket
import sys
import tarfile
import threading
import time
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from skill3d.cluster import FrozenIdentity, JobSpec, Lease, SQLiteJobQueue, hash_file
from skill3d.cluster import scene_compute
from skill3d.cluster.publication import inspect_receipt, validate_artifacts


ROOT = Path(__file__).resolve().parents[2]
TOKEN = "local-worker-test-only"
SCENE = "scene-cpu"
QA_IDS = ["qa-0", "qa-1"]
METHOD = {"runtime_config": {}, "min_free_gib": 0, "seed": 7}
PAYLOAD = b"CPU semantic fixture; no model calls"


def load_script(name):
    spec = importlib.util.spec_from_file_location(
        name + "_worker_test", ROOT / "scripts" / (name + ".py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def write_tar(path, name="scene.json", content=None):
    if content is None:
        content = json.dumps({"scene_id": SCENE, "qa_ids": QA_IDS}).encode()
    with tarfile.open(path, "w:") as stream:
        member = tarfile.TarInfo(name)
        member.size = len(content)
        stream.addfile(member, io.BytesIO(content))


def never(*_args, **_kwargs):
    raise AssertionError("unexpected model or recompute operation")


class Clock:
    def __init__(self):
        self.now = time.time()

    def __call__(self):
        return self.now

    def advance(self, seconds):
        self.now += seconds


class WorkerCase:
    def __init__(self, worker, coordinator, jobs, root, clock):
        self.worker = worker
        self.coordinator = coordinator
        self.jobs = jobs
        self.root = root
        self.clock = clock
        self.endpoint = None
        self.routes = []
        self.http_errors = []
        self.publish_headers = []
        self.publish_acks = []
        self.heartbeat_seen = threading.Event()
        self.faults = {
            "input": None, "reject": {}, "reject_count": {},
            "drop_before": set(), "drop_ack": {},
        }
        self.key = hashlib.sha256(next(iter(coordinator.specs)).encode()).hexdigest()

    @property
    def queue(self):
        return self.coordinator.queue

    @property
    def job_key(self):
        return next(iter(self.coordinator.specs))

    @property
    def staging(self):
        return self.root / "jobs" / self.key

    @property
    def checkpoint_path(self):
        return self.staging / "attempt-1/checkpoint.json"

    @property
    def pending_path(self):
        return self.staging / "attempt-1/pending.json"

    @property
    def archive(self):
        return self.staging / "attempt-1/output.tar"

    @property
    def ack_path(self):
        return self.root / "acks" / (self.key + ".json")

    def client(self, module=None):
        return (module or self.worker).Client(self.endpoint, TOKEN)

    def claim(self, index=0):
        body = self.client().request("/claim", {
            "worker": {"worker_id": f"cpu-worker-{index}", "host": "local",
                       "gpu_id": str(index)},
            "lease_seconds": 600,
            "stages": [self.coordinator.specs[self.job_key].stage],
        }, attempts=1)["lease"]
        return None if body is None else Lease.from_dict(body)

    def reopen(self):
        self.coordinator = self.coordinator.__class__(
            self.coordinator.root, self.jobs, reserve_gib=0)
        self.coordinator.queue = SQLiteJobQueue(
            self.coordinator.queue.db_path, clock=self.clock)


@pytest.fixture
def case(tmp_path, monkeypatch, request):
    coordinator_module = load_script("cluster_coordinator")
    monkeypatch.setitem(sys.modules, "cluster_coordinator", coordinator_module)
    worker = load_script("cluster_worker")
    # Keep the real request retry count, without waiting through backoff.
    monkeypatch.setattr(worker, "time", SimpleNamespace(time=time.time, sleep=Mock()))
    monkeypatch.setenv("HARNESS3D_CLUSTER_TOKEN", TOKEN)
    monkeypatch.setenv("NO_PROXY", "127.0.0.1,localhost")
    monkeypatch.setenv("no_proxy", "127.0.0.1,localhost")

    options = getattr(request, "param", {})
    if isinstance(options, str):
        options = {"stage": options}
    method = dict(METHOD)
    if options.get("oversized"):
        method["frozen_implementation"] = {"public_source": "x" * 120_000}
    identity = FrozenIdentity.freeze(
        {"scenes": {SCENE: {"qa_ids": QA_IDS}}}, method)
    stage = options.get("stage", "reconstruction")
    job = JobSpec(f"scenes/{SCENE}/{stage}", f"completed/{SCENE}",
                  identity, stage, max_attempts=1)
    jobs = tmp_path / "jobs.json"
    jobs.write_text(json.dumps({"jobs": [job.to_dict()]}))
    coordinator = coordinator_module.Coordinator(
        tmp_path / "coordinator", jobs, reserve_gib=0)
    clock = Clock()
    coordinator.queue = SQLiteJobQueue(coordinator.queue.db_path, clock=clock)
    root = tmp_path / "shared-worker-root"
    root.mkdir()
    state = WorkerCase(worker, coordinator, jobs, root, clock)
    state.method = method
    state.server_publish = Mock(wraps=coordinator.publish)
    monkeypatch.setattr(coordinator, "publish", state.server_publish)
    input_path = coordinator.root / "inputs/jobs" / (state.key + ".tar")
    input_path.parent.mkdir(parents=True)
    write_tar(input_path)
    state.input_path = input_path
    state.result = {
        "scene_id": SCENE, "qa_ids": QA_IDS, "no_solver": True, "no_labels": True,
    }

    def compute(inputs, output, method, *, client_factory):
        assert json.loads((inputs / "scene.json").read_text()) == {
            "scene_id": SCENE, "qa_ids": QA_IDS,
        }
        assert method == state.method and callable(client_factory)
        assert state.queue.get_job(job.job_key)["attempts"] == 1
        artifact = output / "prepared/binding.bin"
        artifact.parent.mkdir()
        artifact.write_bytes(PAYLOAD)
        return state.result

    def semantic_check(output, *, expected_scene_id, expected_qa_ids,
                       expected_method_config):
        assert expected_scene_id == SCENE and expected_qa_ids == QA_IDS
        assert expected_method_config == state.method
        assert (output / "prepared/binding.bin").read_bytes() == PAYLOAD
        assert json.loads((output / "scene_result.json").read_text()) == state.result
        return {"status": "validated", "scene_id": SCENE, "qa_ids": QA_IDS}

    state.compute = Mock(side_effect=compute)
    state.validate = Mock(side_effect=semantic_check)
    monkeypatch.setattr(scene_compute, "compute_scene", state.compute)
    monkeypatch.setattr(scene_compute, "validate_published_scene", state.validate)
    from skill3d.cluster import perception_client

    monkeypatch.setattr(perception_client, "RemoteBailianClient", Mock(side_effect=never))

    ready = threading.Event()
    servers = []
    errors = []

    def make_server(address, handler):
        class FaultHandler(handler):
            def disconnect(self):
                self.close_connection = True
                self.connection.shutdown(socket.SHUT_RDWR)

            def do_POST(self):
                state.routes.append(self.path)
                assert self.headers["Authorization"] == "Bearer " + TOKEN
                if self.path == "/publish":
                    state.publish_headers.append(dict(self.headers))
                remaining = state.faults["reject_count"].get(self.path, 0)
                if remaining:
                    state.faults["reject_count"][self.path] = remaining - 1
                    self.rfile.read(int(self.headers["Content-Length"]))
                    self.reply(503, {"error": "injected"})
                    return
                if self.path in state.faults["drop_before"]:
                    self.rfile.read(int(self.headers["Content-Length"]))
                    self.disconnect()
                    return
                if self.path in state.faults["reject"]:
                    self.rfile.read(int(self.headers["Content-Length"]))
                    self.reply(state.faults["reject"][self.path], {"error": "injected"})
                    return
                if self.path == "/input" and state.faults["input"]:
                    self.read_json()
                    raw = state.input_path.read_bytes()
                    self.send_response(200)
                    self.send_header("Content-Length", str(len(raw)))
                    self.send_header("X-Content-SHA256",
                                     "0" * 64 if state.faults["input"] == "digest"
                                     else hashlib.sha256(raw).hexdigest())
                    self.end_headers()
                    self.wfile.write(raw[:128] if state.faults["input"] == "truncated"
                                     else raw)
                    return
                return super().do_POST()

            def reply(self, status, value):
                if self.path == "/heartbeat" and status == 200:
                    state.heartbeat_seen.set()
                if self.path == "/publish" and status == 200:
                    state.publish_acks.append(value)
                remaining = state.faults["drop_ack"].get(self.path, 0)
                if status == 200 and remaining:
                    state.faults["drop_ack"][self.path] = remaining - 1
                    self.disconnect()
                    return
                return super().reply(status, value)

        class LocalServer(ThreadingHTTPServer):
            def serve_forever(self, **_kwargs):
                ready.set()
                return super().serve_forever(poll_interval=0.01)

            def handle_error(self, *_args):
                state.http_errors.append(sys.exc_info()[1])

        server = LocalServer(address, FaultHandler)
        servers.append(server)
        return server

    monkeypatch.setattr(coordinator_module, "ThreadingHTTPServer", make_server)

    def serve():
        try:
            coordinator_module.serve(state.coordinator, "127.0.0.1", 0)
        except BaseException as exc:
            errors.append(exc)
            ready.set()

    thread = threading.Thread(target=serve, daemon=True)
    thread.start()
    try:
        assert ready.wait(timeout=5), "local coordinator did not start"
        assert not errors
        state.endpoint = f"http://127.0.0.1:{servers[0].server_address[1]}"
        yield state
    finally:
        if servers:
            if not errors:
                servers[0].shutdown()
            servers[0].server_close()
        thread.join(timeout=5)
        assert not thread.is_alive()
        assert not errors
        assert not state.http_errors


def assert_complete(case):
    row = case.queue.get_job(case.job_key)
    assert row["status"] == "complete" and row["attempts"] == 1
    ack = json.loads(case.ack_path.read_text())
    assert ack["status"] == "complete" and ack["job_key"] == case.job_key
    assert ack["fingerprint"] == row["completion"]["fingerprint"]
    published = case.coordinator.root / ack["publication_root"]
    assert (published / "prepared/binding.bin").read_bytes() == PAYLOAD
    publication = inspect_receipt(
        published / "publication.json",
        expected_receipt_hash=row["output_metadata"]["receipt_sha256"])
    validate_artifacts(publication, published)
    assert publication.receipt.attempt == 1
    assert publication.receipt.identity_hash == row["spec"]["identity"]["identity_hash"]
    assert not case.staging.exists()
    assert not list((case.root / "acks").glob("*.tmp"))
    assert not list((case.coordinator.root / ".uploads").glob("*.tar"))
    assert case.claim(7) is None
    assert case.queue.progress_summary()["model_attempts"] == 1
    assert [event["event"] for event in case.queue.events()].count("completed") == 1


def assert_retained(case):
    metadata = json.loads(case.checkpoint_path.read_text())
    assert metadata == case.queue.get_job(case.job_key)["output_metadata"]
    assert case.worker.local_path(case.root, metadata["output_locator"]) == case.archive
    assert hash_file(case.archive) == (metadata["archive_sha256"], metadata["archive_bytes"])
    assert not case.ack_path.exists()
    return metadata


def test_success_uses_real_checkpoint_publication_and_ack_cleanup(case):
    unrelated = case.root / "jobs/other-worker/keep.bin"
    unrelated.parent.mkdir(parents=True)
    unrelated.write_bytes(b"another active worker")
    lease = case.claim()
    assert lease.phase == "leased" and lease.attempt == 0
    assert case.worker.run_job(case.client(), case.root, lease, reserve=0) == {
        "status": "complete", "job_key": case.job_key,
    }
    case.compute.assert_called_once()
    case.validate.assert_not_called()
    assert case.routes[:5] == ["/claim", "/input", "/start", "/finish", "/publish"]
    assert_complete(case)
    assert unrelated.read_bytes() == b"another active worker"


@pytest.mark.parametrize("case", [{"oversized": True}], indirect=True)
def test_oversized_frozen_method_uses_compact_http_header_and_exact_durable_ack(case):
    lease = case.claim()
    identity = lease.spec.identity
    assert len(base64.b64encode(json.dumps(lease.to_dict()).encode())) > 100 * 1024
    assert len(identity.method_config_json) > 100 * 1024
    assert case.worker.run_job(case.client(), case.root, lease, 0)["status"] == "complete"
    assert len(case.publish_headers) == 1
    header = case.publish_headers[0]["X-Lease-JSON"]
    assert len(header) < 1024
    wire = json.loads(base64.b64decode(header, validate=True))
    assert wire == {
        "schema": "harness3d-lease-header/1", "job_key": lease.job_key,
        "worker": lease.worker.to_dict(), "token": lease.token,
        "expires_at": lease.expires_at, "attempt": 1,
    }
    case.server_publish.assert_called_once()
    rebuilt = case.server_publish.call_args.args[0]
    assert rebuilt.phase == "publish" and rebuilt.attempt == 1
    assert rebuilt.spec == lease.spec and rebuilt.spec.identity == identity
    assert json.loads(rebuilt.output_metadata_json) == case.queue.get_job(
        case.job_key)["output_metadata"]
    assert len(case.publish_acks) == 1
    assert json.loads(case.ack_path.read_text()) == case.publish_acks[0]
    assert case.ack_path.read_bytes() == json.dumps(
        case.publish_acks[0], sort_keys=True, allow_nan=False).encode()
    assert case.publish_acks[0]["duplicate"] is False
    case.compute.assert_called_once()
    assert_complete(case)
    assert case.queue.get_job(case.job_key)["spec"]["identity"] == identity.to_dict()


@pytest.mark.parametrize("case", ["gpu_aggregate"], indirect=True)
def test_gpu_publication_with_semantic_stub_can_commit_and_ack(case):
    result = case.worker.run_job(case.client(), case.root, case.claim(), 0)
    assert result["status"] == "complete"
    assert case.validate.call_count == 2
    assert_complete(case)


@pytest.mark.parametrize("failure", ["http_503", "disconnect"])
def test_return_failure_reclaims_archive_without_compute_and_cleans_after_ack(
        case, monkeypatch, failure):
    if failure == "http_503":
        case.faults["reject"]["/publish"] = 503
    else:
        case.faults["drop_before"].add("/publish")
    old = case.claim()
    assert case.worker.run_job(case.client(), case.root, old, reserve=0)["status"] == "publish_retry"
    metadata = assert_retained(case)
    saved = case.archive.read_bytes()
    checkpoint_bytes = case.checkpoint_path.read_bytes()
    row = case.queue.get_job(case.job_key)
    assert row["status"] == "queued" and row["resume_phase"] == "publish"
    assert row["attempts"] == row["return_failures"] == 1
    assert case.routes.count("/publish") == 4
    assert case.routes.count("/transfer-failed") == 1
    assert not case.pending_path.exists()
    assert row["completion"] is None and case.claim(1) is None
    case.compute.assert_called_once()

    # Only the durable tar and metadata survive; use a fresh worker/client.
    shutil.rmtree(case.staging / "input")
    shutil.rmtree(case.staging / "attempt-1/output")
    case.reopen()
    case.clock.advance(6)
    resumed = case.claim(1)
    assert resumed.phase == "publish" and resumed.attempt == 1
    assert resumed.token != old.token and resumed.worker != old.worker
    assert json.loads(resumed.output_metadata_json) == metadata
    assert case.archive.read_bytes() == saved
    assert case.checkpoint_path.read_bytes() == checkpoint_bytes
    case.faults["reject"].clear()
    case.faults["drop_before"].clear()
    fresh_worker = load_script("cluster_worker")
    client = case.client(fresh_worker)
    monkeypatch.setattr(client, "download", Mock(side_effect=never))
    case.compute.side_effect = never
    case.compute.reset_mock()
    routes_before = len(case.routes)
    assert fresh_worker.run_job(client, case.root, resumed, reserve=0)["status"] == "complete"
    case.compute.assert_not_called()
    client.download.assert_not_called()
    assert case.routes[routes_before:] == ["/publish"]
    assert_complete(case)
    assert case.queue.progress_summary()["return_failures"] == 1


@pytest.mark.parametrize("failure", ["http_503", "digest", "truncated", "disk", "unsafe_tar"])
def test_prefetch_failure_consumes_no_model_attempt_and_can_retry(case, failure):
    reserve = 0
    if failure == "http_503":
        case.faults["reject"]["/input"] = 503
    elif failure in {"digest", "truncated"}:
        case.faults["input"] = failure
    elif failure == "disk":
        reserve = sys.maxsize
    else:
        write_tar(case.input_path, "../escaped.bin", b"must not extract")
    lease = case.claim()
    assert case.worker.run_job(case.client(), case.root, lease, reserve)["status"] == "prefetch_retry"
    row = case.queue.get_job(case.job_key)
    assert row["status"] == "queued" and row["resume_phase"] == "prefetch"
    assert row["attempts"] == 0 and row["prefetch_failures"] == 1
    assert row["completion"] is None and row["output_metadata"] == {}
    assert case.queue.progress_summary()["model_attempts"] == 0
    case.compute.assert_not_called()
    assert "/start" not in case.routes and "/publish" not in case.routes
    assert not case.ack_path.exists() and not case.checkpoint_path.exists()
    assert not (case.staging / "input.part").exists()
    assert not (case.staging / "escaped.bin").exists()
    if failure in {"http_503", "digest", "truncated"}:
        assert not (case.staging / "input.tar").exists()

    case.faults["input"] = None
    case.faults["reject"].clear()
    if failure == "unsafe_tar":
        (case.staging / "input.tar").unlink()
        write_tar(case.input_path)
    case.clock.advance(6)
    resumed = case.claim(1)
    assert resumed.attempt == 0 and resumed.phase == "leased"
    assert case.worker.run_job(case.client(), case.root, resumed, reserve=0)["status"] == "complete"
    case.compute.assert_called_once()
    assert_complete(case)


@pytest.mark.parametrize("tamper", ["same_size", "truncated"])
def test_retained_archive_tampering_never_publishes_or_completes(case, monkeypatch, tamper):
    case.faults["reject"]["/publish"] = 503
    assert case.worker.run_job(case.client(), case.root, case.claim(), 0)["status"] == "publish_retry"
    assert_retained(case)
    original = case.archive.read_bytes()
    case.archive.write_bytes(
        bytes([original[0] ^ 1]) + original[1:] if tamper == "same_size" else original[:-1])
    case.faults["reject"].clear()
    case.clock.advance(6)
    resumed = case.claim(1)
    assert resumed.phase == "publish"
    client = case.client()
    monkeypatch.setattr(client, "publish", Mock(wraps=client.publish))
    case.compute.side_effect = never
    case.compute.reset_mock()
    case.worker.time.sleep.reset_mock()
    assert case.worker.run_job(client, case.root, resumed, 0)["status"] == "publish_retry"
    client.publish.assert_not_called()
    case.worker.time.sleep.assert_not_called()
    case.compute.assert_not_called()
    row = case.queue.get_job(case.job_key)
    assert row["completion"] is None and row["attempts"] == 1
    assert row["status"] == "queued" and row["return_failures"] == 2
    assert case.archive.exists() and not case.ack_path.exists()
    assert not (case.coordinator.root / "results").exists()
    assert not any(event["event"] == "completed" for event in case.queue.events())


def test_exhausted_finish_ack_loss_succeeds_in_place_without_releasing_lease(case):
    case.faults["drop_ack"]["/finish"] = 4
    assert case.worker.run_job(case.client(), case.root, case.claim(), 0)["status"] == "complete"
    assert case.routes.count("/finish") == 5
    assert case.routes.count("/publish") == 1
    assert "/transfer-failed" not in case.routes and "/fail" not in case.routes
    case.compute.assert_called_once()
    assert_complete(case)


@pytest.mark.parametrize("lost_acks", [1, 3])
def test_committed_publish_ack_loss_recovers_ack_and_cleans_checkpoint(case, lost_acks):
    case.faults["drop_ack"]["/publish"] = lost_acks
    result = case.worker.run_job(case.client(), case.root, case.claim(), 0)
    row = case.queue.get_job(case.job_key)
    assert row["status"] == "complete" and row["completion"] is not None
    case.compute.assert_called_once()
    assert result["status"] == "complete"
    assert case.routes.count("/publish") == lost_acks + 1
    assert "/transfer-failed" not in case.routes
    assert all(header["X-Lease-JSON"] == case.publish_headers[0]["X-Lease-JSON"]
               for header in case.publish_headers)
    assert all(header["X-Archive-SHA256"] == case.publish_headers[0]["X-Archive-SHA256"]
               for header in case.publish_headers)
    assert not case.publish_acks[0]["duplicate"] and case.publish_acks[-1]["duplicate"]
    assert json.loads(case.ack_path.read_text()) == case.publish_acks[-1]
    assert_complete(case)


def test_all_publication_ack_retries_exhausted_retains_accepted_lease_for_restart(
        case, monkeypatch):
    case.faults["drop_ack"]["/publish"] = 4
    with pytest.raises(case.worker.CoordinatorRejected, match="/transfer-failed"):
        case.worker.run_job(case.client(), case.root, case.claim(), 0)
    assert case.routes.count("/publish") == 4
    assert_retained(case)
    row = case.queue.get_job(case.job_key)
    assert row["status"] == "complete" and row["return_failures"] == 0
    pending = json.loads(case.pending_path.read_text())
    stored = Lease.from_dict(pending["lease"])
    assert stored.phase == "publish"
    saved = case.archive.read_bytes()
    case.reopen()
    case.clock.advance(601)
    case.compute.side_effect = never
    case.compute.reset_mock()
    client = case.client()
    monkeypatch.setattr(client, "download", Mock(side_effect=never))
    routes_before = len(case.routes)
    assert case.archive.read_bytes() == saved
    result = case.worker.recover_pending(client, case.root, stored.worker.to_dict(), 0)
    assert result["status"] == "complete"
    assert case.routes[routes_before:] == ["/publish"]
    assert case.publish_acks[-1]["duplicate"]
    client.download.assert_not_called()
    case.compute.assert_not_called()
    assert_complete(case)


def test_finish_outage_retries_in_place_with_heartbeat_and_no_additional_model(
        case, monkeypatch):
    case.faults["reject_count"]["/finish"] = 4
    monkeypatch.setattr(case.worker.Heartbeat, "interval", 0.01)

    def backoff(seconds):
        if seconds != 5:
            return
        assert case.routes.count("/finish") == 4
        pending = json.loads(case.pending_path.read_text())
        assert pending["lease"]["phase"] == "running"
        assert pending["lease"]["attempt"] == 1
        assert pending["metadata"] == json.loads(case.checkpoint_path.read_text())
        case.heartbeat_seen.clear()
        # Synchronize with the independent real HTTP heartbeat, not wall time.
        assert case.heartbeat_seen.wait(timeout=5)
        row = case.queue.get_job(case.job_key)
        assert row["status"] == "running" and row["output_metadata"] == {}
        assert "/transfer-failed" not in case.routes and "/fail" not in case.routes

    case.worker.time.sleep.side_effect = backoff
    assert case.worker.run_job(case.client(), case.root, case.claim(), 0)["status"] == "complete"
    assert case.routes.count("/finish") == 5
    assert case.routes.count("/start") == 1 and case.routes.count("/claim") == 1
    assert "/heartbeat" in case.routes
    case.compute.assert_called_once()
    assert_complete(case)


def leave_pending_finish(case, monkeypatch, *, drop_ack=False):
    if drop_ack:
        case.faults["drop_ack"]["/finish"] = 4
    else:
        case.faults["reject"]["/finish"] = 503
    old = case.claim()

    def one_request_batch(client, lease, metadata):
        return Lease.from_dict(client.request(
            "/finish", {"lease": lease.to_dict(), "metadata": metadata})["lease"])

    # Production keeps retrying with heartbeats. Interrupt one exhausted batch
    # to simulate a process restart and bound the permanent-outage test.
    with monkeypatch.context() as patch:
        patch.setattr(case.worker, "finish_with_retry", one_request_batch)
        with pytest.raises(case.worker.TransportError, match="request unavailable"):
            case.worker.run_job(case.client(), case.root, old, 0)
    metadata = json.loads(case.checkpoint_path.read_text())
    assert hash_file(case.archive) == (metadata["archive_sha256"], metadata["archive_bytes"])
    pending = json.loads(case.pending_path.read_text())
    stored = Lease.from_dict(pending["lease"])
    assert pending["schema"] == "harness3d-pending-checkpoint/1"
    assert pending["metadata"] == metadata
    assert stored.worker == old.worker and stored.token == old.token
    assert stored.spec == old.spec and stored.phase == "running" and stored.attempt == 1
    row = case.queue.get_job(case.job_key)
    assert row["attempts"] == 1 and case.routes.count("/finish") == 4
    assert "/transfer-failed" not in case.routes and "/fail" not in case.routes
    assert "/publish" not in case.routes and not case.ack_path.exists()
    case.compute.assert_called_once()
    return stored, metadata


@pytest.mark.parametrize("drop_ack", [False, True])
def test_restart_main_recovers_live_pending_checkpoint_before_claim_without_compute(
        case, monkeypatch, capsys, drop_ack):
    stored, metadata = leave_pending_finish(case, monkeypatch, drop_ack=drop_ack)
    row = case.queue.get_job(case.job_key)
    assert row["status"] == ("publishing" if drop_ack else "running")
    assert row["output_metadata"] == (metadata if drop_ack else {})
    saved = case.archive.read_bytes()
    case.reopen()
    case.clock.advance(500)
    case.faults["reject"].clear()
    shutil.rmtree(case.staging / "input")
    shutil.rmtree(case.staging / "attempt-1/output")
    case.compute.side_effect = never
    case.compute.reset_mock()
    fresh_worker = load_script("cluster_worker")
    monkeypatch.setattr(fresh_worker.Client, "download", Mock(side_effect=never))
    monkeypatch.setattr(sys, "argv", [
        "cluster_worker.py", "--endpoint", case.endpoint, "--root", str(case.root),
        "--worker-id", stored.worker.worker_id, "--host-id", stored.worker.host,
        "--gpu-id", stored.worker.gpu_id, "--reserve-gib", "0", "--max-jobs", "1",
    ])
    assert case.archive.read_bytes() == saved
    routes_before = len(case.routes)
    fresh_worker.main()
    assert case.routes[routes_before:] == (
        ["/heartbeat", "/publish"] if drop_ack else ["/heartbeat", "/finish", "/publish"])
    assert json.loads(capsys.readouterr().out)["status"] == "complete"
    fresh_worker.Client.download.assert_not_called()
    case.compute.assert_not_called()
    assert_complete(case)


def test_expired_pending_without_db_metadata_is_blocked_retained_and_final_attempt_fails(
        case, monkeypatch):
    stored, metadata = leave_pending_finish(case, monkeypatch)
    assert case.queue.get_job(case.job_key)["output_metadata"] == {}
    saved = case.archive.read_bytes()
    case.faults["reject"].clear()
    case.reopen()
    case.clock.advance(601)
    case.compute.side_effect = never
    case.compute.reset_mock()
    routes_before = len(case.routes)
    result = case.worker.recover_pending(case.client(), case.root, stored.worker.to_dict(), 0)
    assert result["status"] == "checkpoint_blocked"
    assert result["output_locator"] == metadata["output_locator"]
    assert "live lease" in result["error"]
    assert case.routes[routes_before:] == ["/heartbeat"]
    assert case.worker.recover_pending(case.client(), case.root, stored.worker.to_dict(), 0) is None
    assert case.claim(1) is None
    row = case.queue.get_job(case.job_key)
    assert row["status"] == "failed" and row["error_code"] == "attempt_budget_exhausted"
    assert row["output_metadata"] == {} and row["completion"] is None
    assert row["attempts"] == 1 and row["return_failures"] == 0
    assert case.archive.read_bytes() == saved
    assert json.loads(case.pending_path.read_text())["blocked"] == result["error"]
    assert json.loads(case.checkpoint_path.read_text()) == metadata
    assert not case.ack_path.exists()
    case.compute.assert_not_called()


def test_other_worker_does_not_recover_owned_pending_checkpoint(case, monkeypatch):
    stored, _ = leave_pending_finish(case, monkeypatch)
    worker = {**stored.worker.to_dict(), "worker_id": "another-worker"}
    routes_before = len(case.routes)
    assert case.worker.recover_pending(case.client(), case.root, worker, 0) is None
    assert case.routes[routes_before:] == []
    assert case.pending_path.exists()
    assert not json.loads(case.pending_path.read_text()).get("blocked")


def test_live_pending_archive_tampering_is_blocked_before_finish_without_retry(
        case, monkeypatch):
    stored, _ = leave_pending_finish(case, monkeypatch)
    original = case.archive.read_bytes()
    case.archive.write_bytes(bytes([original[0] ^ 1]) + original[1:])
    case.faults["reject"].clear()
    case.compute.side_effect = never
    case.compute.reset_mock()
    case.worker.time.sleep.reset_mock()
    client = case.client()
    monkeypatch.setattr(client, "download", Mock(side_effect=never))
    monkeypatch.setattr(client, "publish", Mock(side_effect=never))
    routes_before = len(case.routes)
    result = case.worker.recover_pending(client, case.root, stored.worker.to_dict(), 0)
    assert result["status"] == "checkpoint_blocked"
    assert result["error"] == "retained output checkpoint changed"
    assert case.routes[routes_before:] == ["/heartbeat"]
    assert case.queue.get_job(case.job_key)["output_metadata"] == {}
    assert case.queue.get_job(case.job_key)["completion"] is None
    assert case.archive.exists() and not case.ack_path.exists()
    case.compute.assert_not_called()
    client.download.assert_not_called()
    client.publish.assert_not_called()
    case.worker.time.sleep.assert_not_called()


def test_malformed_input_tar_records_prefetch_failure_without_model_attempt(case):
    case.input_path.write_bytes(b"digest-valid transport, invalid tar archive")
    result = case.worker.run_job(case.client(), case.root, case.claim(), 0)
    case.compute.assert_not_called()
    assert case.queue.get_job(case.job_key)["attempts"] == 0
    assert "/start" not in case.routes and not case.ack_path.exists()
    assert result["status"] == "prefetch_retry"
    row = case.queue.get_job(case.job_key)
    assert row["status"] == "queued" and row["prefetch_failures"] == 1
    assert not (case.staging / "input.tar").exists()
    assert not (case.staging / "input.unpacking").exists()
    write_tar(case.input_path)
    case.clock.advance(6)
    assert case.worker.run_job(case.client(), case.root, case.claim(1), 0)["status"] == "complete"
    case.compute.assert_called_once()
    assert_complete(case)
