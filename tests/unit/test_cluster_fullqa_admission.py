"""Ready inputs enter a full frozen scope without exposing missing/corrupt jobs."""
import hashlib
import importlib
import json
from pathlib import Path

import pytest

from skill3d.cluster import FrozenIdentity, JobSpec, WorkerIdentity
from skill3d.cluster.identity import QueueError


def setup_coordinator(tmp_path, monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[2] / "scripts"))
    module = importlib.import_module("cluster_coordinator")
    identity = FrozenIdentity.freeze(
        {"scenes": {"a": {"qa_ids": ["1"]}, "b": {"qa_ids": ["2"]}}}, {})
    specs = {name: JobSpec(f"run/{name}/gpu_aggregate", f"done-{name}", identity, "gpu_aggregate")
             for name in ("a", "b")}
    jobs = tmp_path / "jobs.json"
    module.atomic_json(jobs, {"jobs": [spec.to_dict() for spec in specs.values()]})
    # Control the watch loop explicitly so the test covers readiness transitions
    # without depending on thread scheduling.
    monkeypatch.setattr(module.Coordinator, "watch_ready_inputs", lambda _self: None)
    coordinator = module.Coordinator(tmp_path, jobs, ready_inputs_only=True)
    return module, coordinator, identity, specs


def ready_file(module, root, spec):
    data = spec.job_key.encode()
    key = hashlib.sha256(data).hexdigest()
    path = root / "inputs/jobs" / f"{key}.tar"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return {"job_key": spec.job_key, "archive_sha256": hashlib.sha256(data).hexdigest(),
            "archive_bytes": len(data)}


def test_only_verified_ready_jobs_can_be_claimed(tmp_path, monkeypatch):
    module, coordinator, identity, specs = setup_coordinator(tmp_path, monkeypatch)
    worker = WorkerIdentity("worker", "host", "0")
    assert coordinator.queue.claim(worker) is None
    progress = {"identity_hash": identity.identity_hash,
                "scenes": {"a": ready_file(module, tmp_path, specs["a"])}}
    module.atomic_json(tmp_path / "receipts/package_progress.json", progress)
    coordinator.admit_ready_inputs()
    assert coordinator.queue.claim(worker).job_key == specs["a"].job_key
    assert coordinator.queue.claim(WorkerIdentity("other", "host", "1")) is None
    progress["scenes"]["b"] = ready_file(module, tmp_path, specs["b"])
    module.atomic_json(tmp_path / "receipts/package_progress.json", progress)
    coordinator.admit_ready_inputs()
    assert coordinator.queue.claim(WorkerIdentity("other", "host", "1")).job_key == specs["b"].job_key


def test_corrupt_ready_file_is_rejected_before_admission(tmp_path, monkeypatch):
    module, coordinator, identity, specs = setup_coordinator(tmp_path, monkeypatch)
    row = ready_file(module, tmp_path, specs["a"])
    row["archive_sha256"] = "0" * 64
    module.atomic_json(tmp_path / "receipts/package_progress.json", {
        "identity_hash": identity.identity_hash, "scenes": {"a": row}})
    with pytest.raises(QueueError, match="digest changed"):
        coordinator.admit_ready_inputs()
    assert coordinator.queue.claim(WorkerIdentity("worker", "host", "0")) is None
