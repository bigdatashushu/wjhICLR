"""Synthetic bounded-dispatch and never-retry continuation check, no real inputs."""
from collections import Counter
from pathlib import Path
import tempfile
import threading
from types import FunctionType

from ..traces import digest, read, write
from . import vggt_location_batch as batch
from .prototype.client import ServiceError


def run():
    settings = lambda api: {"model": "synthetic", "api_id": api}
    request = lambda directory, item, arm: {
        "sample_id": item["problem"]["sample_id"], "api": item["api_id"], "arm": arm,
    }
    barrier = threading.Barrier(2)
    calls = []
    guard = threading.Lock()

    class FakeClient:
        def __init__(self, api, key):
            self.api = api

        def assert_public(self, value):
            pass

        def call(self, body):
            with guard:
                calls.append(body["sample_id"])
            if body["sample_id"] in ("synthetic-0", "synthetic-1"):
                barrier.wait(timeout=5)
                raise ServiceError("synthetic_service_failure", 503)
            return {"text": "A", "tool_call_count": 0, "reported": {}, "usage": {}}

    with tempfile.TemporaryDirectory(prefix="mmsi-location-batch-fake-") as temporary:
        root = Path(temporary)
        items = [
            {"index": i, "problem": {
                "sample_id": f"synthetic-{i}", "input_sha256": digest({"i": i}),
                "options": ["one", "two"], "task_type": "synthetic",
            }, "api_id": f"synthetic-api-{i % 2}", "arm_order": ["relation"], "depth_images": []}
            for i in range(7)
        ]
        for item in items:
            item["payload_sha256"] = {"relation": digest(request(None, item, "relation"))}

        def synthetic_validate(directory, **kwargs):
            return read(Path(directory) / "manifest.json")

        shared = batch.old.shared
        executor = FunctionType(shared.execute.__code__, {
            **vars(shared), "validate": synthetic_validate, "request": request, "inference": settings,
        }, "synthetic_execute")
        executor.__kwdefaults__ = {"client_factory": FakeClient}
        collector = FunctionType(shared.collect.__code__, {
            **vars(shared), "inference": settings,
        }, "synthetic_collect")

        def phase(name, selected):
            directory = root / name
            directory.mkdir()
            manifest = {
                "run_identity": digest({"phase": name}), "items": selected,
                "questions": len(selected), "max_requests": len(selected),
                "max_inflight": 2, "api_ids": ["synthetic-api-0", "synthetic-api-1"],
            }
            write(directory / "manifest.json", manifest)
            write(directory / "preflight.json", {"model_requests": 0})
            value = executor(directory, {api: "synthetic-unused" for api in manifest["api_ids"]})
            records, _ = collector(directory, manifest)
            return value, records

        first, records = phase("first", items)
        assert first["status"] == "stopped_service_error"
        assert first["dispatched"] == 2 and first["inflight"] == 0 and first["peak_inflight"] == 2
        pending = batch.remaining_items(items, [records])
        assert len(pending) == 5
        second, later = phase("untouched", pending)
        assert second["status"] == "complete" and second["dispatched"] == 5
        assert not batch.remaining_items(items, [records, later])
        assert Counter(calls) == Counter(item["problem"]["sample_id"] for item in items)
        try:
            batch.remaining_items(items, [records, later, later])
        except ValueError:
            pass
        else:
            raise AssertionError("duplicate_attempt_not_rejected")
    print('{"status":"passed","fake_attempts":7,"real_requests":0,"labels_read":false}', flush=True)


if __name__ == "__main__":
    run()
