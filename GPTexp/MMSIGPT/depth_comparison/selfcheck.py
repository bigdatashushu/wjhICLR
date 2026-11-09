"""Synthetic critical-path checks: rendering, isolation, paired dispatch and freeze."""
from __future__ import annotations

from contextlib import redirect_stdout
from io import BytesIO, StringIO
from pathlib import Path
import socket
import sys
import tempfile
import threading
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image

from ...traces import digest, file_digest, read, write
from ..prototype.client import ServiceError
from ..prototype.inputs import load_images, original_input_sha256
from .depth import read_depth, render
from .evaluation import score, summarize
from .inputs import request, selection, tokenizer
from .runner import execute, freeze, prepare, validate_frozen, validate
from .continuation import prepare_from_frozen

TOKENIZER = None


class FakeClient:
    calls = []
    fail = False

    def __init__(self, api_id, key):
        self.api_id = api_id

    def assert_public(self, value):
        pass

    def call(self, body):
        self.calls.append((self.api_id, digest(body)))
        if self.fail:
            raise ServiceError("synthetic_service_failure", 524)
        return {"text": "``A``", "tool_call_count": 0}


class Checks(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.network = patch.object(socket.socket, "connect",
                                    side_effect=AssertionError("network_disabled"))
        self.network.start()
        self.addCleanup(self.network.stop)
        self.addCleanup(self.temp.cleanup)
        FakeClient.calls = []
        FakeClient.fail = False

    def fixture(self):
        from skill3d.small_image.precompute import _prepared
        from skill3d.small_image.storage import save_prepared

        paths = []
        frames = [np.full((10, 12, 3), value, np.uint8) for value in (64, 192)]
        for i, frame in enumerate(frames):
            path = self.root / f"rgb-{i}.png"
            Image.fromarray(frame).save(path)
            paths.append(str(path))
        images_hash = [file_digest(path) for path in paths]
        depth = np.arange(1, 121, dtype=np.float32).reshape(10, 12)
        mask = np.ones((10, 12), bool)
        mask[0, 0] = False
        jobs, index_rows = [], []
        for i in range(30):
            problem = {
                "benchmark": "mmsi", "sample_id": f"synthetic_{i}", "source_split": "test",
                "question": "Synthetic image question.", "options": ["one", "two"],
                "image_paths": paths, "task_type": f"synthetic_type_{i % 6}",
            }
            problem["input_sha256"] = original_input_sha256(problem, images_hash)
            prepared = _prepared(
                problem=problem, frames=tuple(frames), candidates=(), depths={0: depth, 1: depth},
                points={}, intrinsics={}, geometry_masks={0: mask, 1: mask},
                geometries={j: {"relative_only": True} for j in (0, 1)}, categories=(),
                provenance={"labels_used": False, "geometry_scope": "per_view_relative_only",
                            "metric_scale_verified": False, "world_frame_verified": False},
                capabilities={"relative_depth": True}, cache_key=digest(problem),
                normalized_image_paths=(), errors=())
            directory = self.root / "prepared" / problem["input_sha256"]
            save_prepared(prepared, directory)
            jobs.append({"problem": problem, "images": None})
            index_rows.append({
                "sample_id": problem["sample_id"], "input_sha256": problem["input_sha256"],
                "artifact_sha256": prepared.artifact_sha256,
                "images": paths, "prepared_directory": str(directory),
            })
        source, index = self.root / "public.json", self.root / "index.json"
        write(source, {"jobs": jobs})
        write(index, {"manifest_sha256": file_digest(source), "records": index_rows})
        chosen = selection(source, index, [], [f"pool-{i + 1:02d}" for i in range(6)])
        return chosen

    def test_gray_invalid_constant_and_grid(self):
        depth = np.array([[1, 2, 3], [4, np.nan, 6]], dtype=float)
        mask = np.ones_like(depth, dtype=bool)
        raw, receipt = render(depth, mask, (3, 2))
        image = np.asarray(Image.open(BytesIO(raw)))
        self.assertEqual(image[1, 1].tolist(), [255, 0, 255])
        self.assertLess(int(image[0, 0, 0]), int(image[1, 2, 0]))
        raw, _ = render(np.ones((2, 3)), mask, (3, 2))
        self.assertTrue(np.all(np.asarray(Image.open(BytesIO(raw))) == 128))
        with self.assertRaises(ValueError):
            render(depth, mask, (2, 3))
        with self.assertRaises(ValueError):
            render(depth, np.zeros_like(mask), (3, 2))
        self.assertEqual(receipt["valid_pixels"], 5)

    def test_native_storage_hash_tamper(self):
        chosen = self.fixture()
        item = chosen["items"][0]
        receipts = [row[0] for row in load_images(item["problem"])]
        depths, provenance = read_depth(item["prepared_directory"], item["problem"],
                                        receipts, item["artifact_sha256"])
        self.assertEqual(len(depths), 2)
        self.assertFalse(provenance["labels_used"])
        path = Path(item["prepared_directory"]) / "arrays.npz"
        with path.open("ab") as stream:
            stream.write(b"tampered")
        with self.assertRaises(ValueError):
            read_depth(item["prepared_directory"], item["problem"],
                       receipts, item["artifact_sha256"])

    def test_full_paired_freeze_score_and_isolation(self):
        chosen = self.fixture()
        directory = self.root / "run"
        manifest = prepare(directory, chosen, TOKENIZER)
        item = manifest["items"][0]
        direct = request(directory, item, "direct")
        depth = request(directory, item, "depth")
        original_read = Path.read_bytes
        allowed_paths = set(item["problem"]["image_paths"])
        def original_only(path):
            if str(path) not in allowed_paths:
                raise AssertionError("prepared_or_depth_opened")
            return original_read(path)
        with patch.object(Path, "read_bytes", autospec=True, side_effect=original_only):
            self.assertEqual(request(directory, item, "direct"), direct)
        image_blocks = lambda body: [block["image_url"] for row in body["input"]
                                    for block in row["content"] if block["type"] == "input_image"]
        self.assertEqual(image_blocks(depth)[:2], image_blocks(direct))
        self.assertEqual(len(image_blocks(depth)), 4)
        self.assertLessEqual(item["annotation"]["total_tokens"], 160)
        keys = {api: "synthetic-memory-only-key" for api in manifest["api_ids"]}
        references = self.root / "references.json"
        write(references, {"references": [
            {"benchmark": "mmsi", "sample_id": entry["problem"]["sample_id"],
             "reference_answer": "A"} for entry in manifest["items"]
        ]})
        with self.assertRaises(FileNotFoundError):
            score(directory, references, file_digest(references))
        with redirect_stdout(StringIO()):
            result = execute(directory, keys, client_factory=FakeClient)
        self.assertEqual(result["dispatched"], 60)
        self.assertEqual(result["status"], "complete")
        self.assertLessEqual(result["peak_inflight"], 6)
        self.assertEqual(len(FakeClient.calls), 60)
        self.assertEqual({api: sum(call[0] == api for call in FakeClient.calls)
                          for api in keys}, {api: 10 for api in keys})
        frozen = freeze(directory)
        scored = score(directory, references, file_digest(references))
        self.assertEqual(scored["overall"]["complete_pairs"], 30)
        self.assertEqual(scored["overall"]["direct_correct"], 30)
        self.assertEqual(scored["overall"]["delta"], 0)
        with self.assertRaises(ValueError):
            execute(directory, keys, client_factory=FakeClient)
        result_path = next((directory / "traces").glob("*.result.json"))
        changed = read(result_path)
        changed["response"]["text"] = "B"
        write(result_path, changed)
        with self.assertRaises(ValueError):
            validate_frozen(directory, validate(directory))
        self.assertEqual(len(frozen["predictions"]), 60)

    def test_service_failure_stops_without_retry(self):
        directory = self.root / "run"
        manifest = prepare(directory, self.fixture(), TOKENIZER)
        FakeClient.fail = True
        with redirect_stdout(StringIO()):
            result = execute(directory, {api: "synthetic-key" for api in manifest["api_ids"]},
                             client_factory=FakeClient)
        self.assertEqual(result["status"], "stopped_service_error")
        self.assertLessEqual(result["dispatched"], 6)
        self.assertEqual(result["dispatched"], len(FakeClient.calls))
        frozen = freeze(directory)
        self.assertEqual(len(frozen["predictions"]), 60)

    def test_paired_gain_loss_denominator(self):
        pairs = [
            {"direct": {"status": "answered", "correct": False},
             "depth": {"status": "answered", "correct": True}},
            {"direct": {"status": "answered", "correct": True},
             "depth": {"status": "answered", "correct": False}},
            {"direct": {"status": "not_dispatched", "correct": False},
             "depth": {"status": "not_dispatched", "correct": False}},
        ]
        result = summarize(pairs)
        self.assertEqual(result["questions"], 3)
        self.assertEqual(result["complete_pairs"], 2)
        self.assertEqual(result["wrong_to_correct"], 1)
        self.assertEqual(result["correct_to_wrong"], 1)
        self.assertEqual(result["delta"], 0)

    def test_untouched_continuation_preserves_payload_and_never_retries(self):
        parent = self.root / "parent"
        manifest = prepare(parent, self.fixture(), TOKENIZER)
        barrier = threading.Barrier(6)
        class SixFailures(FakeClient):
            def call(self, body):
                barrier.wait(timeout=10)
                raise ServiceError("synthetic_service_failure", 503)
        keys = {api: "synthetic-key" for api in manifest["api_ids"]}
        with redirect_stdout(StringIO()):
            execution = execute(parent, keys, client_factory=SixFailures)
        self.assertEqual(execution["dispatched"], 6)
        parent_freeze = freeze(parent)
        attempted = {row["sample_id"] for row in parent_freeze["predictions"]
                     if row["status"] != "not_dispatched"}
        child = self.root / "child"
        continued = prepare_from_frozen(child, parent)
        self.assertEqual(continued["questions"], 24)
        self.assertFalse(attempted & {item["problem"]["sample_id"]
                                     for item in continued["items"]})
        originals = {item["problem"]["sample_id"]: item for item in manifest["items"]}
        for item in continued["items"]:
            self.assertEqual(item["payload_sha256"],
                             originals[item["problem"]["sample_id"]]["payload_sha256"])
        with redirect_stdout(StringIO()):
            result = execute(child, keys, client_factory=FakeClient)
        self.assertEqual(result["dispatched"], 48)
        self.assertEqual(result["status"], "complete")
        self.assertEqual(len(FakeClient.calls), 48)
        self.assertEqual(len(freeze(child)["predictions"]), 48)
        self.assertEqual(result["dispatched"] + execution["dispatched"], 54)
        path = next((parent / "traces").glob("*.dispatch.json"))
        with path.open("a") as stream:
            stream.write(" ")
        with self.assertRaises(ValueError):
            validate(child)

    def test_two_failures_then_18_pairs_keep_original_inputs_and_budget(self):
        parent = self.root / "parent"
        original = prepare(parent, self.fixture(), TOKENIZER)
        keys = {api: "synthetic-key" for api in original["api_ids"]}
        barrier = threading.Barrier(6)
        class SixFailures(FakeClient):
            def call(self, body):
                barrier.wait(timeout=10)
                raise ServiceError("synthetic_service_failure", 503)
        failed_ids = set()
        directories = [parent, self.root / "child"]
        for i, directory in enumerate(directories):
            if i:
                prepare_from_frozen(directory, directories[i - 1])
            with redirect_stdout(StringIO()):
                result = execute(directory, keys, client_factory=SixFailures)
            self.assertEqual(result["dispatched"], 6)
            frozen = freeze(directory)
            attempted = {row["sample_id"] for row in frozen["predictions"]
                         if row["status"] != "not_dispatched"}
            self.assertFalse(attempted & failed_ids)
            failed_ids |= attempted
        recovered = self.root / "recovered"
        manifest = prepare_from_frozen(recovered, directories[-1])
        self.assertEqual(manifest["questions"], 18)
        self.assertEqual(manifest["continuation"]["parent_requests"], 12)
        self.assertEqual(manifest["continuation"]["combined_max_requests"], 48)
        self.assertEqual(set(manifest["continuation"]["excluded_attempted_sample_ids"]), failed_ids)
        originals = {item["problem"]["sample_id"]: item for item in original["items"]}
        for item in manifest["items"]:
            sample = item["problem"]["sample_id"]
            self.assertNotIn(sample, failed_ids)
            self.assertEqual(item["payload_sha256"], originals[sample]["payload_sha256"])
            self.assertEqual(item["images"], originals[sample]["images"])
            self.assertEqual(item["depth_images"], originals[sample]["depth_images"])
        with redirect_stdout(StringIO()):
            result = execute(recovered, keys, client_factory=FakeClient)
        self.assertEqual(result["status"], "complete")
        self.assertEqual(result["dispatched"], 36)
        self.assertEqual(len(FakeClient.calls), 36)
        self.assertEqual(len(freeze(recovered)["predictions"]), 36)
        self.assertEqual(result["dispatched"] + 12, 48)
        path = next((parent / "traces").glob("*.dispatch.json"))
        with path.open("a") as stream:
            stream.write(" ")
        with self.assertRaises(ValueError):
            validate(recovered)


if __name__ == "__main__":
    TOKENIZER = sys.argv.pop(1)
    unittest.main()
