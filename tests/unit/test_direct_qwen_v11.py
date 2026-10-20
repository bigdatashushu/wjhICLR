import json
from pathlib import Path
import threading
import time

import numpy as np
import pytest

from skill3d.adapters import vsibench_loader as vl
from skill3d.adapters.frame_set import with_pixel_identity
from skill3d.evaluation.direct_qwen_v11 import (
    DIRECT_INDEX_SCHEMA,
    DIRECT_RESPONSE_SCHEMA,
    DIRECT_RUN_SCHEMA,
    DirectQwenError,
    execute_batch,
    execute_direct_request,
    import_reused_responses,
    load_input_receipt,
    load_request_index,
    load_safe_metadata,
    prepare_direct_request,
    write_json_atomic,
)
from skill3d.synthesis.client_errors import ServiceUnavailable


def _contract():
    return {
        "model_provider": "bailian",
        "model": "qwen3.8-flash",
        "model_revision": "",
        "prompt_version": "eval-visual-v1",
        "seed": 137,
        "temperature": 0.0,
        "enable_thinking": False,
        "skills_visible": False,
        "tools_visible": False,
        "geometry_visible": False,
        "derived_images_visible": False,
        "reference_answer_visible": False,
    }


def _index_row(qa_id="10"):
    return {
        "qa_id": qa_id,
        "scene_id": "scene-1",
        "question_type": "object_counting",
        "split": "induction",
        "system_variant_ids": [f"{qa_id}@g1m0"],
    }


def _prepared(qa_id="10"):
    row = _index_row(qa_id)
    metadata = {
        "qa_id": qa_id,
        "id": qa_id,
        "dataset": "scannet",
        "scene_name": "scene-1",
        "question_type": "object_counting",
        "question": f"How many chairs are visible in sample {qa_id}?",
        "options": [],
    }
    pixels = [
        np.full((8, 8, 3), index, dtype=np.uint8)
        for index in range(32)
    ]
    planned = vl.uniform_frame_ids(320, 32)
    source = vl.VideoSource(
        path=Path("/frozen/scannet/scene-1.mp4"),
        tag="",
        convention="primary",
    )
    loaded = {
        "pixels": pixels,
        "planned": planned,
        "missing": [],
        "source": source,
        "total": 320,
        "fps": 30.0,
    }
    frame_set = vl.build_frame_set(
        320, n_frames=32, fps=30.0, episode_id=qa_id,
        dataset_id="scannet", video_id="scene-1", scene_name="scene-1",
        readable_frame_ids=planned,
    )
    frame_set = with_pixel_identity(frame_set, pixels)
    receipt = {
        "scene_id": "scene-1",
        "frame_set_hash": frame_set.frame_set_hash,
        "source_frame_indices": frame_set.source_frame_indices,
        "pixel_hashes": frame_set.pixel_sha256,
        "receipt_paths": ["/frozen/input_receipts/scene-1.json"],
    }
    return prepare_direct_request(
        index_row=row,
        metadata_row=metadata,
        loaded=loaded,
        input_receipt=receipt,
        contract=_contract(),
        endpoint_sha256="e" * 64,
        max_tokens=8192,
    )


def test_request_index_hard_locks_qwen38_flash(tmp_path):
    path = tmp_path / "requests.json"
    path.write_text(json.dumps({
        "schema_version": DIRECT_INDEX_SCHEMA,
        "status": "planned_not_executed",
        "contract": _contract(),
        "rows": [_index_row()],
    }))
    loaded = load_request_index(path)
    assert loaded["contract"]["model"] == "qwen3.8-flash"
    assert loaded["rows"][0]["qa_id"] == "10"

    changed = json.loads(path.read_text())
    changed["contract"]["model"] = "qwen3-vl"
    path.write_text(json.dumps(changed))
    with pytest.raises(DirectQwenError, match="qwen3.8-flash"):
        load_request_index(path)


def test_safe_metadata_preserves_numeric_zero_id(tmp_path):
    path = tmp_path / "metadata.jsonl"
    path.write_text(json.dumps({
        "id": 0,
        "dataset": "scannet",
        "scene_name": "scene-0",
        "question_type": "object_counting",
        "question": "How many chairs?",
        "options": None,
        "ground_truth": "3",
    }) + "\n")
    rows = load_safe_metadata(path)
    assert rows["0"]["qa_id"] == "0"
    assert "ground_truth" not in rows["0"]


def test_complete_input_receipt_wins_over_matching_partial(tmp_path):
    complete_root = tmp_path / "complete"
    partial_root = tmp_path / "partial"
    full = {
        "scene_id": "scene-1",
        "frame_set_hash": "f" * 64,
        "pixel_hashes": [f"{index:064x}" for index in range(32)],
        "source_frame_indices": list(range(32)),
    }
    write_json_atomic(complete_root / "scene-1.json", full)
    write_json_atomic(partial_root / "scene-1.json", {
        "scene_id": "scene-1",
        "frame_set_hash": "f" * 64,
    })
    receipt = load_input_receipt(
        "scene-1", [partial_root, complete_root])
    assert receipt["pixel_hashes"] == full["pixel_hashes"]
    assert receipt["receipt_paths"] == [
        str((complete_root / "scene-1.json").resolve())]
    assert receipt["partial_receipt_paths"] == [
        str((partial_root / "scene-1.json").resolve())]


def test_preflight_binds_exact_32_frame_bytes_and_model_contract():
    prepared = _prepared()
    receipt = prepared.preflight
    assert receipt["n_images"] == 32
    assert len(receipt["pixel_sha256"]) == 32
    assert len(prepared.messages[0]["content"]) == 33
    assert receipt["request_identity"] == {
        "schema_version": "v11-direct-qwen-request/1.0",
        "provider": "bailian",
        "model": "qwen3.8-flash",
        "model_revision": "",
        "endpoint_sha256": "e" * 64,
        "prompt_version": "eval-visual-v1",
        "messages_sha256": receipt["messages_sha256"],
        "seed": 137,
        "temperature": 0.0,
        "max_tokens": 8192,
        "enable_thinking": False,
    }
    assert not receipt["reference_answer_visible"]


def test_preflight_rejects_changed_frozen_pixels():
    prepared = _prepared()
    row = _index_row()
    metadata = {
        "qa_id": "10", "id": "10", "dataset": "scannet",
        "scene_name": "scene-1", "question_type": "object_counting",
        "question": "How many chairs?", "options": [],
    }
    pixels = [np.zeros((8, 8, 3), np.uint8) for _ in range(32)]
    planned = vl.uniform_frame_ids(320, 32)
    loaded = {
        "pixels": pixels, "planned": planned, "missing": [],
        "source": vl.VideoSource(
            Path("/frozen/scannet/scene-1.mp4"), "", "primary"),
        "total": 320, "fps": 30.0,
    }
    receipt = {
        "scene_id": "scene-1",
        "frame_set_hash": prepared.preflight["frame_set_hash"],
        "source_frame_indices": prepared.preflight["source_frame_indices"],
        "pixel_hashes": prepared.preflight["pixel_sha256"],
        "receipt_paths": ["/frozen/input_receipts/scene-1.json"],
    }
    with pytest.raises(DirectQwenError, match="pixels changed"):
        prepare_direct_request(
            index_row=row, metadata_row=metadata, loaded=loaded,
            input_receipt=receipt, contract=_contract(),
            endpoint_sha256="e" * 64, max_tokens=8192)


class _Client:
    def __init__(self, *, answer="2", error=None, barrier=None, state=None):
        self.answer = answer
        self.error = error
        self.barrier = barrier
        self.state = state
        self.last_usage = {}
        self.last_generation = {}
        self.last_response_identity = {}

    def chat(self, messages, **kwargs):
        assert len(messages[0]["content"]) == 33
        assert kwargs == {
            "max_tokens": 8192,
            "seed": 137,
            "temperature": 0.0,
            "enable_thinking": False,
        }
        if self.state is not None:
            with self.state["lock"]:
                self.state["active"] += 1
                self.state["max_active"] = max(
                    self.state["max_active"], self.state["active"])
        try:
            if self.barrier is not None:
                self.barrier.wait(timeout=2)
            if self.error is not None:
                raise self.error
            time.sleep(0.01)
            self.last_usage = {"prompt_tokens": 100, "completion_tokens": 1}
            self.last_generation = {"finish_reason": "stop"}
            self.last_response_identity = {
                "id": "response-id",
                "model": "qwen3.8-flash",
                "created": 1,
            }
            return self.answer
        finally:
            if self.state is not None:
                with self.state["lock"]:
                    self.state["active"] -= 1


def test_four_workers_issue_four_requests_concurrently(tmp_path):
    prepared = [_prepared(str(index)) for index in range(4)]
    state = {
        "lock": threading.Lock(),
        "active": 0,
        "max_active": 0,
    }
    barrier = threading.Barrier(4)
    results, batch = execute_batch(
        prepared,
        workers=4,
        client_factory=lambda: _Client(barrier=barrier, state=state),
        response_dir=tmp_path / "responses",
    )
    assert state["max_active"] == 4
    assert all(row["accepted"] for row in results)
    assert batch["workers"] == 4
    assert batch["status_counts"] == {"accepted": 4}
    assert batch["api_wall_elapsed_s"] > 0
    assert batch["api_throughput_requests_per_s"] > 0
    assert batch["receipt_overhead_s"] >= 0
    assert len(list((tmp_path / "responses").glob("*.json"))) == 4


def test_http_429_is_terminal_and_audited():
    result = execute_direct_request(
        _prepared(),
        client_factory=lambda: _Client(
            error=ServiceUnavailable("Bailian HTTP 429: rate limited")),
    )
    assert result["status"] == "service_error"
    assert result["service_error"]
    assert not result["api_response_received"]
    assert "HTTP 429" in result["error"]


def test_prior_exact_request_is_imported_create_only(tmp_path):
    prepared = _prepared()
    source = tmp_path / "source"
    manifest = {
        "schema_version": DIRECT_RUN_SCHEMA,
        "request_index": {"sha256": "i" * 64},
        "metadata": {"sha256": "m" * 64},
        "config": {"sha256": "c" * 64},
        "contract": _contract(),
        "selected_qa_ids": ["10"],
        "request_model": "qwen3.8-flash",
        "request_model_revision": "",
        "endpoint_sha256": "e" * 64,
        "seed": 137,
        "temperature": 0.0,
        "max_tokens": 8192,
        "enable_thinking": False,
        "reference_answer_visible": False,
    }
    write_json_atomic(source / "manifest.json", manifest)
    write_json_atomic(
        source / "preflight" / "10.json", prepared.preflight)
    response = {
        "schema_version": DIRECT_RESPONSE_SCHEMA,
        "qa_id": "10",
        "request_sha256": prepared.preflight["request_sha256"],
        "status": "accepted",
        "accepted": True,
    }
    write_json_atomic(source / "responses" / "10.json", response)

    target = tmp_path / "target"
    receipt = import_reused_responses(
        output_root=target,
        selected_qa_ids=["10", "11"],
        reuse_roots=[source],
        target_manifest=manifest,
    )
    assert receipt["imported_count"] == 1
    assert receipt["imported"][0]["qa_id"] == "10"
    assert (target / "preflight" / "10.json").is_file()
    assert (target / "responses" / "10.json").is_file()
    assert (target / "responses" / "10.json").stat().st_ino == (
        source / "responses" / "10.json").stat().st_ino
