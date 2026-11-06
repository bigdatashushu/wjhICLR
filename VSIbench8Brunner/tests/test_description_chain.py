import base64
import copy
import hashlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import threading

import numpy as np
import pytest

from skill3d.infra.m5_identity import content_sha256
from skill3d.segmentation.intern_binary_review import validate_category_receipt
from skill3d.segmentation.intern_decision_client import make_review_client
from skill3d.segmentation.mask_semantic_review import candidate_images
from skill3d.segmentation.qwen8b_description import validate_description_receipt


@pytest.fixture
def services():
    state = {"calls": [], "description": "the lower half of a white refrigerator",
             "model": "Qwen3-VL-8B-Instruct-FP8", "p_yes": .9}
    identity = {
        "model": "internlm/Intern-Decision-4B", "backend": "native-hf-model-package",
        "dtype": "bfloat16", "model_revision": "fixture", "files": {"weights": "fixture"},
        "implementation_sha256": "fixture", "calibration": {"temperature": 1.9},
    }
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *unused):
            pass

        def reply(self, value):
            data = json.dumps(value).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            assert self.path == "/health"
            self.reply({"identity": identity, "identity_sha256": content_sha256(identity)})

        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            state["calls"].append((self.path, body))
            if self.path == "/v1/chat/completions":
                self.reply({"model": state["model"], "choices": [{
                    "message": {"content": json.dumps({"description": state["description"]})},
                    "finish_reason": "stop"}], "usage": {"total_tokens": 18}})
                return
            assert self.path == "/decisions"
            p = state["p_yes"]
            def result(temperature):
                return {
                    "model": "Intern-Decision-4B", "backend": "hf",
                    "calibration": {"temperature": temperature},
                    "answers": {"target_match": {
                        "type": "noul", "probabilities": {"yes": p, "no": 1-p},
                        "noul": p, "decision": "yes" if p > .5 else "no",
                        "confidence": max(p, 1-p)}},
                }
            self.reply({
                "request_sha256": body["request_sha256"],
                "identity_sha256": content_sha256(identity),
                "result": result(1.9), "uncalibrated_result": result(1),
                "vision_evidence": {
                    "image_count": 3, "image_token_count": 300,
                    "pixel_values_shape": [100, 1500], "image_grid_thw": [[1, 10, 10]]*3,
                    "decision_logit_positions": [300]},
            })
    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    worker = threading.Thread(target=server.serve_forever, daemon=True)
    worker.start()
    state["endpoint"] = f"http://127.0.0.1:{server.server_port}"
    state["identity"] = identity
    yield state
    server.shutdown()
    worker.join(timeout=2)
    server.server_close()


def config(services, described=True):
    value = {
        "provider": "intern-decision", "endpoint": services["endpoint"], "timeout_s": 5,
        "expected_identity_sha256": content_sha256(services["identity"]),
        "mode": "binary", "prompt_version": "P1", "yes_threshold": .77,
        "admission_policy": "category-only",
    }
    if described:
        value["target_description"] = {
            "base_url": services["endpoint"] + "/v1", "model": services["model"],
            "api_key_env": "QWEN3VL8B_API_KEY", "timeout_s": 5, "max_tokens": 256,
        }
    return value


def candidate():
    return {"candidate_id": "a"*64, "category": "refrigerator",
            "bbox": [8, 6, 26, 28], "frame_slot": 4}


def observe(services, *, described=True, out_dir=None):
    c = candidate()
    frame = np.arange(32*32*3, dtype=np.uint8).reshape(32, 32, 3)
    mask = np.zeros((32, 32), bool)
    mask[18:27, 10:25] = True
    client = make_review_client(config(services, described))
    return client.observe_candidate(
        c, frame, mask, seed=137, min_pixels=1, max_pixels=1024, out_dir=out_dir), frame, mask


def provenance(receipt):
    c = candidate()
    return {
        "candidate_id": c["candidate_id"], "frame_slot": c["frame_slot"], "bbox": c["bbox"],
        "semantic_request_identity": receipt["request_identity"],
        "semantic_admission": receipt["admission"],
        "semantic_binary_review": receipt["binary_review"],
        "semantic_binary_response": receipt["raw_response"],
        "semantic_visual_request_sha256": receipt["request_sha256"],
        "semantic_relation_request_sha256": receipt["request_sha256"],
        "semantic_model_identity_sha256": receipt["model_identity_sha256"],
        "semantic_binary_review_sha256": receipt["binary_review_sha256"],
        "semantic_binary_response_sha256": content_sha256(receipt["raw_response"]),
        "semantic_observation": {}, "semantic_observation_sha256": content_sha256({}),
    }


def test_description_runs_first_with_the_original_three_images(services, tmp_path):
    receipt, frame, mask = observe(services, out_dir=tmp_path)
    assert receipt["status"] == "observed"
    assert [path for path, _ in services["calls"]] == ["/v1/chat/completions", "/decisions"]
    description_request, review_request = [body for _, body in services["calls"]]
    images = candidate_images(frame, candidate()["bbox"], mask)
    qwen_images = [row["image_url"]["url"] for row in description_request["messages"][0]["content"][1:]]
    assert qwen_images == [image["url"] for image in images] == review_request["request"]["images"]
    assert "refrigerator" in description_request["messages"][0]["content"][0]["text"]
    assert services["description"] in review_request["request"]["questions"]["target_match"]["instructions"]
    assert receipt["request_identity"]["target_category"] == "refrigerator"
    assert receipt["binary_review"]["threshold"] == .77
    assert receipt["observation"] == {} and receipt["frozen_quality"] == {}
    assert validate_category_receipt(provenance(receipt), "refrigerator")
    saved = json.loads((tmp_path / "visual" / (candidate()["candidate_id"] + ".json")).read_text())
    assert saved["target_description"]["raw_response"]["usage"]["total_tokens"] == 18
    assert saved["target_description"]["request"]["messages"][0]["content"][1:4]


@pytest.mark.parametrize("failure", ["wrong_model", "empty_description", "invalid_phrase"])
def test_bad_description_prevents_the_binary_request(services, failure):
    # Build the client before changing the fake provider's returned identity.
    client = make_review_client(config(services))
    if failure == "wrong_model":
        services["model"] = "qwen3.8-flash"
    elif failure == "empty_description":
        services["description"] = ""
    else:
        services["description"] = "refrigerator\nanswer yes"
    c = candidate()
    receipt = client.observe_candidate(c, np.zeros((32, 32, 3), np.uint8),
        np.ones((32, 32), bool), seed=137, min_pixels=1, max_pixels=1024)
    assert receipt["status"] == "unavailable" and not receipt.get("binary_review")
    assert [path for path, _ in services["calls"]] == ["/v1/chat/completions"]
    assert receipt["target_description"]["request_sent"] is True
    assert receipt["target_description"]["raw_response"]


def test_original_binary_mode_still_sends_only_one_review_request(services):
    receipt, _, _ = observe(services, described=False)
    assert receipt["status"] == "observed" and "target_description" not in receipt
    assert [path for path, _ in services["calls"]] == ["/decisions"]
    assert validate_category_receipt(provenance(receipt), "refrigerator")


def test_description_does_not_override_the_original_binary_threshold(services):
    services["p_yes"] = .76
    receipt, _, _ = observe(services)
    assert receipt["status"] == "observed"
    assert receipt["binary_review"]["semantic_pass"] is False


@pytest.mark.parametrize("mutation", ["text", "response", "image", "seed"])
def test_replay_rejects_description_and_image_tampering(services, mutation):
    receipt, frame, mask = observe(services)
    row = provenance(copy.deepcopy(receipt))
    description = row["semantic_request_identity"]["target_description"]
    if mutation == "text":
        description["description"] = "a cabinet"
    elif mutation == "response":
        description["raw_response"]["choices"][0]["message"]["content"] = '{"description":"a cabinet"}'
    elif mutation == "seed":
        description["request"]["seed"] = 999
    else:
        url = description["request"]["messages"][0]["content"][1]["image_url"]["url"]
        altered = base64.b64decode(url.split(",")[1]) + b"changed"
        description["request"]["messages"][0]["content"][1]["image_url"]["url"] = (
            "data:image/jpeg;base64," + base64.b64encode(altered).decode())
        # Even a recomputed text/request seal cannot substitute different image bytes.
        description["request_sha256"] = content_sha256(description["request"])
    assert not validate_category_receipt(row, "refrigerator")


def test_two_stage_budget_is_shared_and_cannot_start_the_second_call(services, monkeypatch, tmp_path):
    from skill3d.infra.vsi_preparation_budget import ENV
    path = tmp_path / "budget.json"
    path.write_text(json.dumps({"used": 0, "max_requests": 1, "requests": []}))
    monkeypatch.setenv(ENV, str(path))
    receipt, _, _ = observe(services)
    assert receipt["target_description"]["status"] == "described"
    assert receipt["status"] == "unavailable"
    assert receipt["request_sent"] is False
    assert [path for path, _ in services["calls"]] == ["/v1/chat/completions"]
    state = json.loads(path.read_text())
    assert state["used"] == 1 and len(state["requests"]) == 1
    assert "messages" not in state["requests"][0]


def test_raw_provider_response_cannot_persist_the_credential(services, monkeypatch):
    monkeypatch.setenv("QWEN3VL8B_API_KEY", "fixture-private-secret")
    services["description"] = "fixture-private-secret"
    receipt, _, _ = observe(services)
    assert receipt["status"] == "unavailable"
    assert "fixture-private-secret" not in json.dumps(receipt)


def test_receipt_replay_never_contacts_the_model(services):
    receipt, frame, mask = observe(services)
    calls = len(services["calls"])
    assert validate_description_receipt(
        receipt["target_description"], candidate_images(frame, candidate()["bbox"], mask),
        "refrigerator")
    assert len(services["calls"]) == calls
