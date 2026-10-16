from __future__ import annotations

import json

import numpy as np

from skill3d.segmentation import mask_semantic_review as msr
from skill3d.synthesis.request_context import current_request_phase


class Client:
    model = "qwen3.8-flash"
    last_usage = {"total_tokens": 7}
    last_response_identity = {"id": "r1", "model": model}

    def __init__(self):
        self.calls = []

    def audit_identity(self, *, seed=None):
        return {"provider": "test", "model": self.model, "seed": seed,
                "cache_eligible": False}

    def chat(self, messages, **kwargs):
        phase = current_request_phase()
        self.calls.append((phase, messages, kwargs))
        if phase == "m5_mask_visual":
            text = messages[0]["content"][0]["text"]
            assert "paper bag" not in text
            assert [row["type"] for row in messages[0]["content"]] == [
                "text", "image_url", "image_url", "image_url"]
            return json.dumps({
                "observed_object": "clipboard", "alternate_object": "none",
                "visibility": "clear", "mask_purity": "clean",
                "mask_completeness": "whole",
            })
        pairs = json.loads(messages[0]["content"].split("Input pairs:\n", 1)[1])
        return json.dumps({"results": [
            {"id": row["id"], "relation": "mismatch"} for row in pairs]})


def test_target_is_absent_from_visual_request_and_relation_is_separate(tmp_path):
    frame = np.zeros((64, 80, 3), dtype=np.uint8)
    mask = np.zeros((64, 80), dtype=bool)
    mask[10:40, 15:45] = True
    candidate = {
        "candidate_id": "candidate-1", "category": "paper bag",
        "bbox": [14, 9, 46, 41],
    }
    client = Client()
    visual = msr.observe_candidate(
        candidate, frame, mask, client, seed=137,
        min_pixels=65536, max_pixels=1048576, out_dir=tmp_path)
    assert visual["status"] == "observed"
    assert visual["observation"]["observed_object"] == "clipboard"
    assert visual["request_sent"]
    relations, stats = msr.compare_categories([{
        "candidate_id": candidate["candidate_id"],
        "category": candidate["category"],
        "observation": visual["observation"],
    }], client, seed=137, out_dir=tmp_path)
    assert relations["candidate-1"]["relation"] == "mismatch"
    assert stats == {"actual_requests": 1, "n_batches": 1}
    assert [phase for phase, _messages, _kwargs in client.calls] == [
        "m5_mask_visual", "m5_mask_relation"]
    assert client.calls[-1][2]["max_tokens"] == msr.RELATION_MAX_TOKENS
    assert list((tmp_path / "visual").glob("*.json"))
    assert list((tmp_path / "relation").glob("*.json"))


def test_relation_budget_covers_full_batch_of_sha256_ids():
    client = Client()
    rows = [{
        "candidate_id": f"{index:02x}" + "a" * 62,
        "category": "paper bag",
        "observation": {"observed_object": "box"},
    } for index in range(32)]
    relations, stats = msr.compare_categories(
        rows, client, seed=137, batch_size=32)
    assert len(relations) == 32
    assert stats == {"actual_requests": 1, "n_batches": 1}
    assert client.calls[0][2]["max_tokens"] == 2048


def test_visual_and_relation_schema_fail_closed():
    bad_visual = json.dumps({
        "observed_object": "chair", "visibility": "clear",
        "mask_purity": "clean", "mask_completeness": "whole",
    })
    try:
        msr.validate_visual_response(bad_visual)
        assert False, "missing alternate_object must fail"
    except ValueError:
        pass
    bad_relation = '{"results":[{"id":"extra","relation":"match"}]}'
    try:
        msr.validate_relation_response(bad_relation, ["wanted"])
        assert False, "wrong IDs must fail"
    except ValueError:
        pass
