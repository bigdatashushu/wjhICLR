"""Synthetic checks for one attempt per API, no pose context, and immutable output."""
from __future__ import annotations

from dataclasses import asdict
from io import BytesIO
import json
from pathlib import Path
import tempfile
from unittest.mock import patch

from PIL import Image

from ..traces import digest, read, write
from . import camobj_depth_trial as trial
from .depth_comparison.depth import render
from .depth_comparison.inputs import depth_messages
from .depth_prompt import build_annotation
from .prototype.client import ServiceError
from .prototype.inputs import load_images, original_input_sha256
from .prototype.protocol import messages, payload


def main():
    calls = []

    class Fake:
        def __init__(self, api, key):
            self.api = api
        def assert_public(self, value):
            pass
        def call(self, body):
            calls.append(self.api)
            return {"text": "`A`", "tool_call_count": 0}

    with tempfile.TemporaryDirectory(prefix="mmsi-camobj-depth-synthetic-") as temporary:
        root = Path(temporary)
        raw = BytesIO()
        Image.new("RGB", (8, 8), (120, 70, 20)).save(raw, format="PNG")
        paths = [root / f"image-{i}.png" for i in range(2)]
        for path in paths:
            path.write_bytes(raw.getvalue())
        items = []
        for i in range(5):
            api = f"fake-{i}"
            problem = {
                "benchmark": "mmsi", "source_split": "test", "sample_id": f"synthetic-{i}",
                "question": "Which direction?", "options": ["Left", "Right"],
                "image_paths": list(map(str, paths)), "task_type": trial.TASK,
            }
            import hashlib
            problem["input_sha256"] = original_input_sha256(
                problem, [hashlib.sha256(raw.getvalue()).hexdigest()] * 2)
            images = load_images(problem)
            direct = payload(messages(problem, images), trial.inference(api))
            items.append({
                "index": i, "source_index": i, "problem": problem,
                "images": [r[0] for r in images], "api_id": api, "arm_order": ["depth"],
                "prepared_directory": "unused", "artifact_sha256": "0" * 64,
                "parent_direct_payload_sha256": digest(direct),
            })
        selected = {
            "schema": trial.SCHEMA, "items": items, "questions": 5, "max_requests": 5,
            "api_ids": [r["api_id"] for r in items], "reference_answer_visible": False,
            "sample_role": "reused_development_diagnostic", "parent_evidence": {"synthetic": True},
        }
        selected["selection_identity"] = digest(selected)
        import numpy as np

        def fake_prepare(directory, item, count):
            images = load_images(item["problem"], item["images"])
            annotation = build_annotation([(8, 8)] * 2, count)
            depths = [render(np.ones((8, 8)), np.ones((8, 8)), (8, 8)) for _ in range(2)]
            receipts = []
            for j, (data, receipt) in enumerate(depths):
                path = f"depth/{item['index']:04d}-{j:02d}.png"
                (directory / "depth").mkdir(exist_ok=True)
                (directory / path).write_bytes(data)
                receipts.append({**receipt, "path": path})
            body = payload(depth_messages(item["problem"], images, depths, annotation),
                           trial.inference(item["api_id"]))
            return {**item, "depth_images": receipts, "annotation": asdict(annotation),
                    "added_text_tokens": 40,
                    "payload_sha256": {"direct": item["parent_direct_payload_sha256"],
                                       "depth": digest(body)}}

        directory = root / "run"
        with patch.object(trial, "tokenizer", return_value=lambda text: 40), \
                patch.object(trial, "prepare_item", side_effect=fake_prepare):
            trial.prepare(directory, selected, "unused")
        manifest = trial.validate(directory, verify_requests=True)
        for item in manifest["items"]:
            body = trial.request(directory, item, "depth")
            encoded = json.dumps(body)
            assert all(text not in encoded for text in ("VGGT", "Question frame", "camera_center"))
            assert len([c for row in body["input"] for c in row["content"]
                        if c["type"] == "input_image"]) == 4
        keys = {api: "synthetic-key" for api in manifest["api_ids"]}
        assert trial.execute(directory, keys, client_factory=Fake)["dispatched"] == 5
        assert sorted(calls) == sorted(keys)
        trial.freeze(directory)
        try:
            trial.execute(directory, keys, client_factory=Fake)
        except ValueError:
            pass
        else:
            raise AssertionError("repeat_dispatch_allowed")
        assert len(calls) == 5
        result_path = directory / "traces/0000-depth.result.json"
        record = read(result_path)
        record["answer"] = "B"
        write(result_path, record)
        try:
            trial.validate_frozen(directory, manifest)
        except ValueError:
            pass
        else:
            raise AssertionError("frozen_tampering_allowed")

        class Failed(Fake):
            def call(self, body):
                calls.append(self.api)
                raise ServiceError("synthetic_failure", 503)

        failed_dir = root / "failure"
        with patch.object(trial, "tokenizer", return_value=lambda text: 40), \
                patch.object(trial, "prepare_item", side_effect=fake_prepare):
            trial.prepare(failed_dir, selected, "unused")
        calls.clear()
        outcome = trial.execute(failed_dir, keys, client_factory=Failed)
        assert outcome["status"] == "stopped_service_error"
        assert len(calls) <= 5 and len(calls) == len(set(calls))
        trial.freeze(failed_dir)
    print('{"status":"passed","model_requests":0,"checks":'
          '["four_images_no_pose","five_unique_single_attempt_slots","no_reentry",'
          '"frozen_tamper_rejected","service_failure_no_retry"]}')


if __name__ == "__main__":
    main()
