"""Label-free synthetic migration check; only synthetic references are used for scoring."""
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
import socket
import tempfile
import unittest
from unittest.mock import patch

from ..traces import digest, file_digest, read, write
from . import depth_all_api_recovery as recovery
from .prototype.client import ServiceError


class Checks(unittest.TestCase):
    def test_reuse_reallocation_freeze_score_and_tamper(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            parent, run = root / "parent", root / "new"
            parent.mkdir()
            depth = parent / "depth.png"
            depth.write_bytes(b"synthetic-depth-bytes")
            items, predictions = [], []
            for i in range(18):
                item = {
                    "index": i, "api_id": f"pool-{i % 6 + 1:02d}",
                    "problem": {"sample_id": f"synthetic-{i}", "task_type": f"type-{i % 6}",
                                "input_sha256": digest(i), "options": ["first", "second"]},
                    "arm_order": ["direct", "depth"] if i < 9 else ["depth", "direct"],
                    "depth_images": [{"path": "depth.png", "sha256": file_digest(depth)}],
                    "payload_sha256": {arm: digest({"index": i, "arm": arm})
                                       for arm in ("direct", "depth")},
                }
                items.append(item)
                for arm in item["arm_order"]:
                    status = "answered" if i < 13 or (i == 15 and arm == "direct") else (
                        "outcome_unknown" if (i in (13, 15) and arm == "depth") else "not_dispatched")
                    predictions.append({
                        "index": i, "sample_id": item["problem"]["sample_id"],
                        "task_type": item["problem"]["task_type"],
                        "input_sha256": item["problem"]["input_sha256"], "api_id": item["api_id"],
                        "arm": arm, "status": status, "answer": "A" if status == "answered" else None,
                    })
            manifest = {"questions": 18, "items": items}
            manifest["run_identity"] = digest(manifest)
            frozen = {"predictions": predictions, "run_identity": manifest["run_identity"]}
            frozen["freeze_identity"] = digest(frozen)
            write(parent / "manifest.json", manifest)
            write(parent / "frozen.json", frozen)
            execution = {"status": "interrupted", "dispatched": 29}
            write(root / "api-reallocation.json", {
                "parent_run_identity": manifest["run_identity"], "user_instruction": "api 全都用上",
                "planned_new_paired_requests": 10, "planned_cumulative_paired_requests": 51,
                "old_processes_confirmed_exited": True,
            })
            def parent_evidence(directory):
                current, prediction = read(parent / "manifest.json"), read(parent / "frozen.json")
                if (digest({k: v for k, v in current.items() if k != "run_identity"})
                        != current["run_identity"] or digest({
                            k: v for k, v in prediction.items() if k != "freeze_identity"})
                        != prediction["freeze_identity"]):
                    raise ValueError("synthetic_parent_changed")
                return current, prediction, execution
            calls = []
            class FakeClient:
                fail = False
                def __init__(self, api, key):
                    self.api = api
                def call(self, body):
                    calls.append((self.api, body["index"], body["arm"]))
                    if self.fail:
                        raise ServiceError("synthetic_service_error", 503)
                    return {"text": "``A``", "tool_call_count": 0}
            keys = {api: f"synthetic-memory-secret-{api}" for api in recovery.ALL_APIS}
            with (
                patch.object(socket.socket, "connect", side_effect=AssertionError("network_disabled")),
                patch.object(recovery, "parent_evidence", side_effect=parent_evidence),
                patch("GPTexp.MMSIGPT.depth_comparison_report.ancestors",
                      return_value=[(None, None, None, {"dispatched": 6})] * 2),
                patch.object(recovery, "request",
                             side_effect=lambda directory, item, arm: {"index": item["index"], "arm": arm}),
                redirect_stdout(StringIO()),
            ):
                prepared = recovery.prepare(run, parent)
                self.assertEqual(len(prepared["reused_items"]), 13)
                self.assertEqual([i["api_id"] for i in prepared["new_items"]], recovery.NEW_APIS)
                self.assertEqual(prepared["cumulative_paired_requests_limit"], 51)
                result = recovery.execute(run, keys, client_factory=FakeClient)
                self.assertEqual(result["dispatched"], 10)
                self.assertEqual(result["status"], "complete")
                self.assertEqual(len(calls), 10)
                self.assertEqual({i for _, i, _ in calls}, set(range(13, 18)))
                for api in recovery.NEW_APIS:
                    self.assertEqual(sum(call[0] == api for call in calls), 2)
                final = recovery.freeze(run)
                self.assertEqual(len(final["predictions"]), 36)
                self.assertEqual(sum(p["source"] == "parent" for p in final["predictions"]), 26)
                references = root / "synthetic-references.json"
                write(references, {"references": [
                    {"benchmark": "mmsi", "sample_id": f"synthetic-{i}", "reference_answer": "A"}
                    for i in range(18)]})
                scored = recovery.score(run, references, file_digest(references))
                self.assertEqual(scored["overall"]["complete_pairs"], 18)
                self.assertEqual(scored["overall"]["delta"], 0)
                with self.assertRaises(ValueError):
                    recovery.execute(run, keys, client_factory=FakeClient)
                changed = read(parent / "frozen.json")
                changed["predictions"][0]["answer"] = "B"
                write(parent / "frozen.json", changed)
                with self.assertRaises(ValueError):
                    recovery.validate_frozen(run, prepared)
                write(parent / "frozen.json", frozen)
                result_file = next((run / "traces").glob("*.result.json"))
                changed = read(result_file)
                changed["response"]["text"] = "B"
                write(result_file, changed)
                with self.assertRaises(ValueError):
                    recovery.validate_frozen(run, prepared)
                failed = root / "failed"
                recovery.prepare(failed, parent)
                FakeClient.fail = True
                stopped = recovery.execute(failed, keys, client_factory=FakeClient)
                self.assertEqual(stopped["status"], "stopped_service_error")
                recovery.freeze(failed)
                with self.assertRaises(ValueError):
                    recovery.score(failed, root / "must-never-open", "none")


if __name__ == "__main__":
    unittest.main()
