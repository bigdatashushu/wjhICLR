"""Synthetic-only recovery tests, runnable without the ML/experiment environment.

PYTHONPATH=src python3.11 -m unittest discover -s tests/unit \
    -p test_cached_episode_recovery_v11.py
"""

from copy import deepcopy
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import sqlite3
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import uuid

from skill3d.cluster.perception_client import RPC_SCHEMA, validate_result
from skill3d.evaluation.cached_episode_recovery_v11 import (
    GatewaySQLiteCache, RECOVERY_SCHEMA, RecoveryStop, ReviewedSolverCall,
    evidence_sha256, open_cached_episode_recovery, recovery_accounting,
)
from skill3d.synthesis.bailian_client import BailianClient
from skill3d.synthesis.request_audit import describe_request, payload_bytes
from skill3d.synthesis.request_context import request_phase


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


class CachedEpisodeRecoveryTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="h3d-synthetic-recovery-")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.original = self.root / "original_trace"
        self.original.mkdir()
        (self.original / "sentinel.jsonl").write_bytes(b'{"original":true}\n')
        self.continuation = self.root / "continuation"
        self.database = self.root / "synthetic-responses.sqlite3"
        # Use the actual pure Bailian methods without constructing a model client.
        codec = SimpleNamespace(
            model="synthetic-model", provider="bailian", temperature=0.0,
            max_output_tokens=256, model_revision="",
            endpoint_hash=sha(b"synthetic-endpoint"),
        )
        self.model = BailianClient.audit_identity(codec, seed=138)
        self.skills = [{
            "content_sha256": sha(b"synthetic-method"),
        }]
        self.dispatch = {
            "variant_id": "synthetic-variant", "slot_key": "synthetic/g0m0",
            "skill_identity": {"snapshot_ref": "synthetic-snapshot",
                               "manifest_sha256": sha(b"synthetic-manifest")},
            "skills_sha256": evidence_sha256(self.skills),
            "run_identity": sha(b"synthetic-run"),
        }
        self.context = {
            "run_identity": self.dispatch["run_identity"],
            "shared_input_sha256": sha(b"synthetic-input"),
            "skill_identity": deepcopy(self.dispatch["skill_identity"]),
            "skill_content_sha256": [self.skills[0]["content_sha256"]],
            "solver_config_sha256": sha(b"synthetic-config"),
            "concurrency_limit": 1, "scheduler": "synthetic-scheduler",
        }
        self.messages = []
        self.results = []
        self.calls = []
        with sqlite3.connect(self.database) as db:
            db.execute(
                "CREATE TABLE requests (request_id TEXT PRIMARY KEY, "
                "payload_sha256 TEXT NOT NULL, identity_sha256 TEXT NOT NULL, "
                "state TEXT NOT NULL, http_status INTEGER, reply BLOB)")
            for index in range(1, 5):
                messages = [{"role": "user", "content": [
                    {"type": "text", "text": f"synthetic request {index}"},
                    {"type": "image_url", "image_url": {
                        "url": "data:image/jpeg;base64,c3ludGhldGljLW9uZQ=="}},
                    {"type": "image_url", "image_url": {
                        "url": "data:image/jpeg;base64,c3ludGhldGljLXR3bw=="}},
                ]}]
                self.messages.append(messages)
                payload = BailianClient.build_payload(codec, messages, max_tokens=4096, seed=138)
                identity = describe_request(payload, model_identity=self.model)
                request = {
                    "request_index": index, "request_seed": 138, "started_at": "synthetic",
                    "retry_of": None, "request_identity": identity,
                    "frozen_context": deepcopy(self.context),
                    "request_context": {
                        "round_index": index, "phase": "program",
                        "prompt_version": "synthetic-prompt",
                        "execution_contract_sha256": sha(b"synthetic-contract"),
                        "tool_docs_version": "synthetic-docs",
                        "tool_directory_sha256": sha(b"synthetic-tools"),
                        "tool_execution_allowed": True,
                        "delivered_method_hashes": [self.skills[0]["content_sha256"]],
                        "image_round": {"index": index},
                        "images_in_order": [{"frame_slot": 0}, {"frame_slot": 1}],
                    },
                }
                usage = {"prompt_tokens": 10, "completion_tokens": 2, "total_tokens": 12}
                governor = {
                    "request_id": 100 + index, "category": "solver",
                    "context": {"variant_id": self.dispatch["variant_id"],
                                "run_identity": self.dispatch["run_identity"]},
                    "slot": 0, "started_at": float(index),
                    "status": "started" if index == 4 else "completed",
                }
                if index < 4:
                    governor.update(completed_at=float(index) + 0.1,
                                    exception_type=None, usage=deepcopy(usage))
                response_identity = {
                    "id": f"synthetic-response-{index}", "model": "synthetic-served-model",
                    "created": index, "backend_fingerprint": None,
                }
                generation = {
                    "mode": "nonstream", "usage_status": "measured", "provider": "bailian",
                    "endpoint_sha256": self.model["endpoint_sha256"],
                    "request_identity": deepcopy(identity), "finish_reason": "stop",
                    "response_identity": deepcopy(response_identity),
                }
                gateway_id = str(uuid.UUID(int=index))
                result = {
                    "schema_version": RPC_SCHEMA, "request_id": gateway_id,
                    "payload_sha256": sha(payload_bytes(payload)),
                    "audit_identity": deepcopy(self.model),
                    "content": f"synthetic cached response {index}",
                    "last_usage": usage, "last_generation": generation,
                    "last_response_identity": response_identity,
                    "last_request_identity": deepcopy(identity),
                }
                validate_result(result, request_id=gateway_id, payload=payload,
                                audit_identity=self.model)
                self.results.append(deepcopy(result))
                raw = payload_bytes(result)
                db.execute("INSERT INTO requests VALUES (?,?,?,'complete',200,?)", (
                    gateway_id, result["payload_sha256"], sha(payload_bytes(self.model)), raw))
                response = None if index == 4 else {
                    "request_index": index, "request_seed": 138,
                    "request_identity_sha256": evidence_sha256(identity),
                    "completed_at": "synthetic", "elapsed_s": 0.1,
                    "status": "completed", "exception_type": None,
                    "usage": deepcopy(usage), "response_identity": deepcopy(response_identity),
                    "actual_request_identity": deepcopy(identity),
                }
                approval = {
                    "decision": "approved_cache_replay", "review_id": f"synthetic-review-{index}",
                    "seed_record_sha256": evidence_sha256(request),
                    "governor_record_sha256": evidence_sha256(governor),
                    "response_record_sha256": evidence_sha256(response),
                    "source": {
                        "kind": "gateway_cache", "request_id": gateway_id,
                        "payload_sha256": result["payload_sha256"],
                        "identity_sha256": sha(payload_bytes(self.model)),
                        "reply_sha256": sha(raw),
                        "response_identity": deepcopy(response_identity), "usage": deepcopy(usage),
                    },
                }
                self.calls.append(ReviewedSolverCall(request, governor, approval, response))
                (self.original / f"seed-{index:03d}.json").write_bytes(payload_bytes(request))
        self.state = {
            "policy": {"max_requests": 200, "max_inflight": 1, "requests_per_second": 0.0},
            "requests": [deepcopy(call.governor_record) for call in self.calls],
            "next_start": 0.0, "peak_inflight": 1,
        }
        (self.original / "state.json").write_bytes(payload_bytes(self.state))
        self.original_bytes = self.snapshot()
        self.cache_bytes = self.database.read_bytes()
        self.traps = []
        for target in (
            "socket.socket", "socket.getaddrinfo", "urllib.request.urlopen",
            "urllib.request.OpenerDirector.open",
            "skill3d.synthesis.bailian_client.BailianClient.__init__",
            "skill3d.synthesis.bailian_client.BailianClient.chat",
            "skill3d.cluster.perception_client.RemoteBailianClient.chat",
            "skill3d.synthesis.request_governor.RequestGovernor.request",
        ):
            trap = self.enterContext(patch(target, side_effect=AssertionError(
                "network/model construction/new request is forbidden")))
            self.traps.append(trap)

    def tearDown(self):
        self.assertEqual(self.snapshot(), self.original_bytes)
        for trap in self.traps:
            trap.assert_not_called()

    def snapshot(self):
        return {str(path.relative_to(self.original)): path.read_bytes()
                for path in self.original.iterdir() if path.is_file()}

    def open(self, **overrides):
        options = {
            "frozen_dispatch": self.dispatch, "resumed_dispatch": deepcopy(self.dispatch),
            "frozen_context": self.context, "resumed_context": deepcopy(self.context),
            "skills": self.skills, "model_identity": self.model, "calls": self.calls,
            "original_trace_dir": self.original, "continuation_root": self.continuation,
            "cache_path": self.database,
        }
        options.update(overrides)
        return open_cached_episode_recovery(**options)

    def test_original_response_custody_recovers_usage_without_claiming_episode_replay(self):
        client = self.open()
        receipt = client.recover_original_responses()
        self.assertEqual(receipt["status"], "original_cached_responses_recovered")
        self.assertFalse(receipt["episode_execution_reproduced"])
        self.assertEqual(len(receipt["replayed_calls"]), 4)
        self.assertEqual(receipt["recovered_started_request_ids"], [104])
        accounting = recovery_accounting(self.state, [receipt])
        self.assertEqual(accounting["reserved_requests"], 4)
        self.assertEqual(accounting["completed_requests"], 3)
        self.assertEqual(accounting["recovered_requests"], 1)
        self.assertEqual(accounting["unsettled_started_requests"], 0)
        self.assertEqual(accounting["recovered_usage"]["total_tokens"], 12)
        self.assertFalse(client.configured)
        self.assertEqual(self.database.read_bytes(), self.cache_bytes)

    def test_completed_prefix_replays_without_usage_or_reservation_duplication(self):
        calls = self.calls[:2]
        state = {**deepcopy(self.state), "requests": [
            deepcopy(call.governor_record) for call in calls]}
        original_state = deepcopy(state)
        receipt = self.open(calls=calls).recover_original_responses()
        self.assertEqual(receipt["recovered_started_request_ids"], [])
        view = recovery_accounting(state, [receipt])
        self.assertEqual(view["reserved_requests"], 2)
        self.assertEqual(view["completed_requests"], 2)
        self.assertEqual(view["recovered_requests"], 0)
        self.assertEqual(view["recovered_usage"], {})
        self.assertEqual(view["new_requests"], 0)
        self.assertEqual(state, original_state)
        self.assertEqual(self.database.read_bytes(), self.cache_bytes)

    def test_completed_prefix_requires_response_record(self):
        call = self.calls[0]
        approval = {**call.approval, "response_record_sha256": evidence_sha256(None)}
        with self.assertRaisesRegex(RecoveryStop, "seed_response_missing"):
            self.open(calls=[replace(call, response_record=None, approval=approval)])

    def test_completed_last_call_requires_matching_governor_usage(self):
        call = self.calls[0]
        governor = {**call.governor_record, "usage": {"total_tokens": 13}}
        approval = {**call.approval, "governor_record_sha256": evidence_sha256(governor)}
        with self.assertRaisesRegex(RecoveryStop, "governor_usage_mismatch"):
            self.open(calls=[replace(call, governor_record=governor, approval=approval)])

    def test_error_last_call_cannot_be_recovered_as_completed_prefix(self):
        call = self.calls[0]
        governor = {**call.governor_record, "status": "error"}
        approval = {**call.approval, "governor_record_sha256": evidence_sha256(governor)}
        with self.assertRaisesRegex(RecoveryStop, "governor_status_mismatch"):
            self.open(calls=[replace(call, governor_record=governor, approval=approval)])

    def test_empty_replay_receipt_does_not_authorize_zero_settlement(self):
        receipt = self.open(calls=self.calls[:2]).recover_original_responses()
        receipt["replayed_calls"] = []
        receipt["identity_sha256"] = evidence_sha256({
            k:v for k,v in receipt.items() if k != "identity_sha256"})
        with self.assertRaisesRegex(RecoveryStop, "receipt_settlement_mismatch"):
            recovery_accounting(self.state, [receipt])

    def test_original_response_custody_rejects_changed_last_reply_without_settlement(self):
        client = self.open()
        with sqlite3.connect(self.database) as db:
            db.execute("UPDATE requests SET reply=? WHERE request_id=?",
                       (b'{"corrupted":true}', self.calls[-1].approval["source"]["request_id"]))
        with self.assertRaises(RecoveryStop):
            client.recover_original_responses()
        self.assertFalse((self.continuation / "recovery_receipt.json").exists())
        self.assertTrue((self.continuation / "stopped.json").exists())

    def replay(self, client, index, **kwargs):
        client.set_audit_context(self.context)
        client.set_request_context(self.calls[index - 1].seed_record["request_context"])
        with request_phase(self.calls[index - 1].seed_record["request_context"]["phase"]):
            return client.chat(self.messages[index - 1], max_tokens=4096, seed=137, **kwargs)

    def reapprove(self, index, **changes):
        call = replace(self.calls[index - 1], **changes)
        approval = deepcopy(call.approval)
        approval["seed_record_sha256"] = evidence_sha256(call.seed_record)
        approval["governor_record_sha256"] = evidence_sha256(call.governor_record)
        approval["response_record_sha256"] = evidence_sha256(call.response_record)
        self.calls[index - 1] = replace(call, approval=approval)

    def update_cache(self, index, **columns):
        with sqlite3.connect(self.database) as db:
            assignments = ", ".join(f"{key}=?" for key in columns)
            db.execute(f"UPDATE requests SET {assignments} WHERE request_id=?", (
                *columns.values(), self.calls[index - 1].approval["source"]["request_id"]))

    def as_round_trace(self, index):
        result = self.results[index - 1]
        record = {
            "index": index, "response_text": result["content"],
            "generation": {**deepcopy(result["last_generation"]),
                           "usage": deepcopy(result["last_usage"])},
        }
        approval = deepcopy(self.calls[index - 1].approval)
        source = approval["source"]
        source["kind"] = "round_trace"
        del source["reply_sha256"]
        source.update(trace_ref=f"synthetic-original-round:{index}",
                      record_sha256=evidence_sha256(record))
        self.calls[index - 1] = replace(self.calls[index - 1],
                                       round_record=record, approval=approval)

    def test_four_original_replies_then_explicit_started_settlement(self):
        client = self.open()
        for index in range(1, 5):
            self.assertEqual(self.replay(client, index), f"synthetic cached response {index}")
            self.assertEqual(client.last_usage, self.results[index - 1]["last_usage"])
            self.assertEqual(client.last_generation, self.results[index - 1]["last_generation"])
            self.assertEqual(client.last_request_identity,
                             self.calls[index - 1].seed_record["request_identity"])
        receipt = client.finish_replay()
        self.assertEqual(receipt["schema_version"], RECOVERY_SCHEMA)
        self.assertEqual(receipt["recovered_started_request_ids"], [104])
        self.assertEqual(receipt["new_requests"], 0)
        self.assertEqual(self.state["requests"][-1]["status"], "started")
        view = recovery_accounting(self.state, [receipt])
        self.assertEqual(view["reserved_requests"], 4)
        self.assertEqual(view["completed_requests"], 3)
        self.assertEqual(view["recovered_requests"], 1)
        self.assertEqual(view["unsettled_started_requests"], 0)
        self.assertEqual(view["recovered_usage"], self.results[-1]["last_usage"])
        self.assertEqual(self.database.read_bytes(), self.cache_bytes)
        self.assertEqual(len(list(client.continuation_trace_dir.glob("replayed-*.json"))), 4)
        provenance = json.loads((client.continuation_trace_dir / "replayed-004.json").read_bytes())
        self.assertEqual(provenance["source"], self.calls[-1].approval["source"])
        self.assertNotIn("synthetic cached response", json.dumps(provenance))
        self.assertNotIn("response_text", self.calls[-1].seed_record)

    def test_cache_miss_is_terminal_no_fallback(self):
        client = self.open()
        with sqlite3.connect(self.database) as db:
            db.execute("DELETE FROM requests WHERE request_id=?",
                       (self.calls[0].approval["source"]["request_id"],))
        with self.assertRaisesRegex(RecoveryStop, "cache_miss"):
            self.replay(client, 1)
        self.assertFalse(client.configured)
        self.assertEqual(client.last_usage, {})
        self.assertFalse((self.continuation / "recovery_receipt.json").exists())
        with self.assertRaisesRegex(RecoveryStop, "recovery_closed"):
            self.replay(client, 1)

    def test_missing_database_is_not_created(self):
        self.database.unlink()
        client = self.open()
        with self.assertRaisesRegex(RecoveryStop, "cache_unavailable"):
            self.replay(client, 1)
        self.assertFalse(self.database.exists())

    def test_database_has_no_expected_schema(self):
        with sqlite3.connect(self.database) as db:
            db.execute("DROP TABLE requests")
        client = self.open()
        with self.assertRaisesRegex(RecoveryStop, "cache_invalid"):
            self.replay(client, 1)

    def test_payload_and_model_hash_conflicts(self):
        for column in ("payload_sha256", "identity_sha256"):
            with self.subTest(column=column):
                self.update_cache(1, **{column: "f" * 64})
                client = self.open(continuation_root=self.root / column)
                with self.assertRaisesRegex(RecoveryStop, "cache_identity_mismatch"):
                    self.replay(client, 1)

    def test_non_200_pending_indeterminate_and_corrupt_replies(self):
        mutations = [
            {"http_status": 502}, {"state": "pending"}, {"state": "indeterminate"},
            {"reply": b"{}"}, {"reply": None},
        ]
        for index, mutation in enumerate(mutations):
            with self.subTest(mutation=mutation):
                self.update_cache(1, state="complete", http_status=200,
                                  reply=payload_bytes(self.results[0]))
                self.update_cache(1, **mutation)
                client = self.open(continuation_root=self.root / f"bad-cache-{index}")
                with self.assertRaises(RecoveryStop):
                    self.replay(client, 1)

    def test_changed_prompt_image_order_and_effective_sampling_stop(self):
        client = self.open()
        self.messages[0][0]["content"][0]["text"] += " changed"
        with self.assertRaisesRegex(RecoveryStop, "request_hash_mismatch"):
            self.replay(client, 1)
        self.messages[0][0]["content"][0]["text"] = "synthetic request 1"
        self.messages[0][0]["content"][1:] = reversed(self.messages[0][0]["content"][1:])
        client = self.open(continuation_root=self.root / "image-order")
        with self.assertRaisesRegex(RecoveryStop, "request_hash_mismatch"):
            self.replay(client, 1)
        self.messages[0][0]["content"][1:] = reversed(self.messages[0][0]["content"][1:])
        for options in ({"temperature": 0.25}, {"top_p": 0.9},
                        {"response_format": {"type": "json_object"}}, {"enable_thinking": True}):
            with self.subTest(options=options):
                client = self.open(continuation_root=self.root / str(len(list(self.root.iterdir()))))
                with self.assertRaisesRegex(RecoveryStop, "request_hash_mismatch"):
                    self.replay(client, 1, **options)

    def test_seed_and_call_index_mismatch(self):
        for key, value in (("request_seed", 139), ("request_index", 2), ("retry_of", 1)):
            with self.subTest(key=key):
                original = self.calls[0]
                request = deepcopy(original.seed_record)
                request[key] = value
                self.reapprove(1, seed_record=request)
                with self.assertRaises(RecoveryStop):
                    self.open()
                self.calls[0] = original
        self.assertFalse(self.continuation.exists())

    def test_unreviewed_and_extra_calls_are_rejected(self):
        original = self.calls[0]
        for approval in (None, {}, {**original.approval, "decision": "not_approved"}):
            self.calls[0] = replace(original, approval=approval)
            with self.assertRaisesRegex(RecoveryStop, "unreviewed_call"):
                self.open()
        self.calls[0] = original
        client = self.open()
        for index in range(1, 5):
            self.replay(client, index)
        with self.assertRaisesRegex(RecoveryStop, "unreviewed_call"):
            self.replay(client, 4)
        self.assertFalse((self.continuation / "recovery_receipt.json").exists())

    def test_frozen_dispatch_inputs_skills_and_config_must_match(self):
        for key in self.dispatch:
            with self.subTest(dispatch_key=key):
                resumed = deepcopy(self.dispatch)
                resumed[key] = "changed"
                with self.assertRaisesRegex(RecoveryStop, "dispatch_mismatch"):
                    self.open(resumed_dispatch=resumed)
        for key in self.context:
            with self.subTest(context_key=key):
                resumed = deepcopy(self.context)
                resumed[key] = "changed"
                with self.assertRaisesRegex(RecoveryStop, "frozen_context_mismatch"):
                    self.open(resumed_context=resumed)
        with self.assertRaisesRegex(RecoveryStop, "skills_mismatch"):
            self.open(skills=[{**self.skills[0], "method_text": "changed"}])
        with self.assertRaises(RecoveryStop):
            self.open(model_identity={**self.model, "model": "changed"})
        self.assertFalse(self.continuation.exists())

    def test_governor_category_context_status_and_duplicate_id(self):
        for key, value in (
            ("category", "reviser"), ("context", {"run_identity": "changed"}),
            ("status", "error"), ("request_id", self.calls[1].governor_record["request_id"]),
            ("usage", {"total_tokens": 999}),
        ):
            with self.subTest(key=key):
                original = self.calls[0]
                governor = {**deepcopy(original.governor_record), key: value}
                self.reapprove(1, governor_record=governor)
                with self.assertRaises(RecoveryStop):
                    self.open()
                self.calls[0] = original

    def test_changed_record_without_new_review_is_rejected(self):
        call = self.calls[0]
        self.calls[0] = replace(call, seed_record={**call.seed_record, "started_at": "changed"})
        with self.assertRaisesRegex(RecoveryStop, "review_record_hash_mismatch"):
            self.open()

    def test_runtime_frozen_round_context_and_phase_must_match(self):
        client = self.open()
        client.set_audit_context({**self.context, "scheduler": "changed"})
        client.set_request_context(self.calls[0].seed_record["request_context"])
        with request_phase("program"), self.assertRaisesRegex(
                RecoveryStop, "frozen_context_mismatch"):
            client.chat(self.messages[0], max_tokens=4096)
        client = self.open(continuation_root=self.root / "round-context")
        client.set_audit_context(self.context)
        client.set_request_context({**self.calls[0].seed_record["request_context"],
                                    "round_index": 2})
        with request_phase("program"), self.assertRaisesRegex(
                RecoveryStop, "request_context_mismatch"):
            client.chat(self.messages[0], max_tokens=4096)
        client = self.open(continuation_root=self.root / "phase")
        client.set_audit_context(self.context)
        client.set_request_context(self.calls[0].seed_record["request_context"])
        with request_phase("finalization"), self.assertRaisesRegex(
                RecoveryStop, "request_phase_mismatch"):
            client.chat(self.messages[0], max_tokens=4096)

    def test_source_reply_identity_and_usage_must_match_review(self):
        for key, value in (("last_usage", {"total_tokens": 999}),
                           ("last_response_identity", {
                               **self.results[0]["last_response_identity"], "id": "changed"})):
            with self.subTest(key=key):
                original = self.calls[0]
                result = deepcopy(self.results[0])
                result[key] = value
                if key == "last_response_identity":
                    result["last_generation"]["response_identity"] = value
                raw = payload_bytes(result)
                self.update_cache(1, reply=raw)
                approval = deepcopy(original.approval)
                approval["source"]["reply_sha256"] = sha(raw)
                self.calls[0] = replace(original, approval=approval)
                client = self.open(continuation_root=self.root / key)
                with self.assertRaisesRegex(RecoveryStop, "reviewed_response_mismatch"):
                    self.replay(client, 1)
                self.calls[0] = original

    def test_invalid_rpc_reply_is_rejected_even_with_matching_review_hash(self):
        result = deepcopy(self.results[0])
        result["request_id"] = str(uuid.UUID(int=999))
        raw = payload_bytes(result)
        self.update_cache(1, reply=raw)
        approval = deepcopy(self.calls[0].approval)
        approval["source"]["reply_sha256"] = sha(raw)
        self.calls[0] = replace(self.calls[0], approval=approval)
        with self.assertRaisesRegex(RecoveryStop, "invalid_replay_evidence"):
            self.replay(self.open(), 1)

    def test_original_seed_response_is_cross_checked(self):
        response = deepcopy(self.calls[0].response_record)
        response["usage"]["total_tokens"] = 999
        self.reapprove(1, response_record=response)
        with self.assertRaisesRegex(RecoveryStop, "seed_response_mismatch"):
            self.replay(self.open(), 1)

    def test_round_trace_replay_needs_no_database_and_preserves_generation(self):
        for index in range(1, 5):
            self.as_round_trace(index)
        client = self.open(cache_path=None)
        for index in range(1, 5):
            self.assertEqual(self.replay(client, index), self.results[index - 1]["content"])
            self.assertEqual(client.last_generation, self.results[index - 1]["last_generation"])
        receipt = client.finish_replay()
        self.assertTrue(all(r["source"]["kind"] == "round_trace" for r in receipt["replayed_calls"]))

    def test_round_trace_tampering_is_not_a_cache_fallback(self):
        self.as_round_trace(1)
        self.calls[0].round_record["response_text"] = "changed"
        with self.assertRaisesRegex(RecoveryStop, "round_trace_hash_mismatch"):
            self.replay(self.open(), 1)

    def test_missing_completed_response_record_stops_before_claim(self):
        self.reapprove(1, response_record=None)
        with self.assertRaisesRegex(RecoveryStop, "seed_response_missing"):
            self.open()
        self.assertFalse(self.continuation.exists())

    def test_fresh_continuation_required_and_old_trace_overlap_refused(self):
        for path in (self.original, self.original / "nested", self.root):
            with self.subTest(path=path):
                with self.assertRaisesRegex(RecoveryStop, "continuation_overlaps_original"):
                    self.open(continuation_root=path)
        self.continuation.mkdir()
        sentinel = self.continuation / "existing.json"
        sentinel.write_bytes(b"untouched")
        with self.assertRaisesRegex(RecoveryStop, "continuation_already_exists"):
            self.open()
        self.assertEqual(sentinel.read_bytes(), b"untouched")

    def test_symlink_cache_is_rejected(self):
        link = self.root / "cache-link"
        link.symlink_to(self.database)
        with self.assertRaisesRegex(RecoveryStop, "cache_unavailable"):
            self.replay(self.open(cache_path=link), 1)

    def test_incomplete_or_repeated_finish_has_no_second_receipt(self):
        client = self.open()
        self.replay(client, 1)
        with self.assertRaisesRegex(RecoveryStop, "replay_incomplete"):
            client.finish_replay()
        self.assertFalse((self.continuation / "recovery_receipt.json").exists())
        client = self.open(continuation_root=self.root / "finished")
        for index in range(1, 5):
            self.replay(client, index)
        client.finish_replay()
        path = self.root / "finished" / "recovery_receipt.json"
        original = path.read_bytes()
        with self.assertRaisesRegex(RecoveryStop, "recovery_closed"):
            client.finish_replay()
        self.assertEqual(path.read_bytes(), original)

    def test_write_failure_cannot_return_response_or_settle(self):
        client = self.open()
        path = client.continuation_trace_dir / "replayed-001.json"
        path.write_bytes(b"preexisting")
        with self.assertRaisesRegex(RecoveryStop, "invalid_replay_evidence"):
            self.replay(client, 1)
        self.assertEqual(path.read_bytes(), b"preexisting")
        self.assertEqual(client.last_usage, {})
        self.assertFalse((self.continuation / "recovery_receipt.json").exists())

    def test_accounting_rejects_duplicate_receipts_tampering_and_changed_ledger(self):
        client = self.open()
        for index in range(1, 5):
            self.replay(client, index)
        receipt = client.finish_replay()
        with self.assertRaisesRegex(RecoveryStop, "duplicate_recovery_settlement"):
            recovery_accounting(self.state, [receipt, receipt])
        with self.assertRaisesRegex(RecoveryStop, "invalid_recovery_receipt"):
            recovery_accounting(self.state, [{**receipt, "new_requests": 1}])
        changed = deepcopy(self.state)
        changed["requests"][-1]["status"] = "completed"
        with self.assertRaisesRegex(RecoveryStop, "receipt_governor_mismatch"):
            recovery_accounting(changed, [receipt])
        self.assertEqual(recovery_accounting(self.state, [])["unsettled_started_requests"], 1)

    def test_refusal_bypasses_normal_model_error_handler(self):
        self.assertTrue(issubclass(RecoveryStop, BaseException))
        self.assertFalse(issubclass(RecoveryStop, Exception))
        client = self.open()
        self.messages[0][0]["content"][0]["text"] = "changed"
        caught = False
        try:
            try:
                self.replay(client, 1)
            except Exception:
                caught = True
        except RecoveryStop:
            pass
        else:
            self.fail("recovery refusal did not propagate")
        self.assertFalse(caught)

    def test_cache_reader_is_read_only(self):
        reader = GatewaySQLiteCache(self.database)
        self.assertEqual(reader.read(self.calls[0].approval["source"]), self.results[0])
        self.assertEqual(self.database.read_bytes(), self.cache_bytes)
        self.assertFalse(self.database.with_name(self.database.name + "-journal").exists())

    def test_cache_connection_closed_and_only_read_queries_executed(self):
        real_connect = sqlite3.connect
        connections = []
        queries = []

        def connect(*args, **kwargs):
            self.assertTrue(kwargs["uri"])
            self.assertTrue(args[0].endswith("?mode=ro"))
            db = real_connect(*args, **kwargs)
            db.set_trace_callback(queries.append)
            connections.append(db)
            return db

        with patch("skill3d.evaluation.cached_episode_recovery_v11.sqlite3.connect", connect):
            GatewaySQLiteCache(self.database).read(self.calls[0].approval["source"])
        self.assertEqual(len(connections), 1)
        with self.assertRaises(sqlite3.ProgrammingError):
            connections[0].execute("SELECT 1")
        self.assertTrue(queries)
        self.assertTrue(all(q.startswith(("PRAGMA query_only=ON", "SELECT ")) for q in queries))

    def test_fourth_cache_miss_cannot_settle_partial_replay(self):
        client = self.open()
        for index in range(1, 4):
            self.replay(client, index)
        with sqlite3.connect(self.database) as db:
            db.execute("DELETE FROM requests WHERE request_id=?",
                       (self.calls[-1].approval["source"]["request_id"],))
        with self.assertRaisesRegex(RecoveryStop, "cache_miss"):
            self.replay(client, 4)
        self.assertEqual(len(list(client.continuation_trace_dir.glob("replayed-*.json"))), 3)
        self.assertFalse((self.continuation / "recovery_receipt.json").exists())
        self.assertEqual(recovery_accounting(self.state, [])["unsettled_started_requests"], 1)

    def test_original_finalization_context_replays_without_reinterpretation(self):
        request = deepcopy(self.calls[-1].seed_record)
        request["request_context"]["phase"] = "finalization"
        request["request_context"]["tool_execution_allowed"] = False
        self.reapprove(4, seed_record=request)
        client = self.open()
        for index in range(1, 5):
            self.assertEqual(self.replay(client, index), self.results[index - 1]["content"])
        client.finish_replay()

    def test_reviewed_model_seed_must_be_the_original_seed(self):
        with self.assertRaisesRegex(RecoveryStop, "seed_call_identity_mismatch"):
            self.open(model_identity={**self.model, "seed": 139})
        self.assertFalse(self.continuation.exists())

    def test_duplicate_rpc_id_and_reordered_governor_ids_are_rejected(self):
        original = self.calls[1]
        approval = deepcopy(original.approval)
        approval["source"]["request_id"] = self.calls[0].approval["source"]["request_id"]
        self.calls[1] = replace(original, approval=approval)
        with self.assertRaisesRegex(RecoveryStop, "gateway_call_mismatch"):
            self.open()
        self.calls[1] = original
        self.reapprove(1, governor_record={**self.calls[0].governor_record, "request_id": 103})
        with self.assertRaisesRegex(RecoveryStop, "governor_call_mismatch"):
            self.open()

    def test_evidence_is_copied_at_open_not_mutated_by_the_caller(self):
        original_context = deepcopy(self.calls[0].seed_record["request_context"])
        client = self.open()
        self.calls[0].approval["decision"] = "changed after open"
        self.calls[0].seed_record["request_context"]["round_index"] = 999
        client.set_audit_context(self.context)
        client.set_request_context(original_context)
        with request_phase("program"):
            self.assertEqual(client.chat(self.messages[0], max_tokens=4096, seed=137),
                             self.results[0]["content"])

    def test_receipt_write_failure_does_not_overwrite_an_existing_receipt(self):
        client = self.open()
        for index in range(1, 5):
            self.replay(client, index)
        path = self.continuation / "recovery_receipt.json"
        path.write_bytes(b"existing synthetic receipt")
        with self.assertRaisesRegex(RecoveryStop, "receipt_write_failed"):
            client.finish_replay()
        self.assertEqual(path.read_bytes(), b"existing synthetic receipt")

    def test_rpc_audit_rejects_bool_float_equivalence(self):
        result = deepcopy(self.results[0])
        result["audit_identity"]["temperature"] = False
        raw = payload_bytes(result)
        self.update_cache(1, reply=raw)
        approval = deepcopy(self.calls[0].approval)
        approval["source"]["reply_sha256"] = sha(raw)
        self.calls[0] = replace(self.calls[0], approval=approval)
        with self.assertRaisesRegex(RecoveryStop, "response_request_identity_mismatch"):
            self.replay(self.open(), 1)

    def test_invalid_reviewed_usage_rejected_before_claim(self):
        original = self.calls[-1]
        for value in (-1, True, float("nan"), float("inf")):
            with self.subTest(value=value):
                approval = deepcopy(original.approval)
                approval["source"]["usage"]["total_tokens"] = value
                self.calls[-1] = replace(original, approval=approval)
                with self.assertRaisesRegex(RecoveryStop, "review_usage_invalid"):
                    self.open()
        self.assertFalse(self.continuation.exists())

    def test_request_payload_hash_is_not_a_reply_hash(self):
        original = self.calls[0]
        self.assertNotEqual(original.approval["source"]["payload_sha256"],
                            original.approval["source"]["reply_sha256"])
        approval = deepcopy(original.approval)
        approval["source"]["reply_sha256"] = approval["source"]["payload_sha256"]
        self.calls[0] = replace(original, approval=approval)
        with self.assertRaisesRegex(RecoveryStop, "cache_reply_hash_mismatch"):
            self.replay(self.open(), 1)

    def test_canonical_seed_json_key_order_does_not_change_gateway_identity_hash(self):
        # Actual seed files are canonical JSON, not the original wire dict order.
        model = json.loads(json.dumps(self.model, sort_keys=True))
        self.assertNotEqual(sha(payload_bytes(model)), sha(payload_bytes(self.model)))
        self.assertEqual(evidence_sha256(model), evidence_sha256(self.model))
        for index, call in enumerate(self.calls):
            self.calls[index] = replace(
                call,
                seed_record=json.loads(json.dumps(call.seed_record, sort_keys=True)),
                response_record=json.loads(json.dumps(call.response_record, sort_keys=True)),
            )
        client = self.open(model_identity=model)
        self.assertEqual(payload_bytes(client.audit_identity(seed=138)),
                         payload_bytes(self.model))
        for index in range(1, 5):
            self.assertEqual(self.replay(client, index), self.results[index - 1]["content"])
        client.finish_replay()


if __name__ == "__main__":
    unittest.main()
