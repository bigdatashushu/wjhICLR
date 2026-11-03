"""Offline regression tests using real host recovery functions and fake inference."""
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack, nullcontext, redirect_stdout
from copy import deepcopy
import hashlib
import io
import json
from pathlib import Path
import sys
import tempfile
import threading
import unittest
from unittest.mock import patch

import resume07_control as c
import resume07_entry as e

h, pilot = c.timeout_handling07, c.pilot
PRODUCTION_OPS = c.OPS


class ServiceUnavailable(Exception):
    pass


class OfflineRecovery(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="mmsi-timeout07-offline-")
        self.root = Path(self.temp.name)
        self.stack = ExitStack()
        self.stack.enter_context(patch.object(h, "ROOT", self.root))
        self.stack.enter_context(patch.object(c, "ROOT", self.root))
        self.stack.enter_context(patch.object(e, "ROOT", self.root))
        self.stack.enter_context(patch.object(pilot, "ROOT", self.root))
        self.ops = self.root / "ops"
        for folder in ("ops", "results/skill", "results/direct", "request_records", "intents",
                       "requests", "test_budget", "original", "cv", "ops/proofs", "ops/handled"):
            (self.root / folder).mkdir(parents=True, exist_ok=True)
        for module in (c, e):
            self.stack.enter_context(patch.object(module, "OPS", self.ops))
        self.stack.enter_context(patch.object(h, "PROOFS", self.ops / "proofs"))
        self.stack.enter_context(patch.object(h, "HANDLED", self.ops / "handled"))
        for name, filename in (("PREFLIGHT", "preflight.json"), ("RECOVERY", "recovery.json"),
                               ("ORIGINAL_PREFIX", "original_prefix.json"), ("TIMEOUT_POLICY", "policy.json")):
            self.stack.enter_context(patch.object(c, name, self.ops / filename))
        self.stack.enter_context(patch.object(c, "FAILURES", {}))
        self.stack.enter_context(patch.object(c.resume06, "FAILURES", {}))
        self.stack.enter_context(patch.object(c, "audit_shared_state", lambda *a, **k: None))
        self.stack.enter_context(patch.object(c.watcher, "process_state", lambda *a: {"alive": False}))
        self.stack.enter_context(patch.object(c, "retained_audit", lambda *a, **k: None))
        self.stack.enter_context(patch.object(c, "health", lambda: {"status": "offline_ready"}))
        self.stack.enter_context(patch.object(c, "operator_locks", lambda *a: nullcontext()))
        def archive():
            (self.root / "startup.json").rename(self.ops / "archived_startup.json")
            return {}
        self.stack.enter_context(patch.object(c, "archive_runtime", archive))
        self.stack.enter_context(patch.object(c, "process_receipt", lambda: {"pid": 0}))
        self.stack.enter_context(patch.object(pilot, "score", lambda *a, **k: None))
        class Governor:
            def summary(self):
                return {"offline": True}
        self.stack.enter_context(patch.object(pilot, "governor", lambda *a: Governor()))
        # Any accidental network call fails the test immediately.
        self.stack.enter_context(patch("urllib.request.OpenerDirector.open", side_effect=AssertionError("offline_network_forbidden")))
        self.proofs = {}
        self.stack.enter_context(patch.object(h, "get", lambda path: deepcopy(self.proofs[path.rsplit("/", 1)[-1]])))
        self.lock = threading.Lock()
        self.jobs = []
        self.baseline = {"status": "passed", "manifest_sha256": c.MANIFEST_SHA,
                         "pending_never_dispatched": [], "ledger_prefix_hashes": {"requests": [], "test_budget": []},
                         "original_prefix_requests": 0, "shared_ledger_policy": "offline",
                         "verified_other_campaign_requests": 0}
        self.policy = {"max_requests": 1900, "max_inflight": 4, "requests_per_second": 0.0}
        for name in ("requests", "test_budget"):
            (self.root / name / "state.json").write_text(json.dumps(
                {"policy": self.policy, "peak_inflight": 4, "requests": []}))
        (self.root / "original/state.json").write_text('{"requests":[]}')
        (self.root / "original_budget_prefix.json").write_text('{"requests":[]}')
        pilot.write(self.root / "cv/startup.json", {})
        pilot.write(self.root / "startup.json", {})
        pilot.write(c.ORIGINAL_PREFIX, {})
        pilot.write(c.TIMEOUT_POLICY, {})
        self.baseline["original_ledger_prefix_sha256"] = pilot.sha(c.ORIGINAL_PREFIX)
        pilot.write(self.ops / "resume07_preflight_exit.json", {"returncode": 0})
        pilot.write(self.ops / "resume07_preflight_startup.json", {})
        prior = self.ops / "prior_recovery.json"
        pilot.write(self.root / "historical_timeout.json", {})
        pilot.write(prior, {"timeout_result_path": "historical_timeout.json",
                           "timeout_result_sha256": pilot.sha(self.root / "historical_timeout.json")})
        self.stack.enter_context(patch.object(c.resume06, "RECOVERY", prior))

    def tearDown(self):
        self.stack.close()
        self.temp.cleanup()

    def job(self, index, arm="direct"):
        op = hashlib.sha256(("offline-job-" + str(index)).encode()).hexdigest()
        job = {"operation_id": op, "run_identity": h.RUN, "arm": arm, "ordinal": index,
               "input_sha256": "1" * 64, "images": [{"sha256": "2" * 64}],
               "problem": {"benchmark": "mmsi", "sample_id": str(index), "task_type": "offline"},
               "sample_id": str(index)}
        self.jobs.append(job)
        self.baseline["pending_never_dispatched"].append(op)
        return job

    def ledger(self, job, status, index):
        with self.lock:
            for name in ("requests", "test_budget"):
                path = self.root / name / "state.json"
                state = json.loads(path.read_text())
                state["requests"].append({
                    "request_id": len(state["requests"]), "category": "solver", "status": status, "usage": {},
                    "context": {"run_identity": h.RUN, "variant_id": job["operation_id"]},
                    **({"exception_type": "ServiceUnavailable"} if status == "error" else {})})
                temporary = path.with_suffix(".tmp")
                temporary.write_text(json.dumps(state))
                temporary.replace(path)

    def trace(self, job, index=1, timeout=True):
        op = job["operation_id"]
        folder = self.root / "request_records" / op
        folder.mkdir(exist_ok=True)
        path = folder / f"{index:03d}.response.json"
        params = {"model": "qwen3.8-flash", "temperature": 0, "seed": 137,
                  "enable_thinking": True, "max_tokens": 8192, "top_p": None, "response_format": None}
        identity = {"parameters": params, "images_in_order": [{"jpeg_sha256": "2" * 64}],
                    "wire_payload_sha256": "3" * 64}
        pilot.write(folder / f"{index:03d}.intent.json", {
            "images": job["images"], "reference_answer_visible": False, "payload_sha256": "4" * 64,
            "parameters": {k: params[k] for k in ("temperature", "seed", "enable_thinking", "max_tokens")}})
        row = {"status": "error" if timeout else "received", "request_identity": identity,
               "response_identity": {} if timeout else {"model": "qwen3.8-flash", "model_reported": True},
               "seconds": 1200, "model_response": None,
               "error_type": "ServiceUnavailable",
               "service_error": {"reason_code": "http_error", "exception_type": "HTTPError",
                                 "http_status": 502, "request_stage": "awaiting_headers", "timeout_s": 1230}}
        pilot.write(path, row)
        request_id = hashlib.sha256((h.RUN + ":" + op + ":" + str(index)).encode()).hexdigest()
        self.proofs[request_id] = {
            "status": "receipt_available", "request_id": request_id, "run_identity": h.RUN,
            "schema_version": "harness3d.timeout-receipt-reader/1", "reader_source_sha256": h.READER_SOURCE_SHA,
            "gateway_source_sha256": h.GATEWAY_SOURCE_SHA, "gateway_process_verified": True,
            "model_requests": 0, "response_bodies_read": False, "intent_sha256": "5" * 64,
            "response_sha256": "6" * 64,
            "intent": {"request_id": request_id, "audit_identity": {"run_identity": h.RUN, "operation_id": op},
                       "payload_sha256": "3" * 64, "parameters": params, "upstream_timeout_s": 1200},
            "response": {"status": "failed", "error_type": "ServiceUnavailable", "automatic_retry": False,
                         "body_sha256": None, "service_error": {
                             "schema": "bailian-service-error/1", "timeout_s": 1200, "elapsed_seconds": 1200,
                             "reason_code": "timeout", "exception_type": "TimeoutError",
                             "http_status": None, "request_stage": "awaiting_headers"}}}
        self.ledger(job, "error" if timeout else "completed", index)
        return path, self.proofs[request_id]

    def result(self, job, status="answered", count=1):
        folder = self.root / "request_records" / job["operation_id"]
        row = {"operation_id": job["operation_id"], "job_sha256": pilot.sha_value(job),
               "status": "stopped" if status == "service_error" else "complete",
               "reference_answer_visible": False, "seconds": 1200, "observations": [],
               "result": {"input_sha256": job["input_sha256"], "source_split": "test", "status": status,
                          "answer": None if status == "service_error" else "A", "request_count": count},
               "request_records": {str(p.relative_to(self.root)): pilot.sha(p) for p in folder.glob("*.json")}}
        path = self.root / "results" / job["arm"] / (job["operation_id"] + ".json")
        pilot.write(path, row)
        return path

    def manifest(self):
        return {"jobs": self.jobs, "run_identity": h.RUN, "shared_policy": self.policy, "test_policy": self.policy,
                "original_budget_root": str(self.root / "original"), "cv3d_root": str(self.root / "cv")}

    def test_direct_timeout_materializes_once_and_scores_as_failure(self):
        job = self.job(1)
        self.trace(job)
        row = h.handle(job, ServiceUnavailable())
        self.assertEqual(row["result"]["status"], "service_error")
        self.assertIsNone(row["result"]["answer"])
        c.result_audit(job, self.root / "results/direct" / (job["operation_id"] + ".json"))
        with self.assertRaises(AssertionError):
            h.handle(job, ServiceUnavailable())

    def test_skill_multiturn_preserves_frozen_result_bytes(self):
        job = self.job(2, "skill")
        self.trace(job, timeout=False)
        self.trace(job, index=2)
        path = self.result(job, "service_error", 2)
        before = pilot.sha(path)
        h.handle(job, RuntimeError())
        self.assertEqual(before, pilot.sha(path))
        _, responses = c.result_audit(job, path)
        self.assertEqual([r[2] for r in responses], [True, False])

    def test_ledger_error_before_result_is_provisional_only(self):
        job = self.job(3)
        self.trace(job)
        c.ledger_audit(self.manifest(), self.baseline, active_ok=True)
        with self.assertRaises(FileNotFoundError):
            c.ledger_audit(self.manifest(), self.baseline)
        h.handle(job, ServiceUnavailable())
        c.ledger_audit(self.manifest(), self.baseline)

    def test_reject_non_timeout_502_and_identity_tampering(self):
        job = self.job(4)
        path, proof = self.trace(job)
        variations = [
            ("response.service_error.reason_code", "http_error"),
            ("response.service_error.exception_type", "HTTPError"),
            ("response.automatic_retry", True),
            ("response.body_sha256", "7" * 64),
            ("intent.audit_identity.operation_id", "8" * 64),
            ("intent.payload_sha256", "9" * 64),
            ("request_id", "a" * 64),
            ("reader_source_sha256", "b" * 64),
            ("gateway_source_sha256", "c" * 64),
        ]
        for key, value in variations:
            with self.subTest(key=key):
                changed = deepcopy(proof)
                target = changed
                parts = key.split(".")
                for part in parts[:-1]:
                    target = target[part]
                target[parts[-1]] = value
                with self.assertRaises(AssertionError):
                    h.validate_proof(job, path, changed)

    def test_local_429_and_proof_file_tampering_stop(self):
        job = self.job(5)
        path, _ = self.trace(job)
        row = pilot.read(path)
        row.pop("content_sha256")
        row["service_error"]["http_status"] = 429
        pilot.update(path, row)
        with self.assertRaises(AssertionError):
            h.handle(job, ServiceUnavailable())
        self.assertFalse(list(h.HANDLED.glob("*.json")))
        job2 = self.job(6)
        path2, _ = self.trace(job2)
        _, proof_path = h.proof_for(job2, path2)
        data = json.loads(proof_path.read_text())
        data["automatic_retry"] = True
        proof_path.write_text(json.dumps(data))
        with self.assertRaises(AssertionError):
            h.proof_for(job2, path2)

    def test_first_response_and_full_response_pair_counts_differ(self):
        a, b = self.job(7, "skill"), self.job(8)
        identity = {"parameters": {}, "images_in_order": []}
        pairs = {"x": {"skill": (a, identity, True, False), "direct": (b, identity, True, True)}}
        self.assertEqual(c.pair_statistics(pairs), (1, 0))
        pairs["x"]["direct"] = (b, identity, False, False)
        self.assertEqual(c.pair_statistics(pairs), (0, 0))

    def test_actual_scheduler_dispatches_all_after_two_timeouts_without_retry(self):
        for index in range(1, 113):
            self.job(index, "skill" if index == 2 else "direct")
        pilot.write(c.PREFLIGHT, self.baseline)
        calls, retained_skill = {}, {}
        def fake_solve(job):
            op, index = job["operation_id"], job["ordinal"]
            with self.lock:
                calls[op] = calls.get(op, 0) + 1
            if index == 1:
                self.trace(job)
                raise ServiceUnavailable()
            if index == 2:
                self.trace(job, timeout=False)
                self.trace(job, index=2)
                path = self.result(job, "service_error", 2)
                retained_skill[op] = pilot.sha(path)
                raise RuntimeError()
            self.trace(job, timeout=False)
            return pilot.read(self.result(job))
        class OfflinePool(ThreadPoolExecutor):
            def __init__(self, max_workers, **kwargs):
                super().__init__(max_workers=max_workers)
        with patch.object(c, "manifest_current", self.manifest), patch.object(c.entry, "solve", fake_solve), \
             patch.object(e, "ProcessPoolExecutor", OfflinePool), \
             patch.object(c, "final_audit", lambda: c.ledger_audit(self.manifest(), self.baseline)), \
             redirect_stdout(io.StringIO()):
            e.run()
        self.assertEqual(len(calls), 112)
        self.assertEqual(set(calls.values()), {1})
        self.assertEqual(len(list(h.HANDLED.glob("*.json"))), 2)
        self.assertEqual(sum(len(list((self.root / "results" / a).glob("*.json"))) for a in ("skill", "direct")), 112)
        for op, expected in retained_skill.items():
            self.assertEqual(pilot.sha(self.root / "results/skill" / (op + ".json")), expected)


if __name__ == "__main__":
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(OfflineRecovery)
    result = unittest.TextTestRunner(stream=io.StringIO()).run(suite)
    if not result.wasSuccessful():
        print(json.dumps({"status": "offline_regression_failed", "tests": result.testsRun,
                          "failed_tests": [test.id().split(".")[-1] for test, _ in result.failures + result.errors],
                          "model_requests": 0, "network_requests": 0}))
        sys.exit(2)
    pilot.write(PRODUCTION_OPS / "resume07_test_receipt.json", {
        "status": "passed", "at": pilot.now(), "tests": result.testsRun, "model_requests": 0,
        "network_requests": 0, "gpu_requests": 0,
        "real_timeout_handling_functions_tested": True, "scheduler_jobs": 112,
        "scheduler_timeout_cases": 2, "scheduler_automatic_retries": 0,
        "source_sha256": pilot.sha(__file__), "controller_sha256": pilot.sha(PRODUCTION_OPS / "resume07_control.py"),
        "entry_sha256": pilot.sha(PRODUCTION_OPS / "resume07_entry.py"),
        "helper_sha256": pilot.sha(PRODUCTION_OPS / "timeout_handling07.py")})
    print(json.dumps({"status": "passed", "tests": result.testsRun, "model_requests": 0, "network_requests": 0}))
