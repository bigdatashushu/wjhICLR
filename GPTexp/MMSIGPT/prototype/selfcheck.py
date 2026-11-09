"""Network-disabled engineering checks over synthetic images and fake responses."""
from __future__ import annotations

import base64
from copy import deepcopy
import hashlib
from io import BytesIO
import json
import os
from pathlib import Path
import socket
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch
import urllib.error
import urllib.request

from PIL import Image

from ...traces import digest, file_digest, read, write
from .client import ResponsesClient, ServiceError, configuration, response_record
from .direct import execute, operator_lock, prepare, trace_paths, validate_manifest
from .evaluation import export, freeze, score
from .inputs import extract_public, load_images, original_input_sha256, select
from .protocol import DIRECT_INSTRUCTION, messages, parse_answer, payload, prompt


class FakeClient:
    def __init__(self, profile, inference, *, texts=None, fail_at=None, crash_at=None):
        self.profile, self.inference = profile, inference
        self.texts = texts or ["`A`", "B", "A or B", "D", "`C`", "Z"]
        self.fail_at, self.crash_at = fail_at, crash_at
        self.calls = []

    def assert_public(self, value):
        return None

    def call(self, body):
        ordinal = len(self.calls)
        self.calls.append(deepcopy(body))
        if ordinal == self.crash_at:
            raise OSError("synthetic interruption")
        if ordinal == self.fail_at:
            raise ServiceError("http_error", 524)
        return response_record(self.profile, {
            "status": "completed", "model": self.inference["model"],
            "reasoning": {"effort": "high"}, "temperature": 1, "seed": None,
            "max_output_tokens": self.inference["max_output_tokens"],
            "output": [{"type": "message", "content": [
                {"type": "output_text", "text": self.texts[ordinal % len(self.texts)]}]}],
            "usage": {"input_tokens": 11, "output_tokens": 3,
                      "output_tokens_details": {"reasoning_tokens": 2}},
        }, self.inference, elapsed=0.01, request_id=f"synthetic-request-{ordinal}")


class PrototypeChecks(unittest.TestCase):
    def setUp(self):
        self.temp = TemporaryDirectory(prefix="mmsi-direct-selfcheck-")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        paths = []
        for index, color in enumerate(((240, 30, 30), (20, 200, 40))):
            path = self.root / f"original-{index}.png"
            Image.new("RGB", (20 + index, 15 + index), color).save(path)
            paths.append(str(path))
        self.rows = []
        tasks = ("Cam.–Cam.", "Cam.–Obj.", "Cam.–Reg.", "Obj.–Obj.", "Obj.–Reg.", "Reg.–Reg.")
        for index, task in enumerate(tasks):
            for variant in range(2):
                problem = {
                    "benchmark": "mmsi", "sample_id": f"SYNTH_{index}_{variant}",
                    "source_split": "test", "task_type": task,
                    "question": f"Synthetic location question {index}/{variant}：方向？",
                    "options": ["Behind", "On the right", "Behind", "In front"],
                    "image_paths": paths if variant == 0 else list(reversed(paths)),
                }
                problem["input_sha256"] = original_input_sha256(
                    problem, [file_digest(path) for path in problem["image_paths"]])
                images = [row[0] for row in load_images(problem)]
                self.rows.append({"problem": problem, "images": images})
        self.source = self.root / "public-source.json"
        write(self.source, {
            "jobs": [{**row, "arm": arm, "prepared_root": "/must-not-be-read",
                      "skill": {"unused": True}} for row in self.rows for arm in ("direct", "skill")]
        })
        self.config = self.root / "endpoint.json"
        write(self.config, {
            "model": "gpt-5.6-terra", "reasoning_effort": "high", "seed": 137, "temperature": 0,
            "response_metadata_policy": "request_context",
            "apis": [{
                "id": "synthetic", "model": "gpt-5.6-terra", "protocol": "responses",
                "base_url": "https://example.invalid/v1", "api_key_env": "XCODE_GPT_API_KEY",
                "credentials_file": "/must-not-be-read", "timeout_seconds": 36000,
                "validated": True, "enabled": True,
            }],
        })
        self.api, self.profile, self.inference, _ = configuration(self.config, "synthetic")
        self.run = self.root / "run"

    def prepared(self):
        prepare(self.run, self.source, self.config, "synthetic")
        return validate_manifest(self.run)

    def executed(self, **kwargs):
        manifest = self.prepared()
        fake = FakeClient(self.profile, self.inference, **kwargs)
        summary = execute(self.run, self.config, client_factory=lambda *args: fake)
        return manifest, fake, summary

    def references(self, manifest):
        path = self.root / "synthetic-host-references.json"
        write(path, [
            {"benchmark": "mmsi", "sample_id": item["problem"]["sample_id"],
             "reference_answer": "D" if index == 3 else "A"}
            for index, item in enumerate(manifest["items"])
        ])
        return path, file_digest(path)

    def test_public_only_dual_arm_deduplication(self):
        source = read(self.source)
        rows = extract_public(source)
        self.assertEqual(len(rows), 12)
        self.assertTrue(all(set(row) == {"problem", "images"} for row in rows))
        source["jobs"][1]["problem"] = deepcopy(source["jobs"][1]["problem"])
        source["jobs"][1]["problem"]["question"] += " changed"
        with self.assertRaisesRegex(ValueError, "duplicate_sample_public_input_disagrees"):
            extract_public(source)

    def test_public_problem_rejects_labels(self):
        source = read(self.source)
        source["jobs"][0]["problem"]["reference_answer"] = "A"
        with self.assertRaisesRegex(ValueError, "public_problem_fields_changed"):
            extract_public(source)

    def test_pairs_manifest_without_prepared_reads(self):
        source = {"pairs": [
            {"dataset": "mmsi", "arms": [
                {"arm": arm, "problem": row["problem"], "prepared_root": "/must-not-be-read"}
                for arm in ("direct", "harness")]}
            for row in self.rows
        ]}
        self.assertEqual(len(extract_public(source)), 12)

    def test_original_hash_order_and_duplicate_options(self):
        row = self.rows[0]
        images = load_images(row["problem"], row["images"])
        self.assertEqual(row["problem"]["options"][0], row["problem"]["options"][2])
        wrong = deepcopy(row["problem"])
        wrong["image_paths"].reverse()
        with self.assertRaisesRegex(ValueError, "original_input_bytes_or_identity_changed"):
            load_images(wrong)
        with self.assertRaisesRegex(ValueError, "original_image_receipts_changed"):
            load_images(row["problem"], list(reversed(row["images"])))
        self.assertEqual(images[0][1], Path(row["problem"]["image_paths"][0]).read_bytes())

    def test_official_prompt_and_original_pixel_payload(self):
        row = self.rows[0]
        body = payload(messages(row["problem"], load_images(row["problem"])), self.inference)
        content = body["input"][0]["content"]
        self.assertEqual([part["type"] for part in content], ["input_image", "input_image", "input_text"])
        self.assertEqual(content[-1]["text"], row["problem"]["question"] +
                         "\nOptions: A: Behind, B: On the right, C: Behind, D: In front, \n" +
                         DIRECT_INSTRUCTION)
        self.assertNotIn("text", body)
        self.assertNotIn("tools", body)
        self.assertNotIn("instructions", body)
        for part, path in zip(content[:2], row["problem"]["image_paths"]):
            raw = base64.b64decode(part["image_url"].split(",", 1)[1])
            self.assertEqual(hashlib.sha256(raw).hexdigest(), file_digest(path))

    def test_exact_project_requested_parameters(self):
        manifest = self.prepared()
        inference = manifest["inference"]
        body = payload(messages(self.rows[0]["problem"], load_images(self.rows[0]["problem"])), inference)
        self.assertEqual((body["model"], body["seed"], body["temperature"], body["max_output_tokens"]),
                         ("gpt-5.6-terra", 137, 0, 8192))
        self.assertEqual(body["reasoning"], {"effort": "high"})
        self.assertEqual(inference["timeout_seconds"], 36000)
        self.assertEqual(manifest["max_requests"], 6)
        self.assertEqual(read(self.run / "preparation.json")["model_requests"], 0)
        self.assertFalse((self.run / "execution.started.json").exists())

    def test_label_free_selection_and_exclusion(self):
        first = select(self.rows)
        self.assertEqual(first, select(list(reversed(self.rows))))
        excluded = [first[0]["problem"]["sample_id"]]
        other = select(self.rows, excluded=excluded)
        self.assertNotIn(excluded[0], [r["problem"]["sample_id"] for r in other])
        self.assertEqual(len(other), 6)
        ids = [self.rows[5]["problem"]["sample_id"], self.rows[1]["problem"]["sample_id"]]
        self.assertEqual([r["problem"]["sample_id"] for r in select(self.rows, sample_ids=ids)], ids)
        with self.assertRaisesRegex(ValueError, "invalid_or_excluded_sample_ids"):
            select(self.rows, sample_ids=[excluded[0]], excluded=excluded)

    def test_conservative_parser_no_default(self):
        for value in ("A", "`A`", "``A``", "(A)", "Answer: A", "The answer is `A`.", "a"):
            self.assertEqual(parse_answer(value, 4)["answer"], "A")
        for value in ("", "A or B", '{"answer":"A"}', "Probably A because...", "A\nB", "Behind"):
            self.assertIsNone(parse_answer(value, 4)["answer"])
        self.assertEqual(parse_answer("Z", 4)["parse_status"], "out_of_range")

    def test_complete_freeze_score_and_review(self):
        manifest, fake, summary = self.executed()
        self.assertEqual(len(fake.calls), 6)
        self.assertEqual(summary["status"], "complete")
        frozen = freeze(self.run)
        self.assertEqual(frozen["status_counts"], {"answered": 4, "invalid_answer": 2})
        path, sha = self.references(manifest)
        report = score(self.run, path, sha)
        self.assertEqual(report["overall"]["questions"], 6)
        self.assertEqual(report["overall"]["legal_answers"], 4)
        self.assertEqual(report["overall"]["correct"], 2)
        self.assertEqual(report["overall"]["accuracy"], 2 / 6)
        self.assertEqual(report["overall"]["accuracy_among_legal"], 2 / 4)
        self.assertEqual(len(report["by_task"]), 6)
        self.assertEqual(report, score(self.run, path, sha))
        export(self.run)
        review = read(self.run / "review.json")
        self.assertEqual(len(review["rows"]), 6)
        self.assertTrue(review["scored"])
        self.assertNotIn("data:image", json.dumps(review))
        response = review["rows"][0]["response"]
        self.assertEqual(response["reported"]["temperature"], 1)
        self.assertIsNone(response["reported"]["seed"])
        self.assertFalse(response["reported_matches_requested"]["temperature"])

    def test_score_requires_freeze_before_reference_open(self):
        self.executed()
        # This absent path must not be touched until a valid freeze exists.
        with patch("GPTexp.MMSIGPT.prototype.evaluation.reference_answers") as parse:
            with self.assertRaises(FileNotFoundError) as error:
                score(self.run, self.root / "never-open-reference", "0" * 64)
            self.assertEqual(Path(error.exception.filename).name, "frozen.json")
            parse.assert_not_called()

    def test_service_error_stops_and_preserves_denominator(self):
        manifest, fake, summary = self.executed(fail_at=1)
        self.assertEqual(len(fake.calls), 2)
        self.assertEqual(summary["status"], "stopped_service_error")
        frozen = freeze(self.run)
        self.assertEqual(frozen["status_counts"],
                         {"answered": 1, "service_error": 1, "not_dispatched": 4})
        path, sha = self.references(manifest)
        report = score(self.run, path, sha)
        self.assertEqual(report["overall"]["questions"], 6)
        self.assertEqual(report["overall"]["accuracy"], 1 / 6)
        with self.assertRaisesRegex(ValueError, "run_already_started_or_frozen_no_resume"):
            execute(self.run, self.config, client_factory=lambda *args: fake)
        self.assertEqual(len(fake.calls), 2)

    def test_interrupted_dispatch_is_not_reissued(self):
        manifest = self.prepared()
        fake = FakeClient(self.profile, self.inference, crash_at=0)
        with self.assertRaises(OSError):
            execute(self.run, self.config, client_factory=lambda *args: fake)
        self.assertEqual(read(self.run / "execution.json")["status"], "interrupted")
        with self.assertRaisesRegex(ValueError, "run_already_started_or_frozen_no_resume"):
            execute(self.run, self.config, client_factory=lambda *args: fake)
        frozen = freeze(self.run)
        self.assertEqual(frozen["status_counts"], {"outcome_unknown": 1, "not_dispatched": 5})
        self.assertEqual(len(fake.calls), 1)

    def test_frozen_trace_and_reference_tampering_rejected(self):
        manifest, fake, summary = self.executed()
        freeze(self.run)
        path, sha = self.references(manifest)
        with self.assertRaisesRegex(ValueError, "host_reference_bytes_changed"):
            score(self.run, path, "0" * 64)
        _, result_path = trace_paths(self.run, manifest["items"][0])
        result = read(result_path)
        result["response"]["text"] = "`D`"
        result["answer"] = "D"
        write(result_path, result)
        with self.assertRaisesRegex(ValueError, "records_changed_after_prediction_freeze"):
            score(self.run, path, sha)

    def test_changed_original_blocks_before_dispatch(self):
        manifest = self.prepared()
        image = Path(manifest["items"][0]["problem"]["image_paths"][0])
        Image.new("RGB", (20, 15), (0, 0, 0)).save(image)
        fake = FakeClient(self.profile, self.inference)
        with self.assertRaisesRegex(ValueError, "original_input_bytes_or_identity_changed"):
            execute(self.run, self.config, client_factory=lambda *args: fake)
        self.assertEqual(fake.calls, [])
        self.assertFalse((self.run / "execution.started.json").exists())

    def test_changed_config_blocks_before_dispatch(self):
        self.prepared()
        config = read(self.config)
        config["unused_new_field"] = True
        write(self.config, config)
        fake = FakeClient(self.profile, self.inference)
        with self.assertRaisesRegex(ValueError, "external_endpoint_configuration_changed"):
            execute(self.run, self.config, client_factory=lambda *args: fake)
        self.assertEqual(fake.calls, [])

    def test_active_operator_blocks_freeze_and_execution(self):
        manifest, fake, summary = self.executed()
        with operator_lock(self.run):
            with self.assertRaisesRegex(ValueError, "run_has_active_operator"):
                freeze(self.run)
            with self.assertRaisesRegex(ValueError, "run_has_active_operator"):
                execute(self.run, self.config, client_factory=lambda *args: fake)
        self.assertEqual(len(fake.calls), 6)

    def test_unexpected_trace_and_wrong_request_count_rejected(self):
        manifest, fake, summary = self.executed()
        unexpected = self.run / "traces" / "9999.dispatch.json"
        write(unexpected, {"unexpected": True})
        with self.assertRaisesRegex(ValueError, "unexpected_trace_outside_request_budget"):
            freeze(self.run)
        unexpected.unlink()
        summary["dispatched"] -= 1
        write(self.run / "execution.json", summary)
        with self.assertRaisesRegex(ValueError, "execution_summary_identity_changed"):
            freeze(self.run)

    def test_review_rejects_changed_score(self):
        manifest, fake, summary = self.executed()
        freeze(self.run)
        path, sha = self.references(manifest)
        score(self.run, path, sha)
        changed = read(self.run / "score.json")
        changed["overall"]["correct"] += 1
        write(self.run / "score.json", changed)
        with self.assertRaisesRegex(ValueError, "score_freeze_identity_changed"):
            export(self.run)

    def test_source_identity_change_rejected(self):
        self.prepared()
        with patch("GPTexp.MMSIGPT.prototype.direct.code_identity", return_value={"changed": "source"}):
            with self.assertRaisesRegex(ValueError, "run_manifest_or_source_identity_changed"):
                validate_manifest(self.run)

    def test_strict_response_policy_detects_service_deviation(self):
        strict = deepcopy(self.inference)
        strict["response_metadata_policy"] = "strict"
        strict["upstream_policy"]["require_reported_sampling_match"] = True
        with self.assertRaisesRegex(ServiceError, "reported_sampling_changed"):
            FakeClient(self.profile, strict).call({})

    def test_http_failure_single_attempt_and_private_error_body(self):
        synthetic_secret = "synthetic_private_marker"
        with patch("GPTexp.MMSIGPT.prototype.client.resolve_key", return_value=synthetic_secret):
            client = ResponsesClient(self.api, self.profile, self.inference)
        error = urllib.error.HTTPError(
            self.profile.endpoint, 524, synthetic_secret, {},
            BytesIO(synthetic_secret.encode()))
        opener = unittest.mock.Mock()
        opener.open.side_effect = error
        with patch("urllib.request.build_opener", return_value=opener):
            with self.assertRaises(ServiceError) as caught:
                client.call({"model": "gpt-5.6-terra"})
        self.assertEqual(opener.open.call_count, 1)
        self.assertEqual(caught.exception.http_status, 524)
        self.assertNotIn(synthetic_secret, str(caught.exception))
        with self.assertRaisesRegex(ValueError, "credential_in_provider_payload"):
            client.assert_public({"text": synthetic_secret})

    def test_empty_response_and_tool_call_are_service_failures(self):
        for output in ([], [{"type": "function_call", "name": "unexpected", "arguments": "{}"}]):
            with self.assertRaises(ServiceError):
                response_record(self.profile, {"status": "completed", "output": output},
                                self.inference, elapsed=0)


def selfcheck():
    # No inherited endpoint/auth environment, real credentials, references or model calls.
    def forbidden(*args, **kwargs):
        raise AssertionError("offline_check_attempted_network_or_real_credentials")

    with patch.dict(os.environ, {}, clear=True), \
            patch.object(socket.socket, "connect", side_effect=forbidden), \
            patch("socket.create_connection", side_effect=forbidden), \
            patch("urllib.request.OpenerDirector.open", side_effect=forbidden), \
            patch("GPTexp.MMSIGPT.prototype.client.resolve_key", side_effect=forbidden):
        suite = unittest.defaultTestLoader.loadTestsFromTestCase(PrototypeChecks)
        result = unittest.TextTestRunner(verbosity=2).run(suite)
    if not result.wasSuccessful():
        raise ValueError("offline_selfcheck_failed")
    return {"status": "passed", "checks": result.testsRun, "model_requests": 0,
            "real_reference_answers_read": False, "network_disabled": True,
            "fixtures": "synthetic_images_and_fake_responses", "temporary_files_removed": True}
