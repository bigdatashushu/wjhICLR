"""Exercise the RPC contract with the actual Bailian response metadata."""

import json
import unittest
from unittest.mock import patch

from skill3d.cluster.perception_client import validate_result
from skill3d.synthesis.bailian_client import BailianClient
from skill3d.synthesis.request_audit import payload_bytes
import hashlib


class Reply:
    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False

    def read(self):
        return json.dumps({
            "id": "response-1", "model": "qwen3.8-flash", "created": 1,
            "usage": {"prompt_tokens": 32, "completion_tokens": 8, "total_tokens": 40},
            "choices": [{"finish_reason": "stop", "message": {"content": "A"}}],
        }).encode()


class PerceptionEvidenceTest(unittest.TestCase):
    def setUp(self):
        self.client = BailianClient(model="qwen3.8-flash", max_output_tokens=8192,
                                    opener=lambda *_args, **_kwargs: Reply())
        self.messages = [{"role": "user", "content": "Choose an answer."}]
        self.payload = self.client.build_payload(
            self.messages, seed=137, max_tokens=8192, enable_thinking=True)
        self.identity = self.client.audit_identity(seed=137)
        with patch.dict("os.environ", {self.client.api_key_env: "unit-test-only"}):
            content = self.client.chat(
                self.messages, seed=137, max_tokens=8192, enable_thinking=True)
        self.result = {
            "schema_version": "harness3d.perception-rpc/1.0",
            "request_id": "request-1",
            "payload_sha256": hashlib.sha256(payload_bytes(self.payload)).hexdigest(),
            "audit_identity": self.identity, "content": content,
            "last_usage": self.client.last_usage,
            "last_generation": self.client.last_generation,
            "last_response_identity": self.client.last_response_identity,
            "last_request_identity": self.client.last_request_identity,
        }

    def validate(self):
        validate_result(self.result, request_id="request-1",
                        payload=self.payload, audit_identity=self.identity)

    def test_actual_bailian_receipt(self):
        self.assertIs(self.result["last_response_identity"]["model_reported"], True)
        self.validate()

    def test_legacy_receipt(self):
        del self.result["last_response_identity"]["model_reported"]
        self.validate()

    def test_invalid_reporting_flag_is_rejected(self):
        self.result["last_response_identity"]["model_reported"] = "true"
        with self.assertRaises(ValueError):
            self.validate()

    def test_unknown_response_field_is_rejected(self):
        self.result["last_response_identity"]["unexpected"] = 1
        with self.assertRaises(ValueError):
            self.validate()

    def test_request_identity_mismatch_is_rejected(self):
        self.result["payload_sha256"] = "0" * 64
        with self.assertRaises(ValueError):
            self.validate()


if __name__ == "__main__":
    unittest.main()
