"""Contract checks with in-memory HTTP replies; no real credentials or network."""

from __future__ import annotations

import io
import http.client
import json
import os
import socket
import tempfile
import unittest
import urllib.error
from concurrent.futures import ThreadPoolExecutor
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

from xcodeapi import (
    ConfigurationError, HTTPAPIError, Profile, ProtocolError, RequestLimitError,
    TransportError, XcodeClient, image_file_part, image_url_part, text_part,
)
from xcodeapi.__main__ import main
from xcodeapi.client import normalize_response
from xcodeapi.config import KEY_NAMES, load_profiles, read_credentials, resolve_api_key
from xcodeapi.transport import HTTPResult, JSONTransport


def profile(protocol="responses", **fields):
    return Profile(protocol, fields.pop("base_url", "https://xcode.best/v1"),
                   fields.pop("model", "test-model"), "XCODE_GPT_API_KEY", **fields)


def response(protocol="responses"):
    if protocol == "responses":
        return {"model": "test-model", "status": "completed",
                "output": [{"type": "message", "content": [{"type": "output_text", "text": "OK"}]}],
                "usage": {"input_tokens": 10, "output_tokens": 2}}
    if protocol == "messages":
        return {"model": "test-model", "stop_reason": "end_turn",
                "content": [{"type": "text", "text": "OK"}],
                "usage": {"input_tokens": 10, "output_tokens": 2}}
    return {"model": "test-model", "choices": [
        {"message": {"content": "OK"}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 10, "completion_tokens": 2}}


class OfflineTest(unittest.TestCase):
    def setUp(self):
        # A future regression must not be able to make a live connection.
        self.network = patch.object(socket.socket, "connect", side_effect=AssertionError("Network forbidden"))
        self.network.start()
        self.env = patch.dict(os.environ, {name: "" for name in KEY_NAMES})
        self.env.start()

    def tearDown(self):
        self.env.stop()
        self.network.stop()

    def test_public_profiles_expose_three_protocols(self):
        configs = load_profiles()
        self.assertTrue({"gpt", "claude", "gemini"}.issubset(configs))
        self.assertEqual(configs["gpt"].protocol, "responses")
        self.assertEqual(configs["claude"].protocol, "messages")
        self.assertEqual(configs["gemini"].protocol, "chat_completions")

    def test_base_url_normalization_and_paths(self):
        for protocol, suffix in (
            ("responses", "/responses"), ("messages", "/messages"),
            ("chat_completions", "/chat/completions"),
        ):
            for base in ("https://xcode.best", "https://xcode.best/", "https://xcode.best/v1/"):
                self.assertEqual(profile(protocol, base_url=base).endpoint, "https://xcode.best/v1" + suffix)

    def test_reject_bad_endpoint_and_timeout(self):
        for url in ("http://xcode.best", "https://secret@xcode.best", "https://xcode.best/v1?key=test",
                    "https://xcode.best/v1/messages"):
            with self.subTest(url=url), self.assertRaises(ConfigurationError):
                profile(base_url=url)
        for timeout in (0, -1, True, float("nan"), float("inf")):
            with self.subTest(timeout=timeout), self.assertRaises(ConfigurationError):
                profile(timeout_seconds=timeout)

    def test_constructor_prepare_and_preflight_never_read_key(self):
        with patch("xcodeapi.client.resolve_api_key", side_effect=AssertionError("Key read forbidden")):
            XcodeClient(profile()).prepare([{"role": "user", "content": "OK"}])
            output = io.StringIO()
            with redirect_stdout(output):
                self.assertEqual(main(["preflight"]), 0)
            summary = json.loads(output.getvalue())
            self.assertEqual(summary["network_requests"], 0)
            self.assertFalse(summary["credentials_read"])
            self.assertTrue(all(p["payload_builder_valid"] for p in summary["profiles"].values()))

    def test_native_payloads_match_protocol(self):
        messages = [{"role": "system", "content": "instruction"},
                    {"role": "user", "content": "question"},
                    {"role": "assistant", "content": "previous answer"},
                    {"role": "user", "content": "continue"}]
        for protocol in ("responses", "messages", "chat_completions"):
            payload = XcodeClient(profile(protocol)).prepare(messages, max_output_tokens=81)
            self.assertFalse(payload["stream"])
            self.assertNotIn("temperature", payload)
            if protocol == "responses":
                self.assertEqual(payload["instructions"], "instruction")
                self.assertEqual(payload["max_output_tokens"], 81)
                self.assertEqual(payload["input"][1]["content"][0]["type"], "output_text")
                self.assertFalse(payload["store"])
            elif protocol == "messages":
                self.assertEqual(payload["system"], "instruction")
                self.assertEqual(payload["max_tokens"], 81)
                self.assertEqual(payload["messages"][0]["role"], "user")
            else:
                self.assertEqual(payload["messages"][0], {"role": "system", "content": "instruction"})
                self.assertEqual(payload["max_tokens"], 81)

    def test_image_conversion_preserves_order_and_hides_path(self):
        with tempfile.TemporaryDirectory() as folder:
            image = Path(folder) / "small.png"
            image.write_bytes(b"\x89PNG\r\n\x1a\nfixture-image")
            part = image_file_part(image)
        messages = [{"role": "user", "content": [text_part("first"), part, text_part("last")]}]
        for protocol in ("responses", "messages", "chat_completions"):
            payload = XcodeClient(profile(protocol)).prepare(messages)
            self.assertNotIn(folder, json.dumps(payload))
            key = "input" if protocol == "responses" else "messages"
            parts = payload[key][0]["content"]
            self.assertEqual(parts[0]["text"], "first")
            self.assertEqual(parts[2]["text"], "last")
            if protocol == "messages":
                self.assertEqual(parts[1]["source"]["type"], "base64")
                self.assertEqual(parts[1]["source"]["media_type"], "image/png")
            else:
                self.assertIn("image", parts[1]["type"])

    def test_image_url_validation_and_anthropic_detail(self):
        image = image_url_part("https://example.test/image.png", detail="high")
        with self.assertRaises(ProtocolError):
            XcodeClient(profile("messages")).prepare([{"role": "user", "content": [image]}])
        for bad in ("file:///tmp/secret.png", "data:image/png;base64,?", "data:text/plain;base64,YQ==",
                    "https://[broken/image.png"):
            with self.subTest(url=bad), self.assertRaises(ProtocolError):
                image_url_part(bad)

    def test_reject_unsupported_content_without_silent_loss(self):
        for message in (
            {"role": "tool", "content": "result"},
            {"role": "user", "content": "question", "other": "drop me"},
            {"role": "assistant", "content": [image_url_part("https://example.test/a.png")]},
            {"role": "user", "content": [{"type": "unknown"}]},
        ):
            with self.subTest(message=message), self.assertRaises(ProtocolError):
                XcodeClient(profile()).prepare([message])

    def test_protect_experiment_model_and_parameters(self):
        client = XcodeClient(profile())
        for extras in ({"model": "other"}, {"stream": True}, {"max_output_tokens": 3}, {"n": 2}):
            target = client if "n" not in extras else XcodeClient(profile("chat_completions"))
            with self.subTest(extras=extras), self.assertRaises(ProtocolError):
                target.prepare([{"role": "user", "content": "question"}], extra_body=extras)
        for payload in ({"model": "other"}, {"model": "test-model", "stream": True},
                        {"model": "test-model", "background": True}):
            with self.subTest(payload=payload), self.assertRaises(ProtocolError):
                client.send(payload)

    def test_native_extra_body_and_snapshot(self):
        extras = {"reasoning": {"effort": "high"}}
        payload = XcodeClient(profile()).prepare([{"role": "user", "content": "question"}], extra_body=extras)
        extras["reasoning"]["effort"] = "low"
        self.assertEqual(payload["reasoning"]["effort"], "high")

    def test_native_required_fields_fail_without_reading_key(self):
        with patch("xcodeapi.client.resolve_api_key", side_effect=AssertionError("Key read forbidden")):
            for protocol, native in (
                ("responses", {"model": "test-model"}),
                ("messages", {"model": "test-model", "messages": [{"role": "user", "content": "question"}]}),
                ("chat_completions", {"model": "test-model", "messages": []}),
            ):
                with self.subTest(protocol=protocol), self.assertRaises(ProtocolError):
                    XcodeClient(profile(protocol)).send(native)

    def test_anthropic_temperature_range(self):
        with self.assertRaises(ProtocolError):
            XcodeClient(profile("messages")).prepare(
                [{"role": "user", "content": "question"}], temperature=1.5
            )

    def test_missing_model_fails_before_credentials(self):
        with patch("xcodeapi.client.resolve_api_key", side_effect=AssertionError):
            with self.assertRaises(ConfigurationError):
                XcodeClient(profile(model="")).prepare([{"role": "user", "content": "question"}])

    def test_literal_credentials_priority_and_no_interpolation(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "key.local.env"
            path.write_text("XCODE_GPT_API_KEY='fake-$VALUE-`literal`'\nXCODE_API_KEY=fake-common\n")
            path.chmod(0o600)
            self.assertEqual(read_credentials(path)["XCODE_GPT_API_KEY"], "fake-$VALUE-`literal`")
            os.environ["XCODE_API_KEY"] = "fake-env-common"
            self.assertEqual(resolve_api_key(profile(), path), "fake-$VALUE-`literal`")
            os.environ["XCODE_GPT_API_KEY"] = "fake-env-dedicated"
            self.assertEqual(resolve_api_key(profile(), path), "fake-env-dedicated")

    def test_reject_insecure_or_duplicate_credential_file(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "key.local.env"
            path.write_text("XCODE_API_KEY=fake-one\nXCODE_API_KEY=fake-two\n")
            path.chmod(0o600)
            with self.assertRaises(ConfigurationError):
                read_credentials(path)
            path.write_text("XCODE_API_KEY=fake-one\n")
            path.chmod(0o644)
            with self.assertRaises(ConfigurationError):
                read_credentials(path)

    def test_missing_key_does_not_count_a_request(self):
        client = XcodeClient(profile(), credentials_file="/nonexistent-xcode-test-path", max_requests=1)
        with self.assertRaises(ConfigurationError):
            client.generate([{"role": "user", "content": "question"}])
        self.assertEqual(client.attempts, 0)

    def test_all_protocol_responses_and_missing_usage(self):
        for protocol in ("responses", "messages", "chat_completions"):
            body = response(protocol)
            normalized = normalize_response(profile(protocol), HTTPResult(body, 1.2))
            self.assertEqual(normalized.text, "OK")
            self.assertEqual(normalized.usage.total_tokens, 12)
            del body["usage"]
            normalized = normalize_response(profile(protocol), HTTPResult(body, 1.2))
            self.assertIsNone(normalized.usage.total_tokens)
            self.assertIsNone(normalized.usage.input_tokens)

    def test_reasoning_tools_and_cache_accounting(self):
        body = response("messages")
        body["content"] = [
            {"type": "thinking", "thinking": "private-thought", "signature": "opaque"},
            {"type": "tool_use", "id": "call-1", "name": "measure", "input": {"x": 2}},
        ]
        body["usage"].update(cache_read_input_tokens=20, cache_creation_input_tokens=5)
        result = normalize_response(profile("messages"), HTTPResult(body, 1.2))
        self.assertEqual(result.reasoning, "private-thought")
        self.assertEqual(result.usage.input_tokens, 35)
        self.assertEqual(result.usage.total_tokens, 37)
        self.assertEqual(result.tool_calls[0]["input"], {"x": 2})
        self.assertEqual(result.raw["content"][0]["signature"], "opaque")
        self.assertNotIn("private-thought", repr(result))
        self.assertNotIn("opaque", json.dumps(result.metadata()))

    def test_responses_reasoning_and_incomplete_status(self):
        body = response()
        body["output"].insert(0, {"type": "reasoning", "summary": [{"type": "summary_text", "text": "summary"}]})
        body["output"].append({"type": "function_call", "call_id": "c", "name": "measure", "arguments": "{}"})
        body["usage"]["output_tokens_details"] = {"reasoning_tokens": 1}
        body["status"] = "incomplete"
        body["incomplete_details"] = {"reason": "max_output_tokens"}
        result = normalize_response(profile(), HTTPResult(body, 1))
        self.assertEqual(result.finish_reason, "max_output_tokens")
        self.assertEqual(result.reasoning, "summary")
        self.assertEqual(result.usage.reasoning_tokens, 1)
        self.assertEqual(result.tool_calls[0]["call_id"], "c")

    def test_chat_tools_reasoning_and_cached_tokens(self):
        body = response("chat_completions")
        message = body["choices"][0]["message"]
        message.update(content=None, reasoning_content="private",
                       tool_calls=[{"id": "c", "type": "function", "function": {"name": "tool", "arguments": "{}"}}])
        body["usage"]["prompt_tokens_details"] = {"cached_tokens": 4}
        result = normalize_response(profile("chat_completions"), HTTPResult(body, 1))
        self.assertEqual(result.reasoning, "private")
        self.assertEqual(result.usage.cached_input_tokens, 4)
        self.assertEqual(len(result.tool_calls), 1)

    def test_do_not_invent_returned_model_or_accept_invalid_response(self):
        body = response()
        del body["model"]
        self.assertIsNone(normalize_response(profile(), HTTPResult(body, 1)).returned_model)
        for invalid in ({"error": {"message": "private-error"}}, {"output": []},
                        {"status": "failed", "output": []}):
            with self.subTest(body=invalid), self.assertRaises(ProtocolError):
                normalize_response(profile(), HTTPResult(invalid, 1))

    def test_responses_accepts_nullable_error_field(self):
        body = response()
        body["error"] = None
        self.assertEqual(normalize_response(profile(), HTTPResult(body, 1)).text, "OK")
        transport = JSONTransport("fake-key", "bearer", 10)
        raw = io.BytesIO(json.dumps(body).encode())
        raw.headers = {}
        with patch.object(transport._opener, "open", return_value=raw):
            result = transport.request("POST", "https://xcode.best/v1/responses", {})
            self.assertEqual(result.body["status"], "completed")

    def test_request_limit_is_atomic_across_workers(self):
        client = XcodeClient(profile(), max_requests=3)
        payload = client.prepare([{"role": "user", "content": "question"}])
        with patch("xcodeapi.client.resolve_api_key", return_value="fake-key"), \
             patch.object(JSONTransport, "request", return_value=HTTPResult(response(), 1)) as request:
            def invoke():
                try:
                    return client.send(payload).text
                except RequestLimitError:
                    return "limited"
            with ThreadPoolExecutor(max_workers=8) as workers:
                outcomes = list(workers.map(lambda _: invoke(), range(12)))
            self.assertEqual(outcomes.count("OK"), 3)
            self.assertEqual(client.attempts, 3)
            self.assertEqual(request.call_count, 3)

    def test_failed_attempt_counts_once_without_retry(self):
        client = XcodeClient(profile(), max_requests=1)
        with patch("xcodeapi.client.resolve_api_key", return_value="fake-key"), \
             patch.object(JSONTransport, "request", side_effect=HTTPAPIError(429)) as request:
            with self.assertRaises(HTTPAPIError):
                client.generate([{"role": "user", "content": "question"}])
            self.assertEqual(request.call_count, 1)
            self.assertEqual(client.attempts, 1)

    def test_models_endpoint_uses_budget_without_generation_model(self):
        client = XcodeClient(profile(model=""), max_requests=1)
        with patch("xcodeapi.client.resolve_api_key", return_value="fake-key"), \
             patch.object(JSONTransport, "request", return_value=HTTPResult({"data": [{"id": "model-a"}]}, 1)) as request:
            self.assertEqual(client.list_models()[0]["id"], "model-a")
            request.assert_called_once_with("GET", "https://xcode.best/v1/models", None)
            self.assertEqual(client.attempts, 1)

    def test_auth_headers_and_normal_http_result(self):
        for auth in ("bearer", "api_key"):
            transport = JSONTransport("fake-key", auth, 10)
            reply = io.BytesIO(json.dumps(response("messages")).encode())
            reply.headers = {"x-request-id": "safe-request-id"}
            with patch.object(transport._opener, "open", return_value=reply) as opened:
                result = transport.request("POST", "https://xcode.best/v1/messages", {"model": "test-model"})
                request = opened.call_args.args[0]
                headers = {key.lower(): value for key, value in request.header_items()}
                self.assertEqual(headers["anthropic-version"], "2023-06-01")
                self.assertEqual(headers["authorization"] if auth == "bearer" else headers["x-api-key"],
                                 "Bearer fake-key" if auth == "bearer" else "fake-key")
                self.assertEqual(result.request_id, "safe-request-id")

    def test_transport_errors_redacted_and_redirects_not_followed(self):
        for status in (302, 401, 429, 500):
            transport = JSONTransport("fake-key", "bearer", 10)
            error = urllib.error.HTTPError("https://xcode.best", status, "private-provider-message", {},
                                           io.BytesIO(b"private-error-body-fake-key"))
            with patch.object(transport._opener, "open", side_effect=error) as opened:
                with self.assertRaises(HTTPAPIError) as caught:
                    transport.request("POST", "https://xcode.best/v1/responses", {"model": "test-model"})
                self.assertNotIn("private", str(caught.exception))
                self.assertNotIn("fake-key", str(caught.exception))
                self.assertEqual(opened.call_count, 1)
            handlers = [h for h in transport._opener.handlers if hasattr(h, "redirect_request")]
            self.assertEqual(len(handlers), 1)
            self.assertIsNone(handlers[0].redirect_request(None, None, 302, "", {}, "https://example.test"))

    def test_timeout_and_non_json_error_are_sanitized(self):
        transport = JSONTransport("fake-key", "bearer", 10)
        with patch.object(transport._opener, "open", side_effect=TimeoutError("private-fake-key")):
            with self.assertRaises(TransportError) as caught:
                transport.request("POST", "https://xcode.best/v1/responses", {})
            self.assertNotIn("private", str(caught.exception))
        with patch.object(transport._opener, "open", side_effect=http.client.IncompleteRead(b"private-fake-key")):
            with self.assertRaises(TransportError):
                transport.request("POST", "https://xcode.best/v1/responses", {})
        reply = io.BytesIO(b"private-non-json-body")
        reply.headers = {}
        with patch.object(transport._opener, "open", return_value=reply):
            with self.assertRaises(ProtocolError) as caught:
                transport.request("POST", "https://xcode.best/v1/responses", {})
            self.assertNotIn("private", str(caught.exception))

    def test_cli_live_guard_and_offline_prepare(self):
        with patch("xcodeapi.client.resolve_api_key", side_effect=AssertionError("Key read forbidden")):
            for args in (["smoke", "--profile", "gpt", "--model", "test-model"],
                         ["models", "--profile", "gpt"]):
                with redirect_stdout(io.StringIO()), patch("sys.stderr", new=io.StringIO()):
                    self.assertEqual(main(args), 2)
            output = io.StringIO()
            with redirect_stdout(output):
                self.assertEqual(main(["prepare", "--profile", "gpt", "--model", "test-model"]), 0)
            self.assertFalse(json.loads(output.getvalue())["credentials_read"])


if __name__ == "__main__":
    unittest.main()
