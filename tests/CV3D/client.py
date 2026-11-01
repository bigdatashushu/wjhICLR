"""Qwen native tool transport. Exactly one HTTP attempt per governed request."""
from __future__ import annotations

import http.client
import json
import math
import os
import time
import urllib.error
import urllib.request

from skill3d.synthesis.bailian_client import BailianClient, _transport_error
from skill3d.synthesis.request_audit import describe_request, payload_bytes


class ServiceError(RuntimeError):
    """Only locally generated error codes; no provider bodies or credential values."""

    def __init__(self, code, *, http_status=None):
        super().__init__(code)
        self.code, self.http_status = code, http_status


class NativeToolClient(BailianClient):
    def __init__(self, config, *, base_url=None, api_key_env="DASHSCOPE_API_KEY",
                 opener=None):
        kwargs = {"base_url": base_url} if base_url else {}
        super().__init__(
            model=config.model, temperature=config.temperature,
            max_output_tokens=config.max_tokens, timeout_s=config.timeout_s,
            api_key_env=api_key_env, opener=opener, **kwargs)
        self.config = config

    def payload(self, messages, tools):
        payload = self.build_payload(
            messages, max_tokens=self.config.max_tokens, seed=self.config.seed,
            temperature=self.config.temperature, enable_thinking=self.config.thinking)
        if tools:
            payload.update(tools=tools, tool_choice="auto")
        return payload

    def chat_tools(self, messages, tools):
        payload = self.payload(messages, tools)
        self.last_usage, self.last_service_error = {}, {}
        self.last_request_identity = describe_request(
            payload, model_identity=self.audit_identity(seed=self.config.seed))
        key = os.environ.get(self.api_key_env, "").strip()
        if not key:
            raise ServiceError("missing_credentials")
        request = urllib.request.Request(
            self.base_url + "/chat/completions", data=payload_bytes(payload),
            headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
            method="POST")
        started = time.monotonic()
        try:
            with self._opener(request, timeout=self.timeout_s) as response:
                body = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            status = int(exc.code)
            exc.close()
            raise ServiceError("http_error", http_status=status) from None
        except (urllib.error.URLError, TimeoutError, OSError, http.client.HTTPException) as exc:
            code, _ = _transport_error(exc)
            raise ServiceError(code) from None
        except (UnicodeError, ValueError):
            raise ServiceError("invalid_response_json") from None
        try:
            if not isinstance(body, dict) or body.get("model") != self.model:
                raise ValueError("model identity mismatch")
            choices = body["choices"]
            if not isinstance(choices, list) or len(choices) != 1:
                raise ValueError("one choice required")
            choice, usage = choices[0], body.get("usage") or {}
            message = choice["message"]
            if not isinstance(message, dict) or message.get("role") != "assistant":
                raise ValueError("invalid assistant message")
            self.last_usage = {
                k: v for k, v in usage.items()
                if k in {"prompt_tokens", "completion_tokens", "total_tokens"}
                and type(v) in (int, float) and math.isfinite(v) and v >= 0
            }
            content, reasoning = message.get("content"), message.get("reasoning_content")
            calls = message.get("tool_calls") or []
            if (content is not None and not isinstance(content, str)
                    or reasoning is not None and not isinstance(reasoning, str)
                    or not isinstance(calls, list)):
                raise ValueError("invalid message fields")
            clean_calls = []
            for call in calls:
                function = call["function"]
                if (call.get("type") != "function" or not isinstance(call.get("id"), str)
                        or not call["id"] or not isinstance(function.get("name"), str)
                        or not isinstance(function.get("arguments"), str)):
                    raise ValueError("invalid native function call")
                clean_calls.append({
                    "id": call["id"], "type": "function",
                    "function": {"name": function["name"], "arguments": function["arguments"]},
                })
            if not clean_calls and not content:
                raise ValueError("empty assistant response")
            clean = {"role": "assistant", "content": content}
            if reasoning is not None:
                clean["reasoning_content"] = reasoning
            if clean_calls:
                clean["tool_calls"] = clean_calls
            return {
                "message": clean, "finish_reason": choice.get("finish_reason"),
                "model": body["model"], "usage": self.last_usage,
                "elapsed_seconds": round(time.monotonic() - started, 6),
            }
        except (ValueError, KeyError, TypeError, AttributeError, IndexError):
            raise ServiceError("response_schema_or_model_mismatch") from None
