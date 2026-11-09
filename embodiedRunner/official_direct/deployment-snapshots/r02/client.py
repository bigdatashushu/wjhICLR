"""Single-attempt OpenAI-compatible HTTP client using only the standard library."""
import json
import os
import socket
import urllib.error
import urllib.parse
import urllib.request

from .sources import ContractError, OperationalError, canonical, digest


class ModelClient:
    def __init__(self, config):
        self.config = config
        self.key = os.environ.get(config["api_key_env"], "")
        base_url = os.environ.get(config["base_url_env"], "https://api.openai.com/v1").rstrip("/")
        parts = urllib.parse.urlsplit(base_url)
        if parts.scheme not in ("http", "https") or not parts.netloc or parts.username or parts.query:
            raise ContractError("invalid_model_endpoint")
        self.url = base_url + "/chat/completions"
        # Prevent redirects from moving Authorization to a different endpoint.
        class NoRedirect(urllib.request.HTTPRedirectHandler):
            def redirect_request(self, *args, **kwargs):
                return None
        self.opener = urllib.request.build_opener(NoRedirect())

    def assert_available(self):
        if not self.key:
            raise OperationalError("missing_model_credentials")

    def safe(self, value):
        if self.key and self.key in canonical(value).decode():
            raise OperationalError("credential_in_model_content")
        return value

    def payload(self, messages, schema):
        c = self.config
        payload = {"model": c["name"], "messages": messages}
        if c["response_format"] == "official_schema":
            payload["response_format"] = {"type": "json_schema", "json_schema": {
                "name": "embodied_planning", "schema": schema}}
        else:
            payload["response_format"] = {"type": "json_object"}
        if c["name"].startswith("gpt-5"):
            payload["max_completion_tokens"] = c["max_tokens"]
        else:
            payload.update(max_tokens=c["max_tokens"], temperature=c["temperature"])
        payload.update(c["extra_body"])
        return self.safe(payload)

    def complete(self, payload):
        self.assert_available()
        request = urllib.request.Request(self.url, data=canonical(self.safe(payload)), headers={
            "Authorization": "Bearer " + self.key, "Content-Type": "application/json"},
            method="POST")
        try:
            with self.opener.open(request, timeout=self.config["timeout_s"]) as response:
                raw = response.read(32 * 1024**2 + 1)
            if len(raw) > 32 * 1024**2:
                raise OperationalError("model_response_too_large")
            value = json.loads(raw)
            message = value["choices"][0]["message"]
            text = message["content"]
            if not isinstance(text, str) or message.get("tool_calls") or message.get("refusal"):
                raise OperationalError("invalid_model_response")
            self.safe(value)
            usage = {k: v for k, v in value.get("usage", {}).items()
                     if k in ("prompt_tokens", "completion_tokens", "total_tokens")
                     and type(v) is int and v >= 0}
            # Preserve the actual returned message, including optional reasoning.
            return {"text": text, "message": message, "usage": usage,
                    "model": value.get("model"), "response_sha256": digest(raw),
                    "finish_reason": value["choices"][0].get("finish_reason")}
        except urllib.error.HTTPError as exc:
            # Never persist service error bodies, URLs or exception messages.
            raise OperationalError("model_http_error", exc.code) from None
        except (TimeoutError, socket.timeout):
            raise OperationalError("model_timeout") from None
        except urllib.error.URLError:
            raise OperationalError("model_transport_error") from None
        except (KeyError, ValueError, TypeError, IndexError, UnicodeError):
            raise OperationalError("invalid_model_response") from None
