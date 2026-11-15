"""One HTTPS JSON attempt per call; no retries, redirects or implicit logging."""

from __future__ import annotations

import json
import http.client
import socket
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from time import monotonic

from .errors import HTTPAPIError, ProtocolError, TransportError


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


@dataclass(frozen=True)
class HTTPResult:
    body: dict = field(repr=False)
    elapsed_seconds: float
    request_id: str | None = None


class JSONTransport:
    def __init__(self, api_key: str, auth: str, timeout_seconds: float):
        self._api_key = api_key
        self.auth = auth
        self.timeout_seconds = timeout_seconds
        # Retain normal TLS verification and the user's normal proxy settings.
        self._opener = urllib.request.build_opener(_NoRedirect())

    def request(self, method: str, url: str, payload: dict | None = None) -> HTTPResult:
        try:
            encoded = None if payload is None else json.dumps(
                payload, ensure_ascii=False, allow_nan=False
            ).encode("utf-8")
        except (ValueError, TypeError):
            raise ProtocolError("Request is not valid JSON.") from None
        headers = {"Accept": "application/json", "Content-Type": "application/json"}
        if self.auth == "api_key":
            headers["x-api-key"] = self._api_key
        else:
            headers["Authorization"] = "Bearer " + self._api_key
        if url.endswith("/messages") or self.auth == "api_key":
            headers["anthropic-version"] = "2023-06-01"
        req = urllib.request.Request(url, data=encoded, headers=headers, method=method)
        started = monotonic()
        try:
            with self._opener.open(req, timeout=self.timeout_seconds) as response:
                raw = response.read()
                request_id = response.headers.get("x-request-id")
        except urllib.error.HTTPError as error:
            status = error.code
            error.close()
            raise HTTPAPIError(status) from None
        except (OSError, urllib.error.URLError, socket.timeout, http.client.HTTPException, ValueError):
            raise TransportError("HTTPS request failed or timed out; outcome unknown. No automatic retry.") from None
        try:
            body = json.loads(raw)
        except (ValueError, UnicodeError):
            raise ProtocolError("Provider returned a non-JSON response.") from None
        # Responses includes "error": null on normal successful responses.
        if not isinstance(body, dict) or body.get("error") is not None:
            raise ProtocolError("Provider returned an invalid response or an API error object.")
        if (
            not isinstance(request_id, str) or len(request_id) > 160
            or self._api_key in request_id
            or not all(c.isalnum() or c in "-_." for c in request_id)
        ):
            request_id = None
        return HTTPResult(body, monotonic() - started, request_id)


class ResponsesStreamTransport(JSONTransport):
    """Read one native Responses SSE attempt through its authoritative terminal event."""

    def request(self, method: str, url: str, payload: dict | None = None) -> HTTPResult:
        if method != "POST" or not isinstance(payload, dict) or payload.get("stream") is not True:
            raise ProtocolError("Native streaming requires a streaming POST payload.")
        try:
            encoded = json.dumps(payload, ensure_ascii=False, allow_nan=False).encode("utf-8")
        except (ValueError, TypeError):
            raise ProtocolError("Request is not valid JSON.") from None
        headers = {"Accept": "text/event-stream", "Content-Type": "application/json"}
        if self.auth == "api_key":
            headers["x-api-key"] = self._api_key
        else:
            headers["Authorization"] = "Bearer " + self._api_key
        if url.endswith("/messages") or self.auth == "api_key":
            headers["anthropic-version"] = "2023-06-01"
        req = urllib.request.Request(url, data=encoded, method=method, headers=headers)
        started = monotonic()
        try:
            with self._opener.open(req, timeout=self.timeout_seconds) as response:
                request_id = response.headers.get("x-request-id")
                body = self._read_terminal(response, started)
        except urllib.error.HTTPError as error:
            status = error.code
            error.close()
            raise HTTPAPIError(status) from None
        except (OSError, urllib.error.URLError, socket.timeout, http.client.HTTPException, ValueError):
            raise TransportError("HTTPS stream failed or timed out; outcome unknown. No automatic retry.") from None
        if (
            not isinstance(request_id, str) or len(request_id) > 160
            or self._api_key in request_id
            or not all(c.isalnum() or c in "-_." for c in request_id)
        ):
            request_id = None
        return HTTPResult(body, monotonic() - started, request_id)

    def _read_terminal(self, response, started: float) -> dict:
        data = []
        event_bytes = 0
        total_bytes = 0
        while True:
            if monotonic() - started > self.timeout_seconds:
                raise TransportError("Responses stream deadline exceeded; outcome unknown.")
            line = response.readline(16 * 1024 * 1024 + 1)
            total_bytes += len(line)
            event_bytes += len(line)
            if event_bytes > 16 * 1024 * 1024 or total_bytes > 64 * 1024 * 1024:
                raise ProtocolError("Responses stream exceeded the event-size limit.")
            if not line:
                raise TransportError("Responses stream ended without a terminal response; outcome unknown.")
            line = line.rstrip(b"\r\n")
            if line:
                if line.startswith(b"data:"):
                    data.append(line[5:].lstrip(b" "))
                continue
            if not data:
                event_bytes = 0
                continue
            try:
                event = json.loads(b"\n".join(data))
            except (ValueError, UnicodeError):
                raise ProtocolError("Responses stream contained an invalid JSON event.") from None
            data = []
            event_bytes = 0
            if not isinstance(event, dict):
                raise ProtocolError("Responses stream contained an invalid event object.")
            kind = event.get("type")
            if kind in {"response.completed", "response.incomplete", "response.failed"}:
                body = event.get("response")
                if not isinstance(body, dict) or body.get("status") != kind.split(".")[1]:
                    raise ProtocolError("Responses terminal event did not match its response status.")
                # Keep terminal usage even when failed/incomplete answer parsing fails.
                return body
            if kind == "error":
                raise ProtocolError("Responses stream returned an API error event.")


class MessagesStreamTransport(ResponsesStreamTransport):
    """Assemble native Messages blocks only through a complete message_stop."""

    def _read_terminal(self, response, started: float) -> dict:
        data, blocks, fragments, active = [], {}, {}, set()
        body = None
        total_bytes = event_bytes = 0
        while True:
            if monotonic() - started > self.timeout_seconds:
                raise TransportError("Messages stream deadline exceeded; outcome unknown.")
            line = response.readline(16 * 1024 * 1024 + 1)
            total_bytes += len(line)
            event_bytes += len(line)
            if event_bytes > 16 * 1024 * 1024 or total_bytes > 64 * 1024 * 1024:
                raise ProtocolError("Messages stream exceeded the event-size limit.")
            if not line:
                raise TransportError("Messages stream ended without a terminal message; outcome unknown.")
            line = line.rstrip(b"\r\n")
            if line:
                if line.startswith(b"data:"):
                    data.append(line[5:].lstrip(b" "))
                continue
            if not data:
                event_bytes = 0
                continue
            try:
                event = json.loads(b"\n".join(data))
            except (ValueError, UnicodeError):
                raise ProtocolError("Messages stream contained an invalid JSON event.") from None
            data, event_bytes = [], 0
            if not isinstance(event, dict):
                raise ProtocolError("Messages stream contained an invalid event object.")
            kind = event.get("type")
            if kind == "ping":
                continue
            if kind == "error":
                raise ProtocolError("Messages stream returned an API error event.")
            if kind == "message_start":
                if body is not None or not isinstance(event.get("message"), dict):
                    raise ProtocolError("Invalid Messages start event.")
                body = event["message"]
                if not isinstance(body.get("usage"), dict) or body.get("content") != []:
                    raise ProtocolError("Invalid Messages initial content or usage.")
                continue
            if body is None:
                raise ProtocolError("Messages event preceded message_start.")
            if kind == "content_block_start":
                index, block = event.get("index"), event.get("content_block")
                if type(index) is not int or index != len(blocks) or not isinstance(block, dict):
                    raise ProtocolError("Invalid Messages block start.")
                blocks[index], fragments[index] = block, []
                active.add(index)
            elif kind == "content_block_delta":
                index, delta = event.get("index"), event.get("delta")
                if index not in active or not isinstance(delta, dict):
                    raise ProtocolError("Invalid Messages block delta.")
                fields = {"text_delta": ("text", "text"), "thinking_delta": ("thinking", "thinking"),
                          "signature_delta": ("signature", "signature"),
                          "input_json_delta": ("input", "partial_json")}
                if delta.get("type") not in fields:
                    raise ProtocolError("Unsupported Messages delta.")
                field, source = fields[delta["type"]]
                if not isinstance(delta.get(source), str):
                    raise ProtocolError("Invalid Messages delta text.")
                if field == "input":
                    fragments[index].append(delta[source])
                else:
                    blocks[index][field] = blocks[index].get(field, "") + delta[source]
            elif kind == "content_block_stop":
                index = event.get("index")
                if index not in active:
                    raise ProtocolError("Invalid Messages block stop.")
                if fragments[index]:
                    try:
                        blocks[index]["input"] = json.loads("".join(fragments[index]))
                    except ValueError:
                        raise ProtocolError("Invalid Messages tool input.") from None
                active.remove(index)
            elif kind == "message_delta":
                delta, usage = event.get("delta"), event.get("usage")
                if not isinstance(delta, dict) or not isinstance(usage, dict):
                    raise ProtocolError("Invalid Messages final delta or usage.")
                body.update(delta)
                body["usage"].update(usage)
            elif kind == "message_stop":
                if active or not isinstance(body.get("stop_reason"), str):
                    raise ProtocolError("Messages terminal event lacks completed blocks or stop_reason.")
                body["content"] = [blocks[index] for index in sorted(blocks)]
                return body
            else:
                raise ProtocolError("Unsupported Messages event.")
