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
            raise ProtocolError("Responses streaming requires a streaming POST payload.")
        try:
            encoded = json.dumps(payload, ensure_ascii=False, allow_nan=False).encode("utf-8")
        except (ValueError, TypeError):
            raise ProtocolError("Request is not valid JSON.") from None
        req = urllib.request.Request(
            url, data=encoded, method=method,
            headers={"Accept": "text/event-stream", "Content-Type": "application/json",
                     "Authorization": "Bearer " + self._api_key})
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
