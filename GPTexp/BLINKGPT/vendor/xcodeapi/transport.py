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
