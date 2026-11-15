"""Terminal SSE evidence must survive chunked transport without inventing usage."""
import io
import json
from time import monotonic

import pytest
from xcodeapi.errors import ProtocolError, TransportError
from xcodeapi.transport import ResponsesStreamTransport


def event(kind, **fields):
    return ("event: " + kind + "\r\ndata: " +
            json.dumps({"type": kind, **fields}) + "\r\n\r\n").encode()


@pytest.mark.parametrize("status", ["completed", "incomplete", "failed"])
def test_terminal_response_keeps_original_output_and_usage(status):
    terminal = {"status": status, "output": [{"text": "authoritative"}],
                "usage": {"input_tokens": 12, "output_tokens": 8, "total_tokens": 20}}
    raw = (b": keepalive\r\n\r\n" +
           event("response.output_text.delta", delta="partial") +
           event("response." + status, response=terminal))
    transport = ResponsesStreamTransport("offline-key", "bearer", 10)
    assert transport._read_terminal(io.BytesIO(raw), monotonic()) == terminal


def test_truncated_stream_does_not_promote_deltas_to_a_paid_final_result():
    transport = ResponsesStreamTransport("offline-key", "bearer", 10)
    with pytest.raises(TransportError, match="without a terminal"):
        transport._read_terminal(
            io.BytesIO(event("response.output_text.delta", delta="partial")), monotonic())


def test_invalid_terminal_status_is_rejected():
    transport = ResponsesStreamTransport("offline-key", "bearer", 10)
    with pytest.raises(ProtocolError, match="status"):
        transport._read_terminal(
            io.BytesIO(event("response.completed", response={"status": "in_progress"})),
            monotonic())
