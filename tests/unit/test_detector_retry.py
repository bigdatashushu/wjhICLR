"""Transport retries are bounded and distinct from healthy empty detections."""

from unittest.mock import Mock

import numpy as np
import pytest
import requests

from skill3d.segmentation import open_vocab_detector as ovd


def _response(status=200, data=None):
    response = Mock(status_code=status)
    response.raise_for_status.side_effect = requests.HTTPError(str(status)) if status >= 400 else None
    response.json.return_value = {"success": True, "detections": []} if data is None else data
    return response


def _detect():
    return ovd.detect(np.zeros((16, 16, 3), dtype=np.uint8), "chair.", endpoint="http://detector")


@pytest.mark.parametrize("first", [
    requests.Timeout("timeout"), requests.ConnectionError("refused"),
    _response(429), _response(503),
])
def test_transient_recovery_records_attempts_without_final_failure(monkeypatch, first):
    post = Mock(side_effect=[first, _response(data={
        "success": True,
        "detections": [{"label": "chair", "bbox": [1, 1, 12, 12], "confidence": .9}],
    })])
    monkeypatch.setattr(requests, "post", post)
    monkeypatch.setattr(ovd.time, "sleep", lambda _: None)
    with ovd.capture_detector_failures() as failures, ovd.capture_detector_attempts() as attempts:
        assert len(_detect()) == 1
    assert not failures and not ovd.detect.last_error
    assert post.call_count == 2
    assert attempts[0]["will_retry"] and attempts[1]["recovered"]
    assert post.call_args_list[0].kwargs["timeout"] == 120


@pytest.mark.parametrize("response", [
    _response(400), _response(data={"success": False, "error": "bad model input"}),
    _response(data=[]), _response(data={"success": True}),
])
def test_business_or_schema_errors_do_not_retry(monkeypatch, response):
    post = Mock(return_value=response)
    monkeypatch.setattr(requests, "post", post)
    with ovd.capture_detector_failures() as failures, ovd.capture_detector_attempts() as attempts:
        assert _detect() == []
    assert post.call_count == 1
    assert len(failures) == 1 and ovd.detect.last_error
    assert not attempts[0]["retryable"]


def test_exhausted_retries_report_one_final_failure(monkeypatch):
    post = Mock(side_effect=requests.Timeout("timeout"))
    monkeypatch.setattr(requests, "post", post)
    monkeypatch.setattr(ovd.time, "sleep", lambda _: None)
    with ovd.capture_detector_failures() as failures, ovd.capture_detector_attempts() as outer:
        with ovd.capture_detector_attempts() as inner:
            assert _detect() == []
    assert post.call_count == 3
    assert len(failures) == 1
    assert outer == inner and len(inner) == 3
    assert not inner[-1]["will_retry"]


def test_healthy_empty_is_not_an_outage(monkeypatch):
    post = Mock(return_value=_response())
    monkeypatch.setattr(requests, "post", post)
    with ovd.capture_detector_failures() as failures, ovd.capture_detector_attempts() as attempts:
        assert _detect() == []
    assert post.call_count == 1 and not failures
    assert attempts[0]["status"] == "empty"


def test_total_budget_prevents_retry_after_slow_failure(monkeypatch):
    clock = [0.0]
    def fail(*args, **kwargs):
        clock[0] = 181.0
        raise requests.Timeout("timeout")
    post = Mock(side_effect=fail)
    monkeypatch.setattr(requests, "post", post)
    monkeypatch.setattr(ovd.time, "monotonic", lambda: clock[0])
    with ovd.capture_detector_failures() as failures:
        assert _detect() == []
    assert post.call_count == 1 and len(failures) == 1


def test_slow_timeout_leaves_bounded_budget_for_recovery(monkeypatch):
    clock, timeouts = [0.0], []
    def post(*args, **kwargs):
        timeouts.append(kwargs["timeout"])
        if len(timeouts) == 1:
            clock[0] = 120.0
            raise requests.Timeout("first timeout")
        return _response()
    monkeypatch.setattr(requests, "post", post)
    monkeypatch.setattr(ovd.time, "monotonic", lambda: clock[0])
    monkeypatch.setattr(ovd.time, "sleep", lambda delay: clock.__setitem__(0, clock[0] + delay))
    with ovd.capture_detector_failures() as failures:
        assert _detect() == []
    assert not failures
    assert timeouts[0] == 120
    assert 59 < timeouts[1] < 60
