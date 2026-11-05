"""Offline checks for Terra identity, passive TCP visibility and failed requests."""
from __future__ import annotations

from copy import deepcopy
import json
import os
from pathlib import Path
import tempfile
from unittest.mock import patch

from .api_pool import ApiPool
from .connection_monitor import collect, report
from .launch import validate_api
from .model_binding import bound_spec
from .probe import png
from .selfcheck import Opener
from .terra_ops import ADAPTER_HASHES, terra_config
from .traces import file_digest, read, write
from .transport import ResponsesClient


def main():
    config = terra_config()
    assert config["model"] == "gpt-5.6-terra"
    assert all(a["model"] == config["model"] and not a["credentials_file"] for a in config["apis"])
    assert config["max_requests"] == 13070 and config["max_inflight"] == 10
    assert {n: file_digest(Path(__file__).parent / n) for n in ADAPTER_HASHES} == ADAPTER_HASHES
    for original in (
            {"solver": {"expected_model": "gpt-6.1-sol", "max_tokens": 8192}, "arm": "harness"},
            {"config": {"model": "gpt-6.1-sol", "max_rounds": 6}, "arm": "direct"}):
        bound, changed = bound_spec(original, config["model"])
        assert changed in (["solver.expected_model"], ["config.model"])
        assert original != bound and original.get("solver", original.get("config")) != bound.get(
            "solver", bound.get("config"))
    with tempfile.TemporaryDirectory(prefix="terra-offline-check-") as temp:
        root = Path(temp)
        api = config["apis"][0]
        client = ResponsesClient(api, ApiPool(root / "payload-pool", config), root / "req", "offline")
        messages = [{"role": "user", "content": [{"type": "text", "text": "preserve\noriginal"}, png(255, 0, 0)]}]
        payload = client.build_payload(messages, max_tokens=8192, seed=137, temperature=0)
        assert payload["model"] == "gpt-5.6-terra" and payload["reasoning"] == {"effort": "high"}
        assert payload["seed"] == 137 and payload["temperature"] == 0 and payload["max_output_tokens"] == 8192
        body = {"model": config["model"], "status": "completed", "reasoning": {"effort": "high"},
                "output": [{"type": "message", "content": [
                    {"type": "output_text", "text": '{"first":"red","second":"blue"}'}]}]}
        with patch("GPTexp.transport.resolve_key", return_value="offline-only-sentinel"), patch(
                "GPTexp.transport.urllib.request.build_opener", return_value=Opener(body)):
            validation = validate_api(api, config, root / "validation")
        assert validation["validated"] and validation["response_identity"]["model"] == "gpt-5.6-terra"
        wrong = deepcopy(body)
        wrong["model"] = "gpt-6.1-sol"
        with patch("GPTexp.transport.resolve_key", return_value="offline-only-sentinel"), patch(
                "GPTexp.transport.urllib.request.build_opener", return_value=Opener(wrong)):
            validation = validate_api(api, config, root / "wrong-validation")
        assert not validation["validated"] and validation["reason_code"] == "actual_model_changed"
        run = root / "run"
        proc = root / "proc"
        proc.mkdir()
        (proc / "net").mkdir()
        write(run / "launch.config.json", config)
        write(run / "bootstrap.json", {"stage": "validating_apis", "ignored_secret": "sk-fake-private"})
        owner = proc / "100"
        (owner / "fd").mkdir(parents=True)
        (owner / "stat").write_text("100 (python) S 1 " + "0 " * 17 + "42 0\n")
        (owner / "fd/3").symlink_to("socket:[900]")
        (proc / "net/tcp").write_text(
            "header\n0: 00000000:1234 00000000:01BB 01 0:0 0:0 0 0 0 900\n")
        write(run / "api-probes/xcode-1/pool/state.json", {
            "requests": [{"request_id": 0, "api": "xcode-1", "status": "started", "started": 10,
                          "ignored_secret": "sk-fake-private"}]})
        status, events = collect(run, 100, 42, proc=proc, now=60)
        assert status["owner_alive"] and status["https_tcp_states"] == {"ESTABLISHED": 1}
        assert status["apis"][0]["active_requests"][0]["age_seconds"] == 50
        assert status["apis"][0]["https_tcp_states"] == {}
        assert not status["validated_apis"] and not status["monitor_requests"]
        assert "sk-fake-private" not in json.dumps(status)
        write(run / "api-probes/xcode-1/pool/state.json", {
            "requests": [{"request_id": 0, "api": "xcode-1", "status": "error", "started": 10,
                          "finished": 60, "http_status": 524, "reason_code": "http_error"}]})
        os.unlink(owner / "fd/3")
        status, events = collect(run, 100, 43, proc=proc, now=65)
        assert not status["owner_alive"] and not status["https_tcp_states"]
        assert status["stage"] == "owner_exited"
        assert events[0]["http_status"] == 524 and not status["apis"][0]["active_requests"]
        assert "524" in report(status)
        assert not status["automatic_retry"] and not status["reference_answers_read"]
    print(json.dumps({
        "status": "passed", "live_model_requests": 0,
        "checks": ["terra_model_high_and_original_payload", "new_image_validation_and_wrong_model_rejection",
                   "runtime_binding_changes_only_model_identity",
                   "frozen_adapter_hashes", "tcp_is_not_api_validation", "active_request_age",
                   "owner_pid_reuse_detection", "http524_and_disconnected_observation",
                   "monitor_secret_allowlist_and_no_requests"]}))


if __name__ == "__main__":
    main()
