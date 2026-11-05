"""One offline adapter/capture check; synthetic responses, never benchmark scores."""
from __future__ import annotations

import argparse
from pathlib import Path
from unittest.mock import patch

from .adapters import ROOT, jsonable, setup, solve
from .api_pool import ApiPool
from .traces import digest, read, write
from .transport import ResponsesClient


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--preflight", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--arm", choices=["direct", "harness"], default="harness")
    args = parser.parse_args()
    receipt = read(args.preflight)
    dataset = receipt["dataset"]
    setup(dataset)
    spec = next(s for s in receipt["specs"] if s["arm"] == args.arm)
    if args.arm == "direct" and any(key in spec for key in (
            "prepared_root", "prepared_directory", "artifact_sha256", "skill", "allowed_tools")):
        raise AssertionError("direct_received_geometry")
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    config = read(ROOT / "config.json")

    class Synthetic(ResponsesClient):
        def chat(self, messages, **kwargs):
            self.index += 1
            payload = self.build_payload(messages, **kwargs)
            if args.arm == "direct":
                text = '{"answer":"A"}'
            elif self.index == 1:
                program = ('def solve(ctx):\n'
                           '    r = ctx.tools.view_depth_summary()\n'
                           '    YieldObservations([r["result_id"]])\n')
                text = "```python\n" + program + "```"
            elif dataset == "mindcube-tiny":
                text = '```python\ndef solve(ctx):\n    ReturnAnswer("A")\n```'
            else:
                text = ('```python\ndef solve(ctx):\n'
                        '    ReturnAnswer(AnswerPayload(value="A", unit="option", basis="visual_estimate"))\n```')
            # The reserved final slot requires exactly the original JSON format.
            if self.index >= (6 if dataset in ("cv3d", "mindcube-tiny") else 3):
                text = '{"answer":"A"}'
            self.last_response_identity = {"model": self.model, "model_reported": True}
            self.last_generation = {"finish_reason": "stop", "status": "completed"}
            self.last_usage = {}
            write(output / "synthetic_requests" / f"{self.index:03d}.json", {
                "native_payload": payload, "text": text, "synthetic_response": True,
                "benchmark_evidence": False, "reference_answer_visible": False}, exclusive=True)
            return text

    client = Synthetic(config["apis"][0], ApiPool(output / "unused-pool", config),
                       output / "unused-requests", "offline-smoke")
    with patch("socket.create_connection", side_effect=AssertionError("live_network_forbidden")), patch(
            "urllib.request.OpenerDirector.open", side_effect=AssertionError("live_network_forbidden")):
        result = solve(spec, dataset, client, output, digest({"offline": dataset}))
    result = jsonable(result)
    cells = result.get("execution_trace", [])
    if dataset == "mindcube-tiny":
        traces = [read(path) for path in output.glob("traces/*/trace.json")]
        tool_events = [event for trace in traces for event in trace["events"]
                       if event["kind"] == "cell_completed"]
        captured_calls = sum(len(e["data"]["calls"]) for e in tool_events)
        authorization = sum(len(e["data"]["authorization_receipts"]) for e in tool_events)
    else:
        captured_calls = sum(len(c["tool_calls"]) for c in cells)
        authorization = sum(len(c["authorization_receipts"]) for c in cells)
    write(output / "smoke.json", {
        "dataset": dataset, "arm": args.arm, "benchmark_evidence": False, "synthetic_responses": True,
        "live_model_requests": 0, "result": result, "captured_tool_calls": captured_calls,
        "captured_authorization_receipts": authorization,
        "reference_answer_visible": False}, exclusive=True)
    expected_capture = (captured_calls == authorization == 0 if args.arm == "direct"
                        else captured_calls >= 1 and authorization >= 1)
    if result["status"] != "answered" or not expected_capture:
        raise AssertionError("adapter_or_capture_smoke_failed")
    print(dataset, args.arm, "offline adapter/capture check passed", captured_calls, authorization)


if __name__ == "__main__":
    main()
