"""Offline checks of the new protocol and credential/unknown-request boundaries."""
from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from pathlib import Path
import tempfile
from unittest.mock import patch

from .api_pool import ApiPool, resolve_key
from .export import export
from .launch import validate_api
from .probe import png
from .traces import digest, read, write
from .transport import APIError, ResponsesClient


class Reply:
    headers = {}

    def __init__(self, body):
        self.body = body

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass

    def read(self):
        return json.dumps(self.body).encode()


class Opener:
    def __init__(self, body):
        self.body = body
        self.attempts = 0

    def open(self, request, timeout):
        self.attempts += 1
        return Reply(self.body)


def main():
    config = read(Path(__file__).with_name("config.json"))
    config["max_requests"] = 1
    config["upstream_policy"] = {
        "require_reported_sampling_match": True,
        "require_reported_instructions_match": True,
        "require_reported_output_limit_match": True}
    key = "offline-only-credential-sentinel"
    body = {"model": config["model"], "status": "completed",
            "reasoning": {"effort": "high"}, "temperature": 0, "max_output_tokens": 8192,
            "output": [{"type": "message", "content": [{"type": "output_text", "text": '{"answer":"A"}'}]}],
            "usage": {"input_tokens": 10, "output_tokens": 5}}
    passed = []
    with tempfile.TemporaryDirectory(prefix="gptexp-selfcheck-") as temp:
        api = config["apis"][0]
        pool = ApiPool(Path(temp) / "good-pool", config)
        client = ResponsesClient(api, pool, Path(temp) / "good", "offline")
        messages = [{"role": "user", "content": [
            {"type": "text", "text": "Exact original text\nA. left\nB. right"},
            png(240, 10, 10), {"type": "text", "text": "second"}, png(10, 10, 240)]}]
        payload = client.build_payload(messages, max_tokens=8192, temperature=0, seed=137,
                                       enable_thinking=True)
        converted = payload["input"][0]["content"]
        assert converted[0]["text"] == messages[0]["content"][0]["text"]
        assert [p["image_url"] for p in converted if p["type"] == "input_image"] == [
            p["image_url"]["url"] for p in messages[0]["content"] if p["type"] == "image_url"]
        assert payload["seed"] == 137 and payload["temperature"] == 0
        assert payload["reasoning"] == {"effort": "high"} and payload["max_output_tokens"] == 8192
        opener = Opener(body)
        with patch("GPTexp.transport.resolve_key", return_value=key), patch(
                "GPTexp.transport.urllib.request.build_opener", return_value=opener):
            assert client.chat(messages, max_tokens=8192, seed=137) == '{"answer":"A"}'
        assert opener.attempts == 1 and client.last_response_identity["model"] == config["model"]
        passed.append("lossless_text_image_order_and_client_compatibility")
        for case, native in (("sampling", {**body, "temperature": 1}),
                             ("output_limit", {**body, "max_output_tokens": None}),
                             ("instructions", {**body, "instructions": "unsent synthetic instruction"}),
                             ("credential", {**body, "extra": key})):
            pool = ApiPool(Path(temp) / (case + "-pool"), config)
            client = ResponsesClient(api, pool, Path(temp) / case, "offline")
            opener = Opener(native)
            with patch("GPTexp.transport.resolve_key", return_value=key), patch(
                    "GPTexp.transport.urllib.request.build_opener", return_value=opener):
                try:
                    client.chat(messages, max_tokens=8192, seed=137)
                except (APIError, ValueError):
                    pass
                else:
                    raise AssertionError("unsafe_provider_deviation_accepted")
            assert opener.attempts == 1
            assert pool.paused(api)
            for path in (Path(temp) / case).glob("*.json"):
                assert key not in path.read_text()
            passed.append(case + "_rejected_and_no_automatic_retry")
        multi = deepcopy(config)
        multi["max_requests"] = multi["max_inflight"] = 2
        multi["apis"] = [deepcopy(config["apis"][0])]
        second = {**multi["apis"][0], "id": "offline-second",
                  "api_key_env": "OFFLINE_SECOND_API_KEY"}
        multi["apis"].append(second)
        keys = {a["id"]: key + "-" + a["id"] for a in multi["apis"]}
        routed = []

        class RoutedOpener(Opener):
            def open(self, request, timeout):
                routed.append(request.get_header("Authorization"))
                return super().open(request, timeout)

        pool = ApiPool(Path(temp) / "multi-pool", multi)
        opener = RoutedOpener(body)

        def routed_call(api):
            client = ResponsesClient(api, pool, Path(temp) / "multi" / api["id"], "offline-multi")
            return client.chat(messages, max_tokens=8192, seed=137)

        with patch("GPTexp.transport.resolve_key", side_effect=lambda a: keys[a["id"]]), patch(
                "GPTexp.transport.urllib.request.build_opener", return_value=opener):
            with ThreadPoolExecutor(max_workers=2) as executor:
                outputs = list(executor.map(routed_call, multi["apis"]))
        assert outputs == ['{"answer":"A"}'] * 2
        assert set(routed) == {"Bearer " + v for v in keys.values()} and opener.attempts == 2
        assert sum(row["requests"] for row in pool.summary()) == 2
        assert len(pool.summary()) == 2
        for path in (Path(temp) / "multi").rglob("*.json"):
            assert not any(v in path.read_text() for v in keys.values())
        passed.append("independent_api_credentials_and_shared_accounting")

        credentials = Path(temp) / "synthetic-credentials.env"
        credentials.write_text(
            "OFFLINE_KEY=offline-main\n# ordinary comment\n# sk-offline-first\n"
            "OTHER_KEY=offline-other\n# sk-offline-second\n", encoding="utf-8")
        credentials.chmod(0o600)
        with patch.dict("os.environ", {}, clear=True):
            assert resolve_key({"api_key_env": "OFFLINE_KEY", "credentials_file": str(credentials)}) == "offline-main"
            assert resolve_key({"api_key_env": "OFFLINE_COMMENT_KEY", "credentials_file": str(credentials),
                                "credentials_comment_index": 1}) == "sk-offline-second"
        passed.append("comment_credentials_selected_without_copying_keys")

        relaxed = deepcopy(config)
        relaxed["upstream_policy"] = {}
        native = {**body, "temperature": 1, "max_output_tokens": None,
                  "instructions": "unsent synthetic instruction"}
        pool = ApiPool(Path(temp) / "accepted-pool", relaxed)
        client = ResponsesClient(api, pool, Path(temp) / "accepted", "offline-accepted")
        opener = Opener(native)
        with patch("GPTexp.transport.resolve_key", return_value=key), patch(
                "GPTexp.transport.urllib.request.build_opener", return_value=opener):
            assert client.chat(messages, max_tokens=8192, seed=137) == '{"answer":"A"}'
        intent = read(Path(temp) / "accepted/001.intent.json")
        assert intent["native_payload"]["temperature"] == 0
        assert intent["native_payload"]["seed"] == 137
        assert intent["native_payload"]["max_output_tokens"] == 8192
        assert "instructions" not in intent["native_payload"]
        assert read(Path(temp) / "accepted/001.response.json")["native_response"] == native
        assert not pool.paused(api) and opener.attempts == 1
        passed.append("authorized_reported_differences_retained_with_original_sent_parameters")

        image_body = {**body, "output": [{"type": "message", "content": [
            {"type": "output_text", "text": '{"first":"red","second":"blue"}'}]}]}
        with patch("GPTexp.transport.resolve_key", return_value=key), patch(
                "GPTexp.transport.urllib.request.build_opener", return_value=Opener(image_body)):
            validation = validate_api(api, relaxed, Path(temp) / "validation")
        assert validation["validated"] is True and validation["max_requests"] == 1
        passed.append("single_attempt_image_order_validation_before_api_activation")

        directory = Path(temp) / "interrupted-export"
        pair = {"dataset": "blink-multiview", "sample_id": "offline-synthetic",
                "pair_identity": digest("offline-pair"),
                "arms": [{"arm": arm, "problem": {"sample_id": "offline-synthetic"}}
                         for arm in ("direct", "harness")]}
        manifest = {"counts": {"blink-multiview": 93}, "questions": 93, "arms": 186,
                    "pairs": [pair], "synthetic_fixture": True}
        manifest["run_identity"] = digest(manifest)
        write(directory / "manifest.json", manifest)
        root = directory / "episodes" / pair["dataset"] / pair["pair_identity"]
        write(root / "direct/episode.json", {
            "dataset": pair["dataset"], "sample_id": pair["sample_id"], "arm": "direct",
            "pair_identity": pair["pair_identity"], "run_identity": manifest["run_identity"],
            "reference_answer_visible": False, "result": {
                "status": "answered", "answer": "A",
                "execution_trace": [{"program_source": "offline full-trace marker",
                                     "tool_results": [{"synthetic_fixture": True}]}]},
            "request_files": {}, "failure": None})
        write(root / "harness/requests/001.intent.json", {
            "synthetic_fixture": True, "reference_answer_visible": False})
        outcome = export(directory)
        assert outcome == {"episodes": 1, "partial_arms": 1, "host_scored": False}
        exported = read(directory / "results_and_full_trajectories.json")
        assert exported["expected_arms"] == 186
        direct = next(row for row in exported["episodes"] if row["arm"] == "direct")
        assert direct["result"]["execution_trace"][0]["program_source"] == "offline full-trace marker"
        partial = next(row for row in exported["episodes"] if not row["episode_complete"])
        assert "requests/001.intent.json" in partial["full_trajectory"]
        statistics = read(directory / "comparison.json")["statistics"]
        assert statistics[1]["completed_arms"] == 0 and statistics[1]["missing"] == 93
        assert statistics[1]["requests"] == 0 and statistics[1]["request_intents"] == 1
        assert statistics[1]["partial_arms"] == 1
        write(root / "harness/requests/001.admission.json", {"synthetic_fixture": True})
        write(root / "harness/requests/001.response.json", {
            "synthetic_fixture": True, "status": "error", "reason_code": "transport_outcome_unknown"})
        export(directory)
        statistics = read(directory / "comparison.json")["statistics"]
        assert statistics[1]["requests"] == 1 and statistics[1]["received_responses"] == 0
        assert statistics[1]["transport_outcome_unknown"] == 1
        passed.append("interrupted_request_export_and_fixed_denominator")
    print(json.dumps({"model_requests": 0, "checks": passed}))


if __name__ == "__main__":
    main()
