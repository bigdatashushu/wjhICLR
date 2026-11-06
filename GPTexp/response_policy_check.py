"""Offline regression checks for correlation, truthful receipts and layered reuse."""
from __future__ import annotations

from contextlib import contextmanager
from copy import deepcopy
import json
import io
from pathlib import Path
import tempfile
from types import SimpleNamespace
import urllib.error
from unittest.mock import patch

from .api_pool import ApiPool
from .continuation import prepare
from .model_binding import bound_spec, mind_without_metadata_gate
from .partial_score import freeze
from .response_policy import CORRELATION, continuation_config
from .run import code_identity
from .selfcheck import Opener
from .subset import arm_cap, missing_pairs, select_pairs, verify_reuse
from .subset_check import fixture, rejects, seal
from .subset_export import records
from .terra_ops import ADAPTER_HASHES, terra_config
from .traces import digest, file_digest, read, write
from .transport import ResponsesClient


def transport_checks(base):
    config = continuation_config(terra_config(), 1)
    # Even stale require-match flags cannot enable metadata blocking in this mode.
    config["upstream_policy"] = {k: True for k in config["upstream_policy"]}
    compact = {"id": "provider-response", "status": "completed", "output": [
        {"type": "message", "content": [{"type": "output_text", "text": '{"answer":"A"}'}]}]}
    mismatch = {**compact, "model": "reported-other-model", "reasoning": {"effort": "low"},
                "temperature": 1, "seed": 99, "max_output_tokens": 1,
                "instructions": "reported-other-instructions"}
    for i, body in enumerate((compact, mismatch)):
        directory = base / str(i)
        api = config["apis"][0]
        pool = ApiPool(directory / "pool", config)
        client = ResponsesClient(api, pool, directory / "requests", "operation-" + str(i))
        opener = Opener(body)
        with patch("GPTexp.transport.resolve_key", return_value="offline-sentinel"), patch(
                "GPTexp.transport.urllib.request.build_opener", return_value=opener):
            answer = client.chat([{"role": "system", "content": "original instructions"},
                                  {"role": "user", "content": "original question"}], seed=137)
        assert answer == '{"answer":"A"}' and opener.attempts == 1
        intent = read(directory / "requests/001.intent.json")
        receipt = read(directory / "requests/001.response.json")
        admission = read(directory / "requests/001.admission.json")
        assert intent["native_payload"]["model"] == "gpt-5.6-terra"
        assert intent["native_payload"]["reasoning"] == {"effort": "high"}
        assert intent["native_payload"]["instructions"] == "original instructions"
        assert receipt["native_response"] == body
        assert receipt["response_identity"] == {
            "model": body.get("model"), "model_reported": bool(body.get("model"))}
        association = receipt["request_association"]
        assert association["operation"] == client.operation
        assert association["payload_sha256"] == digest(intent["native_payload"])
        assert association["admission_request_id"] == admission["request_id"]
        assert association["ordinal"] == 1
        assert read(directory / "pool/state.json")["requests"][0]["status"] == "completed"
    for i, body in enumerate(({"status": "completed", "output": []},
                              {"error": {"code": "service_error"}}, [])):
        directory = base / ("malformed-" + str(i))
        api = config["apis"][0]
        client = ResponsesClient(api, ApiPool(directory / "pool", config),
                                 directory / "requests", "malformed")
        with patch("GPTexp.transport.resolve_key", return_value="offline-sentinel"), patch(
                "GPTexp.transport.urllib.request.build_opener", return_value=Opener(body)):
            try:
                client.chat([{"role": "user", "content": "question"}])
            except Exception:
                pass
            else:
                raise AssertionError("malformed response accepted")
        assert read(directory / "requests/001.response.json")["status"] == "error"
    directory = base / "http"
    client = ResponsesClient(config["apis"][0], ApiPool(directory / "pool", config),
                             directory / "requests", "http")
    with patch("GPTexp.transport.resolve_key", return_value="offline-sentinel"), patch(
            "GPTexp.transport.urllib.request.build_opener") as opener:
        opener.return_value.open.side_effect = urllib.error.HTTPError(
            "", 524, "", {}, io.BytesIO(b'{"error":{"code":"timeout"}}'))
        try:
            client.chat([{"role": "user", "content": "question"}])
        except Exception:
            pass
        else:
            raise AssertionError("HTTP error accepted")
    assert read(directory / "requests/001.response.json")["http_status"] == 524


def mind_checks(base):
    import sys
    root = Path(__file__).resolve().parent.parent
    with patch.object(sys, "path", [str(root / "src"), str(root / "scripts"), *sys.path]):
        from MindcubeRunner import worker

    relaxed = mind_without_metadata_gate(worker.solve)
    config = SimpleNamespace(model="gpt-5.6-terra", request_limit=lambda arm: 1,
                             kwargs=lambda: {}, public=lambda: {})
    receipt = {"phase": "final"}
    client = SimpleNamespace(chat=lambda *a, **k: '{"answer":"A"}', last_usage={},
                             last_response_identity={"model": None, "model_reported": False})

    class Budget:
        @contextmanager
        def request(self):
            yield {"usage": {}}

    job = {"config": {}, "item": {"sample_id": "sample", "task_type": "Depth", "labels": ["A", "B"]},
           "arm": "direct", "output_directory": str(base), "run_identity": "run",
           "initial_receipt": receipt, "stop_file": str(base / "stop.json")}
    with patch.object(worker, "Config", return_value=config), patch.object(
            worker, "load", return_value=SimpleNamespace(provenance={})), patch.object(
            worker, "pair_identity", return_value="pair"), patch.object(
            worker, "messages", return_value=([{"role": "user", "content": "question"}], receipt)):
        result = relaxed(job, client, Budget())
    assert result["status"] == "answered" and result["answer"] == "A"
    assert result["rounds"][0]["response_identity"] == client.last_response_identity


def continuation_checks(base):
    config = terra_config()
    pairs = []
    for i in range(10):
        public = {"sample_id": str(i), "task_type": "Multi-view_Reasoning"}
        pairs.append({"pair_identity": digest(i), "dataset": "blink-multiview", "sample_id": str(i),
                      "arms": [{"arm": a, "problem": public} for a in ("direct", "harness")]})
    source, first, second, third = (base / n for n in ("original", "first", "second", "third"))
    previous = seal({"pairs": pairs, "counts": {"blink-multiview": 10}, "arms": 20,
                     "config": config})
    write(source / "manifest.json", previous)
    write(source / "pool/state.json", {"policy_identity": digest(config), "requests": []})
    (source / "pool/state.lock").touch()
    fixture(source, previous, pairs[0], "direct", failure={"error_type": "APIError"})
    first.mkdir()
    freeze(source, first / "reuse_source")
    snapshot = read(first / "reuse_source/snapshot.json")
    selected, strata = select_pairs(previous, {(pairs[0]["pair_identity"], "direct")})
    budget = sum(arm_cap(p["dataset"], s["arm"]) for p in selected for s in p["arms"]) - 1
    manifest = seal({
        "schema": "gptexp-type30-paired-reuse/1", "pairs": selected, "strata": strata,
        "counts": {"blink-multiview": 3}, "questions": 3, "arms": 6,
        "config": {**config, "max_requests": budget}, "max_requests": budget,
        "reused_arms": 1, "missing_arms": 5, "source_run_identity": previous["run_identity"],
        "reuse_snapshot_identity": snapshot["snapshot_identity"],
        "reused_api_pins": {pairs[0]["pair_identity"]: "xcode-1"},
        "validation_receipt_sha256": {}, "selection_policy": "fixed fixture",
        "source_receipts": {}, "preflight": {}, "adapter_source": code_identity()})
    write(first / "manifest.json", manifest)
    write(first / "pool/state.json", {"policy_identity": digest(manifest["config"]), "requests": []})
    (first / "pool/state.lock").touch()
    fixture(first, manifest, selected[0], "harness", answer="invalid")
    for origin, destination in ((first, second), (second, third)):
        original_manifest = read(origin / "manifest.json")
        write(origin / "stopped_by_user.json", {"completed_episode_sha256": {
            str(p.relative_to(origin)): file_digest(p) for p in (origin / "episodes").rglob("episode.json")}})
        def validation(directory, source, config):
            write(directory / "api_validation.json", {"receipt_sha256": {}})
            return deepcopy(config), original_manifest

        with patch("GPTexp.continuation.reuse_validation", side_effect=validation):
            audit = prepare(destination, origin)
        current = read(destination / "manifest.json")
        reused = verify_reuse(destination, current)
        jobs, unresolved, _ = missing_pairs(destination, current, reused)
        assert current["pairs"] == selected and current["strata"] == strata
        assert audit["model_requests"] == 0 and not unresolved
        assert len(reused) == (2 if destination == second else 3)
        assert sum(len(specs) for _, specs, _ in jobs) == current["missing_arms"]
        exported = records(destination, current)
        assert any(e[3]["failure"] for e in exported)
        assert any((e[3]["result"] or {}).get("answer") == "invalid" for e in exported)
        for _, _, root, episode, _, provenance in exported:
            assert provenance["source_run_identity"] == episode["run_identity"]
        if destination == second:
            write(destination / "pool/state.json", {"policy_identity": digest(current["config"]), "requests": []})
            (destination / "pool/state.lock").touch()
            pair, specs, pin = jobs[0]
            assert pin == "xcode-1" or pin is None
            # The fixture also writes the current policy's binding receipt.
            spec = specs[0]
            output = fixture(destination, current, pair, spec["arm"])
            from .model_binding import binding_record
            _, binding = binding_record(spec, config["model"], current["run_identity"], CORRELATION)
            write(output / "model_binding.json", binding)
    entry = next(iter(verify_reuse(third, read(third / "manifest.json")).values()))
    path = third / "reuse_source" / entry["episode"]
    path.write_bytes(path.read_bytes() + b" ")
    rejects(lambda: verify_reuse(third, read(third / "manifest.json")))


def main():
    root = Path(__file__).parent
    adapter_hashes = {n: file_digest(root / n) for n in ADAPTER_HASHES}
    offline_config = read(root / "config.example.json")
    offline_config["apis"] = [offline_config["apis"][0]]
    offline_config["apis"][0].update(base_url="https://offline.invalid/v1",
                                     api_key_env="OFFLINE_TEST_API_KEY",
                                     credentials_file="", enabled=True, validated=True)
    spec = {"solver": {"expected_model": "gpt-6.1-sol", "max_requests": 4, "seed": 137}}
    bound, fields = bound_spec(spec, "gpt-5.6-terra", CORRELATION)
    assert fields == ["solver.expected_model"] and bound["solver"]["expected_model"] == ""
    assert bound["solver"]["max_requests"] == 4 and bound["solver"]["seed"] == 137
    with tempfile.TemporaryDirectory(prefix="gptexp-response-policy-") as temporary, patch(
            "GPTexp.terra_ops.read", return_value=offline_config):
        base = Path(temporary)
        transport_checks(base / "transport")
        mind_checks(base / "mind")
        continuation_checks(base / "reuse")
    assert {n: file_digest(root / n) for n in ADAPTER_HASHES} == adapter_hashes
    print(json.dumps({"status": "passed", "live_model_requests": 0,
                      "checks": ["compact_and_mismatched_metadata_accepted",
                                 "sent_model_high_and_original_prompt_preserved",
                                 "request_association_and_truthful_missing_metadata",
                                 "malformed_and_http_errors_rejected_without_retry",
                                 "mind_direct_passes_with_unreported_model",
                                 "three_generations_preserve_all_failures_and_invalid_answers",
                                 "fixed_selection_missing_only_api_pins_and_tamper_rejection",
                                 "adapter_bytes_unchanged_during_check"]}))


if __name__ == "__main__":
    main()
