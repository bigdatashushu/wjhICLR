"""Fresh full pass using verified prior input and API receipts, without new probes."""
from __future__ import annotations

import argparse
from copy import deepcopy
import fcntl
from pathlib import Path
import shutil
import tarfile
import time

from .api_pool import resolve_key
from .launch import reuse_preflight
from .run import ROOT, code_identity, execute, prepare, verify_manifest
from .traces import digest, file_digest, read, write


def reuse_validation(directory, source, config):
    previous = read(source / "manifest.json")
    if digest({k: v for k, v in previous.items() if k != "run_identity"}) != previous["run_identity"]:
        raise ValueError("validation_source_manifest_changed")
    audit = read(source / "api_validation.json")
    rows = {row["api_id"]: row for row in audit["results"]}
    old_apis = {a["id"]: a for a in previous["config"]["apis"]}
    final = deepcopy(config)
    checks = ("base_url", "protocol", "model", "api_key_env", "credentials_comment_index",
              "timeout_seconds", "max_inflight", "requests_per_second")
    hashes = {}
    for api in final["apis"]:
        old = old_apis[api["id"]]
        if any(api.get(k) != old.get(k) for k in checks):
            raise ValueError("validation_api_binding_changed")
        original = source / "api-probes" / api["id"]
        row = rows[api["id"]]
        if (read(original / "summary.json") != row or row["validated"] is not True
                or row["image_order_verified"] is not True
                or not old["enabled"] or not old["validated"]):
            raise ValueError("validation_receipt_not_successful")
        intent = read(original / "requests/001.intent.json")
        response = read(original / "requests/001.response.json")
        ledger = read(original / "pool/state.json")["requests"]
        if (intent["api_id"] != api["id"] or intent["ordinal"] != 1
                or intent["payload_sha256"] != digest(intent["native_payload"])
                or intent["native_payload"]["model"] != config["model"]
                or intent["native_payload"]["reasoning"] != {"effort": "high"}
                or response["status"] != "received"
                or response["native_response"]["model"] != config["model"]
                or response["native_response"]["reasoning"]["effort"] != "high"
                or len(ledger) != 1 or ledger[0]["status"] != "completed"
                or ledger[0]["payload_sha256"] != intent["payload_sha256"]):
            raise ValueError("validation_native_response_changed")
        for path in original.rglob("*.json"):
            relative = path.relative_to(source)
            expected = file_digest(path)
            target = directory / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(path, target)
            if file_digest(target) != expected:
                raise ValueError("validation_receipt_copy_changed")
            hashes[str(relative)] = expected
        api.update(validated=True, enabled=True,
                   validation_receipt=f"api-probes/{api['id']}/summary.json")
    final.update(max_inflight=len(final["apis"]), max_workers=2 * len(final["apis"]))
    write(directory / "api_validation.json", {
        **audit, "reused_from": str(source), "source_run_identity": previous["run_identity"],
        "new_engineering_requests": 0, "receipt_sha256": hashes})
    return final, previous


def launch(directory, config, preflight_source, validation_source):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / "launch.lock").open("a+b") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if any((directory / name).exists() for name in ("bootstrap.json", "manifest.json", "startup.json")):
            raise ValueError("rerun_already_launched")
        keys = [resolve_key(api) for api in config["apis"]]
        if len(keys) != len(set(keys)):
            raise ValueError("duplicate_api_credentials")
        del keys
        status = {"schema": "gptexp-rerun-bootstrap/1", "at": time.time(),
                  "stage": "reusing_verified_receipts", "automatic_retry": False,
                  "reference_answers_read": False}
        write(directory / "bootstrap.json", status, exclusive=True)
        write(directory / "launch.config.json", config, exclusive=True)
        snapshot = directory / "public_source_snapshot.tar.gz"
        with tarfile.open(snapshot, "w:gz") as archive:
            for name in code_identity():
                archive.add(ROOT / name, arcname=name)
            archive.add(directory / "launch.config.json", arcname="config.json")
        status["public_source_snapshot_sha256"] = file_digest(snapshot)
        original_preflight = reuse_preflight(directory, Path(preflight_source), config["datasets"])
        final, previous = reuse_validation(directory, Path(validation_source), config)
        write(directory / "run.config.json", final)
        prepare(directory, final)
        manifest = read(directory / "manifest.json")
        verify_manifest(manifest)
        if (manifest["pairs"] != previous["pairs"]
                or manifest["counts"] != previous["counts"]
                or manifest["max_requests"] != previous["max_requests"]
                or manifest["preflight"] != {
                    name: original_preflight["preflight"][name] for name in config["datasets"]}):
            raise ValueError("fresh_pass_frozen_input_changed")
        write(directory / "readiness_audit.json", {
            "run_identity": manifest["run_identity"], "questions": manifest["questions"],
            "arms": manifest["arms"], "max_requests": manifest["max_requests"],
            "datasets": manifest["counts"], "excluded": ["blink-sr"],
            "pair_order_and_specs_identical_to_previous": True,
            "cached_preflight_unchanged": True, "validated_apis": len(final["apis"]),
            "preparation_model_requests": 0, "new_api_probes": 0,
            "source_and_vendor_verified": True, "reference_answers_read": False,
            "adapter_source": code_identity(), "service_failure_policy": final["service_failure_policy"]})
        status.update(stage="executing", run_identity=manifest["run_identity"], at=time.time(),
                      validated_apis=len(final["apis"]))
        write(directory / "bootstrap.json", status)
        print(f"Fresh full pass: {manifest['questions']} pairs, {len(final['apis'])} verified APIs.", flush=True)
        execute(directory)
        status.update(stage=read(directory / "execution.json")["status"], at=time.time())
        write(directory / "bootstrap.json", status)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", required=True)
    parser.add_argument("--config", default=str(ROOT / "config.json"))
    parser.add_argument("--preflight-source", required=True)
    parser.add_argument("--validation-source", required=True)
    args = parser.parse_args()
    try:
        launch(args.run, read(args.config), args.preflight_source, args.validation_source)
    except Exception as error:
        write(Path(args.run) / "bootstrap_failure.json", {
            "error_type": type(error).__name__, "at": time.time(), "automatic_retry": False})
        print("Fresh pass stopped:", type(error).__name__, flush=True)
        raise SystemExit(1) from None


if __name__ == "__main__":
    main()
