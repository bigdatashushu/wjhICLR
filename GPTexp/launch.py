"""Validate independent APIs once, reuse frozen inputs, and own the full run."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from copy import deepcopy
import fcntl
import json
from pathlib import Path
import re
import shutil
import tarfile
import time

from .api_pool import ApiPool, resolve_key
from .probe import png
from .run import ROOT, code_identity, execute, prepare, verify_manifest
from .traces import digest, file_digest, read, write
from .transport import ResponsesClient

PREVIOUS_RUN = "gpt61-high-full2098-20261105-r01"


def reuse_preflight(directory, source, datasets):
    previous = read(source / "manifest.json")
    if digest({k: v for k, v in previous.items() if k != "run_identity"}) != previous["run_identity"]:
        raise ValueError("previous_manifest_changed")
    audit = read(source / "readiness_audit.json")
    if audit["run_identity"] != previous["run_identity"] or audit["formal_model_requests"] != 0:
        raise ValueError("previous_preparation_identity_changed")
    for name in ("adapters.py", "worker.py"):
        if previous["adapter_source"][name] != file_digest(ROOT / name):
            raise ValueError("cached_preflight_adapter_changed")
    target = directory / "preflight"
    target.mkdir(exist_ok=True)
    files = {}
    for dataset in datasets:
        name = dataset + ".json"
        original = source / "preflight" / name
        expected = audit["preflight_file_sha256"][name]
        if file_digest(original) != expected:
            raise ValueError("cached_preflight_receipt_changed")
        receipt = read(original)
        if receipt["identity"] != previous["preflight"][dataset]:
            raise ValueError("cached_dataset_identity_changed")
        if (target / name).exists():
            raise ValueError("preflight_target_already_exists")
        shutil.copyfile(original, target / name)
        if file_digest(target / name) != expected:
            raise ValueError("preflight_copy_changed")
        files[name] = expected
    write(directory / "preflight_reuse.json", {
        "source_run_identity": previous["run_identity"], "source_run": str(source),
        "preflight_file_sha256": files, "datasets": datasets,
        "model_requests": 0, "reference_answers_read": False})
    return previous


def validate_api(api, config, directory):
    isolated = deepcopy(config)
    isolated.update(max_requests=1, max_inflight=1, max_workers=2, apis=[deepcopy(api)])
    root = directory / "api-probes" / api["id"]
    pool = ApiPool(root / "pool", isolated)
    client = ResponsesClient(api, pool, root / "requests", "api-validation-" + api["id"])
    messages = [{"role": "user", "content": [
        {"type": "text", "text": "Name the dominant color of image 1, then image 2. Return JSON with keys first and second."},
        png(240, 10, 10), png(10, 10, 240)]}]
    started = time.monotonic()
    row = {"api_id": api["id"], "validated": False, "benchmark_evidence": False,
           "automatic_retry": False, "max_requests": 1, "max_inflight": 1}
    try:
        text = client.chat(messages, max_tokens=8192, seed=config["seed"],
                           temperature=config["temperature"], enable_thinking=True)
        text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip(), flags=re.IGNORECASE)
        colors = json.loads(text)
        if (str(colors.get("first", "")).strip().lower() != "red"
                or str(colors.get("second", "")).strip().lower() != "blue"):
            raise ValueError("image_order_validation_failed")
        row.update(validated=True, status="received", image_order_verified=True,
                   response_identity=client.last_response_identity, usage=client.last_usage)
    except Exception as error:
        row.update(status="error", error_type=type(error).__name__,
                   reason_code=client.last_service_error.get("reason_code", "validation_failed"))
    response_path = root / "requests/001.response.json"
    if response_path.exists():
        native = read(response_path).get("native_response") or {}
        row["reported_parameters"] = {
            "temperature": native.get("temperature"), "seed": native.get("seed"),
            "seed_reported": "seed" in native, "max_output_tokens": native.get("max_output_tokens"),
            "reasoning": native.get("reasoning"),
            "instructions_length": len(native.get("instructions") or ""),
            "instructions_sha256": digest(native.get("instructions") or "")}
    row.update(elapsed_seconds=time.monotonic() - started, accounting=pool.summary())
    write(root / "summary.json", row)
    return row


def launch(directory, config, source):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / "launch.lock").open("a+b") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if any((directory / name).exists() for name in ("bootstrap.json", "manifest.json", "startup.json")):
            raise ValueError("run_already_launched")
        apis = [api for api in config["apis"] if api["enabled"]]
        keys = [resolve_key(api) for api in apis]
        if len(keys) != len(set(keys)):
            raise ValueError("duplicate_api_credentials")
        del keys
        status = {"schema": "gptexp-bootstrap/1", "stage": "reusing_frozen_preflight",
                  "at": time.time(), "apis": len(apis), "probe_results": [],
                  "reference_answers_read": False, "automatic_retry": False}
        write(directory / "bootstrap.json", status, exclusive=True)
        write(directory / "launch.config.json", config)
        snapshot = directory / "public_source_snapshot.tar.gz"
        with tarfile.open(snapshot, "w:gz") as archive:
            for name in code_identity():
                archive.add(ROOT / name, arcname=name)
            archive.add(directory / "launch.config.json", arcname="config.json")
        status["public_source_snapshot_sha256"] = file_digest(snapshot)
        previous = reuse_preflight(directory, Path(source), config["datasets"])
        status["stage"] = "validating_apis"
        write(directory / "bootstrap.json", status)
        print(f"Validating {len(apis)} APIs with one image request per API.", flush=True)
        rows = []
        with ThreadPoolExecutor(max_workers=len(apis)) as executor:
            jobs = {executor.submit(validate_api, api, config, directory): api["id"] for api in apis}
            for future in as_completed(jobs):
                row = future.result()
                rows.append(row)
                status.update(probe_results=sorted(rows, key=lambda r: r["api_id"]), at=time.time())
                write(directory / "bootstrap.json", status)
                print(f"API {row['api_id']}: {row['status']}, validated={row['validated']}", flush=True)
        validation = {row["api_id"]: row for row in rows}
        final_config = deepcopy(config)
        for api in final_config["apis"]:
            api["validated"] = bool(validation.get(api["id"], {}).get("validated"))
            api["enabled"] = api["enabled"] and api["validated"]
            api["validation_receipt"] = f"api-probes/{api['id']}/summary.json"
        ready = sum(api["enabled"] for api in final_config["apis"])
        final_config["max_inflight"] = max(1, ready)
        final_config["max_workers"] = max(2, ready * 2)
        write(directory / "api_validation.json", {
            "results": sorted(rows, key=lambda r: r["api_id"]), "validated_apis": ready,
            "engineering_request_limit": len(apis), "benchmark_evidence": False,
            "automatic_retry": False, "upstream_policy": final_config["upstream_policy"]})
        write(directory / "run.config.json", final_config)
        prepare(directory, final_config)
        manifest = read(directory / "manifest.json")
        verify_manifest(manifest)
        if manifest["preflight"] != {name: previous["preflight"][name] for name in config["datasets"]}:
            raise ValueError("frozen_question_identity_changed")
        write(directory / "readiness_audit.json", {
            "run_identity": manifest["run_identity"], "questions": manifest["questions"],
            "arms": manifest["arms"], "max_requests": manifest["max_requests"],
            "datasets": manifest["counts"], "excluded": ["vsi-bench", "omni3d-bench", "blink-sr"],
            "cached_preflight_unchanged": True, "adapter_source": code_identity(),
            "source_and_vendor_verified": True, "validated_apis": ready,
            "preparation_model_requests": 0, "reference_answers_read": False,
            "upstream_policy": final_config["upstream_policy"]})
        status.update(stage="executing" if ready else "blocked_no_validated_api",
                      run_identity=manifest["run_identity"], validated_apis=ready, at=time.time())
        write(directory / "bootstrap.json", status)
        if not ready:
            print("No API passed; retained all validation evidence, no benchmark requests.", flush=True)
            return
        execute(directory)
        status.update(stage=read(directory / "execution.json")["status"], at=time.time())
        write(directory / "bootstrap.json", status)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", required=True)
    parser.add_argument("--config", default=str(ROOT / "config.json"))
    parser.add_argument("--preflight-source", default=str(ROOT / "runs" / PREVIOUS_RUN))
    args = parser.parse_args()
    try:
        launch(Path(args.run), read(args.config), Path(args.preflight_source))
    except Exception as error:
        write(Path(args.run) / "bootstrap_failure.json", {
            "error_type": type(error).__name__, "automatic_retry": False, "at": time.time()})
        print("Launch stopped:", type(error).__name__, flush=True)
        raise SystemExit(1) from None


if __name__ == "__main__":
    main()
