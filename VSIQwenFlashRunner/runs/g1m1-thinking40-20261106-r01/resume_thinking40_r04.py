#!/usr/bin/env python3
"""User-authorized remaining-QA attempt with isolated, explicitly audited transport."""

from collections import Counter
from dataclasses import dataclass
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import sys
import time
import traceback
import urllib.request

HERE = Path(__file__).resolve()
spec = importlib.util.spec_from_file_location("thinking40_recovery_base", HERE.with_name("resume_thinking40.py"))
b = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = b
spec.loader.exec_module(b)
BASE_SHA = "03b4f33343bdddd6893071afc35494a152cc5a6d9ee20436a4c85040a7ee8b1c"
OLD_RECOVERY = b.RECOVERY
b.RECOVERY = b.ROOT / "resume-20261106-r04"
ACTUAL_ENDPOINT = "http://10.10.0.198:27451"
ACTUAL_CLIENT_WAIT = 1260
_original_make_runner = b.make_runner
_original_preserved_files = b.preserved_files
_original_load_controller = b.load_controller


class RouteOpener:
    def __init__(self, delegate):
        self.delegate = delegate

    def __call__(self, request, timeout):
        assert request.full_url == b.ENDPOINT + "/chat"
        # Change only the transport origin. Preserve all frozen RPC bytes and headers.
        routed = urllib.request.Request(
            ACTUAL_ENDPOINT + "/chat", data=request.data,
            headers=dict(request.header_items()), method=request.get_method(),
            origin_req_host=request.origin_req_host, unverifiable=request.unverifiable)
        routed.unredirected_hdrs = dict(request.unredirected_hdrs)
        assert routed.data is request.data
        assert dict(routed.header_items()) == dict(request.header_items())
        return self.delegate(routed, timeout=ACTUAL_CLIENT_WAIT)


def install_route(client):
    seen = set()
    while id(client) not in seen:
        seen.add(id(client))
        if "_rpc_opener" in vars(client):
            before = client.audit_identity()
            if not isinstance(client._rpc_opener, RouteOpener):
                client._rpc_opener = RouteOpener(client._rpc_opener)
            client.transport_retries = 0
            assert client.audit_identity() == before
            return
        next_client = next((vars(client)[key] for key in ("delegate", "client")
                            if key in vars(client)), None)
        assert next_client is not None
        client = next_client
    raise AssertionError("unresolved_transport_delegate")


@dataclass(frozen=True)
class RoutedFactory:
    delegate: object

    def __call__(self, variant_id):
        client = self.delegate(variant_id)
        install_route(client)
        return client


def load_controller(root):
    controller = _original_load_controller(root)
    from skill3d.infra.experiment_stages_v11 import fitset_cli
    factory = fitset_cli()._RevisionClientFactory
    if not getattr(factory, "_thinking40_route_installed", False):
        original_call = factory.__call__

        def routed_call(self):
            client = original_call(self)
            install_route(client)
            return client

        factory.__call__ = routed_call
        factory._thinking40_route_installed = True
    return controller


def make_runner(controller, root, manifest, setup):
    runner = _original_make_runner(controller, root, manifest, setup)
    # Spawned episode processes install transport on their own newly-created client.
    runner.client_factory = RoutedFactory(runner.client_factory)
    return runner


def result_rows():
    return [row for path in (b.ROOT / "system/results").glob("*.json")
            if (row := b.read(path))["status"] == "completed"
            and row["final_state"] == "answer" and not row.get("service_errors")]


def direct_rows():
    return [b.read(path) for root in (
        b.ROOT / "direct", b.DIRECT_ROOT / "direct", OLD_RECOVERY / "direct", b.RECOVERY / "direct")
        for path in (root / "responses").glob("*.json")]


def route_preflight():
    seen = []
    marker = object()

    def fake(request, *, timeout):
        seen.append((request, timeout))
        return marker

    raw = b'{"schema_version":"fake","request_id":"fake","payload":{"seed":137}}'
    request = urllib.request.Request(
        b.ENDPOINT + "/chat", data=raw, method="POST", headers={"Content-Type": "application/json"})
    assert RouteOpener(fake)(request, timeout=180) is marker
    routed, wait = seen[0]
    assert len(seen) == 1 and wait == 1260 and routed.data is raw
    assert routed.full_url == ACTUAL_ENDPOINT + "/chat"
    return {"model_requests": 0, "same_rpc_bytes": True, "same_headers": True,
            "single_transport_attempt": True, "actual_client_timeout_s": 1260}


def health_check(controller, config, max_tokens):
    from skill3d.infra.experiment_stages_v11 import fitset_cli
    original = controller.authenticate(b.ENDPOINT)
    controller.authenticate(ACTUAL_ENDPOINT)
    client = fitset_cli()._RevisionClientFactory(config, max_tokens, b.ENDPOINT)()
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    with opener.open(ACTUAL_ENDPOINT + "/health", timeout=10) as response:
        raw = response.read(65537)
    assert os.environ["HARNESS3D_PERCEPTION_TOKEN"].encode() not in raw
    health = json.loads(raw)
    assert health["identity"] == original["identity"] == client.audit_identity()
    assert health["identity"] == b.read(b.ROOT / "solve_health.json")["identity"]
    assert health["status"] == "ready" and health["max_concurrency"] == 8
    operational = health["transport_recovery"]
    assert operational["actual_upstream_timeout_s"] == 1200
    assert operational["actual_client_timeout_s"] == 1260
    assert operational["automatic_model_retry"] is False
    assert operational["transport_retries"] == 0
    return {"status": "ready", "max_concurrency": 8, "original_identity_verified": True,
            "identity_sha256": hashlib.sha256(
                json.dumps(health["identity"], sort_keys=True).encode()).hexdigest(),
            "transport_recovery": operational, "model_requests": 0}


def preserved_files():
    values = _original_preserved_files()
    # The unfinished canonical result and outcome move together to an explicit archive.
    finished_keys = {path.stem for path in (b.ROOT / "system/results").glob("*.json")
                     if b.read(path)["status"] == "completed"}
    for name in list(values):
        path = Path(name)
        if path.parent == b.ROOT / "system/results" and path.stem not in finished_keys:
            values.pop(name)
        elif path.name == "outcome.json" and path.parent.name not in finished_keys:
            values.pop(name)
    for root in (OLD_RECOVERY,):
        for path in root.glob("**/*.json"):
            if path.name != "status.json":
                values[str(path)] = b.sha(path)
    for path in (b.ROOT / "system/variants").glob("*/plan.json"):
        if path.parent.name in finished_keys:
            values[str(path)] = b.sha(path)
    values[str(HERE.with_name("resume_thinking40.py"))] = BASE_SHA
    values[str(b.ROOT / "system/manifest.json")] = b.sha(b.ROOT / "system/manifest.json")
    return values


def plan():
    assert not b.owners(), "previous_owner_active"
    assert b.sha(HERE.with_name("resume_thinking40.py")) == BASE_SHA
    route_check = route_preflight()
    controller = load_controller(b.ROOT)
    manifest = b.verify_frozen(b.ROOT, b.EXPECTED)
    direct = b.read(b.DIRECT_ROOT / "diagnostic.json")
    from skill3d.evaluation.fitset_v11 import content_sha256
    assert direct["identity_sha256"] == b.EXPECTED_DIRECT
    assert content_sha256({k: v for k, v in direct.items()
                           if k != "identity_sha256"}) == b.EXPECTED_DIRECT
    assert direct["input_files"] == manifest["input_files"]
    assert direct["runtime"] == manifest["runtime"]
    assert b.sha(b.DIRECT_ROOT / "controller.py") == direct["controller_sha256"]
    receipt = b.read(b.DIRECT_ROOT / "code/code_identity.json")
    assert receipt["source_sha256"] == direct["source_sha256"]
    for row in receipt["files"]:
        assert b.sha(b.DIRECT_ROOT / "code/source" / row["path"]) == row["sha256"]
    setup = b.select(b.ROOT, manifest)
    cli, raw, cfg, panel, variants, split, config = setup
    state = b.read(b.ROOT / "requests/state.json")
    assert state["policy"] == {"max_requests": 320, "max_inflight": 8, "requests_per_second": 0}
    assert all(row["status"] in ("completed", "error") for row in state["requests"])
    system_done = {row["variant_id"] for row in result_rows()}
    direct_done = {row["qa_id"] for row in direct_rows()
                   if row["status"] == "completed" and row["final_state"] == "answer"}
    pending_system = sorted(v["variant_id"] for v in variants if v["variant_id"] not in system_done)
    pending_direct = sorted(v["qa_id"] for v in variants if v["qa_id"] not in direct_done)
    assert len(system_done) == 22 and len(direct_done) == 14
    assert len(state["requests"]) == 107
    additional = 7 * len(pending_system) + len(pending_direct)
    assert additional == 152 and len(state["requests"]) + additional <= 320
    failures = [row for row in state["requests"] if row["status"] == "error"]
    system_errors = sorted({row["context"]["variant_id"] for row in failures}
                           & set(pending_system))
    direct_errors = sorted({row["qa_id"] for row in direct_rows()
                            if row["status"] == "service_error"} & set(pending_direct))
    system_count = sum(not row["context"]["variant_id"].startswith("baseline:")
                       for row in state["requests"])
    assert system_count + 7 * len(pending_system) <= 280
    assert shutil.disk_usage(b.ROOT).free / 2**30 >= 80
    health = health_check(controller, config, cfg.max_tokens)
    assert not health["transport_recovery"]["invocation_counts"]
    os.environ["SKILL3D_CODE_RECEIPT"] = str(b.ROOT / "code/code_identity.json")
    os.environ["SKILL3D_RUN_MODE"] = "formal"
    runner = make_runner(controller, b.ROOT, manifest, setup)
    runner.initialize(resume=True)
    value = {
        "schema_version": "thinking40-explicit-recovery-v2",
        "created_at": time.time(), "origin_identity": b.EXPECTED,
        "direct_origin_identity": b.EXPECTED_DIRECT,
        "source_sha256": manifest["source_sha256"], "direct_source_sha256": direct["source_sha256"],
        "recovery_script_sha256": b.sha(HERE), "base_recovery_script_sha256": BASE_SHA,
        "automatic_model_retry": False, "reference_answer_visible": False,
        "same_40_questions": True, "same_scientific_runtime": True,
        "system_manifest_recomputed_and_verified": True,
        "original_start_and_exit_records_preserved": True, "original_errors_preserved": True,
        "system_completed_before": len(system_done), "direct_completed_before": len(direct_done),
        "pending_system_variant_ids": pending_system, "pending_direct_qa_ids": pending_direct,
        "explicitly_authorized_recovery_system_variant_ids": system_errors,
        "explicitly_authorized_recovery_direct_qa_ids": direct_errors,
        "system_requests_before": system_count, "reserved_requests_before": len(state["requests"]),
        "global_request_cap": 320, "global_remaining_requests": 320 - len(state["requests"]),
        "maximum_additional_requests": additional,
        "max_requests_per_system_attempt": 7, "max_requests_per_direct_attempt": 1,
        "direct_original_request_cap": 40,
        "additional_direct_recovery_attempt_cap": len(direct_errors),
        "configured_transport_endpoint_sha256": hashlib.sha256(b.ENDPOINT.encode()).hexdigest(),
        "actual_transport_endpoint_sha256": hashlib.sha256(ACTUAL_ENDPOINT.encode()).hexdigest(),
        "physical_transport_configuration_changed": True,
        "actual_client_timeout_s": 1260, "actual_upstream_timeout_s": 1200,
        "identity_configured_timeout_s": 180, "transport_retries": 0,
        "route_preflight": route_check, "health": health, "preserved_files": preserved_files(),
    }
    b.RECOVERY.mkdir(exist_ok=True)
    b.write(b.RECOVERY / "request_state_before.json", state, exclusive=True)
    b.write(b.RECOVERY / "plan.json", value, exclusive=True)
    print(json.dumps({k: value[k] for k in (
        "system_completed_before", "direct_completed_before", "global_remaining_requests",
        "maximum_additional_requests", "system_manifest_recomputed_and_verified")}), flush=True)


def verify_plan():
    value = b.read(b.RECOVERY / "plan.json")
    assert b.sha(HERE) == value["recovery_script_sha256"]
    for path, expected in value["preserved_files"].items():
        assert b.sha(Path(path)) == expected
    before = b.read(b.RECOVERY / "request_state_before.json")["requests"]
    current = b.read(b.ROOT / "requests/state.json")["requests"]
    assert current[:len(before)] == before and len(current) <= 320
    archive_path = b.RECOVERY / "archive_receipt.json"
    if archive_path.exists():
        for path, expected in b.read(archive_path)["archived_result_and_outcome_hashes"].items():
            assert b.sha(Path(path)) == expected
    return value


def archive_unfinished(value):
    finished_keys = {path.stem for path in (b.ROOT / "system/results").glob("*.json")
                     if b.read(path)["status"] == "completed"}
    moved = []
    hashes = {}

    def move(path, target):
        assert not target.exists()
        target.parent.mkdir(parents=True, exist_ok=True)
        path.rename(target)
        moved.append({"from": str(path), "to": str(target)})

    for path in (b.ROOT / "system/results").glob("*.json"):
        if path.stem not in finished_keys:
            assert b.read(path)["status"] == "incomplete"
            target = b.RECOVERY / "prior_results" / path.name
            hashes[str(target)] = b.sha(path)
            move(path, target)
    for path in (b.ROOT / "system/variants").iterdir():
        if path.name in finished_keys:
            continue
        target = b.RECOVERY / "prior_system_attempts" / path.name
        if (path / "outcome.json").exists():
            hashes[str(target / "outcome.json")] = b.sha(path / "outcome.json")
        move(path, target)
        dispatch = b.ROOT / "system/dispatch" / (path.name + ".json")
        if dispatch.exists():
            move(dispatch, b.RECOVERY / "prior_dispatch" / dispatch.name)
    b.write(b.RECOVERY / "archive_receipt.json", {
        "moved_without_deletion": moved, "archived_result_and_outcome_hashes": hashes,
        "original_request_ledger_and_completed_results_preserved": True,
    }, exclusive=True)


b.load_controller = load_controller
b.make_runner = make_runner
b.result_rows = result_rows
b.direct_rows = direct_rows
b.health_check = health_check
b.plan = plan
b.verify_plan = verify_plan
b.archive_unfinished = archive_unfinished


if __name__ == "__main__":
    try:
        # Base launch/arm use their own __file__ for child dispatch and script pin.
        b.__file__ = str(HERE)
        b.main()
    except BaseException as error:
        print(json.dumps({
            "status": "stopped", "error_type": type(error).__name__,
            "stack": [{"file": Path(row.filename).name, "line": row.lineno, "function": row.name}
                      for row in traceback.extract_tb(error.__traceback__)],
        }), flush=True)
        raise SystemExit(1)
