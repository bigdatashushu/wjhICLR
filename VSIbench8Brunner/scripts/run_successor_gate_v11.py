"""Independent native dual-seed Gate, with label-free execution and no publication."""
from __future__ import annotations

import argparse
import copy
from dataclasses import asdict
import hashlib
import importlib.util
import json
import multiprocessing
from pathlib import Path
import sys
import time
from types import MethodType, SimpleNamespace

# This checkout supplies the Gate wrapper, admission policy and scoped guard.
_script_dir = Path(__file__).resolve().parent
sys.path[:] = [p for p in sys.path if Path(p).resolve() != _script_dir]
# The uniquely named helper is importable by spawn, behind frozen native modules.
sys.path.append(str(_script_dir))
from successor_gate_execution_v11 import (
    CheckedClient, CheckedFactory, GateAbort, GateClientError, IsolatedRunner,
)
EXECUTION_HELPER_PATH = _script_dir / "successor_gate_execution_v11.py"

ADMISSION_POLICY_PATH = (
    Path(__file__).resolve().parents[1] / "src/skill3d/evolution/gate_admission_v11.py")


class PreparationComplete(BaseException):
    pass


def identity(path):
    data = Path(path).read_bytes()
    return {"size": len(data), "sha256": hashlib.sha256(data).hexdigest()}


def read(path):
    return json.loads(Path(path).read_text())


def campaign_root(args, selection):
    # Each source identity owns new arm libraries even when a zero-request
    # preparation previously stopped in the same admitted workspace.
    return args.output / "campaigns" / identity(Path(__file__))["sha256"][:20] / selection["campaign_id"]


def seal(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    with path.open("x") as stream:
        path.chmod(0o600)
        json.dump(value, stream, ensure_ascii=False, sort_keys=True)
        stream.write("\n")


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def bind_gate_admission(worker):
    policy = load("successor_gate_admission", ADMISSION_POLICY_PATH)
    native_gate = worker._gate

    def evaluate(_worker, *args, **kwargs):
        return policy.gate_admission_decision(native_gate(*args, **kwargs))

    worker._gate = MethodType(evaluate, worker)


def setup(args):
    from skill3d.evaluation.fitset_v11 import SystemVariantRunner, content_sha256
    from skill3d.evolution.fitset_stream_v11 import FitSetWorkerEvolution
    from skill3d.evolution.fitset_revision_v11 import _static_validate as native_static_validate
    from skill3d.schemas import SkillSpecV11
    from skill3d.storage.framework_state_io import FrameworkStateIO
    from skill3d.storage.hot_cache import HotCache
    from skill3d.storage.state_workspace import StateWorkspace
    from skill3d.synthesis.request_governor import RequestGovernorConfig
    import freeze_runner_binding_ops

    guard = load("successor_gate_scoped_guard",
                 args.revision_root.parent / "source/src/skill3d/evolution/revision_guard_v11.py")
    _static_validate = guard.scoped_static_validator(native_static_validate)
    revision = read(args.revision_root / "revision_final.json")
    ready = read(args.revision_root / "preflight.json")
    if not revision["protected_inputs_unchanged"] or revision["status"] != "revision_complete":
        raise GateAbort("revision_not_complete_or_assets_changed")
    native_helper = load("successor_gate_native_helper",
                        args.frozen_root / "ops/posthoc_reviser_rescue_v5_ops.py")
    tools = load("successor_gate_revision_tools",
                 args.revision_root.parent / "source/scripts/run_reviser_successor_v11.py")
    native, state, runtime, manifest, io = freeze_runner_binding_ops.configured_runner()
    if state != args.state_root or io._future is not None:
        raise GateAbort("gate_native_input_state")
    config = native_helper.configuration(state)
    candidates = []
    for row in revision["results"]:
        if row["status"] != "static_valid":
            continue
        selection = next(item for item in ready["selected"]
                         if item["campaign_id"] == row["source_campaign"])
        parent = SkillSpecV11.model_validate(read(selection["input_refs"]["request"])["parent"])
        if not row["all_evidence_delivered"] or not row.get("all_batches_resolved", True):
            raise GateAbort("gate_candidate_evidence_unresolved")
        path = args.revision_root / row.get("candidate_ref", row["source_kind"] + "/SKILL.md")
        candidate, errors = _static_validate(
            parent=parent, candidate_text=path.read_text(),
            forbidden_sample_ids={str(v) for item in config["panel"]["system_variants"]
                                  for v in (item["qa_id"], item["scene_id"], item["variant_id"])},
            method_context_max_chars=config["base"].retrieval_policy.method_context_max_chars)
        if errors or candidate is None or candidate.content_sha256 != row["candidate_sha256"]:
            raise GateAbort("gate_candidate_static_or_content_identity")
        candidates.append((selection, parent, candidate))
    if not candidates:
        raise GateAbort("gate_has_no_static_candidate")

    descriptor = read(args.output / "framework_io_binding.json")
    cache = HotCache(Path(descriptor["cache_root"]),
                     quota_bytes=descriptor["quota_bytes"],
                     min_free_bytes=descriptor["min_free_bytes"])
    workspace = StateWorkspace(cache, descriptor["key"], bounded=True)
    gate_io = FrameworkStateIO(
        workspace, job="reviser-successor-gate-20261029",
        max_workspace_bytes=descriptor["workspace_bytes"])
    if gate_io.root != args.output or cache.min_free_bytes != io._workspace.cache.min_free_bytes:
        raise GateAbort("gate_workspace_identity_or_storage_floor")

    coordinator = IsolatedRunner.__new__(IsolatedRunner)
    coordinator.__dict__ = copy.copy(vars(native))
    coordinator.root = args.output
    coordinator._output_scope = args.output
    coordinator.framework_state_io = gate_io
    coordinator.deployment_continuation = coordinator._continuation = None
    coordinator.cached_prefix_recovery = coordinator._cached_prefix_spec = None
    coordinator._request_governor_root = args.output / "request_governor"
    request_cap = min(ready["gate_request_cap"],
                      getattr(args, "request_cap", None) or ready["gate_request_cap"])
    coordinator.request_governor_config = RequestGovernorConfig(
        max_requests=request_cap, max_inflight=getattr(args, "request_concurrency", 4),
        requests_per_second=0.0)
    workers = getattr(args, "workers", 4)
    coordinator._episode_semaphore = multiprocessing.get_context("spawn").BoundedSemaphore(workers)
    coordinator.workers = coordinator.episode_concurrency = workers
    coordinator.run_identity = content_sha256({
        "schema": "harness3d-independent-successor-gate/1",
        "original_run": native.run_identity,
        "revision_final": identity(args.revision_root / "revision_final.json"),
        "gate_source": identity(Path(__file__)),
        "execution_helper": identity(EXECUTION_HELPER_PATH),
        "workers": workers, "request_concurrency": coordinator.request_governor_config.max_inflight,
    })
    coordinator.gate_client_factory = CheckedFactory(
        config["settings"], config["transport"]["endpoint"], ready["model_identity"]["model"], gate_io)
    worker = FitSetWorkerEvolution.__new__(FitSetWorkerEvolution)
    worker.runner, worker.config = coordinator, native.evolution_config
    worker.rows = native.gate_variant_rows
    worker.root = args.output
    if worker.config.gate_seeds != (137, 138) or worker.config.gate_mode != "qa_disjoint":
        raise GateAbort("gate_protocol_identity")
    bind_gate_admission(worker)
    return tools, coordinator, worker, candidates, ready


def preflight(args):
    from skill3d.evaluation.fitset_v11 import content_sha256
    args.output.mkdir(parents=True, exist_ok=True, mode=0o700)
    tools, runner, worker, candidates, revision_ready = setup(args)
    if tools.protected(args.state_root, args.frozen_root) != revision_ready["protected_inputs"]:
        raise GateAbort("gate_original_assets_changed")
    plans, outputs = {}, []

    def collect(coordinator, episodes):
        if len(plans) + len(episodes) > runner.request_governor_config.max_requests:
            raise GateAbort("gate_minimum_request_budget")
        pairs = {}
        for arm, variant in episodes:
            with arm._input_episode_lease(variant):
                prepared = arm._prepare_one(variant, preflight=True)
            if (prepared["item"].episode.ground_truth or arm.label_metadata
                    or arm.capture_offline_case_input):
                raise GateAbort("gate_validation_labels_visible")
            shared = {key: value for key, value in prepared["plan"].items() if key != "skill_identity"}
            hashed = content_sha256(shared)
            relative = str(arm.root.relative_to(args.output))
            key = relative + "/" + str(variant["variant_id"])
            plans[key] = hashed
            pair = (relative.rsplit("/", 1)[0], str(variant["variant_id"]))
            if pair in pairs and pairs[pair] != hashed:
                raise GateAbort("gate_pair_input_mismatch")
            pairs[pair] = hashed
        raise PreparationComplete()

    runner._run_gate_episodes = MethodType(collect, runner)
    for selection, parent, candidate in candidates:
        request = SimpleNamespace(slot_key=parent.slot_key,
                                  question_type=parent.question_type, evidence_state=parent.evidence_state)
        root = campaign_root(args, selection)
        try:
            result = worker._gate(request, parent, candidate, root)
        except PreparationComplete:
            outputs.append({"campaign_id": selection["campaign_id"],
                            "slot_key": parent.slot_key, "status": "ready",
                            "parent_sha256": parent.content_sha256,
                            "candidate_sha256": candidate.content_sha256})
        else:
            outputs.append({"campaign_id": selection["campaign_id"], "status": "panel_ineligible",
                            "decision": result})
    if tools.protected(args.state_root, args.frozen_root) != revision_ready["protected_inputs"]:
        raise GateAbort("gate_original_assets_changed")
    value = {
        "status": "ready", "at": time.time(), "model_requests": 0, "candidates": outputs,
        "prepared_input_identities": plans, "planned_episodes": len(plans),
        "gate_source_identity": identity(Path(__file__)),
        "execution_helper_source_identity": identity(EXECUTION_HELPER_PATH),
        "admission_policy_source_identity": identity(ADMISSION_POLICY_PATH),
        "native_sources": revision_ready["source_identity"],
        "revision_final_identity": identity(args.revision_root / "revision_final.json"),
        "gate_config": asdict(worker.config), "seeds": [137, 138],
        "model_identity": revision_ready["model_identity"],
        "solver_config": {key: val for key, val in asdict(runner.base_cfg).items()
                          if key in {"seed", "temperature", "max_tokens", "max_rounds"}},
        "request_cap": runner.request_governor_config.max_requests, "per_episode_request_cap": 64,
        "max_inflight": getattr(args, "request_concurrency", 4),
        "workers": getattr(args, "workers", 4), "transport_retries": 0,
        "executor": "frozen_native_SystemVariantRunner_without_original_receipt_writer",
        "validation_labels_hidden_during_execution": True, "published": False,
    }
    value["framework_workspace_binding"] = read(args.output / "framework_io_binding.json")
    runner.framework_state_io.write_json_once(args.output / "preflight.json", value)
    return {key: value[key] for key in ("status", "model_requests", "candidates", "planned_episodes")}


def run(args):
    from skill3d.synthesis.request_governor import RequestGovernor
    ready = read(args.output / "preflight.json")
    if ready["gate_source_identity"] != identity(Path(__file__)):
        raise GateAbort("gate_source_changed")
    if ready["admission_policy_source_identity"] != identity(ADMISSION_POLICY_PATH):
        raise GateAbort("gate_admission_policy_source_changed")
    if ready["execution_helper_source_identity"] != identity(EXECUTION_HELPER_PATH):
        raise GateAbort("gate_execution_helper_changed")
    service = __import__("storage_scaleup_service_ops").facade()
    service.load_service_environment()
    tools, runner, worker, candidates, revision_ready = setup(args)
    if runner.request_governor_config.max_requests != ready["request_cap"]:
        raise GateAbort("gate_request_allowance_changed")
    if (runner.workers != ready["workers"]
            or runner.request_governor_config.max_inflight != ready["max_inflight"]):
        raise GateAbort("gate_concurrency_changed")
    if (args.output / "startup.json").exists():
        raise GateAbort("gate_existing_startup_no_automatic_resume")
    runner.framework_state_io.write_json_once(
        args.output / "startup.json", {"at": time.time(), "preflight": identity(args.output / "preflight.json")})
    if ready["revision_final_identity"] != identity(args.revision_root / "revision_final.json"):
        raise GateAbort("gate_revision_changed")
    for path, expected in ready["native_sources"].items():
        if identity(path) != expected:
            raise GateAbort("gate_native_source_changed")
    if tools.protected(args.state_root, args.frozen_root) != revision_ready["protected_inputs"]:
        raise GateAbort("gate_original_assets_changed")
    runner._prepared_identity = ready["prepared_input_identities"]
    governor = runner._request_governor()

    results, failure = [], None
    try:
        for selection, parent, candidate in candidates:
            if next(row for row in ready["candidates"]
                    if row["campaign_id"] == selection["campaign_id"])["status"] != "ready":
                continue
            request = SimpleNamespace(slot_key=parent.slot_key, question_type=parent.question_type,
                                      evidence_state=parent.evidence_state)
            root = campaign_root(args, selection)
            decision = worker._gate(request, parent, candidate, root)
            runner.framework_state_io.write_json_once(root / "decision.json", decision)
            results.append({"campaign_id": selection["campaign_id"], "slot_key": parent.slot_key,
                            "decision": decision})
    except BaseException as exc:
        failure = {"exception_type": type(exc).__name__,
                   "reason_code": str(exc) if isinstance(exc, GateAbort) else "gate_execution_refused",
                   "candidate_failure": False}
    unchanged = tools.protected(args.state_root, args.frozen_root) == revision_ready["protected_inputs"]
    final = {"status": "stopped_on_operational_failure" if failure else "gate_complete",
             "results": results, "failure": failure, "request_governor": governor.summary(),
             "protected_inputs_unchanged": unchanged, "published": False, "at": time.time()}
    runner.framework_state_io.write_json_once(args.output / "final.json", final)
    return final


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("preflight", "run"))
    for name in ("frozen-root", "state-root", "revision-root", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--request-cap", type=int)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--request-concurrency", type=int, default=4)
    args = parser.parse_args()
    if not 1 <= args.workers <= 64 or not 1 <= args.request_concurrency <= 64:
        raise SystemExit("Gate concurrency must be within 1..64")
    if args.request_cap is not None and args.request_cap <= 0:
        raise SystemExit("remaining Gate allowance must be positive")
    if args.output.is_relative_to(args.frozen_root) or args.output.is_relative_to(args.state_root):
        raise SystemExit("Gate must be outside frozen assets")
    try:
        result = preflight(args) if args.action == "preflight" else run(args)
        # Decisions include opaque sample identifiers; emit only aggregate status.
        print(json.dumps({key: result[key] for key in (
            "status", "model_requests", "planned_episodes", "request_governor",
            "failure", "protected_inputs_unchanged") if key in result}, sort_keys=True))
    except BaseException as exc:
        import traceback
        print(json.dumps({"status": "operational_failure", "exception_type": type(exc).__name__,
                          "frames": [{"file": Path(f.filename).name, "line": f.lineno}
                                     for f in traceback.extract_tb(exc.__traceback__)]}))
        raise SystemExit(2) from None


if __name__ == "__main__":
    main()
