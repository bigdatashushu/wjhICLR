"""Independent native dual-seed Gate, with label-free execution and no publication."""
from __future__ import annotations

import argparse
import copy
from dataclasses import asdict
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import time
from types import MethodType, SimpleNamespace


class GateAbort(BaseException):
    """A service, identity or budget refusal must escape model failure handlers."""


class PreparationComplete(BaseException):
    pass


class GateClientError(RuntimeError):
    """Fixed-code error, recorded by the native seeded-client receipt writer."""


def identity(path):
    data = Path(path).read_bytes()
    return {"size": len(data), "sha256": hashlib.sha256(data).hexdigest()}


def read(path):
    return json.loads(Path(path).read_text())


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


class CheckedClient:
    def __init__(self, client, model, seed):
        self.client, self.model, self.seed = client, model, seed
        self.calls = 0
        client.transport_retries = 0

    def __getattr__(self, name):
        return getattr(self.client, name)

    def chat(self, *args, **kwargs):
        from skill3d.synthesis.request_audit import describe_request
        from skill3d.evaluation.fitset_v11 import content_sha256
        if (self.calls >= 64 or kwargs.get("seed") != self.seed
                or self.client.transport_retries != 0
                or self.client.audit_identity(seed=self.seed).get("model") != self.model):
            raise GateClientError("gate_request_identity_or_episode_budget")
        expected = describe_request(
            self.client.build_payload(*args, **kwargs),
            model_identity=self.client.audit_identity(seed=self.seed),
            wire_encoding_known=True)
        self.calls += 1
        try:
            result = self.client.chat(*args, **kwargs)
        except Exception:
            raise GateClientError("gate_service_failure") from None
        if (self.client.last_response_identity.get("model") != self.model
                or content_sha256(self.client.last_request_identity) != content_sha256(expected)):
            raise GateClientError("gate_response_or_wire_identity")
        return result


class CheckedFactory:
    def __init__(self, config, endpoint, model):
        self.config, self.endpoint, self.model = config, endpoint, model

    def __call__(self, seed, output):
        from run_fitset_v11 import _RevisionClientFactory
        from skill3d.evolution.fitset_stream_v11 import SeededGateClient
        from skill3d.evaluation.fitset_v11 import _variant_key
        from skill3d.online.config import online_model_from

        class FailClosedSeededClient(SeededGateClient):
            def chat(self, *args, **kwargs):
                try:
                    return super().chat(*args, **kwargs)
                except Exception as exc:
                    # Native receipts finish before the error escapes online
                    # model-failure handlers as an operational stop.
                    code = str(exc) if isinstance(exc, GateClientError) else "gate_client_refused"
                    raise GateAbort(code) from None

        def make(variant_id):
            raw = _RevisionClientFactory(
                self.config, online_model_from(self.config).max_tokens, self.endpoint)()
            if not raw.configured:
                raise GateAbort("gate_service_unconfigured")
            return FailClosedSeededClient(
                CheckedClient(raw, self.model, seed), seed,
                output / "request_seeds" / (_variant_key(variant_id) + ".json"))
        return make


def setup(args):
    from skill3d.evaluation.fitset_v11 import SystemVariantRunner, content_sha256
    from skill3d.evolution.fitset_stream_v11 import FitSetWorkerEvolution
    from skill3d.evolution.fitset_revision_v11 import _static_validate
    from skill3d.schemas import SkillSpecV11
    from skill3d.synthesis.request_governor import RequestGovernorConfig
    import freeze_runner_binding_ops

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
        path = args.revision_root / row["source_kind"] / "SKILL.md"
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

    class IsolatedRunner(SystemVariantRunner):
        def _write_create_once(self, path, value):
            if not Path(path).is_relative_to(args.output):
                raise GateAbort("gate_write_scope")
            return super()._write_create_once(path, value)

        def _prepare_one(self, variant, *, preflight=False):
            prepared = super()._prepare_one(variant, preflight=preflight)
            plan = {key: value for key, value in prepared["plan"].items() if key != "skill_identity"}
            key = str(self.root.relative_to(args.output)) + "/" + str(variant["variant_id"])
            expected = getattr(self, "_prepared_identity", {}).get(key)
            if expected is not None and content_sha256(plan) != expected:
                raise GateAbort("gate_prepared_input_identity")
            return prepared

    coordinator = IsolatedRunner.__new__(IsolatedRunner)
    coordinator.__dict__ = copy.copy(vars(native))
    coordinator.root = args.output
    coordinator.framework_state_io = None
    coordinator.deployment_continuation = coordinator._continuation = None
    coordinator.cached_prefix_recovery = coordinator._cached_prefix_spec = None
    coordinator._request_governor_root = args.output / "request_governor"
    coordinator.request_governor_config = RequestGovernorConfig(
        max_requests=512, max_inflight=1, requests_per_second=0.0)
    coordinator._episode_semaphore = None
    coordinator.workers = coordinator.episode_concurrency = 1
    coordinator.run_identity = content_sha256({
        "schema": "harness3d-independent-successor-gate/1",
        "original_run": native.run_identity,
        "revision_final": identity(args.revision_root / "revision_final.json"),
        "gate_source": identity(Path(__file__)),
    })
    coordinator.gate_client_factory = CheckedFactory(
        config["settings"], config["transport"]["endpoint"], ready["model_identity"]["model"])
    worker = FitSetWorkerEvolution.__new__(FitSetWorkerEvolution)
    worker.runner, worker.config = coordinator, native.evolution_config
    worker.rows = native.gate_variant_rows
    worker.root = args.output
    if worker.config.gate_seeds != (137, 138) or worker.config.gate_mode != "qa_disjoint":
        raise GateAbort("gate_protocol_identity")
    return tools, coordinator, worker, candidates, ready


def preflight(args):
    from skill3d.evaluation.fitset_v11 import content_sha256
    args.output.mkdir(parents=True, exist_ok=True, mode=0o700)
    tools, runner, worker, candidates, revision_ready = setup(args)
    if tools.protected(args.state_root, args.frozen_root) != revision_ready["protected_inputs"]:
        raise GateAbort("gate_original_assets_changed")
    plans, outputs = {}, []

    def collect(coordinator, episodes):
        if len(episodes) > 512:
            raise GateAbort("gate_minimum_request_budget")
        pairs = {}
        for arm, variant in episodes:
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
        root = args.output / "campaigns" / selection["campaign_id"]
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
        "native_sources": revision_ready["source_identity"],
        "revision_final_identity": identity(args.revision_root / "revision_final.json"),
        "gate_config": asdict(worker.config), "seeds": [137, 138],
        "model_identity": revision_ready["model_identity"],
        "solver_config": {key: val for key, val in asdict(runner.base_cfg).items()
                          if key in {"seed", "temperature", "max_tokens", "max_rounds"}},
        "request_cap": 512, "per_episode_request_cap": 64,
        "max_inflight": 1, "transport_retries": 0,
        "executor": "frozen_native_SystemVariantRunner_without_original_receipt_writer",
        "validation_labels_hidden_during_execution": True, "published": False,
    }
    seal(args.output / "preflight.json", value)
    return {key: value[key] for key in ("status", "model_requests", "candidates", "planned_episodes")}


def run(args):
    from skill3d.synthesis.request_governor import RequestGovernor
    ready = read(args.output / "preflight.json")
    if ready["gate_source_identity"] != identity(Path(__file__)):
        raise GateAbort("gate_source_changed")
    seal(args.output / "startup.json", {"at": time.time(), "preflight": identity(args.output / "preflight.json")})
    service = __import__("storage_scaleup_service_ops").facade()
    service.load_service_environment()
    tools, runner, worker, candidates, revision_ready = setup(args)
    if ready["revision_final_identity"] != identity(args.revision_root / "revision_final.json"):
        raise GateAbort("gate_revision_changed")
    for path, expected in ready["native_sources"].items():
        if identity(path) != expected:
            raise GateAbort("gate_native_source_changed")
    if tools.protected(args.state_root, args.frozen_root) != revision_ready["protected_inputs"]:
        raise GateAbort("gate_original_assets_changed")
    runner._prepared_identity = ready["prepared_input_identities"]
    governor = RequestGovernor(runner.request_governor_root, runner.request_governor_config)

    def execute(coordinator, episodes):
        for arm, variant in episodes:
            key = str(arm.root.relative_to(args.output)) + "/" + str(variant["variant_id"])
            if key not in ready["prepared_input_identities"]:
                raise GateAbort("gate_unprepared_episode")
            if arm._result_path(variant["variant_id"]).exists():
                raise GateAbort("gate_existing_result_no_automatic_resume")
            result = arm._run_one(variant)
            if result.get("service_errors"):
                raise GateAbort("gate_episode_service_failure")
            seal(arm.root / "progress" / (hashlib.sha256(key.encode()).hexdigest() + ".json"), {
                "status": result["status"], "requests": result.get("model_request_count"),
                "at": time.time(), "gate_request_summary": governor.summary()})

    runner._run_gate_episodes = MethodType(execute, runner)
    results, failure = [], None
    try:
        for selection, parent, candidate in candidates:
            if next(row for row in ready["candidates"]
                    if row["campaign_id"] == selection["campaign_id"])["status"] != "ready":
                continue
            request = SimpleNamespace(slot_key=parent.slot_key, question_type=parent.question_type,
                                      evidence_state=parent.evidence_state)
            root = args.output / "campaigns" / selection["campaign_id"]
            decision = worker._gate(request, parent, candidate, root)
            seal(root / "decision.json", decision)
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
    seal(args.output / "final.json", final)
    return final


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("preflight", "run"))
    for name in ("frozen-root", "state-root", "revision-root", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
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
