#!/usr/bin/env python3
"""Run a frozen diagnostic with existing solver, bounded requests and JSON traces."""

import argparse
from collections import Counter
import copy
from dataclasses import asdict, dataclass
import hashlib
import json
import os
from pathlib import Path
import shlex
import sys
import threading
import time
import traceback
import urllib.error
import urllib.request


class DiagnosticOperationalStop(BaseException):
    pass


class SceneParallelInputs:
    """Independent unmodified providers remove the global cross-scene lock."""

    def __init__(self, base):
        self.base = base
        self.children = {}
        self.lock = threading.RLock()

    def __getattr__(self, name):
        if name == "base":
            raise AttributeError(name)
        return getattr(self.base, name)

    def __getstate__(self):
        return {"base": self.base}

    def __setstate__(self, state):
        self.__init__(state["base"])

    def _child(self, variant):
        scene = variant["scene_id"]
        with self.lock:
            if scene not in self.children:
                child = type(self.base).__new__(type(self.base))
                child.__setstate__(copy.deepcopy(self.base.__getstate__()))
                self.children[scene] = child
            return self.children[scene]

    def __call__(self, variant):
        return self._child(variant)(variant)

    def receipt(self, variant, item):
        return self._child(variant).receipt(variant, item)

    def prepare_published(self, variant, item):
        return self._child(variant).prepare_published(variant, item)

    def for_spawned_episode(self, variant):
        return self._child(variant).for_spawned_episode(variant)

    def frozen_frame_bytes(self, variant):
        return self._child(variant).frozen_frame_bytes(variant)


class FailClosedClient:
    """Stop the scheduler on a transport failure without another model call."""

    def __init__(self, delegate, enable_thinking=False):
        self.delegate = delegate
        self.calls = 0
        self.enable_thinking = enable_thinking

    def __getattr__(self, name):
        if name == "delegate":
            raise AttributeError(name)
        return getattr(self.delegate, name)

    def chat(self, *args, **kwargs):
        self.calls += 1
        if self.calls > 7:
            raise DiagnosticOperationalStop("episode_request_cap_exceeded")
        kwargs["enable_thinking"] = self.enable_thinking
        try:
            return self.delegate.chat(*args, **kwargs)
        except Exception as exc:
            raise DiagnosticOperationalStop(type(exc).__name__) from None


@dataclass(frozen=True)
class DiagnosticClientFactory:
    config: dict
    output: Path
    endpoint: str
    enable_thinking: bool = False

    def __call__(self, variant_id):
        from skill3d.infra.experiment_stages_v11 import fitset_cli
        return FailClosedClient(fitset_cli()._SeededClientFactory(
            self.config, 137, self.output, self.endpoint,
            perception_capacity=8, deferred_admission=True)(variant_id),
            enable_thinking=self.enable_thinking)


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n")
    temporary.replace(path)


def authenticate(endpoint):
    env_file = Path("/home/wangy/wangjh/harness3d/precompute-20261022-r01/.transfer.env")
    values = {}
    for line in env_file.read_text().splitlines():
        words = shlex.split(line)
        if words and words[0] == "export":
            words = words[1:]
        if len(words) == 1 and "=" in words[0]:
            key, value = words[0].split("=", 1)
            values[key] = value
    token = values.get("HARNESS3D_PERCEPTION_TOKEN") or values.get("HARNESS3D_TRANSFER_TOKEN")
    if not token:
        raise DiagnosticOperationalStop("authorized_gateway_token_unavailable")
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    request = urllib.request.Request(
        endpoint + "/diagnostic-auth-check", headers={"Authorization": "Bearer " + token})
    try:
        opener.open(request, timeout=10)
    except urllib.error.HTTPError as exc:
        if exc.code != 404:
            raise DiagnosticOperationalStop("gateway_authentication_failed") from None
    os.environ["HARNESS3D_PERCEPTION_TOKEN"] = token
    with opener.open(endpoint + "/health", timeout=10) as response:
        raw = response.read(65537)
    assert token.encode() not in raw
    health = json.loads(raw)
    assert health["status"] == "ready" and health["max_concurrency"] == 8
    return {k: health[k] for k in ("schema_version", "status", "identity",
                                  "max_concurrency", "upstream") if k in health}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--action", choices=("preflight", "solve", "direct"), required=True)
    parser.add_argument("--skip-validated-preflight", action="store_true")
    args = parser.parse_args()
    root = args.run_root.resolve()
    manifest = json.loads((root / "diagnostic.json").read_text())
    source = root / "code/source"
    sys.path.insert(0, str(source / "src"))
    from skill3d.evaluation.fitset_v11 import (
        SystemVariantRunner, build_fit_skill_bundle, content_sha256, file_identity,
        load_fitset_panel,
    )
    from skill3d.evaluation.derived_published_fitset_v11 import DerivedPublishedFitSetInputProvider
    from skill3d.evolution.fitset_revision_v11 import load_active_parent_slots
    from skill3d.infra.experiment_stages_v11 import fitset_cli
    from skill3d.online.config import load_config, online_model_from, resolve_runtime
    from skill3d.online.scene_overview import SceneOverviewStore
    from skill3d.synthesis.request_governor import RequestGovernorConfig
    assert content_sha256({k: v for k, v in manifest.items() if k != "identity_sha256"
                          }) == manifest["identity_sha256"]
    assert file_identity(root / "controller.py")["sha256"] == manifest["controller_sha256"]
    for relative, expected in manifest["input_files"].items():
        assert file_identity(root / "inputs" / relative)["sha256"] == expected
    code_receipt = json.loads((root / "code/code_identity.json").read_text())
    assert code_receipt["source_sha256"] == manifest["source_sha256"]
    for row in code_receipt["files"]:
        assert file_identity(source / row["path"])["sha256"] == row["sha256"]
    os.environ["SKILL3D_CODE_RECEIPT"] = str(root / "code/code_identity.json")
    os.environ["SKILL3D_RUN_MODE"] = "formal"
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    inputs, output = root / "inputs", root / "system"
    cli = fitset_cli()
    raw = resolve_runtime(load_config(inputs / "online.yaml"), seed=137)
    cfg = cli._base_config(raw)
    assert cfg.max_solver_rounds == 6
    selector = argparse.Namespace(
        experiment_split_manifest=str(inputs / "split.json"), execution_role="test",
        phase="final_system", gate_mode="scene_disjoint", gate_variant_ids="",
        gate_panel_name="", panel_name="ready_579", metadata=str(inputs / "host_evaluation/metadata.jsonl"),
        variant_ids="", seed=137,
    )
    panel = load_fitset_panel(inputs / "panel.json")
    variants, gates, split = cli._select_experiment_variants(panel, selector)
    assert len(variants) == 40 and not gates
    assert Counter(row["question_type"] for row in variants) == manifest["task_counts"]
    endpoint = "http://10.10.0.198:27035"
    health = authenticate(endpoint)
    model_config = {"online_model": asdict(online_model_from(raw))}
    client = cli._RevisionClientFactory(model_config, cfg.max_tokens, endpoint)()
    assert health["identity"] == client.audit_identity()
    write(root / (args.action + "_health.json"), health)
    thinking = manifest["runtime"].get("enable_thinking", False)
    if args.action == "direct":
        from skill3d.infra.experiment_stages_v11 import baseline32
        assert thinking is True
        governor_root = None
        continuation = manifest.get("direct_continuation")
        if continuation:
            origin = Path(continuation["origin_native_root"])
            origin_manifest = json.loads((origin / "diagnostic.json").read_text())
            assert origin_manifest["identity_sha256"] == continuation["origin_identity"]
            assert origin_manifest["source_sha256"] == continuation["origin_source_sha256"]
            assert origin_manifest["input_files"] == manifest["input_files"]
            assert origin_manifest["runtime"] == manifest["runtime"]
            for relative, expected in continuation["files"].items():
                assert file_identity(origin / relative)["sha256"] == expected
            done = set(continuation["completed_qa_ids"])
            responses = [json.loads(p.read_text()) for p in (origin / "direct/responses").glob("*.json")]
            assert {row["qa_id"] for row in responses} == done
            assert len(responses) == continuation["origin_model_requests"] == len(done)
            assert all(row["status"] == "completed" for row in responses)
            assert len(list((origin / "direct/intents").glob("*.json"))) == len(done)
            state = json.loads((origin / "requests/state.json").read_text())
            previous_requests = [row for row in state["requests"]
                                 if row["context"]["variant_id"].startswith("baseline:")]
            assert len(previous_requests) == len(done)
            assert all(row["status"] == "completed" for row in previous_requests)
            variants = [row for row in variants if row["qa_id"] not in done]
            assert {row["qa_id"] for row in variants} == set(continuation["remaining_qa_ids"])
            governor_root = origin / "requests"
        start = time.time()
        assert not (root / "direct_start.json").exists(), "duplicate direct launch forbidden"
        write(root / "direct_start.json", {
            "started_at": start, "pid": os.getpid(), "diagnostic_identity": manifest["identity_sha256"],
            "automatic_model_retry": False, "reference_answer_visible": False,
            "enable_thinking": True, "max_requests_per_question": 1})
        direct_manifest = {
            "run_root": str(root), "identity_sha256": manifest["identity_sha256"],
            "profile": {
                "inputs": {"published_inputs": str(inputs / "published_inputs.json"),
                           "metadata": str(inputs / "host_evaluation/metadata.jsonl"),
                           "online_config": str(inputs / "online.yaml")},
                "runtime": {**manifest["runtime"], "seed": 137, "requests_per_second": 0,
                            "perception_endpoint": endpoint},
                "baseline": {"workers": 4},
            },
        }
        summary = baseline32(direct_manifest, root / "direct", variants=variants,
                             enable_thinking=True, rgb_only=True,
                             request_governor_root=governor_root)
        write(root / "direct_receipt.json", {
            "status": "complete", "started_at": start, "finished_at": time.time(),
            "summary": summary, "reference_answer_visible": False})
        print(json.dumps(summary, sort_keys=True), flush=True)
        return
    bundle = root / "seed_bundle/manifest.json"
    if not bundle.exists():
        parents, snapshot = load_active_parent_slots(source / "skill_library")
        build_fit_skill_bundle(
            output_dir=bundle.parent, bundle_id="diagnostic40-" + manifest["identity_sha256"][:16],
            candidates={}, active_parents=parents, source_run_identity=snapshot)
    provider = SceneParallelInputs(DerivedPublishedFitSetInputProvider(
        inputs / "published_inputs.json",
        derivation_root=root / "prepared_cache/derived_bindings_v4_metric",
        binding_cfg=cfg, max_scenes=manifest["runtime"]["scene_cache_size"],
        audit_mode="reuse_source_audit"))
    runner = SystemVariantRunner(
        output_root=output, panel=panel, panel_name="ready_579",
        variants=variants, gate_variants=[], item_provider=provider,
        input_receipt_provider=provider.receipt,
        input_contract_identity={
            **provider.input_contract_identity,
            "offline_metadata": file_identity(inputs / "host_evaluation/metadata.jsonl"),
            "diagnostic_identity": manifest["identity_sha256"],
            "model_transport": {"kind": "perception_rpc", "upstream_capacity": 8,
                                "endpoint_sha256": hashlib.sha256(endpoint.encode()).hexdigest()},
        },
        binding_refs={row["variant_id"]: provider.public_entry(row)["reference"] for row in variants},
        base_cfg=cfg, phase="final_system", published_inputs=provider,
        experiment_split=split, scene_overviews=SceneOverviewStore(inputs / "scene_overviews/manifest.json"),
        skills_for_variant=cli._fit_skill_provider(bundle),
        min_free_gib=80, workers=16, scheduler="episode_pool", skill_owners=4,
        offline_frame_storage="shared_scene_v2", prefetch_scenes=True,
        episode_concurrency=16,
        request_governor_config=RequestGovernorConfig(manifest["runtime"]["max_requests"], 8, 0),
        client_factory=DiagnosticClientFactory(model_config, output, endpoint, thinking),
    )
    runner._request_governor_root = root / "requests"
    start = time.time()
    if args.action == "preflight":
        assert not (output / "manifest.json").exists()
        summary = runner.preflight(resume=False)
        assert summary["status"] == "complete" and summary["completed_preflights"] == 40
        assert summary["model_requests"] == 0
        for receipt in (output / "preflight").glob("*.json"):
            value = json.loads(receipt.read_text())
            assert value["runtime_evidence_state"] == "g1m1"
    else:
        if args.skip_validated_preflight:
            assert manifest["runtime"]["question_preflight"] == "skipped_per_user_instruction"
        else:
            preflight = json.loads((output / "preflight_summary.json").read_text())
            assert preflight["status"] == "complete" and preflight["model_requests"] == 0
        assert not (root / "solve_start.json").exists(), "duplicate model launch forbidden"
        write(root / "solve_start.json", {
            "started_at": start, "pid": os.getpid(), "diagnostic_identity": manifest["identity_sha256"],
            "automatic_model_retry": False, "reference_answer_visible": False,
            "enable_thinking": thinking,
            "question_preflight_skipped": args.skip_validated_preflight})
        summary = runner.run(resume=(output / "manifest.json").exists())
        assert summary["status"] == "complete" and summary["completed_variants"] == 40
    write(root / (args.action + "_receipt.json"), {
        "status": "complete", "started_at": start, "finished_at": time.time(),
        "summary": summary, "reference_answer_visible": False})
    print(json.dumps(summary, sort_keys=True), flush=True)


if __name__ == "__main__":
    try:
        main()
    except BaseException as exc:
        # Do not emit credential-bearing service exception text.
        print(json.dumps({"status": "stopped", "error_type": type(exc).__name__,
                          "stack": [{"file": Path(row.filename).name, "line": row.lineno,
                                     "function": row.name}
                                    for row in traceback.extract_tb(exc.__traceback__)]}), flush=True)
        raise SystemExit(1)
