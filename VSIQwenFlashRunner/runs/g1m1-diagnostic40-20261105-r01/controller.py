#!/usr/bin/env python3
"""Run a frozen diagnostic with existing solver, bounded requests and JSON traces."""

import argparse
from collections import Counter
from dataclasses import asdict, dataclass
import hashlib
import json
import os
from pathlib import Path
import shlex
import sys
import time
import traceback
import urllib.error
import urllib.request


class DiagnosticOperationalStop(BaseException):
    pass


class FailClosedClient:
    """Stop the scheduler on a transport failure without another model call."""

    def __init__(self, delegate):
        self.delegate = delegate

    def __getattr__(self, name):
        if name == "delegate":
            raise AttributeError(name)
        return getattr(self.delegate, name)

    def chat(self, *args, **kwargs):
        try:
            return self.delegate.chat(*args, **kwargs)
        except Exception as exc:
            raise DiagnosticOperationalStop(type(exc).__name__) from None


@dataclass(frozen=True)
class DiagnosticClientFactory:
    config: dict
    output: Path
    endpoint: str

    def __call__(self, variant_id):
        from skill3d.infra.experiment_stages_v11 import fitset_cli
        return FailClosedClient(fitset_cli()._SeededClientFactory(
            self.config, 137, self.output, self.endpoint,
            perception_capacity=8, deferred_admission=True)(variant_id))


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
    parser.add_argument("--action", choices=("preflight", "solve"), required=True)
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
    write(root / ("preflight_health.json" if args.action == "preflight" else "solve_health.json"), health)
    bundle = root / "seed_bundle/manifest.json"
    if not bundle.exists():
        parents, snapshot = load_active_parent_slots(source / "skill_library")
        build_fit_skill_bundle(
            output_dir=bundle.parent, bundle_id="diagnostic40-" + manifest["identity_sha256"][:16],
            candidates={}, active_parents=parents, source_run_identity=snapshot)
    provider = DerivedPublishedFitSetInputProvider(
        inputs / "published_inputs.json",
        derivation_root=root / "prepared_cache/derived_bindings_v4_metric",
        binding_cfg=cfg, max_scenes=2, audit_mode="reuse_source_audit")
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
        request_governor_config=RequestGovernorConfig(240, 8, 0),
        client_factory=DiagnosticClientFactory(model_config, output, endpoint),
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
            assert value["evidence_state"] == "g1m1"
    else:
        preflight = json.loads((output / "preflight_summary.json").read_text())
        assert preflight["status"] == "complete" and preflight["model_requests"] == 0
        assert not (root / "solve_start.json").exists(), "duplicate model launch forbidden"
        write(root / "solve_start.json", {
            "started_at": start, "pid": os.getpid(), "diagnostic_identity": manifest["identity_sha256"],
            "automatic_model_retry": False, "reference_answer_visible": False})
        summary = runner.run(resume=True)
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
