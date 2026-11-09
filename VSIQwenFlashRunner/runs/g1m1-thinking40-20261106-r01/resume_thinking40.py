#!/usr/bin/env python3
"""Explicit, bounded recovery of the frozen thinking40 pair; never retry automatically."""

import argparse
from collections import Counter
from dataclasses import asdict
import fcntl
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback

ROOT = Path("/home/wangy/wangjh/harness3d/VSIQwenFlashRunner/runs/g1m1-thinking40-20261106-r01")
DIRECT_ROOT = ROOT.parent / "g1m1-thinking40-direct-continuation-20261106-r02"
RECOVERY = ROOT / "resume-20261106-r03"
ENDPOINT = "http://10.10.0.198:27035"
PYTHON = "/home/wangy/wangjh/skill3d-lab/pilot-20261021-01/runtime/skill3d-exp/bin/python"
EXPECTED = "3d6a5d3b2729ee173c48b03dbf264dccf4afe74f109166cc20d8e292480f6735"
EXPECTED_DIRECT = "f6faad3820b6e3bc9fa60309adc56f641d6d5cf982d299b09d1c4cdba08e5f72"


def read(path):
    return json.loads(path.read_text())


def sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for data in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(data)
    return digest.hexdigest()


def write(path, value, *, exclusive=False):
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(value, sort_keys=True, indent=2) + "\n"
    if exclusive:
        with path.open("x") as stream:
            stream.write(text)
    else:
        temporary = path.with_name("." + path.name + "." + str(os.getpid()) + ".tmp")
        temporary.write_text(text)
        temporary.replace(path)


def environment():
    env = dict(os.environ)
    for key in list(env):
        if any(word in key.upper() for word in ("TOKEN", "API_KEY", "SECRET", "PASSWORD")):
            env.pop(key)
    env.update(
        LD_PRELOAD="/home/wangy/wangjh/harness3d/precompute-20261022-r01/cpu_sqlite/libsqlite3.so.0",
        PYTHONPATH="/home/wangy/wangjh/skill3d-lab/pilot-20261021-01/runtime/cpu_compat",
        OMP_NUM_THREADS="2", OPENBLAS_NUM_THREADS="2", MKL_NUM_THREADS="2",
        CUDA_VISIBLE_DEVICES="", PYTHONUNBUFFERED="1")
    return env


def owners():
    result = []
    for path in Path("/proc").glob("[0-9]*/cmdline"):
        try:
            words = path.read_bytes().split(b"\0")
            pid = int(path.parent.name)
            if pid == os.getpid() or b"-c" in words:
                continue
            if not Path(words[0].decode()).name.startswith("python"):
                continue
            if any(str(root).encode() in b"\0".join(words) for root in (ROOT, DIRECT_ROOT)):
                result.append(pid)
        except (OSError, UnicodeError):
            continue
    return result


def process_identity(pid):
    try:
        stat = Path("/proc") / str(pid) / "stat"
        start_ticks = int(stat.read_text().split(") ", 1)[1].split()[19])
        raw = (stat.parent / "cmdline").read_bytes()
        return {"pid": pid, "start_ticks": start_ticks,
                "cmdline_sha256": hashlib.sha256(raw).hexdigest()}
    except OSError:
        return {"pid": pid, "exited_before_identity_capture": True}


def load_controller(root):
    sys.path.insert(0, str(root / "code/source/src"))
    sys.path.insert(0, str(root))
    spec = importlib.util.spec_from_file_location("controller", root / "controller.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules["controller"] = module
    spec.loader.exec_module(module)
    return module


def verify_frozen(root, expected):
    from skill3d.evaluation.fitset_v11 import content_sha256
    manifest = read(root / "diagnostic.json")
    assert manifest["identity_sha256"] == expected
    assert content_sha256({k: v for k, v in manifest.items()
                           if k != "identity_sha256"}) == expected
    assert sha(root / "controller.py") == manifest["controller_sha256"]
    for relative, expected_hash in manifest["input_files"].items():
        # Host labels are only hashed as bytes, never parsed or passed to a solver.
        assert sha(root / "inputs" / relative) == expected_hash
    receipt = read(root / "code/code_identity.json")
    assert receipt["source_sha256"] == manifest["source_sha256"]
    for row in receipt["files"]:
        assert sha(root / "code/source" / row["path"]) == row["sha256"]
    from skill3d.infra.experiment_startup import verify_source_receipt
    verify_source_receipt(root / "code/code_identity.json", root / "code/source")
    assert manifest["runtime"]["enable_thinking"] is True
    assert manifest["runtime"]["max_requests"] == 320
    assert manifest["runtime"]["max_requests_per_episode"] == 7
    assert manifest["runtime"]["automatic_model_retry"] is False
    return manifest


def select(root, manifest):
    from skill3d.infra.experiment_stages_v11 import fitset_cli
    from skill3d.evaluation.fitset_v11 import load_fitset_panel
    from skill3d.online.config import load_config, resolve_runtime, online_model_from
    inputs = root / "inputs"
    cli = fitset_cli()
    raw = resolve_runtime(load_config(inputs / "online.yaml"), seed=137)
    cfg = cli._base_config(raw)
    selector = argparse.Namespace(
        experiment_split_manifest=str(inputs / "split.json"), execution_role="test",
        phase="final_system", gate_mode="scene_disjoint", gate_variant_ids="",
        gate_panel_name="", panel_name="ready_579",
        metadata=str(inputs / "host_evaluation/metadata.jsonl"),
        variant_ids="", seed=137)
    panel = load_fitset_panel(inputs / "panel.json")
    variants, gates, split = cli._select_experiment_variants(panel, selector)
    assert len(variants) == 40 and not gates
    assert Counter(v["question_type"] for v in variants) == manifest["task_counts"]
    assert cfg.max_solver_rounds == 6
    return cli, raw, cfg, panel, variants, split, {"online_model": asdict(online_model_from(raw))}


def health_check(controller, model_config, max_tokens):
    from skill3d.infra.experiment_stages_v11 import fitset_cli
    health = controller.authenticate(ENDPOINT)
    client = fitset_cli()._RevisionClientFactory(model_config, max_tokens, ENDPOINT)()
    assert health["identity"] == client.audit_identity()
    assert health["identity"] == read(ROOT / "solve_health.json")["identity"]
    assert health["status"] == "ready" and health["max_concurrency"] == 8
    return {"status": health["status"], "max_concurrency": 8,
            "identity_sha256": hashlib.sha256(
                json.dumps(health["identity"], sort_keys=True).encode()).hexdigest(),
            "original_identity_verified": True, "model_requests": 0}


def result_rows():
    return [read(p) for p in (ROOT / "system/results").glob("*.json")]


def direct_rows():
    rows = []
    for root in (ROOT / "direct", DIRECT_ROOT / "direct", RECOVERY / "direct"):
        rows.extend(read(p) for p in (root / "responses").glob("*.json"))
    return rows


def preserved_files():
    paths = []
    for root in (ROOT, DIRECT_ROOT):
        for name in ("diagnostic.json", "controller.py", "code/code_identity.json",
                     "solve_start.json", "direct_start.json", "system_exit.json",
                     "direct_exit.json", "pair_start.json", "exit.json"):
            path = root / name
            if path.is_file():
                paths.append(path)
        paths.extend((root / "direct").glob("**/*.json"))
    paths.extend((ROOT / "system/results").glob("*.json"))
    paths.extend((ROOT / "system/variants").glob("*/outcome.json"))
    paths.extend((ROOT / "seed_bundle").glob("**/*.json"))
    return {str(p): sha(p) for p in sorted(set(paths))}


def make_runner(controller, root, manifest, setup):
    from skill3d.evaluation.fitset_v11 import SystemVariantRunner, file_identity
    from skill3d.evaluation.derived_published_fitset_v11 import DerivedPublishedFitSetInputProvider
    from skill3d.online.scene_overview import SceneOverviewStore
    from skill3d.synthesis.request_governor import RequestGovernorConfig
    cli, raw, cfg, panel, variants, split, model_config = setup
    inputs, output = root / "inputs", root / "system"
    provider = controller.SceneParallelInputs(DerivedPublishedFitSetInputProvider(
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
                                "endpoint_sha256": hashlib.sha256(ENDPOINT.encode()).hexdigest()}},
        binding_refs={v["variant_id"]: provider.public_entry(v)["reference"] for v in variants},
        base_cfg=cfg, phase="final_system", published_inputs=provider,
        experiment_split=split,
        scene_overviews=SceneOverviewStore(inputs / "scene_overviews/manifest.json"),
        skills_for_variant=cli._fit_skill_provider(root / "seed_bundle/manifest.json"),
        min_free_gib=80, workers=16, scheduler="episode_pool", skill_owners=4,
        offline_frame_storage="shared_scene_v2", prefetch_scenes=True,
        episode_concurrency=16,
        request_governor_config=RequestGovernorConfig(320, 8, 0),
        # Seed/response artifacts belong to this explicit recovery attempt.
        client_factory=controller.DiagnosticClientFactory(
            model_config, RECOVERY / "system_attempt", ENDPOINT, True))
    runner._request_governor_root = root / "requests"
    return runner


def plan():
    assert not owners(), "a prior experiment owner is still active"
    controller = load_controller(ROOT)
    manifest = verify_frozen(ROOT, EXPECTED)
    # Validate the direct snapshot without importing a second scientific runtime.
    direct = read(DIRECT_ROOT / "diagnostic.json")
    assert direct["identity_sha256"] == EXPECTED_DIRECT
    from skill3d.evaluation.fitset_v11 import content_sha256
    assert content_sha256({k: v for k, v in direct.items()
                           if k != "identity_sha256"}) == EXPECTED_DIRECT
    assert direct["input_files"] == manifest["input_files"]
    assert direct["runtime"] == manifest["runtime"]
    assert sha(DIRECT_ROOT / "controller.py") == direct["controller_sha256"]
    receipt = read(DIRECT_ROOT / "code/code_identity.json")
    assert receipt["source_sha256"] == direct["source_sha256"]
    for row in receipt["files"]:
        assert sha(DIRECT_ROOT / "code/source" / row["path"]) == row["sha256"]
    setup = select(ROOT, manifest)
    cli, raw, cfg, panel, variants, split, config = setup
    state = read(ROOT / "requests/state.json")
    assert state["policy"] == {"max_requests": 320, "max_inflight": 8, "requests_per_second": 0}
    assert all(r["status"] in ("completed", "error") for r in state["requests"])
    system = result_rows()
    assert all(r["status"] == "completed" for r in system)
    system_done = {r["variant_id"] for r in system}
    direct_done = {r["qa_id"] for r in direct_rows() if r["status"] == "completed"}
    pending_system = sorted(v["variant_id"] for v in variants if v["variant_id"] not in system_done)
    pending_direct = sorted(v["qa_id"] for v in variants if v["qa_id"] not in direct_done)
    failures = [r for r in state["requests"] if r["status"] == "error"]
    system_errors = sorted({r["context"]["variant_id"] for r in failures
                            if not r["context"]["variant_id"].startswith("baseline:")})
    direct_errors = sorted({r["qa_id"] for r in direct_rows() if r["status"] == "service_error"})
    assert set(system_errors) <= set(pending_system)
    assert set(direct_errors) <= set(pending_direct)
    assert len(state["requests"]) + 7 * len(pending_system) + len(pending_direct) <= 320
    system_count = sum(not r["context"]["variant_id"].startswith("baseline:")
                       for r in state["requests"])
    assert system_count + 7 * len(pending_system) <= 280
    assert shutil.disk_usage(ROOT).free / 2**30 >= 80
    health = health_check(controller, config, cfg.max_tokens)
    os.environ["SKILL3D_CODE_RECEIPT"] = str(ROOT / "code/code_identity.json")
    os.environ["SKILL3D_RUN_MODE"] = "formal"
    # Recompute the frozen system manifest, without solver/model execution.
    runner = make_runner(controller, ROOT, manifest, setup)
    runner.initialize(resume=True)
    plan_value = {
        "schema_version": "thinking40-explicit-recovery-v1",
        "created_at": time.time(), "origin_identity": EXPECTED,
        "direct_origin_identity": EXPECTED_DIRECT,
        "source_sha256": manifest["source_sha256"],
        "direct_source_sha256": direct["source_sha256"],
        "recovery_script_sha256": sha(Path(__file__).resolve()),
        "automatic_model_retry": False, "reference_answer_visible": False,
        "same_40_questions": True, "same_scientific_runtime": True,
        "system_manifest_recomputed_and_verified": True,
        "original_start_and_exit_records_preserved": True,
        "original_errors_preserved": True,
        "system_completed_before": len(system_done), "direct_completed_before": len(direct_done),
        "pending_system_variant_ids": pending_system, "pending_direct_qa_ids": pending_direct,
        "explicitly_authorized_recovery_system_variant_ids": system_errors,
        "explicitly_authorized_recovery_direct_qa_ids": direct_errors,
        "system_requests_before": system_count, "reserved_requests_before": len(state["requests"]),
        "global_request_cap": 320, "global_remaining_requests": 320 - len(state["requests"]),
        "maximum_additional_requests": 7 * len(pending_system) + len(pending_direct),
        "max_requests_per_system_attempt": 7, "max_requests_per_direct_attempt": 1,
        "direct_original_request_cap": 40,
        "additional_direct_recovery_attempt_cap": len(direct_errors),
        "health": health, "preserved_files": preserved_files()}
    RECOVERY.mkdir(exist_ok=True)
    write(RECOVERY / "request_state_before.json", state, exclusive=True)
    write(RECOVERY / "plan.json", plan_value, exclusive=True)
    print(json.dumps({k: plan_value[k] for k in (
        "system_completed_before", "direct_completed_before", "global_remaining_requests",
        "maximum_additional_requests", "system_manifest_recomputed_and_verified")}), flush=True)


def verify_plan():
    value = read(RECOVERY / "plan.json")
    assert sha(Path(__file__).resolve()) == value["recovery_script_sha256"]
    for path, expected_hash in value["preserved_files"].items():
        assert sha(Path(path)) == expected_hash
    return value


def archive_unfinished(plan_value):
    finished_keys = {p.stem for p in (ROOT / "system/results").glob("*.json")}
    moved = []
    for path in (ROOT / "system/variants").iterdir():
        if path.name in finished_keys:
            continue
        assert not (path / "outcome.json").exists()
        archive = RECOVERY / "prior_system_attempts" / path.name
        archive.parent.mkdir(parents=True, exist_ok=True)
        assert not archive.exists()
        path.rename(archive)
        moved.append({"from": str(path), "to": str(archive)})
        dispatch = ROOT / "system/dispatch" / (path.name + ".json")
        if dispatch.is_file():
            target = RECOVERY / "prior_dispatch" / dispatch.name
            target.parent.mkdir(parents=True, exist_ok=True)
            dispatch.rename(target)
    write(RECOVERY / "archive_receipt.json", {
        "moved_without_deletion": moved,
        "original_seed_response_files_preserved": True,
        "original_model_request_ledger_preserved": True})


def arm(action):
    with (RECOVERY / (action + ".lock")).open("a+b") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        value = verify_plan()
        assert not (RECOVERY / (action + "_start.json")).exists()
        root = ROOT if action == "solve" else DIRECT_ROOT
        controller = load_controller(root)
        manifest = verify_frozen(root, EXPECTED if action == "solve" else EXPECTED_DIRECT)
        setup = select(root, manifest)
        cli, raw, cfg, panel, variants, split, model_config = setup
        health = health_check(controller, model_config, cfg.max_tokens)
        os.environ["SKILL3D_CODE_RECEIPT"] = str(root / "code/code_identity.json")
        os.environ["SKILL3D_RUN_MODE"] = "formal"
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        write(RECOVERY / (action + "_start.json"), {
            **process_identity(os.getpid()), "started_at": time.time(),
            "origin_identity": manifest["identity_sha256"],
            "automatic_model_retry": False, "reference_answer_visible": False,
            "enable_thinking": True, "health": health}, exclusive=True)
        if action == "solve":
            runner = make_runner(controller, root, manifest, setup)
            runner.initialize(resume=True)
            archive_unfinished(value)
            summary = runner.run(resume=True)
            assert summary["status"] == "complete" and summary["completed_variants"] == 40
        else:
            from skill3d.infra.experiment_stages_v11 import baseline32
            selected = [v for v in variants if v["qa_id"] in value["pending_direct_qa_ids"]]
            assert len(selected) == len(value["pending_direct_qa_ids"])
            direct_manifest = {
                "run_root": str(root), "identity_sha256": manifest["identity_sha256"],
                "profile": {
                    "inputs": {"published_inputs": str(root / "inputs/published_inputs.json"),
                               "metadata": str(root / "inputs/host_evaluation/metadata.jsonl"),
                               "online_config": str(root / "inputs/online.yaml")},
                    "runtime": {**manifest["runtime"], "seed": 137, "requests_per_second": 0,
                                "perception_endpoint": ENDPOINT},
                    "baseline": {"workers": 4}}}
            summary = baseline32(
                direct_manifest, RECOVERY / "direct", variants=selected,
                enable_thinking=True, rgb_only=True, request_governor_root=ROOT / "requests")
            completed = {r["qa_id"] for r in direct_rows() if r["status"] == "completed"}
            assert len(completed) == 40
        verify_plan()
        write(RECOVERY / (action + "_receipt.json"), {
            "status": "complete", "finished_at": time.time(), "summary": summary,
            "original_results_hashes_verified": True, "automatic_model_retry": False})
        print(json.dumps({"action": action, "status": "complete"}), flush=True)


def launch():
    # The detached pair owner has a single launch record; the arm owners acquire OS locks.
    with (RECOVERY / "pair.lock").open("a+b") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        value = verify_plan()
        assert not (RECOVERY / "pair_start.json").exists()
        # This process is the only permitted owner at this boundary.
        assert not owners(), "another owner already launched"
        write(RECOVERY / "pair_start.json", {
            **process_identity(os.getpid()), "started_at": time.time(),
            "automatic_model_retry": False, "origin_identity": EXPECTED}, exclusive=True)
        children = {}
        for action in ("solve", "direct"):
            with (RECOVERY / (action + ".log")).open("xb") as log:
                process = subprocess.Popen(
                    [PYTHON, str(Path(__file__).resolve()), "--action", action],
                    env=environment(), stdout=log, stderr=subprocess.STDOUT)
            children[action] = process
            write(RECOVERY / (action + "_process.json"), process_identity(process.pid),
                  exclusive=True)
        exits = {}
        for action, process in children.items():
            exits[action] = process.wait()
            write(RECOVERY / (action + "_exit.json"), {
                "exit_code": exits[action], "finished_at": time.time(), "pid": process.pid})
        verify_plan()
        write(RECOVERY / "exit.json", {
            "status": "complete" if all(c == 0 for c in exits.values()) else "stopped",
            "exits": exits, "finished_at": time.time(),
            "automatic_model_retry": False, "original_results_hashes_verified": True})
        status()


def status():
    state = read(ROOT / "requests/state.json")
    system = result_rows()
    direct = direct_rows()
    complete_direct = {r["qa_id"] for r in direct if r["status"] == "completed"}
    result = {
        "queried_at": time.time(), "system_completed": len(system),
        "direct_completed": len(complete_direct), "total_questions": 40,
        "request_status_counts": dict(Counter(r["status"] for r in state["requests"])),
        "reserved_requests": len(state["requests"]), "max_requests": 320,
        "recovery_direct_status_counts": dict(Counter(
            read(p)["status"] for p in (RECOVERY / "direct/responses").glob("*.json"))),
        "owners": owners(),
        "exit": read(RECOVERY / "exit.json") if (RECOVERY / "exit.json").exists() else None}
    if RECOVERY.exists():
        write(RECOVERY / "status.json", result)
    print(json.dumps(result, sort_keys=True), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--action", choices=("plan", "launch", "solve", "direct", "status"),
                        required=True)
    args = parser.parse_args()
    if args.action == "plan":
        plan()
    elif args.action in ("solve", "direct"):
        arm(args.action)
    elif args.action == "launch":
        launch()
    else:
        status()


if __name__ == "__main__":
    try:
        main()
    except BaseException as exc:
        print(json.dumps({
            "status": "stopped", "error_type": type(exc).__name__,
            "stack": [{"file": Path(row.filename).name, "line": row.lineno,
                       "function": row.name}
                      for row in traceback.extract_tb(exc.__traceback__)]}), flush=True)
        raise SystemExit(1)
