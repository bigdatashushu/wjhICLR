"""Keep the selected pairs fixed and preserve every generation of finalized arms."""
from __future__ import annotations

from collections import defaultdict
from copy import deepcopy
import fcntl
from pathlib import Path
import shutil
import tarfile
import time

from .adapters import ROOT
from .partial_score import freeze
from .relaunch import reuse_validation
from .response_policy import continuation_config
from .run import code_identity, verify_manifest
from .subset import arm_cap, identity_check, safe_file, selection_report, verified_record, verify_reuse
from .traces import digest, file_digest, read, write


def inherited_entry(entry, previous):
    value = deepcopy(entry)
    value["episode"] = "reuse_source/" + entry["episode"]
    value["files"] = {"reuse_source/" + n: h for n, h in entry["files"].items()}
    value.setdefault("origin_snapshot_identity", previous["reuse_snapshot_identity"])
    return value


def check_snapshot(source, snapshot):
    if (digest({k: v for k, v in snapshot.items() if k != "snapshot_identity"})
            != snapshot["snapshot_identity"]
            or file_digest(source / "manifest.json") != snapshot["manifest_sha256"]
            or file_digest(source / "request_ledger_snapshot.json") != snapshot["request_ledger_sha256"]):
        raise ValueError("continuation_snapshot_changed")


def verify_continuation(directory, manifest):
    source = directory / "reuse_source"
    snapshot = read(source / "snapshot.json")
    check_snapshot(source, snapshot)
    if (snapshot["snapshot_identity"] != manifest["reuse_snapshot_identity"]
            or file_digest(source / "fresh_snapshot.json") != snapshot["fresh_snapshot_sha256"]):
        raise ValueError("continuation_snapshot_binding_changed")
    previous = read(source / "manifest.json")
    identity_check(previous)
    if (previous["run_identity"] != manifest["source_run_identity"]
            or previous["pairs"] != manifest["pairs"] or previous["strata"] != manifest["strata"]
            or previous["counts"] != manifest["counts"]):
        raise ValueError("fixed_selection_changed")
    inherited = verify_reuse(source, previous)
    fresh = read(source / "fresh_snapshot.json")
    check_snapshot(source, fresh)
    combined = [inherited_entry(e, previous) for e in inherited.values()]
    ledger = read(source / "request_ledger_snapshot.json")
    pairs = {p["pair_identity"]: p for p in manifest["pairs"]}
    for entry in fresh["index"]:
        pair = pairs[entry["pair_identity"]]
        spec = next(s for s in pair["arms"] if s["arm"] == entry["arm"])
        verified_record(safe_file(source, entry["episode"]).parent, previous, pair, spec,
                        Path(fresh["source"]), ledger)
        combined.append({**entry, "origin_snapshot_identity": fresh["snapshot_identity"]})
    if combined != snapshot["index"] or len(combined) != manifest["reused_arms"]:
        raise ValueError("continuation_reuse_index_changed")
    index, endpoints = {}, defaultdict(set)
    for entry in combined:
        key = entry["pair_identity"], entry["arm"]
        if key in index:
            raise ValueError("duplicate_continuation_arm")
        for name, expected in entry["files"].items():
            if file_digest(safe_file(source, name)) != expected:
                raise ValueError("continuation_trajectory_changed")
        episode = read(source / entry["episode"])
        endpoints[entry["pair_identity"]].add(episode["api_id"])
        index[key] = entry
    if any(len(ids) != 1 for ids in endpoints.values()):
        raise ValueError("continuation_api_affinity_changed")
    if {p: next(iter(ids)) for p, ids in endpoints.items()} != manifest["reused_api_pins"]:
        raise ValueError("continuation_api_pins_changed")
    budget = sum(arm_cap(p["dataset"], s["arm"]) for p in manifest["pairs"] for s in p["arms"]
                 if (p["pair_identity"], s["arm"]) not in index)
    if (budget != manifest["max_requests"]
            or manifest["config"] != continuation_config(previous["config"], budget)
            or manifest["missing_arms"] != manifest["arms"] - len(index)):
        raise ValueError("continuation_configuration_changed")
    for name, expected in manifest["validation_receipt_sha256"].items():
        if file_digest(safe_file(directory, name)) != expected:
            raise ValueError("continuation_validation_changed")
    return index


def prepare(directory, source):
    directory, source = Path(directory).absolute(), Path(source).absolute()
    previous = read(source / "manifest.json")
    identity_check(previous)
    if previous.get("schema") not in {"gptexp-type30-paired-reuse/1", "gptexp-type30-paired-reuse/2"}:
        raise ValueError("continuation_requires_existing_selected_pairs")
    stop = read(source / "stopped_by_user.json")
    actual = {str(p.relative_to(source)): file_digest(p)
              for p in (source / "episodes").rglob("episode.json")}
    if actual != stop["completed_episode_sha256"]:
        raise ValueError("continuation_source_changed_after_stop")
    directory.mkdir(exist_ok=False)
    target = directory / "reuse_source"
    with (source / "operator.lock").open("a+b") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        inherited = verify_reuse(source, previous)
        freeze(source, target)
        fresh = read(target / "snapshot.json")
        shutil.copyfile(target / "snapshot.json", target / "fresh_snapshot.json")
        shutil.copytree(source / "reuse_source", target / "reuse_source")
        # The previous selection's verifier also binds the original API probes.
        for name, expected in previous["validation_receipt_sha256"].items():
            path = safe_file(target, name)
            path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(safe_file(source, name), path)
            if file_digest(path) != expected:
                raise ValueError("prior_validation_copy_changed")
    combined = [inherited_entry(e, previous) for e in inherited.values()]
    combined.extend({**e, "origin_snapshot_identity": fresh["snapshot_identity"]}
                    for e in fresh["index"])
    keys = {(e["pair_identity"], e["arm"]) for e in combined}
    if len(keys) != len(combined):
        raise ValueError("duplicate_prior_result")
    snapshot = {
        **fresh, "schema": "gptexp-continuation-freeze/1", "index": combined,
        "completed_arms": len(combined), "inherited_arms": len(inherited),
        "fresh_arms": len(fresh["index"]),
        "fresh_snapshot_sha256": file_digest(target / "fresh_snapshot.json")}
    snapshot.pop("snapshot_identity")
    snapshot["snapshot_identity"] = digest(snapshot)
    write(target / "snapshot.json", snapshot)
    budget = sum(arm_cap(p["dataset"], s["arm"]) for p in previous["pairs"] for s in p["arms"]
                 if (p["pair_identity"], s["arm"]) not in keys)
    config = continuation_config(previous["config"], budget)
    final, validation_previous = reuse_validation(directory, source, config)
    if final != config or validation_previous != previous:
        raise ValueError("continuation_api_configuration_changed")
    pins = {}
    for entry in combined:
        episode = read(target / entry["episode"])
        old = pins.setdefault(entry["pair_identity"], episode["api_id"])
        if old != episode["api_id"]:
            raise ValueError("prior_pair_api_affinity_changed")
    manifest = {
        **{k: deepcopy(previous[k]) for k in ("questions", "arms", "counts", "pairs", "strata",
                                             "source_receipts", "preflight")},
        "schema": "gptexp-type30-paired-reuse/2", "config": final, "max_requests": budget,
        "reused_arms": len(combined), "missing_arms": previous["arms"] - len(combined),
        "source_run_identity": previous["run_identity"], "source_directory": str(source),
        "reuse_snapshot_identity": snapshot["snapshot_identity"], "reused_api_pins": pins,
        "adapter_source": code_identity(),
        "validation_receipt_sha256": read(directory / "api_validation.json")["receipt_sha256"],
        "reference_answer_visible": False, "automatic_retry": False, "skill_evolution": False,
        "selection_policy": previous["selection_policy"], "fixed_selection_preserved": True,
        "allowed_changes": ["response metadata recorded without blocking",
                            "immutable reuse from all previous generations",
                            "remaining-arm global budget and scheduling"],
        "prepared_at": time.time()}
    manifest["run_identity"] = digest(manifest)
    write(directory / "manifest.json", manifest, exclusive=True)
    write(directory / "run.config.json", config, exclusive=True)
    write(directory / "launch.config.json", config, exclusive=True)
    verify_manifest(manifest)
    verify_reuse(directory, manifest)
    with tarfile.open(directory / "public_source_snapshot.tar.gz", "w:gz") as archive:
        for name in code_identity():
            archive.add(ROOT / name, arcname=name)
    receipt = {
        "status": "ready", "run_identity": manifest["run_identity"], "questions": manifest["questions"],
        "arms": manifest["arms"], "reused_arms": len(combined), "inherited_arms": len(inherited),
        "prior_fresh_arms": len(fresh["index"]), "missing_arms": manifest["missing_arms"],
        "max_requests": budget, "validated_apis": len(config["apis"]), "model_requests": 0,
        "new_api_probes": 0, "fixed_pairs_and_original_specs_preserved": True,
        "all_finalized_outcomes_reused": True, "reference_answers_read": False,
        "original_episode_bytes_preserved": True, "partial_pair_api_affinity_preserved": True,
        "response_metadata_policy": config["response_metadata_policy"],
        "reported_metadata_synthesized": False,
        "public_source_snapshot_sha256": file_digest(directory / "public_source_snapshot.tar.gz"),
        "at": time.time()}
    write(directory / "readiness_audit.json", receipt, exclusive=True)
    write(directory / "preparation.json", receipt, exclusive=True)
    text = selection_report(manifest)
    text += "\n本次沿用上一轮固定题单；回包元数据只记录，按同步请求上下文对应回复。\n"
    (directory / "selection_report.md").write_text(text, encoding="utf-8")
    return receipt
