"""Small, stratified, label-free trajectory diagnostics from immutable inputs."""
from __future__ import annotations

from collections import Counter, defaultdict
from copy import deepcopy
from pathlib import Path
import time


def select_pairs(pairs, *, excluded=(), per_type=5, seed=137):
    from ..traces import digest

    if not 1 <= per_type <= 5:
        raise ValueError("diagnostic_requires_one_to_five_per_type")
    excluded = set(excluded)
    groups = defaultdict(list)
    for pair in pairs:
        if pair["sample_id"] not in excluded:
            groups[pair["arms"][0]["problem"]["task_type"]].append(pair)
    if len(groups) != 6 or any(len(rows) < per_type for rows in groups.values()):
        raise ValueError("diagnostic_requires_six_complete_task_strata")
    ordered = {task: sorted(rows, key=lambda row: digest(
        {"seed": seed, "sample_id": row["sample_id"], "pair_identity": row["pair_identity"]}))[:per_type]
        for task, rows in sorted(groups.items())}
    return [rows[index] for index in range(per_type) for rows in ordered.values()]


def validate_diagnostic(manifest):
    pairs = manifest["pairs"]
    if (
        manifest.get("diagnostic_only") is not True
        or manifest.get("reference_answer_visible") is not False
        or manifest.get("automatic_retry") is not False
        or manifest.get("host_scoring") != "disabled"
        or not 1 <= len(pairs) <= 30
        or manifest["questions"] != len(pairs)
        or manifest["arms"] != 2 * len(pairs)
        or manifest["counts"] != {"mmsi": len(pairs)}
        or manifest["max_requests"] != 5 * len(pairs)
        or manifest["config"]["max_requests"] != manifest["max_requests"]
        or len({row["sample_id"] for row in pairs}) != len(pairs)
    ):
        raise ValueError("diagnostic_bounds_or_contract_changed")
    for pair in pairs:
        if pair["dataset"] != "mmsi" or sorted(a["arm"] for a in pair["arms"]) != ["direct", "harness"]:
            raise ValueError("diagnostic_pair_invalid")
        direct = next(a for a in pair["arms"] if a["arm"] == "direct")
        harness = next(a for a in pair["arms"] if a["arm"] == "harness")
        if direct["problem"] != harness["problem"]:
            raise ValueError("diagnostic_input_pair_changed")
        if any(k in direct for k in ("prepared_root", "artifact_sha256", "skill", "allowed_tools")):
            raise ValueError("diagnostic_direct_geometry_exposure")
    counts = Counter(p["arms"][0]["problem"]["task_type"] for p in pairs)
    if len(counts) != 6 or len(set(counts.values())) != 1 or max(counts.values()) > 5:
        raise ValueError("diagnostic_strata_changed")


def prepare_diagnostic(directory, source_run, *, per_type=5, seed=137, stage="all"):
    from ..adapters import ROOT, problem_for, setup
    from ..run import code_identity
    from ..traces import digest, file_digest, read, write

    directory, source_run = Path(directory), Path(source_run)
    source = read(source_run / "manifest.json")
    if digest({k: v for k, v in source.items() if k != "run_identity"}) != source["run_identity"]:
        raise ValueError("source_manifest_changed")
    if source["counts"] != {"mmsi": 365} or source["reference_answer_visible"] is not False:
        raise ValueError("diagnostic_requires_label_free_mmsi_source")
    for name, expected in source["source_receipts"]["mmsi"]["files"].items():
        if file_digest(ROOT / "frozen/mmsi" / name) != expected:
            raise ValueError("frozen_source_changed")
    # Avoid both completed and uncertain in-flight requests from the stopped run.
    excluded = [
        pair["sample_id"] for pair in source["pairs"]
        if (source_run / "episodes/mmsi" / pair["pair_identity"]).exists()
    ]
    selected = select_pairs(source["pairs"], excluded=excluded, per_type=per_type, seed=seed)
    if stage not in {"first", "remaining", "all"}:
        raise ValueError("diagnostic_stage_invalid")
    pairs = selected[:6] if stage == "first" else selected[6:] if stage == "remaining" else selected
    pairs = deepcopy(pairs)
    if not pairs:
        raise ValueError("diagnostic_stage_empty")
    setup("mmsi")
    from skill3d.small_image.blink_protocol import image_identities
    from skill3d.small_image.runner import _validate_prepared, load_problem_frames
    from skill3d.small_image.storage import load_prepared

    original = read(ROOT / "frozen/mmsi/manifest.json")
    image_binding = {job["problem"]["sample_id"]: job["images"] for job in original["jobs"]}
    for pair in pairs:
        harness = next(a for a in pair["arms"] if a["arm"] == "harness")
        problem = problem_for(harness["problem"])
        if image_identities(problem) != image_binding[problem.sample_id]:
            raise ValueError("diagnostic_original_image_identity_changed")
        prepared = load_prepared(Path(harness["prepared_root"]) / problem.input_sha256, problem=problem)
        if prepared.artifact_sha256 != harness["artifact_sha256"] or _validate_prepared(
            prepared, problem, load_problem_frames(problem)
        ):
            raise ValueError("diagnostic_prepared_identity_changed")
    config = deepcopy(source["config"])
    config["max_requests"] = 5 * len(pairs)
    config.pop("service_failure_policy", None)
    # A source-run service failure must not be retried through the same endpoint.
    paused = {p.name.removeprefix("paused-").removesuffix(".json")
              for p in (source_run / "pool").glob("paused-*.json")}
    for api in config["apis"]:
        if api["id"] in paused:
            api["enabled"] = False
    selection = {
        "schema": "mmsigpt-stratified-selection/1", "seed": seed, "per_type": per_type,
        "stage": stage, "task_type_only_sampling": True, "selection_reads_labels": False,
        "source_split_preserved": True, "learning_role": "diagnostic",
        "excluded_dispatched_sample_ids": sorted(excluded),
        "all_selected_sample_ids": [p["sample_id"] for p in selected],
        "campaign_questions": len(selected), "campaign_request_cap": 5 * len(selected),
    }
    manifest = {
        "schema": "mmsigpt-stratified-diagnostic/1", "questions": len(pairs),
        "arms": 2 * len(pairs), "max_requests": config["max_requests"],
        "counts": {"mmsi": len(pairs)}, "config": config, "pairs": pairs,
        "source_receipts": source["source_receipts"], "adapter_source": code_identity(),
        "source_run_identity": source["run_identity"],
        "source_manifest_sha256": file_digest(source_run / "manifest.json"),
        "preflight": {"mmsi": digest(pairs)}, "selection_policy": selection,
        "reference_answer_visible": False, "automatic_retry": False, "skill_evolution": False,
        "diagnostic_only": True, "host_scoring": "disabled",
        "allowed_changes": ["bounded stratified sampling", "public evidence summary capture",
                            "new MMSI prompt/interface version", "label-free export"],
        "prepared_at": time.time(),
    }
    validate_diagnostic(manifest)
    manifest["run_identity"] = digest(manifest)
    directory.mkdir(parents=True, exist_ok=True)
    write(directory / "manifest.json", manifest, exclusive=True)
    write(directory / "preparation.json", {
        "status": "ready", "questions": len(pairs), "arms": 2 * len(pairs), "model_requests": 0,
        "run_identity": manifest["run_identity"], "reference_answers_read": False,
        "original_images_verified": True, "prepared_identity_verified": True,
        "task_counts": dict(Counter(p["arms"][0]["problem"]["task_type"] for p in pairs)),
        "disabled_source_paused_apis": sorted(paused),
    }, exclusive=True)
    print(f"Prepared {len(pairs)} stratified diagnostic pairs; zero model requests.", flush=True)
