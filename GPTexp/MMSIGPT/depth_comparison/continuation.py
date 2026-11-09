"""Move only untouched pairs from a frozen failed run; never retry an attempted pair."""
from __future__ import annotations

from collections import Counter
from copy import deepcopy
from pathlib import Path

from ...traces import digest, read
from ..prototype.inputs import load_images
from .runner import save_preparation, validate_frozen


def untouched(directory):
    """Validate the historical envelope and actual traces, independent of new source/env."""
    directory = Path(directory)
    manifest = read(directory / "manifest.json")
    total = manifest["questions"]
    if (manifest["schema"] != "mmsi-depth-paired/1"
            or digest({k: v for k, v in manifest.items() if k != "run_identity"})
            != manifest["run_identity"] or total not in (24, 30)
            or len(manifest["items"]) != total or manifest["max_requests"] != 2 * total
            or manifest["reference_answer_visible"] is not False
            or manifest["automatic_retry"] is not False or manifest["tools_enabled"] is not False
            or (directory / "score.json").exists()):
        raise ValueError("parent_envelope_or_label_isolation_changed")
    if total == 24:
        validate_continuation(manifest)
    elif "continuation" in manifest:
        raise ValueError("unexpected_parent_continuation")
    frozen = validate_frozen(directory, manifest)
    execution = read(directory / "execution.json")
    attempted = sorted({row["sample_id"] for row in frozen["predictions"]
                        if row["status"] != "not_dispatched"})
    remaining = [item for item in manifest["items"]
                 if item["problem"]["sample_id"] not in attempted]
    if (execution["status"] != "stopped_service_error" or execution["dispatched"] != 6
            or len(attempted) != 6 or len(remaining) != total - 6
            or Counter(row["status"] for row in frozen["predictions"])
            != {"service_error": 6, "not_dispatched": 2 * total - 6}
            or len(manifest["api_ids"]) != 6
            or Counter(item["api_id"] for item in remaining)
            != {api: (total - 6) // 6 for api in manifest["api_ids"]}
            or len(Counter(item["problem"]["task_type"] for item in remaining)) != 6
            or set(Counter(item["problem"]["task_type"] for item in remaining).values())
            != {(total - 6) // 6}):
        raise ValueError("parent_requires_six_failed_first_requests_and_balanced_untouched_pairs")
    return manifest, frozen, attempted, remaining


def child_items(remaining):
    ranked = sorted(range(len(remaining)), key=lambda i: digest({
        "seed": 137, "purpose": "arm-order",
        "sample": remaining[i]["problem"]["sample_id"],
    }))
    direct_first = set(ranked[:len(remaining) // 2])
    items = deepcopy(remaining)
    for i, item in enumerate(items):
        item["index"] = i
        item["arm_order"] = ["direct", "depth"] if i in direct_first else ["depth", "direct"]
    return items


def binding(directory, parent, frozen, attempted):
    prior = parent.get("continuation", {})
    parent_requests = prior.get("parent_requests", 0) + 6
    child_max_requests = 2 * (parent["questions"] - 6)
    excluded = sorted(set(prior.get("excluded_attempted_sample_ids", [])) | set(attempted))
    return {
        "schema": "mmsi-depth-untouched-continuation/1",
        "parent_directory": str(Path(directory).resolve()),
        "parent_run_identity": parent["run_identity"],
        "parent_freeze_identity": frozen["freeze_identity"],
        "excluded_attempted_sample_ids": excluded, "parent_requests": parent_requests,
        "child_max_requests": child_max_requests,
        "combined_max_requests": parent_requests + child_max_requests,
        "original_budget": 60, "attempted_pairs_retried": False,
        "selection_rule": "all_parent_pairs_with_neither_arm_dispatched",
        "original_payloads_preserved": True,
    }


def validate_continuation(manifest):
    info = manifest["continuation"]
    parent, frozen, attempted, remaining = untouched(info["parent_directory"])
    if (info != binding(info["parent_directory"], parent, frozen, attempted)
            or manifest["items"] != child_items(remaining)
            or manifest["selection"]["items"] != manifest["items"]
            or manifest["api_ids"] != parent["api_ids"]
            or manifest["inference_by_api"] != parent["inference_by_api"]):
        raise ValueError("continuation_changed_or_attempted_pair_reintroduced")


def prepare_from_frozen(directory, parent_directory):
    parent, frozen, attempted, remaining = untouched(parent_directory)
    items = child_items(remaining)
    # Full original bytes and all public input hashes are checked before writing.
    for item in items:
        load_images(item["problem"], item["images"])
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    if any(directory.iterdir()):
        raise ValueError("requires_empty_new_run_directory")
    for item in items:
        for receipt in item["depth_images"]:
            relative = Path(receipt["path"])
            if relative.is_absolute() or ".." in relative.parts:
                raise ValueError("unsafe_depth_path")
            target = directory / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            raw = (Path(parent_directory) / relative).read_bytes()
            with target.open("xb") as stream:
                stream.write(raw)
    selected = {**deepcopy(parent["selection"]), "per_type": len(items) // 6,
                "questions": len(items), "max_requests": 2 * len(items), "items": items}
    selected["selection_identity"] = digest({
        k: v for k, v in selected.items() if k != "selection_identity"})
    return save_preparation(
        directory, selected, items,
        continuation=binding(parent_directory, parent, frozen, attempted))
