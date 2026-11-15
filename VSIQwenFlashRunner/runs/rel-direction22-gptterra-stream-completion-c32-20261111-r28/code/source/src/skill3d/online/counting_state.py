"""Counting state presentation and deterministic image-reference checks.

No pixel semantics, category ontology or physical-instance coverage is inferred.
"""

import copy
import json
import math


def episode_feedback(kernel, observations, *, max_page_chars=16000):
    """Rehydrate every recovery from the same valid episode ledger, with bounded pages."""
    valid_ids = {o.result_id for o in observations}
    progress = {}
    for item in getattr(kernel, "vsi_reader_progress", []):
        if item["result_id"] in valid_ids:
            key = (item["result_id"], tuple(item["data_path"]),
                   None if item["fields"] is None else tuple(sorted(item["fields"])))
            progress[key] = item
    pages, seen, chars = [], set(), 0
    for page in reversed(getattr(kernel, "observation_reads", [])):
        if page["result_id"] not in valid_ids:
            continue
        key = json.dumps({k: page.get(k) for k in (
            "result_id", "payload_hash", "data_path", "returned_range",
            "fields_omitted", "data")}, sort_keys=True)
        if key in seen:
            continue
        seen.add(key)
        size = len(json.dumps(page, ensure_ascii=False, sort_keys=True))
        if chars + size > max_page_chars:
            continue
        pages.append(page)
        chars += size
    lines = [
        "Episode read state (cumulative selected-field coverage; summary omission is separate):",
        json.dumps(list(progress.values()), ensure_ascii=False, sort_keys=True),
        "Retained requested pages (raw data and source paths; older pages may be omitted by "
        "the feedback cap; omission does not reset completed reads):",
        json.dumps(list(reversed(pages)), ensure_ascii=False, sort_keys=True),
    ]
    records = getattr(kernel, "counting_evidence", [])
    if records:
        lines += ["Last declared instance evidence (image references checked; pixel semantics, "
                  "identity and coverage remain unverified):",
                  json.dumps(records[-1]["items"], ensure_ascii=False, sort_keys=True)]
    return "\n".join(lines)


def count_tool_status(handle):
    """Use the actual gate, preserving authorization separately from callability."""
    if handle is None or not callable(getattr(handle, "category_binding_complete", None)):
        return ""
    requirements = getattr(handle, "grounding_coverage", {}).get("requirements", [])
    rows = []
    for requirement in requirements:
        name = requirement["name"]
        callable_now = handle.category_binding_complete(name)
        reasons = []
        if not callable_now:
            if not requirement.get("scan_complete"):
                reasons.append("technical_scan_incomplete")
            if requirement.get("status") != "candidates_available":
                reasons.append("candidate_set_unavailable")
            if requirement.get("binding_failures"):
                reasons.append("binding_failures")
            if not requirement.get("usable_object_ids"):
                reasons.append("no_usable_bindings")
            if not reasons:
                reasons.append("binding_validation_failed")
        rows.append({"category": name, "callable_now": callable_now, "blocked_by": reasons})
    if not rows:
        return ""
    return (
        "\ncount_objects current preconditions (authorization remains in the tool catalog): "
        + json.dumps(rows, sort_keys=True, separators=(",", ":"))
        + "\nA blocked count call remains blocked until the host records a complete candidate scan "
        "and usable bindings. Viewing/cropping images does not change this gate. For visual "
        "identity doubts, inspect delivered images and targeted regions; a visual answer is legal.")


def validate_instance_evidence(items, ledger):
    """Check only record shape, completed image delivery, coordinates and merge references."""
    if not isinstance(items, list) or not items or len(items) > 128:
        raise ValueError("CountingEvidence requires 1..128 instance records")
    if ledger is None:
        raise ValueError("CountingEvidence requires the current image ledger")
    current = next((r for r in ledger.rounds
                    if r.round_index == int(ledger.current_round)), None)
    observed = set(current.image_ids) if current is not None and current.observed else set()
    clean, included, merges = [], set(), []
    for row in items:
        if not isinstance(row, dict) or set(row) - {
                "instance_id", "image_id", "region", "decision", "same_as", "reason"}:
            raise ValueError("Invalid instance record fields")
        iid, image = row.get("instance_id"), row.get("image_id")
        decision, reason = row.get("decision"), row.get("reason")
        if not isinstance(iid, str) or not iid or len(iid) > 80:
            raise ValueError("Instance ID must be a nonempty string <=80 characters")
        if image not in observed:
            raise ValueError(f"Image {image!r} is not observed in this completed request")
        region = row.get("region")
        if (not isinstance(region, list) or len(region) != 4
                or any(type(v) not in (int, float) or not math.isfinite(v)
                       or not 0 <= v <= 1000 for v in region)
                or region[0] >= region[2] or region[1] >= region[3]):
            raise ValueError("region must be sorted [x0,y0,x1,y1] in image coordinates 0..1000")
        if decision not in {"include", "exclude", "merge", "uncertain"}:
            raise ValueError("decision must be include/exclude/merge/uncertain")
        if not isinstance(reason, str) or not reason.strip() or len(reason) > 400:
            raise ValueError("Each instance record needs a reason <=400 characters")
        if decision == "include":
            included.add(iid)
        if decision == "merge":
            merges.append((iid, row.get("same_as")))
        elif row.get("same_as") is not None:
            raise ValueError("same_as is only valid for a merge")
        clean.append(copy.deepcopy(row))
    for iid, target in merges:
        if not isinstance(target, str) or target not in included or iid == target:
            raise ValueError("Merge target must be an included, different instance ID")
    return clean


def verification_scope(verify, kernel):
    """Expose performed checks without interpreting submission admission as correctness."""
    reference_calls = [r for r in getattr(kernel, "cell_vsi_interface_calls", [])
                       if r["name"] == "CountingEvidence" and r["allowed"]]
    return {
        "submission_contract": verify.check_status.get("answer_contract", "not_performed"),
        "result_checks": dict(verify.check_status),
        "derivation_replay": verify.check_status.get("derivation_replay", "not_performed"),
        "instance_image_references": "pass" if reference_calls else "not_performed",
        "instance_pixel_semantics": "not_performed",
        "instance_identity_and_coverage": "not_performed",
        "admission_passed": verify.passed,
        "meaning": "Admission checks the execution/evidence contract, not count correctness.",
    }
